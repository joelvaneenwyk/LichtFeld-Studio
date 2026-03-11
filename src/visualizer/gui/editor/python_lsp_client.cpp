/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "python_lsp_client.hpp"

#include "stdio_process.hpp"

#include <core/logger.hpp>
#include <core/path_utils.hpp>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <mutex>
#include <numeric>
#include <optional>
#include <stop_token>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <unordered_set>

#ifndef _WIN32
#include <unistd.h>
#else
#include <processthreadsapi.h>
#endif

namespace lfs::vis::editor {

    namespace {

        using json = nlohmann::json;
        namespace fs = std::filesystem;

        constexpr auto WORKER_IDLE_WAIT = std::chrono::milliseconds(16);
        constexpr auto RESTART_DELAY = std::chrono::milliseconds(400);

        struct ServerCommand {
            std::string program;
            std::vector<std::string> args;
            std::string label;
        };

        struct PendingCompletionRequest {
            int document_version = 0;
            int line = 0;
            int character = 0;
            bool manual = false;
        };

        bool is_unreserved(const unsigned char ch) {
            return (ch >= 'A' && ch <= 'Z') || (ch >= 'a' && ch <= 'z') ||
                   (ch >= '0' && ch <= '9') || ch == '-' || ch == '_' || ch == '.' ||
                   ch == '~' || ch == '/' || ch == ':';
        }

        std::string percent_encode(std::string_view text) {
            static constexpr std::array<char, 16> HEX = {
                '0', '1', '2', '3', '4', '5', '6', '7',
                '8', '9', 'A', 'B', 'C', 'D', 'E', 'F',
            };

            std::string encoded;
            encoded.reserve(text.size());
            for (const unsigned char ch : text) {
                if (is_unreserved(ch)) {
                    encoded.push_back(static_cast<char>(ch));
                    continue;
                }

                encoded.push_back('%');
                encoded.push_back(HEX[(ch >> 4) & 0xF]);
                encoded.push_back(HEX[ch & 0xF]);
            }
            return encoded;
        }

        std::string file_uri_from_path(const fs::path& path) {
            fs::path absolute = path;
            std::error_code ec;
            if (absolute.is_relative()) {
                absolute = fs::absolute(absolute, ec);
            }
            if (!ec) {
                absolute = absolute.lexically_normal();
            }

            std::string utf8 = lfs::core::path_to_utf8(absolute);
#ifdef _WIN32
            return "file:///" + percent_encode(fs::path(utf8).generic_string());
#else
            return "file://" + percent_encode(fs::path(utf8).generic_string());
#endif
        }

        void append_candidate(std::vector<ServerCommand>& commands,
                              std::unordered_set<std::string>& seen,
                              ServerCommand command) {
            const std::string key = command.program + '\n' + std::accumulate(
                command.args.begin(), command.args.end(), std::string(),
                [](std::string acc, const std::string& arg) {
                    acc += arg;
                    acc.push_back('\n');
                    return acc;
                });
            if (seen.insert(key).second) {
                commands.push_back(std::move(command));
            }
        }

        std::vector<ServerCommand> discover_server_commands() {
            std::vector<ServerCommand> commands;
            std::unordered_set<std::string> seen;

            if (const char* override_program = std::getenv("LFS_PYTHON_LSP");
                override_program && *override_program) {
                append_candidate(commands, seen,
                                 {.program = override_program,
                                  .args = {},
                                  .label = "custom Python language server"});
                return commands;
            }

            const fs::path project_root(PROJECT_ROOT_PATH);
#ifdef _WIN32
            constexpr std::array<std::string_view, 4> DIRECT_EXECUTABLES = {
                "py/.venv/Scripts/basedpyright-langserver.exe",
                ".venv/Scripts/basedpyright-langserver.exe",
                "py/.venv/Scripts/pyright-langserver.exe",
                ".venv/Scripts/pyright-langserver.exe",
            };
#else
            constexpr std::array<std::string_view, 4> DIRECT_EXECUTABLES = {
                "py/.venv/bin/basedpyright-langserver",
                ".venv/bin/basedpyright-langserver",
                "py/.venv/bin/pyright-langserver",
                ".venv/bin/pyright-langserver",
            };
#endif

            for (const auto relative : DIRECT_EXECUTABLES) {
                const fs::path candidate = project_root / relative;
                if (!fs::exists(candidate)) {
                    continue;
                }

                const bool is_basedpyright =
                    candidate.filename().string().find("basedpyright") != std::string::npos;
                append_candidate(commands, seen,
                                 {.program = lfs::core::path_to_utf8(candidate),
                                  .args = {"--stdio"},
                                  .label = is_basedpyright ? "basedpyright" : "pyright"});
            }

            append_candidate(commands, seen,
                             {.program = "basedpyright-langserver",
                              .args = {"--stdio"},
                              .label = "basedpyright"});
            append_candidate(commands, seen,
                             {.program = "pyright-langserver",
                              .args = {"--stdio"},
                              .label = "pyright"});
            append_candidate(commands, seen,
                             {.program = "uv",
                              .args = {"tool", "run", "--from", "basedpyright",
                                       "basedpyright-langserver", "--stdio"},
                              .label = "basedpyright via uv"});
            return commands;
        }

        uint64_t current_process_id() {
#ifdef _WIN32
            return static_cast<uint64_t>(GetCurrentProcessId());
#else
            return static_cast<uint64_t>(getpid());
#endif
        }

        std::string json_rpc_frame(const json& payload) {
            const std::string body = payload.dump();
            return "Content-Length: " + std::to_string(body.size()) + "\r\n\r\n" + body;
        }

        std::optional<PythonLspClient::TextEdit> parse_text_edit(const json& node) {
            if (!node.is_object()) {
                return std::nullopt;
            }

            const json* range = nullptr;
            if (node.contains("range") && node["range"].is_object()) {
                range = &node["range"];
            } else if (node.contains("replace") && node["replace"].is_object()) {
                range = &node["replace"];
            } else if (node.contains("insert") && node["insert"].is_object()) {
                range = &node["insert"];
            }

            if (range == nullptr || !range->contains("start") || !range->contains("end")) {
                return std::nullopt;
            }

            const auto& start = (*range)["start"];
            const auto& end = (*range)["end"];
            if (!start.is_object() || !end.is_object()) {
                return std::nullopt;
            }

            return PythonLspClient::TextEdit{
                .start_line = start.value("line", 0),
                .start_character = start.value("character", 0),
                .end_line = end.value("line", 0),
                .end_character = end.value("character", 0),
                .new_text = node.value("newText", std::string()),
            };
        }

        std::optional<json> pop_json_rpc_message(std::string& buffer) {
            while (true) {
                const size_t header_end = buffer.find("\r\n\r\n");
                if (header_end == std::string::npos) {
                    return std::nullopt;
                }

                std::string_view headers(buffer.data(), header_end);
                size_t content_length = 0;

                size_t line_start = 0;
                while (line_start < headers.size()) {
                    const size_t line_end = headers.find("\r\n", line_start);
                    const std::string_view line = headers.substr(
                        line_start,
                        line_end == std::string::npos ? headers.size() - line_start
                                                      : line_end - line_start);
                    if (line.rfind("Content-Length:", 0) == 0) {
                        const std::string_view value = line.substr(std::string_view("Content-Length:").size());
                        content_length = static_cast<size_t>(std::stoull(std::string(value)));
                        break;
                    }
                    if (line_end == std::string::npos) {
                        break;
                    }
                    line_start = line_end + 2;
                }

                if (content_length == 0) {
                    buffer.erase(0, header_end + 4);
                    continue;
                }

                const size_t body_offset = header_end + 4;
                if (buffer.size() < body_offset + content_length) {
                    return std::nullopt;
                }

                const std::string body = buffer.substr(body_offset, content_length);
                buffer.erase(0, body_offset + content_length);

                try {
                    return json::parse(body);
                } catch (const json::parse_error& error) {
                    LOG_WARN("PythonLspClient: failed to parse LSP JSON: {}", error.what());
                }
            }
        }

    } // namespace

    struct PythonLspClient::Impl {
        enum class State {
            Starting,
            Ready,
            Failed,
        };

        Impl()
            : commands(discover_server_commands()),
              workspace_root(PROJECT_ROOT_PATH),
              document_path(workspace_root / ".lichtfeld" / "script_editor.py"),
              root_uri(file_uri_from_path(workspace_root)),
              document_uri(file_uri_from_path(document_path)),
              next_retry_at(std::chrono::steady_clock::now()) {
            if (commands.empty()) {
                state = State::Failed;
                status = "No Python language server configured";
                return;
            }

            status = "Starting Python completions...";
            worker = std::jthread([this](std::stop_token stop_token) { run(stop_token); });
        }

        ~Impl() {
            cv.notify_all();
        }

        int updateDocument(const std::string& text) {
            std::scoped_lock lock(mutex);
            if (document_text == text) {
                return document_version;
            }

            document_text = text;
            ++document_version;
            document_dirty = true;
            latest_completion.reset();
            cv.notify_all();
            return document_version;
        }

        void requestCompletion(const PendingCompletionRequest& request) {
            std::scoped_lock lock(mutex);
            queued_completion = request;
            cv.notify_all();
        }

        std::optional<CompletionList> takeLatestCompletion() {
            std::scoped_lock lock(mutex);
            auto result = std::move(latest_completion);
            latest_completion.reset();
            return result;
        }

        bool isReady() const {
            std::scoped_lock lock(mutex);
            return state == State::Ready;
        }

        bool isAvailable() const {
            std::scoped_lock lock(mutex);
            return state != State::Failed;
        }

        std::string statusText() const {
            std::scoped_lock lock(mutex);
            return status;
        }

        void run(const std::stop_token stop_token) {
            while (!stop_token.stop_requested()) {
                bool did_work = false;

                if (active_candidate_index.has_value() && !process.isRunning()) {
                    handle_process_exit();
                    did_work = true;
                }

                if (!active_candidate_index.has_value() &&
                    attempt_candidate_index < commands.size() &&
                    std::chrono::steady_clock::now() >= next_retry_at) {
                    did_work = try_start_candidate(attempt_candidate_index) || did_work;
                }

                if (active_candidate_index.has_value()) {
                    did_work = drain_stdout() || did_work;
                    did_work = drain_stderr(false) || did_work;
                    did_work = flush_document_sync() || did_work;
                    did_work = flush_completion_request() || did_work;
                }

                if (!did_work) {
                    std::unique_lock lock(mutex);
                    cv.wait_for(lock, WORKER_IDLE_WAIT);
                }
            }

            process.kill();
            drain_stderr(true);
        }

        bool try_start_candidate(size_t index) {
            if (index >= commands.size()) {
                return false;
            }

            const auto& command = commands[index];
            if (!process.start(command.program, command.args)) {
                LOG_WARN("PythonLspClient: failed to start {}", command.label);
                attempt_candidate_index = index + 1;
                next_retry_at = std::chrono::steady_clock::now() + RESTART_DELAY;
                if (attempt_candidate_index >= commands.size()) {
                    std::scoped_lock lock(mutex);
                    state = State::Failed;
                    status = "Python completions unavailable";
                }
                return false;
            }

            LOG_INFO("PythonLspClient: starting {}", command.label);

            {
                std::scoped_lock lock(mutex);
                state = State::Starting;
                status = "Starting " + command.label + "...";
                document_dirty = true;
            }

            active_candidate_index = index;
            initialized = false;
            ready_once = false;
            document_opened = false;
            stdout_buffer.clear();
            stderr_buffer.clear();
            inflight_completions.clear();
            last_completion_request.reset();

            initialize_request_id = next_request_id++;
            json initialize = {
                {"jsonrpc", "2.0"},
                {"id", initialize_request_id},
                {"method", "initialize"},
                {"params",
                 {
                     {"processId", current_process_id()},
                     {"rootUri", root_uri},
                     {"rootPath", lfs::core::path_to_utf8(workspace_root)},
                     {"clientInfo", {{"name", "LichtFeld Studio"}}},
                     {"workspaceFolders",
                      json::array({{{"uri", root_uri},
                                    {"name", workspace_root.filename().empty()
                                                 ? "LichtFeld Studio"
                                                 : lfs::core::path_to_utf8(workspace_root.filename())}}})},
                     {"capabilities",
                      {
                          {"general", {{"positionEncodings", json::array({"utf-16"})}}},
                          {"textDocument",
                           {
                               {"synchronization",
                                {{"willSave", false},
                                 {"willSaveWaitUntil", false},
                                 {"didSave", false}}},
                               {"completion",
                                {{"contextSupport", true},
                                 {"completionItem",
                                  {{"snippetSupport", false},
                                   {"documentationFormat", json::array({"markdown", "plaintext"})},
                                   {"labelDetailsSupport", true},
                                   {"deprecatedSupport", true}}}}}
                           }}
                      }}
                 }}
            };

            if (!process.writeAll(json_rpc_frame(initialize))) {
                process.kill();
                return false;
            }

            return true;
        }

        void handle_process_exit() {
            drain_stderr(true);

            const size_t failed_candidate = active_candidate_index.value_or(0);
            const int exit_code = process.exitCode();
            const bool restart_same_candidate = ready_once;

            LOG_WARN("PythonLspClient: {} exited with code {}",
                     commands[failed_candidate].label, exit_code);

            process.kill();
            active_candidate_index.reset();
            initialized = false;
            document_opened = false;
            inflight_completions.clear();
            last_completion_request.reset();

            {
                std::scoped_lock lock(mutex);
                state = State::Starting;
                status = restart_same_candidate ? "Restarting Python completions..."
                                                : "Trying fallback Python completions...";
                document_dirty = true;
            }

            if (restart_same_candidate) {
                attempt_candidate_index = failed_candidate;
            } else {
                attempt_candidate_index = failed_candidate + 1;
                if (attempt_candidate_index >= commands.size()) {
                    std::scoped_lock lock(mutex);
                    state = State::Failed;
                    status = "Python completions unavailable";
                }
            }

            next_retry_at = std::chrono::steady_clock::now() + RESTART_DELAY;
            ready_once = false;
        }

        bool drain_stdout() {
            bool did_work = false;
            std::array<char, 4096> buffer = {};

            while (true) {
                const ssize_t read = process.readStdout(buffer.data(), buffer.size());
                if (read <= 0) {
                    break;
                }

                stdout_buffer.append(buffer.data(), static_cast<size_t>(read));
                did_work = true;

                while (auto message = pop_json_rpc_message(stdout_buffer)) {
                    handle_message(*message);
                }
            }

            return did_work;
        }

        bool drain_stderr(const bool flush_all) {
            bool did_work = false;
            std::array<char, 2048> buffer = {};

            while (true) {
                const ssize_t read = process.readStderr(buffer.data(), buffer.size());
                if (read <= 0) {
                    break;
                }
                stderr_buffer.append(buffer.data(), static_cast<size_t>(read));
                did_work = true;
            }

            size_t line_end = 0;
            while ((line_end = stderr_buffer.find('\n')) != std::string::npos) {
                std::string line = stderr_buffer.substr(0, line_end);
                if (!line.empty() && line.back() == '\r') {
                    line.pop_back();
                }
                if (!line.empty()) {
                    LOG_DEBUG("PythonLspClient: {}", line);
                }
                stderr_buffer.erase(0, line_end + 1);
                did_work = true;
            }

            if (flush_all && !stderr_buffer.empty()) {
                LOG_DEBUG("PythonLspClient: {}", stderr_buffer);
                stderr_buffer.clear();
                did_work = true;
            }

            return did_work;
        }

        bool flush_document_sync() {
            std::string text;
            int version = 0;
            bool open_document = false;

            {
                std::scoped_lock lock(mutex);
                if (!initialized || !document_dirty) {
                    return false;
                }

                text = document_text;
                version = document_version;
                open_document = !document_opened;
                document_dirty = false;
            }

            json payload = {
                {"jsonrpc", "2.0"},
                {"method", open_document ? "textDocument/didOpen" : "textDocument/didChange"},
                {"params",
                 open_document
                     ? json{{"textDocument",
                             {{"uri", document_uri},
                              {"languageId", "python"},
                              {"version", version},
                              {"text", text}}}}
                     : json{{"textDocument",
                             {{"uri", document_uri},
                              {"version", version}}},
                            {"contentChanges", json::array({{{"text", text}}})}}}
            };

            if (!process.writeAll(json_rpc_frame(payload))) {
                std::scoped_lock lock(mutex);
                document_dirty = true;
                return false;
            }

            if (open_document) {
                document_opened = true;
            }
            return true;
        }

        bool flush_completion_request() {
            PendingCompletionRequest request;
            uint64_t request_id = 0;

            {
                std::scoped_lock lock(mutex);
                if (!initialized || !document_opened || document_dirty || !queued_completion.has_value()) {
                    return false;
                }

                if (last_completion_request.has_value() &&
                    last_completion_request->document_version == queued_completion->document_version &&
                    last_completion_request->line == queued_completion->line &&
                    last_completion_request->character == queued_completion->character &&
                    last_completion_request->manual == queued_completion->manual) {
                    queued_completion.reset();
                    return false;
                }

                request = *queued_completion;
                queued_completion.reset();
                request_id = next_request_id++;
                inflight_completions.emplace(request_id, request);
                last_completion_request = request;
            }

            json payload = {
                {"jsonrpc", "2.0"},
                {"id", request_id},
                {"method", "textDocument/completion"},
                {"params",
                 {{"textDocument", {{"uri", document_uri}}},
                  {"position", {{"line", request.line}, {"character", request.character}}},
                  {"context", {{"triggerKind", 1}}}}}
            };

            if (!process.writeAll(json_rpc_frame(payload))) {
                std::scoped_lock lock(mutex);
                queued_completion = request;
                inflight_completions.erase(request_id);
                last_completion_request.reset();
                return false;
            }

            return true;
        }

        void handle_message(const json& message) {
            if (message.contains("method")) {
                const std::string method = message.value("method", std::string());
                if (method == "window/logMessage" || method == "window/showMessage") {
                    const auto& params = message.value("params", json::object());
                    const std::string text = params.value("message", std::string());
                    if (!text.empty()) {
                        LOG_DEBUG("PythonLspClient: {}", text);
                    }
                }
                return;
            }

            if (!message.contains("id")) {
                return;
            }

            const uint64_t id = message["id"].get<uint64_t>();
            if (id == initialize_request_id) {
                if (message.contains("result")) {
                    initialized = true;
                    ready_once = true;
                    document_opened = false;

                    {
                        std::scoped_lock lock(mutex);
                        state = State::Ready;
                        status = "Python completions ready";
                        document_dirty = true;
                    }

                    json initialized_payload = {
                        {"jsonrpc", "2.0"},
                        {"method", "initialized"},
                        {"params", json::object()},
                    };
                    process.writeAll(json_rpc_frame(initialized_payload));
                } else {
                    LOG_WARN("PythonLspClient: initialize failed");
                    process.kill();
                }
                return;
            }

            auto request_it = inflight_completions.find(id);
            if (request_it == inflight_completions.end()) {
                return;
            }

            CompletionList completion = {
                .document_version = request_it->second.document_version,
                .line = request_it->second.line,
                .character = request_it->second.character,
            };

            inflight_completions.erase(request_it);

            const json& result = message.contains("result") ? message["result"] : json();
            json items = json::array();
            if (result.is_array()) {
                items = result;
            } else if (result.is_object()) {
                completion.is_incomplete = result.value("isIncomplete", false);
                items = result.value("items", json::array());
            }

            if (items.is_array()) {
                completion.items.reserve(items.size());
                for (const auto& item : items) {
                    if (!item.is_object()) {
                        continue;
                    }

                    CompletionItem parsed = {
                        .label = item.value("label", std::string()),
                        .detail = item.value("detail", std::string()),
                        .description = item.value("labelDetails", json::object()).value("description", std::string()),
                        .sort_text = item.value("sortText", item.value("label", std::string())),
                        .filter_text = item.value("filterText", item.value("label", std::string())),
                        .insert_text = item.value("insertText", item.value("label", std::string())),
                        .kind = item.value("kind", 0),
                        .deprecated = item.value("deprecated", false),
                    };

                    if (item.contains("tags") && item["tags"].is_array()) {
                        parsed.deprecated = parsed.deprecated ||
                                            std::find(item["tags"].begin(), item["tags"].end(), 1) !=
                                                item["tags"].end();
                    }

                    if (item.contains("textEdit")) {
                        parsed.text_edit = parse_text_edit(item["textEdit"]);
                    }

                    if (item.contains("additionalTextEdits") && item["additionalTextEdits"].is_array()) {
                        for (const auto& edit : item["additionalTextEdits"]) {
                            if (auto parsed_edit = parse_text_edit(edit)) {
                                parsed.additional_text_edits.push_back(*parsed_edit);
                            }
                        }
                    }

                    if (!parsed.label.empty()) {
                        completion.items.push_back(std::move(parsed));
                    }
                }
            }

            std::ranges::sort(completion.items, [](const CompletionItem& lhs, const CompletionItem& rhs) {
                if (lhs.sort_text != rhs.sort_text) {
                    return lhs.sort_text < rhs.sort_text;
                }
                return lhs.label < rhs.label;
            });

            std::scoped_lock lock(mutex);
            latest_completion = std::move(completion);
        }

        std::vector<ServerCommand> commands;
        fs::path workspace_root;
        fs::path document_path;
        std::string root_uri;
        std::string document_uri;

        mutable std::mutex mutex;
        std::condition_variable_any cv;
        StdioProcess process;

        State state = State::Starting;
        std::string status;

        std::optional<size_t> active_candidate_index;
        size_t attempt_candidate_index = 0;
        std::chrono::steady_clock::time_point next_retry_at;

        bool initialized = false;
        bool ready_once = false;
        bool document_opened = false;
        std::string document_text;
        int document_version = 1;
        bool document_dirty = true;

        uint64_t next_request_id = 1;
        uint64_t initialize_request_id = 0;
        std::unordered_map<uint64_t, PendingCompletionRequest> inflight_completions;
        std::optional<PendingCompletionRequest> queued_completion;
        std::optional<PendingCompletionRequest> last_completion_request;
        std::optional<CompletionList> latest_completion;

        std::string stdout_buffer;
        std::string stderr_buffer;
        std::jthread worker;
    };

    PythonLspClient::PythonLspClient()
        : impl_(std::make_unique<Impl>()) {
    }

    PythonLspClient::~PythonLspClient() = default;

    int PythonLspClient::updateDocument(const std::string& text) {
        return impl_->updateDocument(text);
    }

    void PythonLspClient::requestCompletion(const int document_version,
                                            const int line,
                                            const int character,
                                            const bool manual) {
        impl_->requestCompletion({
            .document_version = document_version,
            .line = line,
            .character = character,
            .manual = manual,
        });
    }

    std::optional<PythonLspClient::CompletionList> PythonLspClient::takeLatestCompletion() {
        return impl_->takeLatestCompletion();
    }

    bool PythonLspClient::isReady() const {
        return impl_->isReady();
    }

    bool PythonLspClient::isAvailable() const {
        return impl_->isAvailable();
    }

    std::string PythonLspClient::statusText() const {
        return impl_->statusText();
    }

} // namespace lfs::vis::editor
