/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "app/mcp_gui_tools.hpp"

#include "core/base64.hpp"
#include "core/event_bridge/command_center_bridge.hpp"
#include "core/logger.hpp"
#include "core/scene.hpp"
#include "core/tensor.hpp"
#include "io/exporter.hpp"
#include "mcp/llm_client.hpp"
#include "mcp/mcp_tools.hpp"
#include "mcp/selection_client.hpp"
#include "python/runner.hpp"
#include "rendering/gs_rasterizer_tensor.hpp"
#include "rendering/rasterizer/rasterization/include/rasterization_api_tensor.h"
#include "visualizer/visualizer.hpp"

#include <stb_image_write.h>

#include <cassert>
#include <future>
#include <string>

namespace lfs::app {

    using json = nlohmann::json;
    using mcp::McpTool;
    using mcp::ToolRegistry;

    namespace {

        void stbi_write_callback(void* context, void* data, int size) {
            auto* buf = static_cast<std::vector<uint8_t>*>(context);
            auto* bytes = static_cast<const uint8_t*>(data);
            buf->insert(buf->end(), bytes, bytes + size);
        }

        std::expected<std::string, std::string> render_scene_to_base64(
            core::Scene& scene,
            int camera_index = 0) {

            auto* model = scene.getTrainingModel();
            if (!model)
                return std::unexpected("No model to render");

            auto cameras = scene.getAllCameras();
            if (cameras.empty())
                return std::unexpected("No cameras available");

            if (camera_index < 0 || camera_index >= static_cast<int>(cameras.size()))
                camera_index = 0;

            auto& camera = cameras[camera_index];
            if (!camera)
                return std::unexpected("Failed to get camera");

            core::Tensor bg = core::Tensor::zeros({3}, core::Device::CUDA);

            try {
                auto [image, alpha] = rendering::rasterize_tensor(*camera, *model, bg);

                image = image.clone().to(core::Device::CPU).to(core::DataType::Float32);
                if (image.ndim() == 4)
                    image = image.squeeze(0);
                if (image.ndim() == 3 && image.shape()[0] <= 4)
                    image = image.permute({1, 2, 0});
                image = (image.clamp(0, 1) * 255.0f).to(core::DataType::UInt8).contiguous();

                const int h = static_cast<int>(image.shape()[0]);
                const int w = static_cast<int>(image.shape()[1]);
                const int c = static_cast<int>(image.shape()[2]);
                assert(c >= 1 && c <= 4);

                std::vector<uint8_t> png_buf;
                png_buf.reserve(static_cast<size_t>(w) * h * c);
                int ok = stbi_write_png_to_func(
                    stbi_write_callback, &png_buf, w, h, c,
                    image.ptr<uint8_t>(), w * c);
                if (!ok)
                    return std::unexpected("PNG encoding failed");

                return core::base64_encode(png_buf);
            } catch (const std::exception& e) {
                return std::unexpected(std::string("Render failed: ") + e.what());
            }
        }

        std::expected<core::Tensor, std::string> compute_screen_positions(
            core::Scene& scene,
            int camera_index = 0) {

            auto* model = scene.getTrainingModel();
            if (!model)
                return std::unexpected("No model loaded");

            auto cameras = scene.getAllCameras();
            if (cameras.empty())
                return std::unexpected("No cameras available");

            if (camera_index < 0 || camera_index >= static_cast<int>(cameras.size()))
                camera_index = 0;

            auto& camera = cameras[camera_index];
            if (!camera)
                return std::unexpected("Failed to get camera");

            core::Tensor bg = core::Tensor::zeros({3}, core::Device::CUDA);
            core::Tensor screen_positions;

            try {
                auto [image, alpha] = rendering::rasterize_tensor(
                    *camera, *model, bg,
                    false, 0.01f,
                    nullptr, nullptr, nullptr,
                    &screen_positions);

                return screen_positions;
            } catch (const std::exception& e) {
                return std::unexpected(std::string("Screen position computation failed: ") + e.what());
            }
        }

        template <typename F>
        auto post_and_wait(vis::Visualizer* viewer, F&& fn) {
            using R = std::invoke_result_t<F>;
            auto task = std::make_shared<std::packaged_task<R()>>(std::forward<F>(fn));
            auto f = task->get_future();
            viewer->postWork([task]() { (*task)(); });
            return f.get();
        }

        json apply_selection_and_count(
            core::Scene& scene,
            const core::Tensor& selection,
            const std::string& mode) {

            const auto N = static_cast<size_t>(selection.shape()[0]);

            auto existing_mask = scene.getSelectionMask();
            if (!existing_mask) {
                existing_mask = std::make_shared<core::Tensor>(
                    core::Tensor::zeros({N}, core::Device::CUDA, core::DataType::UInt8));
            }

            core::Tensor output_mask = core::Tensor::zeros({N}, core::Device::CUDA, core::DataType::UInt8);
            uint32_t locked_groups[8] = {0};

            bool add_mode = (mode != "remove");
            rendering::apply_selection_group_tensor(
                selection, *existing_mask, output_mask,
                1, locked_groups, add_mode);

            scene.setSelectionMask(std::make_shared<core::Tensor>(std::move(output_mask)));

            const auto count = static_cast<int64_t>(scene.getSelectionMask()->count_nonzero());
            return json{{"success", true}, {"selected_count", count}};
        }

    } // namespace

    void register_gui_scene_tools(vis::Visualizer* viewer) {
        assert(viewer);
        auto& registry = ToolRegistry::instance();

        // --- Scene operations (posted to GUI thread) ---

        registry.register_tool(
            McpTool{
                .name = "scene.load_ply",
                .description = "Load a PLY file for viewing",
                .input_schema = {
                    .type = "object",
                    .properties = json{
                        {"path", json{{"type", "string"}, {"description", "Path to PLY file"}}}},
                    .required = {"path"}}},
            [viewer](const json& args) -> json {
                std::filesystem::path path = args["path"].get<std::string>();

                auto result = post_and_wait(viewer, [viewer, path]() {
                    return viewer->loadPLY(path);
                });

                if (!result)
                    return json{{"error", result.error()}};

                return json{{"success", true}, {"path", path.string()}};
            });

        registry.register_tool(
            McpTool{
                .name = "scene.load_dataset",
                .description = "Load a COLMAP dataset for training",
                .input_schema = {
                    .type = "object",
                    .properties = json{
                        {"path", json{{"type", "string"}, {"description", "Path to COLMAP dataset directory"}}},
                        {"images_folder", json{{"type", "string"}, {"description", "Images subfolder (default: images)"}}},
                        {"max_iterations", json{{"type", "integer"}, {"description", "Maximum training iterations (default: 30000)"}}},
                        {"strategy", json{{"type", "string"}, {"enum", json::array({"mcmc", "default"})}, {"description", "Training strategy"}}}},
                    .required = {"path"}}},
            [viewer](const json& args) -> json {
                std::filesystem::path path = args["path"].get<std::string>();

                core::param::TrainingParameters params;
                params.dataset.data_path = path;
                if (args.contains("images_folder"))
                    params.dataset.images = args["images_folder"].get<std::string>();
                if (args.contains("max_iterations"))
                    params.optimization.iterations = args["max_iterations"].get<size_t>();
                if (args.contains("strategy"))
                    params.optimization.strategy = args["strategy"].get<std::string>();

                auto result = post_and_wait(viewer, [viewer, params, path]() {
                    viewer->setParameters(params);
                    return viewer->loadDataset(path);
                });

                if (!result)
                    return json{{"error", result.error()}};

                return json{{"success", true}, {"path", path.string()}};
            });

        registry.register_tool(
            McpTool{
                .name = "scene.load_checkpoint",
                .description = "Load a training checkpoint (.resume file)",
                .input_schema = {
                    .type = "object",
                    .properties = json{
                        {"path", json{{"type", "string"}, {"description", "Path to checkpoint file"}}}},
                    .required = {"path"}}},
            [viewer](const json& args) -> json {
                std::filesystem::path path = args["path"].get<std::string>();

                auto result = post_and_wait(viewer, [viewer, path]() {
                    return viewer->loadCheckpointForTraining(path);
                });

                if (!result)
                    return json{{"error", result.error()}};

                return json{{"success", true}, {"path", path.string()}};
            });

        registry.register_tool(
            McpTool{
                .name = "scene.save_checkpoint",
                .description = "Save current training state to checkpoint (uses configured output path)",
                .input_schema = {.type = "object", .properties = json::object(), .required = {}}},
            [viewer](const json&) -> json {
                auto result = post_and_wait(viewer, [viewer]() {
                    return viewer->saveCheckpoint();
                });

                if (!result)
                    return json{{"error", result.error()}};

                return json{{"success", true}};
            });

        registry.register_tool(
            McpTool{
                .name = "training.start",
                .description = "Start training in background",
                .input_schema = {.type = "object", .properties = json::object(), .required = {}}},
            [viewer](const json&) -> json {
                auto result = post_and_wait(viewer, [viewer]() {
                    return viewer->startTraining();
                });

                if (!result)
                    return json{{"error", result.error()}};

                return json{{"success", true}, {"message", "Training started"}};
            });

        registry.register_tool(
            McpTool{
                .name = "scene.save_ply",
                .description = "Save current model as PLY file",
                .input_schema = {
                    .type = "object",
                    .properties = json{
                        {"path", json{{"type", "string"}, {"description", "Path to save PLY file"}}}},
                    .required = {"path"}}},
            [viewer](const json& args) -> json {
                std::filesystem::path path = args["path"].get<std::string>();

                return post_and_wait(viewer, [viewer, path]() -> json {
                    auto& scene = viewer->getScene();
                    auto* model = scene.getTrainingModel();
                    if (!model)
                        return json{{"error", "No model to save"}};

                    io::PlySaveOptions options{.output_path = path, .binary = true};
                    auto result = io::save_ply(*model, options);
                    if (!result)
                        return json{{"error", result.error().message}};

                    return json{{"success", true}, {"path", path.string()}};
                });
            });

        registry.register_tool(
            McpTool{
                .name = "render.capture",
                .description = "Render current scene and return as base64 PNG",
                .input_schema = {
                    .type = "object",
                    .properties = json{
                        {"camera_index", json{{"type", "integer"}, {"description", "Camera index (default: 0)"}}}},
                    .required = {}}},
            [viewer](const json& args) -> json {
                int camera_index = args.value("camera_index", 0);

                return post_and_wait(viewer, [viewer, camera_index]() -> json {
                    auto result = render_scene_to_base64(viewer->getScene(), camera_index);
                    if (!result)
                        return json{{"error", result.error()}};

                    json response;
                    response["success"] = true;
                    response["mime_type"] = "image/png";
                    response["data"] = *result;
                    return response;
                });
            });

        // --- Selection tools ---

        registry.register_tool(
            McpTool{
                .name = "selection.rect",
                .description = "Select Gaussians inside a screen rectangle",
                .input_schema = {
                    .type = "object",
                    .properties = json{
                        {"x0", json{{"type", "number"}, {"description", "Left edge X coordinate"}}},
                        {"y0", json{{"type", "number"}, {"description", "Top edge Y coordinate"}}},
                        {"x1", json{{"type", "number"}, {"description", "Right edge X coordinate"}}},
                        {"y1", json{{"type", "number"}, {"description", "Bottom edge Y coordinate"}}},
                        {"camera_index", json{{"type", "integer"}, {"description", "Camera index (default: 0)"}}},
                        {"mode", json{{"type", "string"}, {"enum", json::array({"replace", "add", "remove"})}, {"description", "Selection mode (default: replace)"}}}},
                    .required = {"x0", "y0", "x1", "y1"}}},
            [viewer](const json& args) -> json {
                const float x0 = args["x0"].get<float>();
                const float y0 = args["y0"].get<float>();
                const float x1 = args["x1"].get<float>();
                const float y1 = args["y1"].get<float>();
                const std::string mode = args.value("mode", "replace");
                const int camera_index = args.value("camera_index", 0);

                mcp::SelectionClient client;
                if (client.is_gui_running()) {
                    auto result = client.select_rect(x0, y0, x1, y1, mode, camera_index);
                    if (!result)
                        return json{{"error", result.error()}};
                    return json{{"success", true}, {"via_gui", true}};
                }

                return post_and_wait(viewer, [viewer, x0, y0, x1, y1, mode, camera_index]() -> json {
                    auto& scene = viewer->getScene();
                    auto screen_pos_result = compute_screen_positions(scene, camera_index);
                    if (!screen_pos_result)
                        return json{{"error", screen_pos_result.error()}};

                    const auto& screen_positions = *screen_pos_result;
                    const auto N = static_cast<size_t>(screen_positions.shape()[0]);

                    core::Tensor selection = core::Tensor::zeros({N}, core::Device::CUDA, core::DataType::UInt8);

                    if (mode == "replace") {
                        rendering::rect_select_tensor(screen_positions, x0, y0, x1, y1, selection);
                    } else {
                        bool add = (mode == "add");
                        rendering::rect_select_mode_tensor(screen_positions, x0, y0, x1, y1, selection, add);
                    }

                    return apply_selection_and_count(scene, selection, mode);
                });
            });

        registry.register_tool(
            McpTool{
                .name = "selection.polygon",
                .description = "Select Gaussians inside a screen polygon",
                .input_schema = {
                    .type = "object",
                    .properties = json{
                        {"points", json{{"type", "array"}, {"items", json{{"type", "array"}, {"items", json{{"type", "number"}}}}}, {"description", "Polygon vertices [[x0,y0], [x1,y1], ...]"}}},
                        {"camera_index", json{{"type", "integer"}, {"description", "Camera index (default: 0)"}}},
                        {"mode", json{{"type", "string"}, {"enum", json::array({"replace", "add", "remove"})}, {"description", "Selection mode (default: replace)"}}}},
                    .required = {"points"}}},
            [viewer](const json& args) -> json {
                const auto& points = args["points"];
                const size_t num_vertices = points.size();
                if (num_vertices < 3)
                    return json{{"error", "Polygon requires at least 3 vertices"}};

                std::vector<float> vertex_data;
                vertex_data.reserve(num_vertices * 2);
                for (const auto& pt : points) {
                    vertex_data.push_back(pt[0].get<float>());
                    vertex_data.push_back(pt[1].get<float>());
                }

                const std::string mode = args.value("mode", "replace");
                const int camera_index = args.value("camera_index", 0);

                return post_and_wait(viewer, [viewer, vertex_data, num_vertices, mode, camera_index]() -> json {
                    auto& scene = viewer->getScene();
                    auto screen_pos_result = compute_screen_positions(scene, camera_index);
                    if (!screen_pos_result)
                        return json{{"error", screen_pos_result.error()}};

                    core::Tensor polygon_vertices = core::Tensor::from_vector(
                        vertex_data, {num_vertices, 2}, core::Device::CUDA);

                    const auto& screen_positions = *screen_pos_result;
                    const auto N = static_cast<size_t>(screen_positions.shape()[0]);

                    core::Tensor selection = core::Tensor::zeros({N}, core::Device::CUDA, core::DataType::UInt8);

                    if (mode == "replace") {
                        rendering::polygon_select_tensor(screen_positions, polygon_vertices, selection);
                    } else {
                        bool add = (mode == "add");
                        rendering::polygon_select_mode_tensor(screen_positions, polygon_vertices, selection, add);
                    }

                    return apply_selection_and_count(scene, selection, mode);
                });
            });

        registry.register_tool(
            McpTool{
                .name = "selection.click",
                .description = "Select Gaussians near a screen point (brush selection)",
                .input_schema = {
                    .type = "object",
                    .properties = json{
                        {"x", json{{"type", "number"}, {"description", "X coordinate"}}},
                        {"y", json{{"type", "number"}, {"description", "Y coordinate"}}},
                        {"radius", json{{"type", "number"}, {"description", "Selection radius in pixels (default: 20)"}}},
                        {"camera_index", json{{"type", "integer"}, {"description", "Camera index (default: 0)"}}},
                        {"mode", json{{"type", "string"}, {"enum", json::array({"replace", "add", "remove"})}, {"description", "Selection mode (default: replace)"}}}},
                    .required = {"x", "y"}}},
            [viewer](const json& args) -> json {
                const float x = args["x"].get<float>();
                const float y = args["y"].get<float>();
                const float radius = args.value("radius", 20.0f);
                const std::string mode = args.value("mode", "replace");
                const int camera_index = args.value("camera_index", 0);

                return post_and_wait(viewer, [viewer, x, y, radius, mode, camera_index]() -> json {
                    auto& scene = viewer->getScene();
                    auto screen_pos_result = compute_screen_positions(scene, camera_index);
                    if (!screen_pos_result)
                        return json{{"error", screen_pos_result.error()}};

                    const auto& screen_positions = *screen_pos_result;
                    const auto N = static_cast<size_t>(screen_positions.shape()[0]);

                    core::Tensor selection = core::Tensor::zeros({N}, core::Device::CUDA, core::DataType::UInt8);
                    rendering::brush_select_tensor(screen_positions, x, y, radius, selection);

                    return apply_selection_and_count(scene, selection, mode);
                });
            });

        registry.register_tool(
            McpTool{
                .name = "selection.get",
                .description = "Get current selection (returns selected Gaussian indices)",
                .input_schema = {
                    .type = "object",
                    .properties = json{
                        {"max_indices", json{{"type", "integer"}, {"description", "Maximum indices to return (default: 100000)"}}}},
                    .required = {}}},
            [viewer](const json& args) -> json {
                const int max_indices = args.value("max_indices", 100000);

                return post_and_wait(viewer, [viewer, max_indices]() -> json {
                    auto& scene = viewer->getScene();

                    auto mask = scene.getSelectionMask();
                    if (!mask)
                        return json{{"success", true}, {"selected_count", 0}, {"indices", json::array()}};

                    auto mask_vec = mask->to_vector_uint8();

                    int64_t count = 0;
                    std::vector<int64_t> indices;
                    for (size_t i = 0; i < mask_vec.size(); ++i) {
                        if (mask_vec[i] > 0) {
                            ++count;
                            if (static_cast<int>(indices.size()) < max_indices)
                                indices.push_back(static_cast<int64_t>(i));
                        }
                    }

                    json result;
                    result["success"] = true;
                    result["selected_count"] = count;
                    result["indices"] = indices;
                    result["truncated"] = count > static_cast<int64_t>(indices.size());
                    return result;
                });
            });

        registry.register_tool(
            McpTool{
                .name = "selection.clear",
                .description = "Clear all selection",
                .input_schema = {.type = "object", .properties = json::object(), .required = {}}},
            [viewer](const json&) -> json {
                return post_and_wait(viewer, [viewer]() -> json {
                    auto& scene = viewer->getScene();

                    auto* model = scene.getTrainingModel();
                    if (!model)
                        return json{{"error", "No model loaded"}};

                    const auto N = model->size();
                    auto empty_mask = std::make_shared<core::Tensor>(
                        core::Tensor::zeros({N}, core::Device::CUDA, core::DataType::UInt8));
                    scene.setSelectionMask(empty_mask);

                    return json{{"success", true}};
                });
            });

        // --- Plugin tools ---

        registry.register_tool(
            McpTool{
                .name = "plugin.invoke",
                .description = "Invoke a plugin capability by name. Use plugin.list to see available capabilities.",
                .input_schema = {
                    .type = "object",
                    .properties = json{
                        {"capability", json{{"type", "string"}, {"description", "Capability name (e.g., 'selection.by_text')"}}},
                        {"args", json{{"type", "object"}, {"description", "Arguments to pass to the capability"}}}},
                    .required = {"capability"}}},
            [](const json& args) -> json {
                const auto capability = args.value("capability", "");
                if (capability.empty())
                    return json{{"error", "Missing capability name"}};

                mcp::SelectionClient client;
                if (!client.is_gui_running())
                    return json{{"error", "GUI not running"}};

                const std::string args_json = args.contains("args") ? args["args"].dump() : "{}";
                auto result = client.invoke_capability(capability, args_json);
                if (!result)
                    return json{{"error", result.error()}};

                if (!result->success)
                    return json{{"success", false}, {"error", result->error}};

                try {
                    return json::parse(result->result_json);
                } catch (const std::exception& e) {
                    LOG_WARN("Failed to parse capability result: {}", e.what());
                    return json{{"success", true}, {"raw_result", result->result_json}};
                }
            });

        registry.register_tool(
            McpTool{
                .name = "plugin.list",
                .description = "List all registered plugin capabilities",
                .input_schema = {.type = "object", .properties = json::object(), .required = {}}},
            [](const json&) -> json {
                auto capabilities = python::list_capabilities();
                json result = json::array();
                for (const auto& cap : capabilities) {
                    result.push_back({{"name", cap.name}, {"description", cap.description}, {"plugin", cap.plugin_name}});
                }
                return json{{"success", true}, {"capabilities", result}};
            });

        // --- LLM-powered tools ---

        registry.register_tool(
            McpTool{
                .name = "training.ask_advisor",
                .description = "Ask an LLM for training advice based on current state and render",
                .input_schema = {
                    .type = "object",
                    .properties = json{
                        {"problem", json{{"type", "string"}, {"description", "Description of the problem or question"}}},
                        {"include_render", json{{"type", "boolean"}, {"description", "Include current render in request (default: true)"}}},
                        {"camera_index", json{{"type", "integer"}, {"description", "Camera index for render (default: 0)"}}}},
                    .required = {}}},
            [viewer](const json& args) -> json {
                auto api_key = mcp::LLMClient::load_api_key_from_env();
                if (!api_key)
                    return json{{"error", api_key.error()}};

                mcp::LLMClient client;
                client.set_api_key(*api_key);

                auto* cc = event::command_center();
                if (!cc)
                    return json{{"error", "Training system not initialized"}};

                auto snapshot = cc->snapshot();

                std::string base64_render;
                bool include_render = args.value("include_render", true);
                if (include_render) {
                    int camera_index = args.value("camera_index", 0);
                    auto render_result = post_and_wait(viewer, [viewer, camera_index]() {
                        return render_scene_to_base64(viewer->getScene(), camera_index);
                    });
                    if (render_result)
                        base64_render = *render_result;
                }

                std::string problem = args.value("problem", "");

                auto result = mcp::ask_training_advisor(
                    client,
                    snapshot.iteration,
                    snapshot.loss,
                    snapshot.num_gaussians,
                    base64_render,
                    problem);

                if (!result)
                    return json{{"error", result.error()}};

                json response;
                response["success"] = result->success;
                response["advice"] = result->content;
                response["model"] = result->model;
                response["input_tokens"] = result->input_tokens;
                response["output_tokens"] = result->output_tokens;
                if (!result->success)
                    response["error"] = result->error;
                return response;
            });

        registry.register_tool(
            McpTool{
                .name = "selection.by_description",
                .description = "Select Gaussians by natural language description using LLM vision",
                .input_schema = {
                    .type = "object",
                    .properties = json{
                        {"description", json{{"type", "string"}, {"description", "Natural language description of what to select (e.g., 'the bicycle wheel')"}}},
                        {"camera_index", json{{"type", "integer"}, {"description", "Camera index for rendering (default: 0)"}}}},
                    .required = {"description"}}},
            [viewer](const json& args) -> json {
                auto api_key = mcp::LLMClient::load_api_key_from_env();
                if (!api_key)
                    return json{{"error", api_key.error()}};

                const int camera_index = args.value("camera_index", 0);
                const std::string description = args["description"].get<std::string>();

                auto render_result = post_and_wait(viewer, [viewer, camera_index]() {
                    return render_scene_to_base64(viewer->getScene(), camera_index);
                });
                if (!render_result)
                    return json{{"error", render_result.error()}};

                mcp::LLMClient client;
                client.set_api_key(*api_key);

                mcp::LLMRequest request;
                request.prompt = "Look at this 3D scene render. I need you to identify the bounding box for: \"" + description + "\"\n\n"
                                                                                                                                 "Return ONLY a JSON object with the bounding box coordinates in pixel space:\n"
                                                                                                                                 "{\"x0\": <left>, \"y0\": <top>, \"x1\": <right>, \"y1\": <bottom>}\n\n"
                                                                                                                                 "The coordinates should be integers representing pixel positions. "
                                                                                                                                 "If you cannot identify the object, return: {\"error\": \"Object not found\"}";
                request.attachments.push_back(mcp::ImageAttachment{.base64_data = *render_result, .media_type = "image/png"});
                request.temperature = 0.0f;
                request.max_tokens = 256;

                auto response = client.complete(request);
                if (!response)
                    return json{{"error", response.error()}};

                if (!response->success)
                    return json{{"error", response->error}};

                json bbox;
                try {
                    auto content = response->content;
                    auto json_start = content.find('{');
                    auto json_end = content.rfind('}');
                    if (json_start == std::string::npos || json_end == std::string::npos)
                        return json{{"error", "LLM response did not contain valid JSON"}};
                    bbox = json::parse(content.substr(json_start, json_end - json_start + 1));
                } catch (const std::exception& e) {
                    return json{{"error", std::string("Failed to parse LLM response: ") + e.what()}};
                }

                if (bbox.contains("error"))
                    return json{{"error", bbox["error"].get<std::string>()}};

                if (!bbox.contains("x0") || !bbox.contains("y0") || !bbox.contains("x1") || !bbox.contains("y1"))
                    return json{{"error", "LLM response missing bounding box coordinates"}};

                const float x0 = bbox["x0"].get<float>();
                const float y0 = bbox["y0"].get<float>();
                const float x1 = bbox["x1"].get<float>();
                const float y1 = bbox["y1"].get<float>();

                mcp::SelectionClient selection_client;
                if (selection_client.is_gui_running()) {
                    auto sel_result = selection_client.select_rect(x0, y0, x1, y1, "replace", camera_index);
                    if (!sel_result)
                        return json{{"error", sel_result.error()}};

                    json gui_response;
                    gui_response["success"] = true;
                    gui_response["via_gui"] = true;
                    gui_response["bounding_box"] = bbox;
                    gui_response["description"] = description;
                    return gui_response;
                }

                return post_and_wait(viewer, [viewer, x0, y0, x1, y1, camera_index, bbox, description]() -> json {
                    auto& scene = viewer->getScene();
                    auto screen_pos_result = compute_screen_positions(scene, camera_index);
                    if (!screen_pos_result)
                        return json{{"error", screen_pos_result.error()}};

                    const auto& screen_positions = *screen_pos_result;
                    const auto N = static_cast<size_t>(screen_positions.shape()[0]);

                    core::Tensor selection = core::Tensor::zeros({N}, core::Device::CUDA, core::DataType::UInt8);
                    rendering::rect_select_tensor(screen_positions, x0, y0, x1, y1, selection);

                    auto result = apply_selection_and_count(scene, selection, "replace");
                    result["bounding_box"] = bbox;
                    result["description"] = description;
                    return result;
                });
            });

        LOG_INFO("Registered GUI-native MCP scene tools");
    }

} // namespace lfs::app
