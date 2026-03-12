/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#define GLM_ENABLE_EXPERIMENTAL

#include "app/mcp_gui_tools.hpp"

#include "core/base64.hpp"
#include "core/event_bridge/command_center_bridge.hpp"
#include "core/events.hpp"
#include "core/logger.hpp"
#include "core/scene.hpp"
#include "core/splat_data_transform.hpp"
#include "core/tensor.hpp"
#include "io/exporter.hpp"
#include "mcp/llm_client.hpp"
#include "mcp/mcp_tools.hpp"
#include "python/python_runtime.hpp"
#include "python/runner.hpp"
#include "rendering/gs_rasterizer_tensor.hpp"
#include "visualizer/gui/panels/python_console_panel.hpp"
#include "visualizer/operation/undo_entry.hpp"
#include "visualizer/operation/undo_history.hpp"
#include "visualizer/gui_capabilities.hpp"
#include "visualizer/rendering/rendering_manager.hpp"
#include "visualizer/scene/scene_manager.hpp"
#include "visualizer/visualizer.hpp"

#include <stb_image_write.h>

#include <atomic>
#include <cassert>
#include <future>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/euler_angles.hpp>

namespace lfs::app {

    using json = nlohmann::json;
    using mcp::McpResource;
    using mcp::McpResourceContent;
    using mcp::McpTool;
    using mcp::ResourceRegistry;
    using mcp::ToolRegistry;

    namespace {

        template <typename T>
        struct dependent_false : std::false_type {};

        template <typename T>
        struct is_string_expected : std::false_type {};

        template <typename T>
        struct is_string_expected<std::expected<T, std::string>> : std::true_type {};

        using TransformComponents = vis::cap::TransformComponents;

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

        std::expected<std::vector<McpResourceContent>, std::string> single_json_resource(
            const std::string& uri,
            json payload) {
            return std::vector<McpResourceContent>{
                McpResourceContent{
                    .uri = uri,
                    .mime_type = "application/json",
                    .content = payload.dump(2)}};
        }

        std::expected<std::vector<McpResourceContent>, std::string> single_blob_resource(
            const std::string& uri,
            const std::string& mime_type,
            std::string base64_payload) {
            return std::vector<McpResourceContent>{
                McpResourceContent{
                    .uri = uri,
                    .mime_type = mime_type,
                    .content = std::move(base64_payload)}};
        }

        json selection_state_json(core::Scene& scene, const int max_indices = 100000) {
            auto mask = scene.getSelectionMask();
            if (!mask)
                return json{{"selected_count", 0}, {"indices", json::array()}, {"truncated", false}};

            auto mask_vec = mask->to_vector_uint8();

            int64_t count = 0;
            std::vector<int64_t> indices;
            for (size_t i = 0; i < mask_vec.size(); ++i) {
                if (mask_vec[i] == 0)
                    continue;
                ++count;
                if (static_cast<int>(indices.size()) < max_indices)
                    indices.push_back(static_cast<int64_t>(i));
            }

            return json{
                {"selected_count", count},
                {"indices", indices},
                {"truncated", count > static_cast<int64_t>(indices.size())}};
        }

        template <typename R>
        R make_post_failure(const std::string& error) {
            if constexpr (std::is_same_v<R, json>) {
                return json{{"error", error}};
            } else if constexpr (is_string_expected<R>::value) {
                return std::unexpected(error);
            } else {
                static_assert(dependent_false<R>::value, "Unsupported post_and_wait return type");
            }
        }

        template <typename F>
        auto post_and_wait(vis::Visualizer* viewer, F&& fn) {
            using R = std::invoke_result_t<F>;
            constexpr const char* shutdown_error = "Viewer is shutting down";

            auto task = std::make_shared<std::decay_t<F>>(std::forward<F>(fn));
            auto promise = std::make_shared<std::promise<R>>();
            auto completed = std::make_shared<std::atomic_bool>(false);
            auto future = promise->get_future();

            auto finish_with_value = [promise, completed](auto&& value) mutable {
                if (!completed->exchange(true))
                    promise->set_value(std::forward<decltype(value)>(value));
            };
            auto finish_with_exception = [promise, completed](std::exception_ptr error) {
                if (!completed->exchange(true))
                    promise->set_exception(std::move(error));
            };

            const bool posted = viewer->postWork(vis::Visualizer::WorkItem{
                .run =
                    [task, finish_with_value, finish_with_exception]() mutable {
                        try {
                            finish_with_value(std::invoke(*task));
                        } catch (...) {
                            finish_with_exception(std::current_exception());
                        }
                    },
                .cancel =
                    [finish_with_value]() mutable {
                        finish_with_value(make_post_failure<R>(shutdown_error));
                    }});

            if (!posted)
                return make_post_failure<R>(shutdown_error);

            return future.get();
        }

        json vec3_to_json(const glm::vec3& value) {
            return json::array({value.x, value.y, value.z});
        }

        json mat4_to_json(const glm::mat4& value) {
            return json::array({
                json::array({value[0][0], value[1][0], value[2][0], value[3][0]}),
                json::array({value[0][1], value[1][1], value[2][1], value[3][1]}),
                json::array({value[0][2], value[1][2], value[2][2], value[3][2]}),
                json::array({value[0][3], value[1][3], value[2][3], value[3][3]}),
            });
        }

        const char* node_type_to_string(const core::NodeType type) {
            switch (type) {
            case core::NodeType::SPLAT:
                return "splat";
            case core::NodeType::POINTCLOUD:
                return "pointcloud";
            case core::NodeType::GROUP:
                return "group";
            case core::NodeType::CROPBOX:
                return "crop_box";
            case core::NodeType::ELLIPSOID:
                return "ellipsoid";
            case core::NodeType::DATASET:
                return "dataset";
            case core::NodeType::CAMERA_GROUP:
                return "camera_group";
            case core::NodeType::CAMERA:
                return "camera";
            case core::NodeType::IMAGE_GROUP:
                return "image_group";
            case core::NodeType::IMAGE:
                return "image";
            case core::NodeType::MESH:
                return "mesh";
            case core::NodeType::KEYFRAME_GROUP:
                return "keyframe_group";
            case core::NodeType::KEYFRAME:
                return "keyframe";
            }
            return "unknown";
        }

        TransformComponents decompose_transform(const glm::mat4& matrix) {
            return vis::cap::decomposeTransform(matrix);
        }

        glm::mat4 compose_transform(const TransformComponents& components) {
            return vis::cap::composeTransform(components);
        }

        int64_t selected_gaussian_count(const core::Scene& scene) {
            const auto mask = scene.getSelectionMask();
            if (!mask || !mask->is_valid())
                return 0;
            return static_cast<int64_t>(mask->count_nonzero());
        }

        json node_summary_json(const core::Scene& scene, const core::SceneNode& node) {
            json result{
                {"name", node.name},
                {"type", node_type_to_string(node.type)},
                {"visible", static_cast<bool>(node.visible)},
                {"locked", static_cast<bool>(node.locked)},
                {"gaussian_count", node.gaussian_count},
            };

            if (node.parent_id != core::NULL_NODE) {
                if (const auto* const parent = scene.getNodeById(node.parent_id)) {
                    result["parent"] = parent->name;
                }
            }

            if (node.type == core::NodeType::SPLAT || node.type == core::NodeType::POINTCLOUD) {
                result["has_crop_box"] = scene.getCropBoxForSplat(node.id) != core::NULL_NODE;
                result["has_ellipsoid"] = scene.getEllipsoidForSplat(node.id) != core::NULL_NODE;
            }

            return result;
        }

        json transform_info_json(const core::Scene& scene, const core::SceneNode& node) {
            const glm::mat4 local = scene.getNodeTransform(node.name);
            const glm::mat4 world = scene.getWorldTransform(node.id);
            const auto local_components = decompose_transform(local);
            const auto world_components = decompose_transform(world);

            return json{
                {"name", node.name},
                {"type", node_type_to_string(node.type)},
                {"local", json{
                              {"translation", vec3_to_json(local_components.translation)},
                              {"rotation", vec3_to_json(local_components.rotation)},
                              {"scale", vec3_to_json(local_components.scale)},
                              {"matrix", mat4_to_json(local)},
                          }},
                {"world", json{
                              {"translation", vec3_to_json(world_components.translation)},
                              {"rotation", vec3_to_json(world_components.rotation)},
                              {"scale", vec3_to_json(world_components.scale)},
                              {"matrix", mat4_to_json(world)},
                          }},
            };
        }

        json selection_result_json(const vis::SceneManager& scene_manager, const vis::SelectionResult& result) {
            if (!result.success)
                return json{{"error", result.error}};

            return json{
                {"success", true},
                {"affected_count", static_cast<int64_t>(result.affected_count)},
                {"selected_count", selected_gaussian_count(scene_manager.getScene())},
            };
        }

        std::optional<std::string> optional_string_arg(const json& args, const char* key) {
            if (!args.contains(key) || args[key].is_null())
                return std::nullopt;
            return args[key].get<std::string>();
        }

        std::expected<std::optional<glm::vec3>, std::string> optional_vec3_arg(const json& args, const char* key) {
            if (!args.contains(key) || args[key].is_null())
                return std::optional<glm::vec3>{};

            const auto& value = args[key];
            if (!value.is_array() || value.size() != 3)
                return std::unexpected(std::string("Field '") + key + "' must be a 3-element array");

            return glm::vec3(value[0].get<float>(), value[1].get<float>(), value[2].get<float>());
        }

        void show_python_console() {
            core::events::cmd::ShowWindow{.window_name = "python_console", .show = true}.emit();
        }

        json text_payload_json(std::string text, const int max_chars, const bool tail = true) {
            if (max_chars >= 0 && static_cast<int>(text.size()) > max_chars) {
                const size_t keep = static_cast<size_t>(max_chars);
                if (tail) {
                    text = text.substr(text.size() - keep);
                } else {
                    text.resize(keep);
                }
                return json{
                    {"text", std::move(text)},
                    {"truncated", true},
                };
            }

            return json{
                {"text", std::move(text)},
                {"truncated", false},
            };
        }

        std::expected<std::vector<std::string>, std::string> resolve_transform_targets(
            const vis::SceneManager& scene_manager,
            const std::optional<std::string>& requested_node) {
            return vis::cap::resolveTransformTargets(scene_manager, requested_node);
        }

        std::expected<core::NodeId, std::string> resolve_cropbox_parent_id(
            const vis::SceneManager& scene_manager,
            const std::optional<std::string>& requested_node) {
            return vis::cap::resolveCropBoxParentId(scene_manager, requested_node);
        }

        std::expected<core::NodeId, std::string> resolve_cropbox_id(
            const vis::SceneManager& scene_manager,
            const std::optional<std::string>& requested_node) {
            return vis::cap::resolveCropBoxId(scene_manager, requested_node);
        }

        json crop_box_info_json(const vis::SceneManager& scene_manager,
                                const vis::RenderingManager* rendering_manager,
                                const core::NodeId cropbox_id) {
            const auto& scene = scene_manager.getScene();
            const auto* const node = scene.getNodeById(cropbox_id);
            assert(node && node->cropbox);

            const auto components = decompose_transform(scene.getNodeTransform(node->name));
            json crop_box{
                {"node", node->name},
                {"type", node_type_to_string(node->type)},
                {"min", vec3_to_json(node->cropbox->min)},
                {"max", vec3_to_json(node->cropbox->max)},
                {"inverse", node->cropbox->inverse},
                {"enabled", node->cropbox->enabled},
                {"translation", vec3_to_json(components.translation)},
                {"rotation", vec3_to_json(components.rotation)},
                {"scale", vec3_to_json(components.scale)},
                {"local_matrix", mat4_to_json(scene.getNodeTransform(node->name))},
                {"world_matrix", mat4_to_json(scene.getWorldTransform(cropbox_id))},
            };

            if (node->parent_id != core::NULL_NODE) {
                if (const auto* const parent = scene.getNodeById(node->parent_id)) {
                    crop_box["parent"] = parent->name;
                }
            }

            if (rendering_manager) {
                const auto settings = rendering_manager->getSettings();
                crop_box["show"] = settings.show_crop_box;
                crop_box["use"] = settings.use_crop_box;
            }

            return json{{"success", true}, {"crop_box", crop_box}};
        }

        std::expected<core::NodeId, std::string> ensure_cropbox(
            vis::SceneManager& scene_manager,
            vis::RenderingManager* rendering_manager,
            const core::NodeId parent_id) {
            return vis::cap::ensureCropBox(scene_manager, rendering_manager, parent_id);
        }

        std::expected<void, std::string> fit_cropbox_to_parent(
            vis::SceneManager& scene_manager,
            vis::RenderingManager* rendering_manager,
            const core::NodeId cropbox_id,
            const bool use_percentile) {
            return vis::cap::fitCropBoxToParent(scene_manager, rendering_manager, cropbox_id, use_percentile);
        }

        std::expected<void, std::string> reset_cropbox(
            vis::SceneManager& scene_manager,
            vis::RenderingManager* rendering_manager,
            const core::NodeId cropbox_id) {
            return vis::cap::resetCropBox(scene_manager, rendering_manager, cropbox_id);
        }

    } // namespace

    void register_gui_scene_tools(vis::Visualizer* viewer) {
        assert(viewer);
        auto* const viewer_impl = viewer;
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

                auto immediate_params = params;
                immediate_params.dataset.data_path.clear();

                auto result = post_and_wait(viewer, [viewer, params = std::move(immediate_params), path]() {
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
            [viewer_impl](const json& args) -> json {
                const float x0 = args["x0"].get<float>();
                const float y0 = args["y0"].get<float>();
                const float x1 = args["x1"].get<float>();
                const float y1 = args["y1"].get<float>();
                const std::string mode = args.value("mode", "replace");
                const int camera_index = args.value("camera_index", 0);

                return post_and_wait(viewer_impl, [viewer_impl, x0, y0, x1, y1, mode, camera_index]() -> json {
                    auto* const scene_manager = viewer_impl->getSceneManager();
                    if (!scene_manager)
                        return json{{"error", "Scene manager not initialized"}};
                    return selection_result_json(*scene_manager,
                                                 scene_manager->selectRect(x0, y0, x1, y1, mode, camera_index));
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
            [viewer_impl](const json& args) -> json {
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

                return post_and_wait(viewer_impl, [viewer_impl, vertex_data = std::move(vertex_data), mode, camera_index]() -> json {
                    auto* const scene_manager = viewer_impl->getSceneManager();
                    if (!scene_manager)
                        return json{{"error", "Scene manager not initialized"}};
                    return selection_result_json(*scene_manager,
                                                 scene_manager->selectPolygon(vertex_data, mode, camera_index));
                });
            });

        registry.register_tool(
            McpTool{
                .name = "selection.lasso",
                .description = "Select Gaussians inside a screen-space lasso path",
                .input_schema = {
                    .type = "object",
                    .properties = json{
                        {"points", json{{"type", "array"}, {"items", json{{"type", "array"}, {"items", json{{"type", "number"}}}}}, {"description", "Lasso points [[x0,y0], [x1,y1], ...]"}}},
                        {"camera_index", json{{"type", "integer"}, {"description", "Camera index (default: 0)"}}},
                        {"mode", json{{"type", "string"}, {"enum", json::array({"replace", "add", "remove"})}, {"description", "Selection mode (default: replace)"}}}},
                    .required = {"points"}}},
            [viewer_impl](const json& args) -> json {
                const auto& points = args["points"];
                const size_t num_vertices = points.size();
                if (num_vertices < 3)
                    return json{{"error", "Lasso requires at least 3 points"}};

                std::vector<float> vertex_data;
                vertex_data.reserve(num_vertices * 2);
                for (const auto& pt : points) {
                    vertex_data.push_back(pt[0].get<float>());
                    vertex_data.push_back(pt[1].get<float>());
                }

                const std::string mode = args.value("mode", "replace");
                const int camera_index = args.value("camera_index", 0);

                return post_and_wait(viewer_impl, [viewer_impl, vertex_data = std::move(vertex_data), mode, camera_index]() -> json {
                    auto* const scene_manager = viewer_impl->getSceneManager();
                    if (!scene_manager)
                        return json{{"error", "Scene manager not initialized"}};
                    return selection_result_json(*scene_manager,
                                                 scene_manager->selectLasso(vertex_data, mode, camera_index));
                });
            });

        registry.register_tool(
            McpTool{
                .name = "selection.ring",
                .description = "Select the front-most Gaussian under a screen point using ring-mode picking",
                .input_schema = {
                    .type = "object",
                    .properties = json{
                        {"x", json{{"type", "number"}, {"description", "X coordinate"}}},
                        {"y", json{{"type", "number"}, {"description", "Y coordinate"}}},
                        {"camera_index", json{{"type", "integer"}, {"description", "Camera index (default: 0)"}}},
                        {"mode", json{{"type", "string"}, {"enum", json::array({"replace", "add", "remove"})}, {"description", "Selection mode (default: replace)"}}}},
                    .required = {"x", "y"}}},
            [viewer_impl](const json& args) -> json {
                const float x = args["x"].get<float>();
                const float y = args["y"].get<float>();
                const std::string mode = args.value("mode", "replace");
                const int camera_index = args.value("camera_index", 0);

                return post_and_wait(viewer_impl, [viewer_impl, x, y, mode, camera_index]() -> json {
                    auto* const scene_manager = viewer_impl->getSceneManager();
                    if (!scene_manager)
                        return json{{"error", "Scene manager not initialized"}};
                    return selection_result_json(*scene_manager,
                                                 scene_manager->selectRing(x, y, mode, camera_index));
                });
            });

        registry.register_tool(
            McpTool{
                .name = "selection.brush",
                .description = "Select Gaussians near a screen point using a brush radius",
                .input_schema = {
                    .type = "object",
                    .properties = json{
                        {"x", json{{"type", "number"}, {"description", "X coordinate"}}},
                        {"y", json{{"type", "number"}, {"description", "Y coordinate"}}},
                        {"radius", json{{"type", "number"}, {"description", "Selection radius in pixels (default: 20)"}}},
                        {"camera_index", json{{"type", "integer"}, {"description", "Camera index (default: 0)"}}},
                        {"mode", json{{"type", "string"}, {"enum", json::array({"replace", "add", "remove"})}, {"description", "Selection mode (default: replace)"}}}},
                    .required = {"x", "y"}}},
            [viewer_impl](const json& args) -> json {
                const float x = args["x"].get<float>();
                const float y = args["y"].get<float>();
                const float radius = args.value("radius", 20.0f);
                const std::string mode = args.value("mode", "replace");
                const int camera_index = args.value("camera_index", 0);

                return post_and_wait(viewer_impl, [viewer_impl, x, y, radius, mode, camera_index]() -> json {
                    auto* const scene_manager = viewer_impl->getSceneManager();
                    if (!scene_manager)
                        return json{{"error", "Scene manager not initialized"}};
                    return selection_result_json(*scene_manager,
                                                 scene_manager->selectBrush(x, y, radius, mode, camera_index));
                });
            });

        registry.register_tool(
            McpTool{
                .name = "selection.click",
                .description = "Alias for selection.brush",
                .input_schema = {
                    .type = "object",
                    .properties = json{
                        {"x", json{{"type", "number"}, {"description", "X coordinate"}}},
                        {"y", json{{"type", "number"}, {"description", "Y coordinate"}}},
                        {"radius", json{{"type", "number"}, {"description", "Selection radius in pixels (default: 20)"}}},
                        {"camera_index", json{{"type", "integer"}, {"description", "Camera index (default: 0)"}}},
                        {"mode", json{{"type", "string"}, {"enum", json::array({"replace", "add", "remove"})}, {"description", "Selection mode (default: replace)"}}}},
                    .required = {"x", "y"}}},
            [viewer_impl](const json& args) -> json {
                const float x = args["x"].get<float>();
                const float y = args["y"].get<float>();
                const float radius = args.value("radius", 20.0f);
                const std::string mode = args.value("mode", "replace");
                const int camera_index = args.value("camera_index", 0);

                return post_and_wait(viewer_impl, [viewer_impl, x, y, radius, mode, camera_index]() -> json {
                    auto* const scene_manager = viewer_impl->getSceneManager();
                    if (!scene_manager)
                        return json{{"error", "Scene manager not initialized"}};
                    return selection_result_json(*scene_manager,
                                                 scene_manager->selectBrush(x, y, radius, mode, camera_index));
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
                    const auto selection = vis::cap::getSelectionSnapshot(viewer->getScene(), max_indices);
                    return json{
                        {"success", true},
                        {"selected_count", selection.selected_count},
                        {"indices", selection.indices},
                        {"truncated", selection.truncated},
                    };
                });
            });

        registry.register_tool(
            McpTool{
                .name = "selection.clear",
                .description = "Clear all selection",
                .input_schema = {.type = "object", .properties = json::object(), .required = {}}},
            [viewer_impl](const json&) -> json {
                return post_and_wait(viewer_impl, [viewer_impl]() -> json {
                    auto* const scene_manager = viewer_impl->getSceneManager();
                    if (!scene_manager)
                        return json{{"error", "Scene manager not initialized"}};
                    if (auto result = vis::cap::clearGaussianSelection(*scene_manager); !result)
                        return json{{"error", result.error()}};
                    return json{{"success", true}, {"selected_count", 0}};
                });
            });

        registry.register_tool(
            McpTool{
                .name = "scene.list_nodes",
                .description = "List scene nodes visible to the shared GUI scene",
                .input_schema = {
                    .type = "object",
                    .properties = json{
                        {"include_hidden", json{{"type", "boolean"}, {"description", "Include hidden nodes (default: true)"}}},
                        {"include_auxiliary", json{{"type", "boolean"}, {"description", "Include helper nodes like crop boxes, ellipsoids, cameras, and keyframes (default: true)"}}}},
                    .required = {}}},
            [viewer_impl](const json& args) -> json {
                const bool include_hidden = args.value("include_hidden", true);
                const bool include_auxiliary = args.value("include_auxiliary", true);

                return post_and_wait(viewer_impl, [viewer_impl, include_hidden, include_auxiliary]() -> json {
                    auto* const scene_manager = viewer_impl->getSceneManager();
                    if (!scene_manager)
                        return json{{"error", "Scene manager not initialized"}};

                    const auto& scene = scene_manager->getScene();
                    json nodes = json::array();
                    for (const auto* const node : scene.getNodes()) {
                        if (!node)
                            continue;
                        if (!include_hidden && !static_cast<bool>(node->visible))
                            continue;
                        if (!include_auxiliary) {
                            switch (node->type) {
                            case core::NodeType::CROPBOX:
                            case core::NodeType::ELLIPSOID:
                            case core::NodeType::CAMERA_GROUP:
                            case core::NodeType::CAMERA:
                            case core::NodeType::IMAGE_GROUP:
                            case core::NodeType::IMAGE:
                            case core::NodeType::KEYFRAME_GROUP:
                            case core::NodeType::KEYFRAME:
                                continue;
                            default:
                                break;
                            }
                        }
                        nodes.push_back(node_summary_json(scene, *node));
                    }

                    return json{{"success", true}, {"count", nodes.size()}, {"nodes", nodes}};
                });
            });

        registry.register_tool(
            McpTool{
                .name = "scene.get_selected_nodes",
                .description = "Get the current shared node selection from the GUI scene",
                .input_schema = {.type = "object", .properties = json::object(), .required = {}}},
            [viewer_impl](const json&) -> json {
                return post_and_wait(viewer_impl, [viewer_impl]() -> json {
                    auto* const scene_manager = viewer_impl->getSceneManager();
                    if (!scene_manager)
                        return json{{"error", "Scene manager not initialized"}};

                    const auto& scene = scene_manager->getScene();
                    json nodes = json::array();
                    for (const auto& name : scene_manager->getSelectedNodeNames()) {
                        if (const auto* const node = scene.getNode(name))
                            nodes.push_back(node_summary_json(scene, *node));
                    }

                    return json{{"success", true}, {"count", nodes.size()}, {"nodes", nodes}};
                });
            });

        registry.register_tool(
            McpTool{
                .name = "scene.select_node",
                .description = "Change the shared GUI node selection",
                .input_schema = {
                    .type = "object",
                    .properties = json{
                        {"name", json{{"type", "string"}, {"description", "Node name to select"}}},
                        {"mode", json{{"type", "string"}, {"enum", json::array({"replace", "add"})}, {"description", "Selection update mode (default: replace)"}}}},
                    .required = {"name"}}},
            [viewer_impl](const json& args) -> json {
                const std::string name = args["name"].get<std::string>();
                const std::string mode = args.value("mode", "replace");

                return post_and_wait(viewer_impl, [viewer_impl, name, mode]() -> json {
                    auto* const scene_manager = viewer_impl->getSceneManager();
                    if (!scene_manager)
                        return json{{"error", "Scene manager not initialized"}};

                    if (auto result = vis::cap::selectNode(*scene_manager, name, mode); !result)
                        return json{{"error", result.error()}};

                    json nodes = json::array();
                    for (const auto& selected_name : scene_manager->getSelectedNodeNames()) {
                        if (const auto* const node = scene_manager->getScene().getNode(selected_name))
                            nodes.push_back(node_summary_json(scene_manager->getScene(), *node));
                    }

                    return json{{"success", true}, {"count", nodes.size()}, {"nodes", nodes}};
                });
            });

        registry.register_tool(
            McpTool{
                .name = "transform.get",
                .description = "Inspect local and world transforms for a node or the current shared node selection",
                .input_schema = {
                    .type = "object",
                    .properties = json{
                        {"node", json{{"type", "string"}, {"description", "Optional node name; defaults to the current selected node(s)"}}}},
                    .required = {}}},
            [viewer_impl](const json& args) -> json {
                const auto requested_node = optional_string_arg(args, "node");

                return post_and_wait(viewer_impl, [viewer_impl, requested_node]() -> json {
                    auto* const scene_manager = viewer_impl->getSceneManager();
                    if (!scene_manager)
                        return json{{"error", "Scene manager not initialized"}};

                    auto targets = resolve_transform_targets(*scene_manager, requested_node);
                    if (!targets)
                        return json{{"error", targets.error()}};

                    json nodes = json::array();
                    for (const auto& name : *targets) {
                        if (const auto* const node = scene_manager->getScene().getNode(name))
                            nodes.push_back(transform_info_json(scene_manager->getScene(), *node));
                    }

                    return json{{"success", true}, {"count", nodes.size()}, {"nodes", nodes}};
                });
            });

        registry.register_tool(
            McpTool{
                .name = "transform.set",
                .description = "Set absolute local transform components for a node or the current shared node selection",
                .input_schema = {
                    .type = "object",
                    .properties = json{
                        {"node", json{{"type", "string"}, {"description", "Optional node name; defaults to the current selected node(s)"}}},
                        {"translation", json{{"type", "array"}, {"items", json{{"type", "number"}}}, {"description", "Optional XYZ translation"}}},
                        {"rotation", json{{"type", "array"}, {"items", json{{"type", "number"}}}, {"description", "Optional XYZ Euler rotation in radians"}}},
                        {"scale", json{{"type", "array"}, {"items", json{{"type", "number"}}}, {"description", "Optional XYZ scale"}}}},
                    .required = {}}},
            [viewer_impl](const json& args) -> json {
                const auto requested_node = optional_string_arg(args, "node");
                auto translation = optional_vec3_arg(args, "translation");
                if (!translation)
                    return json{{"error", translation.error()}};
                auto rotation = optional_vec3_arg(args, "rotation");
                if (!rotation)
                    return json{{"error", rotation.error()}};
                auto scale = optional_vec3_arg(args, "scale");
                if (!scale)
                    return json{{"error", scale.error()}};
                if (!translation->has_value() && !rotation->has_value() && !scale->has_value())
                    return json{{"error", "At least one of translation, rotation, or scale must be provided"}};

                return post_and_wait(viewer_impl, [viewer_impl, requested_node, translation = *translation, rotation = *rotation, scale = *scale]() -> json {
                    auto* const scene_manager = viewer_impl->getSceneManager();
                    if (!scene_manager)
                        return json{{"error", "Scene manager not initialized"}};

                    auto targets = resolve_transform_targets(*scene_manager, requested_node);
                    if (!targets)
                        return json{{"error", targets.error()}};

                    if (auto result = vis::cap::setTransform(
                            *scene_manager, *targets, translation, rotation, scale, "mcp.transform.set");
                        !result) {
                        return json{{"error", result.error()}};
                    }

                    json nodes = json::array();
                    for (const auto& name : *targets) {
                        if (const auto* const node = scene_manager->getScene().getNode(name))
                            nodes.push_back(transform_info_json(scene_manager->getScene(), *node));
                    }

                    return json{{"success", true}, {"count", nodes.size()}, {"nodes", nodes}};
                });
            });

        registry.register_tool(
            McpTool{
                .name = "transform.translate",
                .description = "Translate a node or the current shared node selection",
                .input_schema = {
                    .type = "object",
                    .properties = json{
                        {"node", json{{"type", "string"}, {"description", "Optional node name; defaults to the current selected node(s)"}}},
                        {"value", json{{"type", "array"}, {"items", json{{"type", "number"}}}, {"description", "XYZ translation delta"}}}},
                    .required = {"value"}}},
            [viewer_impl](const json& args) -> json {
                const auto requested_node = optional_string_arg(args, "node");
                auto value = optional_vec3_arg(args, "value");
                if (!value)
                    return json{{"error", value.error()}};
                if (!value->has_value())
                    return json{{"error", "Field 'value' must be provided"}};

                return post_and_wait(viewer_impl, [viewer_impl, requested_node, value = **value]() -> json {
                    auto* const scene_manager = viewer_impl->getSceneManager();
                    if (!scene_manager)
                        return json{{"error", "Scene manager not initialized"}};

                    auto targets = resolve_transform_targets(*scene_manager, requested_node);
                    if (!targets)
                        return json{{"error", targets.error()}};

                    if (auto result = vis::cap::translateNodes(
                            *scene_manager, *targets, value, "mcp.transform.translate");
                        !result) {
                        return json{{"error", result.error()}};
                    }

                    json nodes = json::array();
                    for (const auto& name : *targets) {
                        if (const auto* const node = scene_manager->getScene().getNode(name))
                            nodes.push_back(transform_info_json(scene_manager->getScene(), *node));
                    }

                    return json{{"success", true}, {"count", nodes.size()}, {"nodes", nodes}};
                });
            });

        registry.register_tool(
            McpTool{
                .name = "transform.rotate",
                .description = "Rotate a node or the current shared node selection by XYZ Euler deltas in radians",
                .input_schema = {
                    .type = "object",
                    .properties = json{
                        {"node", json{{"type", "string"}, {"description", "Optional node name; defaults to the current selected node(s)"}}},
                        {"value", json{{"type", "array"}, {"items", json{{"type", "number"}}}, {"description", "XYZ Euler delta in radians"}}}},
                    .required = {"value"}}},
            [viewer_impl](const json& args) -> json {
                const auto requested_node = optional_string_arg(args, "node");
                auto value = optional_vec3_arg(args, "value");
                if (!value)
                    return json{{"error", value.error()}};
                if (!value->has_value())
                    return json{{"error", "Field 'value' must be provided"}};

                return post_and_wait(viewer_impl, [viewer_impl, requested_node, value = **value]() -> json {
                    auto* const scene_manager = viewer_impl->getSceneManager();
                    if (!scene_manager)
                        return json{{"error", "Scene manager not initialized"}};

                    auto targets = resolve_transform_targets(*scene_manager, requested_node);
                    if (!targets)
                        return json{{"error", targets.error()}};

                    if (auto result = vis::cap::rotateNodes(
                            *scene_manager, *targets, value, "mcp.transform.rotate");
                        !result) {
                        return json{{"error", result.error()}};
                    }

                    json nodes = json::array();
                    for (const auto& name : *targets) {
                        if (const auto* const node = scene_manager->getScene().getNode(name))
                            nodes.push_back(transform_info_json(scene_manager->getScene(), *node));
                    }

                    return json{{"success", true}, {"count", nodes.size()}, {"nodes", nodes}};
                });
            });

        registry.register_tool(
            McpTool{
                .name = "transform.scale",
                .description = "Scale a node or the current shared node selection by XYZ factors",
                .input_schema = {
                    .type = "object",
                    .properties = json{
                        {"node", json{{"type", "string"}, {"description", "Optional node name; defaults to the current selected node(s)"}}},
                        {"value", json{{"type", "array"}, {"items", json{{"type", "number"}}}, {"description", "XYZ scale multiplier"}}}},
                    .required = {"value"}}},
            [viewer_impl](const json& args) -> json {
                const auto requested_node = optional_string_arg(args, "node");
                auto value = optional_vec3_arg(args, "value");
                if (!value)
                    return json{{"error", value.error()}};
                if (!value->has_value())
                    return json{{"error", "Field 'value' must be provided"}};

                return post_and_wait(viewer_impl, [viewer_impl, requested_node, value = **value]() -> json {
                    auto* const scene_manager = viewer_impl->getSceneManager();
                    if (!scene_manager)
                        return json{{"error", "Scene manager not initialized"}};

                    auto targets = resolve_transform_targets(*scene_manager, requested_node);
                    if (!targets)
                        return json{{"error", targets.error()}};

                    if (auto result = vis::cap::scaleNodes(
                            *scene_manager, *targets, value, "mcp.transform.scale");
                        !result) {
                        return json{{"error", result.error()}};
                    }

                    json nodes = json::array();
                    for (const auto& name : *targets) {
                        if (const auto* const node = scene_manager->getScene().getNode(name))
                            nodes.push_back(transform_info_json(scene_manager->getScene(), *node));
                    }

                    return json{{"success", true}, {"count", nodes.size()}, {"nodes", nodes}};
                });
            });

        registry.register_tool(
            McpTool{
                .name = "crop_box.add",
                .description = "Add or reuse a crop box for a node in the shared GUI scene",
                .input_schema = {
                    .type = "object",
                    .properties = json{
                        {"node", json{{"type", "string"}, {"description", "Optional splat or pointcloud node name; defaults to the current selected node"}}}},
                    .required = {}}},
            [viewer_impl](const json& args) -> json {
                const auto requested_node = optional_string_arg(args, "node");

                return post_and_wait(viewer_impl, [viewer_impl, requested_node]() -> json {
                    auto* const scene_manager = viewer_impl->getSceneManager();
                    auto* const rendering_manager = viewer_impl->getRenderingManager();
                    if (!scene_manager)
                        return json{{"error", "Scene manager not initialized"}};

                    auto parent_id = resolve_cropbox_parent_id(*scene_manager, requested_node);
                    if (!parent_id)
                        return json{{"error", parent_id.error()}};

                    auto cropbox_id = ensure_cropbox(*scene_manager, rendering_manager, *parent_id);
                    if (!cropbox_id)
                        return json{{"error", cropbox_id.error()}};

                    return crop_box_info_json(*scene_manager, rendering_manager, *cropbox_id);
                });
            });

        registry.register_tool(
            McpTool{
                .name = "crop_box.get",
                .description = "Inspect a crop box in the shared GUI scene",
                .input_schema = {
                    .type = "object",
                    .properties = json{
                        {"node", json{{"type", "string"}, {"description", "Optional crop box node or parent node name; defaults to the current selected crop box"}}}},
                    .required = {}}},
            [viewer_impl](const json& args) -> json {
                const auto requested_node = optional_string_arg(args, "node");

                return post_and_wait(viewer_impl, [viewer_impl, requested_node]() -> json {
                    auto* const scene_manager = viewer_impl->getSceneManager();
                    auto* const rendering_manager = viewer_impl->getRenderingManager();
                    if (!scene_manager)
                        return json{{"error", "Scene manager not initialized"}};

                    auto cropbox_id = resolve_cropbox_id(*scene_manager, requested_node);
                    if (!cropbox_id)
                        return json{{"error", cropbox_id.error()}};

                    return crop_box_info_json(*scene_manager, rendering_manager, *cropbox_id);
                });
            });

        registry.register_tool(
            McpTool{
                .name = "crop_box.set",
                .description = "Update crop box bounds, transform, or render toggles in the shared GUI scene",
                .input_schema = {
                    .type = "object",
                    .properties = json{
                        {"node", json{{"type", "string"}, {"description", "Optional crop box node or parent node name; defaults to the current selected crop box"}}},
                        {"min", json{{"type", "array"}, {"items", json{{"type", "number"}}}, {"description", "Optional local minimum bounds"}}},
                        {"max", json{{"type", "array"}, {"items", json{{"type", "number"}}}, {"description", "Optional local maximum bounds"}}},
                        {"translation", json{{"type", "array"}, {"items", json{{"type", "number"}}}, {"description", "Optional local XYZ translation"}}},
                        {"rotation", json{{"type", "array"}, {"items", json{{"type", "number"}}}, {"description", "Optional local XYZ Euler rotation in radians"}}},
                        {"scale", json{{"type", "array"}, {"items", json{{"type", "number"}}}, {"description", "Optional local XYZ scale"}}},
                        {"inverse", json{{"type", "boolean"}, {"description", "Invert the crop volume"}}},
                        {"enabled", json{{"type", "boolean"}, {"description", "Enable crop filtering for this crop box"}}},
                        {"show", json{{"type", "boolean"}, {"description", "Show crop boxes in the viewport"}}},
                        {"use", json{{"type", "boolean"}, {"description", "Use crop box filtering in rendering"}}}},
                    .required = {}}},
            [viewer_impl](const json& args) -> json {
                const auto requested_node = optional_string_arg(args, "node");
                auto min_bounds = optional_vec3_arg(args, "min");
                if (!min_bounds)
                    return json{{"error", min_bounds.error()}};
                auto max_bounds = optional_vec3_arg(args, "max");
                if (!max_bounds)
                    return json{{"error", max_bounds.error()}};
                auto translation = optional_vec3_arg(args, "translation");
                if (!translation)
                    return json{{"error", translation.error()}};
                auto rotation = optional_vec3_arg(args, "rotation");
                if (!rotation)
                    return json{{"error", rotation.error()}};
                auto scale = optional_vec3_arg(args, "scale");
                if (!scale)
                    return json{{"error", scale.error()}};

                const bool has_inverse = args.contains("inverse");
                const bool has_enabled = args.contains("enabled");
                const bool has_show = args.contains("show");
                const bool has_use = args.contains("use");

                if (!min_bounds->has_value() && !max_bounds->has_value() &&
                    !translation->has_value() && !rotation->has_value() && !scale->has_value() &&
                    !has_inverse && !has_enabled && !has_show && !has_use) {
                    return json{{"error", "No crop box fields were provided"}};
                }

                const bool inverse = has_inverse ? args["inverse"].get<bool>() : false;
                const bool enabled = has_enabled ? args["enabled"].get<bool>() : false;
                const bool show = has_show ? args["show"].get<bool>() : false;
                const bool use = has_use ? args["use"].get<bool>() : false;

                return post_and_wait(viewer_impl, [viewer_impl, requested_node, min_bounds = *min_bounds, max_bounds = *max_bounds, translation = *translation, rotation = *rotation, scale = *scale, has_inverse, inverse, has_enabled, enabled, has_show, show, has_use, use]() -> json {
                    auto* const scene_manager = viewer_impl->getSceneManager();
                    auto* const rendering_manager = viewer_impl->getRenderingManager();
                    if (!scene_manager)
                        return json{{"error", "Scene manager not initialized"}};

                    auto cropbox_id = resolve_cropbox_id(*scene_manager, requested_node);
                    if (!cropbox_id)
                        return json{{"error", cropbox_id.error()}};

                    vis::cap::CropBoxUpdate update;
                    update.min_bounds = min_bounds;
                    update.max_bounds = max_bounds;
                    update.translation = translation;
                    update.rotation = rotation;
                    update.scale = scale;
                    update.has_inverse = has_inverse;
                    update.inverse = inverse;
                    update.has_enabled = has_enabled;
                    update.enabled = enabled;
                    update.has_show = has_show;
                    update.show = show;
                    update.has_use = has_use;
                    update.use = use;

                    if (auto result = vis::cap::updateCropBox(
                            *scene_manager, rendering_manager, *cropbox_id, update);
                        !result) {
                        return json{{"error", result.error()}};
                    }

                    return crop_box_info_json(*scene_manager, rendering_manager, *cropbox_id);
                });
            });

        registry.register_tool(
            McpTool{
                .name = "crop_box.fit",
                .description = "Fit a crop box to its parent node bounds",
                .input_schema = {
                    .type = "object",
                    .properties = json{
                        {"node", json{{"type", "string"}, {"description", "Optional crop box node or parent node name; defaults to the current selected crop box"}}},
                        {"use_percentile", json{{"type", "boolean"}, {"description", "Use percentile bounds instead of strict min/max (default: true)"}}}},
                    .required = {}}},
            [viewer_impl](const json& args) -> json {
                const auto requested_node = optional_string_arg(args, "node");
                const bool use_percentile = args.value("use_percentile", true);

                return post_and_wait(viewer_impl, [viewer_impl, requested_node, use_percentile]() -> json {
                    auto* const scene_manager = viewer_impl->getSceneManager();
                    auto* const rendering_manager = viewer_impl->getRenderingManager();
                    if (!scene_manager)
                        return json{{"error", "Scene manager not initialized"}};

                    auto cropbox_id = resolve_cropbox_id(*scene_manager, requested_node);
                    if (!cropbox_id)
                        return json{{"error", cropbox_id.error()}};

                    auto result = fit_cropbox_to_parent(*scene_manager, rendering_manager, *cropbox_id, use_percentile);
                    if (!result)
                        return json{{"error", result.error()}};

                    return crop_box_info_json(*scene_manager, rendering_manager, *cropbox_id);
                });
            });

        registry.register_tool(
            McpTool{
                .name = "crop_box.reset",
                .description = "Reset a crop box to default bounds and identity transform",
                .input_schema = {
                    .type = "object",
                    .properties = json{
                        {"node", json{{"type", "string"}, {"description", "Optional crop box node or parent node name; defaults to the current selected crop box"}}}},
                    .required = {}}},
            [viewer_impl](const json& args) -> json {
                const auto requested_node = optional_string_arg(args, "node");

                return post_and_wait(viewer_impl, [viewer_impl, requested_node]() -> json {
                    auto* const scene_manager = viewer_impl->getSceneManager();
                    auto* const rendering_manager = viewer_impl->getRenderingManager();
                    if (!scene_manager)
                        return json{{"error", "Scene manager not initialized"}};

                    auto cropbox_id = resolve_cropbox_id(*scene_manager, requested_node);
                    if (!cropbox_id)
                        return json{{"error", cropbox_id.error()}};

                    auto result = reset_cropbox(*scene_manager, rendering_manager, *cropbox_id);
                    if (!result)
                        return json{{"error", result.error()}};

                    return crop_box_info_json(*scene_manager, rendering_manager, *cropbox_id);
                });
            });

        // --- Editor tools ---

        registry.register_tool(
            McpTool{
                .name = "editor.set_code",
                .description = "Populate the visible Python editor with code in the integrated Python console",
                .input_schema = {
                    .type = "object",
                    .properties = json{
                        {"code", json{{"type", "string"}, {"description", "Python code to place into the visible editor"}}},
                        {"show_console", json{{"type", "boolean"}, {"description", "Show the Python console window (default: true)"}}}},
                    .required = {"code"}}},
            [viewer_impl](const json& args) -> json {
                const std::string code = args["code"].get<std::string>();
                const bool show_console_window = args.value("show_console", true);

                return post_and_wait(viewer_impl, [code, show_console_window]() -> json {
                    if (show_console_window)
                        show_python_console();

                    auto& console = vis::gui::panels::PythonConsoleState::getInstance();
                    auto* const editor = console.getEditor();
                    if (!editor)
                        return json{{"error", "Python editor not initialized"}};

                    console.setEditorText(code);
                    console.focusEditor();
                    console.setModified(true);

                    return json{
                        {"success", true},
                        {"chars", static_cast<int64_t>(code.size())},
                        {"running", console.isScriptRunning()},
                    };
                });
            });

        registry.register_tool(
            McpTool{
                .name = "editor.run",
                .description = "Run code through the integrated Python console; optionally replace the visible editor contents first",
                .input_schema = {
                    .type = "object",
                    .properties = json{
                        {"code", json{{"type", "string"}, {"description", "Optional Python code to set and run; defaults to current editor contents"}}},
                        {"show_console", json{{"type", "boolean"}, {"description", "Show the Python console window (default: true)"}}}},
                    .required = {}}},
            [viewer_impl](const json& args) -> json {
                const auto code = optional_string_arg(args, "code");
                const bool show_console_window = args.value("show_console", true);

                return post_and_wait(viewer_impl, [code, show_console_window]() -> json {
                    if (show_console_window)
                        show_python_console();

                    auto& console = vis::gui::panels::PythonConsoleState::getInstance();
                    auto* const editor = console.getEditor();
                    if (!editor)
                        return json{{"error", "Python editor not initialized"}};
                    if (console.isScriptRunning())
                        return json{{"error", "A script is already running"}};

                    std::string code_to_run;
                    if (code) {
                        console.setEditorText(*code);
                        console.focusEditor();
                        console.setModified(true);
                        code_to_run = *code;
                    } else {
                        code_to_run = console.getEditorTextStripped();
                    }

                    if (code_to_run.empty())
                        return json{{"error", "Editor is empty"}};

                    console.runScriptAsync(code_to_run);
                    return json{
                        {"success", true},
                        {"chars", static_cast<int64_t>(code_to_run.size())},
                        {"running", console.isScriptRunning()},
                    };
                });
            });

        registry.register_tool(
            McpTool{
                .name = "editor.get_code",
                .description = "Read the current contents of the visible integrated Python editor",
                .input_schema = {
                    .type = "object",
                    .properties = json{
                        {"max_chars", json{{"type", "integer"}, {"description", "Maximum characters to return; defaults to no limit"}}}},
                    .required = {}}},
            [viewer_impl](const json& args) -> json {
                const int max_chars = args.value("max_chars", -1);

                return post_and_wait(viewer_impl, [max_chars]() -> json {
                    auto& console = vis::gui::panels::PythonConsoleState::getInstance();
                    auto* const editor = console.getEditor();
                    if (!editor)
                        return json{{"error", "Python editor not initialized"}};

                    std::string code = console.getEditorText();
                    auto result = text_payload_json(code, max_chars, false);
                    result["success"] = true;
                    result["total_chars"] = static_cast<int64_t>(code.size());
                    result["modified"] = console.isModified();
                    if (!console.getScriptPath().empty())
                        result["path"] = console.getScriptPath().string();
                    return result;
                });
            });

        registry.register_tool(
            McpTool{
                .name = "editor.get_output",
                .description = "Read captured output from the integrated Python console output terminal",
                .input_schema = {
                    .type = "object",
                    .properties = json{
                        {"max_chars", json{{"type", "integer"}, {"description", "Maximum characters to return (default: 20000)"}}},
                        {"tail", json{{"type", "boolean"}, {"description", "Return the most recent output when truncated (default: true)"}}}},
                    .required = {}}},
            [viewer_impl](const json& args) -> json {
                const int max_chars = args.value("max_chars", 20000);
                const bool tail = args.value("tail", true);

                return post_and_wait(viewer_impl, [max_chars, tail]() -> json {
                    auto& console = vis::gui::panels::PythonConsoleState::getInstance();
                    auto* const output = console.getOutputTerminal();
                    if (!output)
                        return json{{"error", "Python output terminal not initialized"}};

                    std::string text = console.getOutputText();
                    auto result = text_payload_json(text, max_chars, tail);
                    result["success"] = true;
                    result["total_chars"] = static_cast<int64_t>(text.size());
                    result["running"] = console.isScriptRunning();
                    return result;
                });
            });

        registry.register_tool(
            McpTool{
                .name = "editor.is_running",
                .description = "Check whether the integrated Python console is currently running a script",
                .input_schema = {.type = "object", .properties = json::object(), .required = {}}},
            [viewer_impl](const json&) -> json {
                return post_and_wait(viewer_impl, []() -> json {
                    auto& console = vis::gui::panels::PythonConsoleState::getInstance();
                    return json{
                        {"success", true},
                        {"running", console.isScriptRunning()},
                        {"modified", console.isModified()},
                    };
                });
            });

        registry.register_tool(
            McpTool{
                .name = "editor.interrupt",
                .description = "Interrupt the currently running script in the integrated Python console",
                .input_schema = {.type = "object", .properties = json::object(), .required = {}}},
            [viewer_impl](const json&) -> json {
                return post_and_wait(viewer_impl, []() -> json {
                    auto& console = vis::gui::panels::PythonConsoleState::getInstance();
                    const bool was_running = console.isScriptRunning();
                    if (was_running)
                        console.interruptScript();

                    return json{
                        {"success", true},
                        {"was_running", was_running},
                        {"running", console.isScriptRunning()},
                    };
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
            [viewer](const json& args) -> json {
                const auto capability = args.value("capability", "");
                if (capability.empty())
                    return json{{"error", "Missing capability name"}};

                const std::string args_json = args.contains("args") ? args["args"].dump() : "{}";

                return post_and_wait(viewer, [viewer, capability, args_json]() -> json {
                    python::SceneContextGuard ctx(&viewer->getScene());
                    auto result = python::invoke_capability(capability, args_json);
                    if (!result.success)
                        return json{{"success", false}, {"error", result.error}};

                    if (auto* const rendering_manager = viewer->getRenderingManager())
                        rendering_manager->markDirty(vis::DirtyFlag::ALL);

                    try {
                        return json::parse(result.result_json);
                    } catch (const std::exception& e) {
                        LOG_WARN("Failed to parse capability result: {}", e.what());
                        return json{{"success", true}, {"raw_result", result.result_json}};
                    }
                });
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
            [viewer_impl](const json& args) -> json {
                auto api_key = mcp::LLMClient::load_api_key_from_env();
                if (!api_key)
                    return json{{"error", api_key.error()}};

                const int camera_index = args.value("camera_index", 0);
                const std::string description = args["description"].get<std::string>();

                auto render_result = post_and_wait(viewer_impl, [viewer_impl, camera_index]() {
                    return render_scene_to_base64(viewer_impl->getScene(), camera_index);
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

                return post_and_wait(viewer_impl, [viewer_impl, x0, y0, x1, y1, camera_index, bbox, description]() -> json {
                    auto* const scene_manager = viewer_impl->getSceneManager();
                    if (!scene_manager)
                        return json{{"error", "Scene manager not initialized"}};

                    auto result = selection_result_json(*scene_manager,
                                                        scene_manager->selectRect(x0, y0, x1, y1, "replace", camera_index));
                    if (!result.value("success", false))
                        return result;
                    result["bounding_box"] = bbox;
                    result["description"] = description;
                    return result;
                });
            });

        LOG_INFO("Registered GUI-native MCP scene tools");
    }

    void register_gui_scene_resources(vis::Visualizer* viewer) {
        assert(viewer);
        auto* const viewer_impl = viewer;
        auto& registry = ResourceRegistry::instance();

        registry.register_resource(
            McpResource{
                .uri = "lichtfeld://render/current",
                .name = "Current Render",
                .description = "Base64-encoded PNG render from the current GUI scene using camera 0",
                .mime_type = "image/png"},
            [viewer](const std::string& uri) -> std::expected<std::vector<McpResourceContent>, std::string> {
                auto result = post_and_wait(viewer, [viewer]() {
                    return render_scene_to_base64(viewer->getScene(), 0);
                });
                if (!result)
                    return std::unexpected(result.error());

                return single_blob_resource(uri, "image/png", *result);
            });

        registry.register_resource_prefix(
            "lichtfeld://render/camera/",
            [viewer](const std::string& uri) -> std::expected<std::vector<McpResourceContent>, std::string> {
                constexpr std::string_view camera_prefix = "lichtfeld://render/camera/";
                int camera_index = 0;
                const auto idx_str = uri.substr(camera_prefix.size());
                try {
                    camera_index = std::stoi(idx_str);
                } catch (...) {
                    return std::unexpected("Invalid camera resource URI: " + uri);
                }

                auto result = post_and_wait(viewer, [viewer, camera_index]() {
                    return render_scene_to_base64(viewer->getScene(), camera_index);
                });
                if (!result)
                    return std::unexpected(result.error());

                return single_blob_resource(uri, "image/png", *result);
            });

        registry.register_resource(
            McpResource{
                .uri = "lichtfeld://gaussians/stats",
                .name = "Gaussian Statistics",
                .description = "Statistics about the current GUI scene Gaussian model",
                .mime_type = "application/json"},
            [viewer](const std::string& uri) -> std::expected<std::vector<McpResourceContent>, std::string> {
                return post_and_wait(viewer, [viewer, uri]() -> std::expected<std::vector<McpResourceContent>, std::string> {
                    json payload;
                    auto& scene = viewer->getScene();
                    payload["count"] = scene.getTotalGaussianCount();

                    const auto selection = selection_state_json(scene, 0);
                    payload["selected_count"] = selection.value("selected_count", 0);

                    if (auto* cc = event::command_center()) {
                        auto snapshot = cc->snapshot();
                        payload["is_refining"] = snapshot.is_refining;
                    }

                    return single_json_resource(uri, std::move(payload));
                });
            });

        registry.register_resource(
            McpResource{
                .uri = "lichtfeld://selection/current",
                .name = "Current Selection",
                .description = "Current Gaussian selection from the shared GUI scene",
                .mime_type = "application/json"},
            [viewer](const std::string& uri) -> std::expected<std::vector<McpResourceContent>, std::string> {
                return post_and_wait(viewer, [viewer, uri]() -> std::expected<std::vector<McpResourceContent>, std::string> {
                    auto payload = selection_state_json(viewer->getScene());
                    payload["success"] = true;
                    return single_json_resource(uri, std::move(payload));
                });
            });

        registry.register_resource(
            McpResource{
                .uri = "lichtfeld://scene/nodes",
                .name = "Scene Nodes",
                .description = "All nodes from the shared GUI scene",
                .mime_type = "application/json"},
            [viewer_impl](const std::string& uri) -> std::expected<std::vector<McpResourceContent>, std::string> {
                return post_and_wait(viewer_impl, [viewer_impl, uri]() -> std::expected<std::vector<McpResourceContent>, std::string> {
                    auto* const scene_manager = viewer_impl->getSceneManager();
                    if (!scene_manager)
                        return std::unexpected("Scene manager not initialized");

                    const auto& scene = scene_manager->getScene();
                    json nodes = json::array();
                    for (const auto* const node : scene.getNodes()) {
                        if (!node)
                            continue;
                        nodes.push_back(node_summary_json(scene, *node));
                    }

                    return single_json_resource(uri, json{{"count", nodes.size()}, {"nodes", std::move(nodes)}});
                });
            });

        registry.register_resource(
            McpResource{
                .uri = "lichtfeld://scene/selected_nodes",
                .name = "Selected Scene Nodes",
                .description = "Currently selected nodes from the shared GUI scene",
                .mime_type = "application/json"},
            [viewer_impl](const std::string& uri) -> std::expected<std::vector<McpResourceContent>, std::string> {
                return post_and_wait(viewer_impl, [viewer_impl, uri]() -> std::expected<std::vector<McpResourceContent>, std::string> {
                    auto* const scene_manager = viewer_impl->getSceneManager();
                    if (!scene_manager)
                        return std::unexpected("Scene manager not initialized");

                    const auto& scene = scene_manager->getScene();
                    json nodes = json::array();
                    for (const auto& name : scene_manager->getSelectedNodeNames()) {
                        if (const auto* const node = scene.getNode(name))
                            nodes.push_back(node_summary_json(scene, *node));
                    }

                    return single_json_resource(uri, json{{"count", nodes.size()}, {"nodes", std::move(nodes)}});
                });
            });

        LOG_INFO("Registered GUI-native MCP scene resources");
    }

} // namespace lfs::app
