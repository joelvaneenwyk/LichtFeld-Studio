/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#define GLM_ENABLE_EXPERIMENTAL

#include "app/mcp_gui_tools.hpp"

#include "core/base64.hpp"
#include "core/events.hpp"
#include "core/event_bridge/command_center_bridge.hpp"
#include "core/logger.hpp"
#include "core/scene.hpp"
#include "core/splat_data_transform.hpp"
#include "core/tensor.hpp"
#include "io/exporter.hpp"
#include "mcp/llm_client.hpp"
#include "mcp/mcp_tools.hpp"
#include "mcp/selection_client.hpp"
#include "python/runner.hpp"
#include "rendering/gs_rasterizer_tensor.hpp"
#include "visualizer/operation/undo_entry.hpp"
#include "visualizer/operation/undo_history.hpp"
#include "visualizer/rendering/rendering_manager.hpp"
#include "visualizer/scene/scene_manager.hpp"
#include "visualizer/visualizer.hpp"

#include <stb_image_write.h>

#include <cassert>
#include <future>
#include <optional>
#include <string>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/euler_angles.hpp>

namespace lfs::app {

    using json = nlohmann::json;
    using mcp::McpTool;
    using mcp::ToolRegistry;

    namespace {

        struct TransformComponents {
            glm::vec3 translation{0.0f};
            glm::vec3 rotation{0.0f};
            glm::vec3 scale{1.0f};
        };

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

        template <typename F>
        auto post_and_wait(vis::Visualizer* viewer, F&& fn) {
            using R = std::invoke_result_t<F>;
            auto task = std::make_shared<std::packaged_task<R()>>(std::forward<F>(fn));
            auto f = task->get_future();
            viewer->postWork([task]() { (*task)(); });
            return f.get();
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
            TransformComponents result;
            result.translation = glm::vec3(matrix[3]);

            glm::vec3 col0 = glm::vec3(matrix[0]);
            glm::vec3 col1 = glm::vec3(matrix[1]);
            glm::vec3 col2 = glm::vec3(matrix[2]);

            result.scale.x = glm::length(col0);
            result.scale.y = glm::length(col1);
            result.scale.z = glm::length(col2);

            if (result.scale.x > 0.0f)
                col0 /= result.scale.x;
            if (result.scale.y > 0.0f)
                col1 /= result.scale.y;
            if (result.scale.z > 0.0f)
                col2 /= result.scale.z;

            const glm::mat3 rotation_matrix(col0, col1, col2);
            glm::extractEulerAngleXYZ(glm::mat4(rotation_matrix), result.rotation.x, result.rotation.y, result.rotation.z);

            return result;
        }

        glm::mat4 compose_transform(const TransformComponents& components) {
            const glm::mat4 translation = glm::translate(glm::mat4(1.0f), components.translation);
            const glm::mat4 rotation = glm::eulerAngleXYZ(components.rotation.x, components.rotation.y, components.rotation.z);
            const glm::mat4 scale = glm::scale(glm::mat4(1.0f), components.scale);
            return translation * rotation * scale;
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

        std::expected<std::vector<std::string>, std::string> resolve_transform_targets(
            const vis::SceneManager& scene_manager,
            const std::optional<std::string>& requested_node) {

            if (requested_node) {
                if (!scene_manager.getScene().getNode(*requested_node))
                    return std::unexpected("Node not found: " + *requested_node);
                return std::vector<std::string>{*requested_node};
            }

            auto names = scene_manager.getSelectedNodeNames();
            if (names.empty())
                return std::unexpected("No node specified and no node selected");
            return names;
        }

        std::expected<core::NodeId, std::string> resolve_cropbox_parent_id(
            const vis::SceneManager& scene_manager,
            const std::optional<std::string>& requested_node) {

            const auto& scene = scene_manager.getScene();
            const auto resolve = [&scene](const core::SceneNode* node) -> std::expected<core::NodeId, std::string> {
                if (!node)
                    return std::unexpected("Node not found");
                if (node->type == core::NodeType::CROPBOX)
                    return node->parent_id;
                if (node->type == core::NodeType::SPLAT || node->type == core::NodeType::POINTCLOUD)
                    return node->id;
                return std::unexpected("Crop boxes can only target splat or pointcloud nodes");
            };

            if (requested_node)
                return resolve(scene.getNode(*requested_node));

            const auto selected_name = scene_manager.getSelectedNodeName();
            if (selected_name.empty())
                return std::unexpected("No node specified and no node selected");
            return resolve(scene.getNode(selected_name));
        }

        std::expected<core::NodeId, std::string> resolve_cropbox_id(
            const vis::SceneManager& scene_manager,
            const std::optional<std::string>& requested_node) {

            const auto& scene = scene_manager.getScene();
            if (requested_node) {
                const auto* const node = scene.getNode(*requested_node);
                if (!node)
                    return std::unexpected("Node not found: " + *requested_node);

                if (node->type == core::NodeType::CROPBOX && node->cropbox)
                    return node->id;

                if (node->type == core::NodeType::SPLAT || node->type == core::NodeType::POINTCLOUD) {
                    const core::NodeId cropbox_id = scene.getCropBoxForSplat(node->id);
                    if (cropbox_id == core::NULL_NODE)
                        return std::unexpected("Node has no crop box: " + *requested_node);
                    return cropbox_id;
                }

                return std::unexpected("Node does not reference a crop box: " + *requested_node);
            }

            const core::NodeId cropbox_id = scene_manager.getSelectedNodeCropBoxId();
            if (cropbox_id == core::NULL_NODE)
                return std::unexpected("No crop box specified and no crop box selected");
            return cropbox_id;
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

            auto& scene = scene_manager.getScene();
            const auto* const parent = scene.getNodeById(parent_id);
            if (!parent)
                return std::unexpected("Target node not found");

            if (parent->type != core::NodeType::SPLAT && parent->type != core::NodeType::POINTCLOUD)
                return std::unexpected("Crop boxes can only be attached to splat or pointcloud nodes");

            if (const core::NodeId existing = scene.getCropBoxForSplat(parent_id); existing != core::NULL_NODE) {
                if (rendering_manager) {
                    auto settings = rendering_manager->getSettings();
                    settings.show_crop_box = true;
                    rendering_manager->updateSettings(settings);
                }
                return existing;
            }

            const std::string cropbox_name = parent->name + "_cropbox";
            const core::NodeId cropbox_id = scene.addCropBox(cropbox_name, parent_id);
            if (cropbox_id == core::NULL_NODE)
                return std::unexpected("Failed to create crop box for node: " + parent->name);

            core::CropBoxData data;
            glm::vec3 min_bounds, max_bounds;
            if (scene.getNodeBounds(parent_id, min_bounds, max_bounds)) {
                data.min = min_bounds;
                data.max = max_bounds;
            }
            data.enabled = true;
            scene.setCropBoxData(cropbox_id, data);

            if (const auto* const cropbox_node = scene.getNodeById(cropbox_id)) {
                core::events::state::PLYAdded{
                    .name = cropbox_node->name,
                    .node_gaussians = 0,
                    .total_gaussians = scene.getTotalGaussianCount(),
                    .is_visible = cropbox_node->visible,
                    .parent_name = parent->name,
                    .is_group = false,
                    .node_type = static_cast<int>(core::NodeType::CROPBOX)}
                    .emit();
            }

            if (rendering_manager) {
                auto settings = rendering_manager->getSettings();
                settings.show_crop_box = true;
                rendering_manager->updateSettings(settings);
            }

            return cropbox_id;
        }

        std::expected<void, std::string> fit_cropbox_to_parent(
            vis::SceneManager& scene_manager,
            vis::RenderingManager* rendering_manager,
            const core::NodeId cropbox_id,
            const bool use_percentile) {

            auto& scene = scene_manager.getScene();
            const auto* const cropbox_node = scene.getNodeById(cropbox_id);
            if (!cropbox_node || cropbox_node->type != core::NodeType::CROPBOX || !cropbox_node->cropbox)
                return std::unexpected("Invalid crop box target");

            const auto* const parent = scene.getNodeById(cropbox_node->parent_id);
            if (!parent)
                return std::unexpected("Crop box parent not found");

            glm::vec3 min_bounds, max_bounds;
            bool bounds_valid = false;
            if (parent->type == core::NodeType::SPLAT && parent->model && parent->model->size() > 0) {
                bounds_valid = core::compute_bounds(*parent->model, min_bounds, max_bounds, 0.0f, use_percentile);
            } else if (parent->type == core::NodeType::POINTCLOUD && parent->point_cloud && parent->point_cloud->size() > 0) {
                bounds_valid = core::compute_bounds(*parent->point_cloud, min_bounds, max_bounds, 0.0f, use_percentile);
            }

            if (!bounds_valid)
                return std::unexpected("Cannot compute bounds for node: " + parent->name);

            const auto before_data = *cropbox_node->cropbox;
            const auto before_transform = scene_manager.getNodeTransform(cropbox_node->name);

            const glm::vec3 center = (min_bounds + max_bounds) * 0.5f;
            const glm::vec3 half_size = (max_bounds - min_bounds) * 0.5f;

            auto updated_data = before_data;
            updated_data.min = -half_size;
            updated_data.max = half_size;
            scene.setCropBoxData(cropbox_id, updated_data);
            scene.setNodeTransform(cropbox_node->name, glm::translate(glm::mat4(1.0f), center));

            if (rendering_manager)
                rendering_manager->markDirty(vis::DirtyFlag::SPLATS | vis::DirtyFlag::OVERLAY);

            auto entry = std::make_unique<vis::op::CropBoxUndoEntry>(
                scene_manager, cropbox_node->name, before_data, before_transform);
            if (entry->hasChanges())
                vis::op::undoHistory().push(std::move(entry));

            return {};
        }

        std::expected<void, std::string> reset_cropbox(
            vis::SceneManager& scene_manager,
            vis::RenderingManager* rendering_manager,
            const core::NodeId cropbox_id) {

            auto& scene = scene_manager.getScene();
            const auto* const cropbox_node = scene.getNodeById(cropbox_id);
            if (!cropbox_node || cropbox_node->type != core::NodeType::CROPBOX || !cropbox_node->cropbox)
                return std::unexpected("Invalid crop box target");

            const auto before_data = *cropbox_node->cropbox;
            const auto before_transform = scene_manager.getNodeTransform(cropbox_node->name);

            auto reset_data = before_data;
            reset_data.min = glm::vec3(-1.0f);
            reset_data.max = glm::vec3(1.0f);
            reset_data.inverse = false;
            scene.setCropBoxData(cropbox_id, reset_data);
            scene.setNodeTransform(cropbox_node->name, glm::mat4(1.0f));

            if (rendering_manager) {
                auto settings = rendering_manager->getSettings();
                settings.use_crop_box = false;
                rendering_manager->updateSettings(settings);
                rendering_manager->markDirty(vis::DirtyFlag::SPLATS | vis::DirtyFlag::OVERLAY);
            }

            auto entry = std::make_unique<vis::op::CropBoxUndoEntry>(
                scene_manager, cropbox_node->name, before_data, before_transform);
            if (entry->hasChanges())
                vis::op::undoHistory().push(std::move(entry));

            return {};
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
            [viewer_impl](const json&) -> json {
                return post_and_wait(viewer_impl, [viewer_impl]() -> json {
                    auto* const scene_manager = viewer_impl->getSceneManager();
                    if (!scene_manager)
                        return json{{"error", "Scene manager not initialized"}};
                    scene_manager->deselectAllGaussians();
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

                    if (!scene_manager->getScene().getNode(name))
                        return json{{"error", "Node not found: " + name}};

                    if (mode == "add")
                        scene_manager->addToSelection(name);
                    else
                        scene_manager->selectNode(name);

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

                    auto entry = std::make_unique<vis::op::SceneSnapshot>(*scene_manager, "mcp.transform.set");
                    entry->captureTransforms(*targets);

                    for (const auto& name : *targets) {
                        auto components = decompose_transform(scene_manager->getScene().getNodeTransform(name));
                        if (translation)
                            components.translation = *translation;
                        if (rotation)
                            components.rotation = *rotation;
                        if (scale)
                            components.scale = *scale;
                        scene_manager->setNodeTransform(name, compose_transform(components));
                    }

                    entry->captureAfter();
                    vis::op::undoHistory().push(std::move(entry));

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

                    auto entry = std::make_unique<vis::op::SceneSnapshot>(*scene_manager, "mcp.transform.translate");
                    entry->captureTransforms(*targets);

                    for (const auto& name : *targets) {
                        auto transform = scene_manager->getScene().getNodeTransform(name);
                        transform[3] += glm::vec4(value, 0.0f);
                        scene_manager->setNodeTransform(name, transform);
                    }

                    entry->captureAfter();
                    vis::op::undoHistory().push(std::move(entry));

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

                    auto entry = std::make_unique<vis::op::SceneSnapshot>(*scene_manager, "mcp.transform.rotate");
                    entry->captureTransforms(*targets);

                    const glm::mat4 rotation_delta = glm::eulerAngleXYZ(value.x, value.y, value.z);
                    for (const auto& name : *targets) {
                        auto components = decompose_transform(scene_manager->getScene().getNodeTransform(name));
                        const glm::mat4 current_rotation =
                            glm::eulerAngleXYZ(components.rotation.x, components.rotation.y, components.rotation.z);
                        const glm::mat4 new_rotation = rotation_delta * current_rotation;
                        glm::extractEulerAngleXYZ(new_rotation, components.rotation.x, components.rotation.y, components.rotation.z);
                        scene_manager->setNodeTransform(name, compose_transform(components));
                    }

                    entry->captureAfter();
                    vis::op::undoHistory().push(std::move(entry));

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

                    auto entry = std::make_unique<vis::op::SceneSnapshot>(*scene_manager, "mcp.transform.scale");
                    entry->captureTransforms(*targets);

                    for (const auto& name : *targets) {
                        auto components = decompose_transform(scene_manager->getScene().getNodeTransform(name));
                        components.scale *= value;
                        scene_manager->setNodeTransform(name, compose_transform(components));
                    }

                    entry->captureAfter();
                    vis::op::undoHistory().push(std::move(entry));

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

                return post_and_wait(viewer_impl, [viewer_impl, requested_node,
                                                   min_bounds = *min_bounds,
                                                   max_bounds = *max_bounds,
                                                   translation = *translation,
                                                   rotation = *rotation,
                                                   scale = *scale,
                                                   has_inverse, inverse,
                                                   has_enabled, enabled,
                                                   has_show, show,
                                                   has_use, use]() -> json {
                    auto* const scene_manager = viewer_impl->getSceneManager();
                    auto* const rendering_manager = viewer_impl->getRenderingManager();
                    if (!scene_manager)
                        return json{{"error", "Scene manager not initialized"}};

                    auto cropbox_id = resolve_cropbox_id(*scene_manager, requested_node);
                    if (!cropbox_id)
                        return json{{"error", cropbox_id.error()}};

                    auto& scene = scene_manager->getScene();
                    const auto* const cropbox_node = scene.getNodeById(*cropbox_id);
                    if (!cropbox_node || !cropbox_node->cropbox)
                        return json{{"error", "Invalid crop box target"}};

                    const auto before_data = *cropbox_node->cropbox;
                    const auto before_transform = scene_manager->getNodeTransform(cropbox_node->name);

                    auto updated_data = before_data;
                    auto updated_components = decompose_transform(before_transform);

                    bool cropbox_changed = false;
                    bool transform_changed = false;

                    if (min_bounds) {
                        updated_data.min = *min_bounds;
                        cropbox_changed = true;
                    }
                    if (max_bounds) {
                        updated_data.max = *max_bounds;
                        cropbox_changed = true;
                    }
                    if (has_inverse) {
                        updated_data.inverse = inverse;
                        cropbox_changed = true;
                    }
                    if (has_enabled) {
                        updated_data.enabled = enabled;
                        cropbox_changed = true;
                    }
                    if (translation) {
                        updated_components.translation = *translation;
                        transform_changed = true;
                    }
                    if (rotation) {
                        updated_components.rotation = *rotation;
                        transform_changed = true;
                    }
                    if (scale) {
                        updated_components.scale = *scale;
                        transform_changed = true;
                    }

                    if (cropbox_changed)
                        scene.setCropBoxData(*cropbox_id, updated_data);
                    if (transform_changed)
                        scene_manager->setNodeTransform(cropbox_node->name, compose_transform(updated_components));

                    if (rendering_manager && (cropbox_changed || transform_changed)) {
                        rendering_manager->markDirty(vis::DirtyFlag::SPLATS | vis::DirtyFlag::OVERLAY);
                    }

                    if (rendering_manager && (has_show || has_use)) {
                        auto settings = rendering_manager->getSettings();
                        if (has_show)
                            settings.show_crop_box = show;
                        if (has_use)
                            settings.use_crop_box = use;
                        rendering_manager->updateSettings(settings);
                    }

                    if (cropbox_changed || transform_changed) {
                        auto entry = std::make_unique<vis::op::CropBoxUndoEntry>(
                            *scene_manager, cropbox_node->name, before_data, before_transform);
                        if (entry->hasChanges())
                            vis::op::undoHistory().push(std::move(entry));
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

} // namespace lfs::app
