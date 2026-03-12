/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "core/parameters.hpp"
#include "core/scene.hpp"
#include "core/tensor.hpp"
#include "training/trainer.hpp"

#include <array>
#include <atomic>
#include <expected>
#include <filesystem>
#include <future>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace lfs::mcp {

    class LFS_MCP_API TrainingContext {
    public:
        static TrainingContext& instance();

        std::expected<void, std::string> load_dataset(
            const std::filesystem::path& path,
            const core::param::TrainingParameters& params);

        std::expected<void, std::string> load_checkpoint(
            const std::filesystem::path& path);

        std::expected<void, std::string> save_checkpoint(
            const std::filesystem::path& path);

        std::expected<void, std::string> save_ply(
            const std::filesystem::path& path);

        std::expected<std::string, std::string> render_to_base64(
            int camera_index = 0,
            int width = 0,
            int height = 0);

        std::expected<core::Tensor, std::string> compute_screen_positions(
            int camera_index = 0);

        std::expected<void, std::string> start_training();
        void stop_training();
        void pause_training();
        void resume_training();

        bool is_loaded() const { return scene_ != nullptr; }
        bool is_training() const { return training_thread_ != nullptr; }

        std::shared_ptr<core::Scene> scene() {
            std::lock_guard lock(mutex_);
            return scene_;
        }
        training::Trainer* trainer() { return trainer_.get(); }
        const core::param::TrainingParameters& params() const { return params_; }
        core::param::TrainingParameters& params_mutable() { return params_; }
        core::Tensor& selection_locked_groups_device_mask() { return locked_groups_device_mask_; }
        core::Tensor& selection_scratch_buffer() { return selection_scratch_buffer_; }
        core::Tensor& selection_polygon_vertex_buffer() { return selection_polygon_vertex_buffer_; }
        std::array<core::Tensor, 2>& selection_output_buffers() { return selection_output_buffers_; }
        size_t& selection_output_buffer_index() { return selection_output_buffer_index_; }

        void shutdown();

    private:
        TrainingContext() = default;
        ~TrainingContext();

        // shared_ptr allows tool lambdas to hold references across async boundaries.
        // INVARIANT: stop_training() must complete before scene_.reset().
        std::shared_ptr<core::Scene> scene_;
        std::unique_ptr<training::Trainer> trainer_;
        core::param::TrainingParameters params_;
        core::Tensor locked_groups_device_mask_;
        core::Tensor selection_scratch_buffer_;
        core::Tensor selection_polygon_vertex_buffer_;
        std::array<core::Tensor, 2> selection_output_buffers_;
        size_t selection_output_buffer_index_ = 0;

        std::unique_ptr<std::jthread> training_thread_;
        std::mutex mutex_;
    };

    void register_scene_tools();

} // namespace lfs::mcp
