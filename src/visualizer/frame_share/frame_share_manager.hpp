/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include <chrono>
#include <memory>
#include <string>
#include <utility>

namespace lfs::vis {

    class IFrameShareSink;

    enum class FrameShareResolution : uint8_t {
        Native,
        R720p,
        R1080p,
        R1440p,
    };

    enum class FrameShareBackend : uint8_t {
        None,
#ifdef LFS_SPOUT_ENABLED
        Spout,
#endif
#if defined(__linux__)
        SharedMemory,
        V4L2,
#endif
    };

    class LFS_VIS_API FrameShareManager {
    public:
        FrameShareManager();
        ~FrameShareManager();

        void setBackend(FrameShareBackend backend);
        FrameShareBackend getBackend() const { return backend_; }

        void setEnabled(bool enabled);
        bool isEnabled() const { return enabled_; }
        bool isActive() const;

        void setSenderName(const std::string& name);
        const std::string& getSenderName() const { return sender_name_; }

        void setTargetFps(float fps);
        float getTargetFps() const { return target_fps_; }

        void setOutputResolution(FrameShareResolution res);
        FrameShareResolution getOutputResolution() const { return output_resolution_; }

        void onFrame(unsigned int gl_texture_id, int width, int height);
        void onFrameFromViewport(int viewport_x, int viewport_y, int width, int height);

        int connectedReceivers() const;
        const std::string& getStatusMessage() const { return status_message_; }

    private:
        void createSink(int width, int height);
        void destroySink();
        void ensureCaptureTexture(int width, int height);
        void destroyCaptureTexture();
        std::pair<int, int> computeOutputSize(int src_w, int src_h) const;
        void ensureScaledResources(int target_w, int target_h);
        void destroyScaledResources();

        std::unique_ptr<IFrameShareSink> sink_;
        FrameShareBackend backend_ = FrameShareBackend::None;
        bool enabled_ = false;
        std::string sender_name_ = "LichtFeld Studio";
        std::string status_message_;
        int last_width_ = 0;
        int last_height_ = 0;
        int sink_width_ = 0;
        int sink_height_ = 0;
        std::chrono::steady_clock::time_point next_retry_time_{};
        float target_fps_ = 0.0f;
        std::chrono::steady_clock::time_point last_frame_time_{};
        unsigned int capture_texture_ = 0;
        int capture_width_ = 0;
        int capture_height_ = 0;
        FrameShareResolution output_resolution_ = FrameShareResolution::Native;
        unsigned int scale_fbo_ = 0;
        unsigned int scale_draw_fbo_ = 0;
        unsigned int scaled_texture_ = 0;
        int scaled_tex_width_ = 0;
        int scaled_tex_height_ = 0;
    };

} // namespace lfs::vis
