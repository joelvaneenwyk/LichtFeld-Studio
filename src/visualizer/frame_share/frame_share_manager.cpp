/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "frame_share/frame_share_manager.hpp"
#include "core/logger.hpp"
#include "frame_share/frame_share_sink.hpp"
#include <glad/glad.h>

#ifdef LFS_SPOUT_ENABLED
#include "frame_share/spout_sink.hpp"
#endif

#if defined(__linux__)
#include "frame_share/shm_sink.hpp"
#include "frame_share/v4l2_sink.hpp"
#endif

namespace lfs::vis {

    FrameShareManager::FrameShareManager() {
#ifdef LFS_SPOUT_ENABLED
        backend_ = FrameShareBackend::Spout;
#elif defined(__linux__)
        backend_ = FrameShareBackend::SharedMemory;
#endif
        status_message_ = "Inactive";
        next_retry_time_ = std::chrono::steady_clock::now();
    }

    FrameShareManager::~FrameShareManager() {
        destroySink();
        destroyCaptureTexture();
        destroyScaledResources();
    }

    void FrameShareManager::setBackend(FrameShareBackend backend) {
        if (backend_ == backend)
            return;

        const bool was_enabled = enabled_;
        if (was_enabled)
            setEnabled(false);

        backend_ = backend;

        if (was_enabled)
            setEnabled(true);
    }

    void FrameShareManager::setEnabled(bool enabled) {
        if (enabled_ == enabled)
            return;
        enabled_ = enabled;

        if (enabled_) {
            next_retry_time_ = std::chrono::steady_clock::now();
            createSink(last_width_, last_height_);
        } else {
            destroySink();
            status_message_ = "Inactive";
        }
    }

    bool FrameShareManager::isActive() const {
        return sink_ && sink_->isActive();
    }

    void FrameShareManager::setSenderName(const std::string& name) {
        if (sender_name_ == name)
            return;

        const bool was_enabled = enabled_;
        if (was_enabled)
            setEnabled(false);

        sender_name_ = name;

        if (was_enabled)
            setEnabled(true);
    }

    void FrameShareManager::setTargetFps(float fps) {
        target_fps_ = std::max(fps, 0.0f);
    }

    void FrameShareManager::onFrame(unsigned int gl_texture_id, int width, int height) {
        if (!enabled_ || gl_texture_id == 0 || width <= 0 || height <= 0)
            return;

        if (target_fps_ > 0.0f) {
            const auto now = std::chrono::steady_clock::now();
            const auto interval = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                std::chrono::duration<float>(1.0f / target_fps_));
            if (now - last_frame_time_ < interval)
                return;
            last_frame_time_ = now;
        }

        last_width_ = width;
        last_height_ = height;

        if (!sink_ || !sink_->isActive()) {
            const auto now = std::chrono::steady_clock::now();
            if (now < next_retry_time_) {
                return;
            }
            createSink(width, height);
            if (!sink_ || !sink_->isActive()) {
                return;
            }
        }

        auto [out_w, out_h] = computeOutputSize(width, height);

        if (out_w != sink_width_ || out_h != sink_height_) {
            LOG_INFO("Frame share: output resolution changed to {}x{}, restarting sink", out_w, out_h);
            createSink(out_w, out_h);
            if (!sink_ || !sink_->isActive())
                return;
        }

        if (out_w == width && out_h == height) {
            sink_->sendFrame(gl_texture_id, width, height);
        } else {
            ensureScaledResources(out_w, out_h);

            GLint prev_fbo = 0;
            glGetIntegerv(GL_FRAMEBUFFER_BINDING, &prev_fbo);

            glBindFramebuffer(GL_READ_FRAMEBUFFER, scale_fbo_);
            glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, gl_texture_id, 0);

            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, scale_draw_fbo_);
            glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, scaled_texture_, 0);

            glBlitFramebuffer(0, 0, width, height, 0, 0, out_w, out_h, GL_COLOR_BUFFER_BIT, GL_LINEAR);

            glBindFramebuffer(GL_FRAMEBUFFER, static_cast<GLuint>(prev_fbo));

            sink_->sendFrame(scaled_texture_, out_w, out_h);
        }
    }

    void FrameShareManager::onFrameFromViewport(int viewport_x, int viewport_y, int width, int height) {
        if (!enabled_ || width <= 0 || height <= 0)
            return;

        if (target_fps_ > 0.0f) {
            const auto now = std::chrono::steady_clock::now();
            const auto interval = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                std::chrono::duration<float>(1.0f / target_fps_));
            if (now - last_frame_time_ < interval)
                return;
        }

        const auto now = std::chrono::steady_clock::now();
        if (!sink_ && now < next_retry_time_) {
            return;
        }

        ensureCaptureTexture(width, height);
        if (capture_texture_ == 0) {
            return;
        }

        GLint prev_read_framebuffer = 0;
        GLint prev_read_buffer = 0;
        GLint double_buffered = GL_TRUE;
        glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &prev_read_framebuffer);
        glGetIntegerv(GL_READ_BUFFER, &prev_read_buffer);
        glGetIntegerv(GL_DOUBLEBUFFER, &double_buffered);

        glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
        glReadBuffer(double_buffered ? GL_BACK : GL_FRONT);
        glBindTexture(GL_TEXTURE_2D, capture_texture_);
        glCopyTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, viewport_x, viewport_y, width, height);
        glBindTexture(GL_TEXTURE_2D, 0);

        glBindFramebuffer(GL_READ_FRAMEBUFFER, static_cast<GLuint>(prev_read_framebuffer));
        glReadBuffer(static_cast<GLenum>(prev_read_buffer));

        onFrame(capture_texture_, width, height);
    }

    int FrameShareManager::connectedReceivers() const {
        return sink_ ? sink_->connectedReceivers() : 0;
    }

    void FrameShareManager::createSink(int width, int height) {
        destroySink();

        switch (backend_) {
#ifdef LFS_SPOUT_ENABLED
        case FrameShareBackend::Spout:
            sink_ = std::make_unique<SpoutSink>();
            break;
#endif
#if defined(__linux__)
        case FrameShareBackend::SharedMemory:
            sink_ = std::make_unique<ShmSink>();
            break;
        case FrameShareBackend::V4L2:
            sink_ = std::make_unique<V4L2Sink>();
            break;
#endif
        case FrameShareBackend::None:
            status_message_ = "No backend selected";
            next_retry_time_ = std::chrono::steady_clock::now() + std::chrono::seconds(1);
            return;
        }

        if (sink_) {
            if (sink_->start(sender_name_, width, height)) {
                sink_width_ = width;
                sink_height_ = height;
                status_message_ = "Active";
                LOG_INFO("Frame share started: backend={}, name='{}'",
                         static_cast<int>(backend_), sender_name_);
            } else {
                status_message_ = "Failed to start";
                LOG_ERROR("Frame share failed to start");
                sink_.reset();
                sink_width_ = 0;
                sink_height_ = 0;
                next_retry_time_ = std::chrono::steady_clock::now() + std::chrono::seconds(1);
            }
        }
    }

    void FrameShareManager::destroySink() {
        if (sink_) {
            sink_->stop();
            sink_.reset();
        }
        sink_width_ = 0;
        sink_height_ = 0;
    }

    void FrameShareManager::ensureCaptureTexture(int width, int height) {
        if (capture_texture_ == 0) {
            glGenTextures(1, &capture_texture_);
            if (capture_texture_ == 0) {
                LOG_ERROR("Frame share: failed to create capture texture");
                return;
            }

            glBindTexture(GL_TEXTURE_2D, capture_texture_);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        } else {
            glBindTexture(GL_TEXTURE_2D, capture_texture_);
        }

        if (capture_width_ != width || capture_height_ != height) {
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
            capture_width_ = width;
            capture_height_ = height;
        }

        glBindTexture(GL_TEXTURE_2D, 0);
    }

    void FrameShareManager::destroyCaptureTexture() {
        if (capture_texture_ != 0) {
            glDeleteTextures(1, &capture_texture_);
            capture_texture_ = 0;
        }
        capture_width_ = 0;
        capture_height_ = 0;
    }

    void FrameShareManager::setOutputResolution(FrameShareResolution res) {
        if (output_resolution_ == res)
            return;
        output_resolution_ = res;

        if (res == FrameShareResolution::Native)
            destroyScaledResources();
    }

    std::pair<int, int> FrameShareManager::computeOutputSize(int src_w, int src_h) const {
        int target_w = 0;
        switch (output_resolution_) {
        case FrameShareResolution::Native:
            return {src_w, src_h};
        case FrameShareResolution::R720p:
            target_w = 1280;
            break;
        case FrameShareResolution::R1080p:
            target_w = 1920;
            break;
        case FrameShareResolution::R1440p:
            target_w = 2560;
            break;
        }

        if (target_w >= src_w)
            return {src_w, src_h};

        const int target_h = (target_w * src_h) / src_w;
        return {target_w, target_h > 0 ? target_h : 1};
    }

    void FrameShareManager::ensureScaledResources(int target_w, int target_h) {
        if (scale_fbo_ == 0)
            glGenFramebuffers(1, &scale_fbo_);
        if (scale_draw_fbo_ == 0)
            glGenFramebuffers(1, &scale_draw_fbo_);

        if (scaled_tex_width_ == target_w && scaled_tex_height_ == target_h)
            return;

        if (scaled_texture_ == 0) {
            glGenTextures(1, &scaled_texture_);
            glBindTexture(GL_TEXTURE_2D, scaled_texture_);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        } else {
            glBindTexture(GL_TEXTURE_2D, scaled_texture_);
        }

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, target_w, target_h, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        glBindTexture(GL_TEXTURE_2D, 0);

        scaled_tex_width_ = target_w;
        scaled_tex_height_ = target_h;
    }

    void FrameShareManager::destroyScaledResources() {
        if (scale_fbo_ != 0) {
            glDeleteFramebuffers(1, &scale_fbo_);
            scale_fbo_ = 0;
        }
        if (scale_draw_fbo_ != 0) {
            glDeleteFramebuffers(1, &scale_draw_fbo_);
            scale_draw_fbo_ = 0;
        }
        if (scaled_texture_ != 0) {
            glDeleteTextures(1, &scaled_texture_);
            scaled_texture_ = 0;
        }
        scaled_tex_width_ = 0;
        scaled_tex_height_ = 0;
    }

} // namespace lfs::vis
