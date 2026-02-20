/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/panels/frame_share_panel.hpp"
#include "core/event_bridge/localization_manager.hpp"
#include "frame_share/frame_share_manager.hpp"
#include "gui/string_keys.hpp"
#include "theme/theme.hpp"
#include <cassert>
#include <cstring>
#include <imgui.h>

using namespace lichtfeld::Strings;

namespace lfs::vis::gui::panels {

    FrameSharePanel::FrameSharePanel(FrameShareManager* manager)
        : manager_(manager) {
        assert(manager_);
        std::strncpy(name_buf_, manager_->getSenderName().c_str(), sizeof(name_buf_) - 1);
        fps_limit_ = manager_->getTargetFps();
        resolution_index_ = static_cast<int>(manager_->getOutputResolution());
    }

    void FrameSharePanel::draw(const PanelDrawContext& /*ctx*/) {
        bool enabled = manager_->isEnabled();
        if (ImGui::Checkbox(LOC(FrameShare::ENABLE), &enabled)) {
            manager_->setEnabled(enabled);
        }

        ImGui::Separator();

        ImGui::TextUnformatted(LOC(FrameShare::BACKEND));
        ImGui::SameLine();

        auto current = manager_->getBackend();
        const char* backend_label = "None";

#ifdef LFS_SPOUT_ENABLED
        if (current == FrameShareBackend::Spout)
            backend_label = "Spout";
#endif
#if defined(__linux__)
        if (current == FrameShareBackend::SharedMemory)
            backend_label = "Shared Memory";
        if (current == FrameShareBackend::V4L2)
            backend_label = "V4L2 Loopback";
#endif

        if (ImGui::BeginCombo("##backend", backend_label)) {
#ifdef LFS_SPOUT_ENABLED
            if (ImGui::Selectable("Spout", current == FrameShareBackend::Spout)) {
                manager_->setBackend(FrameShareBackend::Spout);
            }
#endif
#if defined(__linux__)
            if (ImGui::Selectable("Shared Memory", current == FrameShareBackend::SharedMemory)) {
                manager_->setBackend(FrameShareBackend::SharedMemory);
            }
            if (ImGui::Selectable("V4L2 Loopback", current == FrameShareBackend::V4L2)) {
                manager_->setBackend(FrameShareBackend::V4L2);
            }
#endif
            ImGui::EndCombo();
        }

        ImGui::TextUnformatted(LOC(FrameShare::SENDER_NAME));
        ImGui::SetNextItemWidth(-1);
        if (ImGui::InputText("##sender_name", name_buf_, sizeof(name_buf_),
                             ImGuiInputTextFlags_EnterReturnsTrue)) {
            manager_->setSenderName(name_buf_);
        }

        ImGui::Spacing();
        ImGui::TextUnformatted(LOC(FrameShare::FPS_LIMIT));
        ImGui::SetNextItemWidth(-1);
        if (fps_limit_ < 1.0f) {
            ImGui::SliderFloat("##fps_limit", &fps_limit_, 0.0f, 120.0f, "Unlimited");
        } else {
            ImGui::SliderFloat("##fps_limit", &fps_limit_, 0.0f, 120.0f, "%.0f");
        }
        if (ImGui::IsItemDeactivatedAfterEdit()) {
            if (fps_limit_ < 1.0f)
                fps_limit_ = 0.0f;
            manager_->setTargetFps(fps_limit_);
        }

        ImGui::Spacing();
        ImGui::TextUnformatted(LOC(FrameShare::OUTPUT_RESOLUTION));
        ImGui::SetNextItemWidth(-1);
        static constexpr const char* resolution_labels[] = {"Native", "720p", "1080p", "1440p"};
        if (ImGui::Combo("##output_resolution", &resolution_index_, resolution_labels, IM_ARRAYSIZE(resolution_labels))) {
            manager_->setOutputResolution(static_cast<FrameShareResolution>(resolution_index_));
        }

        ImGui::Separator();

        const auto& t = theme();
        ImGui::TextUnformatted(LOC(FrameShare::STATUS));
        ImGui::SameLine();

        if (manager_->isActive()) {
            ImGui::TextColored(t.palette.success, "%s", LOC(FrameShare::ACTIVE));
            const int receivers = manager_->connectedReceivers();
            if (receivers >= 0) {
                ImGui::SameLine();
                ImGui::Text("(%d %s)", receivers, LOC(FrameShare::RECEIVERS));
            }
        } else if (manager_->isEnabled()) {
            ImGui::TextColored(t.palette.error, "%s", LOC(FrameShare::ERROR_STATE));
            const auto& msg = manager_->getStatusMessage();
            if (!msg.empty()) {
                ImGui::SameLine();
                ImGui::TextUnformatted(msg.c_str());
            }
        } else {
            ImGui::TextDisabled("%s", LOC(FrameShare::INACTIVE));
        }
    }

} // namespace lfs::vis::gui::panels
