/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "gui/panel_registry.hpp"

namespace lfs::vis {
    class FrameShareManager;
}

namespace lfs::vis::gui::panels {

    class FrameSharePanel : public IPanel {
    public:
        explicit FrameSharePanel(FrameShareManager* manager);

        void draw(const PanelDrawContext& ctx) override;

    private:
        FrameShareManager* manager_;
        char name_buf_[256] = {};
        float fps_limit_ = 0.0f;
        int resolution_index_ = 0;
    };

} // namespace lfs::vis::gui::panels
