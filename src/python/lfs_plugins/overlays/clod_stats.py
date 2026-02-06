"""CLoD runtime stats overlay for viewport."""

import lichtfeld as lf

from ..types import Panel

OVERLAY_FLAGS = (
    lf.ui.UILayout.WindowFlags.NoTitleBar
    | lf.ui.UILayout.WindowFlags.NoResize
    | lf.ui.UILayout.WindowFlags.NoMove
    | lf.ui.UILayout.WindowFlags.NoScrollbar
    | lf.ui.UILayout.WindowFlags.NoInputs
    | lf.ui.UILayout.WindowFlags.NoFocusOnAppearing
    | lf.ui.UILayout.WindowFlags.NoBringToFrontOnFocus
    | lf.ui.UILayout.WindowFlags.AlwaysAutoResize
)


class ClodStatsOverlay(Panel):
    label = "##CLoDStats"
    space = "VIEWPORT_OVERLAY"
    order = 20

    @classmethod
    def poll(cls, context):
        rendered, total, active = lf.ui.get_clod_render_stats()
        return active and total > 0 and not lf.ui.is_startup_visible()

    def draw(self, layout):
        rendered, total, active = lf.ui.get_clod_render_stats()
        if not active or total <= 0:
            return

        scale = layout.get_dpi_scale()
        theme = lf.ui.theme()

        vp_x, vp_y = layout.get_viewport_pos()
        vp_w, _ = layout.get_viewport_size()

        panel_w = 220.0 * scale
        margin = 12.0 * scale
        panel_x = vp_x + vp_w - panel_w - margin
        panel_y = vp_y + margin

        layout.set_next_window_pos((panel_x, panel_y))
        layout.set_next_window_size((panel_w, 0.0))

        surface = theme.palette.surface
        layout.push_style_color("WindowBg", (surface[0], surface[1], surface[2], 0.75))
        layout.push_style_var("WindowRounding", theme.sizes.popup_rounding * scale)
        layout.push_style_var_vec2("WindowPadding", (12.0 * scale, 10.0 * scale))

        if not layout.begin_window("##CLoDStatsOverlay", OVERLAY_FLAGS):
            layout.end_window()
            layout.pop_style_var(2)
            layout.pop_style_color(1)
            return

        text_color = theme.palette.overlay_text
        dim_color = theme.palette.overlay_text_dim

        layout.text_colored("CLoD", text_color)

        ratio = rendered / total
        culled = total - rendered

        layout.text_colored(f"{rendered:,} / {total:,}", dim_color)
        layout.text_colored(f"Culled: {culled:,} ({(1.0 - ratio) * 100.0:.1f}%)", dim_color)

        layout.progress_bar(ratio, f"{ratio * 100.0:.1f}%", -1)

        layout.end_window()
        layout.pop_style_var(2)
        layout.pop_style_color(1)
