# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Plugin template generator for scaffolding new plugins."""

import logging
from pathlib import Path
from typing import Optional

_log = logging.getLogger(__name__)

PYPROJECT_TOML = '''[project]
name = "{name}"
version = "0.1.0"
description = "A new LichtFeld plugin"

[tool.lichtfeld]
hot_reload = true
'''

INIT_PY = '''"""
{name} - A LichtFeld Studio plugin.
"""

import lichtfeld as lf
from .panels.main_panel import MainPanel

_classes = [MainPanel]


def on_load():
    """Called when plugin is loaded."""
    for cls in _classes:
        lf.register_class(cls)
    lf.log.info("{name} plugin loaded")


def on_unload():
    """Called when plugin is unloaded."""
    for cls in reversed(_classes):
        lf.unregister_class(cls)
    lf.log.info("{name} plugin unloaded")
'''

MAIN_PANEL_PY = '''"""Main panel for {name} plugin."""

import lichtfeld as lf


class MainPanel(lf.ui.Panel):
    """Example plugin panel using the unified Panel API."""

    idname = "{name}.main_panel"
    label = "{title}"
    space = "MAIN_PANEL_TAB"
    order = 100

    def __init__(self):
        self._click_count = 0
        self._enabled = True
        self._strength = 0.35

    def draw(self, ui):
        ui.heading("{title}")
        ui.text_disabled("Simple plugin panel using the unified Panel API")

        if ui.button(f"Click Me ({{self._click_count}})"):
            self._click_count += 1
            lf.log.info("{name}: Button clicked")
        ui.same_line()
        if ui.button_styled("Reset", "secondary"):
            self._click_count = 0
            self._enabled = True
            self._strength = 0.35

        _, self._enabled = ui.checkbox("Enabled", self._enabled)
        _, self._strength = ui.slider_float("Strength", self._strength, 0.0, 1.0)

        ui.separator()
        ui.text_wrapped(
            f"enabled={{self._enabled}} | strength={{self._strength:.2f}} | clicks={{self._click_count}}"
        )
'''


def create_plugin(name: str, target_dir: Optional[Path] = None) -> Path:
    """Create a new plugin from template.

    Args:
        name: Plugin name (used for directory and module)
        target_dir: Optional target directory (defaults to ~/.lichtfeld/plugins)

    Returns:
        Path to created plugin directory

    Raises:
        FileExistsError: If plugin directory already exists
    """
    if target_dir is None:
        target_dir = Path.home() / ".lichtfeld" / "plugins"

    plugin_dir = target_dir / name
    if plugin_dir.exists():
        raise FileExistsError(f"Plugin directory already exists: {plugin_dir}")

    title = name.replace("_", " ").title()

    plugin_dir.mkdir(parents=True, exist_ok=True)
    (plugin_dir / "panels").mkdir(exist_ok=True)

    (plugin_dir / "pyproject.toml").write_text(PYPROJECT_TOML.format(name=name))
    (plugin_dir / "__init__.py").write_text(INIT_PY.format(name=name))
    (plugin_dir / "panels" / "__init__.py").write_text("")
    (plugin_dir / "panels" / "main_panel.py").write_text(
        MAIN_PANEL_PY.format(name=name, title=title)
    )

    _log.info("Created plugin template at %s", plugin_dir)
    return plugin_dir
