# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regression tests for generated plugin templates."""

from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from types import ModuleType
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PYTHON = PROJECT_ROOT / "src" / "python"
if str(SOURCE_PYTHON) in sys.path:
    sys.path.remove(str(SOURCE_PYTHON))
sys.path.insert(0, str(SOURCE_PYTHON))

from lfs_plugins.templates import create_plugin


def test_create_plugin_generates_unified_panel_template(tmp_path):
    plugin_dir = create_plugin("example_plugin", tmp_path)

    panel_py = plugin_dir / "panels" / "main_panel.py"
    panel_rml = plugin_dir / "panels" / "main_panel.rml"
    panel_rcss = plugin_dir / "panels" / "main_panel.rcss"

    assert panel_py.exists()
    assert not panel_rml.exists()
    assert not panel_rcss.exists()

    sys.path.insert(0, str(tmp_path))
    original_lf = sys.modules.get("lichtfeld")
    try:
        fake_lf = ModuleType("lichtfeld")
        fake_lf.ui = SimpleNamespace(Panel=type("Panel", (), {}))
        fake_lf.log = SimpleNamespace(info=lambda *args, **kwargs: None)
        fake_lf.register_class = lambda _cls: None
        fake_lf.unregister_class = lambda _cls: None
        sys.modules["lichtfeld"] = fake_lf
        sys.modules.pop("example_plugin", None)
        sys.modules.pop("example_plugin.panels", None)
        sys.modules.pop("example_plugin.panels.main_panel", None)

        module = import_module("example_plugin.panels.main_panel")
        panel_cls = module.MainPanel

        assert panel_cls.__mro__[1].__name__ == "Panel"
        assert "def draw(self, ui):" in panel_py.read_text()
        assert "lf.ui.Panel" in panel_py.read_text()
    finally:
        if original_lf is None:
            sys.modules.pop("lichtfeld", None)
        else:
            sys.modules["lichtfeld"] = original_lf
        sys.path.remove(str(tmp_path))
