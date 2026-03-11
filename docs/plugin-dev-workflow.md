# Plugin Developer Workflow

This guide explains the current plugin scaffolding workflow and the intended UI progression from a single `draw(ui)` file to full retained RML/RCSS.

## Scaffold paths

There are two supported ways to create a plugin:

### CLI scaffold

```bash
LichtFeld-Studio plugin create my_plugin
```

This creates the minimal plugin package and also adds local development helpers:

```text
~/.lichtfeld/plugins/my_plugin/
├── .venv/
├── .vscode/
│   ├── launch.json
│   └── settings.json
├── pyproject.toml
├── pyrightconfig.json
├── __init__.py
└── panels/
    ├── __init__.py
    └── main_panel.py
```

Useful companion commands:

```bash
LichtFeld-Studio plugin check my_plugin
LichtFeld-Studio plugin list
```

### Python scaffold

```python
import lichtfeld as lf

path = lf.plugins.create("my_plugin")
print(path)
```

This creates only the plugin package itself:

```text
~/.lichtfeld/plugins/my_plugin/
├── pyproject.toml
├── __init__.py
└── panels/
    ├── __init__.py
    └── main_panel.py
```

## Why the scaffold does not create `.rml` and `.rcss`

The scaffold is intentionally minimal. Most plugins start as a pure immediate-mode panel:

```python
import lichtfeld as lf


class MainPanel(lf.ui.Panel):
    id = "my_plugin.main_panel"
    label = "My Plugin"
    space = lf.ui.PanelSpace.MAIN_PANEL_TAB

    def draw(self, ui):
        ui.heading("Hello")
        ui.text_disabled("Start simple and keep state on self.")
```

That panel does not need any extra files. Creating `main_panel.rml` and `main_panel.rcss` up front would add noise to the default workflow and push authors into the advanced path before they need it.

Use the scaffold as step 1, then add files only when the panel actually needs retained DOM structure or a standalone stylesheet.

## Styling progression

### 1. Immediate-only panel

Stay in `main_panel.py` when you only need widgets, layout composition, and panel-local state.

Typical tools:

- `draw(self, ui)`
- `button_styled()`
- `row()`, `column()`, `box()`, `grid_flow()`
- `push_style_var()` and `push_style_color()` for local widget overrides

### 2. Mixed panel with inline RCSS

When you want retained shell styling but still want Python to generate the content, add `style`, `height_mode`, or retained hooks on the same panel class:

```python
class StatusPanel(lf.ui.Panel):
    id = "my_plugin.status"
    label = "Status"
    space = lf.ui.PanelSpace.STATUS_BAR
    height_mode = lf.ui.PanelHeightMode.CONTENT
    style = """
body.status-bar-panel { padding: 0 12dp; }
#im-root .im-label { color: #f3c96d; font-weight: bold; }
"""

    def draw(self, ui):
        ui.label("Ready")
```

Notes:

- `style` is inline RCSS text, not a file path.
- `STATUS_BAR` automatically gets the status-bar shell.
- `draw(ui)` still provides the actual panel content.

### 3. Hybrid panel with template and stylesheet

Add a template when you need custom DOM structure, model binding, or direct document event handling:

```python
from pathlib import Path


class HybridPanel(lf.ui.Panel):
    id = "my_plugin.hybrid"
    label = "Hybrid"
    template = str(Path(__file__).resolve().with_name("main_panel.rml"))
    height_mode = lf.ui.PanelHeightMode.CONTENT

    def draw(self, ui):
        ui.text_disabled("Rendered into #im-root")
```

At that point, add the sibling files:

```text
~/.lichtfeld/plugins/my_plugin/
└── panels/
    ├── main_panel.py
    ├── main_panel.rml
    └── main_panel.rcss
```

Important details:

- Use an absolute path for plugin-local `template` values.
- A sibling `main_panel.rcss` is loaded automatically when `main_panel.rml` exists.
- Keep `<div id="im-root"></div>` in the template if you want embedded `draw(ui)` content.

## Development loop

Load and reload from the app:

```python
import lichtfeld as lf

lf.plugins.discover()
lf.plugins.load("my_plugin")
lf.plugins.reload("my_plugin")
lf.plugins.start_watcher()
```

Inspect failures:

```python
state = lf.plugins.get_state("my_plugin")
error = lf.plugins.get_error("my_plugin")
traceback = lf.plugins.get_traceback("my_plugin")
```

Unload when needed:

```python
lf.plugins.unload("my_plugin")
```

## Dependencies

Declare plugin dependencies in `pyproject.toml`:

```toml
[project]
dependencies = [
    "numpy>=1.26",
]
```

Plugin environments are isolated per plugin. The CLI scaffold creates `.venv` immediately. The Python scaffold creates only the source files; dependency installation still happens through the plugin system when the plugin is loaded.

## IDE support

`LichtFeld-Studio plugin create` also writes:

- `.vscode/settings.json`
- `.vscode/launch.json`
- `pyrightconfig.json`

Those files point the editor at the plugin venv and the LichtFeld typings.

If you scaffold through `lf.plugins.create()`, configure your IDE manually or run the CLI scaffold path for new plugins when you want a ready-made editor setup.

## Recommended reading order

1. Read [docs/plugins/getting-started.md](plugins/getting-started.md).
2. Walk through [docs/plugins/examples/README.md](plugins/examples/README.md).
3. Use [docs/plugins/api-reference.md](plugins/api-reference.md) for exact panel, hook, and widget APIs.
