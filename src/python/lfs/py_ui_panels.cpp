/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/logger.hpp"
#include "py_rml.hpp"
#include "py_ui.hpp"
#include "python/python_runtime.hpp"
#include "python_panel_adapter.hpp"
#include "rml_im_mode_panel_adapter.hpp"
#include "rml_python_panel_adapter.hpp"
#include "visualizer/gui/panel_registry.hpp"
#include "visualizer/gui/rmlui/rml_theme.hpp"

#include <algorithm>
#include <array>
#include <mutex>
#include <optional>
#include <stdexcept>

namespace lfs::python {

    namespace gui = lfs::vis::gui;

    namespace {
        std::string get_class_id(nb::object cls) {
            auto mod = nb::cast<std::string>(cls.attr("__module__"));
            auto name = nb::cast<std::string>(cls.attr("__qualname__"));
            return mod + "." + name;
        }

        std::optional<PanelSpace> parse_panel_space(const std::string& str) {
            if (str == "SIDE_PANEL" || str == "PROPERTIES")
                return PanelSpace::SidePanel;
            if (str == "VIEWPORT_OVERLAY")
                return PanelSpace::ViewportOverlay;
            if (str == "DOCKABLE")
                return PanelSpace::Dockable;
            if (str == "MAIN_PANEL_TAB")
                return PanelSpace::MainPanelTab;
            if (str == "SCENE_HEADER")
                return PanelSpace::SceneHeader;
            if (str == "STATUS_BAR")
                return PanelSpace::StatusBar;
            if (str == "FLOATING")
                return PanelSpace::Floating;
            return std::nullopt;
        }

        PanelSpace normalize_panel_space(const PanelSpace space, const std::string& panel_idname) {
            if (space != PanelSpace::Dockable)
                return space;

            LOG_WARN("Panel '{}' requested DOCKABLE space. Docking has been retired, so the panel "
                     "will use the retained FLOATING window path instead.",
                     panel_idname);
            return PanelSpace::Floating;
        }

        nb::object panel_base_type() {
            static nb::object panel_type = nb::module_::import_("lfs_ui_panel").attr("Panel");
            return panel_type;
        }

        bool class_overrides(nb::object cls, nb::object base, const char* attr_name) {
            return nb::hasattr(cls, attr_name) &&
                   nb::hasattr(base, attr_name) &&
                   !cls.attr(attr_name).is(base.attr(attr_name));
        }

        std::string default_template_for_space(const PanelSpace space) {
            if (space == PanelSpace::Floating)
                return "rmlui/floating_window.rml";
            if (space == PanelSpace::StatusBar)
                return "rmlui/status_bar_panel.rml";
            return "rmlui/docked_panel.rml";
        }

        std::string default_immediate_document_for_space(const PanelSpace space) {
            if (space == PanelSpace::StatusBar)
                return "rmlui/im_mode_status_bar_panel.rml";
            return "rmlui/im_mode_panel.rml";
        }

        std::string resolve_template_identifier(const std::string& template_name,
                                                const PanelSpace space) {
            if (template_name.empty())
                return default_template_for_space(space);
            if (template_name == "builtin:docked-panel")
                return "rmlui/docked_panel.rml";
            if (template_name == "builtin:floating-window")
                return "rmlui/floating_window.rml";
            if (template_name == "builtin:status-bar")
                return "rmlui/status_bar_panel.rml";
            return template_name;
        }

        const std::string& retained_immediate_mode_style() {
            static std::string cached = []() {
                try {
                    return gui::rml_theme::loadBaseRCSS("rmlui/im_mode_panel.rcss");
                } catch (const std::exception& e) {
                    LOG_ERROR("Failed to load retained immediate-mode RCSS: {}", e.what());
                    return std::string{};
                }
            }();
            return cached;
        }

        std::string compose_retained_style(const std::string& style,
                                           const bool has_immediate_draw) {
            if (!has_immediate_draw)
                return style;

            const auto& im_mode_style = retained_immediate_mode_style();
            if (im_mode_style.empty())
                return style;
            if (style.empty())
                return im_mode_style;
            return im_mode_style + "\n" + style;
        }

        int parse_height_mode(nb::object panel_class) {
            if (!nb::hasattr(panel_class, "height_mode"))
                return 0;

            const auto mode = nb::cast<std::string>(panel_class.attr("height_mode"));
            return mode == "content" ? 1 : 0;
        }

        std::pair<float, float> parse_panel_size(nb::object panel_class) {
            if (!nb::hasattr(panel_class, "size"))
                return {0.0f, 0.0f};

            nb::object size_obj = panel_class.attr("size");
            if (!size_obj.is_valid() || size_obj.is_none())
                return {0.0f, 0.0f};

            if (!nb::isinstance<nb::tuple>(size_obj))
                throw std::runtime_error("size must be a tuple[float, float] or None");

            const nb::tuple size_tuple = nb::cast<nb::tuple>(size_obj);
            if (size_tuple.size() != 2)
                throw std::runtime_error("size must contain exactly two values");

            return {nb::cast<float>(size_tuple[0]), nb::cast<float>(size_tuple[1])};
        }

        bool uses_retained_panel(nb::object panel_class, nb::object panel_base,
                                 const PanelSpace space, const std::string& template_name,
                                 const std::string& style, const int height_mode) {
            if (space == PanelSpace::ViewportOverlay || !lfs::python::get_rml_manager())
                return false;

            if (!template_name.empty() || !style.empty() || height_mode != 0)
                return true;

            for (const auto* hook :
                 std::array{"on_bind_model", "on_mount", "on_unmount", "on_update", "on_scene_changed"}) {
                if (class_overrides(panel_class, panel_base, hook))
                    return true;
            }

            return false;
        }
    } // namespace

    PyPanelRegistry& PyPanelRegistry::instance() {
        static PyPanelRegistry registry;
        return registry;
    }

    void PyPanelRegistry::register_panel(nb::object panel_class) {
        std::lock_guard lock(mutex_);

        if (!panel_class.is_valid()) {
            LOG_ERROR("register_panel: invalid panel_class");
            return;
        }

        std::string label = "Python Panel";
        std::string idname = get_class_id(panel_class);
        PanelSpace space = PanelSpace::MainPanelTab;
        int order = 100;
        uint32_t options = 0;
        PollDependency poll_deps = PollDependency::ALL;
        std::string parent_idname;
        std::string template_name;
        std::string style;
        int height_mode = 0;
        float initial_width = 0.0f;
        float initial_height = 0.0f;

        try {
            if (nb::hasattr(panel_class, "idname")) {
                idname = nb::cast<std::string>(panel_class.attr("idname"));
            }
            if (nb::hasattr(panel_class, "label")) {
                label = nb::cast<std::string>(panel_class.attr("label"));
            }
            if (nb::hasattr(panel_class, "space")) {
                std::string space_str = nb::cast<std::string>(panel_class.attr("space"));
                if (!space_str.empty()) {
                    if (auto ps = parse_panel_space(space_str)) {
                        space = *ps;
                    } else {
                        LOG_WARN("Unknown panel space '{}' for panel '{}', defaulting to MainPanelTab", space_str, label);
                    }
                }
            }
            if (nb::hasattr(panel_class, "order")) {
                order = nb::cast<int>(panel_class.attr("order"));
            }
            if (nb::hasattr(panel_class, "parent")) {
                parent_idname = nb::cast<std::string>(panel_class.attr("parent"));
            }
            if (nb::hasattr(panel_class, "template")) {
                template_name = nb::cast<std::string>(panel_class.attr("template"));
            }
            if (nb::hasattr(panel_class, "style")) {
                style = nb::cast<std::string>(panel_class.attr("style"));
            }
            height_mode = parse_height_mode(panel_class);
            auto [parsed_width, parsed_height] = parse_panel_size(panel_class);
            initial_width = parsed_width;
            initial_height = parsed_height;
            nb::object opts;
            if (nb::hasattr(panel_class, "options")) {
                opts = panel_class.attr("options");
            }
            if (opts.is_valid() && nb::isinstance<nb::set>(opts)) {
                nb::set opts_set = nb::cast<nb::set>(opts);
                for (auto item : opts_set) {
                    std::string opt_str = nb::cast<std::string>(item);
                    if (opt_str == "DEFAULT_CLOSED") {
                        options |= static_cast<uint32_t>(gui::PanelOption::DEFAULT_CLOSED);
                    } else if (opt_str == "HIDE_HEADER") {
                        options |= static_cast<uint32_t>(gui::PanelOption::HIDE_HEADER);
                    }
                }
            }
            if (nb::hasattr(panel_class, "poll_deps")) {
                nb::object deps_obj = panel_class.attr("poll_deps");
                if (deps_obj.is_valid() && nb::isinstance<nb::set>(deps_obj)) {
                    poll_deps = PollDependency::NONE;
                    nb::set deps_set = nb::cast<nb::set>(deps_obj);
                    for (auto item : deps_set) {
                        std::string dep = nb::cast<std::string>(item);
                        if (dep == "SELECTION")
                            poll_deps = poll_deps | PollDependency::SELECTION;
                        else if (dep == "TRAINING")
                            poll_deps = poll_deps | PollDependency::TRAINING;
                        else if (dep == "SCENE")
                            poll_deps = poll_deps | PollDependency::SCENE;
                        else
                            LOG_WARN("Unknown poll dependency '{}' for panel '{}', ignoring", dep, label);
                    }
                }
            }
        } catch (const std::exception& e) {
            LOG_ERROR("register_panel: failed to extract attributes: {}", e.what());
            return;
        }

        space = normalize_panel_space(space, idname);

        LOG_DEBUG("Panel '{}' registered (space={})", label, static_cast<int>(space));

        nb::object instance;
        try {
            instance = panel_class();
        } catch (const std::exception& e) {
            LOG_ERROR("register_panel: failed to create instance for '{}': {}", label, e.what());
            return;
        }

        if (!instance.is_valid()) {
            LOG_ERROR("register_panel: invalid instance for '{}'", label);
            return;
        }

        const bool has_poll = nb::hasattr(panel_class, "poll");
        const nb::object panel_base = panel_base_type();
        const bool has_immediate_draw = class_overrides(panel_class, panel_base, "draw");
        const bool use_retained = uses_retained_panel(
            panel_class, panel_base, space, template_name, style, height_mode);
        const bool use_rml = (space != PanelSpace::ViewportOverlay) && lfs::python::get_rml_manager();

        if (space == PanelSpace::Floating && !use_rml) {
            LOG_ERROR("Panel '{}' ({}) requires the retained UI manager for FLOATING windows. "
                      "Window panels no longer fall back to the legacy ImGui wrapper path.",
                      label, idname);
            return;
        }

        std::shared_ptr<gui::IPanel> adapter;
        if (use_retained) {
            auto retained_adapter = std::make_shared<gui::RmlPythonPanelAdapter>(
                lfs::python::get_rml_manager(),
                instance,
                idname,
                resolve_template_identifier(template_name, space),
                compose_retained_style(style, has_immediate_draw),
                has_poll,
                height_mode,
                has_immediate_draw);
            if (to_gui_space(space) == gui::PanelSpace::Floating)
                retained_adapter->setForeground(true);
            adapter = retained_adapter;
        } else if (use_rml) {
            adapter = std::make_shared<gui::RmlImModePanelAdapter>(
                lfs::python::get_rml_manager(),
                instance,
                has_poll,
                default_immediate_document_for_space(space));
        } else {
            adapter = std::make_shared<PythonPanelAdapter>(instance, has_poll);
        }

        gui::PanelInfo info;
        info.panel = adapter;
        info.label = label;
        info.idname = idname;
        info.parent_idname = parent_idname;
        info.space = to_gui_space(space);
        info.order = order;
        info.options = options;
        info.poll_deps = static_cast<gui::PollDependency>(poll_deps);
        info.is_native = false;
        info.initial_width = initial_width;
        info.initial_height = initial_height;

        const bool default_closed =
            (options & static_cast<uint32_t>(gui::PanelOption::DEFAULT_CLOSED)) &&
            (info.space == gui::PanelSpace::Floating || info.space == gui::PanelSpace::Dockable);
        info.enabled = !default_closed;

        std::string module_prefix;
        try {
            module_prefix = nb::cast<std::string>(panel_class.attr("__module__"));
        } catch (...) {
        }

        if (!gui::PanelRegistry::instance().register_panel(std::move(info)))
            return;
        panels_[idname] = {adapter, module_prefix};
    }

    void PyPanelRegistry::unregister_panel(nb::object panel_class) {
        std::lock_guard lock(mutex_);

        std::string idname;
        if (nb::hasattr(panel_class, "idname")) {
            idname = nb::cast<std::string>(panel_class.attr("idname"));
        }
        if (idname.empty()) {
            idname = get_class_id(panel_class);
        }

        if (on_gl_thread()) {
            gui::PanelRegistry::instance().unregister_panel(idname);
        } else {
            schedule_gl_callback([id = idname]() {
                gui::PanelRegistry::instance().unregister_panel(id);
            });
        }
        panels_.erase(idname);
    }

    void PyPanelRegistry::unregister_all() {
        std::lock_guard lock(mutex_);
        if (on_gl_thread()) {
            gui::PanelRegistry::instance().unregister_all_non_native();
        } else {
            schedule_gl_callback([]() {
                gui::PanelRegistry::instance().unregister_all_non_native();
            });
        }
        panels_.clear();
    }

    void PyPanelRegistry::unregister_for_module(const std::string& prefix) {
        std::lock_guard lock(mutex_);

        std::vector<std::string> to_remove;
        for (const auto& [idname, entry] : panels_) {
            if (entry.module_prefix == prefix || entry.module_prefix.starts_with(prefix + ".")) {
                to_remove.push_back(idname);
            }
        }

        for (const auto& idname : to_remove) {
            if (on_gl_thread()) {
                gui::PanelRegistry::instance().unregister_panel(idname);
            } else {
                schedule_gl_callback([id = idname]() {
                    gui::PanelRegistry::instance().unregister_panel(id);
                });
            }
            panels_.erase(idname);
            LOG_INFO("Unregistered panel '{}' for module '{}'", idname, prefix);
        }
    }

    void register_ui_panels(nb::module_& m) {
        nb::enum_<PanelSpace>(m, "PanelSpace")
            .value("SIDE_PANEL", PanelSpace::SidePanel)
            .value("FLOATING", PanelSpace::Floating)
            .value("VIEWPORT_OVERLAY", PanelSpace::ViewportOverlay)
            .value("DOCKABLE", PanelSpace::Dockable)
            .value("MAIN_PANEL_TAB", PanelSpace::MainPanelTab)
            .value("SCENE_HEADER", PanelSpace::SceneHeader)
            .value("STATUS_BAR", PanelSpace::StatusBar);
        m.attr("Panel") = panel_base_type();

        m.def(
            "unregister_all_panels", []() {
                PyPanelRegistry::instance().unregister_all();
            },
            "Unregister all Python panels");

        m.def(
            "unregister_panels_for_module",
            [](const std::string& prefix) {
                PyPanelRegistry::instance().unregister_for_module(prefix);
            },
            nb::arg("module_prefix"),
            "Unregister all panels registered by a given module prefix");

        m.def(
            "get_panel_names", [](const std::string& space) {
                auto ps = parse_panel_space(space).value_or(PanelSpace::Floating);
                return gui::PanelRegistry::instance().get_panel_names(to_gui_space(ps));
            },
            nb::arg("space") = "FLOATING", "Get registered panel names for a given space");

        m.def(
            "set_panel_enabled", [](const std::string& idname, bool enabled) {
                gui::PanelRegistry::instance().set_panel_enabled(idname, enabled);
            },
            nb::arg("idname"), nb::arg("enabled"), "Enable or disable a panel by idname");

        m.def(
            "is_panel_enabled", [](const std::string& idname) {
                return gui::PanelRegistry::instance().is_panel_enabled(idname);
            },
            nb::arg("idname"), "Check if a panel is enabled");

        m.def(
            "get_main_panel_tabs", []() {
                auto tabs = gui::PanelRegistry::instance().get_panels_for_space(gui::PanelSpace::MainPanelTab);
                nb::list result;
                for (const auto& tab : tabs) {
                    nb::dict info;
                    info["idname"] = tab.idname;
                    info["label"] = tab.label;
                    info["order"] = tab.order;
                    info["enabled"] = tab.enabled;
                    result.append(info);
                }
                return result;
            },
            "Get all main panel tabs as list of dicts");

        m.def(
            "get_panel", [](const std::string& idname) -> nb::object {
                auto panel = gui::PanelRegistry::instance().get_panel(idname);
                if (!panel.has_value()) {
                    return nb::none();
                }
                nb::dict info;
                info["idname"] = panel->idname;
                info["label"] = panel->label;
                info["order"] = panel->order;
                info["enabled"] = panel->enabled;
                info["space"] = static_cast<int>(panel->space);
                return info;
            },
            nb::arg("idname"), "Get panel info by idname (None if not found)");

        m.def(
            "set_panel_label", [](const std::string& idname, const std::string& new_label) {
                return gui::PanelRegistry::instance().set_panel_label(idname, new_label);
            },
            nb::arg("idname"), nb::arg("label"), "Set the display label for a panel");

        m.def(
            "set_panel_order", [](const std::string& idname, int new_order) {
                return gui::PanelRegistry::instance().set_panel_order(idname, new_order);
            },
            nb::arg("idname"), nb::arg("order"), "Set the sort order for a panel");

        m.def(
            "set_panel_space", [](const std::string& idname, const std::string& space_str) {
                auto ps = parse_panel_space(space_str);
                if (!ps) {
                    LOG_WARN("Unknown panel space '{}' for panel '{}', defaulting to MainPanelTab", space_str, idname);
                }
                auto gui_space = to_gui_space(normalize_panel_space(ps.value_or(PanelSpace::MainPanelTab), idname));
                return gui::PanelRegistry::instance().set_panel_space(idname, gui_space);
            },
            nb::arg("idname"), nb::arg("space"), "Set the panel space (where it renders)");

        m.def(
            "set_panel_parent", [](const std::string& idname, const std::string& parent_idname) {
                return gui::PanelRegistry::instance().set_panel_parent(idname, parent_idname);
            },
            nb::arg("idname"), nb::arg("parent"), "Set the parent panel (embeds as collapsible section)");

        m.def(
            "has_main_panel_tabs", []() {
                return gui::PanelRegistry::instance().has_panels(gui::PanelSpace::MainPanelTab);
            },
            "Check if any main panel tabs are registered");
    }

} // namespace lfs::python
