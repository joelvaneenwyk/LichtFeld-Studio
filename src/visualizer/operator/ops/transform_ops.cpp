/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "transform_ops.hpp"
#include "operation/undo_entry.hpp"
#include "operation/undo_history.hpp"
#include "operator/operator_registry.hpp"
#include "visualizer/gui_capabilities.hpp"

namespace lfs::vis::op {

    const OperatorDescriptor TransformSetOperator::DESCRIPTOR = {
        .builtin_id = BuiltinOp::TransformSet,
        .python_class_id = {},
        .label = "Set Transform",
        .description = "Set absolute transform values",
        .icon = "",
        .shortcut = "",
        .flags = OperatorFlags::REGISTER | OperatorFlags::UNDO,
        .source = OperatorSource::CPP,
    };

    bool TransformSetOperator::poll(const OperatorContext& ctx) const {
        return ctx.hasSelection();
    }

    OperatorResult TransformSetOperator::invoke(OperatorContext& ctx, OperatorProperties& props) {
        const auto nodes = ctx.selectedNodes();
        if (nodes.empty()) {
            return OperatorResult::CANCELLED;
        }

        const auto translation = props.get_or<glm::vec3>("translation", glm::vec3(0.0f));
        const auto rotation = props.get_or<glm::vec3>("rotation", glm::vec3(0.0f));
        const auto scale = props.get_or<glm::vec3>("scale", glm::vec3(1.0f));
        const auto result = cap::setTransform(
            ctx.scene(), nodes, translation, rotation, scale, "transform.set");
        return result ? OperatorResult::FINISHED : OperatorResult::CANCELLED;
    }

    const OperatorDescriptor TransformTranslateOperator::DESCRIPTOR = {
        .builtin_id = BuiltinOp::TransformTranslate,
        .python_class_id = {},
        .label = "Translate",
        .description = "Move selected nodes",
        .icon = "translate",
        .shortcut = "G",
        .flags = OperatorFlags::REGISTER | OperatorFlags::UNDO,
        .source = OperatorSource::CPP,
    };

    bool TransformTranslateOperator::poll(const OperatorContext& ctx) const {
        return ctx.hasSelection();
    }

    OperatorResult TransformTranslateOperator::invoke(OperatorContext& ctx, OperatorProperties& props) {
        const auto nodes = ctx.selectedNodes();
        if (nodes.empty()) {
            return OperatorResult::CANCELLED;
        }

        const auto value = props.get_or<glm::vec3>("value", glm::vec3(0.0f));
        const auto result = cap::translateNodes(ctx.scene(), nodes, value, "transform.translate");
        return result ? OperatorResult::FINISHED : OperatorResult::CANCELLED;
    }

    const OperatorDescriptor TransformRotateOperator::DESCRIPTOR = {
        .builtin_id = BuiltinOp::TransformRotate,
        .python_class_id = {},
        .label = "Rotate",
        .description = "Rotate selected nodes",
        .icon = "rotate",
        .shortcut = "R",
        .flags = OperatorFlags::REGISTER | OperatorFlags::UNDO,
        .source = OperatorSource::CPP,
    };

    bool TransformRotateOperator::poll(const OperatorContext& ctx) const {
        return ctx.hasSelection();
    }

    OperatorResult TransformRotateOperator::invoke(OperatorContext& ctx, OperatorProperties& props) {
        const auto nodes = ctx.selectedNodes();
        if (nodes.empty()) {
            return OperatorResult::CANCELLED;
        }

        const auto value = props.get_or<glm::vec3>("value", glm::vec3(0.0f));
        const auto result = cap::rotateNodes(ctx.scene(), nodes, value, "transform.rotate");
        return result ? OperatorResult::FINISHED : OperatorResult::CANCELLED;
    }

    const OperatorDescriptor TransformScaleOperator::DESCRIPTOR = {
        .builtin_id = BuiltinOp::TransformScale,
        .python_class_id = {},
        .label = "Scale",
        .description = "Scale selected nodes",
        .icon = "scale",
        .shortcut = "S",
        .flags = OperatorFlags::REGISTER | OperatorFlags::UNDO,
        .source = OperatorSource::CPP,
    };

    bool TransformScaleOperator::poll(const OperatorContext& ctx) const {
        return ctx.hasSelection();
    }

    OperatorResult TransformScaleOperator::invoke(OperatorContext& ctx, OperatorProperties& props) {
        const auto nodes = ctx.selectedNodes();
        if (nodes.empty()) {
            return OperatorResult::CANCELLED;
        }

        const auto value = props.get_or<glm::vec3>("value", glm::vec3(1.0f));
        const auto result = cap::scaleNodes(ctx.scene(), nodes, value, "transform.scale");
        return result ? OperatorResult::FINISHED : OperatorResult::CANCELLED;
    }

    const OperatorDescriptor TransformApplyBatchOperator::DESCRIPTOR = {
        .builtin_id = BuiltinOp::TransformApplyBatch,
        .python_class_id = {},
        .label = "Apply Batch Transform",
        .description = "Apply pre-computed transforms with undo support",
        .icon = "",
        .shortcut = "",
        .flags = OperatorFlags::UNDO,
        .source = OperatorSource::CPP,
    };

    bool TransformApplyBatchOperator::poll(const OperatorContext& /*ctx*/) const {
        return true;
    }

    OperatorResult TransformApplyBatchOperator::invoke(OperatorContext& ctx, OperatorProperties& props) {
        auto node_names = props.get<std::vector<std::string>>("node_names");
        auto old_transforms = props.get<std::vector<glm::mat4>>("old_transforms");
        if (!node_names || !old_transforms || node_names->empty()) {
            return OperatorResult::CANCELLED;
        }

        auto entry = std::make_unique<SceneSnapshot>(ctx.scene(), "transform.batch");
        if (!entry->captureTransformsBefore(*node_names, *old_transforms)) {
            return OperatorResult::CANCELLED;
        }
        entry->captureAfter();
        undoHistory().push(std::move(entry));

        return OperatorResult::FINISHED;
    }

    void registerTransformOperators() {
        operators().registerOperator(BuiltinOp::TransformSet, TransformSetOperator::DESCRIPTOR,
                                     [] { return std::make_unique<TransformSetOperator>(); });
        operators().registerOperator(BuiltinOp::TransformTranslate, TransformTranslateOperator::DESCRIPTOR,
                                     [] { return std::make_unique<TransformTranslateOperator>(); });
        operators().registerOperator(BuiltinOp::TransformRotate, TransformRotateOperator::DESCRIPTOR,
                                     [] { return std::make_unique<TransformRotateOperator>(); });
        operators().registerOperator(BuiltinOp::TransformScale, TransformScaleOperator::DESCRIPTOR,
                                     [] { return std::make_unique<TransformScaleOperator>(); });
        operators().registerOperator(BuiltinOp::TransformApplyBatch, TransformApplyBatchOperator::DESCRIPTOR,
                                     [] { return std::make_unique<TransformApplyBatchOperator>(); });
    }

    void unregisterTransformOperators() {
        operators().unregisterOperator(BuiltinOp::TransformSet);
        operators().unregisterOperator(BuiltinOp::TransformTranslate);
        operators().unregisterOperator(BuiltinOp::TransformRotate);
        operators().unregisterOperator(BuiltinOp::TransformScale);
        operators().unregisterOperator(BuiltinOp::TransformApplyBatch);
    }

} // namespace lfs::vis::op
