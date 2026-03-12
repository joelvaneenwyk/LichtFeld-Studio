/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/event_bridge/event_bridge.hpp"
#include "core/event_bus.hpp"
#include "core/services.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "mcp/selection_client.hpp"
#include "operation/undo_history.hpp"
#include "rendering/rendering_manager.hpp"
#include "scene/scene_manager.hpp"
#include "visualizer/ipc/selection_server.hpp"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <gtest/gtest.h>
#include <memory>
#include <nlohmann/json.hpp>
#include <thread>
#include <vector>

using lfs::core::DataType;
using lfs::core::Device;
using lfs::core::Tensor;

namespace {

    Tensor make_uint8_mask(const std::vector<uint8_t>& values) {
        auto tensor = Tensor::empty({values.size()}, Device::CPU, DataType::UInt8);
        std::copy(values.begin(), values.end(), tensor.ptr<uint8_t>());
        return tensor.cuda();
    }

    std::shared_ptr<Tensor> make_screen_positions(const std::vector<float>& xy) {
        return std::make_shared<Tensor>(
            Tensor::from_vector(xy, {xy.size() / 2, size_t{2}}, Device::CUDA).to(DataType::Float32));
    }

    std::unique_ptr<lfs::core::SplatData> make_test_splat(const std::vector<float>& xyz) {
        const size_t count = xyz.size() / 3;
        auto means = Tensor::from_vector(xyz, {count, size_t{3}}, Device::CUDA).to(DataType::Float32);
        auto sh0 = Tensor::zeros({count, size_t{1}, size_t{3}}, Device::CUDA, DataType::Float32);
        auto shN = Tensor::zeros({count, size_t{3}, size_t{3}}, Device::CUDA, DataType::Float32);
        auto scaling = Tensor::zeros({count, size_t{3}}, Device::CUDA, DataType::Float32);

        std::vector<float> rotation_data(count * 4, 0.0f);
        for (size_t i = 0; i < count; ++i) {
            rotation_data[i * 4] = 1.0f;
        }
        auto rotation = Tensor::from_vector(rotation_data, {count, size_t{4}}, Device::CUDA).to(DataType::Float32);
        auto opacity = Tensor::zeros({count, size_t{1}}, Device::CUDA, DataType::Float32);

        return std::make_unique<lfs::core::SplatData>(
            1,
            std::move(means),
            std::move(sh0),
            std::move(shN),
            std::move(scaling),
            std::move(rotation),
            std::move(opacity),
            1.0f);
    }

    std::vector<uint8_t> selection_values(const lfs::vis::SceneManager& scene_manager) {
        const auto mask = scene_manager.getScene().getSelectionMask();
        if (!mask || !mask->is_valid()) {
            return {};
        }
        return mask->cpu().to_vector_uint8();
    }

} // namespace

class SelectionIpcRoundTripTest : public ::testing::Test {
protected:
    void SetUp() override {
        lfs::event::EventBridge::instance().clear_all();
        lfs::core::event::bus().clear_all();
        lfs::vis::services().clear();
        lfs::vis::op::undoHistory().clear();

        rendering_manager_ = std::make_unique<lfs::vis::RenderingManager>();
        scene_manager_ = std::make_unique<lfs::vis::SceneManager>();
        lfs::vis::services().set(rendering_manager_.get());
        lfs::vis::services().set(scene_manager_.get());

        scene_manager_->getScene().addNode(
            "test",
            make_test_splat({
                0.0f, 0.0f, 0.0f,
                1.0f, 0.0f, 0.0f,
            }));
        scene_manager_->initSelectionService();

        auto* const service = scene_manager_->getSelectionService();
        ASSERT_NE(service, nullptr);
        service->setTestingViewport({
            .x = 0.0f,
            .y = 0.0f,
            .width = 100.0f,
            .height = 100.0f,
            .render_width = 100,
            .render_height = 100,
        });

        const auto unique_id = std::chrono::steady_clock::now().time_since_epoch().count();
        socket_path_ = (std::filesystem::temp_directory_path() /
                        ("lichtfeld-selection-test-" + std::to_string(unique_id) + ".sock")).string();
        server_ = std::make_unique<lfs::vis::SelectionServer>();
        ASSERT_TRUE(server_->start(socket_path_));

        client_ = std::make_unique<lfs::mcp::SelectionClient>(socket_path_);
        wait_for_server();
    }

    void TearDown() override {
        client_.reset();
        if (server_) {
            server_->stop();
        }
        server_.reset();
        std::error_code ec;
        std::filesystem::remove(socket_path_, ec);

        lfs::event::EventBridge::instance().clear_all();
        lfs::core::event::bus().clear_all();
        lfs::vis::services().clear();
        scene_manager_.reset();
        rendering_manager_.reset();
        lfs::vis::op::undoHistory().clear();
    }

    void set_initial_selection(const std::vector<uint8_t>& values) {
        scene_manager_->getScene().setSelectionMask(std::make_shared<Tensor>(make_uint8_mask(values)));
    }

    lfs::vis::SelectionService& service() {
        return *scene_manager_->getSelectionService();
    }

    void process_server_queue() {
        server_->process_pending_commands();
    }

    void wait_for_server() {
        constexpr int MAX_ATTEMPTS = 50;
        for (int i = 0; i < MAX_ATTEMPTS; ++i) {
            if (client_->is_gui_running()) {
                return;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        FAIL() << "Selection server did not accept connections";
    }

    std::unique_ptr<lfs::vis::RenderingManager> rendering_manager_;
    std::unique_ptr<lfs::vis::SceneManager> scene_manager_;
    std::unique_ptr<lfs::vis::SelectionServer> server_;
    std::unique_ptr<lfs::mcp::SelectionClient> client_;
    std::string socket_path_;
};

TEST_F(SelectionIpcRoundTripTest, RectRoundTripUsesQueuedCommandProcessing) {
    service().setTestingScreenPositions(make_screen_positions({
        80.0f, 80.0f,
        10.0f, 10.0f,
    }));

    auto result = client_->select_rect(0.0f, 0.0f, 30.0f, 30.0f, "replace", 0);
    ASSERT_TRUE(result.has_value()) << result.error();

    process_server_queue();

    EXPECT_EQ(selection_values(*scene_manager_), (std::vector<uint8_t>{0, 1}));
}

TEST_F(SelectionIpcRoundTripTest, PolygonRoundTripUsesCameraSpecificProjection) {
    service().setTestingScreenPositions(make_screen_positions({
        10.0f, 10.0f,
        80.0f, 80.0f,
    }));
    service().setTestingScreenPositionsForCamera(7, make_screen_positions({
        80.0f, 80.0f,
        10.0f, 10.0f,
    }));

    auto result = client_->select_polygon({
        0.0f, 0.0f,
        30.0f, 0.0f,
        0.0f, 30.0f,
    }, "replace", 7);
    ASSERT_TRUE(result.has_value()) << result.error();

    process_server_queue();

    EXPECT_EQ(selection_values(*scene_manager_), (std::vector<uint8_t>{0, 1}));
}

TEST_F(SelectionIpcRoundTripTest, LassoRoundTripUsesQueuedCommandProcessing) {
    service().setTestingScreenPositions(make_screen_positions({
        80.0f, 80.0f,
        10.0f, 10.0f,
    }));

    auto result = client_->select_lasso({
        0.0f, 0.0f,
        30.0f, 0.0f,
        0.0f, 30.0f,
    }, "replace", 0);
    ASSERT_TRUE(result.has_value()) << result.error();

    process_server_queue();

    EXPECT_EQ(selection_values(*scene_manager_), (std::vector<uint8_t>{0, 1}));
}

TEST_F(SelectionIpcRoundTripTest, PolygonRoundTripRejectsMalformedPointLists) {
    auto result = client_->select_polygon({
        0.0f, 0.0f,
        30.0f, 0.0f,
    }, "replace", 0);

    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), "Polygon requires at least 3 points");
    EXPECT_TRUE(selection_values(*scene_manager_).empty());
}

TEST_F(SelectionIpcRoundTripTest, LassoRoundTripRejectsOddLengthPointLists) {
    auto result = client_->select_lasso({
        0.0f, 0.0f,
        30.0f, 0.0f,
        0.0f, 30.0f,
        5.0f,
    }, "replace", 0);

    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), "Lasso points must be x/y pairs");
    EXPECT_TRUE(selection_values(*scene_manager_).empty());
}

TEST_F(SelectionIpcRoundTripTest, RingRoundTripUsesQueuedCommandProcessing) {
    service().setTestingHoveredGaussianId(1);

    auto result = client_->select_ring(50.0f, 50.0f, "replace", 0);
    ASSERT_TRUE(result.has_value()) << result.error();

    process_server_queue();

    EXPECT_EQ(selection_values(*scene_manager_), (std::vector<uint8_t>{0, 1}));
}

TEST_F(SelectionIpcRoundTripTest, BrushAndDeselectAllRoundTrip) {
    service().setTestingScreenPositions(make_screen_positions({
        10.0f, 10.0f,
        80.0f, 80.0f,
    }));

    auto brush_result = client_->select_brush(10.0f, 10.0f, 8.0f, "replace", 0);
    ASSERT_TRUE(brush_result.has_value()) << brush_result.error();
    process_server_queue();
    EXPECT_EQ(selection_values(*scene_manager_), (std::vector<uint8_t>{1, 0}));

    auto clear_result = client_->deselect_all();
    ASSERT_TRUE(clear_result.has_value()) << clear_result.error();
    process_server_queue();
    EXPECT_TRUE(selection_values(*scene_manager_).empty());
}

TEST_F(SelectionIpcRoundTripTest, ApplyMaskRoundTripReplacesSelection) {
    set_initial_selection({1, 0});

    auto result = client_->apply_mask({0, 1});
    ASSERT_TRUE(result.has_value()) << result.error();

    process_server_queue();

    EXPECT_EQ(selection_values(*scene_manager_), (std::vector<uint8_t>{0, 1}));
}

TEST_F(SelectionIpcRoundTripTest, RectRoundTripRespectsDepthFilterSettings) {
    service().setTestingScreenPositions(make_screen_positions({
        10.0f, 10.0f,
        20.0f, 20.0f,
    }));

    auto settings = rendering_manager_->getSettings();
    settings.depth_filter_enabled = true;
    settings.depth_filter_min = {-0.5f, -10000.0f, -1.0f};
    settings.depth_filter_max = {0.5f, 10000.0f, 1.0f};
    rendering_manager_->updateSettings(settings);

    auto result = client_->select_rect(0.0f, 0.0f, 30.0f, 30.0f, "replace", 0);
    ASSERT_TRUE(result.has_value()) << result.error();

    process_server_queue();

    EXPECT_EQ(selection_values(*scene_manager_), (std::vector<uint8_t>{1, 0}));
}

TEST_F(SelectionIpcRoundTripTest, RectRoundTripRespectsCropFilterSettings) {
    service().setTestingScreenPositions(make_screen_positions({
        10.0f, 10.0f,
        20.0f, 20.0f,
    }));

    const auto splat_id = scene_manager_->getScene().getNodeIdByName("test");
    ASSERT_NE(splat_id, lfs::core::NULL_NODE);

    const auto crop_id = scene_manager_->getScene().addCropBox("test.crop", splat_id);
    lfs::core::CropBoxData crop_data;
    crop_data.min = {-0.5f, -0.5f, -0.5f};
    crop_data.max = {0.5f, 0.5f, 0.5f};
    crop_data.enabled = true;
    scene_manager_->getScene().setCropBoxData(crop_id, crop_data);

    auto settings = rendering_manager_->getSettings();
    settings.crop_filter_for_selection = true;
    rendering_manager_->updateSettings(settings);

    auto result = client_->select_rect(0.0f, 0.0f, 30.0f, 30.0f, "replace", 0);
    ASSERT_TRUE(result.has_value()) << result.error();

    process_server_queue();

    EXPECT_EQ(selection_values(*scene_manager_), (std::vector<uint8_t>{1, 0}));
}

TEST_F(SelectionIpcRoundTripTest, CapabilityRoundTripReturnsStructuredJson) {
    server_->setInvokeCapabilityCallback([](const std::string& name, const std::string& args) {
        nlohmann::json payload;
        payload["capability"] = name;
        payload["args"] = nlohmann::json::parse(args);
        payload["ok"] = true;
        return lfs::vis::CapabilityInvokeResult{
            .success = true,
            .result_json = payload.dump(),
            .error = {},
        };
    });

    auto result = client_->invoke_capability("selection.by_text", R"({"query":"wheel"})");
    ASSERT_TRUE(result.has_value()) << result.error();
    ASSERT_TRUE(result->success);

    const auto parsed = nlohmann::json::parse(result->result_json);
    EXPECT_EQ(parsed.value("capability", ""), "selection.by_text");
    EXPECT_EQ(parsed["args"].value("query", ""), "wheel");
    EXPECT_TRUE(parsed.value("ok", false));
}
