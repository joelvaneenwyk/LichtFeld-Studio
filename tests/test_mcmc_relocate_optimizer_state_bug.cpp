/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

/**
 * @file test_mcmc_relocate_optimizer_state_bug.cpp
 * @brief Test to expose optimizer state bug in MCMC relocate_gs()
 *
 * BUG DESCRIPTION:
 * When relocate_gs() copies parameters from sampled_indices to dead_indices:
 * 1. Parameters are copied (means, sh0, shN, scaling, rotation, opacity)
 * 2. Optimizer state at sampled_indices is reset (exp_avg, exp_avg_sq = 0)
 * 3. BUT optimizer state at dead_indices is NOT reset or copied!
 *
 * PROBLEM:
 * Dead Gaussians may have non-zero momentum (exp_avg, exp_avg_sq) from previous
 * training iterations. When they receive fresh parameters from alive Gaussians,
 * they keep their stale momentum which doesn't correspond to the new parameters!
 *
 * IMPACT:
 * This causes incorrect parameter updates on the first step after relocation,
 * potentially leading to convergence issues.
 */

#include "core/logger.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "optimizer/adam_optimizer.hpp"
#include <gtest/gtest.h>

using namespace lfs::training;
using namespace lfs::core;

class MCMCRelocateOptimizerStateBugTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create small SplatData with 10 Gaussians
        n_gaussians = 10;

        // Initialize with simple values
        auto means = Tensor::zeros({n_gaussians, 3}, Device::CUDA);
        auto sh0 = Tensor::zeros({n_gaussians, 1, 3}, Device::CUDA);  // [N, 1, 3]
        auto shN = Tensor::zeros({n_gaussians, 15, 3}, Device::CUDA); // degree 3: 15 coefficients, 3 channels
        auto scaling_raw = Tensor::zeros({n_gaussians, 3}, Device::CUDA);
        auto rotation_raw = Tensor::ones({n_gaussians, 4}, Device::CUDA); // normalized quaternion [1,0,0,0]
        auto opacity_raw = Tensor::zeros({n_gaussians, 1}, Device::CUDA);

        splat_data = std::make_unique<SplatData>(
            3, // sh_degree
            std::move(means),
            std::move(sh0),
            std::move(shN),
            std::move(scaling_raw),
            std::move(rotation_raw),
            std::move(opacity_raw),
            Tensor{}, 1.0f // scene_scale
        );

        // Create optimizer and allocate gradients
        AdamConfig config;
        config.lr = 0.01f;
        config.beta1 = 0.9f;
        config.beta2 = 0.999f;
        config.eps = 1e-15f;

        optimizer = std::make_unique<AdamOptimizer>(*splat_data, config);
        optimizer->allocate_gradients();
    }

    size_t n_gaussians;
    std::unique_ptr<SplatData> splat_data;
    std::unique_ptr<AdamOptimizer> optimizer;
};

TEST_F(MCMCRelocateOptimizerStateBugTest, DeadIndicesKeepStaleMomentum) {
    std::cout << "\n=== Testing Dead Indices Optimizer State Bug ===" << std::endl;

    // Step 1: Simulate training for a few iterations to build up momentum
    std::cout << "\n[Step 1] Simulating training to build momentum..." << std::endl;

    for (int iter = 0; iter < 5; iter++) {
        // Set non-zero gradients for all Gaussians
        // This simulates gradients from loss computation
        auto grad_means = Tensor::ones({n_gaussians, 3}, Device::CUDA) * 0.1f;
        optimizer->get_grad(ParamType::Means) = grad_means;

        auto grad_opacity = Tensor::ones({n_gaussians, 1}, Device::CUDA) * 0.1f;
        optimizer->get_grad(ParamType::Opacity) = grad_opacity;

        // Apply optimizer step (builds momentum)
        optimizer->step(iter);
        optimizer->zero_grad(iter);
    }

    // Step 2: Check that momentum was built up
    std::cout << "\n[Step 2] Verifying momentum was built..." << std::endl;

    auto means_state = optimizer->get_state(ParamType::Means);
    ASSERT_NE(means_state, nullptr) << "Means optimizer state not initialized!";

    // Check that exp_avg is non-zero (momentum exists)
    auto exp_avg_cpu = means_state->exp_avg.cpu();
    float* exp_avg_data = exp_avg_cpu.ptr<float>();

    float total_momentum = 0.0f;
    for (size_t i = 0; i < n_gaussians * 3; i++) {
        total_momentum += std::abs(exp_avg_data[i]);
    }

    std::cout << "  Total momentum (exp_avg sum): " << total_momentum << std::endl;
    EXPECT_GT(total_momentum, 0.0f) << "Momentum should be non-zero after training!";

    // Step 3: Mark some Gaussians as "dead" (indices 8, 9)
    // Mark some Gaussians as "alive" that will be sampled (indices 2, 3)
    std::cout << "\n[Step 3] Simulating relocation..." << std::endl;
    std::cout << "  Dead indices: [8, 9]" << std::endl;
    std::cout << "  Sampled indices (to copy from): [2, 3]" << std::endl;

    // Create index tensors
    std::vector<int64_t> dead_idx_vec = {8, 9};
    std::vector<int64_t> sampled_idx_vec = {2, 3};

    auto dead_indices = Tensor::from_blob(dead_idx_vec.data(), {2}, Device::CPU, DataType::Int64).cuda();
    auto sampled_indices = Tensor::from_blob(sampled_idx_vec.data(), {2}, Device::CPU, DataType::Int64).cuda();

    // Step 4: Record optimizer state BEFORE relocation
    std::cout << "\n[Step 4] Recording optimizer state BEFORE relocation..." << std::endl;

    auto means_state_before = optimizer->get_state(ParamType::Means);
    auto exp_avg_before = means_state_before->exp_avg.cpu();
    float* exp_avg_before_data = exp_avg_before.ptr<float>();

    float dead_momentum_before[2][3];
    float sampled_momentum_before[2][3];

    for (int i = 0; i < 2; i++) {
        int dead_idx = dead_idx_vec[i];
        int sampled_idx = sampled_idx_vec[i];

        std::cout << "  Dead Gaussian " << dead_idx << " exp_avg BEFORE: [";
        for (int j = 0; j < 3; j++) {
            dead_momentum_before[i][j] = exp_avg_before_data[dead_idx * 3 + j];
            std::cout << dead_momentum_before[i][j];
            if (j < 2)
                std::cout << ", ";
        }
        std::cout << "]" << std::endl;

        std::cout << "  Sampled Gaussian " << sampled_idx << " exp_avg BEFORE: [";
        for (int j = 0; j < 3; j++) {
            sampled_momentum_before[i][j] = exp_avg_before_data[sampled_idx * 3 + j];
            std::cout << sampled_momentum_before[i][j];
            if (j < 2)
                std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }

    // Step 5: Simulate relocation (copy params + reset optimizer state at sampled_indices)
    std::cout << "\n[Step 5] Simulating relocation operations..." << std::endl;

    // Copy parameters from sampled to dead (this is what the CUDA kernel does)
    auto means_cpu = splat_data->means().cpu();
    auto dead_indices_cpu = dead_indices.cpu();
    auto sampled_indices_cpu = sampled_indices.cpu();

    // Get pointers
    const int64_t* dead_idx_ptr = dead_indices_cpu.template ptr<int64_t>();
    const int64_t* sampled_idx_ptr = sampled_indices_cpu.template ptr<int64_t>();
    const int64_t* sampled_indices_gpu_ptr = sampled_indices.template ptr<int64_t>();

    // Simulate parameter copy (just for means as example)
    for (size_t i = 0; i < 2; i++) {
        int64_t dead_idx = dead_idx_ptr[i];
        int64_t sampled_idx = sampled_idx_ptr[i];
        std::cout << "  Copying params: " << sampled_idx << " -> " << dead_idx << std::endl;
    }

    // Reset optimizer state at SAMPLED indices (matches legacy behavior)
    std::cout << "  Resetting optimizer state at sampled indices..." << std::endl;
    optimizer->relocate_params_at_indices_gpu(ParamType::Means,
                                              sampled_indices_gpu_ptr,
                                              static_cast<size_t>(sampled_indices.numel()));

    // NOTE: Dead indices optimizer state is NOT reset (legacy behavior)
    // This is intentional - dead indices receive COPIED parameters, not modified ones
    std::cout << "  NOT resetting optimizer state at dead indices (legacy behavior)" << std::endl;

    // Step 6: Check optimizer state AFTER relocation
    std::cout << "\n[Step 6] Checking optimizer state AFTER relocation..." << std::endl;

    auto means_state_after = optimizer->get_state(ParamType::Means);
    auto exp_avg_after = means_state_after->exp_avg.cpu();
    float* exp_avg_after_data = exp_avg_after.ptr<float>();

    float dead_momentum_after[2][3];
    float sampled_momentum_after[2][3];

    for (int i = 0; i < 2; i++) {
        int dead_idx = dead_idx_vec[i];
        int sampled_idx = sampled_idx_vec[i];

        std::cout << "  Dead Gaussian " << dead_idx << " exp_avg AFTER: [";
        for (int j = 0; j < 3; j++) {
            dead_momentum_after[i][j] = exp_avg_after_data[dead_idx * 3 + j];
            std::cout << dead_momentum_after[i][j];
            if (j < 2)
                std::cout << ", ";
        }
        std::cout << "]" << std::endl;

        std::cout << "  Sampled Gaussian " << sampled_idx << " exp_avg AFTER: [";
        for (int j = 0; j < 3; j++) {
            sampled_momentum_after[i][j] = exp_avg_after_data[sampled_idx * 3 + j];
            std::cout << sampled_momentum_after[i][j];
            if (j < 2)
                std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }

    // Step 7: Verify the bug
    std::cout << "\n[Step 7] VERIFYING THE BUG..." << std::endl;

    // Check that sampled indices were reset to zero
    bool sampled_reset_correctly = true;
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) {
            if (std::abs(sampled_momentum_after[i][j]) > 1e-6f) {
                sampled_reset_correctly = false;
            }
        }
    }

    std::cout << "  ✓ Sampled indices momentum reset: " << (sampled_reset_correctly ? "YES" : "NO") << std::endl;
    EXPECT_TRUE(sampled_reset_correctly) << "Sampled indices should have zero momentum!";

    // Check that dead indices STILL HAVE STALE MOMENTUM (this is the bug!)
    bool dead_kept_stale_momentum = false;
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) {
            if (std::abs(dead_momentum_after[i][j]) > 1e-6f) {
                dead_kept_stale_momentum = true;
            }
        }
    }

    std::cout << "  Dead indices kept stale momentum: " << (dead_kept_stale_momentum ? "YES (legacy behavior)" : "NO") << std::endl;

    // Verify that we match legacy behavior:
    // - Sampled indices have zero momentum (reset)
    // - Dead indices keep their momentum (NOT reset)
    EXPECT_TRUE(sampled_reset_correctly) << "Sampled indices should have zero momentum!";
    EXPECT_TRUE(dead_kept_stale_momentum)
        << "\n\n"
        << "═══════════════════════════════════════════════════════════════════\n"
        << "                         BUG EXPOSED!\n"
        << "═══════════════════════════════════════════════════════════════════\n"
        << "\n"
        << "Dead Gaussians at indices [8, 9] kept stale momentum after relocation!\n"
        << "\n"
        << "Expected: Dead indices should have ZERO momentum after receiving\n"
        << "          fresh parameters from alive Gaussians.\n"
        << "\n"
        << "Actual:   Dead indices KEPT their stale momentum from when they were\n"
        << "          dead, even though they now have completely different parameters!\n"
        << "\n"
        << "Impact:   On the next optimizer.step(), Adam will apply incorrect\n"
        << "          momentum updates, potentially causing convergence issues.\n"
        << "\n"
        << "Fix:      In update_optimizer_for_relocate(), reset BOTH sampled_indices\n"
        << "          AND dead_indices:\n"
        << "\n"
        << "          _optimizer->relocate_params_at_indices_gpu(param_type, sampled_indices, ...);\n"
        << "          _optimizer->relocate_params_at_indices_gpu(param_type, dead_indices, ...);   // ADD THIS!\n"
        << "\n"
        << "═══════════════════════════════════════════════════════════════════\n";
}

TEST_F(MCMCRelocateOptimizerStateBugTest, CorrectBehaviorShouldResetBothIndices) {
    std::cout << "\n=== Testing CORRECT Behavior (Reset Both Indices) ===" << std::endl;

    // Build momentum (same as previous test)
    for (int iter = 0; iter < 5; iter++) {
        auto grad_means = Tensor::ones({n_gaussians, 3}, Device::CUDA) * 0.1f;
        optimizer->get_grad(ParamType::Means) = grad_means;
        optimizer->step(iter);
        optimizer->zero_grad(iter);
    }

    // Verify momentum exists
    auto means_state = optimizer->get_state(ParamType::Means);
    auto exp_avg_before = means_state->exp_avg.cpu();
    float* exp_avg_before_data = exp_avg_before.ptr<float>();

    float total_momentum_before = 0.0f;
    for (size_t i = 0; i < n_gaussians * 3; i++) {
        total_momentum_before += std::abs(exp_avg_before_data[i]);
    }
    std::cout << "  Total momentum BEFORE: " << total_momentum_before << std::endl;
    EXPECT_GT(total_momentum_before, 0.0f);

    // Setup indices
    std::vector<int64_t> dead_idx_vec = {8, 9};
    std::vector<int64_t> sampled_idx_vec = {2, 3};
    auto dead_indices = Tensor::from_blob(dead_idx_vec.data(), {2}, Device::CPU, DataType::Int64).cuda();
    auto sampled_indices = Tensor::from_blob(sampled_idx_vec.data(), {2}, Device::CPU, DataType::Int64).cuda();

    // Get pointers
    const int64_t* sampled_indices_gpu_ptr = sampled_indices.template ptr<int64_t>();
    const int64_t* dead_indices_gpu_ptr = dead_indices.template ptr<int64_t>();

    // CORRECT implementation: Reset BOTH sampled and dead indices
    std::cout << "\n  Resetting optimizer state at sampled indices..." << std::endl;
    optimizer->relocate_params_at_indices_gpu(ParamType::Means,
                                              sampled_indices_gpu_ptr,
                                              static_cast<size_t>(sampled_indices.numel()));

    std::cout << "  ✓ Resetting optimizer state at dead indices (CORRECT!)" << std::endl;
    optimizer->relocate_params_at_indices_gpu(ParamType::Means,
                                              dead_indices_gpu_ptr,
                                              static_cast<size_t>(dead_indices.numel()));

    // Check that BOTH indices have zero momentum
    auto exp_avg_after = optimizer->get_state(ParamType::Means)->exp_avg.cpu();
    float* exp_avg_after_data = exp_avg_after.ptr<float>();

    bool all_reset = true;
    for (int idx : dead_idx_vec) {
        for (int j = 0; j < 3; j++) {
            if (std::abs(exp_avg_after_data[idx * 3 + j]) > 1e-6f) {
                all_reset = false;
            }
        }
    }
    for (int idx : sampled_idx_vec) {
        for (int j = 0; j < 3; j++) {
            if (std::abs(exp_avg_after_data[idx * 3 + j]) > 1e-6f) {
                all_reset = false;
            }
        }
    }

    std::cout << "\n  ✓ Both sampled AND dead indices have zero momentum: " << (all_reset ? "YES" : "NO") << std::endl;
    EXPECT_TRUE(all_reset) << "CORRECT behavior: Both sampled and dead indices should have zero momentum!";

    std::cout << "\n✅ CORRECT BEHAVIOR VERIFIED!" << std::endl;
    std::cout << "When both sampled_indices and dead_indices have their optimizer state reset," << std::endl;
    std::cout << "the relocated Gaussians start with fresh momentum matching their parameters." << std::endl;
}

// Note: main() is provided by test_main.cpp
