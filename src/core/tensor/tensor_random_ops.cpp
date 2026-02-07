/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/logger.hpp"
#include "internal/tensor_impl.hpp"
#include "internal/tensor_ops.hpp"
#include <atomic>
#include <curand.h>
#include <curand_kernel.h>
#include <mutex>
#include <random>

#define CHECK_CUDA(call)                              \
    do {                                              \
        cudaError_t error = call;                     \
        if (error != cudaSuccess) {                   \
            LOG_ERROR("CUDA error at {}:{} - {}: {}", \
                      __FILE__, __LINE__,             \
                      cudaGetErrorName(error),        \
                      cudaGetErrorString(error));     \
        }                                             \
    } while (0)

#define CHECK_CURAND(call)                             \
    do {                                               \
        curandStatus_t error = call;                   \
        if (error != CURAND_STATUS_SUCCESS) {          \
            LOG_ERROR("CURAND error at {}:{} - {}",    \
                      __FILE__, __LINE__, (int)error); \
        }                                              \
    } while (0)

namespace lfs::core {

    // ============= RandomGenerator Implementation =============

    // Private implementation class to hide atomic counter
    class RandomGeneratorImpl {
    public:
        std::atomic<uint64_t> call_counter_{0};
        uint64_t seed_ = 42;
        void* cuda_generator_ = nullptr;
        std::mt19937_64 cpu_generator_;
        std::mutex cuda_generator_mutex_;

        RandomGeneratorImpl() : seed_(42),
                                cpu_generator_(seed_) {
            // Initialize CUDA random generator with Philox (same as PyTorch - much faster!)
            curandGenerator_t* gen = new curandGenerator_t;
            CHECK_CURAND(curandCreateGenerator(gen, CURAND_RNG_PSEUDO_PHILOX4_32_10));
            CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(*gen, seed_));
            cuda_generator_ = gen;
        }

        ~RandomGeneratorImpl() {
            if (cuda_generator_) {
                curandGenerator_t* gen = static_cast<curandGenerator_t*>(cuda_generator_);
                curandDestroyGenerator(*gen);
                delete gen;
            }
        }
    };

    RandomGenerator& RandomGenerator::instance() {
        static RandomGenerator instance;
        return instance;
    }

    RandomGenerator::RandomGenerator() : seed_(42),
                                         cpu_generator_(seed_) {
        // Create implementation
        impl_ = new RandomGeneratorImpl();
    }

    RandomGenerator::~RandomGenerator() {
        if (impl_) {
            delete static_cast<RandomGeneratorImpl*>(impl_);
        }
    }

    void RandomGenerator::manual_seed(uint64_t seed) {
        seed_ = seed;
        cpu_generator_.seed(seed);

        auto* impl = static_cast<RandomGeneratorImpl*>(impl_);
        std::lock_guard<std::mutex> lock(impl->cuda_generator_mutex_);
        impl->seed_ = seed;
        impl->cpu_generator_.seed(seed);
        impl->call_counter_.store(0, std::memory_order_relaxed); // Reset call counter when seed is set

        if (impl->cuda_generator_) {
            curandGenerator_t* gen = static_cast<curandGenerator_t*>(impl->cuda_generator_);
            CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(*gen, seed));
            // IMPORTANT: Reset the offset to ensure reproducibility
            CHECK_CURAND(curandSetGeneratorOffset(*gen, 0));
        }
    }

    uint64_t RandomGenerator::get_next_cuda_seed() {
        auto* impl = static_cast<RandomGeneratorImpl*>(impl_);
        uint64_t base_seed = impl->seed_;
        uint64_t counter = impl->call_counter_.fetch_add(1);
        return base_seed + counter * 1000000ULL;
    }

    uint64_t RandomGenerator::get_next_cuda_offset() {
        auto* impl = static_cast<RandomGeneratorImpl*>(impl_);
        uint64_t counter = impl->call_counter_.fetch_add(1);
        return counter * 1000000ULL;
    }

    void* RandomGenerator::get_generator(Device device) {
        auto* impl = static_cast<RandomGeneratorImpl*>(impl_);
        if (device == Device::CUDA) {
            return impl->cuda_generator_;
        } else {
            return &impl->cpu_generator_;
        }
    }

    // ============= In-place Random Operations =============

    Tensor& Tensor::uniform_(float low, float high) {
        if (dtype_ != DataType::Float32) {
            LOG_ERROR("uniform_ only implemented for float32");
            return *this;
        }

        if (!is_valid() || numel() == 0) {
            return *this;
        }

        size_t n = numel();

        if (device_ == Device::CUDA) {
            const cudaStream_t active_stream = resolveCUDAStream(stream_);
            CHECK_CUDA(waitForCUDAStream(active_stream, stream_));
            stream_ = active_stream;

            auto* impl = static_cast<RandomGeneratorImpl*>(
                RandomGenerator::instance().get_impl());
            curandGenerator_t* gen = static_cast<curandGenerator_t*>(impl->cuda_generator_);

            {
                std::lock_guard<std::mutex> lock(impl->cuda_generator_mutex_);
                uint64_t offset = impl->call_counter_.fetch_add(1, std::memory_order_relaxed) * 1000000ULL;
                CHECK_CURAND(curandSetGeneratorOffset(*gen, offset));
                CHECK_CURAND(curandSetStream(*gen, active_stream));
                CHECK_CURAND(curandGenerateUniform(*gen, ptr<float>(), n));
            }

            // Scale from (0,1] to [low,high)
            if (low != 0.0f || high != 1.0f) {
                const float range = high - low;
                tensor_ops::launch_scalar_op_generic(
                    ptr<float>(), range, ptr<float>(), n,
                    ops::mul_op{}, active_stream);
                if (low != 0.0f) {
                    tensor_ops::launch_scalar_op_generic(
                        ptr<float>(), low, ptr<float>(), n,
                        ops::add_op{}, active_stream);
                }
            }
        } else {
            // CPU uses stateful generator
            auto* impl = static_cast<RandomGeneratorImpl*>(
                RandomGenerator::instance().get_impl());
            std::uniform_real_distribution<float> dist(low, high);

            float* data = ptr<float>();
            for (size_t i = 0; i < n; ++i) {
                data[i] = dist(impl->cpu_generator_);
            }
        }

        return *this;
    }

    Tensor& Tensor::normal_(float mean, float std) {
        if (dtype_ != DataType::Float32) {
            LOG_ERROR("normal_ only implemented for float32");
            return *this;
        }

        if (!is_valid() || numel() == 0) {
            return *this;
        }

        size_t n = numel();

        if (device_ == Device::CUDA) {
            const cudaStream_t active_stream = resolveCUDAStream(stream_);
            CHECK_CUDA(waitForCUDAStream(active_stream, stream_));
            stream_ = active_stream;

            auto* impl = static_cast<RandomGeneratorImpl*>(
                RandomGenerator::instance().get_impl());
            curandGenerator_t* gen = static_cast<curandGenerator_t*>(impl->cuda_generator_);

            {
                std::lock_guard<std::mutex> lock(impl->cuda_generator_mutex_);
                uint64_t offset = impl->call_counter_.fetch_add(1, std::memory_order_relaxed) * 1000000ULL;
                CHECK_CURAND(curandSetGeneratorOffset(*gen, offset));
                CHECK_CURAND(curandSetStream(*gen, active_stream));

                if (n % 2 == 1) {
                    CHECK_CURAND(curandGenerateNormal(*gen, ptr<float>(), n + 1, mean, std));
                } else {
                    CHECK_CURAND(curandGenerateNormal(*gen, ptr<float>(), n, mean, std));
                }
            }
        } else {
            // CPU uses stateful generator
            auto* impl = static_cast<RandomGeneratorImpl*>(
                RandomGenerator::instance().get_impl());
            std::normal_distribution<float> dist(mean, std);

            float* data = ptr<float>();
            for (size_t i = 0; i < n; ++i) {
                data[i] = dist(impl->cpu_generator_);
            }
        }

        return *this;
    }

#undef CHECK_CUDA
#undef CHECK_CURAND

} // namespace lfs::core
