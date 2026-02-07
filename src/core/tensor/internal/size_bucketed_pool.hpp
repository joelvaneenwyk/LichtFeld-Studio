/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/logger.hpp"
#include "cuda_stream_context.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cuda_runtime.h>
#include <mutex>
#include <vector>

namespace lfs::core {

    // Size-bucketed memory pool. Rounds allocations to bucket boundaries and caches
    // freed memory per bucket to maximize reuse and reduce fragmentation.
    class SizeBucketedPool {
    public:
        static constexpr size_t MIN_BUCKET_SIZE = 256 * 1024;
        static constexpr size_t MAX_TRACKED_SIZE = 16ULL * 1024 * 1024 * 1024;
        static constexpr size_t CACHE_SIZE_PER_BUCKET = 4;
        static constexpr size_t NUM_BUCKETS = 128;

        struct Stats {
            std::atomic<uint64_t> cache_hits{0};
            std::atomic<uint64_t> cache_misses{0};
            std::atomic<uint64_t> alloc_count{0};
            std::atomic<uint64_t> free_count{0};
            std::atomic<uint64_t> bytes_cached{0};
            std::atomic<uint64_t> bytes_wasted{0};
        };

        static SizeBucketedPool& instance() {
            static SizeBucketedPool pool;
            return pool;
        }

        void shutdown() {
            bool expected = false;
            if (!shutdown_.compare_exchange_strong(expected, true))
                return;
            trim_cache();
        }

        static size_t get_bucket_size(size_t bytes) {
            if (bytes <= MIN_BUCKET_SIZE)
                return MIN_BUCKET_SIZE;
            if (bytes <= 1024 * 1024)
                return ((bytes + 256 * 1024 - 1) / (256 * 1024)) * (256 * 1024);
            if (bytes <= 16 * 1024 * 1024)
                return ((bytes + 1024 * 1024 - 1) / (1024 * 1024)) * (1024 * 1024);
            if (bytes <= 256 * 1024 * 1024)
                return ((bytes + 16 * 1024 * 1024 - 1) / (16 * 1024 * 1024)) * (16 * 1024 * 1024);
            if (bytes <= 1024ULL * 1024 * 1024)
                return ((bytes + 64 * 1024 * 1024 - 1) / (64 * 1024 * 1024)) * (64 * 1024 * 1024);
            if (bytes <= 8ULL * 1024 * 1024 * 1024)
                return ((bytes + 256ULL * 1024 * 1024 - 1) / (256ULL * 1024 * 1024)) * (256ULL * 1024 * 1024);
            return ((bytes + 1024ULL * 1024 * 1024 - 1) / (1024ULL * 1024 * 1024)) * (1024ULL * 1024 * 1024);
        }

        static size_t get_bucket_index(size_t bucket_size) {
            if (bucket_size <= 1024 * 1024)
                return (bucket_size / (256 * 1024)) - 1;
            if (bucket_size <= 16 * 1024 * 1024)
                return 4 + (bucket_size / (1024 * 1024)) - 1;
            if (bucket_size <= 256 * 1024 * 1024)
                return 20 + (bucket_size / (16 * 1024 * 1024)) - 1;
            if (bucket_size <= 1024ULL * 1024 * 1024)
                return 36 + (bucket_size / (64 * 1024 * 1024)) - 4;
            if (bucket_size <= 8ULL * 1024 * 1024 * 1024)
                return 48 + (bucket_size / (256ULL * 1024 * 1024)) - 4;
            const size_t idx = 76 + (bucket_size / (1024ULL * 1024 * 1024)) - 8;
            return std::min(idx, NUM_BUCKETS - 1);
        }

        void* try_allocate_cached(size_t bytes, cudaStream_t consumer_stream = nullptr) {
            const size_t bucket_size = get_bucket_size(bytes);
            const size_t bucket_idx = get_bucket_index(bucket_size);
            if (bucket_idx >= NUM_BUCKETS)
                return nullptr;

            CachedBlock block{};
            {
                std::lock_guard<std::mutex> lock(buckets_[bucket_idx].mutex);
                if (!buckets_[bucket_idx].cache.empty()) {
                    block = buckets_[bucket_idx].cache.back();
                    buckets_[bucket_idx].cache.pop_back();
                    buckets_[bucket_idx].cached_bytes -= bucket_size;
                    stats_.cache_hits.fetch_add(1, std::memory_order_relaxed);
                    stats_.bytes_cached.fetch_sub(bucket_size, std::memory_order_relaxed);
                    stats_.bytes_wasted.fetch_add(bucket_size - bytes, std::memory_order_relaxed);
                } else {
                    stats_.cache_misses.fetch_add(1, std::memory_order_relaxed);
                    return nullptr;
                }
            }
            wait_for_block(block, consumer_stream);
            return block.ptr;
        }

        bool cache_free(void* ptr, size_t bytes, cudaStream_t producer_stream = nullptr) {
            const size_t bucket_size = get_bucket_size(bytes);
            const size_t bucket_idx = get_bucket_index(bucket_size);
            if (bucket_idx >= NUM_BUCKETS)
                return false;

            cudaEvent_t ready_event = nullptr;
            cudaError_t err = cudaEventCreateWithFlags(&ready_event, cudaEventDisableTiming);
            if (err != cudaSuccess) {
                LOG_WARN("SizeBucketedPool: cudaEventCreateWithFlags failed, bypassing cache: {}",
                         cudaGetErrorString(err));
                cudaFree(ptr);
                return true;
            }

            err = cudaEventRecord(ready_event, producer_stream);
            if (err != cudaSuccess) {
                LOG_WARN("SizeBucketedPool: cudaEventRecord failed, bypassing cache: {}",
                         cudaGetErrorString(err));
                cudaEventDestroy(ready_event);
                cudaFree(ptr);
                return true;
            }

            CachedBlock evicted{};
            bool has_evicted = false;
            {
                std::lock_guard<std::mutex> lock(buckets_[bucket_idx].mutex);
                if (buckets_[bucket_idx].cache.size() >= CACHE_SIZE_PER_BUCKET) {
                    evicted = buckets_[bucket_idx].cache.front();
                    buckets_[bucket_idx].cache.erase(buckets_[bucket_idx].cache.begin());
                    buckets_[bucket_idx].cached_bytes -= bucket_size;
                    stats_.bytes_cached.fetch_sub(bucket_size, std::memory_order_relaxed);
                    has_evicted = true;
                }
                buckets_[bucket_idx].cache.push_back({ptr, ready_event});
                buckets_[bucket_idx].cached_bytes += bucket_size;
                stats_.free_count.fetch_add(1, std::memory_order_relaxed);
                stats_.bytes_cached.fetch_add(bucket_size, std::memory_order_relaxed);
            }

            if (has_evicted) {
                release_block(evicted);
            }
            return true;
        }

        void* allocate(size_t bytes, cudaStream_t stream = nullptr) {
            void* ptr = try_allocate_cached(bytes, stream);
            if (ptr)
                return ptr;

            const size_t bucket_size = get_bucket_size(bytes);
            cudaError_t err = cudaMallocAsync(&ptr, bucket_size, stream);
            if (err != cudaSuccess) {
                trim_cache();
                err = cudaMallocAsync(&ptr, bucket_size, stream);
                if (err != cudaSuccess) {
                    LOG_ERROR("cudaMallocAsync failed for {} bytes: {}", bucket_size, cudaGetErrorString(err));
                    cudaGetLastError(); // Clear sticky error state for clean recovery
                    return nullptr;
                }
            }
            stats_.alloc_count.fetch_add(1, std::memory_order_relaxed);
            stats_.bytes_wasted.fetch_add(bucket_size - bytes, std::memory_order_relaxed);
            return ptr;
        }

        void deallocate(void* ptr, size_t bytes, cudaStream_t stream = nullptr) {
            if (!ptr)
                return;
            if (!cache_free(ptr, bytes, stream)) {
                cudaFreeAsync(ptr, stream);
            }
        }

        void trim_cache() {
            for (size_t i = 0; i < NUM_BUCKETS; ++i) {
                std::vector<CachedBlock> blocks_to_release;
                {
                    std::lock_guard<std::mutex> lock(buckets_[i].mutex);
                    blocks_to_release.swap(buckets_[i].cache);
                    buckets_[i].cached_bytes = 0;
                }
                for (auto& block : blocks_to_release) {
                    release_block(block);
                }
            }
            stats_.bytes_cached.store(0, std::memory_order_relaxed);
        }

        const Stats& stats() const { return stats_; }

        void print_stats() const {
            uint64_t hits = stats_.cache_hits.load();
            uint64_t misses = stats_.cache_misses.load();
            double hit_rate = (hits + misses > 0) ? (100.0 * hits / (hits + misses)) : 0.0;

            LOG_INFO("SizeBucketedPool Statistics:");
            LOG_INFO("  Cache hits: {} ({:.1f}%)", hits, hit_rate);
            LOG_INFO("  Cache misses: {}", misses);
            LOG_INFO("  Bytes cached: {:.2f} MB", stats_.bytes_cached.load() / (1024.0 * 1024.0));
            LOG_INFO("  Bytes wasted (rounding): {:.2f} MB", stats_.bytes_wasted.load() / (1024.0 * 1024.0));
        }

        // Calculate waste percentage for a given size
        static double get_waste_percentage(size_t bytes) {
            size_t bucket = get_bucket_size(bytes);
            return 100.0 * (bucket - bytes) / bucket;
        }

        SizeBucketedPool(const SizeBucketedPool&) = delete;
        SizeBucketedPool& operator=(const SizeBucketedPool&) = delete;

    private:
        struct CachedBlock {
            void* ptr{nullptr};
            cudaEvent_t ready_event{nullptr};
        };

        struct Bucket {
            std::vector<CachedBlock> cache;
            std::mutex mutex;
            size_t cached_bytes{0};

            Bucket() {
                cache.reserve(CACHE_SIZE_PER_BUCKET);
            }
        };

        SizeBucketedPool() = default;

        static void wait_for_block(CachedBlock& block, cudaStream_t consumer_stream) {
            if (!block.ready_event)
                return;

            cudaError_t err = cudaStreamWaitEvent(consumer_stream, block.ready_event, 0);
            if (err != cudaSuccess) {
                LOG_WARN("SizeBucketedPool: cudaStreamWaitEvent failed, synchronizing event: {}",
                         cudaGetErrorString(err));
                err = cudaEventSynchronize(block.ready_event);
                if (err != cudaSuccess) {
                    LOG_WARN("SizeBucketedPool: cudaEventSynchronize failed: {}",
                             cudaGetErrorString(err));
                }
            }

            err = cudaEventDestroy(block.ready_event);
            if (err != cudaSuccess) {
                LOG_WARN("SizeBucketedPool: cudaEventDestroy failed: {}",
                         cudaGetErrorString(err));
            }
            block.ready_event = nullptr;
        }

        static void release_block(CachedBlock& block) {
            if (!block.ptr)
                return;

            if (block.ready_event) {
                cudaError_t err = cudaEventSynchronize(block.ready_event);
                if (err != cudaSuccess) {
                    LOG_WARN("SizeBucketedPool: cudaEventSynchronize failed during eviction: {}",
                             cudaGetErrorString(err));
                }
                err = cudaEventDestroy(block.ready_event);
                if (err != cudaSuccess) {
                    LOG_WARN("SizeBucketedPool: cudaEventDestroy failed during eviction: {}",
                             cudaGetErrorString(err));
                }
                block.ready_event = nullptr;
            }

            cudaError_t err = cudaFree(block.ptr);
            if (err != cudaSuccess) {
                LOG_WARN("SizeBucketedPool: cudaFree failed during eviction: {}",
                         cudaGetErrorString(err));
            }
            block.ptr = nullptr;
        }

        ~SizeBucketedPool() {
            shutdown();
        }

        std::array<Bucket, NUM_BUCKETS> buckets_;
        std::atomic<bool> shutdown_{false};
        Stats stats_;
    };

} // namespace lfs::core
