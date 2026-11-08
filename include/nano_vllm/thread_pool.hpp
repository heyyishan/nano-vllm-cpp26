#pragma once

#include <atomic>
#include <concepts>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace nano_vllm {

/// Lightweight data-parallel execution pool backed by C++20/26 std::jthread workers.
class ThreadPool {
public:
    explicit ThreadPool(std::size_t num_threads = std::thread::hardware_concurrency()) {
        if (num_threads == 0) {
            num_threads = 4;
        }
        num_threads_ = num_threads;
        workers_.reserve(num_threads_);
        for (std::size_t tid = 0; tid < num_threads_; ++tid) {
            workers_.emplace_back([this, tid](std::stop_token stoken) {
                worker_loop(tid, stoken);
            });
        }
    }

    ~ThreadPool() {
        {
            std::scoped_lock lock(mutex_);
            stopping_ = true;
        }
        cv_start_.notify_all();
    }

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    /// Executes `fn(idx)` for `idx` in `[0, total_items)` across worker threads.
    template <std::invocable<std::size_t> Func>
    void parallel_for(std::size_t total_items, Func&& fn) {
        if (total_items == 0) {
            return;
        }
        if (total_items == 1 || num_threads_ <= 1) {
            for (std::size_t i = 0; i < total_items; ++i) {
                fn(i);
            }
            return;
        }

        {
            std::scoped_lock lock(mutex_);
            current_task_ = [&fn](std::size_t idx) { fn(idx); };
            total_items_.store(total_items, std::memory_order::relaxed);
            next_index_.store(0, std::memory_order::relaxed);
            active_workers_.store(num_threads_, std::memory_order::relaxed);
            ++generation_;
        }
        cv_start_.notify_all();

        // Main thread also participates in work stealing
        run_work_stealing_step();

        std::unique_lock lock(mutex_);
        cv_done_.wait(lock, [this] {
            return active_workers_.load(std::memory_order::acquire) == 0;
        });
        current_task_ = nullptr;
    }

private:
    void run_work_stealing_step() {
        const std::size_t total = total_items_.load(std::memory_order::relaxed);
        while (true) {
            const std::size_t idx = next_index_.fetch_add(1, std::memory_order::relaxed);
            if (idx >= total) {
                break;
            }
            current_task_(idx);
        }
    }

    void worker_loop(std::size_t /*tid*/, std::stop_token stoken) {
        std::uint64_t seen_generation = 0;
        while (!stoken.stop_requested()) {
            {
                std::unique_lock lock(mutex_);
                cv_start_.wait(lock, [this, &seen_generation, &stoken] {
                    return stopping_ || stoken.stop_requested() || generation_ != seen_generation;
                });
                if (stopping_ || stoken.stop_requested()) {
                    return;
                }
                seen_generation = generation_;
            }

            run_work_stealing_step();

            if (active_workers_.fetch_sub(1, std::memory_order::acq_rel) == 1) {
                std::scoped_lock lock(mutex_);
                cv_done_.notify_one();
            }
        }
    }

    std::size_t num_threads_{1};
    std::vector<std::jthread> workers_;
    std::mutex mutex_;
    std::condition_variable cv_start_;
    std::condition_variable cv_done_;
    std::function<void(std::size_t)> current_task_;
    std::atomic<std::size_t> total_items_{0};
    std::atomic<std::size_t> next_index_{0};
    std::atomic<std::size_t> active_workers_{0};
    std::uint64_t generation_{0};
    bool stopping_{false};
};

} // namespace nano_vllm
