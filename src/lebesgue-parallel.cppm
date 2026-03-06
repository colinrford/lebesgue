/*
 *  lebesgue-parallel.cppm
 *  see github.com/colinrford/lebesgue for GPL 3.0 license and for more info
 *
 *  Parallel execution primitives (wrapping TBB or custom thread pool).
 */

module;
#ifdef LEB_HAS_TBB
#include <tbb/blocked_range.h>
#include <tbb/parallel_reduce.h>
#endif

export module lam.lebesgue:parallel;

import :config;

import std;

export namespace lam::leb::parallel
{

// Ensure config matches preprocessor state
#ifdef LEB_HAS_TBB
constexpr bool tbb_compiled = true;
#else
constexpr bool tbb_compiled = false;
#endif
static_assert(lam::lebesgue::config::use_tbb == tbb_compiled,
              "lam::lebesgue::config::use_tbb mismatch with LEB_HAS_TBB");

inline unsigned thread_count() noexcept
{
  unsigned n = std::thread::hardware_concurrency();
  return n > 0 ? n : 4;
}


class thread_pool
{
  std::vector<std::jthread> workers;
  std::deque<std::function<void()>> tasks;
  std::mutex queue_mutex;
  std::condition_variable condition;
  bool stop = false;

public:
  explicit thread_pool(std::size_t num_threads)
  {
    for (std::size_t i = 0; i < num_threads; ++i)
    {
      workers.emplace_back([this](std::stop_token st) {
        while (true)
        {
          std::function<void()> task;
          {
            std::unique_lock<std::mutex> lock(queue_mutex);
            condition.wait(lock, [this, &st] { return stop || st.stop_requested() || !tasks.empty(); });

            if ((stop || st.stop_requested()) && tasks.empty())
              return;

            if (tasks.empty())
              continue;
            task = std::move(tasks.front());
            tasks.pop_front();
          }
          task();
        }
      });
    }
  }

  ~thread_pool()
  {
    {
      std::unique_lock<std::mutex> lock(queue_mutex);
      stop = true;
    }
    condition.notify_all();
  }

  template<class F>
  void enqueue(F&& f)
  {
    {
      std::unique_lock<std::mutex> lock(queue_mutex);
      tasks.emplace_back(std::forward<F>(f));
    }
    condition.notify_one();
  }

  static thread_pool& instance()
  {
    static thread_pool pool(thread_count());
    return pool;
  }
};

/**
 * Parallel transform_reduce.
 * Range [0, n).
 * Identity value 'init'.
 * Transform function 'transform_op(size_t i) -> T'.
 * Reduce function 'reduce_op(T a, T b) -> T'.
 */
template<typename T, typename TransformOp, typename ReduceOp>
T transform_reduce(std::size_t n, T init, TransformOp&& transform_op, ReduceOp&& reduce_op)
{
  if constexpr (lam::lebesgue::config::use_tbb)
  {
#ifdef LEB_HAS_TBB
    return tbb::parallel_reduce(
      tbb::blocked_range<std::size_t>(0, n), init,
      [&](const tbb::blocked_range<std::size_t>& r, T local_init) -> T {
        for (std::size_t i = r.begin(); i != r.end(); ++i)
        {
          local_init = reduce_op(local_init, transform_op(i));
        }
        return local_init;
      },
      std::forward<ReduceOp>(reduce_op));
#else
    // This branch should be unreachable due to static_assert checking usage == capability
    return init; // Dummy return to satisfy compiler if it doesn't see strict unreachability
#endif
  }
  else
  {
    // Fallback to thread pool
    auto& pool = thread_pool::instance();
    unsigned n_threads = thread_count();

    if (n < 1000)
      n_threads = 1; // Sequential for small N

    if (n_threads == 1)
    {
      T val = init;
      for (std::size_t i = 0; i < n; ++i)
        val = reduce_op(val, transform_op(i));
      return val;
    }

    std::size_t chunk_size = (n + n_threads - 1) / n_threads;
    std::vector<T> partials(n_threads, init);
    std::latch done(n_threads);

    for (unsigned t = 0; t < n_threads; ++t)
    {
      pool.enqueue([&, t, n_threads, chunk_size, n, &latch = done] {
        std::size_t start = t * chunk_size;
        std::size_t end = std::min(start + chunk_size, n);
        T local_acc = init; // Assuming init is identity for reduce_op

        // Note: reduce_op usually commutative/associative.
        // If init is NOT identity, we must be careful.
        // Standard transform_reduce assumes init is applied once at end or involves monoid.
        // Usually parallel_reduce (TBB) style assumes we can combine partials.
        // We'll accumulate into partials[t] starting from init.

        for (std::size_t i = start; i < end; ++i)
        {
          local_acc = reduce_op(local_acc, transform_op(i));
        }
        partials[t] = local_acc;
        latch.count_down();
      });
    }
    done.wait();

    T final_res = init;
    for (const auto& p : partials)
      final_res = reduce_op(final_res, p);

    return final_res;
  }
}

} // namespace lam::leb::parallel
