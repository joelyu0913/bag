#include "yang/sim/scheduler.h"

#include <atomic>
#include <functional>
#include <future>
#include <queue>
#include <stdexcept>
#include <thread_pool.hpp>

#include "yang/base/exception.h"
#include "yang/util/logging.h"

namespace yang {

void Scheduler::Run(int num_threads, const unordered_map<Id, std::vector<Id>> &dep_map,
                    const std::vector<std::pair<Id, Func>> &tasks) {
  if (tasks.empty()) return;

  // Topological sort - Kahn's algorithm
  // compute auxiliary info
  std::vector<std::atomic<int>> in_degrees(tasks.size());
  std::vector<std::vector<int>> out_edges(tasks.size());
  {
    unordered_map<Id, int> id_map;
    for (int i = 0; i < static_cast<int>(tasks.size()); ++i) {
      ENSURE(!id_map.count(tasks[i].first), "Duplicate task id: {}", tasks[i].first);
      id_map[tasks[i].first] = i;
    }
    for (auto &deps : dep_map) {
      auto &consumer = deps.first;
      auto consumer_it = id_map.find(consumer);
      if (consumer_it == id_map.end()) continue;

      int consumer_id = consumer_it->second;
      for (auto &producer : deps.second) {
        auto producer_it = id_map.find(producer);
        if (producer_it == id_map.end()) continue;
        int producer_id = producer_it->second;
        in_degrees[consumer_id]++;
        out_edges[producer_id].push_back(consumer_id);
      }
    }
  }

  thread_pool worker(num_threads);

  std::atomic<int> status{0};
  std::atomic<int> active_counter{0};
  std::promise<void> run_promise;
  std::function<void(int)> run_task = [&](int task_id) {
    active_counter.fetch_add(1, std::memory_order_relaxed);

    worker.push_task([&, task_id]() {
      if (status.load(std::memory_order_relaxed) == 0) {
        try {
          tasks[task_id].second();
        } catch (const std::exception &e) {
          LOG_ERROR("{} failed: {}", tasks[task_id].first, e.what());
          status.store(-1, std::memory_order_relaxed);
        } catch (...) {
          LOG_ERROR("{} failed", tasks[task_id].first);
          status.store(-1, std::memory_order_relaxed);
        }

        for (auto &downstream_id : out_edges[task_id]) {
          int d = in_degrees[downstream_id].fetch_add(-1, std::memory_order_relaxed) - 1;
          if (d == 0) run_task(downstream_id);
        }
      }

      int n = active_counter.fetch_add(-1, std::memory_order_relaxed) - 1;
      if (n == 0) {
        run_promise.set_value();
      }
    });
  };

  std::vector<int> seed_tasks;
  for (int i = 0; i < static_cast<int>(tasks.size()); ++i) {
    if (in_degrees[i].load(std::memory_order_relaxed) == 0) seed_tasks.push_back(i);
  }
  for (auto &task_id : seed_tasks) run_task(task_id);

  auto run_future = run_promise.get_future();
  run_future.get();
  if (status < 0) throw FatalError("Task failure");

  bool cycle = false;
  for (int i = 0; i < static_cast<int>(tasks.size()); ++i) {
    if (in_degrees[i].load(std::memory_order_relaxed) > 0) {
      LOG_ERROR("Found dependency cycle containing {}", tasks[i].first);
      cycle = true;
    }
  }
  if (cycle) throw FatalError("Found dependency cycle");
}

}  // namespace yang
