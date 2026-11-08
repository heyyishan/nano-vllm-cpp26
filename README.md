# `nano-vllm-cpp26`: High-Throughput Continuous Batching & Paged KV Cache Inference Server in Modern C++26

`nano-vllm-cpp26` is a zero-dependency, high-performance LLM inference engine written in **Modern C++26** (`-std=c++26` / `-std=c++2c`), inspired by **Orca** (OSDI '22) and **vLLM** (SOSP '23). It is engineered around a single objective: **maximizing token generation throughput and minimizing Time-To-First-Token (TTFT) under concurrent, skewed request loads.**

---

## System Architecture

```text
                        Incoming HTTP POST /v1/generate
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                  Zero-Dependency POSIX HTTP/1.1 + SSE Server                 │
│     TokenStreamQueue (CondVar) ◄────────── Per-Request Stream Handle         │
└──────────────────────────────────────────────┬───────────────────────────────┘
                                               │ Ingress Queue + Notify
                                               ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                 Continuous Batching Scheduler (Iteration-Level)              │
│                                                                              │
│  ┌─────────────────────┐    Dynamic Admission     ┌───────────────────────┐  │
│  │   Waiting Deque     │ ───────────────────────► │  Running Batch Vec    │  │
│  │ (New & Preempted)   │ ◄─────────────────────── │ (Mixed Prefill+Decode)│  │
│  └─────────────────────┘   KV OOM Preemption      └───────────┬───────────┘  │
└───────────────────────────────────────────────────────────────┼──────────────┘
                                                                │
                   ┌────────────────────────────────────────────┴───────────┐
                   ▼                                                        ▼
┌─────────────────────────────────────────┐      ┌─────────────────────────────┐
│   Selective Token Flattening Compute    │      │ Paged KV Cache BlockManager │
│                                         │      │                             │
│ 1. Flatten Prefill + Decode tokens      │      │ Logical BlockTable:         │
│    into super-batch [N_tokens, D]       │      │  Seq A: [Blk 7, Blk 2]      │
│ 2. Shared-Weight Batched GEMM (Q,K,V)   │◄────►│  Seq B: [Blk 0, Blk 9, ...] │
│ 3. FlashAttention Online Softmax over   │      │                             │
│    non-contiguous Physical KV Blocks    │      │ Contiguous Arena:           │
│ 4. Batched SwiGLU FFN + LM Head         │      │ [num_blocks, layers, B, D]  │
└─────────────────────────────────────────┘      └─────────────────────────────┘
```

---

## Modern C++26 Features Used

- **`std::expected<T, KVError>`**: Zero-overhead error handling for physical KV block allocations (`allocate_block()`, `allocate_blocks()`).
- **`std::span<const float>` / `std::span<float>`**: Bounds-safe, zero-copy tensor slice views into the contiguous Paged KV Cache arena.
- **`std::jthread` & `std::stop_token`**: Cooperative, RAII-managed background continuous-batching worker loop and work-stealing compute thread pool (`ThreadPool`).
- **`std::format`**: Fast, type-safe SSE JSON frame serialization and benchmark table formatting.
- **Lock-Free `std::atomic` Telemetry**: Real-time tracking of TTFT, token throughput, batch slot utilization, and KV cache pressure with `std::memory_order::relaxed` / `acq_rel`.

---

## How to Build, Test, and Run

### Requirements
- CMake `>= 3.25`
- GCC `14+` or Clang `18+` with C++26 (`-std=c++26` / `-std=c++2c`) support

### 1. Configure & Compile (Release Mode)
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

### 2. Run Unit & Concurrent Integration Tests
```bash
ctest --test-dir build --output-on-failure
```

### 3. Run Continuous vs. Static Batching Benchmark
```bash
./build/nano_vllm_benchmark
```

Sample Output (96 concurrent requests: 80% 16-token completions, 20% 128-token completions, `max_batch_size = 16`):
```text
====================================================================================================
  nano-vllm-cpp26 Benchmark: Continuous Batching vs. Static Batching
  Workload: 96 concurrent requests (80% short 16-tok, 20% long 128-tok) | Max Batch = 16
====================================================================================================

Scheduling Strategy          |    Wall Time |     Throughput |   Slot Util |    p50 TTFT |     p50 E2E
----------------------------------------------------------------------------------------------------
Static Batching (Baseline)   |     482.6 ms |     7732.3 t/s |      30.3 % |    241.0 ms |    246.2 ms
Continuous Batching (vLLM)   |     149.8 ms |    24913.2 t/s |      94.8 % |     24.8 ms |     62.1 ms
----------------------------------------------------------------------------------------------------
Summary: Continuous Batching achieved 3.22x higher token throughput and 9.72x lower p50 TTFT!
```

### 4. Start the HTTP/SSE Inference Server
```bash
./build/nano_vllm_server
```

### 5. Stream Tokens via SSE (`curl`)
```bash
curl -N -X POST http://localhost:8000/v1/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "cpp26 continuous batching paged kv cache", "max_tokens": 16}'
```

### 6. Query Real-Time Metrics
```bash
curl -s http://localhost:8000/v1/metrics
```
