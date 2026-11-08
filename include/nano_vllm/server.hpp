#pragma once

#include "nano_vllm/engine.hpp"

#include <cstdint>
#include <string_view>

namespace nano_vllm {

/// Zero-dependency POSIX HTTP/1.1 Server with Server-Sent Events (SSE) token streaming.
class HttpSseServer {
public:
    HttpSseServer(ContinuousBatchingEngine& engine, std::uint16_t port);
    ~HttpSseServer();

    HttpSseServer(const HttpSseServer&) = delete;
    HttpSseServer& operator=(const HttpSseServer&) = delete;

    /// Starts accepting connections and spawning detached client handler threads.
    void serve_forever();

private:
    void handle_client(int client_fd);

    ContinuousBatchingEngine& engine_;
    std::uint16_t port_;
    int listen_fd_{-1};
};

} // namespace nano_vllm
