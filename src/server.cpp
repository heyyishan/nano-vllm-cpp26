#include "nano_vllm/server.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <signal.h>
#include <sys/socket.h>
#include <unistd.h>

#include <charconv>
#include <cstring>
#include <format>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>

namespace nano_vllm {

namespace {

bool send_all(int fd, std::string_view data) noexcept {
    std::size_t total_sent = 0;
    while (total_sent < data.size()) {
#ifdef MSG_NOSIGNAL
        const ssize_t n = ::send(
            fd, data.data() + total_sent, data.size() - total_sent, MSG_NOSIGNAL);
#else
        const ssize_t n = ::send(
            fd, data.data() + total_sent, data.size() - total_sent, 0);
#endif
        if (n <= 0) {
            return false;
        }
        total_sent += static_cast<std::size_t>(n);
    }
    return true;
}

std::string extract_json_string(std::string_view body, std::string_view key, std::string_view fallback) {
    const std::string pattern = std::format("\"{}\"", key);
    auto pos = body.find(pattern);
    if (pos == std::string_view::npos) {
        return std::string(fallback);
    }
    pos = body.find(':', pos + pattern.size());
    if (pos == std::string_view::npos) {
        return std::string(fallback);
    }
    pos = body.find('"', pos + 1);
    if (pos == std::string_view::npos) {
        return std::string(fallback);
    }
    const std::size_t end = body.find('"', pos + 1);
    if (end == std::string_view::npos) {
        return std::string(fallback);
    }
    return std::string(body.substr(pos + 1, end - (pos + 1)));
}

std::size_t extract_json_usize(std::string_view body, std::string_view key, std::size_t fallback) {
    const std::string pattern = std::format("\"{}\"", key);
    auto pos = body.find(pattern);
    if (pos == std::string_view::npos) {
        return fallback;
    }
    pos = body.find(':', pos + pattern.size());
    if (pos == std::string_view::npos) {
        return fallback;
    }
    ++pos;
    while (pos < body.size() && (body[pos] == ' ' || body[pos] == '\t')) {
        ++pos;
    }
    std::size_t value = fallback;
    std::from_chars(body.data() + pos, body.data() + body.size(), value);
    return value;
}

} // namespace

HttpSseServer::HttpSseServer(ContinuousBatchingEngine& engine, std::uint16_t port)
    : engine_(engine), port_(port) {
    ::signal(SIGPIPE, SIG_IGN);
    listen_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd_ < 0) {
        throw std::runtime_error("Failed to create TCP socket");
    }

    int opt = 1;
    ::setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(port_);

    if (::bind(listen_fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        ::close(listen_fd_);
        throw std::runtime_error(std::format("Failed to bind TCP port {}", port_));
    }

    if (::listen(listen_fd_, 128) < 0) {
        ::close(listen_fd_);
        throw std::runtime_error("Failed to listen on TCP socket");
    }
}

HttpSseServer::~HttpSseServer() {
    if (listen_fd_ >= 0) {
        ::close(listen_fd_);
    }
}

void HttpSseServer::serve_forever() {
    while (true) {
        sockaddr_in client_addr{};
        socklen_t client_len = sizeof(client_addr);
        const int client_fd = ::accept(
            listen_fd_, reinterpret_cast<sockaddr*>(&client_addr), &client_len);
        if (client_fd < 0) {
            continue;
        }
        std::thread([this, client_fd]() {
            handle_client(client_fd);
        }).detach();
    }
}

void HttpSseServer::handle_client(int client_fd) {
    char buffer[8192];
    const ssize_t bytes_read = ::recv(client_fd, buffer, sizeof(buffer) - 1, 0);
    if (bytes_read <= 0) {
        ::close(client_fd);
        return;
    }
    buffer[bytes_read] = '\0';
    const std::string_view req(buffer, static_cast<std::size_t>(bytes_read));

    if (req.starts_with("GET /health")) {
        static constexpr std::string_view kResp =
            "HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: 2\r\nConnection: close\r\n\r\nOK";
        send_all(client_fd, kResp);
        ::close(client_fd);
        return;
    }

    if (req.starts_with("GET /v1/metrics")) {
        const std::string json = engine_.snapshot_metrics().to_json();
        const std::string resp = std::format(
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: application/json\r\n"
            "Content-Length: {}\r\n"
            "Connection: close\r\n\r\n{}",
            json.size(),
            json);
        send_all(client_fd, resp);
        ::close(client_fd);
        return;
    }

    if (req.starts_with("POST /v1/generate")) {
        const auto body_pos = req.find("\r\n\r\n");
        const std::string_view body =
            (body_pos != std::string_view::npos) ? req.substr(body_pos + 4) : "";

        std::string prompt = extract_json_string(body, "prompt", "cpp26 continuous batching");
        std::size_t max_tokens = extract_json_usize(body, "max_tokens", 32);

        static constexpr std::string_view kSseHeader =
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: text/event-stream\r\n"
            "Cache-Control: no-cache\r\n"
            "Connection: keep-alive\r\n\r\n";

        if (!send_all(client_fd, kSseHeader)) {
            ::close(client_fd);
            return;
        }

        auto stream_queue = engine_.submit_request(std::move(prompt), max_tokens);
        while (auto ev_opt = stream_queue->pop_wait()) {
            const auto& ev = *ev_opt;
            std::string frame;
            if (ev.ttft_us.has_value()) {
                frame = std::format(
                    "data: {{\"seq_id\":{},\"token_id\":{},\"text\":\"{}\",\"finished\":{},\"ttft_us\":{}}}\n\n",
                    ev.seq_id,
                    ev.token_id,
                    ev.text,
                    ev.is_finished ? "true" : "false",
                    *ev.ttft_us);
            } else {
                frame = std::format(
                    "data: {{\"seq_id\":{},\"token_id\":{},\"text\":\"{}\",\"finished\":{}}}\n\n",
                    ev.seq_id,
                    ev.token_id,
                    ev.text,
                    ev.is_finished ? "true" : "false");
            }

            if (!send_all(client_fd, frame)) {
                stream_queue->close();
                break;
            }
        }

        ::close(client_fd);
        return;
    }

    static constexpr std::string_view kNotFound =
        "HTTP/1.1 404 Not Found\r\nContent-Length: 9\r\nConnection: close\r\n\r\nNot Found";
    send_all(client_fd, kNotFound);
    ::close(client_fd);
}

} // namespace nano_vllm
