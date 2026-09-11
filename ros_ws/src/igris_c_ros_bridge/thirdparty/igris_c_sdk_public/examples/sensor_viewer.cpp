#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <dds/dds.hpp>
#include <igris_sdk/igris_c_msgs.hpp>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

struct status {
    std::string name;
    float fps           = 0.0f;
    float bytes_per_sec = 0.0f;
};

namespace {
int domain_id = 10;

using CompressedMessage = igris_c::msg::dds::CompressedMessage;

std::atomic<bool> g_stop{false};

void handle_signal(int) { g_stop.store(true); }

int getenv_int(const char *name, int fallback) {
    const char *value = std::getenv(name);
    if (!value || value[0] == '\0') {
        return fallback;
    }
    char *end   = nullptr;
    long parsed = std::strtol(value, &end, 10);
    if (end == value) {
        return fallback;
    }
    return static_cast<int>(parsed);
}

struct StreamStats {
    std::chrono::steady_clock::time_point last_time = std::chrono::steady_clock::now();
    double fps                                      = 0.0;
    double bytes_per_sec                            = 0.0;
};

void update_stats(StreamStats &stats, size_t bytes) {
    auto now     = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(now - stats.last_time).count();
    if (elapsed > 0) {
        double seconds  = static_cast<double>(elapsed) / 1'000'000.0;
        double inst_fps = 1.0 / seconds;
        double inst_bps = static_cast<double>(bytes) / seconds;
        double alpha    = 0.15;
        if (stats.fps <= 0.0) {
            stats.fps           = inst_fps;
            stats.bytes_per_sec = inst_bps;
        } else {
            stats.fps           = (1.0 - alpha) * stats.fps + alpha * inst_fps;
            stats.bytes_per_sec = (1.0 - alpha) * stats.bytes_per_sec + alpha * inst_bps;
        }
    }
    stats.last_time = now;
}

std::string format_rate(double bytes_per_sec) {
    std::ostringstream oss;
    if (bytes_per_sec > 1024.0 * 1024.0) {
        oss << std::fixed << std::setprecision(1) << (bytes_per_sec / (1024.0 * 1024.0)) << " MB/s";
    } else if (bytes_per_sec > 1024.0) {
        oss << std::fixed << std::setprecision(1) << (bytes_per_sec / 1024.0) << " KB/s";
    } else {
        oss << std::fixed << std::setprecision(0) << bytes_per_sec << " B/s";
    }
    return oss.str();
}

struct Stream {
    std::string name;
    dds::topic::Topic<CompressedMessage> topic;
    dds::sub::DataReader<CompressedMessage> reader;
    StreamStats stats;

    Stream(dds::domain::DomainParticipant &participant, dds::sub::Subscriber &subscriber, const std::string &topic_name)
        : name(topic_name), topic(participant, topic_name), reader(subscriber, topic) {}
};
}  // namespace

int main(int argc, char **argv) {
    std::signal(SIGINT, handle_signal);
    std::signal(SIGTERM, handle_signal);

    if (argc > 1) {
        domain_id = std::atoi(argv[1]);
    } else {
        domain_id = 10;  // Default domain ID
    }

    std::vector<std::string> topics;
    for (int i = 1; i < argc; ++i) {
        topics.emplace_back(argv[i]);
    }
    if (topics.empty()) {
        topics = {
            "igris_c/sensor/d435_color", "igris_c/sensor/d435_depth", "igris_c/sensor/eyes_stereo",
            "igris_c/sensor/left_hand",  "igris_c/sensor/right_hand",
        };
    }

    dds::domain::DomainParticipant participant(domain_id);
    dds::sub::Subscriber subscriber(participant);

    std::vector<Stream> streams;
    streams.reserve(topics.size());
    for (const auto &name : topics) {
        streams.emplace_back(participant, subscriber, name);
        cv::namedWindow(name, cv::WINDOW_NORMAL);
    }

    std::vector<status> stream_status(streams.size());
    for (size_t i = 0; i < streams.size(); ++i) {
        stream_status[i].name = streams[i].name;
    }

    while (!g_stop.load()) {
        bool got_data = false;

        for (size_t i = 0; i < streams.size(); ++i) {
            auto &stream = streams[i];
            auto samples = stream.reader.take();
            for (const auto &sample : samples) {
                if (!sample.info().valid()) {
                    continue;
                }
                got_data = true;

                const auto &msg   = sample.data();
                const auto &bytes = msg.image_data();
                if (bytes.empty()) {
                    continue;
                }

                update_stats(stream.stats, bytes.size());

                cv::Mat raw(1, static_cast<int>(bytes.size()), CV_8UC1, const_cast<uint8_t *>(bytes.data()));
                cv::Mat img = cv::imdecode(raw, cv::IMREAD_UNCHANGED);
                if (img.empty()) {
                    continue;
                }

                std::ostringstream oss;
                oss << std::fixed << std::setprecision(1) << stream.stats.fps << " FPS, " << format_rate(stream.stats.bytes_per_sec);
                cv::Scalar text_color = (stream.stats.fps <= 20.0 ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0));
                cv::putText(img, oss.str(), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2);
                stream_status[i].fps           = static_cast<float>(stream.stats.fps);
                stream_status[i].bytes_per_sec = static_cast<float>(stream.stats.bytes_per_sec);
                cv::imshow(stream.name, img);
            }
        }

        std::cout << "\x1B[2J\x1B[H";
        std::cout << "==============================\n\n";
        std::cout << "Stream Status:\n";
        for (const auto &s : stream_status) {
            bool low_fps            = (s.fps <= 20.0f);
            const char *color_start = low_fps ? "\x1B[31m" : "\x1B[0m";
            const char *color_end   = "\x1B[0m";
            std::cout << color_start << s.name << ": " << std::fixed << std::setprecision(1) << s.fps << " FPS, "
                      << format_rate(s.bytes_per_sec) << color_end << "\n";
        }
        std::cout << std::endl;
        std::cout << "Total bytes/sec: "
                  << format_rate(std::accumulate(stream_status.begin(), stream_status.end(), 0.0f,
                                                 [](float sum, const status &s) { return sum + s.bytes_per_sec; }))
                  << "\n";
        std::cout << "Avg FPS: " << std::fixed << std::setprecision(1)
                  << (stream_status.empty() ? 0.0f
                                            : std::accumulate(stream_status.begin(), stream_status.end(), 0.0f,
                                                              [](float sum, const status &s) { return sum + s.fps; }) /
                                                  stream_status.size())
                  << " FPS\n";
        std::cout << "Press 'q' or 'ESC' to quit.\n";
        std::cout << "==============================\n";
        std::cout << std::flush;

        int key = cv::waitKey(1);
        if (key == 27 || key == 'q' || key == 'Q') {
            break;
        }

        if (!got_data) {
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
    }

    for (const auto &stream : streams) {
        cv::destroyWindow(stream.name);
    }
    return 0;
}
