#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>

#include <unistd.h>

#include "igris_c_sdk/msg/low_cmd.hpp"
#include "rclcpp/rclcpp.hpp"

namespace igris_lowcmd_relay
{

class LowCmdRelay final : public rclcpp::Node
{
public:
  using LowCmd = igris_c_sdk::msg::LowCmd;
  using SteadyClock = std::chrono::steady_clock;

  explicit LowCmdRelay(const rclcpp::NodeOptions & options = rclcpp::NodeOptions())
  : Node("igris_lowcmd_relay", options)
  {
    const auto input_topic = declare_parameter<std::string>(
      "input_topic", "rt/lowcmd_desired");
    const auto output_topic = declare_parameter<std::string>(
      "output_topic", "rt/lowcmd");
    parent_pid_ = declare_parameter<std::int64_t>("parent_pid", 0);

    if (input_topic.empty() || output_topic.empty()) {
      throw std::invalid_argument("input_topic and output_topic must not be empty");
    }

    auto qos = rclcpp::QoS(rclcpp::KeepLast(1));
    qos.best_effort();
    qos.durability_volatile();

    publisher_ = create_publisher<LowCmd>(output_topic, qos);
    subscription_ = create_subscription<LowCmd>(
      input_topic, qos,
      [this](LowCmd::ConstSharedPtr message) {
        std::lock_guard<std::mutex> lock(command_mutex_);
        latest_command_ = *message;
        have_command_ = true;
      });

    const std::string resolved_input_topic{subscription_->get_topic_name()};
    const std::string resolved_output_topic{publisher_->get_topic_name()};
    if (resolved_input_topic == resolved_output_topic) {
      throw std::invalid_argument(
              "input_topic and output_topic resolve to the same topic; refusing a relay loop");
    }

    RCLCPP_INFO(
      get_logger(),
      "LowCmd relay ready: %s -> %s at %.1f Hz; waiting for first desired command",
      resolved_input_topic.c_str(), resolved_output_topic.c_str(), kPublishHz);

    running_.store(true, std::memory_order_release);
    publish_thread_ = std::thread(&LowCmdRelay::publish_loop, this);
  }

  ~LowCmdRelay() override
  {
    stop();
  }

  LowCmdRelay(const LowCmdRelay &) = delete;
  LowCmdRelay & operator=(const LowCmdRelay &) = delete;

  void stop()
  {
    running_.store(false, std::memory_order_release);
    if (publish_thread_.joinable()) {
      publish_thread_.join();
    }
  }

private:
  static constexpr double kPublishHz = 300.0;
  static constexpr double kStatisticsPeriodSeconds = 5.0;
  static constexpr std::int64_t kNanosecondsPerSecond = 1000000000LL;

  void publish_loop()
  {
    const auto period = std::chrono::duration_cast<SteadyClock::duration>(
      std::chrono::duration<double>(1.0 / kPublishHz));
    const auto statistics_period = std::chrono::duration_cast<SteadyClock::duration>(
      std::chrono::duration<double>(kStatisticsPeriodSeconds));

    auto next_tick = SteadyClock::now();
    auto window_start = next_tick;
    auto previous_publish = SteadyClock::time_point{};
    auto max_gap = SteadyClock::duration::zero();
    std::uint64_t interval_count = 0;
    std::uint64_t missed_deadlines = 0;
    bool first_publish_logged = false;

    while (running_.load(std::memory_order_acquire) && rclcpp::ok()) {
      next_tick += period;
      std::this_thread::sleep_until(next_tick);

      if (!running_.load(std::memory_order_acquire) || !rclcpp::ok()) {
        break;
      }
      if (parent_pid_ > 0 && static_cast<std::int64_t>(::getppid()) != parent_pid_) {
        RCLCPP_ERROR(
          get_logger(), "Control parent process %lld exited; stopping stale LowCmd output",
          static_cast<long long>(parent_pid_));
        running_.store(false, std::memory_order_release);
        rclcpp::shutdown();
        break;
      }

      const auto wake_time = SteadyClock::now();
      if (wake_time > next_tick + period) {
        ++missed_deadlines;
        // Do not emit catch-up bursts after a long scheduler stall.
        next_tick = wake_time;
      }

      LowCmd outgoing;
      {
        std::lock_guard<std::mutex> lock(command_mutex_);
        if (!have_command_) {
          continue;
        }
        outgoing = latest_command_;
      }

      refresh_header(outgoing);
      publisher_->publish(outgoing);

      const auto publish_time = SteadyClock::now();
      if (!first_publish_logged) {
        RCLCPP_INFO(get_logger(), "First desired LowCmd received; fixed-rate output started");
        first_publish_logged = true;
        window_start = publish_time;
      }

      if (previous_publish != SteadyClock::time_point{}) {
        const auto gap = publish_time - previous_publish;
        max_gap = std::max(max_gap, gap);
        ++interval_count;
      }
      previous_publish = publish_time;

      const auto window_elapsed = publish_time - window_start;
      if (window_elapsed >= statistics_period && interval_count > 0U) {
        const auto elapsed_seconds = std::chrono::duration<double>(window_elapsed).count();
        const auto measured_hz = static_cast<double>(interval_count) / elapsed_seconds;
        const auto max_gap_ms = std::chrono::duration<double, std::milli>(max_gap).count();
        RCLCPP_INFO(
          get_logger(), "LowCmd output: %.2f Hz, max gap %.3f ms, missed deadlines %llu",
          measured_hz, max_gap_ms,
          static_cast<unsigned long long>(missed_deadlines));

        window_start = publish_time;
        max_gap = SteadyClock::duration::zero();
        interval_count = 0;
        missed_deadlines = 0;
      }
    }
  }

  void refresh_header(LowCmd & command)
  {
    command.header.seq = sequence_++;

    const auto wall_time = std::chrono::system_clock::now().time_since_epoch();
    const auto total_nanoseconds =
      std::chrono::duration_cast<std::chrono::nanoseconds>(wall_time).count();
    command.header.sec = static_cast<std::uint32_t>(
      total_nanoseconds / kNanosecondsPerSecond);
    command.header.nanosec = static_cast<std::uint32_t>(
      total_nanoseconds % kNanosecondsPerSecond);
  }

  rclcpp::Publisher<LowCmd>::SharedPtr publisher_;
  rclcpp::Subscription<LowCmd>::SharedPtr subscription_;

  std::mutex command_mutex_;
  LowCmd latest_command_{};
  bool have_command_{false};

  std::atomic<bool> running_{false};
  std::thread publish_thread_;
  std::uint32_t sequence_{0};
  std::int64_t parent_pid_{0};
};

}  // namespace igris_lowcmd_relay

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  std::shared_ptr<igris_lowcmd_relay::LowCmdRelay> relay;
  try {
    relay = std::make_shared<igris_lowcmd_relay::LowCmdRelay>();
    rclcpp::spin(relay);
  } catch (const std::exception & exception) {
    RCLCPP_FATAL(rclcpp::get_logger("igris_lowcmd_relay"), "%s", exception.what());
    if (relay) {
      relay->stop();
    }
    rclcpp::shutdown();
    return 1;
  }

  relay->stop();
  rclcpp::shutdown();
  return 0;
}
