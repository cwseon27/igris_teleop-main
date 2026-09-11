#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <dds/dds.hpp>
#include <igris_sdk/igris_c_msgs.hpp>
#include <memory>
#include <optional>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using HandCmd   = igris_c::msg::dds::HandCmd;
using HandState = igris_c::msg::dds::HandState;
using MotorCmd  = igris_c::msg::dds::MotorCmd;

constexpr std::size_t kHandMotorCount = 12;
constexpr std::array<std::uint16_t, kHandMotorCount> kHandMotorIds = {11, 12, 13, 14, 15, 16, 21, 22, 23, 24, 25, 26};

std::string strip_slashes(std::string value) {
    while (!value.empty() && value.front() == '/') {
        value.erase(value.begin());
    }
    while (!value.empty() && value.back() == '/') {
        value.pop_back();
    }
    return value;
}

std::string scoped_topic(const std::string &dds_namespace, const std::string &topic) {
    const auto ns   = strip_slashes(dds_namespace);
    const auto leaf = strip_slashes(topic);
    if (ns.empty() || leaf.rfind(ns + "/", 0) == 0) {
        return leaf;
    }
    return ns + "/" + leaf;
}

void stamp_header(HandCmd &command, std::uint32_t sequence) {
    const auto now      = std::chrono::system_clock::now().time_since_epoch();
    const auto seconds  = std::chrono::duration_cast<std::chrono::seconds>(now);
    const auto nanosecs = std::chrono::duration_cast<std::chrono::nanoseconds>(now - seconds);
    auto &header        = command.header();
    header.seq(sequence);
    header.sec(static_cast<std::uint32_t>(seconds.count()));
    header.nanosec(static_cast<std::uint32_t>(nanosecs.count()));
    std::array<char, 256> frame_id{};
    constexpr char kFrame[] = "igris_teleop_hand";
    std::copy_n(kFrame, std::min(sizeof(kFrame) - 1, frame_id.size() - 1), frame_id.begin());
    header.frame_id(frame_id);
}

HandCmd build_hand_command(const std::array<float, kHandMotorCount> &targets, std::uint32_t sequence) {
    HandCmd command;
    stamp_header(command, sequence);
    command.motor_cmd().resize(kHandMotorCount);
    for (std::size_t index = 0; index < kHandMotorCount; ++index) {
        auto &motor = command.motor_cmd()[index];
        motor.id(kHandMotorIds[index]);
        motor.q(targets[index]);
        motor.dq(0.0F);
        motor.tau(0.0F);
        motor.kp(0.0F);
        motor.kd(0.0F);
    }
    return command;
}

HandCmd build_init_command(std::uint32_t sequence) {
    HandCmd command;
    stamp_header(command, sequence);
    command.motor_cmd().resize(1);
    auto &trigger = command.motor_cmd().front();
    trigger.id(99);
    trigger.q(0.0F);
    trigger.dq(0.0F);
    trigger.tau(0.0F);
    trigger.kp(0.0F);
    trigger.kd(0.0F);
    return command;
}

}  // namespace

class IgrisCHandNode : public rclcpp::Node {
  public:
    IgrisCHandNode() : Node("igris_c_hand_bridge") {
        domain_id_         = this->declare_parameter<int>("domain_id", 0);
        dds_namespace_     = this->declare_parameter<std::string>("dds_namespace", "igris_c_IG05");
        dds_command_topic_ = this->declare_parameter<std::string>(
            "dds_command_topic", scoped_topic(dds_namespace_, "rt/handcmd"));
        dds_state_topic_ = this->declare_parameter<std::string>(
            "dds_state_topic", scoped_topic(dds_namespace_, "rt/handstate"));
        ros_command_topic_ = this->declare_parameter<std::string>("ros_command_topic", "/igris_teleop/hand/command");
        ros_state_topic_   = this->declare_parameter<std::string>("ros_state_topic", "/igris_teleop/hand/state");
        ros_init_service_  = this->declare_parameter<std::string>("ros_init_service", "/igris_teleop/hand/init");
        publish_rate_hz_   = this->declare_parameter<double>("publish_rate_hz", 100.0);

        if (!std::isfinite(publish_rate_hz_) || publish_rate_hz_ <= 0.0) {
            throw std::runtime_error("publish_rate_hz must be finite and positive");
        }

        participant_.emplace(domain_id_);
        dds_publisher_.emplace(*participant_);
        dds_subscriber_.emplace(*participant_);
        command_topic_.emplace(*participant_, dds_command_topic_);
        state_topic_.emplace(*participant_, dds_state_topic_);

        auto writer_qos = dds_publisher_->default_datawriter_qos();
        writer_qos << dds::core::policy::Reliability::Reliable()
                   << dds::core::policy::History::KeepLast(1);
        command_writer_.emplace(*dds_publisher_, *command_topic_, writer_qos);

        auto reader_qos = dds_subscriber_->default_datareader_qos();
        reader_qos << dds::core::policy::Reliability::BestEffort()
                   << dds::core::policy::History::KeepLast(1);
        state_reader_.emplace(*dds_subscriber_, *state_topic_, reader_qos);

        state_publisher_ = this->create_publisher<std_msgs::msg::Float32MultiArray>(ros_state_topic_, rclcpp::SensorDataQoS());
        command_subscription_ = this->create_subscription<std_msgs::msg::Float32MultiArray>(
            ros_command_topic_, rclcpp::QoS(rclcpp::KeepLast(1)).best_effort(),
            [this](const std_msgs::msg::Float32MultiArray::SharedPtr message) { handle_ros_command(*message); });
        init_service_ = this->create_service<std_srvs::srv::Trigger>(
            ros_init_service_,
            [this](const std::shared_ptr<std_srvs::srv::Trigger::Request>,
                   std::shared_ptr<std_srvs::srv::Trigger::Response> response) { handle_init(response); });

        const auto command_period = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::duration<double>(1.0 / publish_rate_hz_));
        command_timer_ = this->create_wall_timer(command_period, [this]() { publish_latest_command(); });
        state_timer_   = this->create_wall_timer(std::chrono::milliseconds(5), [this]() { poll_state(); });
        status_timer_  = this->create_wall_timer(std::chrono::seconds(5), [this]() {
            const auto command_matches = command_writer_->publication_matched_status().current_count();
            const auto state_matches   = state_reader_->subscription_matched_status().current_count();
            RCLCPP_INFO(this->get_logger(),
                        "Hand bridge traffic: ros_cmd=%lu dds_cmd=%lu dds_state=%lu matches(cmd=%d,state=%d)",
                        static_cast<unsigned long>(ros_command_count_), static_cast<unsigned long>(dds_command_count_),
                        static_cast<unsigned long>(dds_state_count_), command_matches, state_matches);
        });

        RCLCPP_INFO(this->get_logger(),
                    "Robot-compatible hand bridge: DDS %s <-> %s, ROS %s <-> %s, %.1f Hz",
                    dds_state_topic_.c_str(), dds_command_topic_.c_str(), ros_state_topic_.c_str(),
                    ros_command_topic_.c_str(), publish_rate_hz_);
    }

  private:
    void handle_ros_command(const std_msgs::msg::Float32MultiArray &message) {
        if (message.data.size() != kHandMotorCount) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                                 "Ignoring hand command with %zu values; expected %zu", message.data.size(), kHandMotorCount);
            return;
        }
        std::array<float, kHandMotorCount> next{};
        for (std::size_t index = 0; index < kHandMotorCount; ++index) {
            next[index] = std::clamp(message.data[index], 0.0F, 1.0F);
        }
        latest_command_ = next;
        has_command_    = true;
        ++ros_command_count_;
    }

    void publish_latest_command() {
        if (!has_command_ || !command_writer_) {
            return;
        }
        try {
            command_writer_->write(build_hand_command(latest_command_, sequence_++));
            ++dds_command_count_;
        } catch (const std::exception &error) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000, "Hand DDS write failed: %s", error.what());
        }
    }

    void handle_init(const std::shared_ptr<std_srvs::srv::Trigger::Response> &response) {
        has_command_ = false;
        try {
            command_writer_->write(build_init_command(sequence_++));
            response->success = true;
            response->message = "Robot-compatible HandCmd init trigger published";
        } catch (const std::exception &error) {
            response->success = false;
            response->message = error.what();
        }
    }

    void poll_state() {
        if (!state_reader_) {
            return;
        }
        try {
            const auto samples = state_reader_->take();
            for (const auto &sample : samples) {
                if (!sample.info().valid()) {
                    continue;
                }
                std_msgs::msg::Float32MultiArray output;
                const auto &motors = sample.data().motor_state();
                output.data.reserve(motors.size());
                for (const auto &motor : motors) {
                    output.data.push_back(motor.q());
                }
                state_publisher_->publish(output);
                ++dds_state_count_;
            }
        } catch (const std::exception &error) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000, "Hand DDS read failed: %s", error.what());
        }
    }

    int domain_id_{0};
    double publish_rate_hz_{100.0};
    std::string dds_namespace_;
    std::string dds_command_topic_;
    std::string dds_state_topic_;
    std::string ros_command_topic_;
    std::string ros_state_topic_;
    std::string ros_init_service_;
    std::uint32_t sequence_{0};
    std::array<float, kHandMotorCount> latest_command_{};
    bool has_command_{false};
    std::uint64_t ros_command_count_{0};
    std::uint64_t dds_command_count_{0};
    std::uint64_t dds_state_count_{0};

    std::optional<dds::domain::DomainParticipant> participant_;
    std::optional<dds::pub::Publisher> dds_publisher_;
    std::optional<dds::sub::Subscriber> dds_subscriber_;
    std::optional<dds::topic::Topic<HandCmd>> command_topic_;
    std::optional<dds::topic::Topic<HandState>> state_topic_;
    std::optional<dds::pub::DataWriter<HandCmd>> command_writer_;
    std::optional<dds::sub::DataReader<HandState>> state_reader_;

    rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr state_publisher_;
    rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr command_subscription_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr init_service_;
    rclcpp::TimerBase::SharedPtr command_timer_;
    rclcpp::TimerBase::SharedPtr state_timer_;
    rclcpp::TimerBase::SharedPtr status_timer_;
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    try {
        rclcpp::spin(std::make_shared<IgrisCHandNode>());
    } catch (const std::exception &error) {
        RCLCPP_FATAL(rclcpp::get_logger("igris_c_hand_bridge"), "%s", error.what());
        rclcpp::shutdown();
        return 1;
    }
    rclcpp::shutdown();
    return 0;
}
