#include <algorithm>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <array>
#include <builtin_interfaces/msg/time.hpp>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <dds/dds.hpp>
#include <igris_c_sensor/robot_igris_c_msgs.hpp>
#include <memory>
#include <mutex>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <optional>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using CompressedMessage = igris_c::msg::dds::CompressedMessage;
using IgrisHeader       = igris_c::msg::dds::Header;

constexpr char kDefaultDdsNamespace[] = "igris_c_IG05";
constexpr char kD435ColorLeaf[]       = "sensor/d435_color";
constexpr char kD435DepthLeaf[]       = "sensor/d435_depth";
constexpr char kEyesStereoLeaf[]      = "sensor/eyes_stereo";
constexpr char kLeftHandLeaf[]        = "sensor/left_hand";
constexpr char kRightHandLeaf[]       = "sensor/right_hand";

constexpr char kRosHeadColorTopic[]       = "/rs_comp/cam_213622075556/color/image/compressed";
constexpr char kRosHeadDepthTopic[]       = "/rs_comp/cam_213622075556/depth/image/compressed";
constexpr char kRosLeftWristColorTopic[]  = "/rs_comp/cam_335122271161/color/image/compressed";
constexpr char kRosRightWristColorTopic[] = "/rs_comp/cam_335122271403/color/image/compressed";

std::string trim_leading_slashes(std::string value) {
    while (!value.empty() && value.front() == '/') {
        value.erase(value.begin());
    }
    return value;
}

std::string trim_trailing_slashes(std::string value) {
    while (!value.empty() && value.back() == '/') {
        value.pop_back();
    }
    return value;
}

std::string make_scoped_topic(const std::string &dds_namespace, const std::string &topic) {
    const auto ns       = trim_trailing_slashes(trim_leading_slashes(dds_namespace));
    const auto stripped = trim_leading_slashes(topic);
    if (ns.empty()) {
        return stripped;
    }
    const auto prefix = ns + "/";
    if (stripped.rfind(prefix, 0) == 0) {
        return stripped;
    }
    return prefix + stripped;
}

std::string frame_id_to_string(const IgrisHeader &igris_header) {
    const auto &raw = igris_header.frame_id();
    const auto end  = std::find(raw.begin(), raw.end(), '\0');
    return std::string(raw.begin(), end);
}

std::string to_lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return value;
}

std::string encoding_extension_for_format(std::string format) {
    format = to_lower(std::move(format));

    if (format.find("jpeg") != std::string::npos || format.find("jpg") != std::string::npos) {
        return ".jpg";
    }
    if (format.find("png") != std::string::npos) {
        return ".png";
    }

    throw std::runtime_error("Unsupported compressed image format: " + format);
}

builtin_interfaces::msg::Time to_ros_stamp_msg(const IgrisHeader &igris_header, const rclcpp::Time &fallback_stamp) {
    if (igris_header.sec() == 0 && igris_header.nanosec() == 0) {
        const auto fallback_ns = fallback_stamp.nanoseconds();
        builtin_interfaces::msg::Time stamp;
        stamp.sec     = static_cast<std::int32_t>(fallback_ns / 1000000000LL);
        stamp.nanosec = static_cast<std::uint32_t>(fallback_ns % 1000000000LL);
        return stamp;
    }

    builtin_interfaces::msg::Time stamp;
    stamp.sec     = static_cast<std::int32_t>(igris_header.sec());
    stamp.nanosec = igris_header.nanosec();
    return stamp;
}

sensor_msgs::msg::CompressedImage to_ros_image(const CompressedMessage &compressed_message, const IgrisHeader &igris_header,
                                               const std::string &fallback_frame_id, const rclcpp::Time &fallback_stamp,
                                               double resize_scale, bool rotate_180 = false) {
    sensor_msgs::msg::CompressedImage ros_image;
    const auto frame_id       = frame_id_to_string(igris_header);
    ros_image.header.frame_id = frame_id.empty() ? fallback_frame_id : frame_id;
    ros_image.header.stamp    = to_ros_stamp_msg(igris_header, fallback_stamp);
    ros_image.format          = compressed_message.format();

    if (resize_scale == 1.0 && !rotate_180) {
        ros_image.data = compressed_message.image_data();
        return ros_image;
    }

    const auto image_data = compressed_message.image_data();
    cv::Mat decoded       = cv::imdecode(image_data, cv::IMREAD_UNCHANGED);
    if (decoded.empty()) {
        throw std::runtime_error("Failed to decode compressed image");
    }

    if (rotate_180) {
        cv::rotate(decoded, decoded, cv::ROTATE_180);
    }

    const int resized_width  = std::max(1, static_cast<int>(std::lround(decoded.cols * resize_scale)));
    const int resized_height = std::max(1, static_cast<int>(std::lround(decoded.rows * resize_scale)));

    if (resized_width == decoded.cols && resized_height == decoded.rows) {
        ros_image.data = image_data;
        return ros_image;
    }

    cv::Mat resized;
    cv::resize(decoded, resized, cv::Size(resized_width, resized_height), 0.0, 0.0, cv::INTER_AREA);

    if (!cv::imencode(encoding_extension_for_format(ros_image.format), resized, ros_image.data)) {
        throw std::runtime_error("Failed to encode resized image");
    }

    return ros_image;
}

sensor_msgs::msg::CameraInfo make_camera_info(const builtin_interfaces::msg::Time &stamp, const std::string &frame_id,
                                              const cv::Mat &projection, const cv::Size &size) {
    cv::Mat projection64;
    projection.convertTo(projection64, CV_64F);

    sensor_msgs::msg::CameraInfo camera_info;
    camera_info.header.stamp     = stamp;
    camera_info.header.frame_id  = frame_id;
    camera_info.width            = static_cast<std::uint32_t>(size.width);
    camera_info.height           = static_cast<std::uint32_t>(size.height);
    camera_info.distortion_model = "plumb_bob";
    camera_info.d                = {0.0, 0.0, 0.0, 0.0, 0.0};
    camera_info.r                = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};

    camera_info.k = {
        projection64.at<double>(0, 0), projection64.at<double>(0, 1), projection64.at<double>(0, 2),
        projection64.at<double>(1, 0), projection64.at<double>(1, 1), projection64.at<double>(1, 2),
        projection64.at<double>(2, 0), projection64.at<double>(2, 1), projection64.at<double>(2, 2),
    };

    camera_info.p = {
        projection64.at<double>(0, 0), projection64.at<double>(0, 1), projection64.at<double>(0, 2), projection64.at<double>(0, 3),
        projection64.at<double>(1, 0), projection64.at<double>(1, 1), projection64.at<double>(1, 2), projection64.at<double>(1, 3),
        projection64.at<double>(2, 0), projection64.at<double>(2, 1), projection64.at<double>(2, 2), projection64.at<double>(2, 3),
    };

    return camera_info;
}

std::vector<int> jpeg_encode_params(int jpeg_quality, const std::string &format) {
    if (encoding_extension_for_format(format) == ".jpg") {
        return {cv::IMWRITE_JPEG_QUALITY, jpeg_quality};
    }
    return {};
}

sensor_msgs::msg::CompressedImage make_compressed_image(const cv::Mat &image, const std::string &format,
                                                        const builtin_interfaces::msg::Time &stamp, const std::string &frame_id,
                                                        int jpeg_quality) {
    sensor_msgs::msg::CompressedImage msg;
    msg.header.stamp    = stamp;
    msg.header.frame_id = frame_id;
    msg.format          = format;

    const auto params = jpeg_encode_params(jpeg_quality, format);
    if (!cv::imencode(encoding_extension_for_format(format), image, msg.data, params)) {
        throw std::runtime_error("Failed to encode stereo rectified image");
    }

    return msg;
}

cv::Mat decode_color_image(const CompressedMessage &message) {
    cv::Mat decoded = cv::imdecode(message.image_data(), cv::IMREAD_COLOR);
    if (decoded.empty()) {
        throw std::runtime_error("Failed to decode compressed image");
    }
    return decoded;
}

cv::Mat resize_by_scale(const cv::Mat &image, double scale) {
    if (scale == 1.0) {
        return image.clone();
    }

    const int resized_width  = std::max(1, static_cast<int>(std::lround(image.cols * scale)));
    const int resized_height = std::max(1, static_cast<int>(std::lround(image.rows * scale)));

    cv::Mat resized;
    cv::resize(image, resized, cv::Size(resized_width, resized_height), 0.0, 0.0, cv::INTER_AREA);
    return resized;
}

}  // namespace

class IgrisCSensorRobotNode : public rclcpp::Node {
  public:
    IgrisCSensorRobotNode() : Node("igris_c_sensor_robot") {
        domain_id_      = this->declare_parameter<int>("domain_id", 0);
        dds_namespace_  = this->declare_parameter<std::string>("dds_namespace", kDefaultDdsNamespace);
        d435_color_topic_ = this->declare_parameter<std::string>(
            "d435_color_topic", make_scoped_topic(dds_namespace_, kD435ColorLeaf));
        d435_depth_topic_ = this->declare_parameter<std::string>(
            "d435_depth_topic", make_scoped_topic(dds_namespace_, kD435DepthLeaf));
        eyes_stereo_topic_ = this->declare_parameter<std::string>(
            "eyes_stereo_topic", make_scoped_topic(dds_namespace_, kEyesStereoLeaf));
        left_hand_topic_ = this->declare_parameter<std::string>(
            "left_hand_topic", make_scoped_topic(dds_namespace_, kLeftHandLeaf));
        right_hand_topic_ = this->declare_parameter<std::string>(
            "right_hand_topic", make_scoped_topic(dds_namespace_, kRightHandLeaf));

        const std::vector<std::string> default_igris_topics = {
            d435_color_topic_, d435_depth_topic_, eyes_stereo_topic_, left_hand_topic_, right_hand_topic_,
        };

        igris_topics_            = this->declare_parameter<std::vector<std::string>>("igris_topics", default_igris_topics);
        dds_history_depth_       = this->declare_parameter<int>("dds_history_depth", 5);
        dds_poll_period_ms_      = this->declare_parameter<double>("dds_poll_period_ms", 2.0);
        frame_id_                = this->declare_parameter<std::string>("frame_id", "igris_c");
        resize_scale_            = this->declare_parameter<double>("resize_scale", 0.5);
        combined_resize_scale_   = this->declare_parameter<double>("combined_resize_scale", 0.5);
        combined_color_topic_    = this->declare_parameter<std::string>("combined_color_topic", "/rs_comp/combined/color/image/compressed");
        combined_jpeg_quality_   = this->declare_parameter<int>("combined_jpeg_quality", 85);
        hand_rotate_180_         = this->declare_parameter<bool>("hand_rotate_180", true);
        stereo_enabled_          = this->declare_parameter<bool>("stereo_enabled", true);
        stereo_ros_source_topic_ = this->declare_parameter<std::string>(
            "stereo_ros_source_topic", "/igris_c_IG05/sensor/eyes_stereo/compressed");
        stereo_map_path_         = this->declare_parameter<std::string>("stereo_map_path", "stereo_rectify_maps_tuned.yml.gz");
        stereo_swap_lr_          = this->declare_parameter<bool>("stereo_swap_lr", false);
        stereo_jpeg_quality_     = this->declare_parameter<int>("stereo_jpeg_quality", 85);
        stereo_left_topic_       = this->declare_parameter<std::string>("stereo_left_topic", "/left/image_rect/compressed");
        stereo_right_topic_      = this->declare_parameter<std::string>("stereo_right_topic", "/right/image_rect/compressed");
        stereo_left_info_topic_  = this->declare_parameter<std::string>("stereo_left_info_topic", "/igris_c/sensor/left/camera_info");
        stereo_right_info_topic_ = this->declare_parameter<std::string>("stereo_right_info_topic", "/igris_c/sensor/right/camera_info");
        stereo_left_frame_id_    = this->declare_parameter<std::string>("stereo_left_frame_id", "left_camera");
        stereo_right_frame_id_   = this->declare_parameter<std::string>("stereo_right_frame_id", "right_camera");

        if (resize_scale_ <= 0.0 || resize_scale_ > 1.0) {
            throw std::runtime_error("Parameter 'resize_scale' must be in the range (0.0, 1.0]");
        }
        if (combined_resize_scale_ <= 0.0 || combined_resize_scale_ > 1.0) {
            throw std::runtime_error("Parameter 'combined_resize_scale' must be in the range (0.0, 1.0]");
        }
        if (combined_jpeg_quality_ < 1 || combined_jpeg_quality_ > 100) {
            throw std::runtime_error("Parameter 'combined_jpeg_quality' must be in the range [1, 100]");
        }
        if (stereo_jpeg_quality_ < 1 || stereo_jpeg_quality_ > 100) {
            throw std::runtime_error("Parameter 'stereo_jpeg_quality' must be in the range [1, 100]");
        }
        if (dds_history_depth_ <= 0) {
            throw std::runtime_error("Parameter 'dds_history_depth' must be positive");
        }
        if (dds_poll_period_ms_ <= 0.0) {
            throw std::runtime_error("Parameter 'dds_poll_period_ms' must be positive");
        }

        const auto reliable_image_qos = rclcpp::QoS(rclcpp::KeepLast(1)).reliable();
        combined_color_pub_ = this->create_publisher<sensor_msgs::msg::CompressedImage>(combined_color_topic_, reliable_image_qos);
        if (stereo_enabled_) {
            initialize_stereo_support();
        }

        participant_.emplace(domain_id_);
        dds_subscriber_.emplace(*participant_);

        if (igris_topics_.empty()) {
            throw std::runtime_error("Parameter 'igris_topics' must contain at least one DDS topic");
        }

        auto reader_qos = dds_subscriber_->default_datareader_qos();
        reader_qos << dds::core::policy::Reliability::BestEffort()
                   << dds::core::policy::History::KeepLast(static_cast<std::int32_t>(dds_history_depth_));

        streams_.reserve(igris_topics_.size());
        for (const auto &topic_name : igris_topics_) {
            if (topic_name.empty()) {
                continue;
            }

            auto stream          = std::make_unique<Stream>();
            const auto ros_topic = ros_topic_for_igris_topic(topic_name);

            stream->igris_topic = topic_name;
            stream->publisher   = this->create_publisher<sensor_msgs::msg::CompressedImage>(ros_topic, rclcpp::SensorDataQoS());
            stream->dds_topic.emplace(*participant_, topic_name);
            stream->reader.emplace(*dds_subscriber_, *stream->dds_topic, reader_qos);

            RCLCPP_INFO(this->get_logger(), "Bridging IGRIS topic '%s' to ROS2 topic '%s'", topic_name.c_str(), ros_topic.c_str());
            streams_.push_back(std::move(stream));
        }

        if (streams_.empty()) {
            throw std::runtime_error("No valid IGRIS topics configured for bridging");
        }

        const auto poll_period = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::duration<double, std::milli>(dds_poll_period_ms_));
        poll_timer_ = this->create_wall_timer(poll_period, [this]() { this->poll_dds(); });

        RCLCPP_INFO(this->get_logger(), "IGRIS-C robot DDS bridge initialized: domain_id=%d namespace='%s' qos=BestEffort depth=%d",
                    domain_id_, dds_namespace_.c_str(), dds_history_depth_);
    }

    ~IgrisCSensorRobotNode() override = default;

  private:
    struct Stream {
        std::string igris_topic;
        rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr publisher;
        std::optional<dds::topic::Topic<CompressedMessage>> dds_topic;
        std::optional<dds::sub::DataReader<CompressedMessage>> reader;
    };

    std::string ros_topic_for_igris_topic(const std::string &topic_name) const {
        if (topic_name == d435_color_topic_) {
            return kRosHeadColorTopic;
        }
        if (topic_name == d435_depth_topic_) {
            return kRosHeadDepthTopic;
        }
        if (topic_name == left_hand_topic_) {
            return kRosLeftWristColorTopic;
        }
        if (topic_name == right_hand_topic_) {
            return kRosRightWristColorTopic;
        }
        return "/" + trim_leading_slashes(topic_name) + "/compressed";
    }

    std::string resolve_stereo_map_path(const std::string &path) const {
        namespace fs = std::filesystem;

        if (path.empty()) {
            return path;
        }

        const fs::path requested(path);
        std::vector<fs::path> candidates;

        if (requested.is_absolute()) {
            candidates.push_back(requested);
        } else {
            try {
                const fs::path pkg_share(ament_index_cpp::get_package_share_directory("stereo_sbs_cam_pub"));
                candidates.push_back(pkg_share / requested);
                candidates.push_back(pkg_share / "config" / requested);
                candidates.push_back(pkg_share / "config" / requested.filename());
            } catch (const std::exception &) {
            }

            const fs::path source_config_dir = fs::path(__FILE__).parent_path() / ".." / ".." / ".." / "stereo_sbs_cam_pub" / "config";
            candidates.push_back(source_config_dir / requested);
            candidates.push_back(source_config_dir / requested.filename());
            candidates.push_back(fs::absolute(requested));
            candidates.push_back(fs::absolute(requested.filename()));
        }

        for (const auto &candidate : candidates) {
            if (!candidate.empty() && fs::is_regular_file(candidate)) {
                return candidate.lexically_normal().string();
            }
        }

        if (!candidates.empty()) {
            return candidates.front().lexically_normal().string();
        }
        return path;
    }

    cv::Size read_rectified_size(const cv::FileStorage &storage) const {
        cv::Mat size_mat;
        storage["image_size"] >> size_mat;
        if (size_mat.empty()) {
            storage["img_size"] >> size_mat;
        }

        if (!size_mat.empty() && size_mat.total() >= 2) {
            cv::Mat flat64;
            size_mat.reshape(1, 1).convertTo(flat64, CV_64F);
            return {static_cast<int>(std::lround(flat64.at<double>(0))), static_cast<int>(std::lround(flat64.at<double>(1)))};
        }
        return {};
    }

    void initialize_stereo_support() {
        stereo_map_path_ = resolve_stereo_map_path(stereo_map_path_);

        cv::FileStorage storage(stereo_map_path_, cv::FileStorage::READ);
        if (!storage.isOpened()) {
            throw std::runtime_error("Failed to open stereo map file: " + stereo_map_path_);
        }

        storage["map1x"] >> stereo_left_map1_;
        storage["map1y"] >> stereo_left_map2_;
        storage["map2x"] >> stereo_right_map1_;
        storage["map2y"] >> stereo_right_map2_;
        if (stereo_left_map1_.empty()) {
            storage["mapL1"] >> stereo_left_map1_;
            storage["mapL2"] >> stereo_left_map2_;
            storage["mapR1"] >> stereo_right_map1_;
            storage["mapR2"] >> stereo_right_map2_;
        }
        storage["P1"] >> stereo_p1_;
        storage["P2"] >> stereo_p2_;
        stereo_rect_size_ = read_rectified_size(storage);
        storage.release();

        if (stereo_left_map1_.empty() || stereo_left_map2_.empty() || stereo_right_map1_.empty() || stereo_right_map2_.empty()) {
            throw std::runtime_error("Stereo map file is missing remap matrices: " + stereo_map_path_);
        }
        if (stereo_p1_.empty() || stereo_p2_.empty()) {
            throw std::runtime_error("Stereo map file is missing projection matrices: " + stereo_map_path_);
        }
        if (stereo_rect_size_.width <= 0 || stereo_rect_size_.height <= 0) {
            stereo_rect_size_ = stereo_left_map1_.size();
        }

        const auto reliable_image_qos = rclcpp::QoS(rclcpp::KeepLast(1)).reliable();
        stereo_left_pub_              = this->create_publisher<sensor_msgs::msg::CompressedImage>(stereo_left_topic_, reliable_image_qos);
        stereo_right_pub_             = this->create_publisher<sensor_msgs::msg::CompressedImage>(stereo_right_topic_, reliable_image_qos);
        const auto info_qos           = rclcpp::QoS(rclcpp::KeepLast(1)).reliable();
        stereo_left_info_pub_         = this->create_publisher<sensor_msgs::msg::CameraInfo>(stereo_left_info_topic_, info_qos);
        stereo_right_info_pub_        = this->create_publisher<sensor_msgs::msg::CameraInfo>(stereo_right_info_topic_, info_qos);
        stereo_source_sub_ = this->create_subscription<sensor_msgs::msg::CompressedImage>(
            stereo_ros_source_topic_, rclcpp::SensorDataQoS(),
            [this](const sensor_msgs::msg::CompressedImage::SharedPtr message) {
                try {
                    this->publish_stereo_outputs(*message);
                } catch (const std::exception &error) {
                    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                                         "Skipping ROS stereo frame from '%s': %s", stereo_ros_source_topic_.c_str(), error.what());
                }
            });

        RCLCPP_INFO(this->get_logger(), "Stereo rectification subscribes to ROS2 '%s' using map '%s'", stereo_ros_source_topic_.c_str(),
                    stereo_map_path_.c_str());
        RCLCPP_INFO(this->get_logger(), "Stereo outputs: left='%s', right='%s'", stereo_left_topic_.c_str(), stereo_right_topic_.c_str());
    }

    bool is_combined_source_topic(const std::string &topic_name) const {
        return topic_name == left_hand_topic_ || topic_name == d435_color_topic_ || topic_name == right_hand_topic_;
    }

    bool should_rotate_hand_topic(const std::string &topic_name) const {
        return hand_rotate_180_ && (topic_name == left_hand_topic_ || topic_name == right_hand_topic_);
    }

    cv::Mat compose_combined_color_image() const {
        const int output_height = std::max({combined_left_hand_.rows, combined_d435_color_.rows, combined_right_hand_.rows});
        const int output_width  = combined_left_hand_.cols + combined_d435_color_.cols + combined_right_hand_.cols;

        cv::Mat combined(output_height, output_width, CV_8UC3, cv::Scalar::all(0));
        int offset_x = 0;
        for (const auto *image : {&combined_left_hand_, &combined_d435_color_, &combined_right_hand_}) {
            const int offset_y = (output_height - image->rows) / 2;
            image->copyTo(combined(cv::Rect(offset_x, offset_y, image->cols, image->rows)));
            offset_x += image->cols;
        }

        return combined;
    }

    void update_combined_color_output(const std::string &topic_name, const CompressedMessage &message, const IgrisHeader &igris_header) {
        cv::Mat decoded = decode_color_image(message);
        if (should_rotate_hand_topic(topic_name)) {
            cv::rotate(decoded, decoded, cv::ROTATE_180);
        }
        cv::Mat resized = resize_by_scale(decoded, combined_resize_scale_);
        cv::Mat combined_image;

        {
            std::lock_guard<std::mutex> lock(combined_mutex_);
            if (topic_name == left_hand_topic_) {
                combined_left_hand_ = std::move(resized);
            } else if (topic_name == d435_color_topic_) {
                combined_d435_color_ = std::move(resized);
            } else if (topic_name == right_hand_topic_) {
                combined_right_hand_ = std::move(resized);
            } else {
                return;
            }

            if (combined_left_hand_.empty() || combined_d435_color_.empty() || combined_right_hand_.empty()) {
                return;
            }

            combined_image = compose_combined_color_image();
        }

        const auto stamp = to_ros_stamp_msg(igris_header, this->now());
        combined_color_pub_->publish(make_compressed_image(combined_image, "jpeg", stamp, frame_id_, combined_jpeg_quality_));
    }

    void publish_stereo_outputs(const sensor_msgs::msg::CompressedImage &message) {
        cv::Mat stereo_frame = cv::imdecode(message.data, cv::IMREAD_COLOR);
        if (stereo_frame.empty()) {
            throw std::runtime_error("Failed to decode ROS stereo compressed image");
        }
        if (stereo_frame.cols % 2 != 0) {
            throw std::runtime_error("Stereo SBS frame width must be even");
        }

        // The robot publishes the stereo SBS frame upright. Preserve that
        // orientation and rectify each half directly with its calibration map.
        const int half_width = stereo_frame.cols / 2;
        cv::Mat left_raw     = stereo_frame(cv::Rect(0, 0, half_width, stereo_frame.rows)).clone();
        cv::Mat right_raw    = stereo_frame(cv::Rect(half_width, 0, half_width, stereo_frame.rows)).clone();
        if (stereo_swap_lr_) {
            std::swap(left_raw, right_raw);
        }

        if (left_raw.size() != stereo_rect_size_ || right_raw.size() != stereo_rect_size_) {
            RCLCPP_WARN_THROTTLE(
                this->get_logger(), *this->get_clock(), 5000,
                "Skipping stereo frame: split eye size is %dx%d, calibration map requires %dx%d (input topic '%s')",
                left_raw.cols, left_raw.rows, stereo_rect_size_.width, stereo_rect_size_.height,
                stereo_ros_source_topic_.c_str());
            return;
        }

        cv::Mat left_rect;
        cv::Mat right_rect;
        cv::remap(left_raw, left_rect, stereo_left_map1_, stereo_left_map2_, cv::INTER_LINEAR);
        cv::remap(right_raw, right_rect, stereo_right_map1_, stereo_right_map2_, cv::INTER_LINEAR);

        const auto stamp   = message.header.stamp;
        auto output_format = message.format.empty() ? std::string("jpeg") : message.format;
        try {
            (void)encoding_extension_for_format(output_format);
        } catch (const std::exception &) {
            output_format = "jpeg";
        }

        stereo_left_pub_->publish(make_compressed_image(left_rect, output_format, stamp, stereo_left_frame_id_, stereo_jpeg_quality_));
        stereo_right_pub_->publish(make_compressed_image(right_rect, output_format, stamp, stereo_right_frame_id_, stereo_jpeg_quality_));

        const cv::Size output_size  = left_rect.size();

        stereo_left_info_pub_->publish(make_camera_info(stamp, stereo_left_frame_id_, stereo_p1_, output_size));
        stereo_right_info_pub_->publish(make_camera_info(stamp, stereo_right_frame_id_, stereo_p2_, output_size));
    }

    void poll_dds() {
        for (const auto &stream : streams_) {
            if (!stream || !stream->reader) {
                continue;
            }

            try {
                auto samples = stream->reader->take();
                for (const auto &sample : samples) {
                    if (!sample.info().valid()) {
                        continue;
                    }
                    handle_message(*stream, sample.data());
                }
            } catch (const std::exception &error) {
                RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000, "DDS read failed for '%s': %s",
                                     stream->igris_topic.c_str(), error.what());
            }
        }
    }

    void handle_message(const Stream &stream, const CompressedMessage &message) {
        if (message.image_data().empty()) {
            return;
        }

        try {
            // The stereo ROS topic is also the rectifier input. Resizing the
            // native 1280x480 SBS frame here makes each half incompatible with
            // the 640x480 calibration maps.
            const bool preserve_native_size =
                stream.igris_topic == d435_depth_topic_ || stream.igris_topic == eyes_stereo_topic_;
            const double effective_resize_scale = preserve_native_size ? 1.0 : resize_scale_;
            auto ros_image = to_ros_image(message, message.header(), frame_id_, this->now(), effective_resize_scale,
                                          should_rotate_hand_topic(stream.igris_topic));
            stream.publisher->publish(std::move(ros_image));
            if (is_combined_source_topic(stream.igris_topic)) {
                update_combined_color_output(stream.igris_topic, message, message.header());
            }
        } catch (const std::exception &error) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000, "Skipping frame from '%s': %s", stream.igris_topic.c_str(),
                                 error.what());
        }
    }

    int domain_id_{0};
    std::string dds_namespace_;
    std::string d435_color_topic_;
    std::string d435_depth_topic_;
    std::string eyes_stereo_topic_;
    std::string left_hand_topic_;
    std::string right_hand_topic_;
    std::vector<std::string> igris_topics_;
    int dds_history_depth_{5};
    double dds_poll_period_ms_{2.0};
    std::string frame_id_;
    double resize_scale_{1.0};
    double combined_resize_scale_{0.5};
    std::string combined_color_topic_;
    int combined_jpeg_quality_{85};
    bool hand_rotate_180_{true};
    std::mutex combined_mutex_;
    cv::Mat combined_left_hand_;
    cv::Mat combined_d435_color_;
    cv::Mat combined_right_hand_;
    rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr combined_color_pub_;

    bool stereo_enabled_{true};
    std::string stereo_ros_source_topic_;
    std::string stereo_map_path_;
    bool stereo_swap_lr_{false};
    int stereo_jpeg_quality_{85};
    std::string stereo_left_topic_;
    std::string stereo_right_topic_;
    std::string stereo_left_info_topic_;
    std::string stereo_right_info_topic_;
    std::string stereo_left_frame_id_;
    std::string stereo_right_frame_id_;
    cv::Mat stereo_left_map1_;
    cv::Mat stereo_left_map2_;
    cv::Mat stereo_right_map1_;
    cv::Mat stereo_right_map2_;
    cv::Mat stereo_p1_;
    cv::Mat stereo_p2_;
    cv::Size stereo_rect_size_;
    rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr stereo_left_pub_;
    rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr stereo_right_pub_;
    rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr stereo_left_info_pub_;
    rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr stereo_right_info_pub_;
    rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr stereo_source_sub_;

    std::optional<dds::domain::DomainParticipant> participant_;
    std::optional<dds::sub::Subscriber> dds_subscriber_;
    rclcpp::TimerBase::SharedPtr poll_timer_;
    std::vector<std::unique_ptr<Stream>> streams_;
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);

    try {
        auto node = std::make_shared<IgrisCSensorRobotNode>();
        rclcpp::spin(node);
    } catch (const std::exception &error) {
        RCLCPP_FATAL(rclcpp::get_logger("igris_c_sensor_robot"), "%s", error.what());
        rclcpp::shutdown();
        return 1;
    }

    rclcpp::shutdown();
    return 0;
}
