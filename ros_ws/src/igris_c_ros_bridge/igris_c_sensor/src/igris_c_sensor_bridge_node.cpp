#include <algorithm>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <array>
#include <builtin_interfaces/msg/time.hpp>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <igris_sdk/channel_factory.hpp>
#include <igris_sdk/igris_c_msgs.hpp>
#include <igris_sdk/subscriber.hpp>
#include <memory>
#include <mutex>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
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

constexpr char kD435ColorTopic[] = "igris_c/sensor/d435_color";
constexpr char kLeftHandTopic[]  = "igris_c/sensor/left_hand";
constexpr char kRightHandTopic[] = "igris_c/sensor/right_hand";

const std::vector<std::string> kDefaultIgrisTopics = {
    kD435ColorTopic, "igris_c/sensor/d435_depth", "igris_c/sensor/eyes_stereo", kLeftHandTopic, kRightHandTopic,
};

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
    ros_image.header.frame_id = igris_header.frame_id().empty() ? fallback_frame_id : igris_header.frame_id();
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

cv::Mat scaled_projection_matrix(const cv::Mat &projection, double scale_x, double scale_y) {
    cv::Mat scaled = projection.clone();
    if (scale_x == 1.0 && scale_y == 1.0) {
        return scaled;
    }

    scaled.convertTo(scaled, CV_64F);
    for (int col = 0; col < scaled.cols; ++col) {
        scaled.at<double>(0, col) *= scale_x;
        scaled.at<double>(1, col) *= scale_y;
    }
    return scaled;
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

class IgrisCSensorNode : public rclcpp::Node {
  public:
    IgrisCSensorNode() : Node("igris_c_sensor") {
        domain_id_               = this->declare_parameter<int>("domain_id", 10);
        igris_topics_            = this->declare_parameter<std::vector<std::string>>("igris_topics", kDefaultIgrisTopics);
        frame_id_                = this->declare_parameter<std::string>("frame_id", "igris_c");
        resize_scale_            = this->declare_parameter<double>("resize_scale", 0.5);
        combined_resize_scale_   = this->declare_parameter<double>("combined_resize_scale", 0.5);
        combined_color_topic_    = this->declare_parameter<std::string>("combined_color_topic", "/rs_comp/combined/color/image/compressed");
        combined_jpeg_quality_   = this->declare_parameter<int>("combined_jpeg_quality", 85);
        hand_rotate_180_         = this->declare_parameter<bool>("hand_rotate_180", true);
        stereo_enabled_          = this->declare_parameter<bool>("stereo_enabled", true);
        stereo_source_topic_     = this->declare_parameter<std::string>("stereo_source_topic", "igris_c/sensor/eyes_stereo");
        stereo_map_path_         = this->declare_parameter<std::string>("stereo_map_path", "stereo_rectify_maps_tuned.yml.gz");
        stereo_swap_lr_          = this->declare_parameter<bool>("stereo_swap_lr", false);
        stereo_jpeg_quality_     = this->declare_parameter<int>("stereo_jpeg_quality", 85);
        stereo_output_width_     = this->declare_parameter<int>("stereo_output_width", 640);
        stereo_output_height_    = this->declare_parameter<int>("stereo_output_height", 480);
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
        if (stereo_output_width_ <= 0 || stereo_output_height_ <= 0) {
            throw std::runtime_error("Stereo output size must be positive");
        }

        const auto reliable_image_qos = rclcpp::QoS(rclcpp::KeepLast(1)).reliable();
        combined_color_pub_ = this->create_publisher<sensor_msgs::msg::CompressedImage>(combined_color_topic_, reliable_image_qos);
        if (stereo_enabled_) {
            initialize_stereo_support();
        }

        auto *factory = igris_sdk::ChannelFactory::Instance();
        if (!factory->IsInitialized()) {
            factory->Init(domain_id_);
        }

        if (!factory->IsInitialized()) {
            throw std::runtime_error("Failed to initialize igris ChannelFactory");
        }

        if (igris_topics_.empty()) {
            throw std::runtime_error("Parameter 'igris_topics' must contain at least one DDS topic");
        }

        streams_.reserve(igris_topics_.size());
        for (const auto &topic_name : igris_topics_) {
            if (topic_name.empty()) {
                continue;
            }

            auto stream          = std::make_unique<Stream>();
            const auto ros_topic = topic_name + "/compressed";

            stream->igris_topic = topic_name;
            stream->publisher   = this->create_publisher<sensor_msgs::msg::CompressedImage>(ros_topic, rclcpp::SensorDataQoS());
            stream->subscriber  = std::make_unique<igris_sdk::Subscriber<CompressedMessage>>(topic_name);

            const bool initialized = stream->subscriber->init(
                [this, stream_ptr = stream.get()](const CompressedMessage &message) { this->handle_message(*stream_ptr, message); });

            if (!initialized) {
                throw std::runtime_error("Failed to initialize subscriber for topic: " + topic_name);
            }

            RCLCPP_INFO(this->get_logger(), "Bridging IGRIS topic '%s' to ROS2 topic '%s'", topic_name.c_str(), ros_topic.c_str());
            streams_.push_back(std::move(stream));
        }

        if (streams_.empty()) {
            throw std::runtime_error("No valid IGRIS topics configured for bridging");
        }
    }

    ~IgrisCSensorNode() override {
        for (auto &stream : streams_) {
            if (stream->subscriber) {
                stream->subscriber->stop();
            }
        }
        auto *factory = igris_sdk::ChannelFactory::Instance();
        if (factory->IsInitialized()) {
            factory->Release();
        }
    }

  private:
    struct Stream {
        std::string igris_topic;
        rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr publisher;
        std::unique_ptr<igris_sdk::Subscriber<CompressedMessage>> subscriber;
    };

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

        RCLCPP_INFO(this->get_logger(), "Stereo rectification enabled for '%s' using map '%s'", stereo_source_topic_.c_str(),
                    stereo_map_path_.c_str());
        RCLCPP_INFO(this->get_logger(), "Stereo outputs: left='%s', right='%s'", stereo_left_topic_.c_str(), stereo_right_topic_.c_str());
    }

    cv::Mat resize_for_stereo_rectification(const cv::Mat &image, const std::string &label) {
        cv::Mat resized;
        cv::resize(image, resized, stereo_rect_size_, 0.0, 0.0, cv::INTER_LINEAR);
        RCLCPP_DEBUG(this->get_logger(), "Stereo %s resized from %dx%d to rectify size %dx%d", label.c_str(), image.cols, image.rows,
                     stereo_rect_size_.width, stereo_rect_size_.height);
        return resized;
    }

    bool is_combined_source_topic(const std::string &topic_name) const {
        return topic_name == kLeftHandTopic || topic_name == kD435ColorTopic || topic_name == kRightHandTopic;
    }

    bool should_rotate_hand_topic(const std::string &topic_name) const {
        return hand_rotate_180_ && (topic_name == kLeftHandTopic || topic_name == kRightHandTopic);
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
            if (topic_name == kLeftHandTopic) {
                combined_left_hand_ = std::move(resized);
            } else if (topic_name == kD435ColorTopic) {
                combined_d435_color_ = std::move(resized);
            } else if (topic_name == kRightHandTopic) {
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

    void publish_stereo_outputs(const CompressedMessage &message, const IgrisHeader &igris_header) {
        cv::Mat stereo_frame = decode_color_image(message);
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

        left_raw  = resize_for_stereo_rectification(left_raw, "left");
        right_raw = resize_for_stereo_rectification(right_raw, "right");

        cv::Mat left_rect;
        cv::Mat right_rect;
        cv::remap(left_raw, left_rect, stereo_left_map1_, stereo_left_map2_, cv::INTER_LINEAR);
        cv::remap(right_raw, right_rect, stereo_right_map1_, stereo_right_map2_, cv::INTER_LINEAR);

        const cv::Size stereo_output_size(stereo_output_width_, stereo_output_height_);
        cv::resize(left_rect, left_rect, stereo_output_size, 0.0, 0.0, cv::INTER_AREA);
        cv::resize(right_rect, right_rect, stereo_output_size, 0.0, 0.0, cv::INTER_AREA);

        const auto stamp   = to_ros_stamp_msg(igris_header, this->now());
        auto output_format = message.format().empty() ? std::string("jpeg") : std::string(message.format());
        try {
            (void)encoding_extension_for_format(output_format);
        } catch (const std::exception &) {
            output_format = "jpeg";
        }

        stereo_left_pub_->publish(make_compressed_image(left_rect, output_format, stamp, stereo_left_frame_id_, stereo_jpeg_quality_));
        stereo_right_pub_->publish(make_compressed_image(right_rect, output_format, stamp, stereo_right_frame_id_, stereo_jpeg_quality_));

        const double scale_x        = static_cast<double>(stereo_output_size.width) / static_cast<double>(stereo_rect_size_.width);
        const double scale_y        = static_cast<double>(stereo_output_size.height) / static_cast<double>(stereo_rect_size_.height);
        const auto left_projection  = scaled_projection_matrix(stereo_p1_, scale_x, scale_y);
        const auto right_projection = scaled_projection_matrix(stereo_p2_, scale_x, scale_y);
        const cv::Size output_size  = left_rect.size();

        stereo_left_info_pub_->publish(make_camera_info(stamp, stereo_left_frame_id_, left_projection, output_size));
        stereo_right_info_pub_->publish(make_camera_info(stamp, stereo_right_frame_id_, right_projection, output_size));
    }

    void handle_message(const Stream &stream, const CompressedMessage &message) {
        if (message.image_data().empty()) {
            return;
        }

        try {
            if (stereo_enabled_ && stream.igris_topic == stereo_source_topic_) {
                publish_stereo_outputs(message, message.header());
            }

            auto ros_image = to_ros_image(message, message.header(), frame_id_, this->now(), resize_scale_,
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

    int domain_id_{10};
    std::vector<std::string> igris_topics_;
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
    std::string stereo_source_topic_;
    std::string stereo_map_path_;
    bool stereo_swap_lr_{false};
    int stereo_jpeg_quality_{85};
    int stereo_output_width_{640};
    int stereo_output_height_{480};
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

    std::vector<std::unique_ptr<Stream>> streams_;
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);

    try {
        auto node = std::make_shared<IgrisCSensorNode>();
        rclcpp::spin(node);
    } catch (const std::exception &error) {
        RCLCPP_FATAL(rclcpp::get_logger("igris_c_sensor"), "%s", error.what());
        rclcpp::shutdown();
        return 1;
    }

    rclcpp::shutdown();
    return 0;
}
