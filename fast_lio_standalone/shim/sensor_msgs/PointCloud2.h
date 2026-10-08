#pragma once
#include "ros/ros.h"
namespace sensor_msgs {
struct PointField {
    std::string name; std::uint32_t offset{0}; std::uint8_t datatype{0}; std::uint32_t count{1};
};
struct PointCloud2 {
    using Ptr = std::shared_ptr<PointCloud2>;
    using ConstPtr = std::shared_ptr<const PointCloud2>;
    ros::Header header;
    std::uint32_t height{1}, width{0};
    std::vector<PointField> fields;
    bool is_bigendian{false};
    std::uint32_t point_step{0}, row_step{0};
    std::vector<std::uint8_t> data;
    bool is_dense{true};
};
}
