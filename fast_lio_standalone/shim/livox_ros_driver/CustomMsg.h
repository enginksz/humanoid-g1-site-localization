#pragma once
#include "ros/ros.h"
namespace livox_ros_driver {
struct CustomPoint { std::uint32_t offset_time{0}; float x{0}, y{0}, z{0}; std::uint8_t reflectivity{0}, tag{0}, line{0}; };
struct CustomMsg {
    using Ptr = std::shared_ptr<CustomMsg>;
    using ConstPtr = std::shared_ptr<const CustomMsg>;
    ros::Header header; std::uint64_t timebase{0}; std::uint32_t point_num{0};
    std::vector<CustomPoint> points;
};
}
