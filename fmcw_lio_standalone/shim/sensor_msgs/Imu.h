#pragma once
#include "ros/ros.h"
namespace geometry_msgs {
struct Vector3 { double x{0}, y{0}, z{0}; };
struct Quaternion { double x{0}, y{0}, z{0}, w{1}; };
}
namespace sensor_msgs {
struct Imu {
    using Ptr = std::shared_ptr<Imu>;
    using ConstPtr = std::shared_ptr<const Imu>;
    ros::Header header;
    geometry_msgs::Quaternion orientation;
    geometry_msgs::Vector3 angular_velocity;
    geometry_msgs::Vector3 linear_acceleration;
};
}
