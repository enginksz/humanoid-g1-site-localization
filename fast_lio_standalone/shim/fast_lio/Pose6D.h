#pragma once
#include <array>
namespace fast_lio {
// mirrors msg/Pose6D.msg
struct Pose6D {
    double offset_time{0};
    std::array<double, 3> acc{}, gyr{}, vel{}, pos{};
    std::array<double, 9> rot{};
};
}
