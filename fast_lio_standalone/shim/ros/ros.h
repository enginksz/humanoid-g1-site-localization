// Minimal ROS1 API shim so FMCW-LIO's core builds without ROS.
// Only what the estimator core touches: ros::Time and a YAML-backed NodeHandle::param.
#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <sstream>
#include <yaml-cpp/yaml.h>

namespace ros {

struct Time {
    double t{0.0};
    Time() = default;
    explicit Time(double s) : t(s) {}
    Time& fromSec(double s) { t = s; return *this; }
    [[nodiscard]] double toSec() const { return t; }
};

struct Header {
    Time stamp;
    std::string frame_id;
    std::uint32_t seq{0};
};

class NodeHandle {
public:
    NodeHandle() = default;
    explicit NodeHandle(const std::string& yaml_path) : root_(YAML::LoadFile(yaml_path)) {}

    // key like "common/R_bl" -> root_["common"]["R_bl"]
    template <typename T>
    bool param(const std::string& key, T& out, const T& def) const {
        YAML::Node node = lookup(key);
        if (!node || node.IsNull()) { out = def; return false; }
        out = node.as<T>();
        return true;
    }

    void set(const std::string& key, const std::string& value) {
        // override a scalar from the command line: "section/key"
        std::vector<std::string> parts = split(key);
        // note: yaml-cpp's Node::operator= copies content; reset() rebinds
        YAML::Node node;
        node.reset(root_);
        for (std::size_t i = 0; i + 1 < parts.size(); ++i) {
            YAML::Node child = node[parts[i]];
            node.reset(child);
        }
        node[parts.back()] = YAML::Load(value);
    }

private:
    static std::vector<std::string> split(const std::string& key) {
        std::vector<std::string> parts; std::stringstream ss(key); std::string p;
        while (std::getline(ss, p, '/')) if (!p.empty()) parts.push_back(p);
        return parts;
    }
    [[nodiscard]] YAML::Node lookup(const std::string& key) const {
        YAML::Node node;
        node.reset(root_);
        for (const auto& p : split(key)) {
            if (!node.IsMap() || !node[p]) return YAML::Node();
            YAML::Node child = node[p];
            node.reset(child);
        }
        return node;
    }
    YAML::Node root_;
};

}  // namespace ros

// ---- extras used by FAST-LIO
#include <cassert>
#include <cstdio>
#define ROS_ASSERT(x) assert(x)
#define ROS_INFO(...) do { std::printf(__VA_ARGS__); std::printf("\n"); } while (0)
#define ROS_WARN(...) do { } while (0)
namespace ros { struct Publisher {}; }
