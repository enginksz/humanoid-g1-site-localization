#include <filesystem>
#include <iomanip>
#include <pcl/io/pcd_io.h>

#include "bithub/bithub.hpp"
#include "system/fmcw_lio.hpp"

namespace fmcw_lio {

BitHub::BitHub(ros::NodeHandle&,
               const std::shared_ptr<const Config>& config_ptr,
               const std::shared_ptr<const FMCWLIO>& fmcw_lio_ptr)
    : config_ptr_(config_ptr),
      fmcw_lio_ptr_(fmcw_lio_ptr),
      meas_pack_ptr_(std::make_shared<MeasPackLI>()),
      map_wait_dump_ptr_(new PointCloudType()) {
    lidar_scan_time_mean_ = 1.0 / config_ptr_->scan_rate;
    const auto leaf = static_cast<float>(config_ptr_->voxel_filter_size_map);
    voxel_filter_map_.setLeafSize(leaf, leaf, leaf);
}

void BitHub::openOutput(const std::string& out_dir) {
    out_dir_ = out_dir;
    std::filesystem::create_directories(out_dir_);
    poses_file_.open(out_dir_ + "/poses_tum.txt");
    stats_file_.open(out_dir_ + "/scan_stats.csv");
    stats_file_ << "t,proc_ms,n_down,n_dynamic,vel_ok,vx_l,vy_l,vz_l\n";
}

void BitHub::feedIMU(const sensor_msgs::Imu::Ptr& msg) {
    msg->header.stamp = ros::Time(msg->header.stamp.toSec() + config_ptr_->time_diff_bl);
    last_imu_timestamp_ = msg->header.stamp.toSec();
    imu_ptr_buffer_.push_back(msg);
}

void BitHub::feedScan(double t_beg, const std::vector<RawPoint>& pts) {
    // mirrors ConverterLiDAR::convertLiDAR: subsample, skip radius, sort by time
    PointCloudType::Ptr scan(new PointCloudType());
    const std::size_t interval = config_ptr_->point_select_interval;
    std::vector<std::pair<float, std::uint32_t>> keys;
    PointCloudType tmp;
    tmp.reserve(pts.size() / interval + 1);
    for (std::size_t i = 0; i < pts.size(); i += interval) {
        const auto& p = pts[i];
        const double r2 = double(p.x) * p.x + double(p.y) * p.y + double(p.z) * p.z;
        if (r2 < config_ptr_->skip_radius_squared) continue;
        PointType q;
        q.x = p.x; q.y = p.y; q.z = p.z;
        q.intensity = p.doppler;
        q.curvature = p.t_off * 1.0e3f;  // ms, as the core expects
        keys.emplace_back(q.curvature, static_cast<std::uint32_t>(tmp.size()));
        tmp.push_back(q);
    }
    std::sort(keys.begin(), keys.end(), [](auto& a, auto& b) { return a.first < b.first; });
    scan->reserve(keys.size());
    for (const auto& k : keys) scan->push_back(tmp.points[k.second]);
    scan_ptr_buffer_.push_back(scan);
    time_buffer_.push_back(t_beg);
}

bool BitHub::packMeasLI() {
    if (time_buffer_.empty() || imu_ptr_buffer_.empty() || scan_ptr_buffer_.empty()) return false;
    if (!meas_pack_ptr_) meas_pack_ptr_ = std::make_shared<MeasPackLI>();

    if (!is_scan_pushed_) {
        meas_pack_ptr_->scan_ptr = scan_ptr_buffer_.front();
        meas_pack_ptr_->scan_beg_time = time_buffer_.front();
        if (meas_pack_ptr_->scan_ptr->points.empty() ||
            (meas_pack_ptr_->scan_ptr->points.back().curvature * 1.0e-3) > (1.5 * lidar_scan_time_mean_)) {
            scan_ptr_buffer_.pop_front();
            time_buffer_.pop_front();
            return false;
        }
        scan_end_time_ = meas_pack_ptr_->scan_beg_time + meas_pack_ptr_->scan_ptr->points.back().curvature * 1.0e-3;
        meas_pack_ptr_->scan_end_time = scan_end_time_;
        is_scan_pushed_ = true;
    }

    // need one IMU sample beyond the scan end
    if (last_imu_timestamp_ <= scan_end_time_) return false;

    meas_pack_ptr_->imu_msg_ptr_queue.clear();
    while (!imu_ptr_buffer_.empty() && imu_ptr_buffer_.front()->header.stamp.toSec() <= scan_end_time_) {
        meas_pack_ptr_->imu_msg_ptr_queue.emplace_back(imu_ptr_buffer_.front());
        imu_ptr_buffer_.pop_front();
    }
    meas_pack_ptr_->imu_msg_ptr_queue.emplace_back(imu_ptr_buffer_.front());

    scan_ptr_buffer_.pop_front();
    time_buffer_.pop_front();
    is_scan_pushed_ = false;
    return true;
}

void BitHub::publishData() {
    if (!fmcw_lio_ptr_->isInitialized()) return;
    const auto state_ptr = fmcw_lio_ptr_->getState();
    const Eigen::Quaterniond q(state_ptr->getRotation());
    const Eigen::Vector3d p = state_ptr->getPosition();
    if (poses_file_.is_open()) {
        poses_file_ << std::fixed << std::setprecision(6) << state_ptr->getTime() << " "
                    << std::setprecision(9) << p.x() << " " << p.y() << " " << p.z() << " "
                    << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << "\n";
    }
    if (stats_file_.is_open()) {
        const auto v = fmcw_lio_ptr_->getVelocityEstimated();
        const auto vlog = fmcw_lio_ptr_->getLoggerVelocimetry();
        stats_file_ << std::fixed << std::setprecision(6) << state_ptr->getTime() << ","
                    << fmcw_lio_ptr_->getTimeProcessing() << ","
                    << fmcw_lio_ptr_->getScanDownSampledInLiDAR()->size() << ","
                    << fmcw_lio_ptr_->getScanDynamicInLiDAR()->size() << ","
                    << int(vlog->success) << "," << v.x() << "," << v.y() << "," << v.z() << "\n";
    }
    if (config_ptr_->dump_map) {
        *map_wait_dump_ptr_ += *fmcw_lio_ptr_->getScanDownSampledInWorld();
    }
}

void BitHub::dumpMap() {
    if (map_wait_dump_ptr_->empty() || out_dir_.empty()) return;
    PointCloudType::Ptr down(new PointCloudType());
    voxel_filter_map_.setInputCloud(map_wait_dump_ptr_);
    voxel_filter_map_.filter(*down);
    pcl::io::savePCDFileBinary(out_dir_ + "/map.pcd", *down);
}

}  // namespace fmcw_lio
