// Offline FMCW-LIO runner: reads a simulator sequence (imu.bin + lidar.bin),
// runs the unmodified FMCW-LIO core, writes TUM poses.
//   fmcw_lio_offline <config.yaml> <seq_dir> <out_dir> [section/key=value ...]
//   fmcw_lio_offline <config.yaml> -        <out_dir> [...]   stream mode (stdin/stdout)
//
// Stream mode (closed-loop simulation): binary records on stdin
//   'I' t wx wy wz ax ay az (7 x f64)        IMU
//   'G' t vx vy vz sigma n_contact (6 x f64) leg-odometry body velocity
//   'L' t_begin(f64) n(u32) n x RawPoint     LiDAR scan
//   'E'                                      end
// and one text line per processed scan on stdout:
//   P t x y z qx qy qz qw sigma_pos_max sigma_yaw_deg doppler_ok
#include <cstdio>
#include <iostream>

#include "system/fmcw_lio.hpp"

using fmcw_lio::RawPoint;

namespace {

struct ImuRec { double t, wx, wy, wz, ax, ay, az; };

std::vector<ImuRec> readImu(const std::string& path) {
    std::vector<ImuRec> out;
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) { std::cerr << "cannot open " << path << "\n"; std::exit(1); }
    ImuRec r{};
    while (std::fread(&r, sizeof(ImuRec), 1, f) == 1) out.push_back(r);
    std::fclose(f);
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "usage: " << argv[0] << " config.yaml seq_dir out_dir [key=value ...]\n";
        return 1;
    }
    const std::string seq_dir = argv[2], out_dir = argv[3];
    ros::NodeHandle nh(argv[1]);
    for (int i = 4; i < argc; ++i) {
        std::string kv = argv[i];
        auto eq = kv.find('=');
        if (eq != std::string::npos) nh.set(kv.substr(0, eq), kv.substr(eq + 1));
    }

    auto config_ptr = std::make_shared<fmcw_lio::Config>(nh);
    auto sys = std::make_shared<fmcw_lio::FMCWLIO>(nh, config_ptr);
    auto& hub = *sys->bithubForOffline();
    hub.openOutput(out_dir);

    // optional leg odometry fusion
    fmcw_lio::LegParams leg_params;
    nh.param<bool>("leg/use", leg_params.enabled, false);
    nh.param<double>("leg/window", leg_params.window, 0.05);
    nh.param<double>("leg/chi2_gate", leg_params.chi2_gate, 11.34);
    nh.param<bool>("leg/gate_with_doppler", leg_params.gate_with_doppler, true);
    nh.param<double>("leg/sigma_floor", leg_params.sigma_floor, 0.02);
    std::ofstream leg_log;
    if (leg_params.enabled && seq_dir == "-") {
        sys->enableLeg(leg_params);
    } else if (leg_params.enabled) {
        std::vector<fmcw_lio::LegVel> legs;
        FILE* f = std::fopen((seq_dir + "/legvel.bin").c_str(), "rb");
        if (!f) { std::cerr << "leg/use=true but no legvel.bin\n"; return 1; }
        double r[6];
        while (std::fread(r, sizeof(double), 6, f) == 6) {
            legs.push_back({r[0], Eigen::Vector3d(r[1], r[2], r[3]), r[4], r[5]});
        }
        std::fclose(f);
        sys->setLegVelocities(std::move(legs), leg_params);
        leg_log.open(out_dir + "/leg_updates.csv");
        leg_log << "t,n,m2,vbx,vby,vbz,accepted,gate_src\n";
        sys->setLegLog(&leg_log);
    }

    if (seq_dir == "-") {
        // ---------------- stream mode
        if (leg_params.enabled) {
            // leg velocities arrive on stdin instead of legvel.bin (set above would have failed)
        }
        auto emit = [&]() {
            const auto st = sys->getState();
            const Eigen::Quaterniond q(st->getRotation());
            const Eigen::Vector3d p = st->getPosition();
            const Eigen::MatrixXd& P = sys->getCovariance();
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(P.block<3, 3>(6, 6));
            const Eigen::Matrix3d Pth_w = st->getRotation() * P.block<3, 3>(0, 0) * st->getRotation().transpose();
            std::printf("P %.6f %.6f %.6f %.6f %.7f %.7f %.7f %.7f %.6f %.4f %d\n", st->getTime(), p.x(), p.y(), p.z(),
                        q.x(), q.y(), q.z(), q.w(), std::sqrt(std::max(0.0, es.eigenvalues().maxCoeff())),
                        std::sqrt(std::max(0.0, Pth_w(2, 2))) * 180.0 / M_PI,
                        int(sys->getLoggerVelocimetry()->success));
            std::fflush(stdout);
        };
        double last_t = -1.0;
        auto evolve = [&]() {
            for (int k = 0; k < 3; ++k) {
                sys->evolveSystem();
                const double t = sys->getState()->getTime();
                if (sys->isInitialized() && t != last_t) { last_t = t; emit(); }
            }
        };
        std::vector<RawPoint> spts;
        for (;;) {
            int c = std::fgetc(stdin);
            if (c == EOF || c == 'E') break;
            if (c == 'I') {
                double r[7];
                if (std::fread(r, sizeof(double), 7, stdin) != 7) break;
                auto m = std::make_shared<sensor_msgs::Imu>();
                m->header.stamp = ros::Time(r[0]);
                m->angular_velocity = {r[1], r[2], r[3]};
                m->linear_acceleration = {r[4], r[5], r[6]};
                hub.feedIMU(m);
                evolve();
            } else if (c == 'G') {
                double r[6];
                if (std::fread(r, sizeof(double), 6, stdin) != 6) break;
                sys->addLegVelocity({r[0], Eigen::Vector3d(r[1], r[2], r[3]), r[4], r[5]});
            } else if (c == 'L') {
                double tb; std::uint32_t nn;
                if (std::fread(&tb, sizeof(double), 1, stdin) != 1 || std::fread(&nn, 4, 1, stdin) != 1) break;
                spts.resize(nn);
                if (std::fread(spts.data(), sizeof(RawPoint), nn, stdin) != nn) break;
                hub.feedScan(tb, spts);
                evolve();
            } else {
                std::cerr << "bad record " << c << "\n";
                return 2;
            }
        }
        return 0;
    }

    const auto imu = readImu(seq_dir + "/imu.bin");
    FILE* lf = std::fopen((seq_dir + "/lidar.bin").c_str(), "rb");
    if (!lf) { std::cerr << "cannot open lidar.bin\n"; return 1; }

    std::size_t imu_i = 0, n_scans = 0;
    double t_beg;
    std::uint32_t n;
    std::vector<RawPoint> pts;
    auto wall0 = std::chrono::steady_clock::now();
    while (std::fread(&t_beg, sizeof(double), 1, lf) == 1 && std::fread(&n, sizeof(n), 1, lf) == 1) {
        pts.resize(n);
        if (std::fread(pts.data(), sizeof(RawPoint), n, lf) != n) break;
        const double t_end = t_beg + (n ? pts.back().t_off : 0.0);
        // feed IMU up to a little past the scan end (FMCW-LIO needs one sample beyond)
        while (imu_i < imu.size() && imu[imu_i].t <= t_end + 0.02) {
            auto m = std::make_shared<sensor_msgs::Imu>();
            const auto& r = imu[imu_i++];
            m->header.stamp = ros::Time(r.t);
            m->angular_velocity = {r.wx, r.wy, r.wz};
            m->linear_acceleration = {r.ax, r.ay, r.az};
            hub.feedIMU(m);
        }
        hub.feedScan(t_beg, pts);
        sys->evolveSystem();
        if (std::getenv("FMCW_DEBUG") && n_scans < 14) {
            auto c = sys->getScanFullInLiDAR();
            std::cerr << "scan " << n_scans << " raw=" << pts.size() << " comp=" << c->size();
            if (!c->empty()) std::cerr << " p0=" << c->points[0].x << "," << c->points[0].y << "," << c->points[0].z << " w=" << c->width << " h=" << c->height;
            std::cerr << " down=" << sys->getScanDownSampledInLiDAR()->size() << " init=" << sys->isInitialized()
                      << " t=" << sys->getState()->getTime() << " p=" << sys->getState()->getPosition().transpose()
                      << " v=" << sys->getState()->getVelocity().transpose() << std::endl;
        }
        ++n_scans;
    }
    std::fclose(lf);
    // drain
    for (int k = 0; k < 4; ++k) sys->evolveSystem();
    if (config_ptr->dump_map) sys->dumpMap();
    const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - wall0).count();
    std::cout << "scans=" << n_scans << " initialized=" << sys->isInitialized()
              << " wall_s=" << wall << " out=" << out_dir << std::endl;
    return 0;
}
