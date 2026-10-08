"""Export a simulator sequence to a ROS 2 bag (no ROS install needed: rosbags).

Topics
  /aeva/point_cloud   sensor_msgs/PointCloud2  fields x y z velocity(f32) time_offset_ns(i32)
                      (Aeva Aeries II layout, what FMCW-LIO's ROS driver path expects)
  /imu/data           sensor_msgs/Imu
  /joint_states       sensor_msgs/JointState   12 leg joints, position + velocity
  /ground_truth/odom  nav_msgs/Odometry        IMU frame in world
  /tf_static          base_link -> aeva (from meta.json extrinsics)

  python3 tools/to_ros2bag.py data/slab bags/slab
  ros2 bag play bags/slab   # then rviz2, or any ROS 2 LIO node
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from rosbags.rosbag2 import StoragePlugin, Writer
from rosbags.typesys import Stores, get_typestore

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.seqio import iter_scans, read_imu, read_meta  # noqa: E402

JOINTS = [f"{s}_{j}_joint" for s in ("left", "right")
          for j in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")]


def export(seq: Path, out: Path):
    ts = get_typestore(Stores.ROS2_HUMBLE)
    M = ts.types
    Header, Time = M["std_msgs/msg/Header"], M["builtin_interfaces/msg/Time"]
    PointField, PC2 = M["sensor_msgs/msg/PointField"], M["sensor_msgs/msg/PointCloud2"]
    Imu, JS, Odom = M["sensor_msgs/msg/Imu"], M["sensor_msgs/msg/JointState"], M["nav_msgs/msg/Odometry"]
    V3, Q, Pose, PoseC = M["geometry_msgs/msg/Vector3"], M["geometry_msgs/msg/Quaternion"], \
        M["geometry_msgs/msg/Pose"], M["geometry_msgs/msg/PoseWithCovariance"]
    Twist, TwistC, Point = M["geometry_msgs/msg/Twist"], M["geometry_msgs/msg/TwistWithCovariance"], M["geometry_msgs/msg/Point"]
    TFm, TS, Tr = M["tf2_msgs/msg/TFMessage"], M["geometry_msgs/msg/TransformStamped"], M["geometry_msgs/msg/Transform"]

    def stamp(t):
        s = int(np.floor(t))
        return Time(sec=s, nanosec=int(round((t - s) * 1e9)) % 1_000_000_000)

    def hdr(t, frame):
        return Header(stamp=stamp(t), frame_id=frame)

    def ns(t):
        return int(round(t * 1e9)) + 1  # bag time > 0

    meta = read_meta(seq)
    imu = read_imu(seq)
    legs = dict(np.load(seq / "legs.npz"))   # NpzFile re-decompresses on every key access
    gt = dict(np.load(seq / "gt.npz"))
    fields = [PointField(name="x", offset=0, datatype=7, count=1), PointField(name="y", offset=4, datatype=7, count=1),
              PointField(name="z", offset=8, datatype=7, count=1), PointField(name="velocity", offset=12, datatype=7, count=1),
              PointField(name="time_offset_ns", offset=16, datatype=5, count=1)]
    pt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("velocity", "<f4"), ("time_offset_ns", "<i4")])
    zero9 = np.zeros(9)
    zero36 = np.zeros(36)

    if out.exists():
        raise SystemExit(f"{out} exists")
    # MCAP: the sqlite3 backend commits per message and is ~100x slower here
    with Writer(out, version=8, storage_plugin=StoragePlugin.MCAP) as w:
        c_pc = w.add_connection("/aeva/point_cloud", PC2.__msgtype__, typestore=ts)
        c_imu = w.add_connection("/imu/data", Imu.__msgtype__, typestore=ts)
        c_js = w.add_connection("/joint_states", JS.__msgtype__, typestore=ts)
        c_od = w.add_connection("/ground_truth/odom", Odom.__msgtype__, typestore=ts)
        c_tf = w.add_connection("/tf_static", TFm.__msgtype__, typestore=ts)

        R_bl = np.array(meta["R_bl"]).reshape(3, 3)
        import mujoco
        qbl = np.empty(4)
        mujoco.mju_mat2Quat(qbl, R_bl.reshape(-1).copy())
        p = meta["p_bl_b"]
        tf = TFm(transforms=[TS(header=hdr(0.0, "base_link"), child_frame_id="aeva",
                                transform=Tr(translation=V3(x=p[0], y=p[1], z=p[2]),
                                             rotation=Q(x=qbl[1], y=qbl[2], z=qbl[3], w=qbl[0])))])
        w.write(c_tf, 1, ts.serialize_cdr(tf, TFm.__msgtype__))

        for r in imu:
            m = Imu(header=hdr(r[0], "base_link"), orientation=Q(x=0.0, y=0.0, z=0.0, w=1.0),
                    orientation_covariance=np.array([-1.0] + [0.0] * 8),
                    angular_velocity=V3(x=r[1], y=r[2], z=r[3]), angular_velocity_covariance=zero9,
                    linear_acceleration=V3(x=r[4], y=r[5], z=r[6]), linear_acceleration_covariance=zero9)
            w.write(c_imu, ns(r[0]), ts.serialize_cdr(m, Imu.__msgtype__))
        for i, t in enumerate(legs["t"]):
            m = JS(header=hdr(t, "base_link"), name=JOINTS, position=legs["q"][i].astype(np.float64),
                   velocity=legs["dq"][i].astype(np.float64), effort=np.zeros(0))
            w.write(c_js, ns(t), ts.serialize_cdr(m, JS.__msgtype__))
        for i in range(0, len(gt["t"]), 5):     # 50 Hz
            t, pp, q, v = gt["t"][i], gt["p"][i], gt["q"][i], gt["v"][i]
            m = Odom(header=hdr(t, "world"), child_frame_id="base_link",
                     pose=PoseC(pose=Pose(position=Point(x=pp[0], y=pp[1], z=pp[2]),
                                          orientation=Q(x=q[1], y=q[2], z=q[3], w=q[0])), covariance=zero36),
                     twist=TwistC(twist=Twist(linear=V3(x=v[0], y=v[1], z=v[2]), angular=V3(x=0.0, y=0.0, z=0.0)),
                                  covariance=zero36))
            w.write(c_od, ns(t), ts.serialize_cdr(m, Odom.__msgtype__))
        n = 0
        for t0, s in iter_scans(seq):
            a = np.empty(len(s), pt)
            a["x"], a["y"], a["z"], a["velocity"] = s["x"], s["y"], s["z"], s["doppler"]
            a["time_offset_ns"] = np.round(s["t_off"] * 1e9).astype(np.int32)
            m = PC2(header=hdr(t0, "aeva"), height=1, width=len(a), fields=fields, is_bigendian=False,
                    point_step=pt.itemsize, row_step=pt.itemsize * len(a),
                    data=np.frombuffer(a.tobytes(), np.uint8), is_dense=True)
            w.write(c_pc, ns(t0 + 0.1), ts.serialize_cdr(m, PC2.__msgtype__))
            n += 1
    print(f"wrote {out}: {n} scans, {len(imu)} imu, {len(legs['t'])} joint_states")


if __name__ == "__main__":
    export(Path(sys.argv[1]), Path(sys.argv[2]))
