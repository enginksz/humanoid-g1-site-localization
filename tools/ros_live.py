"""ROS 2 publisher for live runs (sim/live_view.py --ros). Needs /opt/ros/humble sourced.

Topics (frame "map" = simulation world):
  /tf                   map -> base_link (estimate), map -> base_link_gt (ground truth)
  /lidar/points         sensor_msgs/PointCloud2 in map frame, fields x y z doppler
  /path/estimate        nav_msgs/Path
  /path/ground_truth    nav_msgs/Path
  /obstacles            visualization_msgs/MarkerArray  (GT costmap cells)
  /local_costmap        nav_msgs/OccupancyGrid          (ego GT free-space)
  /prior_map            sensor_msgs/PointCloud2 (transient local) when a prior map is used
View with:  rviz2 -d configs/live.rviz
"""
from __future__ import annotations

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import OccupancyGrid, Path as PathMsg
from rclpy.qos import DurabilityPolicy, QoSProfile
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import ColorRGBA, Header
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray


def _quat(R):
    import mujoco
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(R).reshape(-1))
    return q  # w x y z


class RosLive:
    def __init__(self, map_path=None):
        rclpy.init()
        self.node = rclpy.create_node("g1_live")
        self.tf = TransformBroadcaster(self.node)
        self.pub_pts = self.node.create_publisher(PointCloud2, "/lidar/points", 5)
        self.pub_est = self.node.create_publisher(PathMsg, "/path/estimate", 5)
        self.pub_gt = self.node.create_publisher(PathMsg, "/path/ground_truth", 5)
        self.pub_obs = self.node.create_publisher(MarkerArray, "/obstacles", 5)
        self.pub_cmap = self.node.create_publisher(OccupancyGrid, "/local_costmap", 5)
        self.path_est, self.path_gt = PathMsg(), PathMsg()
        self.path_est.header.frame_id = self.path_gt.header.frame_id = "map"
        self.n = 0
        if map_path:
            qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
            self.pub_map = self.node.create_publisher(PointCloud2, "/prior_map", qos)
            P = np.load(map_path)["points"]
            P = P[(P[:, 2] > -0.2) & (P[:, 2] < 4.0)][::2]
            self.pub_map.publish(self._cloud(np.c_[P, np.zeros(len(P))], 0.0))

    def _stamp(self, t):
        from builtin_interfaces.msg import Time
        return Time(sec=int(t), nanosec=int((t - int(t)) * 1e9))

    def _cloud(self, P4, t):
        m = PointCloud2()
        m.header.frame_id = "map"
        m.header.stamp = self._stamp(t)
        m.height, m.width = 1, len(P4)
        m.fields = [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1)
                    for i, n in enumerate(["x", "y", "z", "doppler"])]
        m.point_step, m.row_step = 16, 16 * len(P4)
        m.is_dense = True
        m.data = np.ascontiguousarray(P4, np.float32).tobytes()
        return m

    def _tf(self, T, child, t):
        msg = TransformStamped()
        msg.header.frame_id, msg.child_frame_id = "map", child
        msg.header.stamp = self._stamp(t)
        msg.transform.translation.x, msg.transform.translation.y, msg.transform.translation.z = map(float, T[:3, 3])
        q = _quat(T[:3, :3])
        msg.transform.rotation.w, msg.transform.rotation.x, msg.transform.rotation.y, msg.transform.rotation.z = map(float, q)
        return msg

    def _pose(self, T, t):
        p = PoseStamped()
        p.header.frame_id = "map"
        p.header.stamp = self._stamp(t)
        p.pose.position.x, p.pose.position.y, p.pose.position.z = map(float, T[:3, 3])
        q = _quat(T[:3, :3])
        p.pose.orientation.w, p.pose.orientation.x, p.pose.orientation.y, p.pose.orientation.z = map(float, q)
        return p

    def _obstacles(self, costmap, t, z=0.15):
        """CUBE markers on high-cost GT free-space cells."""
        arr = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        arr.markers.append(clear)
        if costmap is None:
            return arr
        m = Marker()
        m.header = Header(frame_id="map", stamp=self._stamp(t))
        m.ns, m.id = "gt_obstacles", 0
        m.type, m.action = Marker.CUBE_LIST, Marker.ADD
        m.scale.x = m.scale.y = m.scale.z = float(costmap.cfg.res) * 0.85
        m.pose.orientation.w = 1.0
        from geometry_msgs.msg import Point
        for x, y, c in costmap.world_cells(max_cost=0.25):
            if len(m.points) >= 800:
                break
            m.points.append(Point(x=float(x), y=float(y), z=z))
            if c > 0.8:
                m.colors.append(ColorRGBA(r=0.95, g=0.15, b=0.1, a=0.85))
            else:
                m.colors.append(ColorRGBA(r=0.95, g=0.7, b=0.1, a=0.65))
        arr.markers.append(m)
        return arr

    def _occupancy(self, costmap, t):
        if costmap is None:
            return None
        g = OccupancyGrid()
        g.header = Header(frame_id="map", stamp=self._stamp(t))
        n = costmap.n
        res = float(costmap.cfg.res)
        g.info.resolution = res
        g.info.width = g.info.height = n
        g.info.origin.position.x = float(costmap.origin[0])
        g.info.origin.position.y = float(costmap.origin[1])
        g.info.origin.orientation.w = 1.0
        # OccupancyGrid is row-major, y then x; our cost is [i=x, j=y]
        cost = costmap.cost
        flat = np.empty(n * n, np.int8)
        for j in range(n):
            for i in range(n):
                c = float(cost[i, j])
                if c < 0.15:
                    v = 0
                elif c < 0.5:
                    v = 50
                else:
                    v = 100
                flat[j * n + i] = v
        g.data = flat.tolist()
        return g

    def publish(self, t, T_gt, T_est, pts_world_dop, costmap=None):
        self.n += 1
        tfs = [self._tf(T_gt, "base_link_gt", t)]
        if T_est is not None:
            tfs.append(self._tf(T_est, "base_link", t))
        self.tf.sendTransform(tfs)
        self.pub_pts.publish(self._cloud(pts_world_dop, t))
        if self.n % 2 == 0:
            self.pub_obs.publish(self._obstacles(costmap, t))
            og = self._occupancy(costmap, t)
            if og is not None:
                self.pub_cmap.publish(og)
        if self.n % 5 == 0:
            self.path_gt.poses.append(self._pose(T_gt, t))
            self.path_gt.header.stamp = self._stamp(t)
            self.pub_gt.publish(self.path_gt)
            if T_est is not None:
                self.path_est.poses.append(self._pose(T_est, t))
                self.path_est.header.stamp = self._stamp(t)
                self.pub_est.publish(self.path_est)
        rclpy.spin_once(self.node, timeout_sec=0.0)

    def shutdown(self):
        self.node.destroy_node()
        rclpy.shutdown()
