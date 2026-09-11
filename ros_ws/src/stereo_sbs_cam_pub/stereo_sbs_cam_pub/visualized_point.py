#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import threading
import numpy as np
import matplotlib.pyplot as plt
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray

POINT_NAMES = ["wrist", "thumb", "index", "middle", "ring", "pinky"]

def set_axes_equal(ax):
    """3D plot 비율 유지"""
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])

    plot_radius = 0.5 * max([x_range, y_range, z_range])

    x_mid = np.mean(x_limits)
    y_mid = np.mean(y_limits)
    z_mid = np.mean(z_limits)

    ax.set_xlim3d([x_mid - plot_radius, x_mid + plot_radius])
    ax.set_ylim3d([y_mid - plot_radius, y_mid + plot_radius])
    ax.set_zlim3d([z_mid - plot_radius, z_mid + plot_radius])

class TwoHandsPoseViewer(Node):
    def __init__(self, left_topic, right_topic):
        super().__init__("two_hands_pose_viewer")
        self._lock = threading.Lock()
        self._data = {
            "left": {"points": None, "frame_id": ""},
            "right": {"points": None, "frame_id": ""},
        }
        self.create_subscription(PoseArray, left_topic, lambda m: self._cb(m, "left"), 10)
        self.create_subscription(PoseArray, right_topic, lambda m: self._cb(m, "right"), 10)
        self.get_logger().info("Ready. Visualizing World Coordinates (Y-Up to Z-Up converted).")

    def _cb(self, msg, hand):
        pts = []
        for p in msg.poses:
            # [중요] Unity/OpenXR(Y-Up) -> Matplotlib(Z-Up) 시각화 변환
            # Unity World: x=Right, y=Up, z=Forward (OpenXR 변환된 상태라면 -z가 Forward)
            # Matplotlib View: x=x, y=z(Forward), z=y(Height) 로 매핑해야 서있는 것처럼 보임
            pts.append([p.position.x, -p.position.z, p.position.y])
            
        with self._lock:
            self._data[hand]["points"] = np.array(pts) if pts else None
            self._data[hand]["frame_id"] = msg.header.frame_id

    def get_latest(self):
        with self._lock:
            return {k: v.copy() for k, v in self._data.items()}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--left_topic", default="/left_hand/poses")
    parser.add_argument("--right_topic", default="/right_hand/poses")
    args = parser.parse_args()

    rclpy.init()
    node = TwoHandsPoseViewer(args.left_topic, args.right_topic)

    plt.ion()
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # 초기 뷰 설정 (월드 좌표 기준 넓게)
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(0.0, 2.0)  # Forward 방향 (Unity Z)
    ax.set_zlim(-0.5, 1.5) # Height 방향 (Unity Y)
    
    ax.set_xlabel("X (Right)")
    ax.set_ylabel("Y (Forward/Depth)") # Matplotlib Y축을 깊이로 사용
    ax.set_zlabel("Z (Height)")        # Matplotlib Z축을 높이로 사용
    
    ax.view_init(elev=20, azim=-45) # 보기 좋은 각도

    # 플롯 객체 초기화
    scats = {
        "left": ax.scatter([], [], [], c="blue", label="Left"),
        "right": ax.scatter([], [], [], c="red", label="Right")
    }
    lines = {"left": [], "right": []}
    for key, color in [("left", "blue"), ("right", "red")]:
        for _ in range(5): # 손가락 5개 선
            (ln,) = ax.plot([], [], [], color=color, linewidth=2)
            lines[key].append(ln)

    ax.legend()

    try:
        while rclpy.ok() and plt.fignum_exists(fig.number):
            rclpy.spin_once(node, timeout_sec=0.01)
            data = node.get_latest()

            for hand in ["left", "right"]:
                pts = data[hand]["points"]
                
                # 점 업데이트
                if pts is not None and len(pts) > 0:
                    scats[hand]._offsets3d = (pts[:,0], pts[:,1], pts[:,2])
                    
                    # 선 업데이트 (Wrist -> Tips)
                    wrist = pts[0]
                    for i in range(1, 6): # 1~5번 팁
                        if i < len(pts):
                            tip = pts[i]
                            ln = lines[hand][i-1]
                            ln.set_data([wrist[0], tip[0]], [wrist[1], tip[1]])
                            ln.set_3d_properties([wrist[2], tip[2]])
                else:
                    # 데이터 없으면 숨김
                    scats[hand]._offsets3d = ([],[],[])
                    for ln in lines[hand]:
                        ln.set_data([], [])
                        ln.set_3d_properties([])

            # Auto Scale (필요시 주석 해제)
            # set_axes_equal(ax) 
            
            fig.canvas.draw_idle()
            plt.pause(0.01)

    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()