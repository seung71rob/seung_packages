#!/usr/bin/env python3

import math
import os
import sys
from pathlib import Path
from collections import deque

# ============================================================
# Python 패키지 추가 검색 경로 (선택)
#
# ultralytics 등을 시스템이 아닌 별도 폴더에 설치해 둔 경우에만 쓴다.
# 기본값은 없음. 필요하면 실행 전에 지정한다.
#   export VISION_PYTHON_PKGS=/원하는/경로/python_pkgs
# 여러 경로는 콜론(:)으로 구분한다.
# ============================================================
for _pkg_dir in os.environ.get("VISION_PYTHON_PKGS", "").split(os.pathsep):
    _pkg_dir = _pkg_dir.strip()
    if _pkg_dir and os.path.isdir(_pkg_dir) and _pkg_dir not in sys.path:
        sys.path.insert(0, _pkg_dir)

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String, Float64MultiArray
from cv_bridge import CvBridge
from ament_index_python.packages import get_package_share_directory

try:
    from ultralytics import YOLO
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "ultralytics 를 불러오지 못했습니다.\n"
        "  설치:  pip install ultralytics\n"
        "  별도 폴더에 설치했다면:  export VISION_PYTHON_PKGS=/그/폴더"
    ) from exc
A3_W_MM = 420.0
A3_H_MM = 297.0

PACKAGE_NAME = 'vision_pkg'
MODEL_FILENAME = 'best_Baseplate_hole.pt'


def find_default_model_path():
    """
    YOLO 모델 파일을 찾는다. 아래 순서로 탐색한다.

      1) 환경변수 BASEPLATE_HOLE_MODEL_PATH
      2) ROS 패키지 vision_pkg/pt/  (설치되어 있을 때만)
      3) 스크립트와 같은 폴더, 그 아래 pt/,
         현재 작업 디렉토리, 그 아래 pt/
      4) 못 찾으면 빈 문자열

    ★ 이 함수는 declare_parameter 의 기본값을 만드는 중에 호출된다.
      여기서 예외가 나면 노드 자체가 뜨지 못하므로, 패키지 조회 실패는
      반드시 삼켜야 한다. (vision_pkg 가 없는 환경에서 실행하면
      PackageNotFoundError 로 죽던 문제)
    """
    names = (MODEL_FILENAME, 'Baseplate_hole.pt', 'baseplate_hole.pt')

    env_path = os.environ.get('BASEPLATE_HOLE_MODEL_PATH', '').strip()
    if env_path:
        return str(Path(env_path).expanduser())

    try:
        package_share = Path(get_package_share_directory(PACKAGE_NAME))
        for name in names:
            cand = package_share / 'pt' / name
            if cand.exists():
                return str(cand)
    except Exception:
        pass

    here = Path(__file__).resolve().parent
    for base in (here, here / 'pt', Path.cwd(), Path.cwd() / 'pt'):
        for name in names:
            cand = base / name
            if cand.exists():
                return str(cand)

    return ''

BASE_LABELS = {'baseplate', 'base_plate'}
DOT_LABELS = {'dot'}
WINDOW_NAME = 'YOLO + OpenCV Precision Hole Detection'
BINARY_WINDOW_NAME = 'Hole Radial Debug'
DEBUG_ROI_SIZE = 210

class PrecisionHoleDetector(Node):

    def __init__(self):
        super().__init__('baseplate_hole')
        self.bridge = CvBridge()

        # ROS parameters: 탑뷰 코드와 비슷하게 파일 경로/토픽을 실행 시 변경 가능
        self.declare_parameter('model_path', find_default_model_path())
        self.declare_parameter('image_topic', 'image_raw')
        self.declare_parameter('conf', 0.25)
        self.declare_parameter('iou', 0.45)
        self.declare_parameter('holes_topic', '/detected_holes')
        self.declare_parameter('mid_topic', '/detected_hole_mid')
        self.declare_parameter('side_topic', '/detected_hole_side')

        self.model_path = str(self.get_parameter('model_path').value)
        self.image_topic = str(self.get_parameter('image_topic').value)
        self.conf_threshold = float(self.get_parameter('conf').value)
        self.iou_threshold = float(self.get_parameter('iou').value)
        self.holes_topic = str(self.get_parameter('holes_topic').value)
        self.mid_topic = str(self.get_parameter('mid_topic').value)
        self.side_topic = str(self.get_parameter('side_topic').value)

        if (not self.model_path) or (not Path(self.model_path).expanduser().exists()):
            raise FileNotFoundError(
                'YOLO 모델을 찾을 수 없습니다: %s\n'
                '  아래 중 하나로 지정하세요.\n'
                '    1) --ros-args -p model_path:="/절대/경로/%s"\n'
                '    2) export BASEPLATE_HOLE_MODEL_PATH=/절대/경로/%s\n'
                '    3) 이 스크립트와 같은 폴더(또는 그 아래 pt/)에 %s 두기'
                % (self.model_path or '(경로 미지정)',
                   MODEL_FILENAME, MODEL_FILENAME, MODEL_FILENAME))

        self.model = YOLO(self.model_path)
        self.subscription = self.create_subscription(
            Image, self.image_topic, self.listener_callback, 10
        )

        # STABLE 상태에서만 mm 좌표 publish
        # /detected_holes: Float64MultiArray
        # [mid_x_mm, mid_y_mm, side_x_mm, side_y_mm, angle_deg, distance_mm]
        self.holes_pub = self.create_publisher(Float64MultiArray, self.holes_topic, 10)
        # 개별 구멍은 기존 /detected_object와 비슷한 String 형식으로도 제공
        self.mid_pub = self.create_publisher(String, self.mid_topic, 10)
        self.side_pub = self.create_publisher(String, self.side_topic, 10)
        self.coordinates_published = False

        self.clicked_pts = []
        self.H = None
        self.h_ready = False
        self.MOVE_THRESHOLD_PX = 2.5
        self.MOVE_CONFIRM_FRAMES = 3
        self.STABLE_STEP_PX = 0.6
        self.STABLE_CONFIRM_FRAMES = 5
        self.MEASURE_FRAMES = 15
        self.BATCH_INLIER_RADIUS_PX = 0.8
        self.RADIAL_ANGLES = 360
        self.RADIAL_STEP_PX = 0.25
        self.RADIUS_MIN_RATIO = 0.1
        self.RADIUS_MAX_RATIO = 0.52
        self.PEAKS_PER_ANGLE = 4
        self.MIN_RADIAL_GRADIENT = 1.2
        self.GRAD_MAD_FACTOR = 2.0
        self.RADIUS_CLUSTER_RATIO = 0.075
        self.RADIUS_CLUSTER_MIN_PX = 5.0
        self.SECOND_PASS_BAND_RATIO = 0.1
        self.SECOND_PASS_BAND_MIN_PX = 4.0
        self.MAX_ELLIPSE_AXIS_RATIO = 1.8
        self.MAX_CENTER_SHIFT_RATIO = 0.28
        self.MIN_FIRST_PASS_COVERAGE = 0.42
        self.MIN_FINAL_COVERAGE = 0.38
        self.MAX_FIT_RESIDUAL_PX = 3.0
        self.MIN_FIT_POINTS = 40
        self.MIN_LOCAL_CONTRAST = 2.5
        # v5: YOLO bbox를 구멍 크기 prior로 사용하고, 내부 반사 영역은 탐색에서 제외한다.
        # v6: 저해상도(반경 5~8 px) 구멍에서도 annulus가 사라지지 않도록
        # bbox 자체보다 '예상 반경'을 기준으로 annulus 폭을 만든다.
        self.YOLO_RADIUS_PRIOR_RATIO = 0.46
        self.ANNULUS_INNER_RADIUS_RATIO = 0.62
        self.ANNULUS_OUTER_RADIUS_RATIO = 1.34
        self.ANNULUS_PRIOR_SIGMA_RATIO = 0.16
        self.CENTER_SEARCH_RATIO = 0.10
        self.MIN_ANNULUS_WIDTH_PX = 2.25
        self.MIN_RING_COVERAGE_V5 = 0.28
        self.track_state = 'MOVING'
        self.prev_raw_mid = None
        self.prev_raw_side = None
        self.stable_frame_count = 0
        self.move_frame_count = 0
        self.measure_mid = deque(maxlen=self.MEASURE_FRAMES)
        self.measure_side = deque(maxlen=self.MEASURE_FRAMES)
        self.locked_mid = None
        self.locked_side = None
        cv2.namedWindow(WINDOW_NAME)
        cv2.setMouseCallback(WINDOW_NAME, self.on_mouse)
        self.get_logger().info(f'Model path: {self.model_path}')
        self.get_logger().info(f'YOLO classes: {self.model.names}')
        self.get_logger().info(f'Image topic: {self.image_topic}')
        self.get_logger().info(f'Holes topic: {self.holes_topic}')
        self.get_logger().info(f'Mid topic: {self.mid_topic}')
        self.get_logger().info(f'Side topic: {self.side_topic}')
        self.get_logger().info('A3 좌상 -> 우상 -> 우하 -> 좌하 순서로 4점을 클릭하세요.')
        self.get_logger().info('좌표는 STABLE 상태가 되면 1회 publish됩니다.')
        self.get_logger().info('/detected_holes = [mid_x, mid_y, side_x, side_y, angle_deg, distance_mm]')

    @staticmethod
    def normalize_label(label):
        return str(label).strip().lower()

    def on_mouse(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if self.h_ready:
            return
        self.clicked_pts.append((x, y))
        self.get_logger().info(f'A3 click ({x}, {y}) {len(self.clicked_pts)}/4')
        if len(self.clicked_pts) == 4:
            src = np.array(self.clicked_pts, dtype=np.float32)
            dst = np.array([[0.0, 0.0], [A3_W_MM, 0.0], [A3_W_MM, A3_H_MM], [0.0, A3_H_MM]], dtype=np.float32)
            self.H = cv2.getPerspectiveTransform(src, dst)
            self.h_ready = True
            self.get_logger().info('A3 pixel -> mm 변환 준비 완료')

    def pixel_to_mm(self, px, py):
        if not self.h_ready or self.H is None:
            return (None, None)
        p = np.array([float(px), float(py), 1.0], dtype=np.float64)
        q = self.H.astype(np.float64) @ p
        if abs(q[2]) < 1e-12:
            return (None, None)
        return (float(q[0] / q[2]), float(q[1] / q[2]))

    def publish_hole_coordinates(self, mid_mx, mid_my, side_mx, side_my, angle_mm, distance_mm):
        """STABLE 상태에서 두 구멍의 A3 기준 mm 좌표를 1회 publish."""
        if self.coordinates_published:
            return

        arr = Float64MultiArray()
        arr.data = [
            float(mid_mx), float(mid_my),
            float(side_mx), float(side_my),
            float(angle_mm), float(distance_mm),
        ]
        self.holes_pub.publish(arr)

        mid_msg = String()
        mid_msg.data = f'hole_mid,{mid_mx:.3f},{mid_my:.3f},0.0,0,OK'
        self.mid_pub.publish(mid_msg)

        side_msg = String()
        side_msg.data = f'hole_side,{side_mx:.3f},{side_my:.3f},0.0,0,OK'
        self.side_pub.publish(side_msg)

        self.coordinates_published = True
        self.get_logger().info(
            '★ Hole mm 좌표 전송 | '
            f'mid=({mid_mx:.3f}, {mid_my:.3f}) mm | '
            f'side=({side_mx:.3f}, {side_my:.3f}) mm | '
            f'angle={angle_mm:.2f} deg | distance={distance_mm:.3f} mm'
        )

    def robust_batch(self, samples):
        if len(samples) == 0:
            return None
        xs = np.array([p[0] for p in samples], dtype=np.float64)
        ys = np.array([p[1] for p in samples], dtype=np.float64)
        med_x = float(np.median(xs))
        med_y = float(np.median(ys))
        dist = np.sqrt((xs - med_x) ** 2 + (ys - med_y) ** 2)
        mask = dist <= self.BATCH_INLIER_RADIUS_PX
        if np.count_nonzero(mask) >= 3:
            return (float(np.mean(xs[mask])), float(np.mean(ys[mask])))
        return (med_x, med_y)

    def reset_tracking_state(self):
        self.track_state = 'MOVING'
        self.prev_raw_mid = None
        self.prev_raw_side = None
        self.stable_frame_count = 0
        self.move_frame_count = 0
        self.measure_mid.clear()
        self.measure_side.clear()
        self.locked_mid = None
        self.locked_side = None
        self.coordinates_published = False

    def update_tracking_state(self, raw_mid, raw_side):
        raw_mid = (float(raw_mid[0]), float(raw_mid[1]))
        raw_side = (float(raw_side[0]), float(raw_side[1]))
        if self.prev_raw_mid is None or self.prev_raw_side is None:
            self.prev_raw_mid = raw_mid
            self.prev_raw_side = raw_side
            return (raw_mid, raw_side, self.track_state, 0)
        mid_step = math.hypot(raw_mid[0] - self.prev_raw_mid[0], raw_mid[1] - self.prev_raw_mid[1])
        side_step = math.hypot(raw_side[0] - self.prev_raw_side[0], raw_side[1] - self.prev_raw_side[1])
        frame_step = max(mid_step, side_step)
        self.prev_raw_mid = raw_mid
        self.prev_raw_side = raw_side
        if self.track_state == 'STABLE':
            mid_from_lock = math.hypot(raw_mid[0] - self.locked_mid[0], raw_mid[1] - self.locked_mid[1])
            side_from_lock = math.hypot(raw_side[0] - self.locked_side[0], raw_side[1] - self.locked_side[1])
            moved_from_lock = max(mid_from_lock, side_from_lock)
            if moved_from_lock > self.MOVE_THRESHOLD_PX:
                self.move_frame_count += 1
            else:
                self.move_frame_count = 0
            if self.move_frame_count >= self.MOVE_CONFIRM_FRAMES:
                self.track_state = 'MOVING'
                self.stable_frame_count = 0
                self.move_frame_count = 0
                self.measure_mid.clear()
                self.measure_side.clear()
                self.locked_mid = None
                self.locked_side = None
                self.coordinates_published = False
                return (raw_mid, raw_side, self.track_state, 0)
            return (self.locked_mid, self.locked_side, self.track_state, self.MEASURE_FRAMES)
        if self.track_state == 'MOVING':
            if frame_step <= self.STABLE_STEP_PX:
                self.stable_frame_count += 1
            else:
                self.stable_frame_count = 0
            if self.stable_frame_count >= self.STABLE_CONFIRM_FRAMES:
                self.track_state = 'MEASURING'
                self.measure_mid.clear()
                self.measure_side.clear()
                self.measure_mid.append(raw_mid)
                self.measure_side.append(raw_side)
                self.move_frame_count = 0
                return (raw_mid, raw_side, self.track_state, 1)
            return (raw_mid, raw_side, self.track_state, 0)
        if self.track_state == 'MEASURING':
            if frame_step > self.MOVE_THRESHOLD_PX:
                self.move_frame_count += 1
            else:
                self.move_frame_count = 0
            if self.move_frame_count >= self.MOVE_CONFIRM_FRAMES:
                self.track_state = 'MOVING'
                self.stable_frame_count = 0
                self.move_frame_count = 0
                self.measure_mid.clear()
                self.measure_side.clear()
                return (raw_mid, raw_side, self.track_state, 0)
            self.measure_mid.append(raw_mid)
            self.measure_side.append(raw_side)
            current_mid = self.robust_batch(self.measure_mid)
            current_side = self.robust_batch(self.measure_side)
            sample_count = min(len(self.measure_mid), len(self.measure_side))
            if sample_count >= self.MEASURE_FRAMES:
                self.locked_mid = self.robust_batch(self.measure_mid)
                self.locked_side = self.robust_batch(self.measure_side)
                self.track_state = 'STABLE'
                self.move_frame_count = 0
                return (self.locked_mid, self.locked_side, self.track_state, self.MEASURE_FRAMES)
            return (current_mid, current_side, self.track_state, sample_count)
        self.reset_tracking_state()
        return (raw_mid, raw_side, self.track_state, 0)

    def local_illumination_normalize(self, gray, box_scale):
        """조명 방향/세기 변화의 저주파 성분을 줄이는 Log-Retinex 정규화.

        v5에서는 이 영상의 '밝기 자체'로 구멍을 자르지 않는다.
        외곽 annulus의 gradient 위치만 찾기 때문에 내부 바닥 반사가 밝아져도
        중심 계산에 직접 들어가지 않는다.
        """
        gray_f = gray.astype(np.float32) + 1.0

        # 구멍보다 훨씬 큰 스케일로 illumination field를 추정한다.
        sigma = max(12.0, float(box_scale) * 0.65)
        illum = cv2.GaussianBlur(gray_f, (0, 0), sigmaX=sigma, sigmaY=sigma)

        retinex = np.log(gray_f) - np.log(illum + 1e-6)

        # 고정 min/max 대신 robust 통계로 표시 범위를 잡는다.
        med = float(np.median(retinex))
        mad = float(np.median(np.abs(retinex - med)))
        robust_sigma = max(1.4826 * mad, 1e-3)
        norm = 128.0 + 34.0 * (retinex - med) / robust_sigma
        norm = np.clip(norm, 0.0, 255.0).astype(np.float32)

        # Subpixel gradient가 픽셀 노이즈에 반응하지 않을 정도만 약하게 평활화.
        norm = cv2.GaussianBlur(norm, (3, 3), 0.65)
        return norm

    @staticmethod
    def subpixel_peak_offset(y_prev, y0, y_next):
        """
        3점 포물선 보간으로 gradient maximum의 subpixel offset 계산.
        반환값은 현재 sample index 기준 -0.75 ~ +0.75.
        """
        denominator = float(y_prev) - 2.0 * float(y0) + float(y_next)
        if abs(denominator) < 1e-09:
            return 0.0
        delta = 0.5 * (float(y_prev) - float(y_next)) / denominator
        return float(np.clip(delta, -0.75, 0.75))

    def radial_profiles(self, norm, cx, cy, angles, radius_grid):
        """
        radius_grid:
          1D이면 모든 angle이 같은 반경 samples 사용.
          2D이면 angle별 서로 다른 반경 samples 사용.

        return:
          profiles, gradients, actual_radius_grid
        """
        cos_a = np.cos(angles).astype(np.float32)
        sin_a = np.sin(angles).astype(np.float32)
        if radius_grid.ndim == 1:
            rr = np.broadcast_to(radius_grid[None, :], (len(angles), len(radius_grid))).astype(np.float32)
        else:
            rr = radius_grid.astype(np.float32)
        map_x = (float(cx) + cos_a[:, None] * rr).astype(np.float32)
        map_y = (float(cy) + sin_a[:, None] * rr).astype(np.float32)
        profiles = cv2.remap(norm, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        profiles = cv2.GaussianBlur(profiles, (5, 1), 0.8)
        if rr.shape[1] >= 2:
            radial_step = float(np.median(rr[:, 1] - rr[:, 0]))
        else:
            radial_step = 1.0
        radial_step = max(abs(radial_step), 1e-06)
        gradients = np.gradient(profiles, radial_step, axis=1).astype(np.float32)
        return (profiles, gradients, rr)

    def collect_first_pass_candidates(self, norm, cx, cy, r_min, r_max, angles):
        radii = np.arange(r_min, r_max + self.RADIAL_STEP_PX * 0.5, self.RADIAL_STEP_PX, dtype=np.float32)
        if len(radii) < 7:
            return (None, None, None)
        profiles, gradients, rr = self.radial_profiles(norm, cx, cy, angles, radii)
        angle_candidates = []
        for angle_index in range(len(angles)):
            g = gradients[angle_index]
            if len(g) < 5:
                angle_candidates.append([])
                continue
            g_med = float(np.median(g))
            g_mad = float(np.median(np.abs(g - g_med)))
            adaptive_threshold = max(self.MIN_RADIAL_GRADIENT, g_med + self.GRAD_MAD_FACTOR * 1.4826 * g_mad)
            peak_idx = np.where((g[1:-1] > g[:-2]) & (g[1:-1] >= g[2:]) & (g[1:-1] >= adaptive_threshold))[0] + 1
            if len(peak_idx) == 0:
                angle_candidates.append([])
                continue
            order = peak_idx[np.argsort(g[peak_idx])[::-1]]
            order = order[:self.PEAKS_PER_ANGLE]
            current = []
            for k in order:
                delta = self.subpixel_peak_offset(g[k - 1], g[k], g[k + 1])
                step = float(rr[angle_index, k + 1] - rr[angle_index, k])
                r_sub = float(rr[angle_index, k] + delta * step)
                current.append({'radius': r_sub, 'strength': float(g[k])})
            angle_candidates.append(current)
        return (angle_candidates, profiles, gradients)

    def dominant_radius_consensus(self, angle_candidates, r_min, r_max, cluster_tolerance):
        """
        내부 초승달 반사나 plate 외곽 그림자는 일부 angle에서만 peak를 만든다.
        실제 hole 외곽은 대부분 angle에서 비슷한 반경에 peak가 생긴다.

        따라서 '강한 peak 하나'가 아니라
        '가장 많은 angle이 동의하는 반경'을 찾는다.
        """
        radius_tests = np.arange(r_min, r_max + 0.25, 0.5, dtype=np.float32)
        best_radius = None
        best_count = -1
        best_strength_score = -1.0
        for radius0 in radius_tests:
            count = 0
            strength_score = 0.0
            for candidates in angle_candidates:
                if len(candidates) == 0:
                    continue
                valid = [c for c in candidates if abs(c['radius'] - float(radius0)) <= cluster_tolerance]
                if len(valid) == 0:
                    continue
                chosen = min(valid, key=lambda c: abs(c['radius'] - float(radius0)))
                distance_weight = math.exp(-0.5 * ((chosen['radius'] - float(radius0)) / max(cluster_tolerance * 0.55, 1e-06)) ** 2)
                count += 1
                strength_score += chosen['strength'] * distance_weight
            if count > best_count or (count == best_count and strength_score > best_strength_score):
                best_count = count
                best_strength_score = strength_score
                best_radius = float(radius0)
        return (best_radius, best_count)

    def points_from_consensus(self, angle_candidates, angles, cx, cy, dominant_radius, cluster_tolerance):
        points = []
        angle_ids = []
        for angle_index, candidates in enumerate(angle_candidates):
            if len(candidates) == 0:
                continue
            valid = [c for c in candidates if abs(c['radius'] - dominant_radius) <= cluster_tolerance]
            if len(valid) == 0:
                continue
            max_strength = max((c['strength'] for c in valid))
            max_strength = max(max_strength, 1e-06)
            chosen = min(valid, key=lambda c: abs(c['radius'] - dominant_radius) / max(cluster_tolerance, 1e-06) - 0.18 * (c['strength'] / max_strength))
            theta = float(angles[angle_index])
            radius = float(chosen['radius'])
            x = float(cx + math.cos(theta) * radius)
            y = float(cy + math.sin(theta) * radius)
            points.append([x, y])
            angle_ids.append(angle_index)
        if len(points) == 0:
            return (np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.int32))
        return (np.asarray(points, dtype=np.float32), np.asarray(angle_ids, dtype=np.int32))

    @staticmethod
    def ellipse_residuals(points, ellipse):
        (cx, cy), (axis1, axis2), angle_deg = ellipse
        a = max(float(axis1) / 2.0, 1e-06)
        b = max(float(axis2) / 2.0, 1e-06)
        theta = math.radians(float(angle_deg))
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        dx = points[:, 0] - float(cx)
        dy = points[:, 1] - float(cy)
        xr = cos_t * dx + sin_t * dy
        yr = -sin_t * dx + cos_t * dy
        rho = np.sqrt((xr / a) ** 2 + (yr / b) ** 2)
        residual = np.abs(rho - 1.0) * min(a, b)
        return residual.astype(np.float32)

    def ellipse_is_plausible(self, ellipse, yolo_cx, yolo_cy, box_scale, r_min, r_max):
        (cx, cy), (axis1, axis2), _ = ellipse
        major = max(float(axis1), float(axis2))
        minor = min(float(axis1), float(axis2))
        if minor <= 1e-06:
            return False
        axis_ratio = major / minor
        if axis_ratio > self.MAX_ELLIPSE_AXIS_RATIO:
            return False
        semi_major = major / 2.0
        semi_minor = minor / 2.0
        if semi_minor < r_min * 0.65:
            return False
        if semi_major > r_max * 1.2:
            return False
        center_shift = math.hypot(float(cx) - float(yolo_cx), float(cy) - float(yolo_cy))
        if center_shift > float(box_scale) * self.MAX_CENTER_SHIFT_RATIO:
            return False
        return True

    def robust_fit_ellipse(self, points, angle_ids, yolo_cx, yolo_cy, box_scale, r_min, r_max):
        if len(points) < self.MIN_FIT_POINTS:
            return (None, None, None)
        current_points = points.copy()
        current_angle_ids = angle_ids.copy()
        ellipse = None
        for _ in range(4):
            if len(current_points) < self.MIN_FIT_POINTS:
                return (None, None, None)
            try:
                ellipse = cv2.fitEllipse(current_points.reshape(-1, 1, 2))
            except cv2.error:
                return (None, None, None)
            if not self.ellipse_is_plausible(ellipse, yolo_cx, yolo_cy, box_scale, r_min, r_max):
                return (None, None, None)
            residual = self.ellipse_residuals(current_points, ellipse)
            med = float(np.median(residual))
            mad = float(np.median(np.abs(residual - med)))
            robust_sigma = 1.4826 * mad
            threshold = max(1.0, med + 3.0 * robust_sigma)
            threshold = min(threshold, self.MAX_FIT_RESIDUAL_PX)
            mask = residual <= threshold
            kept = int(np.count_nonzero(mask))
            if kept == len(current_points):
                break
            if kept < self.MIN_FIT_POINTS:
                break
            current_points = current_points[mask]
            current_angle_ids = current_angle_ids[mask]
        if ellipse is None or len(current_points) < self.MIN_FIT_POINTS:
            return (None, None, None)
        try:
            ellipse = cv2.fitEllipse(current_points.reshape(-1, 1, 2))
        except cv2.error:
            return (None, None, None)
        if not self.ellipse_is_plausible(ellipse, yolo_cx, yolo_cy, box_scale, r_min, r_max):
            return (None, None, None)
        residual = self.ellipse_residuals(current_points, ellipse)
        return (ellipse, current_points, current_angle_ids)

    def expected_radius_from_ellipse(self, ellipse, angles):
        _, (axis1, axis2), angle_deg = ellipse
        a = max(float(axis1) / 2.0, 1e-06)
        b = max(float(axis2) / 2.0, 1e-06)
        theta = math.radians(float(angle_deg))
        delta = angles - theta
        denominator = np.sqrt((np.cos(delta) / a) ** 2 + (np.sin(delta) / b) ** 2)
        denominator = np.maximum(denominator, 1e-06)
        return (1.0 / denominator).astype(np.float32)

    def second_pass_points(self, norm, first_ellipse, angles, box_scale):
        (cx, cy), _, _ = first_ellipse
        expected_r = self.expected_radius_from_ellipse(first_ellipse, angles)
        band = max(self.SECOND_PASS_BAND_MIN_PX, float(box_scale) * self.SECOND_PASS_BAND_RATIO)
        offsets = np.arange(-band, band + self.RADIAL_STEP_PX * 0.5, self.RADIAL_STEP_PX, dtype=np.float32)
        if len(offsets) < 7:
            return (np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.int32))
        radius_grid = (expected_r[:, None] + offsets[None, :]).astype(np.float32)
        radius_grid = np.maximum(radius_grid, 1.0)
        profiles, gradients, rr = self.radial_profiles(norm, float(cx), float(cy), angles, radius_grid)
        points = []
        angle_ids = []
        for angle_index in range(len(angles)):
            g = gradients[angle_index]
            if len(g) < 5:
                continue
            g_med = float(np.median(g))
            g_mad = float(np.median(np.abs(g - g_med)))
            adaptive_threshold = max(self.MIN_RADIAL_GRADIENT, g_med + self.GRAD_MAD_FACTOR * 1.4826 * g_mad)
            peak_idx = np.where((g[1:-1] > g[:-2]) & (g[1:-1] >= g[2:]) & (g[1:-1] >= adaptive_threshold))[0] + 1
            if len(peak_idx) == 0:
                continue
            row_max = max(float(np.max(g[peak_idx])), 1e-06)
            best_k = min(peak_idx, key=lambda k: abs(float(rr[angle_index, k]) - float(expected_r[angle_index])) / max(band, 1e-06) - 0.22 * (float(g[k]) / row_max))
            if best_k <= 0 or best_k >= len(g) - 1:
                continue
            delta = self.subpixel_peak_offset(g[best_k - 1], g[best_k], g[best_k + 1])
            step = float(rr[angle_index, best_k + 1] - rr[angle_index, best_k])
            radius = float(rr[angle_index, best_k] + delta * step)
            theta = float(angles[angle_index])
            x = float(cx + math.cos(theta) * radius)
            y = float(cy + math.sin(theta) * radius)
            points.append([x, y])
            angle_ids.append(angle_index)
        if len(points) == 0:
            return (np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.int32))
        return (np.asarray(points, dtype=np.float32), np.asarray(angle_ids, dtype=np.int32))

    def ellipse_local_contrast(self, norm, ellipse):
        (cx, cy), (axis1, axis2), angle = ellipse
        h, w = norm.shape[:2]
        inner_mask = np.zeros((h, w), dtype=np.uint8)
        outer_big = np.zeros((h, w), dtype=np.uint8)
        outer_small = np.zeros((h, w), dtype=np.uint8)
        center = (int(round(cx)), int(round(cy)))
        inner_axes = (max(1, int(round(axis1 * 0.32))), max(1, int(round(axis2 * 0.32))))
        outer_big_axes = (max(1, int(round(axis1 * 0.68))), max(1, int(round(axis2 * 0.68))))
        outer_small_axes = (max(1, int(round(axis1 * 0.55))), max(1, int(round(axis2 * 0.55))))
        cv2.ellipse(inner_mask, center, inner_axes, float(angle), 0, 360, 255, -1)
        cv2.ellipse(outer_big, center, outer_big_axes, float(angle), 0, 360, 255, -1)
        cv2.ellipse(outer_small, center, outer_small_axes, float(angle), 0, 360, 255, -1)
        outer_ring = cv2.subtract(outer_big, outer_small)
        inner_values = norm[inner_mask > 0]
        outer_values = norm[outer_ring > 0]
        if len(inner_values) < 10 or len(outer_values) < 10:
            return (0.0, 0.0, 0.0)
        inner_level = float(np.percentile(inner_values, 35.0))
        outer_level = float(np.median(outer_values))
        contrast = outer_level - inner_level
        return (float(contrast), inner_level, outer_level)

    def _build_edge_field(self, norm):
        """Scharr gradient field. Absolute image brightness is not used here."""
        sm = cv2.GaussianBlur(norm.astype(np.float32), (0, 0), 0.8)
        gx = cv2.Scharr(sm, cv2.CV_32F, 1, 0) / 16.0
        gy = cv2.Scharr(sm, cv2.CV_32F, 0, 1) / 16.0
        mag = cv2.magnitude(gx, gy)
        return gx, gy, mag

    def _score_ring_for_center(self, gx, gy, mag, cx, cy, r_min, r_max, box_scale, expected_radius):
        """한 중심 후보에서 '외곽 annulus'만 평가한다.

        핵심:
          - r_min보다 안쪽은 완전히 버린다 -> 구멍 바닥의 흰 반사/초승달 제외.
          - gradient 부호는 사용하지 않는다 -> 좌/우 측면 조명에 강함.
          - 한 선의 강한 edge가 아니라 여러 각도에서 같은 반경에 존재하는지를 본다.
          - YOLO bbox로 얻은 예상 반경은 hard threshold가 아니라 부드러운 prior로만 사용한다.
        """
        h, w = mag.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        dx = xx - float(cx)
        dy = yy - float(cy)
        rr = np.sqrt(dx * dx + dy * dy)
        safe = (rr >= float(r_min)) & (rr <= float(r_max))
        if np.count_nonzero(safe) < 18:
            return None

        safe_mag = mag[safe]
        # ROI마다 상대 threshold. 노출이 바뀌어도 자동으로 따라간다.
        edge_thr = max(0.65, float(np.percentile(safe_mag, 58.0)))

        rr_safe = np.maximum(rr, 1e-6)
        # 방향만 본다. 밝아지는 edge와 어두워지는 edge 둘 다 허용.
        radial_dot = (gx * dx + gy * dy) / (np.maximum(mag, 1e-6) * rr_safe)
        align = np.clip(np.abs(radial_dot), 0.0, 1.0)

        edge_mask = safe & (mag >= edge_thr) & (align >= 0.22)
        ys, xs = np.where(edge_mask)
        if len(xs) < 16:
            return None

        radii = rr[ys, xs].astype(np.float32)
        theta = (np.arctan2(dy[ys, xs], dx[ys, xs]) + 2.0 * np.pi) % (2.0 * np.pi)
        weights = (mag[ys, xs] * (0.30 + 0.70 * align[ys, xs] ** 2.0)).astype(np.float32)

        radius_band = max(0.80, float(expected_radius) * 0.13)
        test_step = 0.25 if expected_radius < 12.0 else 0.5
        radius_tests = np.arange(float(r_min), float(r_max) + test_step * 0.5, test_step, dtype=np.float32)
        angle_bins = 72  # 5 degree sectors
        prior_sigma = max(2.0, float(box_scale) * self.ANNULUS_PRIOR_SIGMA_RATIO)
        best = None

        for r0 in radius_tests:
            dm = np.abs(radii - float(r0))
            m = dm <= radius_band
            if np.count_nonzero(m) < 12:
                continue

            a = theta[m]
            ww = weights[m] * np.exp(-0.5 * (dm[m] / max(radius_band * 0.70, 1e-6)) ** 2)
            bins = np.floor(a / (2.0 * np.pi) * angle_bins).astype(np.int32)
            bins = np.clip(bins, 0, angle_bins - 1)
            per_bin = np.zeros(angle_bins, dtype=np.float32)
            np.maximum.at(per_bin, bins, ww)

            nz = per_bin[per_bin > 0]
            if len(nz) == 0:
                continue
            support_thr = max(0.18 * float(np.percentile(nz, 65.0)), 0.03)
            active = per_bin > support_thr
            coverage = float(np.count_nonzero(active)) / float(angle_bins)
            if coverage <= 0.0:
                continue

            strength = float(np.mean(per_bin[active]))
            # YOLO 예상 반경에서 너무 멀리 가는 내부 반사/바깥 그림자를 억제한다.
            prior = math.exp(-0.5 * ((float(r0) - float(expected_radius)) / prior_sigma) ** 2)
            # coverage가 가장 중요하고, prior/strength는 보조 역할만 한다.
            score = (coverage ** 2.55) * math.log1p(max(strength, 0.0)) * (0.55 + 0.45 * prior)

            if best is None or score > best['score']:
                best = {
                    'radius': float(r0),
                    'coverage': coverage,
                    'score': float(score),
                    'edge_thr': edge_thr,
                    'radius_band': radius_band,
                    'xs': xs,
                    'ys': ys,
                    'radii': radii,
                    'theta': theta,
                    'weights': weights,
                    'align_values': align[ys, xs],
                    'expected_radius': float(expected_radius),
                }
        return best
    def _ring_points_from_score(self, best, cx, cy, sectors=180):
        if best is None:
            return np.empty((0, 2), np.float32), np.empty((0,), np.int32)
        r0 = best['radius']
        band = max(best['radius_band'] * 1.35, 2.0)
        radii = best['radii']
        theta = best['theta']
        weights = best['weights']
        xs = best['xs']
        ys = best['ys']
        dm = np.abs(radii - r0)
        valid = dm <= band
        if np.count_nonzero(valid) < 9:
            return np.empty((0, 2), np.float32), np.empty((0,), np.int32)

        bins = np.floor(theta[valid] / (2.0 * np.pi) * sectors).astype(np.int32)
        bins = np.clip(bins, 0, sectors - 1)
        idx_global = np.where(valid)[0]
        points = []
        angle_ids = []
        for b in np.unique(bins):
            candidates = idx_global[bins == b]
            # Prefer strong radial edge, but stay close to the dominant outer radius.
            local_score = weights[candidates] / (1.0 + 0.55 * (dm[candidates] / max(band, 1e-6)) ** 2)
            k = candidates[int(np.argmax(local_score))]
            points.append([float(xs[k]), float(ys[k])])
            angle_ids.append(int(b))
        if not points:
            return np.empty((0, 2), np.float32), np.empty((0,), np.int32)
        return np.asarray(points, np.float32), np.asarray(angle_ids, np.int32)

    @staticmethod
    def _fit_circle_lstsq(points):
        if points is None or len(points) < 3:
            return None
        x = points[:, 0].astype(np.float64)
        y = points[:, 1].astype(np.float64)
        A = np.column_stack((2.0 * x, 2.0 * y, np.ones_like(x)))
        b = x * x + y * y
        try:
            sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            return None
        cx, cy, c = sol
        r2 = c + cx * cx + cy * cy
        if not np.isfinite(r2) or r2 <= 0.0:
            return None
        return float(cx), float(cy), float(math.sqrt(r2))

    def _robust_circle_fit(self, points, min_points=24):
        if points is None or len(points) < min_points:
            return None, None
        cur = points.copy()
        for _ in range(5):
            fit = self._fit_circle_lstsq(cur)
            if fit is None:
                return None, None
            cx, cy, r = fit
            dist = np.sqrt((cur[:, 0] - cx) ** 2 + (cur[:, 1] - cy) ** 2)
            residual = np.abs(dist - r)
            med = float(np.median(residual))
            mad = float(np.median(np.abs(residual - med)))
            sigma = 1.4826 * mad
            thr = min(3.0, max(0.9, med + 2.8 * sigma))
            keep = residual <= thr
            if np.count_nonzero(keep) < min_points:
                break
            if np.count_nonzero(keep) == len(cur):
                break
            cur = cur[keep]
        fit = self._fit_circle_lstsq(cur)
        return fit, cur

    def _find_best_initial_center(self, gx, gy, mag, yolo_cx, yolo_cy, r_min, r_max, box_scale, expected_radius):
        # YOLO 중심을 강한 위치 prior로 사용한다.
        best_center = (float(yolo_cx), float(yolo_cy))
        best_ring = self._score_ring_for_center(
            gx, gy, mag,
            best_center[0], best_center[1],
            r_min, r_max, box_scale, expected_radius,
        )
        best_score = -1.0 if best_ring is None else best_ring['score']

        # YOLO bbox가 약간 치우친 경우만 보정. 너무 넓게 탐색하면 plate 그림자에 끌릴 수 있다.
        span = max(1.5, float(box_scale) * self.CENTER_SEARCH_RATIO)
        offsets = np.linspace(-span, span, 5, dtype=np.float32)
        center_sigma = max(1.5, span * 0.85)

        for oy in offsets:
            for ox in offsets:
                if abs(float(ox)) < 1e-6 and abs(float(oy)) < 1e-6:
                    continue
                cx = float(yolo_cx + ox)
                cy = float(yolo_cy + oy)
                ring = self._score_ring_for_center(
                    gx, gy, mag,
                    cx, cy,
                    r_min, r_max, box_scale, expected_radius,
                )
                if ring is None:
                    continue
                shift = math.hypot(float(ox), float(oy))
                center_prior = math.exp(-0.5 * (shift / center_sigma) ** 2)
                score = ring['score'] * (0.82 + 0.18 * center_prior)
                if score > best_score:
                    best_score = score
                    best_center = (cx, cy)
                    best_ring = ring
        return best_center, best_ring
    def _subpixel_outer_points(self, norm, cx, cy, radius, box_scale, angles):
        """이미 확정된 외곽 ring 주변만 0.25 px 간격으로 다시 측정한다.

        내부 반사 edge를 피하기 위해 중심부는 보지 않으며, gradient 부호도 무시한다.
        후보가 여러 개면 '가장 바깥쪽'이 아니라 현재 외곽 반경에 가장 잘 맞는 edge를 고른다.
        """
        # 작은 구멍에서 고정 2.5px band는 반경의 40% 이상이 되어 내부 반사까지 다시 들어온다.
        # 실제 검출된 반경에 비례하는 좁은 band만 사용한다.
        half_band = max(1.10, min(3.25, float(radius) * 0.22))
        step = 0.20 if radius < 10.0 else 0.25

        # 반경의 78% 안쪽은 최종 단계에서도 절대 탐색하지 않는다.
        inner_limit = max(1.0, float(radius) * 0.78)
        r_start = max(inner_limit, float(radius) - half_band)
        r_end = float(radius) + half_band
        radii = np.arange(r_start, r_end + step * 0.5, step, dtype=np.float32)
        if len(radii) < 7:
            return np.empty((0, 2), np.float32), np.empty((0,), np.int32), np.empty((0,), np.float32)

        profiles, gradients, rr = self.radial_profiles(norm, float(cx), float(cy), angles, radii)
        points = []
        angle_ids = []
        strengths = []

        for i, theta in enumerate(angles):
            g = gradients[i]
            if len(g) < 5:
                continue
            ag = np.abs(g)
            core = ag[1:-1]
            med = float(np.median(core))
            mad = float(np.median(np.abs(core - med)))
            thr = max(0.45, med + 1.55 * 1.4826 * mad)

            peaks = np.where(
                (ag[1:-1] > ag[:-2]) &
                (ag[1:-1] >= ag[2:]) &
                (ag[1:-1] >= thr)
            )[0] + 1
            if len(peaks) == 0:
                continue

            max_g = max(float(np.max(ag[peaks])), 1e-6)
            strong = [int(k) for k in peaks if float(ag[k]) >= max(thr, 0.40 * max_g)]
            if not strong:
                continue

            # 위치 일관성을 우선하고 edge strength를 보조 점수로 쓴다.
            def candidate_cost(k):
                dr_norm = abs(float(rr[i, k]) - float(radius)) / max(half_band, 1e-6)
                strength_norm = float(ag[k]) / max_g
                return dr_norm - 0.22 * strength_norm

            k = min(strong, key=candidate_cost)
            if k <= 0 or k >= len(g) - 1:
                continue

            delta = self.subpixel_peak_offset(ag[k - 1], ag[k], ag[k + 1])
            dr = float(rr[i, k + 1] - rr[i, k])
            r_sub = float(rr[i, k] + delta * dr)
            if abs(r_sub - float(radius)) > half_band * 0.95:
                continue

            x = float(cx + math.cos(float(theta)) * r_sub)
            y = float(cy + math.sin(float(theta)) * r_sub)
            points.append([x, y])
            angle_ids.append(i)
            strengths.append(float(ag[k]))

        if not points:
            return np.empty((0, 2), np.float32), np.empty((0,), np.int32), np.empty((0,), np.float32)

        points = np.asarray(points, np.float32)
        angle_ids = np.asarray(angle_ids, np.int32)
        strengths = np.asarray(strengths, np.float32)

        # 측면 조명 때문에 일부 각도의 edge가 아주 약하면 그 점을 억지로 쓰지 않는다.
        if len(strengths) >= 80:
            s_thr = float(np.percentile(strengths, 10.0))
            keep = strengths >= s_thr
            points = points[keep]
            angle_ids = angle_ids[keep]
            strengths = strengths[keep]

        return points, angle_ids, strengths
    def refine_hole_center(self, frame, yolo_box, debug_name=''):
        img_h, img_w = frame.shape[:2]
        x1, y1, x2, y2 = yolo_box
        box_w = max(1.0, x2 - x1)
        box_h = max(1.0, y2 - y1)
        # 원형 물체 bbox의 한 축이 조명/저해상도 때문에 1~2 px만 줄어도
        # min(width,height)는 반경 prior를 과도하게 작게 만든다. 평균 축 길이를 사용한다.
        box_scale = max(1.0, 0.5 * (box_w + box_h))

        # YOLO가 구멍 위치를 이미 잘 잡으므로 ROI도 과도하게 넓히지 않는다.
        margin_x = max(8, int(round(box_w * 0.45)))
        margin_y = max(8, int(round(box_h * 0.45)))
        rx1 = max(0, int(math.floor(x1)) - margin_x)
        ry1 = max(0, int(math.floor(y1)) - margin_y)
        rx2 = min(img_w, int(math.ceil(x2)) + margin_x)
        ry2 = min(img_h, int(math.ceil(y2)) + margin_y)

        if rx2 <= rx1 or ry2 <= ry1:
            return None, None
        roi = frame[ry1:ry2, rx1:rx2].copy()
        if roi.size == 0:
            return None, None

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        norm = self.local_illumination_normalize(gray, box_scale)
        gx, gy, mag = self._build_edge_field(norm)

        yolo_cx = (x1 + x2) * 0.5 - rx1
        yolo_cy = (y1 + y2) * 0.5 - ry1
        roi_h, roi_w = gray.shape[:2]
        safe_radius = min(yolo_cx, yolo_cy, roi_w - 1.0 - yolo_cx, roi_h - 1.0 - yolo_cy) - 2.0

        # ---------------- v5 핵심 ----------------
        # YOLO bbox에서 '예상 구멍 외곽 반경'을 만들고,
        # 그 주변 annulus만 검색한다. 내부 바닥 반사는 탐색 영역에 들어오지 않는다.
        # YOLO bbox가 실제 구멍 직경과 거의 비슷하므로 radius ~= 0.46 * bbox scale을 prior로 둔다.
        # 저해상도에서는 bbox가 몇 픽셀만 변해도 기존 0.22~0.52*box annulus가 4px 미만으로
        # 줄어들 수 있었기 때문에, v6에서는 예상 반경 R을 먼저 만들고 R의 비율로 annulus를 만든다.
        expected_radius = float(box_scale) * self.YOLO_RADIUS_PRIOR_RATIO
        expected_radius = float(np.clip(expected_radius, 3.0, max(3.0, safe_radius * 0.82)))

        r_min = max(2.0, expected_radius * self.ANNULUS_INNER_RADIUS_RATIO)
        r_max = min(max(3.0, safe_radius), expected_radius * self.ANNULUS_OUTER_RADIUS_RATIO)

        # 최소 annulus 폭을 확보한다. 먼저 바깥쪽으로 넓히고, ROI 한계에 걸리면 안쪽을 조금 연다.
        min_width = max(self.MIN_ANNULUS_WIDTH_PX, expected_radius * 0.34)
        if r_max - r_min < min_width:
            need = min_width - (r_max - r_min)
            grow_out = min(need, max(0.0, safe_radius - r_max))
            r_max += grow_out
            need -= grow_out
            if need > 0.0:
                r_min = max(1.75, r_min - need)

        # prior는 annulus 안에 있어야 한다. 작은 구멍에서 +/-1px 강제 clamp는 너무 크므로 0.35px만 둔다.
        expected_radius = float(np.clip(expected_radius, r_min + 0.35, max(r_min + 0.35, r_max - 0.35)))

        debug = {
            'roi': roi,
            'gray': gray,
            'norm': norm,
            'origin': (rx1, ry1),
            'candidate': None,
            'name': debug_name,
            'debug_img': None,
            'dominant_radius': None,
            'first_coverage': 0.0,
            'final_coverage': 0.0,
            'fit_rmse': None,
            'contrast': None,
            'reason': 'start',
            'prior_center': (float(yolo_cx), float(yolo_cy)),
            'expected_radius': float(expected_radius),
            'annulus_inner': float(r_min),
            'annulus_outer': float(r_max),
            'bbox_wh': (float(box_w), float(box_h)),
            'box_scale': float(box_scale),
        }

        dbg = cv2.cvtColor(np.clip(norm, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        cv2.drawMarker(dbg, (int(round(yolo_cx)), int(round(yolo_cy))), (255, 255, 0), cv2.MARKER_CROSS, 9, 1)
        # 흰색=내부 무시 경계, 청록=외부 검색 경계. 이 사이만 1차 탐색한다.
        cv2.circle(dbg, (int(round(yolo_cx)), int(round(yolo_cy))), int(round(r_min)), (235, 235, 235), 1, cv2.LINE_AA)
        cv2.circle(dbg, (int(round(yolo_cx)), int(round(yolo_cy))), int(round(r_max)), (255, 180, 0), 1, cv2.LINE_AA)

        if r_max <= r_min + 1.75:
            debug['reason'] = 'annulus range too small'
            cv2.putText(dbg, debug['reason'], (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 0, 255), 1)
            debug['debug_img'] = dbg
            return None, debug

        center0, ring0 = self._find_best_initial_center(
            gx, gy, mag,
            yolo_cx, yolo_cy,
            r_min, r_max,
            box_scale,
            expected_radius,
        )
        if ring0 is None:
            debug['reason'] = 'no outer annulus ring'
            cv2.putText(dbg, debug['reason'], (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 0, 255), 1)
            debug['debug_img'] = dbg
            return None, debug

        pts0, ids0 = self._ring_points_from_score(ring0, center0[0], center0[1])
        debug['dominant_radius'] = ring0['radius']
        debug['first_coverage'] = ring0['coverage']
        for p in pts0:
            cv2.circle(dbg, (int(round(p[0])), int(round(p[1]))), 1, (0, 255, 255), -1)

        min_geom_points = 16 if expected_radius < 10.0 else 24
        if ring0['coverage'] < self.MIN_RING_COVERAGE_V5 or len(pts0) < min_geom_points:
            debug['reason'] = f'low annulus cov {ring0["coverage"]:.2f}'
            cv2.putText(dbg, debug['reason'], (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 0, 255), 1)
            debug['debug_img'] = dbg
            return None, debug

        circle0, in0 = self._robust_circle_fit(pts0, min_points=min_geom_points)
        if circle0 is None:
            debug['reason'] = 'circle fit failed'
            cv2.putText(dbg, debug['reason'], (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 0, 255), 1)
            debug['debug_img'] = dbg
            return None, debug

        cx, cy, radius = circle0

        # 한 번만 재평가하되, 이미 찾은 외곽 반경의 좁은 주변만 본다.
        local_band = max(3.0, float(box_scale) * 0.08)
        local_r_min = max(r_min, radius - local_band)
        local_r_max = min(r_max, radius + local_band)
        ring1 = self._score_ring_for_center(
            gx, gy, mag,
            cx, cy,
            local_r_min, local_r_max,
            box_scale,
            radius,
        )
        if ring1 is not None and ring1['coverage'] >= ring0['coverage'] * 0.82:
            pts1, ids1 = self._ring_points_from_score(ring1, cx, cy)
            circle1, in1 = self._robust_circle_fit(pts1, min_points=min_geom_points)
            if circle1 is not None:
                cx, cy, radius = circle1
                ring0 = ring1
                pts0, ids0 = pts1, ids1
                in0 = in1

        # YOLO 위치에서 지나치게 멀리 가면 외곽 rim/그림자를 잘못 잡은 것으로 간주.
        center_shift = math.hypot(cx - yolo_cx, cy - yolo_cy)
        if center_shift > max(5.0, box_scale * 0.20):
            debug['reason'] = f'center shift {center_shift:.1f}px'
            cv2.circle(dbg, (int(round(cx)), int(round(cy))), int(round(radius)), (0, 165, 255), 1)
            cv2.putText(dbg, debug['reason'], (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 0, 255), 1)
            debug['debug_img'] = dbg
            return None, debug

        angles = np.linspace(0.0, 2.0 * np.pi, self.RADIAL_ANGLES, endpoint=False, dtype=np.float32)
        sub_pts, sub_ids, strengths = self._subpixel_outer_points(norm, cx, cy, radius, box_scale, angles)
        coverage = float(len(np.unique(sub_ids))) / float(self.RADIAL_ANGLES) if len(sub_ids) else 0.0
        debug['final_coverage'] = coverage

        if len(sub_pts) < 40 or coverage < 0.28:
            final_pts = in0 if in0 is not None and len(in0) >= min_geom_points else pts0
            final_ids = np.arange(len(final_pts), dtype=np.int32)
            coverage = max(coverage, ring0['coverage'])
        else:
            final_pts = sub_pts
            final_ids = sub_ids

        ellipse = None
        ellipse_pts = final_pts
        if len(final_pts) >= 40:
            ellipse, ellipse_inliers, ellipse_angle_ids = self.robust_fit_ellipse(
                final_pts,
                final_ids,
                yolo_cx,
                yolo_cy,
                box_scale,
                max(2.5, radius * 0.72),
                min(r_max + 2.0, radius * 1.30),
            )
            if ellipse is not None:
                ellipse_pts = ellipse_inliers

        if ellipse is None:
            circle_final, circle_inliers = self._robust_circle_fit(final_pts, min_points=min_geom_points)
            if circle_final is None:
                debug['reason'] = 'final fit failed'
                cv2.putText(dbg, debug['reason'], (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 0, 255), 1)
                debug['debug_img'] = dbg
                return None, debug
            cx, cy, radius = circle_final
            ellipse = ((float(cx), float(cy)), (float(2.0 * radius), float(2.0 * radius)), 0.0)
            ellipse_pts = circle_inliers
        else:
            (cx, cy), (axis1, axis2), _ = ellipse
            radius = 0.25 * (float(axis1) + float(axis2))

        residual = self.ellipse_residuals(ellipse_pts, ellipse)
        fit_rmse = float(np.sqrt(np.mean(residual ** 2))) if len(residual) else 99.0
        contrast, inner_level, outer_level = self.ellipse_local_contrast(norm, ellipse)

        (ecx, ecy), (axis1, axis2), eang = ellipse
        major = max(float(axis1), float(axis2))
        minor = min(float(axis1), float(axis2))
        axis_ratio = major / max(minor, 1e-6)
        final_shift = math.hypot(float(ecx) - yolo_cx, float(ecy) - yolo_cy)

        if axis_ratio > 1.60 or final_shift > max(5.5, box_scale * 0.22):
            debug['reason'] = f'geometry reject ar={axis_ratio:.2f}'
            cv2.ellipse(dbg, ellipse, (0, 165, 255), 1)
            cv2.putText(dbg, debug['reason'], (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 0, 255), 1)
            debug['debug_img'] = dbg
            return None, debug

        # 최종 반경도 YOLO annulus에서 크게 벗어나지 않아야 한다.
        if radius < r_min * 0.90 or radius > r_max * 1.08:
            debug['reason'] = f'radius reject {radius:.1f}'
            cv2.ellipse(dbg, ellipse, (0, 165, 255), 1)
            cv2.putText(dbg, debug['reason'], (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 0, 255), 1)
            debug['debug_img'] = dbg
            return None, debug

        for p in ellipse_pts:
            cv2.circle(dbg, (int(round(p[0])), int(round(p[1]))), 1, (0, 255, 0), -1)
        cv2.ellipse(dbg, ellipse, (0, 0, 255), 1)
        cv2.drawMarker(dbg, (int(round(ecx)), int(round(ecy))), (255, 0, 255), cv2.MARKER_CROSS, 11, 1)
        debug['reason'] = 'OK'
        cv2.putText(
            dbg,
            f'OK cov:{coverage:.2f} r:{radius:.2f} prior:{expected_radius:.1f} err:{fit_rmse:.2f}',
            (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (255, 255, 255), 1,
        )

        candidate = {
            'cx_local': float(ecx),
            'cy_local': float(ecy),
            'ellipse': ellipse,
            'major': major,
            'minor': minor,
            'axis_ratio': axis_ratio,
            'points': ellipse_pts,
            'first_points': pts0,
            'dominant_radius': float(radius),
            'first_coverage': float(ring0['coverage']),
            'final_coverage': float(coverage),
            'fit_rmse': fit_rmse,
            'contrast': float(contrast),
            'inner_level': float(inner_level),
            'outer_level': float(outer_level),
            'expected_radius': float(expected_radius),
            'annulus_inner': float(r_min),
            'annulus_outer': float(r_max),
            'bbox_wh': (float(box_w), float(box_h)),
            'box_scale': float(box_scale),
        }
        debug['candidate'] = candidate
        debug['debug_img'] = dbg
        debug['fit_rmse'] = fit_rmse
        debug['contrast'] = contrast
        debug['dominant_radius'] = float(radius)
        debug['final_coverage'] = float(coverage)

        return (float(rx1 + ecx), float(ry1 + ecy)), debug
    def draw_refined_contour(self, display, debug, color):
        """Draw only the fitted outer boundary on the main view.

        The fitting points are intentionally NOT drawn here. On a small hole,
        dozens of 1-pixel points overlap and make the detected boundary look
        much thicker than it really is. This is visualization only and does
        not change the measured center/ellipse.
        """
        if debug is None:
            return
        candidate = debug.get('candidate')
        if candidate is None:
            return

        ox, oy = debug['origin']
        ellipse = candidate.get('ellipse')
        if ellipse is not None:
            (ecx, ecy), (axis1, axis2), angle = ellipse
            shifted_ellipse = (
                (float(ox + ecx), float(oy + ecy)),
                (float(axis1), float(axis2)),
                float(angle),
            )
            cv2.ellipse(display, shifted_ellipse, color, 1, cv2.LINE_AA)

        # One tiny center pixel only. The larger cross is drawn later after
        # temporal stabilization, so the main screen stays uncluttered.
        cx = float(ox + candidate['cx_local'])
        cy = float(oy + candidate['cy_local'])
        cv2.circle(display, (int(round(cx)), int(round(cy))), 1, color, -1, cv2.LINE_AA)

    def find_best_boxes(self, result):
        best_base = None
        base_conf = -1.0
        dot_boxes = []
        for box in result.boxes:
            class_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = self.model.names[class_id]
            label_l = self.normalize_label(label)
            if label_l in BASE_LABELS:
                if conf > base_conf:
                    base_conf = conf
                    best_base = box
            elif label_l in DOT_LABELS:
                dot_boxes.append({'box': box, 'conf': conf})
        dot_boxes.sort(key=lambda d: d['conf'], reverse=True)
        dot_boxes = dot_boxes[:2]
        return (best_base, dot_boxes, base_conf)

    def classify_dots_by_base_center(self, base_box, dot_boxes):
        if base_box is None or len(dot_boxes) < 2:
            return (None, None)
        bx1, by1, bx2, by2 = base_box.xyxy[0].tolist()
        base_cx = (bx1 + bx2) / 2.0
        base_cy = (by1 + by2) / 2.0
        classified = []
        for item in dot_boxes:
            box = item['box']
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            dot_cx = (x1 + x2) / 2.0
            dot_cy = (y1 + y2) / 2.0
            dist_from_base = math.hypot(dot_cx - base_cx, dot_cy - base_cy)
            classified.append({'box': box, 'conf': item['conf'], 'dist_from_base': float(dist_from_base)})
        classified.sort(key=lambda d: d['dist_from_base'])
        dot_mid = classified[0]
        dot_side = classified[1]
        return (dot_mid, dot_side)

    def fit_debug_panel(self, img, size):
        h, w = img.shape[:2]
        if h <= 0 or w <= 0:
            return np.zeros((size, size, 3), dtype=np.uint8)
        scale = min(size / float(w), size / float(h))
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))

        # INTER_NEAREST made every original 1-pixel debug mark expand into a
        # large square when the ROI was magnified. Linear interpolation keeps
        # the image readable. For successful detections, geometry is redrawn
        # AFTER scaling in make_binary_debug(), so its line width stays 1 px.
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((size, size, 3), dtype=np.uint8)
        x0 = (size - new_w) // 2
        y0 = (size - new_h) // 2
        canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
        return canvas

    def _make_thin_debug_panel(self, debug, size):
        """Magnify the ROI first, then draw sparse 1-pixel overlays.

        This keeps yellow/green point clouds and the red fitted ellipse thin
        even if the original hole ROI is only a few dozen pixels wide.
        """
        norm = debug.get('norm')
        if norm is None:
            img = debug.get('debug_img')
            if img is None:
                return None
            return self.fit_debug_panel(img, size)

        base = cv2.cvtColor(
            np.clip(norm, 0, 255).astype(np.uint8),
            cv2.COLOR_GRAY2BGR,
        )
        h, w = base.shape[:2]
        if h <= 0 or w <= 0:
            return np.zeros((size, size, 3), dtype=np.uint8)

        scale = min(size / float(w), size / float(h))
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        resized = cv2.resize(base, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        panel = np.zeros((size, size, 3), dtype=np.uint8)
        x0 = (size - new_w) // 2
        y0 = (size - new_h) // 2
        panel[y0:y0 + new_h, x0:x0 + new_w] = resized

        def sp(p):
            return (
                int(round(x0 + float(p[0]) * scale)),
                int(round(y0 + float(p[1]) * scale)),
            )

        prior_center = debug.get('prior_center')
        annulus_inner = debug.get('annulus_inner')
        annulus_outer = debug.get('annulus_outer')
        if prior_center is not None and annulus_inner is not None and annulus_outer is not None:
            pc = sp(prior_center)
            rin = max(1, int(round(float(annulus_inner) * scale)))
            rout = max(rin + 1, int(round(float(annulus_outer) * scale)))
            # 흰색 원 안쪽은 아예 사용하지 않는 영역. 청록 원까지가 검색 annulus.
            cv2.circle(panel, pc, rin, (235, 235, 235), 1, cv2.LINE_AA)
            cv2.circle(panel, pc, rout, (255, 180, 0), 1, cv2.LINE_AA)

        candidate = debug.get('candidate')
        if candidate is not None:
            # First-pass edge candidates: sparse yellow dots.
            first_points = candidate.get('first_points')
            if first_points is not None and len(first_points) > 0:
                step = max(1, len(first_points) // 48)
                for p in first_points[::step]:
                    cv2.circle(panel, sp(p), 1, (0, 255, 255), -1, cv2.LINE_AA)

            # Final robust inliers: sparse green dots.
            points = candidate.get('points')
            if points is not None and len(points) > 0:
                step = max(1, len(points) // 72)
                for p in points[::step]:
                    cv2.circle(panel, sp(p), 1, (0, 255, 0), -1, cv2.LINE_AA)

            ellipse = candidate.get('ellipse')
            if ellipse is not None:
                (ecx, ecy), (axis1, axis2), angle = ellipse
                scaled_ellipse = (
                    (float(x0 + ecx * scale), float(y0 + ecy * scale)),
                    (float(axis1 * scale), float(axis2 * scale)),
                    float(angle),
                )
                cv2.ellipse(panel, scaled_ellipse, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.drawMarker(
                    panel,
                    sp((ecx, ecy)),
                    (255, 0, 255),
                    cv2.MARKER_CROSS,
                    9,
                    1,
                )

        name = debug.get('name', '')
        reason = debug.get('reason', '')
        label = name if reason in ('', 'OK', 'start') else f'{name} [{reason}]'
        cv2.putText(panel, label, (6, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (0, 255, 255), 1, cv2.LINE_AA)

        if candidate is not None:
            cov = candidate.get('final_coverage', 0.0)
            err = candidate.get('fit_rmse', 0.0)
            radius = candidate.get('dominant_radius', 0.0)
            info = f'cov:{cov:.2f} r:{radius:.2f} err:{err:.2f}px'
            cv2.putText(panel, info, (6, size - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.27, (255, 255, 255), 1, cv2.LINE_AA)

        bbox_wh = debug.get('bbox_wh')
        exp_r = debug.get('expected_radius')
        rin = debug.get('annulus_inner')
        rout = debug.get('annulus_outer')
        if bbox_wh is not None and exp_r is not None and rin is not None and rout is not None:
            small_info = f'b:{bbox_wh[0]:.1f}x{bbox_wh[1]:.1f} exp:{exp_r:.1f} A:{rin:.1f}-{rout:.1f}'
            cv2.putText(panel, small_info, (6, size - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.24, (220, 220, 220), 1, cv2.LINE_AA)

        return panel

    def make_binary_debug(self, debug_mid, debug_side):
        panels = []
        for debug in (debug_mid, debug_side):
            if debug is None:
                continue

            # Successful detections are rendered from clean normalized data so
            # debug overlays stay 1 pixel wide after magnification.
            if debug.get('candidate') is not None:
                panel = self._make_thin_debug_panel(debug, DEBUG_ROI_SIZE)
            else:
                # On failures keep the original diagnostic marks, but resize
                # smoothly so they are less blocky than the previous version.
                img = debug.get('debug_img')
                if img is None:
                    norm = debug.get('norm')
                    if norm is None:
                        continue
                    img = cv2.cvtColor(np.clip(norm, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
                panel = self.fit_debug_panel(img.copy(), DEBUG_ROI_SIZE)
                name = debug.get('name', '')
                reason = debug.get('reason', '')
                label = name if reason in ('', 'OK', 'start') else f'{name} [{reason}]'
                cv2.putText(panel, label, (6, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (0, 255, 255), 1, cv2.LINE_AA)

            if panel is not None:
                panels.append(panel)

        if len(panels) == 0:
            return None
        if len(panels) == 1:
            panels.append(np.zeros_like(panels[0]))
        return cv2.hconcat(panels[:2])

    def listener_callback(self, data):
        frame = self.bridge.imgmsg_to_cv2(data, 'bgr8')
        display = frame.copy()
        results = self.model(frame, conf=self.conf_threshold, iou=self.iou_threshold, verbose=False)
        result = results[0]
        base_box, dot_boxes, base_conf = self.find_best_boxes(result)
        if base_box is not None:
            bx1, by1, bx2, by2 = base_box.xyxy[0].tolist()
            cv2.rectangle(display, (int(round(bx1)), int(round(by1))), (int(round(bx2)), int(round(by2))), (255, 0, 0), 1)
            cv2.putText(display, f'BasePlate {base_conf:.2f}', (int(round(bx1)), max(20, int(round(by1)) - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 1)
        dot_mid_info, dot_side_info = self.classify_dots_by_base_center(base_box, dot_boxes)
        raw_mid = None
        raw_side = None
        debug_mid = None
        debug_side = None
        if dot_mid_info is not None:
            dot_mid_box = dot_mid_info['box']
            dot_mid_conf = dot_mid_info['conf']
            mx1, my1, mx2, my2 = dot_mid_box.xyxy[0].tolist()
            cv2.rectangle(display, (int(round(mx1)), int(round(my1))), (int(round(mx2)), int(round(my2))), (0, 0, 255), 1)
            cv2.putText(display, f'dot_mid {dot_mid_conf:.2f}', (int(round(mx1)), max(20, int(round(my1)) - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 255), 1)
            raw_mid, debug_mid = self.refine_hole_center(frame, (mx1, my1, mx2, my2), 'dot_mid')
        if dot_side_info is not None:
            dot_side_box = dot_side_info['box']
            dot_side_conf = dot_side_info['conf']
            sx1, sy1, sx2, sy2 = dot_side_box.xyxy[0].tolist()
            cv2.rectangle(display, (int(round(sx1)), int(round(sy1))), (int(round(sx2)), int(round(sy2))), (0, 255, 0), 1)
            cv2.putText(display, f'dot_side {dot_side_conf:.2f}', (int(round(sx1)), max(20, int(round(sy1)) - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1)
            raw_side, debug_side = self.refine_hole_center(frame, (sx1, sy1, sx2, sy2), 'dot_side')
        self.draw_refined_contour(display, debug_mid, (0, 0, 255))
        self.draw_refined_contour(display, debug_side, (0, 255, 0))
        if base_box is None or len(dot_boxes) < 2 or dot_mid_info is None or (dot_side_info is None):
            self.reset_tracking_state()
            cv2.putText(display, 'YOLO: BasePlate / dot x2 NOT FOUND', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 1)
        elif raw_mid is None or raw_side is None:
            reasons = []
            if debug_mid is not None and raw_mid is None:
                reasons.append('mid:' + str(debug_mid.get('reason', '?')))
            if debug_side is not None and raw_side is None:
                reasons.append('side:' + str(debug_side.get('reason', '?')))
            reason_text = ' | '.join(reasons) if reasons else 'unknown'
            cv2.putText(display, 'Hole refinement failed: ' + reason_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 255), 1)
        else:
            dot_mid, dot_side, tracking_state, measure_count = self.update_tracking_state(raw_mid, raw_side)
            if dot_mid is not None and dot_side is not None:
                mid_x, mid_y = dot_mid
                side_x, side_y = dot_side
                cv2.drawMarker(display, (int(round(mid_x)), int(round(mid_y))), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=12, thickness=1)
                cv2.drawMarker(display, (int(round(side_x)), int(round(side_y))), (0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=12, thickness=1)
                cv2.line(display, (int(round(mid_x)), int(round(mid_y))), (int(round(side_x)), int(round(side_y))), (255, 255, 0), 1)
                dx = side_x - mid_x
                dy = side_y - mid_y
                angle_px = math.degrees(math.atan2(dy, dx))
                distance_px = math.hypot(dx, dy)
                raw_delta_mid = math.hypot(raw_mid[0] - mid_x, raw_mid[1] - mid_y)
                raw_delta_side = math.hypot(raw_side[0] - side_x, raw_side[1] - side_y)
                if tracking_state == 'STABLE':
                    status_color = (0, 255, 0)
                elif tracking_state == 'MEASURING':
                    status_color = (0, 255, 255)
                else:
                    status_color = (0, 165, 255)
                cv2.putText(display, f'dot_mid px: {mid_x:.3f}, {mid_y:.3f}', (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
                cv2.putText(display, f'dot_side px: {side_x:.3f}, {side_y:.3f}', (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
                cv2.putText(display, f'{tracking_state} {measure_count}/{self.MEASURE_FRAMES} raw:{raw_delta_mid:.2f}/{raw_delta_side:.2f}px', (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.42, status_color, 1)
                cv2.putText(display, f'Angle:{angle_px:.2f}deg Dist:{distance_px:.2f}px', (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 0), 1)
                if debug_mid is not None and debug_side is not None:
                    mid_candidate = debug_mid.get('candidate')
                    side_candidate = debug_side.get('candidate')
                    if mid_candidate is not None and side_candidate is not None:
                        cv2.putText(display, f"Radial cov:{mid_candidate['final_coverage']:.2f}/{side_candidate['final_coverage']:.2f} err:{mid_candidate['fit_rmse']:.2f}/{side_candidate['fit_rmse']:.2f}px", (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
                if self.h_ready:
                    mid_mx, mid_my = self.pixel_to_mm(mid_x, mid_y)
                    side_mx, side_my = self.pixel_to_mm(side_x, side_y)
                    mm_dx = side_mx - mid_mx
                    mm_dy = side_my - mid_my
                    angle_mm = math.degrees(math.atan2(mm_dy, mm_dx))
                    distance_mm = math.hypot(mm_dx, mm_dy)
                    cv2.putText(display, f'dot_mid mm: {mid_mx:.3f}, {mid_my:.3f}', (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
                    cv2.putText(display, f'dot_side mm: {side_mx:.3f}, {side_my:.3f}', (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
                    cv2.putText(display, f'mm Angle:{angle_mm:.2f}deg Dist:{distance_mm:.3f}mm', (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 0), 1)

                    # 안정화가 끝난 좌표만 ROS2로 1회 publish
                    if tracking_state == 'STABLE':
                        self.publish_hole_coordinates(
                            mid_mx, mid_my,
                            side_mx, side_my,
                            angle_mm, distance_mm,
                        )
        for index, pt in enumerate(self.clicked_pts):
            cv2.circle(display, pt, 4, (0, 255, 255), -1)
            cv2.putText(display, str(index + 1), (pt[0] + 5, pt[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        radial_debug = self.make_binary_debug(debug_mid, debug_side)
        if radial_debug is not None:
            cv2.imshow(BINARY_WINDOW_NAME, radial_debug)
        cv2.imshow(WINDOW_NAME, display)
        key = cv2.waitKey(1) & 255
        if key == ord('r'):
            self.clicked_pts = []
            self.H = None
            self.h_ready = False
            self.reset_tracking_state()
            self.get_logger().info('A3 calibration / history reset')
        elif key == ord('q') or key == 27:
            self.get_logger().info('사용자 종료 요청')
            if rclpy.ok():
                rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    node = PrecisionHoleDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        cv2.destroyAllWindows()
if __name__ == '__main__':
    main()


