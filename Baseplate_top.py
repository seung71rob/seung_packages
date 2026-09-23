#!/usr/bin/env python3

import json
import os
import time
from collections import deque

import cv2
import numpy as np
import rclpy

from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
from ultralytics import YOLO


# ============================================================
# 설정
# ============================================================

MODEL_PATH = '/home/seung/robot_sim/top.pt'

CONF_THRESHOLD = 0.28
IOU_THRESHOLD = 0.45

# BasePlate 좌표 계산에 사용할 최소 confidence
BASEPLATE_COORD_MIN_CONF = 0.25

WINDOW_NAME = 'BasePlate Top Inspection'

# A3 실제 크기 [mm]
A3_W_MM = 420.0
A3_H_MM = 297.0

# Short shot mask 투명도
MASK_ALPHA = 0.30

# 로봇 auto 모드 busy flag
BUSY_FLAG_PATH = '/tmp/robot_auto_busy.flag'


# ============================================================
# 좌표 안정화 설정 (Wonseok 파일에서 이식)
# ============================================================

# EMA smoothing 강도. 클수록 이전값을 더 오래 유지 (0.0~1.0)
SMOOTH_ALPHA = 0.70

# 정지 판정: 이 픽셀 이내 움직임이 STATIONARY_HOLD_SEC 동안 유지되면 정지 상태
STATIONARY_PIXEL_THRESHOLD = 4.0
STATIONARY_HOLD_SEC = 0.9

# 이 이상 크게 움직이면 새 물체 위치로 간주하고 안정화 재시작
MOVE_RESET_THRESHOLD = 12.0

# 최종 확정 조건
# raw center와 smooth center의 차이가 이 픽셀 이하이고,
# 그 상태로 FINAL_LOCK_FRAMES_REQUIRED 프레임 이상 유지되어야 확정
FINAL_LOCK_PIXEL_THRESHOLD = 3.0
FINAL_LOCK_FRAMES_REQUIRED = 8

# 최종 좌표는 최근 안정 구간의 raw center들을 median 처리
FINAL_MEDIAN_SAMPLES = 15


# ============================================================
# 클래스 이름
# ============================================================

BASEPLATE_LABELS = {
    'baseplate',
    'base_plate',
    'base plate',
}

DARK_SPOT_LABELS = {
    'dark spot',
    'dark_spot',
    'darkspot',
}

SHORT_SHOT_LABELS = {
    'short shot',
    'short_shot',
    'shortshot',
}


# ============================================================
# 색상 (OpenCV = BGR)
# ============================================================

BASEPLATE_COLOR = (255, 0, 0)      # 파랑
DARK_SPOT_COLOR = (0, 0, 255)      # 빨강
SHORT_SHOT_COLOR = (0, 165, 255)   # 주황
CENTER_COLOR = (255, 255, 0)       # 시안
CAL_POINT_COLOR = (0, 255, 255)    # 노랑
OK_COLOR = (0, 255, 0)             # 초록
WARN_COLOR = (0, 165, 255)         # 주황
NO_OBJECT_COLOR = (0, 0, 255)      # 빨강
AUX_TEXT_COLOR = (255, 255, 255)   # 흰색

BOX_THICKNESS = 1
TEXT_THICKNESS = 1
FONT = cv2.FONT_HERSHEY_SIMPLEX


class BaseplateTopInspector(Node):

    def __init__(self):
        super().__init__('baseplate_top')

        self.bridge = CvBridge()

        # ========================================================
        # YOLO 모델
        # ========================================================
        self.model = YOLO(MODEL_PATH)

        self.get_logger().info(f'Model path: {MODEL_PATH}')
        self.get_logger().info(f'Model task: {self.model.task}')
        self.get_logger().info(f'Model classes: {self.model.names}')

        if self.model.task != 'segment':
            self.get_logger().warning(
                '현재 모델 task가 segment가 아닙니다. '
                'Short shot Polygon을 사용할 수 없습니다.'
            )

        # ========================================================
        # A3 Calibration
        # 클릭 순서: 좌상 -> 우상 -> 우하 -> 좌하
        # ========================================================
        self.clicked_pts = []
        self.H = None
        self.h_ready = False

        # ========================================================
        # BasePlate 중심 좌표
        # ========================================================
        self.baseplate_center_px = None
        self.baseplate_center_mm = None
        self.baseplate_confidence = None

        # ========================================================
        # 좌표 안정화 상태 (Wonseok 파일에서 이식)
        # ========================================================
        # EMA smoothing 상태
        self.prev_cx = None
        self.prev_cy = None

        # 정지 판정
        self.last_center_px = None
        self.stationary_start_time = None

        # 최종 확정
        self.sent_flag = False              # 한 번 전송했으면 True (재이동 전엔 재전송 안 함)
        self.final_lock_count = 0           # 확정 조건 연속 만족 프레임 수
        self.stable_raw_centers = deque(    # 최종 median 계산용 raw center 버퍼
            maxlen=FINAL_MEDIAN_SAMPLES
        )

        # 확정된 좌표 (값만 보관, 화면에는 표시하지 않음)
        self.locked_px = None
        self.locked_mm = None

        # ========================================================
        # Camera Subscribe
        # ========================================================
        self.subscription = self.create_subscription(
            Image,
            'image_raw',
            self.listener_callback,
            10
        )

        # ========================================================
        # 검사 결과 Publish
        # ========================================================
        self.inspection_pub = self.create_publisher(
            String,
            '/inspection_state',
            10
        )

        # ★ /detected_object 좌표 발행 (Wonseok 파일에서 이식)
        self.det_pub = self.create_publisher(
            String,
            '/detected_object',
            10
        )

        # ========================================================
        # OpenCV Window
        # ========================================================
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW_NAME, self.on_mouse)

        self.get_logger().info('BasePlate Top Inspection Started')
        self.get_logger().info('BasePlate  -> Bounding Box')
        self.get_logger().info('Dark spot  -> Bounding Box')
        self.get_logger().info('Short shot -> Segmentation Mask')
        self.get_logger().info('A3 좌상 -> 우상 -> 우하 -> 좌하 순서로 4점을 클릭하세요.')
        self.get_logger().info("'r' 키를 누르면 A3 calibration이 초기화됩니다.")
        self.get_logger().info('좌표 안정화 후 /detected_object 로 1회 발행합니다.')

    @staticmethod
    def normalize_label(label):
        s = str(label).strip().lower()
        s = s.replace('_', ' ')
        s = s.replace('-', ' ')
        return ' '.join(s.split())

    # ============================================================
    # Calibration
    # ============================================================

    def on_mouse(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if self.h_ready:
            return

        self.clicked_pts.append((x, y))
        self.get_logger().info(f'A3 click ({x}, {y}) {len(self.clicked_pts)}/4')

        if len(self.clicked_pts) == 4:
            src = np.array(self.clicked_pts, dtype=np.float32)
            dst = np.array(
                [
                    [0.0, 0.0],
                    [A3_W_MM, 0.0],
                    [A3_W_MM, A3_H_MM],
                    [0.0, A3_H_MM]
                ],
                dtype=np.float32
            )
            self.H = cv2.getPerspectiveTransform(src, dst).astype(np.float64)
            self.h_ready = True
            self.get_logger().info('A3 pixel -> mm 변환 준비 완료')

    def pixel_to_mm(self, px, py):
        """
        Wonseok 파일과 같이 float64 homogeneous transform 으로 계산.
        cv2.perspectiveTransform(float32) 보다 정밀도가 조금 높다.
        """
        if not self.h_ready or self.H is None:
            return None, None

        p = np.array([float(px), float(py), 1.0], dtype=np.float64)
        q = self.H @ p

        if abs(q[2]) < 1e-12:
            return None, None

        x_mm = q[0] / q[2]
        y_mm = q[1] / q[2]
        return float(x_mm), float(y_mm)

    @staticmethod
    def is_inside_workspace_mm(x_mm, y_mm):
        if x_mm is None or y_mm is None:
            return False
        return (0.0 <= x_mm <= A3_W_MM) and (0.0 <= y_mm <= A3_H_MM)

    # ============================================================
    # 안정화 관련 (Wonseok 파일에서 이식)
    # ============================================================

    def reset_tracking(self, full=False):
        """
        안정화 상태를 초기화한다.
        full=True 이면 EMA 이력까지 완전히 지운다 (BasePlate가 사라졌을 때 등).
        """
        self.sent_flag = False
        self.final_lock_count = 0
        self.stable_raw_centers.clear()

        self.last_center_px = None
        self.stationary_start_time = None

        if full:
            self.prev_cx = None
            self.prev_cy = None

        self.locked_px = None
        self.locked_mm = None

    def smooth_center(self, cx, cy):
        """지수이동평균(EMA) smoothing."""
        cx = float(cx)
        cy = float(cy)

        if self.prev_cx is None:
            self.prev_cx = cx
            self.prev_cy = cy
            return cx, cy

        self.prev_cx = SMOOTH_ALPHA * self.prev_cx + (1.0 - SMOOTH_ALPHA) * cx
        self.prev_cy = SMOOTH_ALPHA * self.prev_cy + (1.0 - SMOOTH_ALPHA) * cy
        return self.prev_cx, self.prev_cy

    def is_stationary(self, cx, cy):
        """
        smoothing된 center 가 STATIONARY_PIXEL_THRESHOLD 이내에서
        STATIONARY_HOLD_SEC 동안 유지되면 정지 상태로 간주.
        반환: (stationary_ok, move_dist)
        """
        now = time.time()

        if self.last_center_px is None:
            self.last_center_px = (float(cx), float(cy))
            self.stationary_start_time = now
            return False, 0.0

        dist = float(
            np.hypot(
                float(cx) - self.last_center_px[0],
                float(cy) - self.last_center_px[1],
            )
        )

        if dist <= STATIONARY_PIXEL_THRESHOLD:
            hold_time = now - self.stationary_start_time
            return (hold_time >= STATIONARY_HOLD_SEC), dist

        # 움직였다고 판단 -> 정지 시계 리셋
        self.last_center_px = (float(cx), float(cy))
        self.stationary_start_time = now

        # 큰 이동은 안정화 상태도 리셋
        self.final_lock_count = 0
        self.stable_raw_centers.clear()
        self.sent_flag = False

        return False, dist

    @staticmethod
    def robot_is_busy():
        """로봇이 auto mode로 작업 중이면 좌표를 보내지 않는다."""
        return os.path.exists(BUSY_FLAG_PATH)

    # ============================================================
    # Publish 검사 상태
    # ============================================================

    def publish_inspection_state(
        self,
        has_baseplate=False,
        has_dark_spot=False,
        dark_spot_count=0,
        has_short_shot=False,
        short_shot_count=0
    ):
        has_ng = has_dark_spot or has_short_shot

        if has_baseplate and has_ng:
            judge = 'NG'
        elif has_baseplate:
            judge = 'OK'
        else:
            judge = 'NO_OBJECT'

        payload = {
            'has_baseplate': bool(has_baseplate),
            'has_dark_spot': bool(has_dark_spot),
            'dark_spot_count': int(dark_spot_count),
            'has_short_shot': bool(has_short_shot),
            'short_shot_count': int(short_shot_count),
            'judge': judge,
            'stamp': time.time()
        }

        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self.inspection_pub.publish(msg)
        return judge

    # ============================================================
    # 시각화 (bbox_wonseok 스타일 유지)
    # ============================================================

    def draw_bounding_box(self, frame, box, color, label, confidence):
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, BOX_THICKNESS)
        cv2.putText(
            frame,
            f'{label} {confidence:.2f}',
            (x1, max(20, y1 - 6)),
            FONT,
            0.45,
            color,
            TEXT_THICKNESS,
            cv2.LINE_AA
        )

    def draw_short_shot_mask(self, frame, polygon, confidence):
        if polygon is None or len(polygon) < 3:
            return

        pts = np.asarray(polygon, dtype=np.int32).reshape((-1, 1, 2))
        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], SHORT_SHOT_COLOR)
        cv2.addWeighted(overlay, MASK_ALPHA, frame, 1.0 - MASK_ALPHA, 0, frame)
        cv2.polylines(frame, [pts], True, SHORT_SHOT_COLOR, BOX_THICKNESS, cv2.LINE_AA)

        x, y, _, _ = cv2.boundingRect(pts)
        cv2.putText(
            frame,
            f'short_shot {confidence:.2f}',
            (x, max(20, y - 6)),
            FONT,
            0.42,
            SHORT_SHOT_COLOR,
            TEXT_THICKNESS,
            cv2.LINE_AA
        )

    def draw_baseplate_center_marker(self, frame, cx, cy):
        center_x = int(round(cx))
        center_y = int(round(cy))

        cv2.drawMarker(
            frame,
            (center_x, center_y),
            CENTER_COLOR,
            markerType=cv2.MARKER_CROSS,
            markerSize=12,
            thickness=1
        )
        cv2.circle(frame, (center_x, center_y), 4, CENTER_COLOR, 1, cv2.LINE_AA)

    def draw_calibration_points(self, frame):
        for index, pt in enumerate(self.clicked_pts):
            cv2.circle(frame, pt, 4, CAL_POINT_COLOR, -1)
            cv2.putText(
                frame,
                str(index + 1),
                (pt[0] + 5, pt[1] - 5),
                FONT,
                0.4,
                CAL_POINT_COLOR,
                1,
                cv2.LINE_AA
            )

    def draw_overlay_text(
        self,
        frame,
        judge,
        has_baseplate,
        has_dark_spot,
        dark_spot_count,
        has_short_shot,
        short_shot_count,
        center_px,
        center_mm,
        coord_status,
    ):
        # 1) 판정 문구
        if judge == 'OK':
            cv2.putText(frame, 'OK - BASEPLATE', (20, 35), FONT, 0.65, OK_COLOR, 1, cv2.LINE_AA)
        elif judge == 'NG':
            if has_dark_spot and has_short_shot:
                cv2.putText(frame, f'NG - DARK SPOT:{dark_spot_count} / SHORT SHOT:{short_shot_count}', (20, 35), FONT, 0.55, NO_OBJECT_COLOR, 1, cv2.LINE_AA)
            elif has_dark_spot:
                cv2.putText(frame, f'NG - DARK SPOT ({dark_spot_count})', (20, 35), FONT, 0.60, DARK_SPOT_COLOR, 1, cv2.LINE_AA)
            else:
                cv2.putText(frame, f'NG - SHORT SHOT ({short_shot_count})', (20, 35), FONT, 0.60, SHORT_SHOT_COLOR, 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, 'NO BASEPLATE', (20, 35), FONT, 0.65, WARN_COLOR, 1, cv2.LINE_AA)

        # 2) Calibration 상태
        if self.h_ready:
            cv2.putText(frame, 'A3 CALIBRATION READY', (20, 58), FONT, 0.55, OK_COLOR, 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, f'A3 CALIBRATION {len(self.clicked_pts)}/4', (20, 58), FONT, 0.55, CAL_POINT_COLOR, 1, cv2.LINE_AA)

        # 3) 부가 정보
        y = 82
        if center_px is not None:
            cx, cy = center_px
            cv2.putText(frame, f'Center px: {cx:.3f}, {cy:.3f}', (20, y), FONT, 0.45, CENTER_COLOR, 1, cv2.LINE_AA)
            y += 20
        if center_mm is not None:
            mx, my = center_mm
            cv2.putText(frame, f'Center mm: {mx:.3f}, {my:.3f}', (20, y), FONT, 0.45, CENTER_COLOR, 1, cv2.LINE_AA)
            y += 20

        # 4) 좌표 안정화 상태
        if coord_status:
            if 'PUBLISHED' in coord_status:
                color = OK_COLOR
            elif 'LOCKING' in coord_status:
                color = CAL_POINT_COLOR
            elif 'BUSY' in coord_status or 'OUTSIDE' in coord_status or 'REQUIRED' in coord_status:
                color = WARN_COLOR
            else:
                color = AUX_TEXT_COLOR
            cv2.putText(frame, f'Coord: {coord_status}', (20, y), FONT, 0.45, color, 1, cv2.LINE_AA)
            y += 20

        if has_dark_spot:
            cv2.putText(frame, f'Dark spot count: {dark_spot_count}', (20, y), FONT, 0.42, DARK_SPOT_COLOR, 1, cv2.LINE_AA)
            y += 18
        if has_short_shot:
            cv2.putText(frame, f'Short shot count: {short_shot_count}', (20, y), FONT, 0.42, SHORT_SHOT_COLOR, 1, cv2.LINE_AA)

    # ============================================================
    # Image callback
    # ============================================================

    def listener_callback(self, data):
        try:
            frame = self.bridge.imgmsg_to_cv2(data, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge error: {e}')
            return

        display_frame = frame.copy()
        results = self.model(frame, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)
        result = results[0]
        boxes = result.boxes
        masks = result.masks
        polygons = list(masks.xy) if masks is not None else []

        has_baseplate = False
        has_dark_spot = False
        dark_spot_count = 0
        has_short_shot = False
        short_shot_count = 0

        best_base_box = None
        best_base_conf = -1.0

        # ========================================================
        # 검출 처리
        # ========================================================
        if boxes is not None and len(boxes) > 0:
            for index, box in enumerate(boxes):
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                label = self.model.names[class_id]
                label_l = self.normalize_label(label)

                if label_l in BASEPLATE_LABELS:
                    has_baseplate = True
                    self.draw_bounding_box(display_frame, box, BASEPLATE_COLOR, 'BasePlate', confidence)
                    # 좌표 계산은 최소 confidence 이상인 것 중 최고를 사용
                    if (confidence > best_base_conf) and (confidence >= BASEPLATE_COORD_MIN_CONF):
                        best_base_conf = confidence
                        best_base_box = box

                elif label_l in DARK_SPOT_LABELS:
                    has_dark_spot = True
                    dark_spot_count += 1
                    self.draw_bounding_box(display_frame, box, DARK_SPOT_COLOR, 'DarkSpot', confidence)

                elif label_l in SHORT_SHOT_LABELS:
                    has_short_shot = True
                    short_shot_count += 1
                    polygon = polygons[index] if index < len(polygons) else None
                    if polygon is not None and len(polygon) >= 3:
                        self.draw_short_shot_mask(display_frame, polygon, confidence)
                    else:
                        x1, y1, _, _ = map(int, box.xyxy[0].tolist())
                        cv2.putText(
                            display_frame,
                            f'short_shot {confidence:.2f} [NO MASK]',
                            (x1, max(20, y1 - 6)),
                            FONT,
                            0.42,
                            SHORT_SHOT_COLOR,
                            1,
                            cv2.LINE_AA
                        )

        # ========================================================
        # 검사 상태 publish (기존 그대로)
        # ========================================================
        judge = self.publish_inspection_state(
            has_baseplate=has_baseplate,
            has_dark_spot=has_dark_spot,
            dark_spot_count=dark_spot_count,
            has_short_shot=has_short_shot,
            short_shot_count=short_shot_count
        )

        # ========================================================
        # BasePlate 중심 좌표 + 안정화 + /detected_object 발행
        # ========================================================
        center_px = None
        center_mm = None
        coord_status = ''

        if best_base_box is not None:
            # 1) raw center (BasePlate bbox 정중앙)
            bx1, by1, bx2, by2 = best_base_box.xyxy[0].tolist()
            raw_cx = (bx1 + bx2) / 2.0
            raw_cy = (by1 + by2) / 2.0

            # 2) EMA smoothing
            smooth_cx, smooth_cy = self.smooth_center(raw_cx, raw_cy)

            # 3) 정지 판정
            stationary_ok, move_dist = self.is_stationary(smooth_cx, smooth_cy)

            # 큰 이동이면 새 위치로 즉시 재설정
            if move_dist > MOVE_RESET_THRESHOLD:
                self.sent_flag = False
                self.final_lock_count = 0
                self.stable_raw_centers.clear()
                # EMA 이력 삭제 -> 새 위치가 바로 반영
                self.prev_cx = float(raw_cx)
                self.prev_cy = float(raw_cy)
                smooth_cx = float(raw_cx)
                smooth_cy = float(raw_cy)

            # 4) raw와 smooth의 차이 (수렴도)
            lock_dist = float(np.hypot(raw_cx - smooth_cx, raw_cy - smooth_cy))

            # 5) mm 변환 (프리뷰)
            preview_x_mm = None
            preview_y_mm = None
            if self.h_ready:
                preview_x_mm, preview_y_mm = self.pixel_to_mm(smooth_cx, smooth_cy)

            inside_workspace = (
                self.is_inside_workspace_mm(preview_x_mm, preview_y_mm)
                if self.h_ready else False
            )

            busy = self.robot_is_busy()

            # 6) 확정 조건: 정지 & 수렴 & 워크스페이스 안 & 로봇 idle
            if (
                stationary_ok
                and lock_dist <= FINAL_LOCK_PIXEL_THRESHOLD
                and self.h_ready
                and inside_workspace
                and not busy
            ):
                self.final_lock_count += 1
                self.stable_raw_centers.append((float(raw_cx), float(raw_cy)))
            else:
                self.final_lock_count = 0
                if not stationary_ok:
                    self.stable_raw_centers.clear()

            # 7) 최종 발행 판단
            publish_ready = (
                self.h_ready
                and stationary_ok
                and self.final_lock_count >= FINAL_LOCK_FRAMES_REQUIRED
                and len(self.stable_raw_centers) >= FINAL_LOCK_FRAMES_REQUIRED
                and inside_workspace
                and not busy
                and not self.sent_flag
                and judge in ('OK', 'NG')
            )

            if publish_ready:
                samples = np.asarray(self.stable_raw_centers, dtype=np.float64)
                final_px = float(np.median(samples[:, 0]))
                final_py = float(np.median(samples[:, 1]))
                final_x_mm, final_y_mm = self.pixel_to_mm(final_px, final_py)

                if self.is_inside_workspace_mm(final_x_mm, final_y_mm):
                    msg = String()
                    msg.data = (
                        'baseplate,'
                        f'{final_x_mm:.1f},'
                        f'{final_y_mm:.1f},'
                        '0.0,0,'
                        f'{judge}'
                    )
                    self.det_pub.publish(msg)
                    self.sent_flag = True
                    self.locked_px = (final_px, final_py)
                    self.locked_mm = (final_x_mm, final_y_mm)
                    self.get_logger().info(f'★ 안정화된 BasePlate 좌표 전송: {msg.data}')

            # 8) 상태 문자열
            if not self.h_ready:
                coord_status = 'A3 CALIBRATION REQUIRED'
            elif busy:
                coord_status = 'ROBOT BUSY'
            elif not inside_workspace:
                coord_status = 'OUTSIDE WORKSPACE'
            elif self.sent_flag:
                coord_status = 'PUBLISHED'
            elif stationary_ok:
                coord_status = f'LOCKING {self.final_lock_count}/{FINAL_LOCK_FRAMES_REQUIRED}'
            else:
                coord_status = 'STABILIZING'

            # 9) 시각화용 최신 값 저장
            center_px = (smooth_cx, smooth_cy)
            self.baseplate_center_px = center_px
            self.baseplate_confidence = best_base_conf

            if self.h_ready:
                center_mm = (preview_x_mm, preview_y_mm)
                self.baseplate_center_mm = center_mm
            else:
                self.baseplate_center_mm = None

            # 10) 화면에 중심 마커 (확정 마커는 그리지 않는다)
            self.draw_baseplate_center_marker(display_frame, smooth_cx, smooth_cy)

        else:
            # BasePlate가 없거나 confidence가 너무 낮으면 안정화 상태를 리셋
            self.baseplate_center_px = None
            self.baseplate_center_mm = None
            self.baseplate_confidence = None
            self.reset_tracking(full=True)
            coord_status = 'NO RELIABLE BASEPLATE'

        # ========================================================
        # Overlay text + calibration point
        # ========================================================
        self.draw_overlay_text(
            display_frame,
            judge,
            has_baseplate,
            has_dark_spot,
            dark_spot_count,
            has_short_shot,
            short_shot_count,
            center_px,
            center_mm,
            coord_status,
        )

        self.draw_calibration_points(display_frame)

        cv2.imshow(WINDOW_NAME, display_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            self.clicked_pts = []
            self.H = None
            self.h_ready = False
            self.baseplate_center_px = None
            self.baseplate_center_mm = None
            self.baseplate_confidence = None
            self.reset_tracking(full=True)
            self.get_logger().info('A3 calibration reset')

    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = BaseplateTopInspector()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
