#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BasePlate Side Inspection - portable version for wonseok laptop

기능
- Side camera ROS2 Image 구독
- Bottom view: bottom.pt 사용
- Back view: side.pt 사용
- Flash view: flash.pt 사용
- Dark spot / Short shot / Flash 결함 표시
- 검사 결과를 JSON String으로 publish
- 로봇의 검사 START/STOP 명령을 ROS2로 구독
- 검사 자세마다 3초 동안 검사하고 불량 누적 2초 이상이면 NG 결과 publish
- NG 확정 시 로봇 시퀀스 시작 이벤트를 해당 검사에서 한 번만 publish
- 모델 경로를 같은 폴더에서 자동 탐색하거나 ROS2 parameter로 지정 가능
- image topic도 ROS2 parameter로 변경 가능

기본 토픽
  image subscribe   : /side/image_raw
  control subscribe : /baseplate_side/inspection_control
  frame state       : /inspection_state_side
  view result       : /baseplate_side/view_result
  NG trigger        : /baseplate_side/defect_sequence_trigger
"""

import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
from ultralytics import YOLO

CONF_THRESHOLD_DEFAULT = 0.15
IOU_THRESHOLD_DEFAULT = 0.45
WINDOW_NAME_DEFAULT = "BasePlate Side Inspection"
MASK_ALPHA = 0.30
INSPECTION_DURATION_SEC_DEFAULT = 3.0
REQUIRED_DEFECT_SEC_DEFAULT = 2.0
MAX_SAMPLE_GAP_SEC_DEFAULT = 0.25

DARK_SPOT_LABELS = {"dark spot"}
SHORT_SHOT_LABELS = {"short shot"}
FLASH_LABELS = {"flash"}

DARK_SPOT_COLOR = (0, 0, 255)
SHORT_SHOT_COLOR = (0, 165, 255)
FLASH_COLOR = (255, 0, 255)


def normalize_label(label):
    return (
        str(label)
        .strip()
        .lower()
        .replace("_", " ")
        .replace("-", " ")
    )


def find_model_path(explicit_path=""):
    if explicit_path:
        p = Path(os.path.expanduser(explicit_path))
        if p.is_file():
            return str(p.resolve())
        raise FileNotFoundError(f"지정한 model_path 파일이 없습니다: {p}")

    env_path = os.environ.get("BASEPLATE_SIDE_MODEL_PATH", "").strip()
    if env_path:
        p = Path(os.path.expanduser(env_path))
        if p.is_file():
            return str(p.resolve())
        raise FileNotFoundError(f"BASEPLATE_SIDE_MODEL_PATH 파일이 없습니다: {p}")

    here = Path(__file__).resolve().parent
    candidates = [
        "Baseplate_top(2).pt",
        "Baseplate_top.pt",
        "Baseplate_top(1).pt",
        "baseplate_top.pt",
        "Baseplate_side.pt",
        "baseplate_side.pt",
    ]

    for name in candidates:
        p = here / name
        if p.is_file():
            return str(p)

    pts = sorted(here.glob("*.pt"))
    if len(pts) == 1:
        return str(pts[0])

    raise FileNotFoundError(
        "YOLO .pt 모델을 찾지 못했습니다.\n"
        f"현재 폴더: {here}\n"
        "Baseplate_top(2).pt를 이 Python 파일과 같은 폴더에 두거나,\n"
        '--ros-args -p model_path:="/절대/경로/model.pt" 로 실행하세요.'
    )


class BaseplateSideInspector(Node):
    def __init__(self):
        super().__init__("baseplate_side")

        model_dir = Path(__file__).resolve().parent
        # model_path는 이전 실행 명령과의 호환용이다. 값이 지정되면 세 view가
        # 모두 그 모델을 사용하고, 비어 있으면 view별 기본 모델을 사용한다.
        self.declare_parameter("model_path", "")
        self.declare_parameter("bottom_model_path", str(model_dir / "bottom.pt"))
        self.declare_parameter("back_model_path", str(model_dir / "side.pt"))
        self.declare_parameter("flash_model_path", str(model_dir / "flash.pt"))
        self.declare_parameter("image_topic", "/side/image_raw")
        self.declare_parameter("inspection_topic", "/inspection_state_side")
        self.declare_parameter(
            "inspection_control_topic",
            "/baseplate_side/inspection_control",
        )
        self.declare_parameter(
            "view_result_topic",
            "/baseplate_side/view_result",
        )
        self.declare_parameter(
            "sequence_trigger_topic",
            "/baseplate_side/defect_sequence_trigger",
        )
        self.declare_parameter("conf_threshold", CONF_THRESHOLD_DEFAULT)
        self.declare_parameter("iou_threshold", IOU_THRESHOLD_DEFAULT)
        self.declare_parameter("window_name", WINDOW_NAME_DEFAULT)
        self.declare_parameter(
            "inspection_duration_sec",
            INSPECTION_DURATION_SEC_DEFAULT,
        )
        self.declare_parameter(
            "required_defect_sec",
            REQUIRED_DEFECT_SEC_DEFAULT,
        )
        self.declare_parameter(
            "max_sample_gap_sec",
            MAX_SAMPLE_GAP_SEC_DEFAULT,
        )

        model_param = str(self.get_parameter("model_path").value).strip()
        bottom_model_param = str(self.get_parameter("bottom_model_path").value).strip()
        back_model_param = str(self.get_parameter("back_model_path").value).strip()
        flash_model_param = str(self.get_parameter("flash_model_path").value).strip()
        self.image_topic = str(self.get_parameter("image_topic").value)
        self.inspection_topic = str(self.get_parameter("inspection_topic").value)
        self.inspection_control_topic = str(
            self.get_parameter("inspection_control_topic").value
        )
        self.view_result_topic = str(
            self.get_parameter("view_result_topic").value
        )
        self.sequence_trigger_topic = str(
            self.get_parameter("sequence_trigger_topic").value
        )
        self.conf_threshold = float(self.get_parameter("conf_threshold").value)
        self.iou_threshold = float(self.get_parameter("iou_threshold").value)
        self.window_name = str(self.get_parameter("window_name").value)
        self.inspection_duration_sec = max(
            0.0,
            float(self.get_parameter("inspection_duration_sec").value),
        )
        self.required_defect_sec = max(
            0.0,
            float(self.get_parameter("required_defect_sec").value),
        )
        self.max_sample_gap_sec = max(
            0.0,
            float(self.get_parameter("max_sample_gap_sec").value),
        )

        if self.required_defect_sec > self.inspection_duration_sec:
            raise ValueError(
                "required_defect_sec must not exceed inspection_duration_sec"
            )

        # One inspection session is created for every START command.
        self.inspection_active = False
        self.inspection_view = ""
        self.inspection_cycle_id = ""
        self.inspection_request_id = ""
        self.inspection_started_at = None
        self.last_sample_at = None
        self.defect_accum_sec = 0.0
        self.inspection_sample_count = 0
        self.max_dark_spot_count = 0
        self.max_short_shot_count = 0
        self.max_flash_count = 0
        self.view_result_id = 0
        self.sequence_event_id = 0
        self.last_view_result = None

        self.bridge = CvBridge()

        def checked_model_path(raw_path, view_name):
            path = Path(os.path.expanduser(raw_path))
            if not path.is_file():
                raise FileNotFoundError(
                    f"{view_name} 모델 파일이 없습니다: {path}"
                )
            return str(path.resolve())

        # 기존 -p model_path:=...가 있으면 이전처럼 한 모델을 공통 사용한다.
        if model_param:
            bottom_model_param = back_model_param = flash_model_param = model_param

        self.model_paths = {
            "bottom": checked_model_path(bottom_model_param, "bottom view"),
            "back": checked_model_path(back_model_param, "back view"),
            "flash": checked_model_path(flash_model_param, "flash view"),
        }
        # 검사 START 후 모델을 로드하면 검사 시간이 손실되므로 시작할 때 모두 로드한다.
        self.models = {
            key: YOLO(path) for key, path in self.model_paths.items()
        }
        self.active_model_key = "bottom"
        self.active_model = self.models[self.active_model_key]

        for key in ("bottom", "back", "flash"):
            model = self.models[key]
            self.get_logger().info(
                f"{key.upper()} model: {self.model_paths[key]} | "
                f"task={model.task} | classes={model.names}"
            )
        self.get_logger().info(f"Image topic: {self.image_topic}")
        self.get_logger().info(f"Inspection topic: {self.inspection_topic}")
        self.get_logger().info(
            f"Inspection control topic: {self.inspection_control_topic}"
        )
        self.get_logger().info(f"View result topic: {self.view_result_topic}")
        self.get_logger().info(
            f"Sequence trigger topic: {self.sequence_trigger_topic}"
        )
        self.get_logger().info(
            f"View inspection: duration={self.inspection_duration_sec:.2f}s, "
            f"required consecutive defect={self.required_defect_sec:.2f}s"
        )

        for key, model in self.models.items():
            if str(model.task).lower() != "segment":
                self.get_logger().warn(
                    f"{key} 모델 task가 segment가 아닙니다. "
                    "마스크 대신 Bounding Box로 표시될 수 있습니다."
                )

        self.subscription = self.create_subscription(
            Image,
            self.image_topic,
            self.listener_callback,
            qos_profile_sensor_data,
        )

        self.inspection_control_sub = self.create_subscription(
            String,
            self.inspection_control_topic,
            self.inspection_control_callback,
            10,
        )

        self.inspection_pub = self.create_publisher(
            String,
            self.inspection_topic,
            10,
        )

        self.view_result_pub = self.create_publisher(
            String,
            self.view_result_topic,
            10,
        )

        self.sequence_trigger_pub = self.create_publisher(
            String,
            self.sequence_trigger_topic,
            10,
        )

        self.create_timer(0.1, self.inspection_watchdog)
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        self.get_logger().info("BasePlate Side Inspection Started")
        self.get_logger().info("Dark spot  -> Bounding Box")
        self.get_logger().info("Short shot -> Segmentation Mask (없으면 BBox fallback)")
        self.get_logger().info("종료: 영상창에서 q / ESC 또는 터미널 Ctrl+C")

    def inspection_control_callback(self, msg):
        """Receive START/STOP commands from the robot controller."""
        raw = msg.data.strip()
        data = {}
        try:
            data = json.loads(raw)
            command = str(data.get("command", "")).strip().upper()
            view = str(data.get("view", "unknown")).strip() or "unknown"
            cycle_id = str(data.get("cycle_id", "")).strip()
        except Exception:
            # Simple text is also accepted: "START left" or "STOP".
            parts = raw.split()
            command = parts[0].upper() if parts else ""
            view = parts[1] if len(parts) >= 2 else "unknown"
            cycle_id = parts[2] if len(parts) >= 3 else ""

        if command == "START":
            if self.inspection_active:
                self.publish_view_result(
                    judge="INCOMPLETE",
                    reason="superseded_by_new_start",
                )
            self.start_view_inspection(view, cycle_id, data.get('request_id','') if isinstance(data,dict) else '')
        elif command == "STOP":
            if self.inspection_active:
                self.publish_view_result(
                    judge="INCOMPLETE",
                    reason="stopped_before_result",
                )
            else:
                self.get_logger().info("Inspection STOP received while idle")
        else:
            self.get_logger().warning(
                f"Unknown inspection command: {raw!r}. Use START or STOP."
            )

    @staticmethod
    def model_key_for_view(view):
        view_upper = str(view).strip().upper()
        if view_upper.startswith("B_BOTTOM_VIEW"):
            return "bottom"
        if view_upper.startswith("B_BACK_VIEW"):
            return "back"
        if view_upper.startswith("B_FLASH_VIEW"):
            return "flash"
        return None

    def start_view_inspection(self, view, cycle_id='', request_id=''):
        model_key = self.model_key_for_view(view)
        if model_key is None:
            self.get_logger().warning(
                f"알 수 없는 view={view!r}; bottom 모델을 사용합니다."
            )
            model_key = "bottom"
        self.active_model_key = model_key
        self.active_model = self.models[model_key]

        now = time.monotonic()
        self.inspection_active = True
        self.inspection_view = view
        self.inspection_cycle_id = cycle_id
        self.inspection_request_id = request_id
        # This three-view process has fixed 3s windows and 2s consecutive NG.
        self.inspection_duration_sec = 3.0
        self.required_defect_sec = 2.0
        self.inspection_started_at = now
        self.last_sample_at = now
        self.defect_accum_sec = 0.0  # compatibility key: now CONSECUTIVE duration
        self.previous_was_ng = False
        self.session_gap = False
        self.inspection_sample_count = 0
        self.max_dark_spot_count = 0
        self.max_short_shot_count = 0
        self.max_flash_count = 0
        self.last_view_result = None
        self.get_logger().info(
            f"START {view} request={request_id} "
            f"model={self.model_paths[model_key]}"
        )


    def publish_view_result(self, judge, reason):
        now = time.monotonic()
        elapsed_sec = (
            0.0
            if self.inspection_started_at is None
            else now - self.inspection_started_at
        )
        self.view_result_id += 1

        result = {
            "event": "VIEW_INSPECTION_RESULT",
            "result_id": self.view_result_id,
            "camera": "baseplate_side",
            "cycle_id": self.inspection_cycle_id,
            "request_id": self.inspection_request_id,
            "view": self.inspection_view,
            "judge": judge,
            "reason": reason,
            "defect_accum_sec": round(self.defect_accum_sec, 3),
            "defect_consecutive_sec": round(self.defect_accum_sec, 3),
            "defect_mode": "consecutive",
            "required_defect_sec": self.required_defect_sec,
            "inspection_elapsed_sec": round(elapsed_sec, 3),
            "inspection_duration_sec": self.inspection_duration_sec,
            "sample_count": self.inspection_sample_count,
            "max_dark_spot_count": self.max_dark_spot_count,
            "max_short_shot_count": self.max_short_shot_count,
            "max_flash_count": self.max_flash_count,
            "model_key": self.active_model_key,
            "model_path": self.model_paths[self.active_model_key],
            "stamp_sec": self.get_clock().now().nanoseconds / 1_000_000_000.0,
        }

        result_msg = String()
        result_msg.data = json.dumps(result, ensure_ascii=False)
        self.view_result_pub.publish(result_msg)
        self.last_view_result = result
        self.inspection_active = False

        self.get_logger().info(
            f"VIEW RESULT: view={self.inspection_view}, "
            f"defect={self.defect_accum_sec:.2f}s -> {judge}"
        )

        if judge == "NG":
            self.sequence_event_id += 1
            event = {
                "event": "START_DEFECT_SEQUENCE",
                "event_id": self.sequence_event_id,
                "source_result_id": self.view_result_id,
                "camera": "baseplate_side",
                "cycle_id": self.inspection_cycle_id,
                "request_id": self.inspection_request_id,
                "view": self.inspection_view,
                "judge": "NG",
                "defect_accum_sec": round(self.defect_accum_sec, 3),
                "defect_consecutive_sec": round(self.defect_accum_sec, 3),
                "defect_mode": "consecutive",
                "max_dark_spot_count": self.max_dark_spot_count,
                "max_short_shot_count": self.max_short_shot_count,
                "max_flash_count": self.max_flash_count,
                "model_key": self.active_model_key,
                "stamp_sec": result["stamp_sec"],
            }
            trigger_msg = String()
            trigger_msg.data = json.dumps(event, ensure_ascii=False)
            self.sequence_trigger_pub.publish(trigger_msg)
            self.get_logger().warning(
                "NG CONFIRMED -> START_DEFECT_SEQUENCE published "
                f"(event_id={self.sequence_event_id}, view={self.inspection_view})"
            )

    def update_view_inspection(self, frame_payload):
        if not self.inspection_active: return
        now = time.monotonic()
        dt = max(0.0, now-self.last_sample_at)
        self.last_sample_at = now
        self.inspection_sample_count += 1
        ng = frame_payload['judge'] == 'NG'
        if dt > self.max_sample_gap_sec:
            self.session_gap = True
            self.defect_accum_sec = 0.0
        elif ng and self.previous_was_ng:
            self.defect_accum_sec += dt
        else:
            self.defect_accum_sec = 0.0
        self.previous_was_ng = ng
        self.max_dark_spot_count = max(self.max_dark_spot_count, frame_payload['dark_spot_count'])
        self.max_short_shot_count = max(self.max_short_shot_count, frame_payload['short_shot_count'])
        self.max_flash_count = max(self.max_flash_count, frame_payload['flash_count'])
        if self.defect_accum_sec + 1e-9 >= self.required_defect_sec:
            self.publish_view_result('NG', 'consecutive_defect_threshold_reached')
        elif now-self.inspection_started_at >= self.inspection_duration_sec:
            if self.session_gap:
                self.publish_view_result('INCOMPLETE', 'camera_sample_gap')
            else:
                self.publish_view_result('OK', 'inspection_window_completed')

    def inspection_watchdog(self):
        if not self.inspection_active: return
        now = time.monotonic()
        if (now-self.inspection_started_at >= self.inspection_duration_sec and
                now-self.last_sample_at > self.max_sample_gap_sec):
            self.publish_view_result('INCOMPLETE', 'camera_frames_missing')


    def draw_dark_spot(self, frame, box, confidence):
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cv2.rectangle(frame, (x1, y1), (x2, y2), DARK_SPOT_COLOR, 2)
        cv2.putText(
            frame,
            f"Dark spot {confidence:.2f}",
            (x1, max(25, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            DARK_SPOT_COLOR,
            2,
            cv2.LINE_AA,
        )

    def draw_flash(self, frame, polygon, box, confidence):
        if polygon is not None and len(polygon) >= 3:
            pts = np.asarray(polygon, dtype=np.int32).reshape((-1, 1, 2))
            overlay = frame.copy()
            cv2.fillPoly(overlay, [pts], FLASH_COLOR)
            cv2.addWeighted(overlay, MASK_ALPHA, frame, 1.0 - MASK_ALPHA, 0, frame)
            cv2.polylines(frame, [pts], True, FLASH_COLOR, 2, cv2.LINE_AA)
            x, y, _, _ = cv2.boundingRect(pts)
        else:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cv2.rectangle(frame, (x1, y1), (x2, y2), FLASH_COLOR, 2)
            x, y = x1, y1
        cv2.putText(
            frame,
            f"Flash {confidence:.2f}",
            (x, max(25, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            FLASH_COLOR,
            2,
            cv2.LINE_AA,
        )

    def draw_short_shot_mask(self, frame, polygon, box, confidence):
        if polygon is not None and len(polygon) >= 3:
            pts = np.asarray(polygon, dtype=np.int32).reshape((-1, 1, 2))
            overlay = frame.copy()
            cv2.fillPoly(overlay, [pts], SHORT_SHOT_COLOR)
            cv2.addWeighted(overlay, MASK_ALPHA, frame, 1.0 - MASK_ALPHA, 0, frame)
            cv2.polylines(frame, [pts], True, SHORT_SHOT_COLOR, 2, cv2.LINE_AA)
            x, y, _, _ = cv2.boundingRect(pts)
            cv2.putText(
                frame,
                f"Short shot {confidence:.2f}",
                (x, max(25, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                SHORT_SHOT_COLOR,
                2,
                cv2.LINE_AA,
            )
        else:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cv2.rectangle(frame, (x1, y1), (x2, y2), SHORT_SHOT_COLOR, 2)
            cv2.putText(
                frame,
                f"Short shot {confidence:.2f}",
                (x1, max(25, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                SHORT_SHOT_COLOR,
                2,
                cv2.LINE_AA,
            )

    def listener_callback(self, data):
        try:
            frame = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except Exception as exc:
            self.get_logger().error(f"cv_bridge error: {exc}")
            return

        display_frame = frame.copy()

        try:
            results = self.active_model(
                frame,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                verbose=False,
            )
        except Exception as exc:
            self.get_logger().error(f"YOLO inference error: {exc}")
            return

        result = results[0]
        boxes = result.boxes
        masks = result.masks

        has_dark_spot = False
        dark_spot_count = 0
        has_short_shot = False
        short_shot_count = 0
        has_flash = False
        flash_count = 0

        polygons = masks.xy if masks is not None else []

        if boxes is not None and len(boxes) > 0:
            for index, box in enumerate(boxes):
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                label = self.active_model.names[class_id]
                label_l = normalize_label(label)

                if label_l in DARK_SPOT_LABELS:
                    has_dark_spot = True
                    dark_spot_count += 1
                    self.draw_dark_spot(display_frame, box, confidence)

                elif label_l in SHORT_SHOT_LABELS:
                    has_short_shot = True
                    short_shot_count += 1
                    polygon = polygons[index] if index < len(polygons) else None
                    self.draw_short_shot_mask(display_frame, polygon, box, confidence)

                elif label_l in FLASH_LABELS:
                    has_flash = True
                    flash_count += 1
                    polygon = polygons[index] if index < len(polygons) else None
                    self.draw_flash(display_frame, polygon, box, confidence)

        judge = "NG" if (has_dark_spot or has_short_shot or has_flash) else "OK"

        payload = {
            "camera": "baseplate_side",
            "has_dark_spot": bool(has_dark_spot),
            "dark_spot_count": int(dark_spot_count),
            "has_short_shot": bool(has_short_shot),
            "short_shot_count": int(short_shot_count),
            "has_flash": bool(has_flash),
            "flash_count": int(flash_count),
            "model_key": self.active_model_key,
            "judge": judge,
        }

        self.update_view_inspection(payload)

        if self.inspection_started_at is None:
            inspection_elapsed_sec = 0.0
        else:
            inspection_elapsed_sec = time.monotonic() - self.inspection_started_at

        payload["inspection_active"] = bool(self.inspection_active)
        payload["inspection_view"] = self.inspection_view
        payload["inspection_cycle_id"] = self.inspection_cycle_id
        payload["inspection_elapsed_sec"] = round(inspection_elapsed_sec, 3)
        payload["defect_accum_sec"] = round(self.defect_accum_sec, 3)
        payload["required_defect_sec"] = self.required_defect_sec
        payload["last_view_judge"] = (
            None
            if self.last_view_result is None
            else self.last_view_result["judge"]
        )

        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self.inspection_pub.publish(msg)

        if has_flash:
            status_text = f"NG - FLASH ({flash_count})"
            status_color = FLASH_COLOR
        elif has_dark_spot and has_short_shot:
            status_text = f"NG - DARK SPOT:{dark_spot_count} / SHORT SHOT:{short_shot_count}"
            status_color = (0, 0, 255)
        elif has_dark_spot:
            status_text = f"NG - DARK SPOT ({dark_spot_count})"
            status_color = (0, 0, 255)
        elif has_short_shot:
            status_text = f"NG - SHORT SHOT ({short_shot_count})"
            status_color = (0, 165, 255)
        else:
            status_text = "OK"
            status_color = (0, 255, 0)

        cv2.putText(
            display_frame,
            status_text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            status_color,
            2,
            cv2.LINE_AA,
        )

        if self.inspection_active:
            trigger_text = (
                f"INSPECT {self.inspection_view}: "
                f"{inspection_elapsed_sec:.1f}/{self.inspection_duration_sec:.1f}s  "
                f"DEFECT {self.defect_accum_sec:.1f}/{self.required_defect_sec:.1f}s"
            )
            trigger_color = (255, 255, 0)
        elif self.last_view_result is not None:
            trigger_text = (
                f"RESULT {self.last_view_result['view']}: "
                f"{self.last_view_result['judge']}"
            )
            trigger_color = (
                (0, 0, 255)
                if self.last_view_result["judge"] == "NG"
                else (0, 255, 0)
            )
        else:
            trigger_text = "WAITING FOR INSPECTION START"
            trigger_color = (255, 255, 0)
        cv2.putText(
            display_frame,
            trigger_text,
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            trigger_color,
            2,
            cv2.LINE_AA,
        )

        cv2.imshow(self.window_name, display_frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            self.get_logger().info("사용자 종료 요청")
            if rclpy.ok():
                rclpy.shutdown()

    def destroy_node(self):
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = BaseplateSideInspector()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
