#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""First centric-gripper process test for the five-axis A-arm.

Sequence
--------
1. Move the rail to the job's absolute position (shaft 625 mm / gear 825 mm).
2. Move A-arm to the job's grasp pose
   (shaft [0,-35,-80,-60,0] / gear [0,-15,-130,-30,0] deg).
3. Close the centric gripper with load-based auto grasp.
4. Return A-arm to the captured [0, 0, 0, 0, 0] deg HOME.
5. Return the rail to the absolute 0 mm position.

The gripper remains closed after A-arm and rail HOME are reached.  The process
starts from the PyBullet START PROCESS button (or the ``start`` command), not
immediately at node startup.  EMERGENCY STOP holds the current A-arm/gripper
position without terminating the ROS 2 node, so another command can follow.
"""

import math
import os
import re
import sys
import tempfile
import threading
import time

try:
    import pybullet as p
    import pybullet_data
except ImportError:
    p = None
    pybullet_data = None

import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from std_msgs.msg import String

try:
    from .centric_gripper_5axis import CentricGripper, GraspResult
    from .dxl_bridge_suction_5axis import DxlBridge
    from .rail_serial_centric_5axis import RAIL_MAX_M, RAIL_MIN_M, RailSerial
except ImportError:
    from centric_gripper_5axis import CentricGripper, GraspResult
    from dxl_bridge_suction_5axis import DxlBridge
    from rail_serial_centric_5axis import RAIL_MAX_M, RAIL_MIN_M, RailSerial


def _deg_to_rad(values):
    return [math.radians(float(value)) for value in values]


def _wrapped_error(a, b):
    return (float(a) - float(b) + math.pi) % (2.0 * math.pi) - math.pi


try:
    PACKAGE_SHARE = get_package_share_directory("vision_pkg")
except Exception:
    PACKAGE_SHARE = os.path.dirname(os.path.abspath(__file__))

DEFAULT_MESH_DIR = next(
    (
        path
        for path in (
            os.path.join(PACKAGE_SHARE, "meshes"),
            os.path.join(PACKAGE_SHARE, "mashes"),
        )
        if os.path.isdir(path)
    ),
    os.path.join(PACKAGE_SHARE, "meshes"),
)

URDF_NAME = "robot_arm_5axis_rail_suction.urdf"
RAIL_JOINT_NAME = "joint_rail"


class SequenceStopped(RuntimeError):
    pass


class CentricSequenceNode(Node):
    def __init__(self):
        super().__init__("centric_sequence_5axis")

        d = self.declare_parameter
        d("use_robot", False)
        d("use_rail", False)
        d("use_centric_gripper", False)
        d("use_simulation", True)

        d("pkg_dir", PACKAGE_SHARE)
        d("mesh_dir", DEFAULT_MESH_DIR)
        d("urdf_name", URDF_NAME)
        d("sim_hz", 60.0)
        d("sim_joint_speed_deg_s", 25.0)
        d("sim_rail_speed_mm_s", 100.0)

        d("dxl_port", "/dev/ttyUSB0")
        d("dxl_baud", 2_000_000)
        d("centric_id", 8)
        d("centric_capture_open_on_start", True)
        d("centric_configured_open_position", 506)

        d("rail_port", "/dev/ttyACM0")
        d("rail_baud", 115200)
        d("rail_auto_zero", True)
        # 공정 2종. 구조는 같고 레일 거리와 파지 자세만 다르다.
        d("shaft_rail_mm", 625.0)          # 샤프트/베어링
        d("shaft_pose_deg", [0.0, -35.0, -80.0, -60.0, 0.0])
        d("gear_rail_mm", 825.0)           # 기어/뚜껑
        d("gear_pose_deg", [0.0, -15.0, -130.0, -30.0, 0.0])
        d("default_job", "shaft")          # start 명령이 실행할 공정
        d("rail_target_mm", 625.0)         # 하위호환. shaft_rail_mm 이 우선
        d("rail_arrival_tol_mm", 2.0)
        d("rail_arrival_timeout_sec", 60.0)

        d("grasp_pose_deg", [0.0, -35.0, -80.0, -60.0, 0.0])
        d("home_pose_deg", [0.0, 0.0, 0.0, 0.0, 0.0])
        d("joint_arrival_tol_deg", 1.5)
        d("joint_arrival_timeout_sec", 25.0)
        d("auto_start", False)

        g = self.get_parameter
        self.use_robot = bool(g("use_robot").value)
        self.use_rail = bool(g("use_rail").value)
        self.use_gripper = bool(g("use_centric_gripper").value)
        self.use_simulation = bool(g("use_simulation").value)
        self.pkg_dir = str(g("pkg_dir").value)
        self.mesh_dir = str(g("mesh_dir").value)
        self.urdf_name = str(g("urdf_name").value)
        self.sim_hz = max(10.0, float(g("sim_hz").value))
        self.sim_joint_speed = max(
            1.0, float(g("sim_joint_speed_deg_s").value)
        )
        self.sim_rail_speed = max(
            1.0, float(g("sim_rail_speed_mm_s").value)
        )
        self.jobs = {
            "shaft": dict(
                name="shaft/bearing",
                rail_mm=float(g("shaft_rail_mm").value),
                pose=[float(v) for v in g("shaft_pose_deg").value],
            ),
            "gear": dict(
                name="gear/cover",
                rail_mm=float(g("gear_rail_mm").value),
                pose=[float(v) for v in g("gear_pose_deg").value],
            ),
        }
        self.default_job = str(g("default_job").value).strip().lower()
        if self.default_job not in self.jobs:
            self.default_job = "shaft"
        self.rail_target_mm = self.jobs[self.default_job]["rail_mm"]
        self.rail_tol_mm = max(0.1, float(g("rail_arrival_tol_mm").value))
        self.rail_timeout = max(1.0, float(g("rail_arrival_timeout_sec").value))
        self.joint_tol_rad = math.radians(
            max(0.1, float(g("joint_arrival_tol_deg").value))
        )
        self.joint_timeout = max(
            1.0, float(g("joint_arrival_timeout_sec").value)
        )
        self.grasp_pose_deg = self._pose_parameter("grasp_pose_deg")
        self.home_pose_deg = self._pose_parameter("home_pose_deg")

        min_mm = RAIL_MIN_M * 1000.0
        max_mm = RAIL_MAX_M * 1000.0
        if not min_mm <= self.rail_target_mm <= max_mm:
            raise ValueError(
                "rail_target_mm %.1f is outside [%.1f, %.1f]"
                % (self.rail_target_mm, min_mm, max_mm)
            )
        self.bridge = None
        self.rail = None
        self.gripper = None
        self.busy = False
        self.phase = "INITIALIZING"
        self._state_lock = threading.Lock()
        self._stop_event = threading.Event()
        self.sim_q_deg = list(self.home_pose_deg)
        self.sim_rail_mm = 0.0
        self.sim_gripper_state = "OPEN"
        self._sim_client = None
        self.robot = None
        self.rev = []
        self.rail_jid = None
        self._button = {}
        self._button_value = {}
        self._debug_text = {}
        self._last_render_text = 0.0

        self._setup_simulation()
        self._setup_hardware(g)
        self.state_pub = self.create_publisher(
            String, "/centric_sequence/state", 10
        )
        self.create_subscription(
            String, "/centric_sequence/command", self._on_command, 10
        )
        self.create_timer(1.0 / self.sim_hz, self._tick)
        self._start_terminal_thread()
        self._set_phase("READY")
        self._print_help()

        if bool(g("auto_start").value):
            self.create_timer(1.0, self._auto_start_once)
        self._auto_started = False

    def _pose_parameter(self, name):
        values = [float(v) for v in self.get_parameter(name).value]
        if len(values) != 5 or not all(math.isfinite(v) for v in values):
            raise ValueError("%s must contain five finite joint angles" % name)
        return values

    # ------------------------------------------------------------- simulation
    def _setup_simulation(self):
        if not self.use_simulation:
            return
        if p is None or pybullet_data is None:
            raise RuntimeError("use_simulation=true but PyBullet is unavailable")

        self._sim_client = p.connect(p.GUI)
        if self._sim_client < 0:
            raise RuntimeError("PyBullet GUI connection failed")
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, 0)
        p.loadURDF("plane.urdf")
        p.resetDebugVisualizerCamera(1.6, 50, -25, [0.0, 0.4, 0.4])
        self.robot = p.loadURDF(
            self._prepare_urdf(self.urdf_name),
            [0, 0, 0],
            useFixedBase=True,
        )

        for joint_id in range(p.getNumJoints(self.robot)):
            info = p.getJointInfo(self.robot, joint_id)
            name = info[1].decode() if isinstance(info[1], bytes) else str(info[1])
            if info[2] == p.JOINT_REVOLUTE:
                self.rev.append(joint_id)
            elif info[2] == p.JOINT_PRISMATIC and name == RAIL_JOINT_NAME:
                self.rail_jid = joint_id

        labels = {
            "start": "START PROCESS",
            "stop": "EMERGENCY STOP",
            "home": "GO A HOME",
            "rail_home": "RAIL HOME",
            "open": "OPEN GRIPPER",
            "grasp": "CENTRIC GRASP ONLY",
        }
        for key, label in labels.items():
            self._button[key] = p.addUserDebugParameter(label, 1, 0, 0)
            self._button_value[key] = 0.0
        self._render_sim(force_text=True)

    def _find_urdf(self, name):
        candidates = [
            os.path.join(self.pkg_dir, "urdf", name),
            os.path.join(self.pkg_dir, name),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), name),
        ]
        for path in candidates:
            if os.path.isfile(path):
                return path
        raise FileNotFoundError("URDF not found: " + " | ".join(candidates))

    def _rewrite_mesh_paths(self, text):
        mesh_dir = os.path.abspath(self.mesh_dir)

        def replace(match):
            quote, raw = match.group(1), match.group(2)
            if not raw.lower().endswith((".stl", ".dae", ".obj")):
                return match.group(0)
            base = os.path.basename(raw.replace("\\", "/"))
            candidate = os.path.join(mesh_dir, base)
            if not os.path.isfile(candidate):
                for root, _dirs, files in os.walk(mesh_dir):
                    if base in files:
                        candidate = os.path.join(root, base)
                        break
            if os.path.isfile(candidate):
                return "filename=%s%s%s" % (quote, candidate, quote)
            return match.group(0)

        return re.sub(r"filename=([\"'])(.*?)(?:\1)", replace, text)

    def _prepare_urdf(self, name):
        source = self._find_urdf(name)
        with open(source, encoding="utf-8") as stream:
            text = self._rewrite_mesh_paths(stream.read())
        text = text.replace(
            "package://robot_description/", self.pkg_dir.rstrip("/") + "/"
        )
        text = text.replace(
            "package://vision_pkg/", self.pkg_dir.rstrip("/") + "/"
        )
        output = os.path.join(
            tempfile.gettempdir(), "centric_sequence_" + os.path.basename(name)
        )
        with open(output, "w", encoding="utf-8") as stream:
            stream.write(text)
        return output

    def _button_clicked(self, key):
        if not self.use_simulation or key not in self._button:
            return False
        try:
            value = p.readUserDebugParameter(self._button[key])
        except Exception as exc:
            if not getattr(self, "_button_read_warned", False):
                self.get_logger().warning(
                    "PyBullet button read failed; ROS node remains active: %s" % exc
                )
                self._button_read_warned = True
            return False
        if value > self._button_value[key]:
            self._button_value[key] = value
            return True
        return False

    def _set_sim_state(self, q_deg=None, rail_mm=None, gripper=None):
        with self._state_lock:
            if q_deg is not None:
                self.sim_q_deg = [float(value) for value in q_deg]
            if rail_mm is not None:
                self.sim_rail_mm = float(rail_mm)
            if gripper is not None:
                self.sim_gripper_state = str(gripper)

    def _render_sim(self, force_text=False):
        if not self.use_simulation or self.robot is None:
            return
        with self._state_lock:
            q_deg = list(self.sim_q_deg)
            rail_mm = self.sim_rail_mm
            gripper_state = self.sim_gripper_state
            busy = self.busy
        if self.rail_jid is not None:
            p.resetJointState(self.robot, self.rail_jid, rail_mm / 1000.0)
        q_rad = _deg_to_rad(q_deg)
        q_rad += [0.0] * max(0, len(self.rev) - len(q_rad))
        for index, joint_id in enumerate(self.rev):
            p.resetJointState(self.robot, joint_id, q_rad[index])
        p.stepSimulation()

        now = time.monotonic()
        if not force_text and now - self._last_render_text < 0.10:
            return
        self._last_render_text = now
        lines = {
            "phase": (
                "PHASE: %s | busy=%s" % (self.phase, busy),
                [1.0, 1.0, 0.2],
                1.35,
            ),
            "pose": (
                "A(deg): %s" % [round(value, 1) for value in q_deg],
                [0.4, 0.9, 1.0],
                1.20,
            ),
            "rail": (
                "RAIL: %.1f mm | CENTRIC: %s" % (rail_mm, gripper_state),
                [0.4, 1.0, 0.4],
                1.05,
            ),
        }
        for key, (text, color, height) in lines.items():
            previous = self._debug_text.get(key)
            if previous is not None:
                p.removeUserDebugItem(previous)
            self._debug_text[key] = p.addUserDebugText(
                text, [0, 0, height], textColorRGB=color, textSize=1.1
            )

    def _tick(self):
        if self._button_clicked("stop"):
            self._emergency_stop("PyBullet EMERGENCY STOP")
            self._set_phase("STOPPED_READY")
        if self._button_clicked("start"):
            self._start_sequence()
        if self._button_clicked("home"):
            self._start_auxiliary("GOING_HOME", self._home_only)
        if self._button_clicked("rail_home"):
            self._start_auxiliary("RAIL_RETURN_HOME", self._rail_home_only)
        if self._button_clicked("open"):
            self._start_auxiliary("OPENING", self._open_gripper)
        if self._button_clicked("grasp"):
            self._start_auxiliary("GRASPING", self._grasp_only)
        self._render_sim()

    # --------------------------------------------------------------- hardware
    def _setup_hardware(self, g):
        try:
            if self.use_rail:
                self.rail = RailSerial(
                    port=str(g("rail_port").value),
                    baud=int(g("rail_baud").value),
                    logger=self.get_logger(),
                )
                self.rail.connect()
                if bool(g("rail_auto_zero").value):
                    if not self.rail.zero(wait=True, timeout=2.0):
                        raise RuntimeError("rail zero acknowledgement timeout")
                if not self.rail.zero_set:
                    raise RuntimeError("rail zero is not set")

            if self.use_robot:
                self.bridge = DxlBridge(
                    port=str(g("dxl_port").value),
                    baud=int(g("dxl_baud").value),
                )
                self.bridge.connect()
                self.bridge.capture_home()
                self.get_logger().warning(
                    "A-arm startup pose was captured as [0, 0, 0, 0, 0] deg HOME"
                )

            if self.use_gripper:
                self.gripper = CentricGripper(
                    port=str(g("dxl_port").value),
                    baud=int(g("dxl_baud").value),
                    dxl_id=int(g("centric_id").value),
                    logger=self.get_logger(),
                )
                connect_kwargs = dict(
                    capture_open_on_start=bool(
                        g("centric_capture_open_on_start").value
                    ),
                    configured_open_position=int(
                        g("centric_configured_open_position").value
                    ),
                )
                if self.bridge is not None:
                    self.gripper.connect_shared(
                        self.bridge.ph,
                        self.bridge.pk,
                        self.bridge.bus_lock,
                        **connect_kwargs,
                    )
                    self.get_logger().info(
                        "A-arm ID 1~7 and centric ID 8 share one U2D2: %s"
                        % str(g("dxl_port").value)
                    )
                else:
                    # Gripper-only real test: this node owns the same U2D2 port.
                    self.gripper.connect(**connect_kwargs)
                if bool(g("centric_capture_open_on_start").value):
                    self.get_logger().warning(
                        "Centric startup position was captured as 0% OPEN; verify it was fully open"
                    )
        except Exception:
            self._close_hardware()
            if self._sim_client is not None and p is not None:
                try:
                    p.disconnect(self._sim_client)
                except Exception:
                    pass
                self._sim_client = None
            raise

    def _close_hardware(self):
        if self.rail is not None:
            try:
                self.rail.estop()
                self.rail.close()
            except Exception:
                pass
            self.rail = None
        if self.gripper is not None:
            try:
                # Detach ID 8 before the A-arm owner closes the shared port.
                self.gripper.close(torque_off=False)
            except Exception:
                pass
            self.gripper = None
        if self.bridge is not None:
            try:
                # Keep A-arm torque enabled so it does not suddenly go limp.
                self.bridge.close(torque_off=False)
            except Exception:
                pass
            self.bridge = None

    # --------------------------------------------------------------- commands
    def _auto_start_once(self):
        if not self._auto_started:
            self._auto_started = True
            self._start_sequence()

    def _on_command(self, msg):
        self._handle_command(msg.data)

    def _handle_command(self, raw):
        parts = str(raw).strip().lower().split()
        if not parts:
            return
        command = parts[0]
        if command in ("start", "run"):
            self._start_sequence(parts[1] if len(parts) > 1 else None)
        elif command in ("shaft", "bearing"):
            self._start_sequence("shaft")
        elif command in ("gear", "cover"):
            self._start_sequence("gear")
        elif command in ("stop", "estop", "0", "s"):
            self._emergency_stop("user command")
            self._set_phase("STOPPED_READY")
        elif command == "status":
            self._log_status()
        elif command == "open":
            self._start_auxiliary("OPENING", self._open_gripper)
        elif command == "grasp":
            self._start_auxiliary("GRASPING", self._grasp_only)
        elif command == "home":
            self._start_auxiliary("GOING_HOME", self._home_only)
        elif command in ("railhome", "rhome"):
            self._start_auxiliary("RAIL_RETURN_HOME", self._rail_home_only)
        else:
            self.get_logger().warning(
                "commands: start [shaft|gear] | shaft | gear | stop | status "
                "| open | grasp | home | railhome"
            )

    def _start_sequence(self, job_key=None):
        if job_key is not None and job_key not in self.jobs:
            self.get_logger().error("unknown job: %s" % job_key)
            return
        with self._state_lock:
            if self.busy:
                self.get_logger().warning("sequence is already busy")
                return
            self.busy = True
            self._stop_event.clear()
        threading.Thread(target=self._run_sequence, args=(job_key,),
                         daemon=True).start()

    def _start_auxiliary(self, phase, callback):
        with self._state_lock:
            if self.busy:
                self.get_logger().warning("sequence is busy")
                return
            self.busy = True
            self._stop_event.clear()

        def worker():
            try:
                self._set_phase(phase)
                callback()
                self._set_phase("READY")
            except SequenceStopped:
                self.get_logger().warning("motion stopped; node is still active")
                self._set_phase("STOPPED_READY")
            except Exception as exc:
                self._fail(str(exc))
            finally:
                with self._state_lock:
                    self.busy = False

        threading.Thread(target=worker, daemon=True).start()

    # --------------------------------------------------------------- sequence
    def _run_sequence(self, job_key=None):
        job = self.jobs[job_key or self.default_job]
        try:
            self.get_logger().info(
                "=== job: %s (rail %.0f mm, pose %s) ==="
                % (job["name"], job["rail_mm"], job["pose"][:4])
            )
            self._set_phase("RAIL_TO_TARGET")
            self._move_rail(job["rail_mm"])

            self._set_phase("A_TO_GRASP_POSE")
            self._move_arm(job["pose"])

            self._set_phase("CENTRIC_GRASP")
            result = self._auto_grasp()
            if not result.success:
                raise RuntimeError("centric grasp failed: " + result.reason)

            self._set_phase("A_RETURN_HOME")
            self._move_arm(self.home_pose_deg)

            self._set_phase("RAIL_RETURN_HOME")
            self._move_rail(0.0)

            self._set_phase("DONE_HOLDING_OBJECT_AT_HOME")
            self.get_logger().info(
                "sequence complete: A-arm HOME, rail 0 mm, gripper holding"
            )
        except SequenceStopped:
            self.get_logger().warning(
                "process stopped at current position; node remains ready for commands"
            )
            self._set_phase("STOPPED_READY")
        except Exception as exc:
            self._fail(str(exc))
        finally:
            with self._state_lock:
                self.busy = False

    def _move_rail(self, target_mm):
        self._check_stop()
        if self.rail is None:
            if self.use_rail:
                raise RuntimeError("rail connection is unavailable")
            self._animate_sim_rail(target_mm)
            self.get_logger().info("[SIM] rail arrived -> %.1f mm" % target_mm)
            return
        if not self.rail.move_abs_mm(float(target_mm)):
            raise RuntimeError("rail move command rejected")

        deadline = time.monotonic() + self.rail_timeout
        next_query = 0.0
        while time.monotonic() < deadline:
            self._check_stop()
            now = time.monotonic()
            if now >= next_query:
                self.rail.query()
                next_query = now + 0.5
            present_mm = float(self.rail.position_mm)
            self._set_sim_state(rail_mm=present_mm)
            error = abs(present_mm - float(target_mm))
            if not self.rail.moving and error <= self.rail_tol_mm:
                self._set_sim_state(rail_mm=present_mm)
                self.get_logger().info(
                    "rail arrived: %.1f mm (error %.1f mm)"
                    % (present_mm, error)
                )
                return
            time.sleep(0.05)
        raise RuntimeError(
            "rail arrival timeout: present %.1f mm, target %.1f mm"
            % (self.rail.position_mm, target_mm)
        )

    def _move_arm(self, goal_deg):
        self._check_stop()
        if self.bridge is None:
            if self.use_robot:
                raise RuntimeError("A-arm connection is unavailable")
            self._animate_sim_arm(goal_deg)
            self.get_logger().info("[SIM] A-arm arrived -> %s deg" % list(goal_deg))
            return

        goal_rad = _deg_to_rad(goal_deg)
        self.bridge.command_joints(goal_rad)
        deadline = time.monotonic() + self.joint_timeout
        while time.monotonic() < deadline:
            self._check_stop()
            actual = self.bridge.read_joints()
            if len(actual) < 5:
                raise RuntimeError("A-arm returned fewer than five joint states")
            actual_deg = [math.degrees(value) for value in actual[:5]]
            self._set_sim_state(q_deg=actual_deg)
            max_error = max(
                abs(_wrapped_error(present, goal))
                for present, goal in zip(actual[:5], goal_rad)
            )
            if max_error <= self.joint_tol_rad:
                self._set_sim_state(q_deg=actual_deg)
                self.get_logger().info(
                    "A-arm arrived: %s deg (max error %.2f deg)"
                    % (list(goal_deg), math.degrees(max_error))
                )
                return
            time.sleep(0.10)
        raise RuntimeError("A-arm arrival timeout")

    def _auto_grasp(self):
        self._check_stop()
        self._set_sim_state(gripper="CLOSING")
        if self.gripper is None:
            if self.use_gripper:
                raise RuntimeError("centric gripper connection is unavailable")
            self._interruptible_wait(0.8)
            self._set_sim_state(gripper="HOLDING")
            self.get_logger().info("[SIM] centric auto grasp -> success")
            return GraspResult(True, "simulation")
        result = self.gripper.auto_grasp(self._stop_event.is_set)
        self._set_sim_state(
            gripper="HOLDING" if result.success else "GRASP FAILED"
        )
        return result

    def _open_gripper(self):
        self._check_stop()
        self._set_sim_state(gripper="OPENING")
        if self.gripper is None:
            if self.use_gripper:
                raise RuntimeError("centric gripper connection is unavailable")
            self._interruptible_wait(0.5)
            self._set_sim_state(gripper="OPEN")
            self.get_logger().info("[SIM] centric open")
            return
        if not self.gripper.open(self._stop_event.is_set):
            raise RuntimeError("centric open failed")
        self._set_sim_state(gripper="OPEN")

    def _grasp_only(self):
        result = self._auto_grasp()
        if not result.success:
            raise RuntimeError("centric grasp failed: " + result.reason)

    def _home_only(self):
        self._move_arm(self.home_pose_deg)

    def _rail_home_only(self):
        self._move_rail(0.0)

    def _interruptible_wait(self, seconds):
        deadline = time.monotonic() + max(0.0, float(seconds))
        while time.monotonic() < deadline:
            self._check_stop()
            time.sleep(0.02)

    def _animate_sim_rail(self, target_mm):
        with self._state_lock:
            start_mm = float(self.sim_rail_mm)
        if not self.use_simulation:
            self._set_sim_state(rail_mm=target_mm)
            return
        distance = abs(float(target_mm) - start_mm)
        duration = distance / self.sim_rail_speed
        self._animate_scalar(
            start_mm,
            float(target_mm),
            duration,
            lambda value: self._set_sim_state(rail_mm=value),
        )

    def _animate_sim_arm(self, goal_deg):
        with self._state_lock:
            start_deg = list(self.sim_q_deg)
        goal = [float(value) for value in goal_deg]
        if not self.use_simulation:
            self._set_sim_state(q_deg=goal)
            return
        duration = max(
            (abs(end - start) for start, end in zip(start_deg, goal)),
            default=0.0,
        ) / self.sim_joint_speed
        started = time.monotonic()
        while True:
            self._check_stop()
            elapsed = time.monotonic() - started
            u = 1.0 if duration <= 0.0 else min(1.0, elapsed / duration)
            smooth = u * u * (3.0 - 2.0 * u)
            present = [
                start + (end - start) * smooth
                for start, end in zip(start_deg, goal)
            ]
            self._set_sim_state(q_deg=present)
            if u >= 1.0:
                return
            time.sleep(1.0 / self.sim_hz)

    def _animate_scalar(self, start, end, duration, setter):
        started = time.monotonic()
        while True:
            self._check_stop()
            elapsed = time.monotonic() - started
            u = 1.0 if duration <= 0.0 else min(1.0, elapsed / duration)
            smooth = u * u * (3.0 - 2.0 * u)
            setter(float(start) + (float(end) - float(start)) * smooth)
            if u >= 1.0:
                return
            time.sleep(1.0 / self.sim_hz)

    def _check_stop(self):
        if self._stop_event.is_set():
            raise SequenceStopped("sequence stopped")

    def _fail(self, reason):
        self.get_logger().error("sequence failed: %s" % reason)
        self._emergency_stop(reason)
        self._set_phase("ERROR_HOLDING")

    def _emergency_stop(self, reason):
        self._stop_event.set()
        if self.rail is not None:
            try:
                self.rail.estop()
            except Exception as exc:
                self.get_logger().error("rail stop failed: %s" % exc)
        if self.bridge is not None:
            try:
                self.bridge.emergency_stop()
            except Exception as exc:
                self.get_logger().error("A-arm stop failed: %s" % exc)
        if self.gripper is not None:
            try:
                self.gripper.emergency_stop(reason)
            except Exception as exc:
                self.get_logger().error("centric stop failed: %s" % exc)

    # ---------------------------------------------------------- status / shell
    def _set_phase(self, phase):
        self.phase = str(phase)
        self.get_logger().info("--- PHASE: %s ---" % self.phase)
        self._publish_state()

    def _publish_state(self):
        msg = String()
        msg.data = "%s|busy=%s" % (self.phase, self.busy)
        self.state_pub.publish(msg)

    def _log_status(self):
        if self.rail is None:
            with self._state_lock:
                rail_text = "%.1f mm (SIM)" % self.sim_rail_mm
        else:
            rail_text = "%.1f mm, moving=%s" % (
                self.rail.position_mm,
                self.rail.moving,
            )
        if self.bridge is None:
            with self._state_lock:
                arm_text = "%s deg (SIM)" % [
                    round(value, 1) for value in self.sim_q_deg
                ]
        else:
            arm_text = "[" + ", ".join(
                "%.1f" % math.degrees(v) for v in self.bridge.read_joints()
            ) + "] deg"
        if self.gripper is None:
            gripper_text = "SIM"
        else:
            status = self.gripper.read_status()
            gripper_text = (
                "pos=%d, close=%.1f%%, load=%.1f%%, temp=%dC"
                % (
                    status["position"],
                    status["close_percent"],
                    status["load_percent"],
                    status["temperature_c"],
                )
            )
        self.get_logger().info(
            "phase=%s, busy=%s | rail=%s | A=%s | centric=%s"
            % (self.phase, self.busy, rail_text, arm_text, gripper_text)
        )

    def _start_terminal_thread(self):
        if sys.stdin is not None and sys.stdin.isatty():
            threading.Thread(target=self._terminal_loop, daemon=True).start()

    def _terminal_loop(self):
        while rclpy.ok():
            try:
                self._handle_command(input("CENTRIC_SEQ> "))
            except (EOFError, KeyboardInterrupt):
                return
            except Exception as exc:
                self.get_logger().error("terminal command failed: %s" % exc)

    @staticmethod
    def _print_help():
        print("\nCommands: start | stop(s) | status | open | grasp | home | railhome")
        print(
            "START PROCESS: rail -> A grasp pose -> grasp "
            "-> A HOME -> rail 0 mm\n"
        )

    def destroy_node(self):
        self._stop_event.set()
        self._close_hardware()
        if self._sim_client is not None and p is not None:
            try:
                p.disconnect(self._sim_client)
            except Exception:
                pass
            self._sim_client = None
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = CentricSequenceNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

