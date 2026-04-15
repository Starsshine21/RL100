import copy
import logging
import time
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp


def _as_6d_float_array(value, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 0:
        array = np.full((6,), float(array), dtype=np.float32)
    array = array.reshape(-1)
    if array.shape[0] != 6:
        raise ValueError(f"{name} must be a scalar or length-6 sequence, got shape={array.shape}.")
    return array.astype(np.float32, copy=False)


def _build_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


def _shape_is_chw(shape) -> bool:
    shape = tuple(shape)
    if len(shape) != 3:
        raise ValueError(f"Expected rank-3 image shape, got {shape}.")
    c_first = shape[0] <= 4 and shape[-1] > 4
    c_last = shape[-1] <= 4 and shape[0] > 4
    if c_first:
        return True
    if c_last:
        return False
    return shape[0] <= 4


def _shape_to_hwc(shape) -> Tuple[int, int, int]:
    shape = tuple(shape)
    if _shape_is_chw(shape):
        return int(shape[1]), int(shape[2]), int(shape[0])
    return int(shape[0]), int(shape[1]), int(shape[2])


class LowPassFilter:
    def __init__(self, alpha: float = 0.1):
        self.alpha = float(alpha)
        self.filtered = None

    def reset(self):
        self.filtered = None

    def filter(self, new_value: np.ndarray) -> np.ndarray:
        if self.filtered is None:
            self.filtered = np.asarray(new_value, dtype=np.float32)
        else:
            self.filtered = (
                self.alpha * np.asarray(new_value, dtype=np.float32)
                + (1.0 - self.alpha) * self.filtered
            )
        return self.filtered.copy()


class UR5eRTDE:
    def __init__(self, robot_ip: str, acceleration: float = 0.1, speed: float = 0.1):
        try:
            import rtde_control
            import rtde_receive
        except ImportError as exc:
            raise ImportError(
                "UR5eRTDE requires `rtde_control` and `rtde_receive`. "
                "Install the UR RTDE python packages on the real-robot machine."
            ) from exc

        self.robot_ip = str(robot_ip)
        self.acceleration = float(acceleration)
        self.speed = float(speed)
        self.rtde_r = rtde_receive.RTDEReceiveInterface(self.robot_ip)
        self.rtde_c = rtde_control.RTDEControlInterface(self.robot_ip)
        self.logger = _build_logger(self.__class__.__name__)
        self.logger.info("UR5e RTDE opened.")

    def get_pos_j(self) -> np.ndarray:
        return np.asarray(self.rtde_r.getActualQ(), dtype=np.float32)

    def get_pos_eef(self) -> np.ndarray:
        return np.asarray(self.rtde_r.getActualTCPPose(), dtype=np.float32)

    def set_pos_j(self, target_qpos, servo: bool = True):
        target_qpos = np.asarray(target_qpos, dtype=np.float64).tolist()
        if servo:
            self.rtde_c.servoJ(
                target_qpos,
                self.speed,
                self.acceleration,
                1 / 500,
                0.1,
                300,
            )
        else:
            self.rtde_c.moveJ(target_qpos, self.speed, self.acceleration)

    def set_pos_l(self, target_pos, servo: bool = True):
        target_pos = np.asarray(target_pos, dtype=np.float64).tolist()
        if servo:
            self.rtde_c.servoL(
                target_pos,
                self.speed,
                self.acceleration,
                1 / 125,
                0.1,
                300,
            )
        else:
            self.rtde_c.moveL(target_pos, self.speed, self.acceleration)

    def stop(self):
        try:
            self.rtde_c.servoStop()
        except Exception:
            pass
        try:
            self.rtde_c.stopScript()
        except Exception:
            pass
        self.logger.info("UR5e RTDE stopped.")


class InspireHandSerial:
    REGDICT = {
        "ID": 1000,
        "baudrate": 1001,
        "clearErr": 1004,
        "forceClb": 1009,
        "posSet": 1474,
        "angleSet": 1486,
        "forceSet": 1498,
        "speedSet": 1522,
        "angleAct": 1546,
        "forceAct": 1582,
        "posAct": 1534,
        "errCode": 1606,
        "statusCode": 1612,
        "temp": 1618,
    }

    def __init__(
        self,
        port: str,
        baudrate: int = 115200,
        hand_id: int = 1,
        init_force=None,
        init_speed=None,
        init_pos=None,
    ):
        try:
            import serial
        except ImportError as exc:
            raise ImportError(
                "InspireHandSerial requires `pyserial`. Install `serial`/`pyserial` on the robot machine."
            ) from exc

        self.serial_mod = serial
        self.port = str(port)
        self.baudrate = int(baudrate)
        self.hand_id = int(hand_id)
        self.init_force = list(init_force if init_force is not None else [500] * 6)
        self.init_speed = list(init_speed if init_speed is not None else [900] * 6)
        self.init_pos = list(init_pos if init_pos is not None else [0] * 6)
        self.ser = None
        self.logger = _build_logger(self.__class__.__name__)

    def open(self):
        self.ser = self.serial_mod.Serial()
        self.ser.port = self.port
        self.ser.baudrate = self.baudrate
        self.ser.open()
        self._write6("forceSet", self.init_force)
        self._write6("speedSet", self.init_speed)
        self.set_hand_pos(self.init_pos)
        self.logger.info("Inspire hand opened.")

    def close(self):
        if self.ser is not None and self.ser.is_open:
            self.ser.close()
        self.logger.info("Inspire hand closed.")

    def _write_register(self, add: int, num: int, values):
        if self.ser is None:
            raise RuntimeError("Inspire hand serial port is not open.")
        packet = [0xEB, 0x90, self.hand_id, num + 3, 0x12, add & 0xFF, (add >> 8) & 0xFF]
        packet.extend(int(v) & 0xFF for v in values)
        checksum = sum(packet[2:]) & 0xFF
        packet.append(checksum)
        self.ser.write(packet)
        time.sleep(0.01)
        self.ser.read_all()

    def _read_register(self, add: int, num: int):
        if self.ser is None:
            raise RuntimeError("Inspire hand serial port is not open.")
        packet = [0xEB, 0x90, self.hand_id, 0x04, 0x11, add & 0xFF, (add >> 8) & 0xFF, num]
        checksum = sum(packet[2:]) & 0xFF
        packet.append(checksum)
        self.ser.write(packet)
        time.sleep(0.01)
        recv = self.ser.read_all()
        if len(recv) == 0:
            return []
        data_len = (recv[3] & 0xFF) - 3
        return [recv[7 + i] for i in range(data_len)]

    def _write6(self, key: str, value):
        payload = []
        for item in value:
            item = int(item)
            payload.append(item & 0xFF)
            payload.append((item >> 8) & 0xFF)
        self._write_register(self.REGDICT[key], 12, payload)

    def _read6(self, key: str):
        raw = self._read_register(self.REGDICT[key], 12)
        if len(raw) < 12:
            raise RuntimeError(f"No valid response while reading {key} from inspire hand.")
        return [
            int((raw[2 * i] & 0xFF) + (raw[2 * i + 1] << 8))
            for i in range(6)
        ]

    def set_hand_pos(self, value):
        if len(value) != 6:
            raise ValueError(f"Inspire hand expects 6 values, got {len(value)}.")
        clipped = [int(np.clip(v, 0, 2000)) for v in value]
        self._write6("posSet", clipped)

    def get_hand_pos(self) -> np.ndarray:
        return np.asarray(self._read6("posAct"), dtype=np.float32)


class L515ColorCamera:
    def __init__(self):
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise ImportError(
                "L515ColorCamera requires `pyrealsense2`. Install Intel RealSense python bindings."
            ) from exc

        self.rs = rs
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline.start(self.config)
        self.logger = _build_logger(self.__class__.__name__)
        self.logger.info("L515 opened.")

    def get_data(self) -> np.ndarray:
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if color_frame is None:
            raise RuntimeError("Failed to read L515 color frame.")
        color_image = np.asanyarray(color_frame.get_data())
        return copy.copy(color_image)

    def close(self):
        try:
            self.pipeline.stop()
        except Exception:
            pass
        self.logger.info("L515 closed.")


class OrbbecFemtoBoltColorCamera:
    def __init__(self):
        try:
            import pyorbbecsdk as sdk
        except ImportError as exc:
            raise ImportError(
                "OrbbecFemtoBoltColorCamera requires `pyorbbecsdk`. Install the Orbbec python SDK."
            ) from exc

        self.sdk = sdk
        self.config = sdk.Config()
        self.pipeline = sdk.Pipeline()
        profile_list = self.pipeline.get_stream_profile_list(sdk.OBSensorType.COLOR_SENSOR)
        color_profile = profile_list.get_video_stream_profile(1280, 720, sdk.OBFormat.BGR, 30)
        self.config.enable_stream(color_profile)
        self.pipeline.start(self.config)
        self.logger = _build_logger(self.__class__.__name__)
        self.logger.info("Orbbec Femto Bolt opened.")

    def get_data(self) -> np.ndarray:
        while True:
            frames = self.pipeline.wait_for_frames(100)
            if frames is None:
                continue
            color_frame = frames.get_color_frame()
            if color_frame is None:
                continue
            width = color_frame.get_width()
            height = color_frame.get_height()
            data = np.asanyarray(color_frame.get_data())
            image_array = np.resize(data, (height, width, 3))
            return copy.deepcopy(image_array)

    def close(self):
        try:
            self.pipeline.stop()
        except Exception:
            pass
        self.logger.info("Orbbec Femto Bolt closed.")


class DummyCamera:
    def __init__(self, width: int = 640, height: int = 480, channels: int = 3, fill_value: int = 0):
        self.width = int(width)
        self.height = int(height)
        self.channels = int(channels)
        self.fill_value = int(fill_value)

    def get_data(self) -> np.ndarray:
        return np.full(
            (self.height, self.width, self.channels),
            fill_value=self.fill_value,
            dtype=np.uint8,
        )

    def close(self):
        return None


class UR5eInspireDualCamEnv:
    def __init__(
        self,
        shape_meta: Dict,
        robot_ip: str = "192.168.1.109",
        hand_port: str = "/dev/ttyUSB0",
        arm_action_mode: str = "eef",
        arm_state_mode: str = "eef",
        arm_encoding: str = "raw",
        arm_norm_center=0.0,
        arm_norm_scale=1.0,
        clip_normalized_arm_action: bool = True,
        hand_encoding: str = "normalized_minus1_1",
        hand_norm_center: float = 1000.0,
        hand_norm_scale: float = 1000.0,
        low_level_control_hz: float = 100.0,
        high_level_step_hz: float = 20.0,
        action_interp_steps: Optional[int] = None,
        arm_acceleration: float = 0.1,
        arm_speed: float = 0.1,
        arm_low_pass_alpha: Optional[float] = None,
        head_camera_type: str = "orbbec_femto_bolt",
        wrist_camera_type: str = "l515",
        convert_bgr_to_rgb: bool = False,
        head_mask_mode: str = "zeros",
        manual_reset: bool = True,
        reset_prompt: str = "[UR5eInspireDualCamEnv] Reset the scene, then press Enter to start rollout.",
        reset_settle_sec: float = 0.5,
        step_settle_sec: float = 0.0,
    ):
        self.shape_meta = dict(shape_meta)
        self.obs_meta = dict(self.shape_meta["obs"])
        self.action_dim = int(np.prod(self.shape_meta["action"]["shape"]))
        if self.action_dim != 12:
            raise ValueError(
                f"UR5eInspireDualCamEnv currently expects 12-D actions "
                f"(6-D UR5e arm + 6-D Inspire hand), got {self.action_dim}."
            )

        self.robot_ip = str(robot_ip)
        self.hand_port = str(hand_port)
        self.arm_action_mode = str(arm_action_mode).lower()
        self.arm_state_mode = str(arm_state_mode).lower()
        self.arm_encoding = str(arm_encoding).lower()
        self.arm_norm_center = _as_6d_float_array(arm_norm_center, "arm_norm_center")
        self.arm_norm_scale = _as_6d_float_array(arm_norm_scale, "arm_norm_scale")
        if np.any(np.abs(self.arm_norm_scale) < 1e-8):
            raise ValueError("arm_norm_scale must be non-zero for all 6 joints.")
        self.clip_normalized_arm_action = bool(clip_normalized_arm_action)
        self.hand_encoding = str(hand_encoding).lower()
        self.hand_norm_center = float(hand_norm_center)
        self.hand_norm_scale = float(hand_norm_scale)
        self.low_level_control_hz = float(low_level_control_hz)
        self.high_level_step_hz = float(high_level_step_hz)
        self.action_interp_steps = (
            int(round(self.low_level_control_hz / self.high_level_step_hz))
            if action_interp_steps is None
            else max(1, int(action_interp_steps))
        )
        self.arm_acceleration = float(arm_acceleration)
        self.arm_speed = float(arm_speed)
        self.convert_bgr_to_rgb = bool(convert_bgr_to_rgb)
        self.head_mask_mode = str(head_mask_mode).lower()
        self.manual_reset = bool(manual_reset)
        self.reset_prompt = str(reset_prompt)
        self.reset_settle_sec = float(reset_settle_sec)
        self.step_settle_sec = float(step_settle_sec)
        self.logger = _build_logger(self.__class__.__name__)

        if self.arm_action_mode not in ("eef", "joint"):
            raise ValueError(f"Unsupported arm_action_mode={arm_action_mode}. Expected 'eef' or 'joint'.")
        if self.arm_state_mode not in ("eef", "joint"):
            raise ValueError(f"Unsupported arm_state_mode={arm_state_mode}. Expected 'eef' or 'joint'.")
        if self.arm_encoding not in ("raw", "normalized_minus1_1"):
            raise ValueError(
                f"Unsupported arm_encoding={arm_encoding}. Expected 'raw' or 'normalized_minus1_1'."
            )
        if self.hand_encoding not in ("normalized_minus1_1", "raw_0_2000"):
            raise ValueError(
                f"Unsupported hand_encoding={hand_encoding}. Expected 'normalized_minus1_1' or 'raw_0_2000'."
            )
        if self.head_mask_mode not in ("zeros", "ones"):
            raise ValueError(f"Unsupported head_mask_mode={head_mask_mode}. Expected 'zeros' or 'ones'.")

        self.robot = None
        self.hand = None
        self.head_camera = None
        self.wrist_camera = None
        try:
            self.robot = UR5eRTDE(
                robot_ip=self.robot_ip,
                acceleration=self.arm_acceleration,
                speed=self.arm_speed,
            )
            self.hand = InspireHandSerial(port=self.hand_port)
            self.hand.open()
            self.head_camera = self._make_camera(head_camera_type)
            self.wrist_camera = self._make_camera(wrist_camera_type)
        except Exception:
            self.stop()
            raise
        self.arm_lpf = LowPassFilter(alpha=arm_low_pass_alpha) if arm_low_pass_alpha is not None else None

        self._first_arm_command = True
        self._prev_arm_target = None
        self._prev_hand_target = None

        self.logger.info(
            "Real-robot env ready: arm_action_mode=%s, arm_state_mode=%s, arm_encoding=%s, hand_encoding=%s, "
            "head_camera=%s, wrist_camera=%s, interp_steps=%d",
            self.arm_action_mode,
            self.arm_state_mode,
            self.arm_encoding,
            self.hand_encoding,
            head_camera_type,
            wrist_camera_type,
            self.action_interp_steps,
        )

    def _make_camera(self, camera_type: str):
        camera_type = str(camera_type).lower()
        if camera_type in ("l515", "realsense_l515"):
            return L515ColorCamera()
        if camera_type in ("orbbec", "orbbec_femto_bolt", "femto_bolt"):
            return OrbbecFemtoBoltColorCamera()
        if camera_type in ("dummy", "none"):
            return DummyCamera()
        raise ValueError(
            f"Unsupported camera_type={camera_type}. "
            "Expected one of ['l515', 'orbbec_femto_bolt', 'dummy']."
        )

    def _current_arm_raw(self, mode: str) -> np.ndarray:
        if mode == "joint":
            return self.robot.get_pos_j().astype(np.float32)
        return self.robot.get_pos_eef().astype(np.float32)

    def _encode_arm_state(self, arm_raw: np.ndarray) -> np.ndarray:
        arm_raw = np.asarray(arm_raw, dtype=np.float32)
        if self.arm_state_mode == "joint" and self.arm_encoding == "normalized_minus1_1":
            return (arm_raw - self.arm_norm_center) / self.arm_norm_scale
        return arm_raw

    def _decode_arm_action(self, arm_action: np.ndarray) -> np.ndarray:
        arm_action = np.asarray(arm_action, dtype=np.float32)
        if self.arm_action_mode == "joint" and self.arm_encoding == "normalized_minus1_1":
            if self.clip_normalized_arm_action:
                arm_action = np.clip(arm_action, -1.0, 1.0)
            return arm_action * self.arm_norm_scale + self.arm_norm_center
        return arm_action

    def _current_arm_state(self) -> np.ndarray:
        return self._encode_arm_state(self._current_arm_raw(self.arm_state_mode))

    def _current_arm_command_state(self) -> np.ndarray:
        return self._current_arm_raw(self.arm_action_mode)

    def _decode_hand_action(self, hand_action: np.ndarray) -> np.ndarray:
        hand_action = np.asarray(hand_action, dtype=np.float32)
        if self.hand_encoding == "raw_0_2000":
            raw = hand_action
        else:
            raw = hand_action * self.hand_norm_scale + self.hand_norm_center
        return np.clip(np.rint(raw), 0, 2000).astype(np.float32)

    def _encode_hand_state(self, hand_raw: np.ndarray) -> np.ndarray:
        hand_raw = np.asarray(hand_raw, dtype=np.float32)
        if self.hand_encoding == "raw_0_2000":
            return hand_raw
        return (hand_raw - self.hand_norm_center) / self.hand_norm_scale

    def _interpolate_arm(self, start: np.ndarray, end: np.ndarray, num_steps: int):
        start = np.asarray(start, dtype=np.float32)
        end = np.asarray(end, dtype=np.float32)
        if num_steps <= 1:
            return [end]

        if self.arm_action_mode == "joint":
            return [
                ((1.0 - alpha) * start + alpha * end).astype(np.float32)
                for alpha in np.linspace(1.0 / num_steps, 1.0, num_steps)
            ]

        rotation_path = R.from_rotvec(np.stack([start[3:6], end[3:6]], axis=0))
        slerp = Slerp([0.0, 1.0], rotation_path)
        trajectory = []
        for alpha in np.linspace(1.0 / num_steps, 1.0, num_steps):
            xyz = (1.0 - alpha) * start[:3] + alpha * end[:3]
            rotvec = slerp(alpha).as_rotvec().astype(np.float32)
            trajectory.append(np.concatenate([xyz.astype(np.float32), rotvec], axis=0))
        return trajectory

    def _interpolate_hand(self, start: np.ndarray, end: np.ndarray, num_steps: int):
        start = np.asarray(start, dtype=np.float32)
        end = np.asarray(end, dtype=np.float32)
        if num_steps <= 1:
            return [end]
        return [
            ((1.0 - alpha) * start + alpha * end).astype(np.float32)
            for alpha in np.linspace(1.0 / num_steps, 1.0, num_steps)
        ]

    def _read_camera_image(self, camera, spec_shape, append_mask: bool) -> np.ndarray:
        image = np.asarray(camera.get_data())
        if image.ndim != 3:
            raise ValueError(f"Expected camera image rank=3, got shape={image.shape}.")
        if image.shape[0] <= 4 and image.shape[-1] > 4:
            image = image.transpose(1, 2, 0)
        if self.convert_bgr_to_rgb and image.shape[-1] >= 3:
            image = image[..., ::-1].copy()

        target_h, target_w, target_c = _shape_to_hwc(spec_shape)
        if image.shape[0] != target_h or image.shape[1] != target_w:
            image = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        if append_mask:
            if target_c != 4:
                raise ValueError(f"rgbm spec must have 4 channels, got shape={spec_shape}.")
            mask_value = 255 if self.head_mask_mode == "ones" and np.issubdtype(image.dtype, np.integer) else 1.0
            if self.head_mask_mode == "zeros":
                mask_value = 0
            mask = np.full((target_h, target_w, 1), fill_value=mask_value, dtype=image.dtype)
            image = np.concatenate([image[..., :3], mask], axis=-1)
        else:
            if target_c < image.shape[-1]:
                image = image[..., :target_c]

        if _shape_is_chw(spec_shape):
            image = image.transpose(2, 0, 1)
        return image

    def _get_obs(self) -> Dict[str, np.ndarray]:
        arm_state = self._current_arm_state()
        hand_state_raw = self.hand.get_hand_pos()
        hand_state = self._encode_hand_state(hand_state_raw)

        rgbm_shape = tuple(self.obs_meta["rgbm"]["shape"])
        wrist_shape = tuple(self.obs_meta["right_cam_img"]["shape"])
        rgbm = self._read_camera_image(self.head_camera, rgbm_shape, append_mask=True)
        right_cam_img = self._read_camera_image(self.wrist_camera, wrist_shape, append_mask=False)

        return {
            "right_state": np.concatenate([arm_state[:6], hand_state[:6]], axis=0).astype(np.float32),
            "rgbm": rgbm,
            "right_cam_img": right_cam_img,
        }

    def _apply_arm_command(self, arm_target: np.ndarray, servo: bool):
        if self.arm_action_mode == "joint":
            self.robot.set_pos_j(arm_target, servo=servo)
        else:
            self.robot.set_pos_l(arm_target, servo=servo)

    def reset(self):
        if self.manual_reset:
            input(self.reset_prompt)
        if self.arm_lpf is not None:
            self.arm_lpf.reset()
        time.sleep(max(self.reset_settle_sec, 0.0))
        arm_state = self._current_arm_command_state()
        hand_state_raw = self.hand.get_hand_pos()
        self._prev_arm_target = arm_state[:6].astype(np.float32)
        self._prev_hand_target = hand_state_raw[:6].astype(np.float32)
        self._first_arm_command = True
        return self._get_obs()

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] != self.action_dim:
            raise ValueError(f"Expected action dim {self.action_dim}, got {action.shape[0]}.")

        arm_target = self._decode_arm_action(action[:6]).astype(np.float32)
        if self.arm_lpf is not None:
            arm_target = self.arm_lpf.filter(arm_target)
        hand_target = self._decode_hand_action(action[6:12])

        if self._prev_arm_target is None or self._prev_hand_target is None:
            self._prev_arm_target = self._current_arm_command_state()[:6].astype(np.float32)
            self._prev_hand_target = self.hand.get_hand_pos()[:6].astype(np.float32)

        arm_traj = self._interpolate_arm(self._prev_arm_target, arm_target, self.action_interp_steps)
        hand_traj = self._interpolate_hand(self._prev_hand_target, hand_target, self.action_interp_steps)
        low_level_dt = 1.0 / self.low_level_control_hz if self.low_level_control_hz > 0 else 0.0

        for arm_cmd, hand_cmd in zip(arm_traj, hand_traj):
            self._apply_arm_command(arm_cmd, servo=not self._first_arm_command)
            self._first_arm_command = False
            self.hand.set_hand_pos(np.rint(hand_cmd).astype(np.int32).tolist())
            if low_level_dt > 0:
                time.sleep(low_level_dt)

        if self.step_settle_sec > 0:
            time.sleep(self.step_settle_sec)

        self._prev_arm_target = arm_target.astype(np.float32)
        self._prev_hand_target = hand_target.astype(np.float32)
        return self._get_obs(), 0.0, False, {}

    def stop(self):
        try:
            self.robot.stop()
        except Exception as exc:
            self.logger.warning("robot.stop() failed: %s: %s", type(exc).__name__, exc)
        for device_name, device in (
            ("hand", self.hand),
            ("head_camera", self.head_camera),
            ("wrist_camera", self.wrist_camera),
        ):
            try:
                if device is not None and hasattr(device, "close"):
                    device.close()
            except Exception as exc:
                self.logger.warning("%s.close() failed: %s: %s", device_name, type(exc).__name__, exc)

    def emergency_stop(self):
        self.stop()

    def close(self):
        self.stop()
