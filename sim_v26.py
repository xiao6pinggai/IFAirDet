# ============================================================
# sim_v24.py — 合并 test/train 仿真，通过 MODE 配置切换
# 用法: 修改下方 MODE = 'test' 或 'train'，然后直接运行
# ============================================================
import re
import matplotlib.pyplot as plt
import random
import numpy as np
from write_xml import write_rotate_xml, bias_cxcywh, get_hbbox, merge_masks
from tqdm import tqdm
from motion_bulr_newV22 import *
from global_noise import PhysicalBasedSimulator, complete_simulation_pipeline
from BlurEdge import blur_full_inner_region

# ======================== 飞机运动参数配置区（集中维护） ========================
# 说明：
# 1. speed_kmh / speed_seq_kmh 表示飞机真实对地速度，单位 km/h
# 2. accel_mps2 表示飞机真实加速度，单位 m/s^2
# 3. turn_total_deg 表示整段轨迹的真实总转角，单位 deg
# 4. turn_rate_deg_per_frame 表示每帧真实航向变化率，单位 deg/frame
MOTION_STATE_PROBABILITIES = {
    "constant": 0.70,    # 整段轨迹保持匀速的概率
    "accelerate": 0.15,  # 整段轨迹保持匀加速的概率
    "decelerate": 0.15,  # 整段轨迹保持匀减速的概率
}

AIRCRAFT_GROUP_RULES = [
    ("战斗机", "战斗机"),
    ("攻击机", "攻击机"),
    ("轰炸机", "轰炸机"),
    ("预警机", "预警机"),
    ("侦察机", "侦察/电子侦察机"),
    ("运输机", "运输机"),
    ("反潜机", "海上巡逻机"),
    ("加油机", "加油机"),
    ("民航机", "客机"),
]

# AIRCRAFT_MOTION_CONFIG 字段说明：
# accel_range_mps2: 真实加速度范围，单位 m/s^2
# speed_scale_range: 相对基准巡航速度允许的上下限倍率
# turn_none_prob: 整段轨迹保持直飞的概率
# turn_angle_range_deg: 常规机动总转角范围，单位 deg
# excited_turn_prob: 明显机动概率（仅战斗机）
# excited_turn_angle_range_deg: 明显机动总转角范围，单位 deg（仅战斗机）
AIRCRAFT_MOTION_CONFIG = {
    "战斗机": {
        "accel_range_mps2": (1.5, 6.0),              # 真实加速度范围，单位 m/s^2
        "speed_scale_range": (0.8, 1.2),             # 相对基准巡航速度允许的倍率范围
        "turn_none_prob": 0.55,                      # 整段轨迹保持直飞的概率
        "turn_angle_range_deg": (5.0, 25.0),         # 常规机动总转角范围，单位 deg
        "excited_turn_prob": 0.05,                   # 明显机动概率
        "excited_turn_angle_range_deg": (35.0, 80.0) # 明显机动总转角范围，单位 deg
    },
    "攻击机": {
        "accel_range_mps2": (1.0, 4.0),
        "speed_scale_range": (0.8, 1.2),
        "turn_none_prob": 0.65,
        "turn_angle_range_deg": (5.0, 20.0),
    },
    "轰炸机": {
        "accel_range_mps2": (0.4, 2.0),
        "speed_scale_range": (0.85, 1.15),
        "turn_none_prob": 0.82,
        "turn_angle_range_deg": (3.0, 12.0),
    },
    "预警机": {
        "accel_range_mps2": (0.3, 1.2),
        "speed_scale_range": (0.85, 1.15),
        "turn_none_prob": 0.88,
        "turn_angle_range_deg": (2.0, 8.0),
    },
    "侦察/电子侦察机": {
        "accel_range_mps2": (0.3, 1.5),
        "speed_scale_range": (0.85, 1.15),
        "turn_none_prob": 0.78,
        "turn_angle_range_deg": (3.0, 15.0),
    },
    "运输机": {
        "accel_range_mps2": (0.4, 2.0),
        "speed_scale_range": (0.85, 1.15),
        "turn_none_prob": 0.82,
        "turn_angle_range_deg": (3.0, 12.0),
    },
    "海上巡逻机": {
        "accel_range_mps2": (0.3, 1.5),
        "speed_scale_range": (0.85, 1.15),
        "turn_none_prob": 0.78,
        "turn_angle_range_deg": (3.0, 15.0),
    },
    "加油机": {
        "accel_range_mps2": (0.3, 1.2),
        "speed_scale_range": (0.85, 1.15),
        "turn_none_prob": 0.88,
        "turn_angle_range_deg": (2.0, 8.0),
    },
    "客机": {
        "accel_range_mps2": (0.3, 1.0),
        "speed_scale_range": (0.85, 1.15),
        "turn_none_prob": 0.90,
        "turn_angle_range_deg": (2.0, 6.0),
    },
}


def get_aircraft_group(img_name_meta):
    """根据机型名返回运动参数大类。"""
    for prefix, group_name in AIRCRAFT_GROUP_RULES:
        if img_name_meta.startswith(prefix):
            return group_name
    return "运输机"


def sample_motion_state():
    """按配置概率抽取整段轨迹的速度状态。"""
    draw = random.random()
    cumulative = 0.0
    for state_name in ("constant", "accelerate", "decelerate"):
        cumulative += MOTION_STATE_PROBABILITIES[state_name]
        if draw <= cumulative:
            return state_name
    return "constant"


def build_speed_sequence_kmh(base_speed_kmh, motion_state, accel_mps2,
                             total_frames, fps, speed_scale_range):
    """生成逐帧真实对地速度序列，单位 km/h。"""
    if total_frames <= 0:
        return []

    min_speed_kmh = base_speed_kmh * speed_scale_range[0]
    max_speed_kmh = base_speed_kmh * speed_scale_range[1]
    accel_kmh_per_s = accel_mps2 * 3.6
    speed_seq_kmh = []

    for frame_idx in range(total_frames):
        time_sec = frame_idx / max(float(fps), 1e-6)
        if motion_state == "accelerate":
            speed_kmh = base_speed_kmh + accel_kmh_per_s * time_sec
        elif motion_state == "decelerate":
            speed_kmh = base_speed_kmh - accel_kmh_per_s * time_sec
        else:
            speed_kmh = base_speed_kmh

        speed_seq_kmh.append(float(np.clip(speed_kmh, min_speed_kmh, max_speed_kmh)))

    return speed_seq_kmh


def sample_turn_profile(aircraft_group, total_frames):
    """按机型大类抽取整段转弯参数。"""
    cfg = AIRCRAFT_MOTION_CONFIG[aircraft_group]
    roll = random.random()
    turn_total_deg = 0.0
    is_excited_turn = False

    if roll >= cfg["turn_none_prob"]:
        if aircraft_group == "战斗机" and roll < cfg["turn_none_prob"] + cfg["excited_turn_prob"]:
            angle_range = cfg["excited_turn_angle_range_deg"]
            is_excited_turn = True
        else:
            angle_range = cfg["turn_angle_range_deg"]
        turn_total_deg = random.uniform(*angle_range) * random.choice([-1.0, 1.0])

    turn_rate_deg_per_frame = turn_total_deg / max(total_frames - 1, 1)
    return {
        "turn_total_deg": turn_total_deg,
        "turn_rate_deg_per_frame": turn_rate_deg_per_frame,
        "is_excited_turn": is_excited_turn,
    }


def sample_motion_profile(img_name_meta, base_speed_kmh, total_frames, fps):
    """按机型大类采样整段轨迹的速度与转弯剖面。"""
    aircraft_group = get_aircraft_group(img_name_meta)
    group_cfg = AIRCRAFT_MOTION_CONFIG[aircraft_group]
    motion_state = sample_motion_state()
    accel_mps2 = 0.0 if motion_state == "constant" else random.uniform(*group_cfg["accel_range_mps2"])
    speed_seq_kmh = build_speed_sequence_kmh(
        base_speed_kmh, motion_state, accel_mps2, total_frames, fps,
        group_cfg["speed_scale_range"]
    )
    turn_profile = sample_turn_profile(aircraft_group, total_frames)

    return {
        "aircraft_group": aircraft_group,
        "motion_state": motion_state,
        "accel_mps2": accel_mps2,
        "speed_seq_kmh": speed_seq_kmh,
        "initial_speed_kmh": speed_seq_kmh[0] if speed_seq_kmh else base_speed_kmh,
        "speed_min_kmh": min(speed_seq_kmh) if speed_seq_kmh else base_speed_kmh,
        "speed_max_kmh": max(speed_seq_kmh) if speed_seq_kmh else base_speed_kmh,
        "turn_total_deg": turn_profile["turn_total_deg"],
        "turn_rate_deg_per_frame": turn_profile["turn_rate_deg_per_frame"],
        "is_excited_turn": turn_profile["is_excited_turn"],
    }


class SatelliteGeometry:
    """卫星观测几何参数 —— 正向公式：真实航向 + 卫星运动 → 像面表观轨迹

    核心公式（正向）：
      V_app = (Hs * V_true - Ha * V_sat) / (Hs - Ha)
      v_app_pix = V_app / (resolution * fps)
      像面轨迹: p_{k+1} = p_k + v_app_pix

    角度约定（与 traces angles 的 arctan2(dy,dx) 一致）：
      图像坐标系 x右 y下，0°=右，90°=下，-90°/270°=上
    """
    def __init__(self, resolution, image_shape,
                 coordinate_mode="image",
                 global_margin=1024):
        # 低轨卫星高度 160-2000 km
        self.H_sat = random.uniform(1600, 2000) * 1000  # m
        # 低轨卫星运行速度 7.2-7.5 km/s
        self.v_sat = random.uniform(7200, 7900)  # m/s
        # 卫星地面轨迹方向（随机），遵循 coordinate_mode 约定
        self.sat_heading = random.uniform(0, 360)  # degrees
        self.resolution = resolution
        self.coordinate_mode = coordinate_mode
        self.global_margin = global_margin

        # 凝视点：位于 patch 周围扩展区域（模拟大图裁剪）
        W, H = image_shape[1], image_shape[0]
        self.gaze_x = random.uniform(-global_margin, W + global_margin)
        self.gaze_y = random.uniform(-global_margin, H + global_margin)

        print(f"[卫星几何] Hs={self.H_sat/1000:.1f}km, "
              f"Vs={self.v_sat/1000:.2f}km/s, "
              f"方向={self.sat_heading:.1f}°, "
              f"凝视点=({self.gaze_x:.0f},{self.gaze_y:.0f}), "
              f"坐标模式={coordinate_mode}, margin={global_margin}")

    def get_sat_velocity_vector(self):
        """卫星速度向量 (m/s)，遵循 coordinate_mode 角度约定"""
        sat_rad = np.radians(self.sat_heading)
        v_sx = self.v_sat * np.cos(sat_rad)
        if self.coordinate_mode == "image":
            # 图像坐标: x右, y下, arctan2(dy,dx) 中 0°=右, 90°=下
            v_sy = self.v_sat * np.sin(sat_rad)
        elif self.coordinate_mode == "enu":
            # ENU: 东=x右, 北=y上 → 图像 y 轴取反
            v_sy = -self.v_sat * np.sin(sat_rad)
        else:
            raise ValueError(f"Unknown coordinate_mode: {self.coordinate_mode}")
        return v_sx, v_sy

    def get_apparent_velocity_from_true_heading(self, theta_heading_true, speed_kmh, H_ac, fps):
        """
        正向计算：真实航向 → 像面表观速度

        V_app = (Hs * V_true - Ha * V_sat) / (Hs - Ha)

        参数:
            theta_heading_true: 飞机真实机头/对地速度方向 (degrees, 图像坐标约定)
            speed_kmh: 飞机真实对地速度 (km/h)
            H_ac: 飞机高度 (m)
            fps: 帧率

        返回:
            v_app_x_pix: 像面表观速度 x 分量 (pixels/frame)
            v_app_y_pix: 像面表观速度 y 分量 (pixels/frame)
            theta_traj_app: 像面表观轨迹角 (degrees, 图像坐标约定)
        """
        Hs = self.H_sat
        Ha = H_ac

        # 飞机真实速度向量 (m/s)
        Va = speed_kmh / 3.6
        heading_rad = np.radians(theta_heading_true)
        V_true_x = Va * np.cos(heading_rad)
        V_true_y = Va * np.sin(heading_rad)

        # 卫星速度向量 (m/s)
        V_sat_x, V_sat_y = self.get_sat_velocity_vector()

        # 像面表观速度 (m/s): V_app = (Hs*V_true - Ha*V_sat) / (Hs-Ha)
        V_app_x = (Hs * V_true_x - Ha * V_sat_x) / (Hs - Ha)
        V_app_y = (Hs * V_true_y - Ha * V_sat_y) / (Hs - Ha)

        # 转换为像素/帧
        v_app_x_pix = V_app_x / (self.resolution * fps)
        v_app_y_pix = V_app_y / (self.resolution * fps)

        # 像面表观轨迹角
        theta_traj_app = np.degrees(np.arctan2(v_app_y_pix, v_app_x_pix))

        # 调试信息
        delta_angle = theta_traj_app - theta_heading_true
        print(f"[视差] θ_true={theta_heading_true:.1f}° θ_traj={theta_traj_app:.1f}° "
              f"Δ={delta_angle:+.1f}° | "
              f"Va={Va:.0f}m/s Vs={self.v_sat:.0f}m/s "
              f"Hs={Hs/1000:.1f}km Ha={Ha/1000:.3f}km | "
              f"v_app_pix=({v_app_x_pix:.2f},{v_app_y_pix:.2f}) px/fr")

        return v_app_x_pix, v_app_y_pix, theta_traj_app


def get_aircraft_altitude(img_name_meta):
    """根据飞机型号确定典型飞行高度 (m)"""
    aircraft_group = get_aircraft_group(img_name_meta)
    if aircraft_group in ("战斗机", "攻击机"):
        return random.uniform(3000, 20000)
    elif aircraft_group == "客机":
        return random.uniform(6800, 11000)
    elif aircraft_group == "轰炸机":
        return random.uniform(5000, 18000)
    elif aircraft_group == "预警机":
        return random.uniform(8000, 15000)
    elif aircraft_group == "侦察/电子侦察机":
        return random.uniform(5000, 15000)
    elif aircraft_group == "运输机":
        return random.uniform(5000, 15000)
    elif aircraft_group == "加油机":
        return random.uniform(8000, 15000)
    elif aircraft_group == "海上巡逻机":
        return random.uniform(3000, 10000)
    else:
        return random.uniform(5000, 15000)


# ======================== 轨迹生成（正向：真实航向 → 表观轨迹） ========================
def get_start_point(image_shape, margin, velocity=None, visible_half_size=0.0, sat_geom=None):
    """依据表观速度生成一个起始可见点，避免整段轨迹都落在视场外。"""
    H, W = image_shape
    safe_x = min(max(float(visible_half_size), 4.0), max(W / 2 - 1.0, 4.0))
    safe_y = min(max(float(visible_half_size), 4.0), max(H / 2 - 1.0, 4.0))
    x_low, x_high = safe_x, max(safe_x, W - safe_x)
    y_low, y_high = safe_y, max(safe_y, H - safe_y)

    if sat_geom is not None:
        gaze_x = float(np.clip(sat_geom.gaze_x, x_low, x_high))
        gaze_y = float(np.clip(sat_geom.gaze_y, y_low, y_high))
    else:
        gaze_x = (x_low + x_high) / 2
        gaze_y = (y_low + y_high) / 2

    if velocity is None:
        return (random.uniform(x_low, x_high), random.uniform(y_low, y_high))

    vx, vy = velocity
    if abs(vx) < 1e-6 and abs(vy) < 1e-6:
        return (random.uniform(x_low, x_high), random.uniform(y_low, y_high))

    if abs(vx) >= abs(vy):
        band = min(W * 0.18, max(12.0, W / 3))
        if vx >= 0:
            x = min(x_high, x_low + random.uniform(0, band))
        else:
            x = max(x_low, x_high - random.uniform(0, band))
        y = float(np.clip(gaze_y + random.uniform(-H * 0.15, H * 0.15), y_low, y_high))
    else:
        band = min(H * 0.18, max(12.0, H / 3))
        if vy >= 0:
            y = min(y_high, y_low + random.uniform(0, band))
        else:
            y = max(y_low, y_high - random.uniform(0, band))
        x = float(np.clip(gaze_x + random.uniform(-W * 0.15, W * 0.15), x_low, x_high))

    return (x, y)


def get_formation_starts(lead_start, theta_heading_true, base_distance_pix, biandui_shuliang):
    """生成编队僚机的起始点（与长机相同航向、相同速度，仅起始位置偏移）

    偏移方向：垂直于航向方向（右侧为正），沿航向方向延伸。
    """
    heading_rad = np.radians(theta_heading_true)
    # 垂直航向的单位向量（右侧）
    perp_x = -np.sin(heading_rad)
    perp_y = np.cos(heading_rad)
    # 沿航向的单位向量（前方）
    along_x = np.cos(heading_rad)
    along_y = np.sin(heading_rad)

    k1 = random.uniform(0.8, 1.0)
    k2 = random.uniform(1.5, 2.0)
    d1 = k1 * base_distance_pix
    d2 = k2 * base_distance_pix
    ext1 = -k1 * base_distance_pix
    ext2 = -k2 * base_distance_pix

    x0, y0 = lead_start
    starts = []

    if biandui_shuliang == 2:
        sx = x0 + perp_x * d1 + along_x * ext1
        sy = y0 + perp_y * d1 + along_y * ext1
        starts.append((sx, sy))

    elif biandui_shuliang == 3:
        # 左侧僚机
        starts.append((x0 - perp_x * d1 + along_x * ext1,
                       y0 - perp_y * d1 + along_y * ext1))
        # 右侧僚机
        starts.append((x0 + perp_x * d1 + along_x * ext1,
                       y0 + perp_y * d1 + along_y * ext1))

    elif biandui_shuliang == 5:
        # 内左、内右
        starts.append((x0 - perp_x * d1 + along_x * ext1,
                       y0 - perp_y * d1 + along_y * ext1))
        starts.append((x0 + perp_x * d1 + along_x * ext1,
                       y0 + perp_y * d1 + along_y * ext1))
        # 外左、外右
        starts.append((x0 - perp_x * d2 + along_x * ext2,
                       y0 - perp_y * d2 + along_y * ext2))
        starts.append((x0 + perp_x * d2 + along_x * ext2,
                       y0 + perp_y * d2 + along_y * ext2))

    if len(starts) != biandui_shuliang - 1:
        raise ValueError(f"编队数量不匹配: 期望{biandui_shuliang-1}个僚机, 生成{len(starts)}个")
    return starts


def generate_apparent_trajectory(start, theta_heading_base, speed_seq_kmh, H_ac,
                                  sat_geom, total_frames, fps, image_shape,
                                  turn_rate_deg_per_frame=0.0,
                                  obj_margin=64, visible_half_size=0.0):
    """从真实航向正向生成像面表观轨迹

    对每一帧 k：
      1. theta_true_k = theta_heading_base + turn_rate_deg_per_frame * k
      2. V_app_k = (Hs*V_true_k - Ha*V_sat) / (Hs-Ha)
      3. v_app_pix_k = V_app_k / (resolution * fps)
      4. p_{k+1} = p_k + v_app_pix_k

    Args:
        start: 起始点 (x, y)
        theta_heading_base: 基础真实航向 (degrees)
        speed_seq_kmh: 逐帧真实速度序列 (km/h)
        H_ac: 飞机高度 (m)
        sat_geom: SatelliteGeometry 实例
        total_frames: 总帧数
        fps: 帧率
        image_shape: patch 尺寸 (H, W)
        turn_rate_deg_per_frame: 转弯速率 (degrees/frame), 0=直线
        obj_margin: 可见性判定外扩像素

    Returns:
        trajectory: [(x,y), ...] 每帧像素坐标
        angles: [theta_traj_app, ...] 每帧表观轨迹角
        is_trace: [0/1, ...] 每帧可见性
        theta_heading_true_seq: [theta_true_k, ...] 每帧真实航向（用于贴图旋转）
    """
    W, H = image_shape[1], image_shape[0]
    trajectory = []
    angles = []
    is_trace = []
    theta_heading_true_seq = []

    x, y = float(start[0]), float(start[1])
    visible_half_size = float(max(0.0, visible_half_size))

    if np.isscalar(speed_seq_kmh):
        speed_seq_kmh = [float(speed_seq_kmh)] * total_frames
    else:
        speed_seq_kmh = list(speed_seq_kmh)

    if len(speed_seq_kmh) < total_frames:
        fill_value = speed_seq_kmh[-1] if speed_seq_kmh else 0.0
        speed_seq_kmh.extend([fill_value] * (total_frames - len(speed_seq_kmh)))

    for k in range(total_frames):
        theta_k = theta_heading_base + turn_rate_deg_per_frame * k
        speed_kmh = speed_seq_kmh[k]
        theta_heading_true_seq.append(theta_k)
        vx_pix, vy_pix, theta_traj = sat_geom.get_apparent_velocity_from_true_heading(
            theta_k, speed_kmh, H_ac, fps)
        trajectory.append((int(x), int(y)))
        angles.append(theta_traj)
        if visible_half_size > 0:
            visible = (x + visible_half_size > 0 and
                       x - visible_half_size < W and
                       y + visible_half_size > 0 and
                       y - visible_half_size < H)
        else:
            visible = (-obj_margin < x < W + obj_margin and
                       -obj_margin < y < H + obj_margin)
        is_trace.append(1 if visible else 0)
        x += vx_pix
        y += vy_pix

    return trajectory, angles, is_trace, theta_heading_true_seq


# ======================== 飞机选型 ========================
def random_select_by_area(planes_list, kjfbl):
    '''
    3m
        极小目标:0.55,小目标:0.3, 中目标:0.1, 大目标:0.04,更大目标:0.01
        10, ['战斗机F15', '战斗机F2', '战斗机F16', '战斗机F18', '战斗机F22', '战斗机F35', '战斗机Su35', '战斗机MiG31', '攻击机Su24', '攻击机Su34']
        1, ['运输机C1A']
        17, ['轰炸机B2', '轰炸机Tu22', '预警机E737', '侦察机EP3', '侦察机P3C', '侦察机Il20', '运输机C130', '运输机An12', '运输机Il18', '运输机C40', '反潜机P8', '反潜机Il38', '反潜机P1', '民航机A220', '民航机A321', '民航机ARJ21', '民航机B737']
        12, ['轰炸机B1B', '轰炸机Tu95', '预警机E3', '预警机E6', '预警机E8', '预警机E767', '预警机A50', '侦察机RC135', '运输机Il62', '运输机XC2', '加油机KC135', '加油机KC46']
        14, ['轰炸机B52', '轰炸机Tu160', '预警机E4', '运输机C17', '运输机C5', '运输机Il76', '运输机An124', '运输机An22', '加油机KC10', '民航机A330', '民航机A350', '民航机B747', '民航机B777', '民航机B787']
    5m
        极小目标:0.4,小目标:0.45, 中目标:0.1, 大目标:0.04,更大目标:0.01
        9, ['战斗机F15', '战斗机F2', '战斗机F16', '战斗机F18', '战斗机F22', '战斗机F35', '战斗机Su35', '战斗机MiG31', '攻击机Su34']
        2, ['攻击机Su24', '运输机C1A']
        16, ['轰炸机B2', '轰炸机Tu22', '预警机E737', '侦察机EP3', '侦察机P3C', '侦察机Il20', '运输机C130', '运输机An12', '运输机Il18', '运输机C40', '反潜机P8', '反潜机P1', '加油机KC135', '民航机A220', '民航机ARJ21', '民航机B737']
        13, ['轰炸机B1B', '预警机E3', '预警机E6', '预警机E8', '预警机E767', '预警机A50', '侦察机RC135', '运输机Il76', '运输机Il62', '运输机XC2', '反潜机Il38', '加油机KC46', '民航机A321']
        14, ['轰炸机B52', '轰炸机Tu95', '轰炸机Tu160', '预警机E4', '运输机C17', '运输机C5', '运输机An124', '运输机An22', '加油机KC10', '民航机A330', '民航机A350', '民航机B747', '民航机B777', '民航机B787']
    :param planes_list:
    :param kjfbl:
    :return:
    '''
    if kjfbl == 3:
        group1_p, group2_p, group3_p, group4_p, group5_p = 0.63, 0.2, 0.1, 0.05, 0.02
    elif kjfbl == 5:
        group1_p, group2_p, group3_p, group4_p, group5_p = 0.32, 0.35, 0.25, 0.05, 0.03
    group1 = []  # 面积 < 20*20=400
    group2 = []  # 400 ≤ 面积 < 900
    group3 = []  # 900 ≤ 面积 < 1600
    group4 = []  # 1600 ≤ 面积 < 2500
    group5 = []  # 2500 ≤ 面积

    for plane in planes_list:
        model = plane[0]
        h = plane[1][1]
        w = plane[1][2]
        area = h * w
        if kjfbl == 3:
            if area < 1139:
                group1.append(model)
            elif area < 1638:
                group2.append(model)
            elif area < 2136:
                group3.append(model)
            elif area < 3133:
                group4.append(model)
            else:
                group5.append(model)
        if kjfbl == 5:
            if area < 641:
                group1.append(model)
            elif area < 1638:
                group2.append(model)
            elif area < 2635:
                group3.append(model)
            elif area < 3632:
                group4.append(model)
            else:
                group5.append(model)

    groups = []
    group_weights = []
    if group1:
        groups.append(group1)
        group_weights.append(group1_p)
    if group2:
        groups.append(group2)
        group_weights.append(group2_p)
    if group3:
        groups.append(group3)
        group_weights.append(group3_p)
    if group4:
        groups.append(group4)
        group_weights.append(group4_p)
    if group5:
        groups.append(group5)
        group_weights.append(group5_p)

    if not groups:
        raise ValueError("没有可用的飞机目标")

    total_weight = sum(group_weights)
    normalized_weights = [w / total_weight for w in group_weights]
    print(f"极小目标:{normalized_weights[0]},小目标:{normalized_weights[1]}, 中目标:{normalized_weights[2]}, 大目标:{normalized_weights[3]},更大目标:{normalized_weights[4]}")

    selected_group = random.choices(groups, weights=normalized_weights, k=1)[0]
    selected_model = random.choice(selected_group)
    print(f"{len(group1)}, {group1}")
    print(f"{len(group2)}, {group2}")
    print(f"{len(group3)}, {group3}")
    print(f"{len(group4)}, {group4}")
    print(f"{len(group5)}, {group5}")
    return selected_model


# ======================== 轨迹 & 飞机信息生成 ========================
def get_info(target_resolution, image_shape, class_mode, key_list, img_ori_path,
             motion_blur, total_frames, fps, kjfbl, sat_geom,
             base_distance_range=(1.5, 2.5), obj_margin=64):
    """正向生成：先随机真实航向，再结合卫星几何计算像面表观轨迹"""
    biandui_shuliang = 1
    plane_num = random.randint(1, 3)

    traces = []
    plane_info = []
    for j in range(plane_num):
        if class_mode == 'quanbu':
            img_class = random_select_by_area(motion_blur.dis_size.items(), kjfbl)
        else:
            img_class_zhongdian = ['民航机B737', '民航机B747', '轰炸机B2', '轰炸机B52', '加油机KC135', '预警机E3', '预警机E767', '战斗机F22']
            img_class = random.sample(img_class_zhongdian, 1)[0]

        img_path_list = glob.glob(img_ori_path + img_class + '/*.png')
        img_path = random.sample(img_path_list, 1)[0]
        img_name = os.path.split(img_path)[-1]
        img_name_meta = img_name[:-4].split('_')[0]
        if img_name_meta != img_class:
            raise ValueError(f"型号不匹配：文件夹{img_class}包含{img_name_meta}的图片")
        dis = motion_blur.dis_size[img_name_meta][0]
        severity = math.ceil(dis / target_resolution[0] / 2)
        w_ori = motion_blur.dis_size[img_name_meta][1]
        h_ori = motion_blur.dis_size[img_name_meta][2]
        speed_kmh = motion_blur.dis_size[img_name_meta][4]
        motion_profile = sample_motion_profile(img_name_meta, speed_kmh, total_frames, fps)
        speed_seq_kmh = motion_profile["speed_seq_kmh"]
        aircraft_group = motion_profile["aircraft_group"]
        altitude = get_aircraft_altitude(img_name_meta)
        print(f"{img_class}: group={aircraft_group}, base_speed={speed_kmh:.1f}km/h, altitude={altitude:.0f}m")

        # ---- 1. 随机生成真实航向（机头/对地速度方向） ----
        theta_heading_true = random.uniform(0, 360)
        turn_rate_deg_per_frame = motion_profile["turn_rate_deg_per_frame"]
        visible_half_size = max(w_ori, h_ori) / target_resolution[0] / 2 + severity + 2

        vx0_pix, vy0_pix, _ = sat_geom.get_apparent_velocity_from_true_heading(
            theta_heading_true, motion_profile["initial_speed_kmh"], altitude, fps)

        # ---- 2. 获取长机起始点 ----
        # 从实际表观速度反推一个“起始即可见”的位置，避免只在视场外生成轨迹。
        lead_start = get_start_point(
            image_shape,
            sat_geom.global_margin,
            velocity=(vx0_pix, vy0_pix),
            visible_half_size=visible_half_size,
            sat_geom=sat_geom
        )

        # ---- 3. 处理编队 ----
        starts = [lead_start]
        current_biandui_shuliang = 1
        biandui_shuliang = current_biandui_shuliang
        if aircraft_group == "战斗机":
            biandui = True if random.random() < 0.8 else False
            if biandui:
                current_biandui_shuliang = random.choice([2, 2, 3, 3, 5])
                biandui_shuliang = current_biandui_shuliang
                plane_num += current_biandui_shuliang - 1
                print(f"biandui_shuliang:{current_biandui_shuliang}")
                base_distance = max(w_ori, h_ori) * random.uniform(*base_distance_range) / target_resolution[0]
                wingman_starts = get_formation_starts(lead_start, theta_heading_true,
                                                       base_distance, current_biandui_shuliang)
                starts.extend(wingman_starts)

        # ---- 4. 对每架飞机（长机+僚机）生成轨迹 ----
        for i, start in enumerate(starts):
            trajectory, angles, is_trace, theta_seq = generate_apparent_trajectory(
                start, theta_heading_true, speed_seq_kmh, altitude,
                sat_geom, total_frames, fps, image_shape,
                turn_rate_deg_per_frame=turn_rate_deg_per_frame,
                obj_margin=obj_margin,
                visible_half_size=visible_half_size)

            traces.append({
                "trajectory": trajectory,
                "angles": angles,                        # 表观轨迹方向 θ_traj_app
                "is_trace": is_trace,
                "theta_heading_true_seq": theta_seq,     # 每帧真实航向（用于贴图旋转）
                "aircraft_group": aircraft_group,
                "motion_state": motion_profile["motion_state"],
                "accel_mps2": motion_profile["accel_mps2"],
                "speed_seq_kmh": speed_seq_kmh,
                "speed_min_kmh": motion_profile["speed_min_kmh"],
                "speed_max_kmh": motion_profile["speed_max_kmh"],
                "turn_total_deg": motion_profile["turn_total_deg"],
                "turn_rate_deg_per_frame": turn_rate_deg_per_frame,
                "is_excited_turn": motion_profile["is_excited_turn"]
            })
            print(f"轨迹[{i}]: 起点{start}, "
                  f"θ_true={theta_heading_true:.1f}°, turn_total={motion_profile['turn_total_deg']:.2f}°, "
                  f"turn_rate={turn_rate_deg_per_frame:.3f}°/fr, "
                  f"speed_state={motion_profile['motion_state']}, "
                  f"有效{sum(is_trace)}/{len(is_trace)}")

            # ---- 5. 处理飞机图像（与原有逻辑一致） ----
            img = cv_imread(img_path, flags=1)
            img_gray = cv_imread(img_path, flags=0)
            plane_shadow_ = cv_imread(img_path, flags=-1)
            _, _, _, a = cv2.split(plane_shadow_)
            plane_shadow = np.zeros_like(img_gray)
            plane_mask = np.zeros_like(img_gray)
            plane_shadow[a == 0] = 255
            plane_mask[a != 0] = 1
            mask = img_gray * plane_mask
            mask_sum = (mask != 0).sum()
            plane_gray_mean = mask.sum() / mask_sum
            print(f"飞机{img_name_meta}飞行高度: {altitude:.0f}m")

            plane_info.append({
                "img": img, "plane_shadow": plane_shadow,
                "img_name_meta": img_name_meta, "severity": severity, "dis": dis,
                "w_ori": w_ori, "h_ori": h_ori, "speed_kmh": speed_kmh,
                "plane_gray_mean": plane_gray_mean,
                "biandui_shuliang": current_biandui_shuliang, "altitude": altitude,
                "theta_heading_true": theta_heading_true,  # 基础真实航向
                "aircraft_group": aircraft_group,
                "motion_state": motion_profile["motion_state"],
                "accel_mps2": motion_profile["accel_mps2"],
                "speed_seq_kmh": speed_seq_kmh,
                "initial_speed_kmh": motion_profile["initial_speed_kmh"],
                "speed_min_kmh": motion_profile["speed_min_kmh"],
                "speed_max_kmh": motion_profile["speed_max_kmh"],
                "turn_total_deg": motion_profile["turn_total_deg"],
                "turn_rate_deg_per_frame": turn_rate_deg_per_frame,
                "is_excited_turn": motion_profile["is_excited_turn"]
            })

    return plane_num, traces, plane_info, biandui_shuliang


# ======================== 主仿真流程 ========================
def main(RS_video_path, image_shape=[270, 480], save_dir_root=r'../../项目_飞机仿真/', video_name='',
         sense='', cloud_dict=None, base_distance_range=(1.5, 2.5),
         global_margin=1024, obj_margin=64, coordinate_mode="image"):
    Add_global_noise = PhysicalBasedSimulator()
    class_mode = 'quanbu'
    ph, pw = 31, 31
    result, save_dir = extract_numbers_from_path(RS_video_path)
    resolution, fps = result
    exposure_time = 0.002 + random.random() * 0.013

    RS_path_list = glob.glob(fr"{RS_video_path}/*.png")
    total_frames = len(RS_path_list)
    img_ori_path = r'../飞机切片/切片6/'
    path_txt = r'./标称巡航速度查询文件4.txt'

    gray_diff_th1 = 150
    diff_radio_max1 = 0

    save_dir_ori2 = save_dir_root
    os.makedirs(save_dir_ori2, exist_ok=True)
    save_xmldir_ori2 = save_dir_ori2
    save_maskdir_ori2 = save_dir_ori2 + '/mask/'
    os.makedirs(save_maskdir_ori2, exist_ok=True)

    RS_path_list = sorted(RS_path_list)

    target_resolution = [resolution, resolution]
    motion_blur = MOTION_BLUR(target_resolution, path_txt, exposure_time)

    # 初始化卫星观测几何（每个视频固定一套卫星参数）
    sat_geom = SatelliteGeometry(resolution, image_shape,
                                  coordinate_mode=coordinate_mode,
                                  global_margin=global_margin)

    key_list, rbboxes_list01, rbboxes_list02, rbboxes_list03, hbboxes_list02, mask_list02 = [], [], [], [], [], []

    for key, point in motion_blur.dis_size.items():
        key_list.append(key)

    flag_get_info = True
    flag_save = True
    active_list = []
    plane_num = 0
    traces = []
    plane_info = []
    successful_frame = 0
    i = 0

    is_active_atmospheric = random.choice([False, False])
    transparency = random.uniform(0.85, 0.98)
    random_atmospheric_light = random.uniform(0.9, 1.1)

    is_active_sensor = random.choice([False, False, False, False])
    noise_qiangdu = [random.uniform(0.05, 0.25), random.uniform(0.01, 0.05)]
    blur_edge_strength = random.uniform(0.3, 0.6)

    with tqdm(total=len(RS_path_list)) as pbar:
        while i < len(RS_path_list):
            flag_save = True
            RS_img_path = RS_path_list[i]

            if flag_get_info:
                plane_num_temp, traces_temp, plane_info_temp, biandui_shuliang = get_info(
                    target_resolution, image_shape, class_mode, key_list, img_ori_path,
                    motion_blur, total_frames, fps, resolution, sat_geom,
                    base_distance_range=base_distance_range, obj_margin=obj_margin)
                plane_num = plane_num + plane_num_temp
                flag_get_info = False
                for _ in range(plane_num_temp):
                    active_list.append(1)
                traces.extend(traces_temp)
                plane_info.extend(plane_info_temp)
                successful_frame = 0
                print(f"num本轮目标类别为：{plane_num_temp + 1 - biandui_shuliang}")

            RS_img = cv_imread(RS_img_path, flags=1)
            RS_img_h, RS_img_w = RS_img.shape[:2]
            save_name = os.path.split(RS_img_path)[-1][:-4]

            RS_img02 = RS_img.copy()
            RS_cxcy_list = []

            print(f"[{i + 1}]:processing {RS_img_path}……")
            print("active object:", active_list)

            cloud_deactive_temp = 0
            diejia_flag = False  # random.random() < 0 恒为 False

            for j in range(plane_num):
                for ii in reversed(range(len(traces[j]['is_trace']))):
                    if traces[j]['is_trace'][ii] == 1:
                        break
                if successful_frame > ii:
                    active_list[j] = 0
                    print(f"id:{j}飞机飞出视野")
                    if sum(active_list) == 0:
                        flag_get_info = True
                        successful_frame = 0
                        i -= 1
                        print("重置flag_get_info")
                        flag_save = False
                        break
                    continue
                if active_list[j] == 0:
                    print(f"id:{j}飞机飞出视野")
                    continue
                if traces[j]['is_trace'][successful_frame] == 1:
                    RS_cx, RS_cy = traces[j]['trajectory'][successful_frame]
                    RS_patch = motion_blur.get_patch2(RS_img, RS_cx, RS_cy, ph, pw)
                else:
                    continue
                RS_gray_patch = cv2.cvtColor(RS_patch, cv2.COLOR_BGR2GRAY)

                img = plane_info[j]['img']
                plane_shadow = plane_info[j]['plane_shadow']
                img_name_meta = plane_info[j]['img_name_meta']
                img, plane_shadow, w, h = motion_blur.shape_size(img, plane_shadow)
                img, plane_shadow = random_transform_img_and_shadow(img, plane_shadow)

                ori_resolution = motion_blur.get_ori_resolution(plane_shadow, img_name_meta, motion_blur.dis_size)

                RS_patch_ori, RS_patch_ori_size = motion_blur.img_resize(RS_patch, target_resolution, ori_resolution)

                img_gd02 = img.astype('int')
                img_gd02[img_gd02 > 255] = 255
                img_gd02 = img_gd02.astype('uint8')

                # ---- 航向视差：直接使用预计算的真实航向与表观轨迹角 ----
                # 表观轨迹方向（像面运动方向，用于运动模糊）
                theta_traj_app = traces[j]['angles'][successful_frame]
                # 真实机头朝向（已由 get_info 正向生成）
                theta_heading_true = traces[j]['theta_heading_true_seq'][successful_frame]

                # 飞机旋转角度：基于真实航向（机头朝向）
                angle2 = -theta_heading_true - 90
                # 运动模糊方向：基于像面表观轨迹方向
                trajectory_angle_for_blur = -theta_traj_app - 90
                angle3 = -((90 + trajectory_angle_for_blur) % 180)
                cloud_mask = None
                severity = plane_info[j]['severity']
                dis = plane_info[j]['dis']
                w_ori = plane_info[j]['w_ori']
                h_ori = plane_info[j]['h_ori']

                RS_patch_temp, _ = motion_blur.img_resize(RS_patch_ori, ori_resolution, target_resolution)

                if sense == 'cloud' or sense == 'cloudA' or sense == 'cloudB':
                    RS_img_gray_avg, plane_img_gray_avg, RS_patch_shadow_gray_min = motion_blur.mix_img(
                        RS_patch_ori, img_gd02, plane_shadow,
                        RS_patch_ori_size[0] / 2, RS_patch_ori_size[1] / 2,
                        rotate_angle=angle2, diejia_flag=diejia_flag, for_avg=True)
                    cloud_area = cloud_dict[video_name]
                    for area in cloud_area:
                        x_left, y_left = area[0]
                        x_right, y_right = area[1]
                        if x_left < RS_cx < x_right and y_left < RS_cy < y_right:
                            print("厚云区域")
                            yun = True
                            break
                        else:
                            yun = False
                    if RS_img_gray_avg > 130 and RS_patch_shadow_gray_min > 100 and yun:
                        cloud_deactive_temp += 1
                        print("云层遮住！")
                        rbboxes_list02.append([])
                        hbboxes_list02.append([])
                        continue
                    elif RS_img_gray_avg > 95 and RS_patch_shadow_gray_min > 85:
                        print("薄云！")
                        diejia_flag = True

                _, _, img_mix02, img_shadow02, _, _ = motion_blur.mix_img(
                    RS_patch_ori, img_gd02, plane_shadow,
                    RS_patch_ori_size[0] / 2, RS_patch_ori_size[1] / 2,
                    rotate_angle=angle2, diejia_flag=diejia_flag)
                if np.min(img_shadow02) == 255:
                    continue

                img_mix02, _ = motion_blur.img_resize(img_mix02, ori_resolution, target_resolution)
                img_shadow02, _ = motion_blur.img_resize(img_shadow02, ori_resolution, target_resolution)

                img_blur02, shadow_blur02 = motion_blur.blur_limpid(img_mix02, img_shadow02, img_name_meta, severity, angle=angle3)
                wwhh = [w_ori / target_resolution[0], h_ori / target_resolution[0]]

                RS_img_out02 = RS_img02.copy()
                RS_cx, RS_cy, RS_mix02, shadow_blur_mix02, plane_cloud_index, wj_index = motion_blur.mix_img(
                    RS_img_out02, img_blur02, shadow_blur02, RS_cy, RS_cx, rotate_angle=0, wwhh=wwhh, RS_cxcy_list=RS_cxcy_list)
                print(f"id:{j}, cx:{RS_cx}, cy:{RS_cy}, xh:{plane_info[j]['img_name_meta']}")

                RS_mix02 = blur_full_inner_region(RS_mix02, shadow_blur_mix02, blur_strength=blur_edge_strength)

                RS_mix02 = Add_global_noise.simulate_atmospheric_effects(
                    RS_mix02, transparency=transparency,
                    random_atmospheric_light=random_atmospheric_light,
                    is_active_atmospheric=is_active_atmospheric)
                RS_mix02 = Add_global_noise.add_sensor_noise_global(
                    RS_mix02, 'mixed', is_active_sensor=is_active_sensor, noise_qiangdu=noise_qiangdu)
                RS_img02 = RS_mix02

                lx, ly, rx, ry, mask = get_hbbox(shadow_blur_mix02, RS_cx, RS_cy, RS_img_out02.shape[:2])
                if lx == ly == rx == ry == 0:
                    continue
                mask_list02.append(mask)
                hbboxes_list02.append([img_name_meta, lx, ly, rx, ry, j, plane_info[j]['biandui_shuliang']])
                r_RS_cx, r_RS_cy, r_w, r_h = bias_cxcywh(RS_cx, RS_cy, w_ori, h_ori, angle2, dis / target_resolution[0], target_resolution)
                rbboxes_list02.append([img_name_meta, r_RS_cx, r_RS_cy, r_w, r_h, float((360 + angle2) / 180 * np.pi), str(j)])

            if flag_save:
                tif_name = os.path.join(save_dir_ori2, save_name + '.png')
                mask_name = os.path.join(save_dir_ori2, 'mask', save_name + '.jpg')
                if mask_list02 == []:
                    print('[mask_list02]')
                    mask_hb02 = np.zeros((512, 512), dtype=np.uint8)
                else:
                    mask_hb02 = merge_masks(mask_list02)
                cv_imwrite(tif_name, RS_img02)
                cv_imwrite(mask_name, mask_hb02)
                rotate_xml_name02 = os.path.join(save_xmldir_ori2, save_name + '.xml')
                write_rotate_xml(rotate_xml_name02, RS_img_h, RS_img_w, rbboxes_list02, hbboxes_list02)
                rbboxes_list02 = []
                hbboxes_list02 = []
                mask_list02 = []
                successful_frame += 1
                print(f"successful_frame:{successful_frame}")

            i += 1
            pbar.update(1)

    print("finished")


# ======================== 工具函数 ========================
def extract_numbers_from_path(path_str):
    path_parts = [part for part in path_str.split("/") if part]
    target_substr = path_parts[-1]
    substr_parts = [sub for sub in target_substr.split("_") if sub]
    numbers = []
    for part in substr_parts[-2:]:
        try:
            num = int(part)
            numbers.append(num)
        except ValueError:
            num = float(part)
            numbers.append(num)
    return numbers, target_substr


def extract_sort_number(file_path):
    file_name = os.path.basename(file_path)
    match = re.search(r'副本 \((\d+)\)', file_name)
    if match:
        return int(match.group(1))
    else:
        return 0


# ======================== 入口 ========================
if __name__ == '__main__':
    from Cv2ForChinese import *

    # ======================== 配置：改为 'test' 或 'train' ========================
    MODE = 'test'  # 'test' 或 'train'

    # ======================== 云层字典 ========================
    cloud_dict_test = {
        'cloudA_852_1_2_3_5': [[(0,51),(55,158)],[(92,0),(260,57)],[(134,55),(192,77)],[(0,205),(146,335)],[(0,344),(119,512)],[(378,454),(512,512)]],
        'cloudA_852_1_2_5_10':[[(0,460),(54,512)],[(26,213),(194,442)],[(188,318),(310,390)],[(135,382),(291,463)],[(313,311),(388,360)],
                               [(331,293),(389,306)],[(347,281),(387,296)],[(452,344),(490,429)],[(472,219),(512,317)]],
        'cloudA_852_1_3_3_10':[[(45,0),(361,121)],[(150,121),(342,154)],[(199,155),(331,213)],[(225,216),(394,316)],[(136,378),(384,468)],
                               [(179,454),(250,498)],[(316,446),(493,512)]],
        'cloudA_852_2_2_3_5': [[(0,0),(208,119)],[(0,114),(40,146)],[(184,62),(411,262)],[(30,383),(129,484)],[(416,313),(512,512)],[(428,364),(512,512)]],
        'cloudA_852_2_2_5_10':[[(339,0),(430,28)],[(374,66),(384,128)],[(307,121),(397,163)],[(300,164),(475,324)],[(218,291),(443,437)],
                               [(444,405),(512,512)],[(67,450),(193,512)],[(90,205),(148,253)]],
        'cloudA_852_2_3_3_10':[[(137,0),(427,48)],[(167,48),(447,304)],[(22,155),(370,398)],[(424,376),(512,512)],[(388,447),(512,512)]],
        'cloudB_852_1_2_3_5': [[(254,0),(429,30)],[(478,0),(512,95)],[(360,147),(512,275)],[(397,273),(512,399)],
                               [(414,395),(512,512)],[(384,441),(512,512)],[(76,411),(146,458)]],
        'cloudB_852_1_2_5_10':[[(0,240),(26,330)],[(0,349),(63,432)],[(0,347),(67,432)],[(96,280),(175,344)],[(106,341),(364,363)],[(124,354),(236,384)],
                               [(173,301),(504,384)],[(192,383),(426,465)],[(234,421),(320,485)],[(343,221),(481,328)],[(421,444),(512,512)],
                               [(255,99),(299,149)],[(85,0),(132,30)]],
        'cloudB_852_1_3_3_10':[[(0,42),(45,196)],[(43,241),(121,251)],[(26,250),(124,264)],[(0,261),(122,353)],[(30,350),(122,377)],
                               [(175,125),(289,225)],[(195,226),(365,299)],[(290,166),(512,312)],[(375,155),(494,171)],[(341,298),(512,398)],
                               [(386,397),(512,512)]],
        'cloudB_852_2_2_3_5': [[(0,0),(141,234)],[(165,255),(378,512)],[(298,389),(512,512)],[(444,61),(512,155)]],
        'cloudB_852_2_2_5_10':[[(0,0),(62,168)],[(134,93),(314,217)],[(391,0),(458,85)],[(476,0),(512,38)],[(9,208),(226,337)],[(48,328),(310,372)],
                               [(97,358),(217,430)],[(135,415),(217,439)],[(138,438),(236,480)],[(461,448),(512,512)]],
        'cloudB_852_2_3_3_10':[[(0,154),(280,481)],[(141,28),(375,242)],[(321,408),(512,512)],[(485,0),(512,34)],[(241,12),(329,47)],[(451,49),(492,70)]]
    }

    cloud_dict_train = {
        'cloudA_86_1_1_5_10': [[(55,22),(103,196)],[(96,43),(131,73)],[(109,0),(161,11)]],
        'cloudA_86_2_1_3_10': [[(194,0),(512,68)],[(319,60),(436,112)],[(489,80),(512,199)]],
        'cloudA_2021_1_1_5_5': [[(97,42),(217,121)],[(262,83),(376,131)],[(83,124),(207,276)],[(52,442),(205,512)]],
        'cloudA_2021_1_2_5_5': [[(167,58),(345,320)],[(72,0),(165,57)],[(174,376),(269,437)]],
        'cloudA_2021_1_3_5_5': [[(0,0),(512,512)]],
        'cloudA_2021_1_4_5_5': [[(0,0),(512,512)]],
        'cloudA_2021_1_5_5_5': [[(0,0),(512,512)]],
        'cloudA_2021_2_7_5_5': [[(0,0),(512,512)]],
        'cloudA_2021_3_2_5_5': [[(0,0),(512,512)]],
        'cloudA_2021_3_7_5_5': [[(0,0),(512,512)]],
        'cloudA_2021_4_2_3_5': [[(0,0),(512,512)]],
        'cloudA_2021_4_3_3_5': [[(0,0),(256,512)]],
        'cloudA_2021_5_2_3_5': [[(0,0),(512,512)]],
        'cloudA_2021_5_3_3_5': [[(0,0),(512,512)]],
        'cloudB_86_1_1_5_10': [[(337,50),(512,131)],[(289,0),(354,26)]],
        'cloudB_86_2_1_3_10': [[(271,352),(512,512)],[(457,253),(512,360)]],
        'cloudB_2021_1_1_5_5': [[(98,0),(262,69)],[(128,207),(265,377)],[(49,360),(512,512)]],
        'cloudB_2021_1_2_5_5': [[(0,0),(512,512)]],
        'cloudB_2021_1_3_5_5': [[(0,0),(512,512)]],
        'cloudB_2021_1_4_5_5': [[(0,0),(512,512)]],
        'cloudB_2021_1_5_5_5': [[(0,0),(512,512)]],
        'cloudB_2021_2_7_5_5': [[(0,0),(512,512)]],
        'cloudB_2021_3_2_5_5': [[(0,0),(512,512)]],
        'cloudB_2021_3_7_5_5': [[(0,0),(512,512)]],
        'cloudB_2021_4_2_3_5': [[(0,0),(512,512)]],
        'cloudB_2021_4_3_3_5': [[(0,0),(512,512)]],
        'cloudB_2021_5_2_3_5': [[(0,0),(512,512)]],
        'cloudB_2021_5_3_3_5': [[(0,0),(512,512)]],
    }

    # ======================== 模式相关配置 ========================
    if MODE == 'test':
        cloud_dict = cloud_dict_test
        RANDOM_SEED = 210
        INPUT_DIR = r"E:/NUDT-Master/Academic/20250919SimData/New3/test_temp3"
        OUTPUT_DIR = r"E:/NUDT-Master/Academic/20250919SimData/New3/test_temp3v26"
        BASE_DISTANCE_RANGE = (2.0, 3.5)
    else:  # 'train'
        cloud_dict = cloud_dict_train
        RANDOM_SEED = 420
        INPUT_DIR = r"E:/NUDT-Master/Academic/20250919SimData/New3/train_temp3"
        OUTPUT_DIR = r"E:/NUDT-Master/Academic/20250919SimData/New3/train_temp3v26"
        BASE_DISTANCE_RANGE = (2.0, 3.5)

    # ======================== 卫星观测几何与高度视差修正 ========================
    GLOBAL_MARGIN = 1024          # 起点/凝视点可在 patch 外部的扩展范围 (pixels)
    OBJ_MARGIN = 64               # is_trace 可见性判定扩展范围 (pixels)
    COORDINATE_MODE = "image"     # "image": 图像坐标 x右y下, arctan2(dy,dx)中 0°=右 90°=下

    print(f"当前模式: {MODE}")
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    video_name_list = os.listdir(INPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    finished = os.listdir(OUTPUT_DIR)

    for video_name in video_name_list:
        sense, fps = video_name.split('_')[0], video_name.split('_')[-1]
        main(RS_video_path=fr"{INPUT_DIR}/{video_name}".replace("\\", "/"),
             save_dir_root=fr"{OUTPUT_DIR}/{video_name}".replace("\\", "/"),
             image_shape=[512, 512],
             video_name=video_name,
             sense=sense,
             cloud_dict=cloud_dict,
             base_distance_range=BASE_DISTANCE_RANGE,
             global_margin=GLOBAL_MARGIN,
             obj_margin=OBJ_MARGIN,
             coordinate_mode=COORDINATE_MODE)
