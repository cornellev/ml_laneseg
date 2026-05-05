import rclpy
from rclpy.node import Node
import pyzed.sl as sl
import torch
import torch.nn.functional as F
import sys
import time
from torchvision import transforms
from sensor_msgs.msg import Image, PointCloud2, PointField
from std_msgs.msg import Header, String
import sensor_msgs_py.point_cloud2 as pc2
from cv_bridge import CvBridge
import cv2
import numpy as np
import os
import json
import struct
import scipy.ndimage as ndimage
from scipy.signal import find_peaks

current_dir   = os.path.dirname(os.path.abspath(__file__))
LFD_REPO_PATH = "/home/mini-dos/ml_laneseg/src/LFD_RoadSeg"
if os.path.exists(LFD_REPO_PATH):
    sys.path.append(LFD_REPO_PATH)

try:
    from models._LFDRoadSeg import LFD_RoadSeg
    print("Success: LFD_RoadSeg imported.")
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)


# ════════════════════════════════════════════════════════════════
#  POINT CLOUD BUILDER
#  Schema published on lane/points/lanes:
#    x, y, z      — 3D position in ZED camera frame (metres)
#    crossable    — 1.0 = may cross (dashed), 0.0 = must not cross (solid)
#    point_type   — 0.0 = lane boundary edge, 1.0 = lane centre line
#    lane_id      — integer lane index (0, 1, 2 …)
# ════════════════════════════════════════════════════════════════

FIELDS = [
    PointField(name='x',          offset=0,  datatype=PointField.FLOAT32, count=1),
    PointField(name='y',          offset=4,  datatype=PointField.FLOAT32, count=1),
    PointField(name='z',          offset=8,  datatype=PointField.FLOAT32, count=1),
    PointField(name='crossable',  offset=12, datatype=PointField.FLOAT32, count=1),
    PointField(name='point_type', offset=16, datatype=PointField.FLOAT32, count=1),
    PointField(name='lane_id',    offset=20, datatype=PointField.FLOAT32, count=1),
]
POINT_STEP = 24   # 6 × float32


def build_pointcloud2(points, frame_id, stamp):
    """
    points : list of (x,y,z,crossable,point_type,lane_id) tuples or Nx6 ndarray
    Returns a PointCloud2 message.
    """
    header           = Header()
    header.stamp     = stamp
    header.frame_id  = frame_id

    if isinstance(points, np.ndarray):
        pts = points.astype(np.float32)
    else:
        pts = np.array(points, dtype=np.float32)

    if pts.ndim != 2 or pts.shape[1] != 6:
        pts = pts.reshape(-1, 6)

    data = pts.tobytes()
    msg               = PointCloud2()
    msg.header        = header
    msg.height        = 1
    msg.width         = len(pts)
    msg.fields        = FIELDS
    msg.is_bigendian  = False
    msg.point_step    = POINT_STEP
    msg.row_step      = POINT_STEP * len(pts)
    msg.data          = data
    msg.is_dense      = True
    return msg


def sample_valid_3d(pc_data, ys, xs, subsample=1):
    """
    Given pixel coordinates, return valid Nx3 XYZ points from ZED point cloud.
    """
    if subsample > 1:
        idx = np.arange(0, len(ys), subsample)
        ys, xs = ys[idx], xs[idx]
    ys = np.clip(ys, 0, pc_data.shape[0] - 1)
    xs = np.clip(xs, 0, pc_data.shape[1] - 1)
    pts = pc_data[ys, xs, :3].astype(np.float32)
    valid = (
        np.isfinite(pts).all(axis=1) &
        (pts[:, 2] > 0.10) &
        (pts[:, 2] < 30.0)
    )
    return pts[valid]


def lane_mask_to_points(pc_data, cam_mask, crossable, lane_id, subsample_boundary=2, subsample_center=4):
    """
    Given a camera-space lane mask:
      - Extracts boundary ring pixels  → point_type = 0
      - Extracts skeleton / centreline → point_type = 1
    Returns Nx6 float32 array.
    """
    rows = []

    # ── BOUNDARY ────────────────────────────────────────────────
    thickness = 5
    erode_k   = np.ones((thickness * 2 + 1, thickness * 2 + 1), np.uint8)
    eroded    = cv2.erode(cam_mask, erode_k, iterations=1)
    boundary  = cv2.bitwise_xor(cam_mask, eroded)

    b_ys, b_xs = np.where(boundary > 0)
    if len(b_ys) > 0:
        pts3d = sample_valid_3d(pc_data, b_ys, b_xs, subsample=subsample_boundary)
        if len(pts3d) > 0:
            extras = np.column_stack([
                np.full(len(pts3d), 1.0 if crossable else 0.0, dtype=np.float32),  # crossable
                np.zeros(len(pts3d), dtype=np.float32),                              # point_type=0 boundary
                np.full(len(pts3d), float(lane_id), dtype=np.float32),              # lane_id
            ])
            rows.append(np.hstack([pts3d, extras]))

    # ── CENTRE LINE ─────────────────────────────────────────────
    # For each row in the mask, find the median x column → centreline pixel
    ys_all, xs_all = np.where(cam_mask > 0)
    if len(ys_all) > 0:
        centre_ys, centre_xs = [], []
        for row in np.unique(ys_all)[::subsample_center]:
            cols = xs_all[ys_all == row]
            if len(cols) > 0:
                centre_ys.append(row)
                centre_xs.append(int(np.median(cols)))
        c_ys = np.array(centre_ys, dtype=np.int32)
        c_xs = np.array(centre_xs, dtype=np.int32)
        if len(c_ys) > 0:
            pts3d = sample_valid_3d(pc_data, c_ys, c_xs, subsample=1)
            if len(pts3d) > 0:
                extras = np.column_stack([
                    np.full(len(pts3d), 1.0 if crossable else 0.0, dtype=np.float32),
                    np.ones(len(pts3d), dtype=np.float32),                            # point_type=1 centre
                    np.full(len(pts3d), float(lane_id), dtype=np.float32),
                ])
                rows.append(np.hstack([pts3d, extras]))

    if not rows:
        return np.zeros((0, 6), dtype=np.float32)
    return np.vstack(rows).astype(np.float32)


def intersection_mask_to_points(pc_data, dir_mask, direction, lane_id, subsample=3):
    """For intersection branches — boundary + centre, crossable always True."""
    return lane_mask_to_points(
        pc_data, dir_mask,
        crossable=True,
        lane_id=lane_id,
        subsample_boundary=subsample,
        subsample_center=subsample * 2,
    )


# ════════════════════════════════════════════════════════════════
#  LANE DETECTION  (straight road)
# ════════════════════════════════════════════════════════════════

def fill_road_convex(road_mask):
    contours, _ = cv2.findContours(
        road_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return road_mask.copy()
    hull   = cv2.convexHull(max(contours, key=cv2.contourArea))
    filled = np.zeros_like(road_mask)
    cv2.drawContours(filled, [hull], -1, 255, cv2.FILLED)
    return filled


def detect_bev_intersection_rows(bright_clean, road_bev, orig_h, orig_w):
    intersection_rows = np.zeros(orig_h, dtype=np.uint8)
    road_widths = road_bev.sum(axis=1) / 255.0
    for row in range(orig_h):
        if road_widths[row] < 10:
            continue
        bright_in_road = bright_clean[row, :][road_bev[row, :] > 0]
        if len(bright_in_road) == 0:
            continue
        if bright_in_road.sum() / 255.0 / road_widths[row] > 0.40:
            intersection_rows[row] = 1
    kernel = np.ones(15, dtype=np.uint8)
    intersection_rows = np.convolve(
        intersection_rows, kernel, mode='same'
    ).clip(0, 1).astype(np.uint8)
    flagged = intersection_rows[road_widths > 10].sum()
    total   = (road_widths > 10).sum()
    return intersection_rows, (total > 0 and flagged / total > 0.60)


def classify_lane_marking(bright_bev_full, peak_col, road_bev, orig_h, road_width):
    half  = max(10, road_width // 25)
    col_l = max(0, peak_col - half)
    col_r = min(bright_bev_full.shape[1], peak_col + half)
    road_rows    = (road_bev[:, col_l:col_r].sum(axis=1)        > 0)
    marking_rows = (bright_bev_full[:, col_l:col_r].sum(axis=1) > 0)
    both         = road_rows & marking_rows
    road_cnt     = road_rows.sum()
    if road_cnt == 0:
        return {'type': 'unknown', 'crossable': True, 'fill_ratio': 0.0}
    fill_ratio = both.sum() / road_cnt
    col_signal = both.astype(np.uint8)
    runs_on, runs_off, last_val, run_len = [], [], int(col_signal[0]), 1
    for px in col_signal[1:]:
        px = int(px)
        if px == last_val:
            run_len += 1
        else:
            (runs_on if last_val == 1 else runs_off).append(run_len)
            last_val, run_len = px, 1
    (runs_on if last_val == 1 else runs_off).append(run_len)
    meaningful_gaps = [g for g in runs_off if g > 8]
    if len(runs_on) >= 2 and len(meaningful_gaps) >= 1:
        t, c = 'dashed', True
    elif fill_ratio > 0.75:
        t, c = 'solid', False
    elif len(meaningful_gaps) >= 1 and fill_ratio < 0.65:
        t, c = 'dashed', True
    elif fill_ratio > 0.55:
        t, c = 'solid', False
    elif fill_ratio > 0.05:
        t, c = 'dashed', True
    else:
        t, c = 'unknown', True
    return {'type': t, 'crossable': c, 'fill_ratio': round(fill_ratio, 2)}


def classify_marking_color(img_bev, bright_bev_full, peak_col, road_width):
    half  = max(10, road_width // 25)
    col_l = max(0, peak_col - half)
    col_r = min(img_bev.shape[1], peak_col + half)
    ys, xs = np.where(bright_bev_full[:, col_l:col_r] > 0)
    if len(ys) == 0:
        return 'unknown'
    hsv    = cv2.cvtColor(img_bev, cv2.COLOR_BGR2HSV)
    px_hsv = hsv[ys, xs + col_l]
    H, S, V = px_hsv[:, 0], px_hsv[:, 1], px_hsv[:, 2]
    n = len(H)
    if ((H >= 15) & (H <= 35) & (S > 80) & (V > 100)).sum() / n > 0.25:
        return 'yellow'
    if ((S < 50) & (V > 160)).sum() / n > 0.25:
        return 'white'
    return 'unknown'


def fit_marking_polyline(bright_clean, peak_col, road_bev,
                         orig_h, road_width, intersection_rows):
    band_height = max(orig_h // 20, 15)
    search_half = road_width // 8
    lane_xs_by_row = {}
    ys, xs = np.where(bright_clean > 0)
    for y, x in zip(ys, xs):
        if intersection_rows[y]:
            continue
        if abs(x - peak_col) < search_half:
            band = (y // band_height) * band_height + band_height // 2
            lane_xs_by_row.setdefault(band, []).append(x)
    band_points = [
        (by, int(np.median(xl)))
        for by, xl in sorted(lane_xs_by_row.items()) if len(xl) >= 3
    ]
    if len(band_points) >= 2:
        bys = np.array([p[0] for p in band_points], dtype=np.float32)
        bxs = np.array([p[1] for p in band_points], dtype=np.float32)
        w   = (bys / orig_h) ** 2
        return np.poly1d(np.polyfit(bys, bxs, deg=1, w=w)), band_points
    return np.poly1d([0, peak_col]), []


def find_valid_seed(mask, cx, cy, radius=80):
    for r in range(0, radius, 3):
        for dx in range(-r, r + 1, 3):
            nx = int(np.clip(cx + dx, 0, mask.shape[1] - 1))
            if mask[cy, nx] > 0:
                return nx, cy
    return None, None


def segment_straight_road(img_bgr, road_mask, orig_h, orig_w):
    """
    Returns:
      colored_cam   (H,W,3) BGR overlay image
      lane_results  list of {'cam_mask', 'crossable', 'lane_id', 'marking_info'}
      marking_infos list of dicts
    """
    src_pts = np.float32([
        [orig_w * 0.43, orig_h * 0.62],
        [orig_w * 0.57, orig_h * 0.62],
        [orig_w * 0.80, orig_h * 0.95],
        [orig_w * 0.20, orig_h * 0.95],
    ])
    dst_pts = np.float32([
        [orig_w * 0.25, 0],       [orig_w * 0.75, 0],
        [orig_w * 0.75, orig_h],  [orig_w * 0.25, orig_h],
    ])
    M    = cv2.getPerspectiveTransform(src_pts, dst_pts)
    Minv = cv2.getPerspectiveTransform(dst_pts, src_pts)

    road_bev_raw = cv2.warpPerspective(road_mask, M, (orig_w, orig_h), flags=cv2.INTER_NEAREST)
    img_bev      = cv2.warpPerspective(img_bgr,   M, (orig_w, orig_h))
    road_bev     = fill_road_convex(road_bev_raw)

    gray_bev      = cv2.cvtColor(img_bev, cv2.COLOR_BGR2GRAY)
    road_gray_bev = cv2.bitwise_and(gray_bev, gray_bev, mask=road_bev)
    valid_px      = road_gray_bev[road_bev > 0]
    if len(valid_px) == 0:
        return np.zeros_like(img_bgr), [], []

    thresh = np.percentile(valid_px, 88)
    _, bright_bev = cv2.threshold(road_gray_bev, thresh, 255, cv2.THRESH_BINARY)

    bright_clean = np.zeros_like(bright_bev)
    n_lbl, lbl_map, stats, _ = cv2.connectedComponentsWithStats(bright_bev, connectivity=8)
    for i in range(1, n_lbl):
        bw, bh, area = (stats[i, cv2.CC_STAT_WIDTH],
                        stats[i, cv2.CC_STAT_HEIGHT],
                        stats[i, cv2.CC_STAT_AREA])
        if bh >= bw * 1.0 and area > 60 and bh < orig_h * 0.7:
            bright_clean[lbl_map == i] = 255

    int_rows, _ = detect_bev_intersection_rows(bright_clean, road_bev, orig_h, orig_w)

    sample_row = road_bev[int(orig_h * 0.85), :]
    road_cols  = np.where(sample_row > 0)[0]
    if len(road_cols) < 4:
        road_cols = np.where(road_bev.sum(axis=0) > orig_h * 0.1)[0]
    if len(road_cols) < 4:
        return np.zeros_like(img_bgr), [], []

    road_left   = int(road_cols[0])
    road_right  = int(road_cols[-1])
    road_width  = road_right - road_left
    road_center = (road_left + road_right) // 2
    edge_margin  = road_width // 8
    inner_margin = road_width // 10
    inner_left   = road_left  + inner_margin
    inner_right  = road_right - inner_margin

    clean_hist = bright_clean.copy()
    clean_hist[int_rows > 0, :] = 0
    col_hist = clean_hist[orig_h // 2:, :].sum(axis=0).astype(np.float32)
    smooth   = ndimage.gaussian_filter1d(col_hist, sigma=max(orig_w // 60, 6))
    gated    = smooth.copy()
    gated[:road_left  + edge_margin] = 0
    gated[road_right - edge_margin:] = 0

    min_lane_w = road_width // 6
    peak_h     = gated.max() * 0.15 if gated.max() > 0 else 1
    peak_pr    = gated.max() * 0.08 if gated.max() > 0 else 1
    peaks, _   = find_peaks(gated, height=peak_h, distance=min_lane_w, prominence=peak_pr)

    wall_half     = max(8, road_width // 22)
    all_walls     = []
    marking_infos = []
    combined_wall = np.zeros_like(road_bev)

    for peak in peaks:
        poly, _ = fit_marking_polyline(
            bright_clean, int(peak), road_bev, orig_h, road_width, int_rows
        )
        wall_mask = np.zeros_like(road_bev)
        for row in range(orig_h):
            if int_rows[row]:
                continue
            cx = int(np.clip(poly(row), road_left + 5, road_right - 5))
            wl, wr = max(0, cx - wall_half), min(orig_w, cx + wall_half)
            if road_bev[row, cx] > 0:
                wall_mask[row, wl:wr] = 255
        col_bottom = int(np.clip(poly(int(orig_h * 0.85)), road_left + 5, road_right - 5))
        all_walls.append((poly, wall_mask, col_bottom, int(peak)))
        combined_wall = cv2.bitwise_or(combined_wall, wall_mask)

        info  = classify_lane_marking(bright_bev, int(peak), road_bev, orig_h, road_width)
        color = classify_marking_color(img_bev, bright_bev, int(peak), road_width)
        info['color'] = color
        info['col']   = col_bottom
        marking_infos.append(info)

    if len(all_walls) == 0:
        seed_xs = [road_center]
    else:
        bxs = [inner_left] + sorted([w[2] for w in all_walls]) + [inner_right]
        seed_xs = [(bxs[i] + bxs[i+1]) // 2 for i in range(len(bxs) - 1)]

    road_carved = cv2.morphologyEx(
        cv2.bitwise_and(road_bev, cv2.bitwise_not(combined_wall)),
        cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    )

    lane_colors_bgr = [
        (255,  80, 180), ( 60, 210,  60), (  0, 160, 255),
        (  0, 200, 180), (255, 160,   0), (200,  60, 200),
    ]

    seed_y      = int(orig_h * 0.88)
    colored_bev = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
    remaining   = road_carved.copy()
    lane_results = []
    lane_count   = 0

    for idx, sx in enumerate(seed_xs):
        sx, sy = find_valid_seed(remaining, sx, seed_y)
        if sx is None:
            continue
        flood_mask = np.zeros((orig_h + 2, orig_w + 2), np.uint8)
        fill_img   = remaining.copy()
        cv2.floodFill(fill_img, flood_mask, (sx, sy), 128)
        lane_px = fill_img == 128
        if lane_px.sum() < 500:
            continue

        bev_mask_uint8 = lane_px.astype(np.uint8) * 255
        colored_bev[lane_px] = lane_colors_bgr[idx % len(lane_colors_bgr)]
        remaining[lane_px]   = 0

        # Unwarp to camera space for 3D lookup
        cam_mask = cv2.warpPerspective(bev_mask_uint8, Minv, (orig_w, orig_h),
                                       flags=cv2.INTER_NEAREST)

        # Crossable = the marking to the RIGHT of this lane
        crossable = True
        if idx < len(marking_infos):
            crossable = marking_infos[idx]['crossable']

        lane_results.append({
            'cam_mask':    cam_mask,
            'crossable':   crossable,
            'lane_id':     lane_count,
            'marking_info': marking_infos[idx] if idx < len(marking_infos) else {},
        })
        lane_count += 1

    colored_cam = cv2.warpPerspective(colored_bev, Minv, (orig_w, orig_h),
                                      flags=cv2.INTER_NEAREST)
    return colored_cam, lane_results, marking_infos


# ════════════════════════════════════════════════════════════════
#  INTERSECTION DETECTION + SEGMENTATION
# ════════════════════════════════════════════════════════════════

def check_intersection(road_mask, orig_h, orig_w):
    hz_top, hz_bottom = int(orig_h * 0.35), int(orig_h * 0.55)
    hz_band   = road_mask[hz_top:hz_bottom, :]
    strip_w   = orig_w // 5
    hz_left   = hz_band[:, :strip_w].mean() / 255.0
    hz_right  = hz_band[:, orig_w - strip_w:].mean() / 255.0
    hz_center = hz_band[:, strip_w:orig_w - strip_w].mean() / 255.0
    lower_band  = road_mask[int(orig_h * 0.45):, :]
    side_w      = orig_w // 6
    lower_left  = lower_band[:, :side_w].mean() / 255.0
    lower_right = lower_band[:, orig_w - side_w:].mean() / 255.0
    mid_w       = (road_mask[int(orig_h * 0.50), :] > 0).sum()
    bot_w       = (road_mask[int(orig_h * 0.85), :] > 0).sum()
    width_ratio = bot_w / max(mid_w, 1)
    has_left    = (hz_left  > 0.05) or (width_ratio > 1.8 and lower_left  > 0.08)
    has_right   = (hz_right > 0.05) or (width_ratio > 1.8 and lower_right > 0.08)
    has_ahead   = hz_center > 0.05
    branches    = []
    if has_ahead: branches.append('ahead')
    if has_left:  branches.append('left')
    if has_right: branches.append('right')
    return (has_left or has_right), branches


def recover_intersection_road(road_mask, orig_h, orig_w):
    close_k  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    closed   = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, close_k)
    dk       = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (orig_w // 10, orig_w // 10))
    dilated  = cv2.dilate(closed, dk, iterations=1)
    dilated[:int(orig_h * 0.38), :] = 0
    full     = fill_road_convex(dilated)
    dk2      = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (orig_w // 6, orig_w // 6))
    generous = cv2.dilate(road_mask, dk2, iterations=1)
    generous[:int(orig_h * 0.38), :] = 0
    return cv2.bitwise_and(full, generous)


def segment_intersection(road_mask, branches, orig_h, orig_w):
    import math
    full_road = recover_intersection_road(road_mask, orig_h, orig_w)
    mid_top, mid_bottom = int(orig_h * 0.35), int(orig_h * 0.65)
    ys, xs = np.where(road_mask[mid_top:mid_bottom, :] > 0)
    cx = int(np.median(xs)) if len(xs) > 0 else orig_w // 2
    cy = int(np.median(ys)) + mid_top if len(ys) > 0 else int(orig_h * 0.50)

    yy, xx    = np.mgrid[0:orig_h, 0:orig_w]
    angle_map = np.arctan2((yy - cy).astype(np.float32), (xx - cx).astype(np.float32))

    def deg(x): return x * math.pi / 180.0
    sector_defs = {
        'ego':   [(deg(55),  deg(125))],
        'ahead': [(deg(-125), deg(-55))],
        'right': [(deg(-55),  deg(55))],
        'left':  [(deg(125),  deg(180)), (deg(-180), deg(-125))],
    }

    direction_masks = {}
    dir_colors_bgr  = {
        'ego':   (  0, 140, 255),
        'ahead': ( 60, 210,  60),
        'left':  (200,  60, 200),
        'right': (220, 160,  40),
    }
    colored = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)

    for direction, intervals in sector_defs.items():
        if direction != 'ego' and direction not in branches:
            continue
        sector = np.zeros((orig_h, orig_w), dtype=np.uint8)
        for (a_min, a_max) in intervals:
            if a_min <= a_max:
                sector |= ((angle_map >= a_min) & (angle_map <= a_max)).astype(np.uint8)
            else:
                sector |= ((angle_map >= a_min) | (angle_map <= a_max)).astype(np.uint8)
        region = cv2.bitwise_and(sector * 255, full_road)
        if (region > 0).sum() >= 300:
            direction_masks[direction] = region
            colored[region > 0] = dir_colors_bgr[direction]

    return colored, direction_masks, (cx, cy)


# ════════════════════════════════════════════════════════════════
#  OVERLAY RENDERER
# ════════════════════════════════════════════════════════════════

def render_overlay(img_bgr, colored_cam, mode, lane_count,
                   marking_infos, found_dirs, int_center=None):
    """
    Returns overlay image (color regions + HUD on black background).
    Raw image is published separately untouched.
    """
    orig_h, orig_w = img_bgr.shape[:2]
    overlay = np.zeros_like(img_bgr)

    # Paint lane/branch colours
    mask = colored_cam.any(axis=2)
    overlay[mask] = colored_cam[mask]

    # Intersection arrows
    if mode == 'intersection' and int_center is not None:
        cx, cy    = int_center
        arrow_len = max(orig_h // 9, 40)
        thickness = max(3, orig_h // 100)
        arrow_cols = {
            'left':  (200,  60, 200),
            'right': (220, 160,  40),
            'ahead': ( 60, 210,  60),
        }
        for d in found_dirs:
            if d == 'left':
                cv2.arrowedLine(overlay, (cx, cy), (cx - arrow_len, cy),
                                arrow_cols['left'], thickness, tipLength=0.35)
            elif d == 'right':
                cv2.arrowedLine(overlay, (cx, cy), (cx + arrow_len, cy),
                                arrow_cols['right'], thickness, tipLength=0.35)
            elif d == 'ahead':
                cv2.arrowedLine(overlay, (cx, cy), (cx, cy - arrow_len),
                                arrow_cols['ahead'], thickness, tipLength=0.35)
        cv2.circle(overlay, (cx, cy), 6, (255, 255, 255), -1)

    # HUD bar
    bar_h = 38
    cv2.rectangle(overlay, (0, orig_h - bar_h), (orig_w, orig_h), (0, 0, 0), -1)
    hud_left = "intersection" if mode == 'intersection' else \
               f"{lane_count} lane{'s' if lane_count != 1 else ''}"
    cv2.putText(overlay, hud_left, (12, orig_h - bar_h + 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 1, cv2.LINE_AA)

    # Chips
    dir_bgr  = {'ahead': (60,160,60), 'left': (180,40,160),
                'right': (40,130,200), 'ego': (0,110,220)}
    x_cursor = 200
    chips    = list(found_dirs) if mode == 'intersection' else \
               [(f"{m['color']} {m['type']}", m['crossable']) for m in marking_infos]

    for item in chips:
        if mode == 'intersection':
            label    = item
            chip_bgr = dir_bgr.get(label, (80, 80, 80))
        else:
            label, crossable = item
            chip_bgr = (0, 0, 180)   if not crossable       else \
                       (0, 140, 220) if 'yellow' in label   else \
                       (0, 160,  60)
        chip_text   = f" {label} "
        (tw, th), _ = cv2.getTextSize(chip_text, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)
        cv2.rectangle(overlay,
                      (x_cursor, orig_h - bar_h + 5),
                      (x_cursor + tw + 4, orig_h - 5), chip_bgr, -1)
        cv2.putText(overlay, chip_text,
                    (x_cursor + 2, orig_h - bar_h + 5 + th + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)
        x_cursor += tw + 10

    return overlay


# ════════════════════════════════════════════════════════════════
#  ROS2 NODE
# ════════════════════════════════════════════════════════════════

class LaneSegmentationNode3D(Node):
    def __init__(self, name='lane_segmentation_node'):
        super().__init__(name)
        self.start_time    = time.time()
        self.frame_counter = 0

        # ── ZED 2 ────────────────────────────────────────────────
        self.zed = sl.Camera()
        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.VGA
        init_params.camera_fps        = 30
        init_params.depth_mode        = sl.DEPTH_MODE.NEURAL
        init_params.coordinate_units  = sl.UNIT.METER

        if self.zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
            self.get_logger().error("Failed to open ZED camera")
            exit(-1)

        tracking_params = sl.PositionalTrackingParameters()
        if self.zed.enable_positional_tracking(tracking_params) == sl.ERROR_CODE.SUCCESS:
            self.get_logger().info("ZED 2 positional tracking enabled.")

        self.image_zed_left = sl.Mat()
        self.point_cloud    = sl.Mat()
        self.runtime_params = sl.RuntimeParameters()

        # ── AI MODEL ─────────────────────────────────────────────
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model  = LFD_RoadSeg(scale_factor=2)
        weights_path = os.path.join(current_dir, 'model_epoch_150.pth')
        checkpoint   = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
        self.model.to(self.device).half().eval()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.3598, 0.3653, 0.3662],
                                 std =[0.2573, 0.2663, 0.2756]),
        ])

        # ── PUBLISHERS  (exactly 3 topics) ───────────────────────
        self.bridge = CvBridge()

        # 1. Raw untouched camera feed
        self.pub_raw = self.create_publisher(
            Image, 'lane/camera_raw', 10
        )
        # 2. Colour overlay (black bg, lane regions + HUD)
        self.pub_overlay = self.create_publisher(
            Image, 'lane/overlay', 10
        )
        # 3. All lane boundaries + centrelines in one PointCloud2
        #    Fields: x, y, z, crossable, point_type, lane_id
        self.pub_lanes_pc = self.create_publisher(
            PointCloud2, 'lane/points/lanes', 10
        )

        self.timer = self.create_timer(1.0 / 30.0, self.timer_callback)
        self.get_logger().info(
            "LaneSegmentationNode3D ready.\n"
            "  Topics:\n"
            "    lane/camera_raw        — raw ZED left image\n"
            "    lane/overlay           — colour overlay + HUD\n"
            "    lane/points/lanes      — PointCloud2 (x,y,z,crossable,point_type,lane_id)"
        )

    # ─────────────────────────────────────────────────────────────
    def timer_callback(self):
        if self.zed.grab(self.runtime_params) != sl.ERROR_CODE.SUCCESS:
            return

        # ── 1. RETRIEVE ZED DATA ─────────────────────────────────
        self.zed.retrieve_image(self.image_zed_left, sl.VIEW.LEFT)
        self.zed.retrieve_measure(self.point_cloud,  sl.MEASURE.XYZ)

        image_data = self.image_zed_left.get_data()   # BGRA uint8
        pc_data    = self.point_cloud.get_data()       # (H, W, 4) float32

        img_bgr  = cv2.cvtColor(image_data, cv2.COLOR_BGRA2BGR)
        img_rgb  = cv2.cvtColor(image_data, cv2.COLOR_BGRA2RGB)
        orig_h, orig_w = img_bgr.shape[:2]

        stamp    = self.get_clock().now().to_msg()
        frame_id = 'zed_left_camera_frame'

        # ── 2. AI ROAD MASK ──────────────────────────────────────
        resized = cv2.resize(img_rgb, (624, 192))
        tensor  = self.transform(resized).unsqueeze(0).to(self.device).half()
        with torch.no_grad():
            output = self.model({"img": tensor})
            if isinstance(output, (list, tuple)):
                output = output[0]
        probs        = F.softmax(output, dim=1)
        mask_float   = F.interpolate(probs, size=(orig_h, orig_w),
                                     mode='bilinear', align_corners=False
                                     ).cpu().float().numpy()
        road_mask_8u = (mask_float[0, 1] > 0.5).astype(np.uint8) * 255

        # ── 3. INTERSECTION CHECK ────────────────────────────────
        at_intersection, branches = check_intersection(road_mask_8u, orig_h, orig_w)

        all_lane_points = []   # will be Nx6 float32, one row per 3D point
        colored_cam     = np.zeros_like(img_bgr)
        lane_count      = 0
        marking_infos   = []
        found_dirs      = {}
        int_center      = None

        # ── 4A. INTERSECTION MODE ────────────────────────────────
        if at_intersection:
            colored_cam, direction_masks, int_center = segment_intersection(
                road_mask_8u, branches, orig_h, orig_w
            )
            dir_id = {'ego': 0, 'ahead': 1, 'left': 2, 'right': 3}
            for direction, mask in direction_masks.items():
                found_dirs[direction] = True
                pts = intersection_mask_to_points(
                    pc_data, mask, direction,
                    lane_id=dir_id.get(direction, 9)
                )
                if len(pts) > 0:
                    all_lane_points.append(pts)

        # ── 4B. STRAIGHT ROAD MODE ───────────────────────────────
        else:
            colored_cam, lane_results, marking_infos = segment_straight_road(
                img_bgr, road_mask_8u, orig_h, orig_w
            )
            lane_count = len(lane_results)
            for lr in lane_results:
                pts = lane_mask_to_points(
                    pc_data,
                    lr['cam_mask'],
                    crossable=lr['crossable'],
                    lane_id=lr['lane_id'],
                )
                if len(pts) > 0:
                    all_lane_points.append(pts)

        # ── 5. PUBLISH RAW IMAGE ─────────────────────────────────
        ros_raw              = self.bridge.cv2_to_imgmsg(img_bgr, encoding='bgr8')
        ros_raw.header.stamp = stamp
        ros_raw.header.frame_id = frame_id
        self.pub_raw.publish(ros_raw)

        # ── 6. PUBLISH OVERLAY ───────────────────────────────────
        overlay = render_overlay(
            img_bgr, colored_cam,
            mode='intersection' if at_intersection else 'road',
            lane_count=lane_count,
            marking_infos=marking_infos,
            found_dirs=found_dirs,
            int_center=int_center,
        )
        ros_overlay              = self.bridge.cv2_to_imgmsg(overlay, encoding='bgr8')
        ros_overlay.header.stamp = stamp
        ros_overlay.header.frame_id = frame_id
        self.pub_overlay.publish(ros_overlay)

        # ── 7. PUBLISH LANE POINT CLOUD ──────────────────────────
        if all_lane_points:
            merged = np.vstack(all_lane_points).astype(np.float32)
            pc_msg = build_pointcloud2(merged, frame_id, stamp)
            self.pub_lanes_pc.publish(pc_msg)

            # Log summary
            boundary_pts = merged[merged[:, 4] == 0.0]
            centre_pts   = merged[merged[:, 4] == 1.0]
            nocross_pts  = merged[merged[:, 3] == 0.0]
            self.get_logger().info(
                f"[{'INT' if at_intersection else 'RD'}] "
                f"total={len(merged)} boundary={len(boundary_pts)} "
                f"centre={len(centre_pts)} no-cross={len(nocross_pts)}",
                throttle_duration_sec=1.0
            )

        # ── 8. FPS ───────────────────────────────────────────────
        self.frame_counter += 1
        now = time.time()
        if now - self.start_time > 2.0:
            fps = self.frame_counter / (now - self.start_time)
            self.get_logger().info(f'--- {fps:.1f} FPS ---')
            self.frame_counter = 0
            self.start_time    = now


# ════════════════════════════════════════════════════════════════
def main(args=None):
    rclpy.init(args=args)
    node = LaneSegmentationNode3D()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.zed.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()