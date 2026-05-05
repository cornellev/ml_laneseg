import rclpy
from rclpy.node import Node
import torch
import torch.nn.functional as F
import sys
from torchvision import transforms
import cv2
import numpy as np
import os
import scipy.ndimage as ndimage
from scipy.signal import find_peaks

current_dir   = os.path.dirname(os.path.abspath(__file__))
LFD_REPO_PATH = "/ros2_ws/src/LFD_RoadSeg"

if os.path.exists(LFD_REPO_PATH):
    sys.path.append(LFD_REPO_PATH)

try:
    from models._LFDRoadSeg import LFD_RoadSeg
    print("Success: LFD_RoadSeg imported.")
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════
#  HELPER: convex hull fill of largest road contour
# ═══════════════════════════════════════════════════════════
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


# ═══════════════════════════════════════════════════════════
#  HELPER: recover full road footprint at intersection
#  Dilates LFD mask to fill gaps, then convex hull
# ═══════════════════════════════════════════════════════════
def recover_intersection_road(road_mask_8u, orig_h, orig_w):
    # Step 1: close small gaps in LFD prediction
    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    closed  = cv2.morphologyEx(road_mask_8u, cv2.MORPH_CLOSE, close_k)

    # Step 2: dilate generously to reach side roads LFD missed
    dilate_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (orig_w // 10, orig_w // 10))
    dilated  = cv2.dilate(closed, dilate_k, iterations=1)

    # Step 3: mask out sky/upper image (above 40% height) — dilation shouldn't go there
    dilated[:int(orig_h * 0.38), :] = 0

    # Step 4: convex hull of result to get clean intersection footprint
    full_road = fill_road_convex(dilated)

    # Step 5: clip back to areas that are plausibly road colored
    # (prevents painting grass/sidewalk that got pulled into the hull)
    # Re-AND with a more aggressively dilated version of the original
    dilate_k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (orig_w // 6, orig_w // 6))
    generous  = cv2.dilate(road_mask_8u, dilate_k2, iterations=1)
    generous[:int(orig_h * 0.38), :] = 0
    full_road = cv2.bitwise_and(full_road, generous)

    return full_road


# ═══════════════════════════════════════════════════════════
#  HELPER: find intersection center point
#  The center is where road from all directions converges —
#  approximately where the bottom of the "ahead" area meets
#  the top of the "ego" area.
# ═══════════════════════════════════════════════════════════
def find_intersection_center(road_mask_8u, orig_h, orig_w):
    # Use the centroid of road pixels in the middle band
    mid_top    = int(orig_h * 0.35)
    mid_bottom = int(orig_h * 0.65)
    mid_band   = road_mask_8u[mid_top:mid_bottom, :]
    ys, xs     = np.where(mid_band > 0)
    if len(xs) == 0:
        return orig_w // 2, int(orig_h * 0.50)
    cx = int(np.median(xs))
    cy = int(np.median(ys)) + mid_top
    return cx, cy


# ═══════════════════════════════════════════════════════════
#  HELPER: intersection check
# ═══════════════════════════════════════════════════════════
def check_intersection(road_mask_camera, orig_h, orig_w):
    hz_top    = int(orig_h * 0.35)
    hz_bottom = int(orig_h * 0.55)
    hz_band   = road_mask_camera[hz_top:hz_bottom, :]
    strip_w   = orig_w // 5

    hz_left   = hz_band[:, :strip_w].mean() / 255.0
    hz_right  = hz_band[:, orig_w - strip_w:].mean() / 255.0
    hz_center = hz_band[:, strip_w:orig_w - strip_w].mean() / 255.0

    lower_top  = int(orig_h * 0.45)
    lower_band = road_mask_camera[lower_top:, :]
    side_w     = orig_w // 6
    lower_left  = lower_band[:, :side_w].mean() / 255.0
    lower_right = lower_band[:, orig_w - side_w:].mean() / 255.0

    mid_row     = int(orig_h * 0.50)
    bot_row     = int(orig_h * 0.85)
    mid_width   = (road_mask_camera[mid_row, :] > 0).sum()
    bot_width   = (road_mask_camera[bot_row, :] > 0).sum()
    width_ratio = bot_width / max(mid_width, 1)

    hz_thresh    = 0.05
    lower_thresh = 0.08

    has_left  = (hz_left  > hz_thresh) or (width_ratio > 1.8 and lower_left  > lower_thresh)
    has_right = (hz_right > hz_thresh) or (width_ratio > 1.8 and lower_right > lower_thresh)
    has_ahead = hz_center > hz_thresh

    at_intersection = has_left or has_right
    branches = []
    if has_ahead: branches.append('ahead')
    if has_left:  branches.append('left')
    if has_right: branches.append('right')

    print(f"Intersection: hz L:{hz_left:.3f} C:{hz_center:.3f} R:{hz_right:.3f} "
          f"width_ratio:{width_ratio:.2f} → {branches}")
    return at_intersection, branches


# ═══════════════════════════════════════════════════════════
#  INTERSECTION MODE
#  Splits the recovered road footprint using ANGULAR SECTORS
#  from the intersection center. Each sector = one direction.
#  This follows actual road shape instead of rectangles.
# ═══════════════════════════════════════════════════════════
def handle_intersection(img, road_mask_8u, branches, orig_h, orig_w):

    # Recover full road footprint including side roads LFD missed
    full_road = recover_intersection_road(road_mask_8u, orig_h, orig_w)

    # Find intersection center
    cx, cy = find_intersection_center(road_mask_8u, orig_h, orig_w)
    print(f"Intersection center: ({cx}, {cy})")

    # Build per-pixel angle map relative to intersection center
    yy, xx  = np.mgrid[0:orig_h, 0:orig_w]
    dx      = (xx - cx).astype(np.float32)
    dy      = (yy - cy).astype(np.float32)
    # atan2 returns angle in radians: 0=right, π/2=down, -π/2=up, ±π=left
    angle_map = np.arctan2(dy, dx)  # shape (H, W)

    # Define angular boundaries for each direction.
    # Camera coords: y increases downward.
    #   EGO   = below center  → angle near +π/2  → [+50°, +130°]
    #   AHEAD = above center  → angle near -π/2  → [-130°, -50°]
    #   LEFT  = left of center → angle near ±π   → split at ±180°
    #   RIGHT = right of center → angle near 0   → [-50°, +50°]
    import math
    def deg(x): return x * math.pi / 180.0

    # Sector definitions: list of (direction, [angle_intervals])
    # Each interval is (min_angle_rad, max_angle_rad)
    sector_defs = {
        'ego':   [(deg(55),  deg(125))],
        'ahead': [(deg(-125), deg(-55))],
        'right': [(deg(-55),  deg(55))],
        'left':  [(deg(125),  deg(180)), (deg(-180), deg(-125))],
    }

    branch_colors_bgr = {
        'ego':   [  0, 140, 255],   # orange  — where we are
        'ahead': [ 60, 210,  60],   # green   — straight ahead
        'left':  [200,  60, 200],   # purple  — left turn
        'right': [220, 160,  40],   # blue    — right turn
    }

    colored    = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
    found_dirs = {}

    # Paint in order: ego last so it's always visible
    paint_order = ['ahead', 'left', 'right', 'ego']
    for direction in paint_order:
        if direction not in sector_defs:
            continue
        # Only paint branches we detected (plus ego always)
        if direction != 'ego' and direction not in branches:
            continue

        intervals = sector_defs[direction]
        sector_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
        for (a_min, a_max) in intervals:
            if a_min <= a_max:
                sector_mask |= ((angle_map >= a_min) & (angle_map <= a_max)).astype(np.uint8)
            else:
                # Wraps around ±π
                sector_mask |= ((angle_map >= a_min) | (angle_map <= a_max)).astype(np.uint8)

        # Intersect with actual road pixels
        region = cv2.bitwise_and(sector_mask * 255, full_road)
        px_count = (region > 0).sum()
        if px_count < 300:
            print(f"  {direction}: only {px_count} px, skipping")
            continue

        colored[region > 0] = branch_colors_bgr[direction]
        found_dirs[direction] = px_count
        print(f"  {direction}: {px_count} px painted")

    blend = cv2.addWeighted(img, 0.45, colored, 0.65, 0)

    # Draw arrows from center indicating available turns
    arrow_len   = max(orig_h // 9, 40)
    arrow_thick = max(3, orig_h // 100)
    arr_colors  = {
        'ahead': (60,  210, 60),
        'left':  (200, 60, 200),
        'right': (220, 160, 40),
    }
    if 'left' in found_dirs:
        cv2.arrowedLine(blend, (cx, cy), (cx - arrow_len, cy),
                        arr_colors['left'], arrow_thick, tipLength=0.35)
    if 'right' in found_dirs:
        cv2.arrowedLine(blend, (cx, cy), (cx + arrow_len, cy),
                        arr_colors['right'], arrow_thick, tipLength=0.35)
    if 'ahead' in found_dirs:
        cv2.arrowedLine(blend, (cx, cy), (cx, cy - arrow_len),
                        arr_colors['ahead'], arrow_thick, tipLength=0.35)

    # Mark intersection center
    cv2.circle(blend, (cx, cy), 6, (255, 255, 255), -1)

    return blend, found_dirs


# ═══════════════════════════════════════════════════════════
#  STRAIGHT ROAD helpers (unchanged from working version)
# ═══════════════════════════════════════════════════════════
def detect_intersection_rows(bright_clean, road_bev, orig_h, orig_w):
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
    multi_dash      = len(runs_on) >= 2
    has_gaps        = len(meaningful_gaps) >= 1
    if multi_dash and has_gaps:
        t, c = 'dashed', True
    elif fill_ratio > 0.75:
        t, c = 'solid', False
    elif has_gaps and fill_ratio < 0.65:
        t, c = 'dashed', True
    elif fill_ratio > 0.55 and not has_gaps:
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


def handle_straight_road(img, img_bev, road_bev, orig_h, orig_w, Minv):
    gray_bev      = cv2.cvtColor(img_bev, cv2.COLOR_BGR2GRAY)
    road_gray_bev = cv2.bitwise_and(gray_bev, gray_bev, mask=road_bev)
    valid_px      = road_gray_bev[road_bev > 0]
    if len(valid_px) == 0:
        return cv2.addWeighted(img, 0.65, np.zeros_like(img), 0.35, 0), 0, []

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

    intersection_rows, _ = detect_intersection_rows(bright_clean, road_bev, orig_h, orig_w)

    sample_row = road_bev[int(orig_h * 0.85), :]
    road_cols  = np.where(sample_row > 0)[0]
    if len(road_cols) < 4:
        road_cols = np.where(road_bev.sum(axis=0) > orig_h * 0.1)[0]
    if len(road_cols) < 4:
        return cv2.addWeighted(img, 0.65, np.zeros_like(img), 0.35, 0), 0, []

    road_left    = int(road_cols[0])
    road_right   = int(road_cols[-1])
    road_width   = road_right - road_left
    road_center  = (road_left + road_right) // 2
    edge_margin  = road_width // 8
    inner_margin = road_width // 10
    inner_left   = road_left  + inner_margin
    inner_right  = road_right - inner_margin

    clean_hist = bright_clean.copy()
    clean_hist[intersection_rows > 0, :] = 0
    col_hist = clean_hist[orig_h // 2:, :].sum(axis=0).astype(np.float32)
    smooth   = ndimage.gaussian_filter1d(col_hist, sigma=max(orig_w // 60, 6))
    gated    = smooth.copy()
    gated[:road_left  + edge_margin] = 0
    gated[road_right - edge_margin:] = 0

    min_lane_w = road_width // 6
    peak_h     = gated.max() * 0.15 if gated.max() > 0 else 1
    peak_pr    = gated.max() * 0.08 if gated.max() > 0 else 1
    peaks, _   = find_peaks(gated, height=peak_h, distance=min_lane_w, prominence=peak_pr)
    print(f"BEV marking peaks: {peaks.tolist()}")

    wall_half     = max(8, road_width // 22)
    all_walls     = []
    marking_infos = []
    combined_wall = np.zeros_like(road_bev)

    for peak in peaks:
        poly, _ = fit_marking_polyline(
            bright_clean, int(peak), road_bev, orig_h, road_width, intersection_rows
        )
        wall_mask = np.zeros_like(road_bev)
        for row in range(orig_h):
            if intersection_rows[row]:
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
        print(f"  Marking @ col {col_bottom}: {info}")

    if len(all_walls) == 0:
        seed_xs = [road_center]
    else:
        bxs = [inner_left] + sorted([w[2] for w in all_walls]) + [inner_right]
        seed_xs = [(bxs[i] + bxs[i+1]) // 2 for i in range(len(bxs) - 1)]

    road_carved = cv2.morphologyEx(
        cv2.bitwise_and(road_bev, cv2.bitwise_not(combined_wall)),
        cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    )
    lane_colors = [
        [180,  80, 255], [ 60, 210,  60], [255, 160,   0],
        [ 60, 180, 255], [255,  60, 120], [  0, 200, 180],
    ]
    seed_y      = int(orig_h * 0.88)
    colored_bev = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
    remaining   = road_carved.copy()
    lane_count  = 0

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
        colored_bev[lane_px] = lane_colors[idx % len(lane_colors)]
        remaining[lane_px]   = 0
        lane_count += 1

    colored_view = cv2.warpPerspective(colored_bev, Minv, (orig_w, orig_h),
                                       flags=cv2.INTER_NEAREST)
    final_blend  = cv2.addWeighted(img, 0.60, colored_view, 0.60, 0)
    return final_blend, lane_count, marking_infos


# ═══════════════════════════════════════════════════════════
#  MAIN NODE
# ═══════════════════════════════════════════════════════════
class MockBirdsEyeSlice(Node):
    def __init__(self, name='mock_birds_eye_slice'):
        super().__init__(name)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model  = LFD_RoadSeg(scale_factor=2)

        weights_path = os.path.join(current_dir, 'model_epoch_150.pth')
        checkpoint   = torch.load(weights_path, map_location=self.device)
        state_dict   = checkpoint.get('model_state_dict', checkpoint)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device).half().eval()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.3598, 0.3653, 0.3662],
                                 std =[0.2573, 0.2663, 0.2756]),
        ])

        self.test_image_path = os.path.join(current_dir, "test_input12.png")
        self.run_inference()

    def run_inference(self):
        img = cv2.imread(self.test_image_path)
        if img is None:
            print("Image not found.")
            return

        orig_h, orig_w = img.shape[:2]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # ── 1. AI ROAD MASK ─────────────────────────────────────
        input_resized = cv2.resize(img_rgb, (624, 192))
        tensor = self.transform(input_resized).unsqueeze(0).to(self.device).half()

        with torch.no_grad():
            output = self.model({"img": tensor})
            if isinstance(output, (list, tuple)):
                output = output[0]

        probs        = F.softmax(output, dim=1)
        mask_float   = F.interpolate(probs, size=(orig_h, orig_w),
                                     mode='bilinear', align_corners=False
                                     ).cpu().float().numpy()
        road_mask_8u = (mask_float[0, 1] > 0.5).astype(np.uint8) * 255

        # ── 2. INTERSECTION CHECK ────────────────────────────────
        at_intersection, branches = check_intersection(road_mask_8u, orig_h, orig_w)

        # ── 3A. INTERSECTION MODE ────────────────────────────────
        if at_intersection:
            print(f"INTERSECTION MODE — branches: {branches}")
            final_blend, found_dirs = handle_intersection(
                img, road_mask_8u, branches, orig_h, orig_w
            )
            hud_mode      = "intersection"
            lane_count    = 0
            marking_infos = []
            chip_data     = [(d, None) for d in found_dirs]

        # ── 3B. STRAIGHT ROAD MODE ───────────────────────────────
        else:
            print("STRAIGHT ROAD MODE")
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

            road_bev_raw = cv2.warpPerspective(road_mask_8u, M, (orig_w, orig_h),
                                               flags=cv2.INTER_NEAREST)
            img_bev  = cv2.warpPerspective(img, M, (orig_w, orig_h))
            road_bev = fill_road_convex(road_bev_raw)

            final_blend, lane_count, marking_infos = handle_straight_road(
                img, img_bev, road_bev, orig_h, orig_w, Minv
            )
            hud_mode  = f"{lane_count} lane{'s' if lane_count != 1 else ''}"
            chip_data = [(f"{info['color']} {info['type']}", info['crossable'])
                         for info in marking_infos]

        # ── 4. HUD ───────────────────────────────────────────────
        bar_h   = 38
        overlay = final_blend.copy()
        cv2.rectangle(overlay, (0, orig_h - bar_h), (orig_w, orig_h), (0, 0, 0), -1)
        final_blend = cv2.addWeighted(overlay, 0.60, final_blend, 0.40, 0)

        cv2.putText(final_blend, hud_mode,
                    (12, orig_h - bar_h + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 1, cv2.LINE_AA)

        dir_colors_bgr = {
            'ahead': (60, 160, 60),
            'left':  (180, 40, 160),
            'right': (40, 130, 200),
            'ego':   (0,  110, 220),
        }
        x_cursor = 200
        for label, crossable in chip_data:
            key = label.strip()
            if key in dir_colors_bgr:
                chip_bgr = dir_colors_bgr[key]
            elif crossable is False:
                chip_bgr = (0, 0, 180)
            elif crossable is True and 'yellow' in label:
                chip_bgr = (0, 140, 220)
            elif crossable is True:
                chip_bgr = (0, 160, 60)
            else:
                chip_bgr = (80, 80, 80)

            chip_text   = f" {label} "
            (tw, th), _ = cv2.getTextSize(chip_text, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)
            cv2.rectangle(final_blend,
                          (x_cursor, orig_h - bar_h + 5),
                          (x_cursor + tw + 4, orig_h - 5),
                          chip_bgr, -1)
            cv2.putText(final_blend, chip_text,
                        (x_cursor + 2, orig_h - bar_h + 5 + th + 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)
            x_cursor += tw + 10

        output_path = os.path.join(current_dir, "output_lanes.jpg")
        cv2.imwrite(output_path, final_blend)
        print(f"Done. mode={'int' if at_intersection else 'road'} lanes={lane_count}")


def main(args=None):
    rclpy.init(args=args)
    node = MockBirdsEyeSlice()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()