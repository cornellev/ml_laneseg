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

current_dir = os.path.dirname(os.path.abspath(__file__))
LFD_REPO_PATH = "/ros2_ws/src/LFD_RoadSeg"

if os.path.exists(LFD_REPO_PATH):
    sys.path.append(LFD_REPO_PATH)

try:
    from models._LFDRoadSeg import LFD_RoadSeg
    print("Success: LFD_RoadSeg imported.")
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

class MockBirdsEyeSlice(Node):
    def __init__(self, name='mock_birds_eye_slice'):
        super().__init__(name)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cfg = {'scale_factor': 2}
        self.model = LFD_RoadSeg(**self.cfg)

        weights_path = os.path.join(current_dir, 'model_epoch_150.pth')
        checkpoint = torch.load(weights_path, map_location=self.device)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        self.model.load_state_dict(state_dict)
        self.model.to(self.device).half().eval()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.3598, 0.3653, 0.3662], std=[0.2573, 0.2663, 0.2756])
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

        # --- 1. GET AI ROAD MASK ---
        input_resized = cv2.resize(img_rgb, (624, 192))
        tensor = self.transform(input_resized).unsqueeze(0).to(self.device).half()

        with torch.no_grad():
            output = self.model({"img": tensor})
            if isinstance(output, (list, tuple)):
                output = output[0]

        output_probs = F.softmax(output, dim=1)
        mask_float = F.interpolate(
            output_probs, size=(orig_h, orig_w),
            mode='bilinear', align_corners=False
        ).cpu().float().numpy()
        road_mask_8u = (mask_float[0, 1, :, :] > 0.5).astype(np.uint8) * 255

        # --- 2. BIRD'S EYE VIEW WARP ---
        src_pts = np.float32([
            [orig_w * 0.43, orig_h * 0.62],
            [orig_w * 0.57, orig_h * 0.62],
            [orig_w * 0.80, orig_h * 0.95],
            [orig_w * 0.20, orig_h * 0.95],
        ])
        dst_pts = np.float32([
            [orig_w * 0.25, 0],      [orig_w * 0.75, 0],
            [orig_w * 0.75, orig_h], [orig_w * 0.25, orig_h],
        ])
        M    = cv2.getPerspectiveTransform(src_pts, dst_pts)
        Minv = cv2.getPerspectiveTransform(dst_pts, src_pts)

        road_bev = cv2.warpPerspective(road_mask_8u, M, (orig_w, orig_h), flags=cv2.INTER_NEAREST)
        img_bev  = cv2.warpPerspective(img, M, (orig_w, orig_h))

        # --- 3. DETECT LANE MARKING BLOBS IN BEV ---
        gray_bev      = cv2.cvtColor(img_bev, cv2.COLOR_BGR2GRAY)
        road_gray_bev = cv2.bitwise_and(gray_bev, gray_bev, mask=road_bev)

        valid_px = road_gray_bev[road_bev > 0]
        if len(valid_px) == 0:
            print("No road pixels.")
            return

        marking_thresh = np.percentile(valid_px, 88)
        _, bright_bev  = cv2.threshold(road_gray_bev, marking_thresh, 255, cv2.THRESH_BINARY)

        # Keep only tall+narrow blobs (lane dashes in BEV are vertical)
        bright_clean = np.zeros_like(bright_bev)
        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bright_bev, connectivity=8)
        dash_centroids = []
        for i in range(1, n_labels):
            bw   = stats[i, cv2.CC_STAT_WIDTH]
            bh   = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]
            if bh > bw * 1.2 and area > 60 and bh < orig_h * 0.7:
                bright_clean[labels == i] = 255
                dash_centroids.append((int(centroids[i][0]), int(centroids[i][1])))

        # --- 4. FIND ROAD BOUNDS & GATE TO CENTER THIRD ---
        sample_row = road_bev[int(orig_h * 0.85), :]
        road_cols  = np.where(sample_row > 0)[0]
        if len(road_cols) < 4:
            road_cols = np.where(road_bev.sum(axis=0) > orig_h * 0.1)[0]
        if len(road_cols) < 4:
            print("Road mask too small.")
            return

        road_left  = int(road_cols[0])
        road_right = int(road_cols[-1])
        road_width = road_right - road_left
        road_center = (road_left + road_right) // 2

        search_l = road_left  + road_width // 4
        search_r = road_right - road_width // 4

        # Keep only dash centroids in the center third
        center_dashes = [(x, y) for (x, y) in dash_centroids if search_l < x < search_r]

        # --- 5. FIT A ROBUST LANE MARKING POLYLINE IN BEV ---
        # Strategy: for each horizontal band, find the median x of marking pixels
        # This gives a per-row x position that follows the actual dash geometry
        band_height = max(orig_h // 20, 15)
        marking_col = road_center  # fallback

        # Collect per-band x positions from bright_clean pixels in center zone
        lane_xs_by_row = {}  # row_band -> list of x positions
        ys, xs = np.where(bright_clean > 0)
        for y, x in zip(ys, xs):
            if search_l < x < search_r:
                band = (y // band_height) * band_height + band_height // 2
                lane_xs_by_row.setdefault(band, []).append(x)

        # Build a smooth per-row marking x using RANSAC-style median per band
        band_points = []  # (y, x) pairs of reliable band medians
        for band_y, xs_list in sorted(lane_xs_by_row.items()):
            if len(xs_list) >= 3:
                band_points.append((band_y, int(np.median(xs_list))))

        print(f"Band points for lane fit: {band_points}")

        if len(band_points) >= 2:
            # Fit a line through band points: x = a*y + b
            band_ys = np.array([p[0] for p in band_points], dtype=np.float32)
            band_xs_arr = np.array([p[1] for p in band_points], dtype=np.float32)
            # Weighted fit: weight bottom rows more (closer to camera = more reliable)
            weights = (band_ys / orig_h) ** 2
            coeffs  = np.polyfit(band_ys, band_xs_arr, deg=1, w=weights)
            poly    = np.poly1d(coeffs)

            # Build the actual wall as a per-row filled strip following the fitted line
            marking_wall = np.zeros_like(road_bev)
            wall_half    = max(8, road_width // 22)
            for row in range(orig_h):
                cx = int(np.clip(poly(row), search_l, search_r))
                wl = max(0, cx - wall_half)
                wr = min(orig_w, cx + wall_half)
                if road_bev[row, cx] > 0:
                    marking_wall[row, wl:wr] = 255

            # Use bottom-row marking column for seed placement
            marking_col = int(np.clip(poly(int(orig_h * 0.85)), search_l, search_r))

        else:
            # No dashes detected — fall back to geometric center wall
            print("No dash bands found, using road center.")
            marking_wall = np.zeros_like(road_bev)
            wall_half    = max(8, road_width // 22)
            for row in range(orig_h):
                if road_bev[row, road_center] > 0:
                    marking_wall[row, road_center - wall_half:road_center + wall_half] = 255

        # --- 6. CARVE WALL & FLOOD FILL ---
        road_carved = cv2.bitwise_and(road_bev, cv2.bitwise_not(marking_wall))
        open_k      = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        road_carved = cv2.morphologyEx(road_carved, cv2.MORPH_OPEN, open_k)

        # Seeds placed at bottom, equidistant between road edges and marking
        seed_y       = int(orig_h * 0.88)
        left_seed_x  = (road_left  + marking_col) // 2
        right_seed_x = (marking_col + road_right)  // 2

        colors = [
            [255,  60,  60],
            [ 60, 210,  60],
            [ 60, 180, 255],
            [255, 160,   0],
        ]

        def find_valid_seed(mask, cx, cy, radius=80):
            for r in range(0, radius, 3):
                for dx in range(-r, r + 1, 3):
                    nx = int(np.clip(cx + dx, 0, mask.shape[1] - 1))
                    if mask[cy, nx] > 0:
                        return nx, cy
            return None, None

        colored_bev = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
        remaining   = road_carved.copy()
        lane_count  = 0

        for idx, (sx, sy) in enumerate([(left_seed_x, seed_y), (right_seed_x, seed_y)]):
            sx, sy = find_valid_seed(remaining, sx, sy)
            if sx is None:
                print(f"Seed {idx}: no valid pixel, skipping.")
                continue

            flood_mask = np.zeros((orig_h + 2, orig_w + 2), np.uint8)
            fill_img   = remaining.copy()
            cv2.floodFill(fill_img, flood_mask, (sx, sy), 128)

            lane_px = fill_img == 128
            if lane_px.sum() < 500:
                print(f"Seed {idx}: {lane_px.sum()} px too small, skipping.")
                continue

            colored_bev[lane_px] = colors[idx % len(colors)]
            remaining[lane_px]   = 0
            lane_count += 1

        # --- 7. DEBUG BEV ---
        debug_bev = cv2.addWeighted(img_bev, 0.45, colored_bev, 0.75, 0)
        wall_vis  = np.zeros_like(img_bev)
        wall_vis[marking_wall > 0] = [0, 255, 255]
        debug_bev = cv2.addWeighted(debug_bev, 1.0, wall_vis, 0.6, 0)
        # Draw the fitted line
        if len(band_points) >= 2:
            for row in range(0, orig_h, 5):
                cx = int(np.clip(poly(row), 0, orig_w - 1))
                cv2.circle(debug_bev, (cx, row), 1, (0, 255, 0), -1)
        cv2.circle(debug_bev, (left_seed_x,  seed_y), 8, (255, 255, 0), -1)
        cv2.circle(debug_bev, (right_seed_x, seed_y), 8, (255, 255, 0), -1)
        cv2.imwrite(os.path.join(current_dir, "debug_bev.jpg"), debug_bev)

        # --- 8. UNWARP & BLEND ---
        colored_view = cv2.warpPerspective(colored_bev, Minv, (orig_w, orig_h), flags=cv2.INTER_NEAREST)
        final_blend  = cv2.addWeighted(img, 0.65, colored_view, 0.55, 0)
        cv2.imwrite(os.path.join(current_dir, "output_lanes.jpg"), final_blend)
        print(f"Found {lane_count} lanes.")
def main(args=None):
    rclpy.init(args=args)
    node = MockBirdsEyeSlice()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()