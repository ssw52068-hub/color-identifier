# ===== Color Identifier App - Final Hybrid Version (No OpenCV) =====
# ML-based color recognition for colorblind assistance
# Algorithms: K-Nearest Neighbors (KNN) + HSV Color Space
# Supports: Web (JSON/Base64) AND Mobile App (Multipart File Upload)

import os
import io
import base64
import colorsys
import numpy as np
from math import sqrt
from PIL import Image, ImageDraw, ImageFont, ImageOps

# [머신러닝 라이브러리]
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier

# [Flask 웹 서버 설정]
from flask import Flask, request, jsonify, render_template

print("=" * 70)
print("🚀 Color Identifier App Starting... (Final Hybrid, No OpenCV)")
print("=" * 70)
print()

try:
    from flask_cors import CORS
    cors_available = True
except ImportError:
    cors_available = False
    print("⚠️  flask-cors not available (optional)")

app = Flask(__name__)

# [설정] 대용량 이미지 허용 (16MB 제한) - 고화질 폰 사진 대응
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

if cors_available:
    CORS(app)

# ==========================================
# [1] 데이터베이스 및 ML 모델 학습
# ==========================================
print("[1/4] Loading color database...")

color_names = []
rgb_data = []

# --- 50가지 색상 데이터 정의 + 추가 색상 ---

# RED family
color_names.extend(['red', 'dark_red', 'light_red', 'crimson'])
rgb_data.extend([[255, 0, 0], [139, 0, 0], [255, 102, 102], [220, 20, 60]])

# ORANGE family
color_names.extend(['orange', 'dark_orange', 'light_orange', 'coral'])
rgb_data.extend([[255, 165, 0], [255, 140, 0], [255, 200, 124], [255, 127, 80]])

# YELLOW family
color_names.extend(['yellow', 'dark_yellow', 'light_yellow', 'gold', 'khaki'])
rgb_data.extend([[255, 255, 0], [204, 204, 0], [255, 255, 153], [255, 215, 0], [240, 230, 140]])

# GREEN family
color_names.extend(['green', 'dark_green', 'light_green', 'lime', 'olive', 'forest_green'])
rgb_data.extend([[0, 255, 0], [0, 100, 0], [144, 238, 144], [50, 205, 50], [128, 128, 0], [34, 139, 34]])

# CYAN family
color_names.extend(['cyan', 'dark_cyan', 'light_cyan', 'turquoise'])
rgb_data.extend([[0, 255, 255], [0, 139, 139], [224, 255, 255], [64, 224, 208]])

# BLUE family
color_names.extend(['blue', 'dark_blue', 'light_blue', 'navy', 'sky_blue',
                    'royal_blue', 'dodger_blue'])
rgb_data.extend([[0, 0, 255], [0, 0, 139], [173, 216, 230], [0, 0, 128], [135, 206, 235],
                 [65, 105, 225], [30, 144, 255]])

# PURPLE family
color_names.extend(['purple', 'dark_purple', 'light_purple', 'violet', 'magenta',
                    'lavender'])
rgb_data.extend([[128, 0, 128], [75, 0, 130], [216, 191, 216], [238, 130, 238], [255, 0, 255],
                 [230, 230, 250]])

# PINK family
color_names.extend(['pink', 'hot_pink', 'light_pink', 'deep_pink'])
rgb_data.extend([[255, 192, 203], [255, 105, 180], [255, 182, 193], [255, 20, 147]])

# BROWN family
color_names.extend(['brown', 'dark_brown', 'light_brown', 'tan', 'beige'])
rgb_data.extend([[165, 42, 42], [101, 67, 33], [222, 184, 135], [210, 180, 140], [245, 245, 220]])

# GRAY family
color_names.extend(['gray', 'dark_gray', 'light_gray', 'silver', 'white', 'black',
                    'charcoal'])
rgb_data.extend([[128, 128, 128], [64, 64, 64], [192, 192, 192], [192, 192, 192],
                 [255, 255, 255], [0, 0, 0], [54, 69, 79]])

# SPECIAL colors
color_names.extend(['ivory', 'cream', 'teal', 'indigo'])
rgb_data.extend([[255, 255, 240], [255, 253, 208],
                 [0, 128, 128], [75, 0, 130]])

rgb_data = np.array(rgb_data)
print(f"✅ Loaded {len(color_names)} colors")

# --- ML 학습 준비 ---
def rgb_to_hsv_features(rgb_arr):
    """RGB를 HSV 특징으로 변환 (정확도 향상용)"""
    hsv_list = []
    for rgb in rgb_arr:
        r, g, b = rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        hsv_list.append([h * 2.0, s * 1.0, v * 1.0])
    return np.array(hsv_list)

print("[2/4] Generating synthetic training data...")
def generate_synthetic_data():
    X_train = []
    y_train = []
    for idx, base_color in enumerate(rgb_data):
        # 원본 색 하나
        X_train.append(base_color)
        y_train.append(idx)
        # 노이즈 추가된 색 여러 개
        for _ in range(30):
            noise = np.random.normal(0, 8, 3)
            noisy_color = np.clip((base_color + noise) *
                                  np.random.uniform(0.9, 1.1), 0, 255)
            X_train.append(noisy_color)
            y_train.append(idx)
    return np.array(X_train), np.array(y_train)

X_train_rgb, y_train = generate_synthetic_data()
X_train_hsv = rgb_to_hsv_features(X_train_rgb)

print("[3/4] Training KNN classifier...")
knn_model = KNeighborsClassifier(n_neighbors=3, weights='distance')
knn_model.fit(X_train_hsv, y_train)
train_accuracy = knn_model.score(X_train_hsv, y_train) * 100
print(f"✓ Model trained successfully! Accuracy: {train_accuracy:.2f}%")

# ==========================================
# [2] 헬퍼 함수 (이미지 처리 등)
# ==========================================
print("[4/4] Setting up helper functions...")


def process_image_data(image_file=None, base64_string=None):
    """
    파일(앱) 또는 Base64 문자열(웹)을 받아서 이미지 배열로 변환
    """
    try:
        img = None
        if image_file:
            # 앱 인벤터 등에서 파일 업로드로 보낸 경우
            img = Image.open(image_file.stream)
        elif base64_string:
            # 웹 브라우저에서 JSON Base64로 보낸 경우
            if ',' in base64_string:
                base64_string = base64_string.split(',')[1]
            img_bytes = base64.b64decode(base64_string)
            img = Image.open(io.BytesIO(img_bytes))

        if img is None:
            raise ValueError("이미지 데이터가 없습니다.")

        # [중요] 폰 사진 회전 보정 (EXIF 태그 처리)
        img = ImageOps.exif_transpose(img)

        img = img.convert('RGB')
        img = img.resize((200, 200))  # 분석용 리사이징
        return np.array(img)
    except Exception as e:
        print(f"❌ 이미지 처리 중 오류: {e}")
        return None


def get_achromatic_color(r, g, b):
    """무채색(검/흰/회) 판별 로직"""
    r_norm, g_norm, b_norm = r / 255.0, g / 255.0, b / 255.0
    h, s, v = colorsys.rgb_to_hsv(r_norm, g_norm, b_norm)
    if v < 0.15:
        return 'black', 95.0
    if s < 0.10 and v > 0.85:
        return 'white', 95.0
    if s < 0.15:
        if v > 0.6:
            return 'light_gray', 90.0
        if v > 0.4:
            return 'gray', 90.0
        return 'dark_gray', 90.0
    return None, None


def predict_color_knn(rgb_value):
    """색상 예측 함수 (무채색 우선 판별 + KNN)"""
    r, g, b = rgb_value
    achromatic_name, achro_conf = get_achromatic_color(r, g, b)
    if achromatic_name:
        return achromatic_name, achro_conf, [(achromatic_name, achro_conf)]

    hsv_input = rgb_to_hsv_features([rgb_value])
    pred_idx = knn_model.predict(hsv_input)[0]
    probabilities = knn_model.predict_proba(hsv_input)[0]
    confidence = probabilities[pred_idx] * 100

    top_3_indices = np.argsort(probabilities)[-3:][::-1]
    top_3 = [(color_names[i], probabilities[i] * 100) for i in top_3_indices]
    return color_names[pred_idx], confidence, top_3


def create_segmented_image(img_array, labels_orig, predictions, rgb_data, color_names):
    """
    결과 이미지 생성 (Base64)

    1) KMeans 라벨맵을 400x400으로 리사이즈
    2) 각 클러스터 영역을 대표 색으로 채움
    3) 얇은 윤곽선(1px) 그림
    4) 각 클러스터에 대해
       - 먼저 영역 '안'에 라벨을 넣을 수 있는지 시도
       - 안 되면 이미지 좌/우측에 라벨 배치 + 화살표로 연결
    """
    display_w, display_h = 400, 400

    # --- 라벨맵 리사이즈 (Nearest) ---
    h_orig, w_orig = labels_orig.shape
    labels_img = Image.fromarray(labels_orig.astype(np.uint8), mode='L')
    labels_disp_img = labels_img.resize((display_w, display_h), resample=Image.NEAREST)
    labels = np.array(labels_disp_img)  # (H, W)

    # --- 세그멘테이션 색 채우기 ---
    segmented = np.zeros((display_h, display_w, 3), dtype=np.uint8)

    for cid, color_idx in predictions.items():
        mask = (labels == cid)
        if np.any(mask):
            segmented[mask] = rgb_data[color_idx]

    # --- 윤곽선(1px) 계산 ---
    H, W = labels.shape
    outline_mask = np.zeros_like(labels, dtype=bool)

    for y in range(H - 1):
        for x in range(W - 1):
            c = labels[y, x]
            if labels[y, x + 1] != c or labels[y + 1, x] != c:
                outline_mask[y, x] = True

    segmented[outline_mask] = [0, 0, 0]

    # --- PIL 객체로 변환 ---
    segmented_pil = Image.fromarray(segmented)
    draw = ImageDraw.Draw(segmented_pil)

    try:
        font = ImageFont.truetype("arial.ttf", 14)  # 조금 더 작은 라벨
    except:
        font = ImageFont.load_default()

    placed_boxes = []  # 이미 배치된 라벨 박스들 (겹침 방지)
    margin_side = 10
    vertical_step_factor = 1.3
    center_x = display_w / 2

    def boxes_overlap(b1, b2):
        x1, y1, x2, y2 = b1
        a1, b1_, a2, b2_ = b2
        return not (x2 < a1 or a2 < x1 or y2 < b1_ or b2_ < y1)

    # --- 각 클러스터별 라벨 + 화살표 ---
    unique_cids = sorted(list(predictions.keys()))

    for cid in unique_cids:
        mask = (labels == cid)
        if not np.any(mask):
            continue

        pixels = np.argwhere(mask)  # (N, 2) [y, x]
        mean_y = float(np.mean(pixels[:, 0]))
        mean_x = float(np.mean(pixels[:, 1]))

        color_idx = predictions[cid]
        name = color_names[color_idx].replace('_', ' ')

        # 라벨 텍스트 크기
        tbx = draw.textbbox((0, 0), name, font=font)
        tw, th = tbx[2] - tbx[0], tbx[3] - tbx[1]

        # ------------------------------------------------
        # 1) 먼저 "영역 안"에 라벨을 넣을 수 있는지 시도
        # ------------------------------------------------
        placed_inside = False

        tx_in = int(mean_x - tw / 2)
        ty_in = int(mean_y - th / 2)

        x1_in = tx_in - 4
        y1_in = ty_in - 2
        x2_in = tx_in + tw + 4
        y2_in = ty_in + th + 2

        if 0 <= x1_in and 0 <= y1_in and x2_in < display_w and y2_in < display_h:
            # 박스 영역이 해당 클러스터 안에 얼마나 포함되는지 체크
            submask = mask[max(0, y1_in):min(H, y2_in),
                           max(0, x1_in):min(W, x2_in)]
            inside_ratio = submask.mean() if submask.size > 0 else 0.0

            if inside_ratio >= 0.7 and not any(
                boxes_overlap((x1_in, y1_in, x2_in, y2_in), b) for b in placed_boxes
            ):
                # → 영역 안에 충분히 들어가고, 다른 라벨과도 안 겹치면
                draw.rounded_rectangle(
                    [x1_in, y1_in, x2_in, y2_in],
                    radius=4,
                    fill=(255, 255, 255),
                    outline=(0, 0, 0),
                    width=1,
                )
                draw.text((tx_in, ty_in), name, fill=(0, 0, 0), font=font)
                placed_boxes.append((x1_in, y1_in, x2_in, y2_in))
                placed_inside = True

        if placed_inside:
            # 안에 잘 들어갔으면 화살표 없이 다음 색으로
            continue

        # ------------------------------------------------
        # 2) 안 되면, 이미지 좌/우측에 라벨 놓고 화살표로 연결
        # ------------------------------------------------
        if mean_x < center_x:
            side = "right"
            tx = display_w - tw - margin_side
        else:
            side = "left"
            tx = margin_side

        ty = int(mean_y) - th // 2
        step = int(th * vertical_step_factor)
        tries = 0
        max_tries = 30

        while tries < max_tries:
            x1 = tx - 4
            y1 = ty - 2
            x2 = tx + tw + 4
            y2 = ty + th + 2
            box = (x1, y1, x2, y2)

            if not any(boxes_overlap(box, b) for b in placed_boxes):
                break

            ty += step
            if ty + th + 2 > display_h:
                ty = max(0, int(mean_y) - th // 2 - step * (tries + 1))
            tries += 1

        x1 = tx - 4
        y1 = ty - 2
        x2 = tx + tw + 4
        y2 = ty + th + 2
        placed_boxes.append((x1, y1, x2, y2))

        # --- 화살표 끝점: 해당 클러스터 "경계" 쪽 픽셀 선택 ---
        band = max(2, H // 50)
        band_mask = np.abs(pixels[:, 0] - mean_y) <= band
        band_pixels = pixels[band_mask]
        if band_pixels.size == 0:
            band_pixels = pixels

        if side == "right":
            idx = np.argmax(band_pixels[:, 1])
        else:
            idx = np.argmin(band_pixels[:, 1])

        end_y, end_x = band_pixels[idx]
        end_x = int(end_x)
        end_y = int(end_y)

        # --- 라벨 그리기 (윤곽선 위에) ---
        draw.rounded_rectangle(
            [x1, y1, x2, y2],
            radius=4,
            fill=(255, 255, 255),
            outline=(0, 0, 0),
            width=1,
        )
        draw.text((tx, ty), name, fill=(0, 0, 0), font=font)

        # --- 화살표 선 + 화살표 머리 ---
        label_cx = (x1 + x2) // 2
        label_cy = (y1 + y2) // 2

        start = (label_cx, label_cy)
        end = (end_x, end_y)

        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = float(np.hypot(dx, dy))

        if length > 0:
            # 선
            draw.line([start, end], fill=(0, 0, 0), width=1)

            # 화살표 머리
            ux = dx / length
            uy = dy / length
            head_len = 6
            head_width = 4

            base_x = end[0] - ux * head_len
            base_y = end[1] - uy * head_len

            left_x = base_x + (-uy) * head_width / 2
            left_y = base_y + (ux) * head_width / 2
            right_x = base_x - (-uy) * head_width / 2
            right_y = base_y - (ux) * head_width / 2

            arrow_head = [
                (end[0], end[1]),
                (int(left_x), int(left_y)),
                (int(right_x), int(right_y)),
            ]
            draw.polygon(arrow_head, fill=(0, 0, 0))

    # --- PNG → Base64 ---
    buffered = io.BytesIO()
    segmented_pil.save(buffered, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode()


# ==========================================
# [3] API 라우트 (핵심 로직)
# ==========================================
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model': 'KNN+HSV',
        'colors': len(color_names)
    })


@app.route('/api/analyze', methods=['POST'])
def analyze_image():
    # [디버깅 로그] 어떤 요청이 왔는지 터미널에 출력
    print("\n" + "=" * 30)
    print("📨 [서버] 분석 요청 도착!")

    try:
        img_array = None

        # [Case A] 앱 인벤터 파일 업로드 처리
        if request.files:
            print("   ✅ 타입: 파일 업로드 (App Inventor)")
            # 첫 번째 파일 가져오기
            file_key = next(iter(request.files))
            file = request.files[file_key]
            print(f"   - 파일명: {file.filename}")
            img_array = process_image_data(image_file=file)

        # [Case B] 웹 브라우저 JSON 처리
        elif request.is_json:
            print("   ✅ 타입: JSON 데이터 (Web)")
            data = request.get_json()
            if 'image' in data:
                img_array = process_image_data(base64_string=data['image'])

        # 이미지 없으면 에러
        if img_array is None:
            print("   ❌ 오류: 이미지 데이터를 찾을 수 없음 (빈 요청)")
            return jsonify({'success': False, 'error': 'No image provided. Check App Inventor path.'}), 400

        # ML 분석 시작
        print("   🔍 ML 분석 중...")
        h, w, c = img_array.shape
        pixels = img_array.reshape(-1, 3)

        k = 6
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels_flat = kmeans.fit_predict(pixels)
        cluster_centers = kmeans.cluster_centers_
        labels = labels_flat.reshape(h, w)

        cluster_predictions = {}
        results = []

        for i in range(k):
            center_rgb = cluster_centers[i].astype(int)
            color_name, confidence, top_3 = predict_color_knn(center_rgb)
            cluster_predictions[i] = color_names.index(color_name)
            coverage = (labels_flat == i).sum() / len(labels_flat) * 100

            # 결과 리스트 (텍스트용) - coverage 제한은 계속 2% 유지
            if coverage > 2.0:
                results.append({
                    'rgb': center_rgb.tolist(),
                    'color_name': color_name,
                    'confidence': round(confidence, 1),
                    'coverage': round(coverage, 1),
                    'top_3': [(name, round(conf, 1)) for name, conf in top_3]
                })

        results.sort(key=lambda x: x['coverage'], reverse=True)
        if results:
            print(f"   🎉 분석 완료! 대표 색상: {results[0]['color_name']} 등 {len(results)}개 색상")
        else:
            print("   ⚠️ 분석은 되었지만 coverage 조건을 만족하는 색상이 없습니다.")

        # 이미지용 세그멘테이션은 모든 클러스터에 라벨/화살표 표시
        segmented_base64 = create_segmented_image(
            img_array, labels, cluster_predictions, rgb_data, color_names
        )

        return jsonify({
            'success': True,
            'results': results,
            'segmented_image': segmented_base64,
            'model_info': {
                'algorithm': 'KNN + HSV Hybrid',
                'training_samples': len(X_train_hsv),
                'accuracy': round(train_accuracy, 2),
                'colors_detected': len(results)
            }
        })

    except Exception as e:
        print(f"   🔥 서버 에러 발생: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    # 외부 접속 허용 (0.0.0.0)
    app.run(host='0.0.0.0', port=5000, debug=True)
