# ===== Color Identifier App - Final Hybrid & Debug Version =====
# ML-based color recognition for colorblind assistance
# Algorithms: K-Nearest Neighbors (KNN) + HSV Color Space
# Supports: Web (JSON/Base64) AND Mobile App (Multipart File Upload)

import os
import io
import base64
import colorsys
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

# [머신러닝 라이브러리]
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier

# [Flask 웹 서버 설정]
from flask import Flask, request, jsonify, render_template

print("=" * 70)
print("🚀 Color Identifier App Starting... (Final Hybrid & Debug Version)")
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
        r, g, b = rgb[0]/255.0, rgb[1]/255.0, rgb[2]/255.0
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        hsv_list.append([h * 2.0, s * 1.0, v * 1.0]) 
    return np.array(hsv_list)

print("[2/4] Generating synthetic training data...")
def generate_synthetic_data():
    X_train = []
    y_train = []
    for idx, base_color in enumerate(rgb_data):
        X_train.append(base_color)
        y_train.append(idx)
        for _ in range(30):
            noise = np.random.normal(0, 8, 3)
            noisy_color = np.clip((base_color + noise) * np.random.uniform(0.9, 1.1), 0, 255)
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
        img = img.resize((200, 200)) # 분석용 리사이징
        return np.array(img)
    except Exception as e:
        print(f"❌ 이미지 처리 중 오류: {e}")
        return None

def get_achromatic_color(r, g, b):
    """무채색(검/흰/회) 판별 로직"""
    r_norm, g_norm, b_norm = r/255.0, g/255.0, b/255.0
    h, s, v = colorsys.rgb_to_hsv(r_norm, g_norm, b_norm)
    if v < 0.15: return 'black', 95.0
    if s < 0.10 and v > 0.85: return 'white', 95.0
    if s < 0.15:
        if v > 0.6: return 'light_gray', 90.0
        if v > 0.4: return 'gray', 90.0
        return 'dark_gray', 90.0
    return None, None

def predict_color_knn(rgb_value):
    """색상 예측 함수"""
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

def create_segmented_image(img_array, labels, predictions, rgb_data, color_names):
    """
    결과 이미지 생성 (Base64)
    - KMeans로 세그멘테이션된 결과를 색으로 칠하고
    - 각 색 클러스터의 중심 근처에 라벨을 찍되,
      이미 찍힌 라벨과 겹치면 아래로 조금씩 밀어서 겹침을 피함
    """
    h, w = labels.shape
    segmented = np.zeros((h, w, 3), dtype=np.uint8)

    # 클러스터 ID → 대표 색, 이름 매핑
    cluster_colors = {cid: rgb_data[pid] for cid, pid in predictions.items()}
    cluster_names = {cid: color_names[pid].replace('_', ' ') for cid, pid in predictions.items()}

    # 세그멘테이션 색칠
    for i in range(h):
        for j in range(w):
            segmented[i, j] = cluster_colors[labels[i, j]]

    # 표시용으로 리사이즈
    display_size = (400, 400)  # (width, height)
    segmented_pil = Image.fromarray(segmented).resize(display_size, Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(segmented_pil)

    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()

    # 원본 좌표 → 표시 이미지 좌표 스케일
    scale_x = display_size[0] / w  # 가로 방향
    scale_y = display_size[1] / h  # 세로 방향

    placed_boxes = []  # 이미 배치한 라벨 박스들 (x1, y1, x2, y2)

    def overlaps(x1, y1, x2, y2, box):
        bx1, by1, bx2, by2 = box
        return not (x2 < bx1 or bx2 < x1 or y2 < by1 or by2 < y1)

    # 각 클러스터(색)에 대해 라벨 하나씩 찍기
    for cid in cluster_names.keys():
        pixels = np.argwhere(labels == cid)
        if len(pixels) == 0:
            continue

        # 해당 색 클러스터 픽셀들의 중심(centroid)
        mean_y = np.mean(pixels[:, 0])
        mean_x = np.mean(pixels[:, 1])

        cx = int(mean_x * scale_x)
        cy = int(mean_y * scale_y)

        text = cluster_names[cid]
        tbx = draw.textbbox((0, 0), text, font=font)
        tw, th = tbx[2] - tbx[0], tbx[3] - tbx[1]

        # 기본 위치: 클러스터 중심에 맞추기
        tx = cx - tw // 2
        ty = cy - th // 2

        # 다른 라벨과 겹치면 아래로 조금씩 내리면서 자리 찾기
        max_tries = 15
        dy_step = th + 6  # 한 번 겹칠 때마다 얼마나 내릴지
        tries = 0
        while tries < max_tries:
            x1, y1 = tx - 4, ty - 4
            x2, y2 = tx + tw + 4, ty + th + 4

            if not any(overlaps(x1, y1, x2, y2, box) for box in placed_boxes):
                break  # 겹치지 않는 위치를 찾음

            # 겹치면 아래로 한 칸 이동
            ty += dy_step
            # 너무 아래로 내려가면 위쪽으로도 한 번 시도
            if ty + th > display_size[1]:
                ty = max(0, cy - th // 2 - dy_step * (tries + 1))
            tries += 1

        # 최종 박스 좌표
        x1, y1 = tx - 4, ty - 4
        x2, y2 = tx + tw + 4, ty + th + 4

        # 흰 배경 박스 + 검은 글자
        draw.rectangle([x1, y1, x2, y2], fill=(255, 255, 255, 220))
        draw.text((tx, ty), text, fill=(0, 0, 0), font=font)

        # 이 라벨 박스 기록 (다음 라벨이 겹치지 않도록)
        placed_boxes.append((x1, y1, x2, y2))

    # PNG → Base64 인코딩
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
    print("\n" + "="*30)
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
        labels = kmeans.fit_predict(pixels)
        cluster_centers = kmeans.cluster_centers_
        
        cluster_predictions = {}
        results = []
        
        for i in range(k):
            center_rgb = cluster_centers[i].astype(int)
            color_name, confidence, top_3 = predict_color_knn(center_rgb)
            cluster_predictions[i] = color_names.index(color_name)
            coverage = (labels == i).sum() / len(labels) * 100
            
            if coverage > 2.0:
                results.append({
                    'rgb': center_rgb.tolist(),
                    'color_name': color_name,
                    'confidence': round(confidence, 1),
                    'coverage': round(coverage, 1),
                    'top_3': [(name, round(conf, 1)) for name, conf in top_3]
                })
        
        results.sort(key=lambda x: x['coverage'], reverse=True)
        print(f"   🎉 분석 완료! 결과: {results[0]['color_name']} 등 {len(results)}개 색상")
        
        segmented_base64 = create_segmented_image(img_array, labels.reshape(h, w), cluster_predictions, rgb_data, color_names)
        
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