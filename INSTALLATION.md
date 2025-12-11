# 🚀 설치 및 실행 가이드

## Step 1: 파일 다운로드

모든 파일을 다운로드하여 다음과 같은 구조로 배치하세요:

```
color_identifier_app/
├── app.py
├── requirements.txt
├── README.md
├── INSTALLATION.md (이 파일)
├── static/
│   ├── style.css
│   └── script.js
└── templates/
    └── index.html
```

## Step 2: Python 환경 설정

### 방법 1: 직접 설치 (추천)

```bash
# 터미널/명령 프롬프트에서 프로젝트 폴더로 이동
cd color_identifier_app

# 패키지 설치
pip install -r requirements.txt
```

### 방법 2: 가상환경 사용 (선택)

```bash
# 가상환경 생성
python -m venv venv

# 가상환경 활성화
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# 패키지 설치
pip install -r requirements.txt
```

## Step 3: Flask 서버 실행

```bash
python app.py
```

다음과 같은 메시지가 나타나면 성공:

```
======================================================================
Color Identifier App Starting...
======================================================================

[1/4] Loading color database...
✓ Loaded 50 colors

[2/4] Generating synthetic training data...
✓ Generated 1050 training samples

[3/4] Training Random Forest classifier...
✓ Model trained successfully!
  - Training accuracy: 98.00%

[4/4] Setting up helper functions...
✓ Helper functions ready

======================================================================
🚀 Starting Flask Development Server
======================================================================

📍 Access the app at: http://localhost:5000
```

## Step 4: 웹 브라우저에서 접속

브라우저를 열고 다음 주소로 이동:

```
http://localhost:5000
```

## ✅ 테스트

1. **파일 업로드**: 이미지 파일을 선택하거나 드래그앤드롭
2. **카메라 사용**: "Use Camera" 버튼 클릭 (카메라 권한 허용 필요)
3. **결과 확인**: 색상 분석 결과 표시
4. **다운로드**: 분석 리포트 다운로드

## 🔧 문제 해결

### 문제 1: 패키지 설치 오류

```bash
# pip 업그레이드
python -m pip install --upgrade pip

# 다시 시도
pip install -r requirements.txt
```

### 문제 2: Flask 서버가 시작되지 않음

```bash
# Python 버전 확인 (3.7 이상 필요)
python --version

# 포트 변경 (5000번 포트가 사용 중인 경우)
# app.py 마지막 줄을 다음과 같이 수정:
app.run(host='0.0.0.0', port=8000, debug=True)
```

### 문제 3: 카메라가 작동하지 않음

- HTTPS 필요 (localhost는 괜찮음)
- 브라우저 카메라 권한 허용
- 대신 파일 업로드 사용

### 문제 4: 이미지 분석 실패

- 이미지 크기 확인 (10MB 이하)
- 지원 형식: JPG, PNG, JPEG
- 서버 콘솔에서 에러 메시지 확인

## 📱 모바일 테스트

같은 Wi-Fi 네트워크에서:

1. 컴퓨터의 IP 주소 확인:
   ```bash
   # Windows
   ipconfig
   # Mac/Linux
   ifconfig
   ```

2. 모바일 브라우저에서 접속:
   ```
   http://YOUR_IP_ADDRESS:5000
   ```

## 🌐 온라인 배포 (다음 단계)

배포 가이드는 README.md 참조

---

**Team: ACDT 31조**  
**문의사항이 있으면 README.md 참조**
