# Ghostdiedie CV

Ghostdiedie 격투 게임의 로컬 Python CV 브리지입니다.
웹캠으로 실제 동작을 감지하고, `localhost` WebSocket으로 프론트엔드에 액션과 CV 프레임을 전달합니다.

---

## 요구 사항

- Python 3.8 이상
- 웹캠
- Ghostdiedie 프론트엔드 실행 중

---

## 설치

### 1. 저장소 클론

```bash
git clone <repo-url>
cd Ghostdiedie-CV
```

### 2. 가상환경 생성 및 활성화

```bash
python3 -m venv .venv
source .venv/bin/activate        # Mac / Linux
.venv\Scripts\activate           # Windows
```

### 3. 의존성 설치

```bash
pip install -r requirements.txt
```

### 4. 환경 변수 설정

```bash
cp .env.example .env
```

`.env` 파일을 열어서 로컬 브리지 포트 설정:

```
CV_BRIDGE_HOST=127.0.0.1
CV_BRIDGE_PORT=8765
CV_SHOW_WINDOW=0
```

---

## 단독 테스트

프론트 연결 없이 카메라와 동작 감지만 테스트할 수 있습니다.

```bash
source .venv/bin/activate
python -c "import queue; from detect import run_detection; run_detection(queue.Queue(), show_window=True)"
```

- 카메라 창이 뜨고 스켈레톤이 오버레이됩니다
- 주먹을 앞으로 뻗으면 터미널에 `오른손 펀치!` / `왼손 펀치!` 출력
- 카메라 창에서 `q` 키를 누르면 종료

---

## 브리지 실행

```bash
source .venv/bin/activate
python main.py
```

실행되면 브리지가 `ws://127.0.0.1:8765` 에서 대기합니다.
프론트엔드는 이 브리지에 연결해서:

1. `cv_frame` 으로 CV 영상 표시
2. `cv_event` 로 이동/펀치/킥/점프 입력 처리

게임 시작 전에 이 브리지를 먼저 실행해야 합니다.

---

## 조작법

| 동작 | 방법 |
|---|---|
| 오른손 펀치 | 오른 손목을 앞으로 뻗기 |
| 왼손 펀치 | 왼 손목을 앞으로 뻗기 |
| 점프 | 몸을 위로 튕기기 |
| 킥 | 무릎을 들었다 내리기 |
| 이동 | 좌우 무릎 들기 |
| 카메라 종료 | `CV_SHOW_WINDOW=1` 일 때 창에서 `q` 키 |

---

## 감지 민감도 조절

너무 민감하거나 잘 안 잡히면 `detect.py` 상단의 값을 수정하세요:

```python
Z_THRESHOLD = 0.15   # 높일수록 더 많이 뻗어야 펀치 인식 (권장: 0.20 ~ 0.30)
```
