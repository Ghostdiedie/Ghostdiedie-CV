# Ghostdiedie CV

Ghostdiedie 격투 게임의 컴퓨터 비전 클라이언트입니다.
웹캠으로 실제 펀치 동작을 감지하고, WebSocket을 통해 게임 서버로 실시간 전송합니다.

---

## 요구 사항

- Python 3.8 이상
- 웹캠
- Ghostdiedie 게임 서버 실행 중

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

`.env` 파일을 열어서 서버 정보 입력:

```
SERVER_URL=http://서버주소:포트
WS_URL=ws://서버주소:포트
API_KEY=발급받은-API-키
```

---

## 카메라 단독 테스트 (서버 없이)

서버 연결 없이 카메라와 동작 감지만 테스트할 수 있습니다.

```bash
source .venv/bin/activate
python -c "import queue; from detect import run_detection; run_detection(queue.Queue())"
```

- 카메라 창이 뜨고 스켈레톤이 오버레이됩니다
- 주먹을 앞으로 뻗으면 터미널에 `오른손 펀치!` / `왼손 펀치!` 출력
- 카메라 창에서 `q` 키를 누르면 종료

---

## 게임 실행

```bash
source .venv/bin/activate
python main.py
```

실행 후 순서대로 입력:

1. 플레이어 이름
2. 캐릭터 ID (1~5)
3. 방 만들기 (`c`) 또는 참가 (`j`)
   - 방 만들기: 생성된 방 코드를 상대방에게 전달
   - 방 참가: 상대방에게 받은 방 코드 입력

두 플레이어가 모두 접속하면 게임이 시작됩니다.

---

## 조작법

| 동작 | 방법 |
|---|---|
| 오른손 펀치 | 오른 손목을 오른 어깨보다 앞으로 뻗기 |
| 왼손 펀치 | 왼 손목을 왼 어깨보다 앞으로 뻗기 |
| 카메라 종료 | 카메라 창에서 `q` 키 |

---

## 감지 민감도 조절

너무 민감하거나 잘 안 잡히면 `detect.py` 상단의 값을 수정하세요:

```python
Z_THRESHOLD = 0.15   # 높일수록 더 많이 뻗어야 펀치 인식 (권장: 0.20 ~ 0.30)
```
