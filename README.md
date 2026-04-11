# Ghostdiedie CV

**Ghostdiedie** 격투 게임의 컴퓨터 비전 클라이언트입니다.
웹캠으로 실제 펀치 동작을 MediaPipe Pose로 감지하고, WebSocket을 통해 게임 서버로 전송합니다.

## 동작 방식

1. **OpenCV + MediaPipe Pose**로 웹캠에서 신체 랜드마크를 추출
2. 손목이 어깨보다 앞으로 나오면 (Z축 임계값 기준) 펀치로 감지
3. 감지된 액션을 WebSocket으로 게임 서버에 실시간 전송 (~60fps)

## 설치 및 실행

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

### 2. 환경 변수 설정

```bash
cp .env.example .env
```

`.env` 파일을 서버 정보에 맞게 수정:

```
SERVER_URL=http://서버주소
WS_URL=ws://서버주소
API_KEY=발급받은-API-키
```

### 3. 실행

```bash
python main.py
```

실행 시 다음을 입력합니다:
- 플레이어 이름
- 캐릭터 ID (1~5)
- 방 만들기 (`c`) 또는 참가 (`j`)

## 조작법

| 동작 | 방법 |
|---|---|
| 오른손 펀치 | 오른 손목을 오른 어깨보다 앞으로 뻗기 |
| 왼손 펀치 | 왼 손목을 왼 어깨보다 앞으로 뻗기 |
| 카메라 종료 | 카메라 창에서 `q` 키 |

## 요구 사항

- Python 3.8 이상
- 웹캠
- Ghostdiedie 게임 서버 실행 중
