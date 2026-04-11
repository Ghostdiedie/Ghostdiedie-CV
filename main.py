import asyncio
import threading
import queue
import json
import os
import requests
import websockets
from dotenv import load_dotenv
from detect import run_detection

load_dotenv()

SERVER_URL = os.getenv("SERVER_URL", "http://localhost:8000")
WS_URL     = os.getenv("WS_URL",     "ws://localhost:8000")
API_KEY    = os.getenv("API_KEY")

HEADERS = {"X-API-Key": API_KEY, "Content-Type": "application/json"}

# OpenCV 스레드 → WebSocket 전달용 큐
action_queue: queue.Queue = queue.Queue()


# ── HTTP: 방 생성 / 참가 ──────────────────────────────────────────────────────

def create_room(player_name: str, character_id: int) -> str:
    res = requests.post(f"{SERVER_URL}/api/rooms", headers=HEADERS, json={
        "player_name": player_name,
        "character_id": character_id,
    })
    return res.json()["room_code"]


def join_room(room_code: str, player_name: str, character_id: int):
    requests.post(f"{SERVER_URL}/api/rooms/{room_code}/join", headers=HEADERS, json={
        "player_name": player_name,
        "character_id": character_id,
    })


# ── WebSocket 루프 ────────────────────────────────────────────────────────────

async def ws_loop(room_code: str):
    uri = f"{WS_URL}/ws/{room_code}"
    async with websockets.connect(uri) as ws:
        print(f"WebSocket 연결됨: {uri}")

        async def sender():
            """action_queue에 쌓인 동작을 서버로 전송"""
            while True:
                try:
                    action = action_queue.get_nowait()
                    await ws.send(json.dumps({
                        "type": "player_action",
                        "action": action,   # "right punch" / "left punch" 등
                        "damage": 15,
                    }))
                    print(f"송신: {action}")
                except queue.Empty:
                    pass
                await asyncio.sleep(0.016)  # ~60fps

        async def receiver():
            """서버에서 오는 메시지 수신 (상대방 동작 / 게임 시작 등)"""
            async for message in ws:
                msg = json.loads(message)
                msg_type = msg.get("type")

                if msg_type == "game_start":
                    print(f"\n게임 시작! 내 포지션: P{msg['player_index'] + 1}")
                    print(f"  나:     {msg['me']['name']} (캐릭터 {msg['me']['character_id']})")
                    print(f"  상대:   {msg['opponent']['name']} (캐릭터 {msg['opponent']['character_id']})\n")

                elif msg_type == "player_action":
                    print(f"상대 동작: {msg.get('action')}  데미지: {msg.get('damage')}")

                elif msg_type == "opponent_left":
                    print("상대방이 게임을 나갔습니다.")

                else:
                    print(f"수신: {msg}")

        await asyncio.gather(sender(), receiver())


# ── 메인 ─────────────────────────────────────────────────────────────────────

def main():
    player_name  = input("플레이어 이름: ").strip()
    character_id = int(input("캐릭터 ID (1~5): ").strip())
    mode         = input("방 만들기(c) / 참가(j): ").strip().lower()

    if mode == "c":
        room_code = create_room(player_name, character_id)
        print(f"방 코드: {room_code}  (상대방에게 알려주세요)")
    else:
        room_code = input("방 코드: ").strip().upper()
        join_room(room_code, player_name, character_id)
        print(f"방 참가 완료: {room_code}")

    # OpenCV 감지 → 별도 스레드
    cv_thread = threading.Thread(target=run_detection, args=(action_queue,), daemon=True)
    cv_thread.start()

    # WebSocket 루프 시작
    asyncio.run(ws_loop(room_code))


if __name__ == "__main__":
    main()
