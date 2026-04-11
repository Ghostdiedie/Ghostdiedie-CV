import asyncio
import json
import logging
import os
import queue
import threading

from dotenv import load_dotenv
import websockets

from detect import run_detection

load_dotenv()

HOST = os.getenv("CV_BRIDGE_HOST", "127.0.0.1")
PORT = int(os.getenv("CV_BRIDGE_PORT", "8765"))
SHOW_WINDOW = os.getenv("CV_SHOW_WINDOW", "0") == "1"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [ghostdiedie-cv] %(message)s",
)
logging.getLogger("websockets").setLevel(logging.WARNING)
logging.getLogger("websockets.server").setLevel(logging.WARNING)
logging.getLogger("websockets.client").setLevel(logging.WARNING)

action_queue: queue.Queue[str] = queue.Queue()
frame_queue: queue.Queue[str] = queue.Queue(maxsize=1)
status_queue: queue.Queue[str] = queue.Queue(maxsize=4)
clients: set = set()


async def broadcast(payload: dict):
    if not clients:
        return
    message = json.dumps(payload)
    dead = []
    for ws in clients:
        try:
            await ws.send(message)
        except Exception:
            dead.append(ws)
    for ws in dead:
        clients.discard(ws)


async def bridge_handler(websocket):
    clients.add(websocket)
    await websocket.send(json.dumps({"type": "cv_status", "status": "connected"}))
    try:
        await websocket.wait_closed()
    finally:
        clients.discard(websocket)


async def broadcast_loop():
    while True:
        drained = False

        while True:
            try:
                status = status_queue.get_nowait()
            except queue.Empty:
                break
            drained = True
            logging.info("status=%s", status)
            await broadcast({"type": "cv_status", "status": status})

        while True:
            try:
                action = action_queue.get_nowait()
            except queue.Empty:
                break
            drained = True
            logging.info("action=%s", action)
            await broadcast({"type": "cv_event", "action": action})

        latest_frame = None
        while True:
            try:
                latest_frame = frame_queue.get_nowait()
            except queue.Empty:
                break
        if latest_frame is not None:
            drained = True
            await broadcast({"type": "cv_frame", "image": latest_frame})

        await asyncio.sleep(0.01 if drained else 0.03)


def start_detection():
    run_detection(action_queue, frame_queue=frame_queue, status_queue=status_queue, show_window=SHOW_WINDOW)


def start_ws_server():
    async def _run():
        async with websockets.serve(bridge_handler, HOST, PORT, max_size=8_000_000):
            logging.info("cv bridge listening ws://%s:%s", HOST, PORT)
            await broadcast_loop()
    asyncio.run(_run())


async def main():
    status_queue.put_nowait("starting")
    cv_thread = threading.Thread(target=start_detection, daemon=True)
    cv_thread.start()

    async with websockets.serve(bridge_handler, HOST, PORT, max_size=8_000_000):
        logging.info("cv bridge listening ws://%s:%s", HOST, PORT)
        await broadcast_loop()


if __name__ == "__main__":
    if SHOW_WINDOW:
        # macOS: cv2.imshow must run on the main thread
        status_queue.put_nowait("starting")
        ws_thread = threading.Thread(target=start_ws_server, daemon=True)
        ws_thread.start()
        run_detection(action_queue, frame_queue=frame_queue, status_queue=status_queue, show_window=True)
    else:
        asyncio.run(main())
