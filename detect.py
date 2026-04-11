import cv2
import mediapipe as mp
import queue
import time
import base64
import platform

from actions import PunchDetector, WalkDetector, JumpDetector, KickDetector

MIN_DETECTION_CONF = 0.5
MIN_TRACKING_CONF  = 0.5
CAMERA_INDEX       = 0
WINDOW_TITLE       = "Ghostdiedie - Motion Tracking"
FRAME_JPEG_QUALITY = 70
MAX_READ_FAILURES  = 45
CAMERA_WARMUP_SECS = 2.5

mp_pose    = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


def _get_body_box(lm, w, h):
    """어깨 + 골반으로 몸통 박스 계산"""
    rs = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    ls = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
    rh = lm[mp_pose.PoseLandmark.RIGHT_HIP]
    lh = lm[mp_pose.PoseLandmark.LEFT_HIP]

    shoulder_y = int(min(rs.y, ls.y) * h)
    hip_y      = int(max(rh.y, lh.y) * h)

    x1 = int(min(rs.x, ls.x, rh.x, lh.x) * w)
    x2 = int(max(rs.x, ls.x, rh.x, lh.x) * w)
    y1 = shoulder_y
    y2 = shoulder_y + int((hip_y - shoulder_y) * 0.55)
    cx = (x1 + x2) // 2

    return x1, y1, x2, y2, cx


def _draw_body_box(image, x1, y1, x2, y2, cx):
    overlay = image.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (220, 220, 220), -1)
    cv2.addWeighted(overlay, 0.2, image, 0.8, 0, image)
    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), 2)
    cv2.line(image, (cx, y1), (cx, y2), (255, 255, 255), 2)


def _push_latest(q: queue.Queue | None, item):
    if q is None:
        return
    while True:
        try:
            q.put_nowait(item)
            return
        except queue.Full:
            try:
                q.get_nowait()
            except queue.Empty:
                return

def _encode_frame(image) -> str | None:
    ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), FRAME_JPEG_QUALITY])
    if not ok:
        return None
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"

def _open_camera():
    backends = [None]
    if platform.system() == "Darwin" and hasattr(cv2, "CAP_AVFOUNDATION"):
        backends = [cv2.CAP_AVFOUNDATION, None]

    for backend in backends:
        cap = cv2.VideoCapture(CAMERA_INDEX) if backend is None else cv2.VideoCapture(CAMERA_INDEX, backend)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return cap, backend
        cap.release()

    return None, None

def run_detection(action_queue: queue.Queue, frame_queue: queue.Queue | None = None, status_queue: queue.Queue | None = None, show_window: bool = False, stop_event: "threading.Event | None" = None):
    import threading as _threading
    if stop_event is None:
        stop_event = _threading.Event()

    pose = mp_pose.Pose(
        model_complexity=2,
        min_detection_confidence=MIN_DETECTION_CONF,
        min_tracking_confidence=MIN_TRACKING_CONF,
    )
    cap, backend = _open_camera()
    if cap is None:
        _push_latest(status_queue, "camera_error")
        print("카메라를 열 수 없습니다.")
        return

    punch = PunchDetector()
    walk  = WalkDetector()
    jump  = JumpDetector()
    kick  = KickDetector()

    print("카메라를 켭니다!")
    print(f"카메라 백엔드: {backend if backend is not None else 'default'}")
    _push_latest(status_queue, "ready")
    read_failures = 0
    started_at = time.time()

    try:
        while cap.isOpened() and not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                read_failures += 1
                if read_failures <= 5 or read_failures % 10 == 0:
                    print(f"카메라 프레임 읽기 실패 ({read_failures}/{MAX_READ_FAILURES})")
                if time.time() - started_at < CAMERA_WARMUP_SECS:
                    time.sleep(0.05)
                    continue
                if read_failures < MAX_READ_FAILURES:
                    time.sleep(0.03)
                    continue
                _push_latest(status_queue, "camera_error")
                print("카메라 스트림이 중단되었습니다.")
                break
            read_failures = 0
            frame = cv2.flip(frame, 1)

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            h, w = image.shape[:2]

            if results.pose_landmarks:
                lm  = results.pose_landmarks.landmark
                now = time.time()

                mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=4, circle_radius=6),
                    mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=3),
                )

                x1, y1, x2, y2, cx = _get_body_box(lm, w, h)
                _draw_body_box(image, x1, y1, x2, y2, cx)
                punch.draw(image, lm, w, h)

                # ── 무릎 visibility 디버그 ──────────────────────────────
                for label, idx in [("RK", mp_pose.PoseLandmark.LEFT_KNEE),
                                    ("LK", mp_pose.PoseLandmark.RIGHT_KNEE)]:
                    kl = lm[idx]
                    kx, ky = int(kl.x * w), int(kl.y * h)
                    color = (0, 255, 0) if kl.visibility > 0.4 else (0, 0, 255)
                    cv2.putText(image, f"{label}:{kl.visibility:.2f}", (kx, ky),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # 각 감지기 실행 (여러 액션 동시 가능)
                for action in filter(None, [
                    punch.detect(lm, w, h, now, x1, x2, y1, y2, cx),
                    kick.detect(lm, w, h, now),
                    jump.detect(lm, w, h, now),
                    walk.detect(lm, w, h, now),
                ]):
                    action_queue.put(action)

            encoded = _encode_frame(image)
            if encoded:
                _push_latest(frame_queue, encoded)

            if show_window:
                cv2.imshow(WINDOW_TITLE, image)
                if cv2.waitKey(10) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        if show_window:
            cv2.destroyAllWindows()
        print("카메라를 닫았습니다.")
        _push_latest(status_queue, "stopped")
