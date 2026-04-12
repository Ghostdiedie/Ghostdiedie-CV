import cv2
import mediapipe as mp
import os
import queue
import time
import base64
import platform

from actions import PunchDetector, WalkDetector, JumpDetector, KickDetector, HeadbuttDetector

MIN_DETECTION_CONF = 0.5
MIN_TRACKING_CONF  = 0.5
CAMERA_INDEX       = 0
WINDOW_TITLE       = "Ghostdiedie - Motion Tracking"
FRAME_JPEG_QUALITY = 70
MAX_READ_FAILURES  = 45
CAMERA_WARMUP_SECS = 2.5

# 사양이 낮은 경우 CV_CAMERA_WIDTH/HEIGHT 를 낮게 설정 (예: 320x240)
CAMERA_WIDTH  = int(os.getenv("CV_CAMERA_WIDTH",  "480"))
CAMERA_HEIGHT = int(os.getenv("CV_CAMERA_HEIGHT", "360"))

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
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
            cap.set(cv2.CAP_PROP_FPS, 30)
            return cap, backend
        cap.release()

    return None, None

def run_detection(action_queue: queue.Queue, frame_queue: queue.Queue | None = None, status_queue: queue.Queue | None = None, show_window: bool = False, stop_event: "threading.Event | None" = None, command_queue: queue.Queue | None = None):
    import threading as _threading
    if stop_event is None:
        stop_event = _threading.Event()

    pose = mp_pose.Pose(
        model_complexity=1,
        min_detection_confidence=MIN_DETECTION_CONF,
        min_tracking_confidence=MIN_TRACKING_CONF,
    )
    cap, backend = _open_camera()
    if cap is None:
        _push_latest(status_queue, "camera_error")
        print("카메라를 열 수 없습니다.")
        return

    punch    = PunchDetector()
    walk     = WalkDetector()
    jump     = JumpDetector()
    kick     = KickDetector()
    headbutt = HeadbuttDetector()
    last_lateral_walk = 0.0   # 좌우 이동 후 펀치 억제용
    last_punch_time   = 0.0   # 펀치 후 depth 억제용

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

            if command_queue is not None:
                while True:
                    try:
                        cmd = command_queue.get_nowait()
                        if cmd == "recalibrate":
                            walk.recalibrate()
                    except queue.Empty:
                        break

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
                walk.draw(image, lm, w, h)

                # 코 랜드마크 강조
                nose = lm[mp_pose.PoseLandmark.NOSE]
                nx, ny = int(nose.x * w), int(nose.y * h)
                cv2.circle(image, (nx, ny), 14, (0, 255, 0), -1)
                cv2.circle(image, (nx, ny), 14, (255, 255, 255), 2)

                # ── 주요 관절 이름 표시 ────────────────────────────────
                JOINT_LABELS = [
                    ("R.Shoulder", mp_pose.PoseLandmark.LEFT_SHOULDER),
                    ("L.Shoulder", mp_pose.PoseLandmark.RIGHT_SHOULDER),
                    ("R.Elbow",    mp_pose.PoseLandmark.LEFT_ELBOW),
                    ("L.Elbow",    mp_pose.PoseLandmark.RIGHT_ELBOW),
                    ("R.Wrist",    mp_pose.PoseLandmark.LEFT_WRIST),
                    ("L.Wrist",    mp_pose.PoseLandmark.RIGHT_WRIST),
                    ("R.Hip",      mp_pose.PoseLandmark.LEFT_HIP),
                    ("L.Hip",      mp_pose.PoseLandmark.RIGHT_HIP),
                    ("R.Knee",     mp_pose.PoseLandmark.LEFT_KNEE),
                    ("L.Knee",     mp_pose.PoseLandmark.RIGHT_KNEE),
                ]
                for label, idx in JOINT_LABELS:
                    jl = lm[idx]
                    jx, jy = int(jl.x * w), int(jl.y * h)
                    color = (0, 255, 0) if jl.visibility > 0.4 else (0, 0, 255)
                    cv2.putText(image, label, (jx + 8, jy - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

                # ── 우선순위: 박치기 > 점프 > 발차기 > 펀치 > 좌우이동 > 앞뒤이동
                headbutt_action = headbutt.detect(lm, w, h, now)
                jump_action     = jump.detect(lm, w, h, now)
                kick_action     = kick.detect(lm, w, h, now)

                walk_action = walk.detect(lm, w, h, now)
                if walk_action in ("walk left", "walk right"):
                    last_lateral_walk = now
                punch_suppressed = (now - last_lateral_walk) < 0.4
                punch_action = None if punch_suppressed \
                               else punch.detect(lm, w, h, now, x1, x2, y1, y2)
                if punch_action:
                    last_punch_time = now

                # 박치기/점프/발차기/펀치/좌우이동 중 하나라도 있으면 앞뒤이동 무시
                # 펀치 후 0.5초 동안도 depth 억제
                depth_suppressed  = (now - last_punch_time) < 0.5
                attack_or_lateral = headbutt_action or jump_action or kick_action \
                                    or punch_action \
                                    or (walk_action in ("walk left", "walk right"))
                depth_action = None if (attack_or_lateral or depth_suppressed) \
                               else (walk_action if walk_action in ("walk forward", "walk backward") else None)

                for action in filter(None, [
                    headbutt_action,
                    jump_action,
                    kick_action,
                    punch_action,
                    walk_action if walk_action in ("walk left", "walk right") else None,
                    depth_action,
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
