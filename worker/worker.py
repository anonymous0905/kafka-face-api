import os
import json
import asyncio
import logging
import random
import tempfile
import httpx
import cv2
import numpy as np
from aiokafka import AIOKafkaConsumer
from deepface import DeepFace
from geopy.distance import geodesic
from scipy.signal import butter, filtfilt
from facenet_pytorch import MTCNN
from PIL import Image
import torch
import mediapipe as mp

from insightface.app import FaceAnalysis
import cv2
from numpy.linalg import norm

# ---------------- CONFIG ----------------
KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
GROUP_ID = "job-workers"
TOPIC = "jobs"

SUPABASE_URL = "https://agrdrscvweokrrcsabsz.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImFncmRyc2N2d2Vva3JyY3NhYnN6Iiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc0OTY5MzU2NCwiZXhwIjoyMDY1MjY5NTY0fQ.OaquJNKMROqZb5qRJufyZhAGgGY6UlQbXcawpZEsP7E"
LAB_COORDS = (12.93562593031098, 77.53476582540279)
CUSTOM_THRESHOLD = 0.5

# ---------------- LOGGING ----------------
log = logging.getLogger("worker")
logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")

# ---------------- UTILS ----------------
def load_video_with_mtcnn_faces(video_path, max_frames=300, min_confidence=0.95, device='cpu'):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames, rois = [], []
    valid_face_count = 0
    mtcnn = MTCNN(keep_all=False, device=device)

    while cap.isOpened() and len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        boxes, probs = mtcnn.detect(pil_img)
        if boxes is not None and probs[0] is not None and probs[0] > min_confidence:
            x1, y1, x2, y2 = map(int, boxes[0])
            x1, y1 = max(0, x1), max(0, y1)
            face = rgb[y1:y2, x1:x2]
            face = cv2.resize(face, (128, 128))
            rois.append(cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
            valid_face_count += 1
        else:
            rois.append(None)

        frames.append(cv2.resize(frame, (320, 240)))

    cap.release()
    return frames, rois, fps, valid_face_count, len(frames)

def download_video_from_url(video_url: str) -> str:
    response = httpx.get(video_url)
    response.raise_for_status()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(response.content)
        return tmp.name  # Return local file path

def chrom_rppg(frames, fps):
    red, green, blue = [], [], []
    for roi in frames:
        if roi is None:
            continue
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        avg_rgb = np.mean(roi_rgb, axis=(0, 1))
        r, g, b = avg_rgb
        red.append(r)
        green.append(g)
        blue.append(b)

    if len(red) < 30:
        return None

    red, green, blue = map(np.array, (red, green, blue))
    X = 3 * red - 2 * green
    Y = 1.5 * red + green - 1.5 * blue
    X = (X - np.mean(X)) / np.std(X)
    Y = (Y - np.mean(Y)) / np.std(Y)
    S = X + Y

    b, a = butter(3, [0.7 / (fps / 2), 4.0 / (fps / 2)], btype='bandpass')
    return filtfilt(b, a, S)

def is_rppg_signal_valid(signal):
    return np.std(signal) > 0.5

def compute_ear(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)

def detect_eye_blinks(frames, ear_thresh=0.2, consec_frames=2):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)
    LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE_IDX = [263, 387, 385, 362, 380, 373]

    blink_count = 0
    frame_counter = 0

    for frame in frames:
        if frame is None:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)
        if not result.multi_face_landmarks:
            continue

        landmarks = result.multi_face_landmarks[0].landmark
        ih, iw = frame.shape[:2]

        left_eye = np.array([[landmarks[i].x * iw, landmarks[i].y * ih] for i in LEFT_EYE_IDX])
        right_eye = np.array([[landmarks[i].x * iw, landmarks[i].y * ih] for i in RIGHT_EYE_IDX])

        ear = (compute_ear(left_eye) + compute_ear(right_eye)) / 2.0
        if ear < ear_thresh:
            frame_counter += 1
        else:
            if frame_counter >= consec_frames:
                blink_count += 1
            frame_counter = 0

    face_mesh.close()
    return blink_count

def detect_spoof_via_landmarks(frames, threshold=1):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)
    prev_landmarks, movements = None, []

    for frame in frames:
        if frame is None:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)
        if not result.multi_face_landmarks:
            continue

        current = np.array([
            [lm.x * frame.shape[1], lm.y * frame.shape[0]]
            for lm in result.multi_face_landmarks[0].landmark
        ])
        if prev_landmarks is not None:
            delta = np.linalg.norm(current - prev_landmarks, axis=1)
            movements.append(np.mean(delta))
        prev_landmarks = current

    face_mesh.close()
    avg_movement = np.mean(movements) if movements else 0
    return avg_movement < threshold

def delete_supabase_video(video_url):
    try:
        path_start = video_url.find("/object/public/") + len("/object/public/")
        object_path = video_url[path_start:]
        bucket, file_path = object_path.split("/", 1)
        response = httpx.delete(
            f"{SUPABASE_URL}/storage/v1/object/{bucket}/{file_path}",
            headers={
                "Authorization": f"Bearer {SUPABASE_KEY}",
                "apikey": SUPABASE_KEY
            }
        )
        log.info(f"Deleted video: {file_path}, status={response.status_code}")
    except Exception as e:
        log.warning(f"Failed to delete video: {e}")

def download_image_as_cv2(url):
    resp = httpx.get(url)
    img_array = np.frombuffer(resp.content, np.uint8)
    return cv2.imdecode(img_array, cv2.IMREAD_COLOR)

# def run_anti_spoof_checks(video_url):
#     response = httpx.get(video_url)
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
#         tmp_file.write(response.content)
#         video_path = tmp_file.name
#
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     frames, rois, fps, valid_faces, total = load_video_with_mtcnn_faces(video_path, device=device)
#
#     reasons = []
#     if valid_faces < total * 0.4:
#         reasons.append("Insufficient valid face frames")
#
#     if detect_eye_blinks(rois) == 0:
#         reasons.append("No eye blinks detected - Possible Spoof")
#
#     if detect_spoof_via_landmarks(rois):
#         reasons.append("Static facial landmarks detected - Possible Spoof")
#     else:
#         signal = chrom_rppg(rois, fps)
#         if signal is None or not is_rppg_signal_valid(signal):
#             reasons.append("Flat pulse detected - Possible spoof")
#
#     os.remove(video_path)
#     return reasons

# ---------------- JOB HANDLER ----------------
async def handle(log_id: str):
    log.info("→ Processing log ID: %s", log_id)
    headers = {"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}"}

    try:
        # Fetch log entry
        log_resp = httpx.get(f"{SUPABASE_URL}/rest/v1/attendance_logs?id=eq.{log_id}&select=*", headers=headers)
        log_resp.raise_for_status()
        log_entry = log_resp.json()[0]

        # Fetch reference image
        student_resp = httpx.get(
            f"{SUPABASE_URL}/rest/v1/students?srn=eq.{log_entry['srn']}&select=reference_image_url",
            headers=headers
        )
        student_resp.raise_for_status()
        student = student_resp.json()[0]

        # Face verification
        # result = DeepFace.verify(
        #     img1_path=log_entry["photo_url"],
        #     img2_path=student["reference_image_url"],
        #     model_name="ArcFace",
        #     enforce_detection=True,
        #     detector_backend="retinaface"
        # )

        img1_path = log_entry["photo_url"]
        img2_path = student["reference_image_url"]
        app = FaceAnalysis(name='buffalo_l')
        app.prepare(ctx_id=-1)
        img1 = download_image_as_cv2(img1_path)
        img2 = download_image_as_cv2(img2_path)

        faces1 = app.get(img1)
        faces2 = app.get(img2)
        is_face_verified = False
        if not faces1 or not faces2:
            print("❌ Face not detected in one or both images.")
        else:
            face1 = max(faces1, key=lambda x: x.det_score)
            face2 = max(faces2, key=lambda x: x.det_score)

            emb1 = face1.embedding
            emb2 = face2.embedding

            # STEP 5: Cosine similarity and auto-thresholding
            score = np.dot(emb1, emb2) / (norm(emb1) * norm(emb2))
            threshold = 0.3  # InsightFace uses ~0.3 as typical threshold for ArcFace-R100

            print(f"\nCosine Similarity Score: {score:.4f}")
            print(f"Auto Threshold (recommended): {threshold:.2f}")

            is_face_verified = bool(score > threshold)

            if score > threshold:
                print("✅ The images are of the SAME person.")
            else:
                print("❌ The images are of DIFFERENT people.")

        #is_face_verified = result["distance"] <= CUSTOM_THRESHOLD

        # Location verification
        loc = (log_entry["location_lat"], log_entry["location_lng"])
        dist = geodesic(LAB_COORDS, loc).meters
        is_location_verified = dist <= 80

        # Load video
        video_path = download_video_from_url(log_entry["video_url"])
        frames, rois, fps, valid_faces, total = load_video_with_mtcnn_faces(video_path)
        is_valid_face_ratio = valid_faces >= total * 0.4

        # Spoof detection
        spoof_reasons = []
        spoof_passes = 0

        if detect_eye_blinks(rois) > 0:
            spoof_passes += 1
        else:
            spoof_reasons.append("No eye blinks detected - Possible Spoof")

        if not detect_spoof_via_landmarks(rois):
            spoof_passes += 1
        else:
            spoof_reasons.append("Static facial landmarks detected - Possible Spoof")

        signal = chrom_rppg(rois, fps)
        if signal is not None and is_rppg_signal_valid(signal):
            spoof_passes += 1
        else:
            spoof_reasons.append("Flat pulse detected - Possible Spoof")

        # Combine all failure reasons
        reasons = []
        if not is_face_verified:
            reasons.append("Face mismatch")
        if not is_location_verified:
            reasons.append(f"Out of range ({dist:.2f}m) — soft warning only")
        if not is_valid_face_ratio:
            reasons.append("Insufficient valid face frames")
        if spoof_passes < 2:
            reasons.extend(spoof_reasons)

        # Final verification decision
        verified = (
            is_face_verified and
            is_valid_face_ratio and
            spoof_passes >= 2
        )

        location_reason = next((r for r in reasons if "Out of range" in r), None)
        non_location_reasons = [r for r in reasons if r != location_reason]

        update = {
            "verified": verified,
            "flagged": bool(non_location_reasons),
            "reason_flagged": ", ".join(reasons) if reasons else None
        }

        # Patch result to Supabase
        patch_resp = httpx.patch(
            f"{SUPABASE_URL}/rest/v1/attendance_logs?id=eq.{log_id}",
            headers={**headers, "Content-Type": "application/json"},
            json=update
        )
        patch_resp.raise_for_status()

        # Send email if flagged
        if not verified:
            email_resp = httpx.post(
                f"{SUPABASE_URL}/functions/v1/flagging-true",
                headers={**headers, "Content-Type": "application/json"},
                json={"id": log_entry["id"]},
                timeout=30
            )
            email_resp.raise_for_status()

        # Clean up
        os.remove(video_path)
        delete_supabase_video(log_entry["video_url"])

        log.info("✓ Finished %s | verified: %s | reasons: %s", log_id, verified, reasons)
    except Exception as e:
        log.exception(f"Job failed for {log_id} — will be retried")

# ---------------- CONSUMER BOOT ----------------
job_queue = asyncio.Queue()

async def connect_to_kafka():
    while True:
        consumer = AIOKafkaConsumer(
            TOPIC,
            bootstrap_servers=KAFKA_BOOTSTRAP,
            group_id=GROUP_ID,
            enable_auto_commit=True,   # ✅ Auto commit enabled
            auto_commit_interval_ms=5000,
            max_poll_records=1,
            max_poll_interval_ms=6000000,
        )
        try:
            await consumer.start()
            log.info("Connected to Kafka at %s", KAFKA_BOOTSTRAP)
            return consumer
        except Exception as e:
            log.warning("Kafka not ready (%s). Retrying in 2s...", str(e))
            await asyncio.sleep(2)

async def consume_and_enqueue(consumer):
    while True:
        try:
            msg = await consumer.getone()
            job = json.loads(msg.value.decode())
            log_id = job["id"]
            await job_queue.put(log_id)
            log.info("✓ Auto-committed and queued log ID: %s", log_id)
        except Exception as e:
            log.exception("Failed to enqueue job — will retry")

async def process_jobs():
    while True:
        log_id = await job_queue.get()
        try:
            await handle(log_id)
        except Exception as e:
            log.exception(f"Error processing log ID {log_id}")
        finally:
            job_queue.task_done()

async def main():
    consumer = await connect_to_kafka()
    consumer_task = asyncio.create_task(consume_and_enqueue(consumer))
    processor_task = asyncio.create_task(process_jobs())
    await asyncio.gather(consumer_task, processor_task)

if __name__ == "__main__":
    asyncio.run(main())