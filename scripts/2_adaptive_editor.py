import os
import json
import math
import urllib.request
import subprocess
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
MODEL_PATH = "work/blaze_face_short_range.tflite"
HAAR_URL = "https://raw.githubusercontent.com/opencv/opencv/4.x/data/haarcascades/haarcascade_frontalface_default.xml"
HAAR_PATH = "work/haarcascade_frontalface_default.xml"

def ensure_face_models():
    if not os.path.exists(MODEL_PATH):
        try:
            print("Downloading MediaPipe BlazeFace task model...", flush=True)
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        except Exception as e:
            print(f"⚠️ BlazeFace download skipped: {e}", flush=True)

    if not os.path.exists(HAAR_PATH):
        try:
            urllib.request.urlretrieve(HAAR_URL, HAAR_PATH)
        except Exception:
            pass

def format_ass_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int(round((seconds - int(seconds)) * 100))
    if cs == 100:
        s += 1
        cs = 0
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"

def generate_ass_subtitles(words: list, clip_start: float, clip_end: float, title: str, theme: str, ass_path: str):
    clip_words = [
        {
            "word": w["word"].upper(),
            "start": max(0.0, w["start"] - clip_start),
            "end": max(0.05, w["end"] - clip_start),
        }
        for w in words
        if w["end"] > clip_start and w["start"] < clip_end and w["word"].strip()
    ]

    if theme == "hype_gaming":
        primary_color = "&H0000FFFF"   # Yellow highlight
        secondary_color = "&H00FFFFFF" # White base
        outline_color = "&H00000000"
    elif theme == "editorial_podcast":
        primary_color = "&H0000E5FF"   # Gold/Amber highlight
        secondary_color = "&H00FFFFFF"
        outline_color = "&H00101010"
    else:
        primary_color = "&H0055FF55"   # Vibrant green highlight
        secondary_color = "&H00FFFFFF"
        outline_color = "&H00000000"

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: HookBanner,Montserrat ExtraBold,64,&H00000000,&H00000000,&H0000FFFF,&H0000FFFF,-1,0,0,0,100,100,0,0,3,18,0,8,60,60,180,1
Style: KineticCaption,Montserrat ExtraBold,86,{primary_color},{secondary_color},{outline_color},&H80000000,-1,0,0,0,100,100,0,0,1,6,3,2,60,60,340,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    duration = max(1.0, clip_end - clip_start)
    events = [
        f"Dialogue: 1,0:00:00.00,{format_ass_time(duration)},HookBanner,,0,0,0,,{{\\fad(200,200)}}{title.upper()}"
    ]

    chunk_size = 3
    for i in range(0, len(clip_words), chunk_size):
        chunk = clip_words[i : i + chunk_size]
        for active_idx, active_w in enumerate(chunk):
            seg_start = active_w["start"]
            seg_end = chunk[active_idx + 1]["start"] if active_idx + 1 < len(chunk) else active_w["end"]
            if seg_end <= seg_start:
                seg_end = seg_start + 0.12

            styled_tokens = []
            for j, token in enumerate(chunk):
                clean_txt = token["word"].replace("{", "").replace("}", "")
                if j == active_idx:
                    styled_tokens.append(f"{{\\c{primary_color}\\fscx112\\fscy112}}{clean_txt}{{\\r}}")
                else:
                    styled_tokens.append(f"{{\\c{secondary_color}}}{clean_txt}")

            line_text = " ".join(styled_tokens)
            events.append(
                f"Dialogue: 0,{format_ass_time(seg_start)},{format_ass_time(seg_end)},KineticCaption,,0,0,0,,{line_text}"
            )

    with open(ass_path, "w", encoding="utf-8") as f:
        f.write(header + "\n".join(events) + "\n")

def detect_faces_tasks(detector, frame_bgr):
    h, w, _ = frame_bgr.shape
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)
    centers = []
    for det in result.detections:
        bbox = det.bounding_box
        cx = (bbox.origin_x + bbox.width / 2.0) / float(w)
        cy = (bbox.origin_y + bbox.height / 2.0) / float(h)
        centers.append((max(0.0, min(1.0, cx)), max(0.0, min(1.0, cy))))
    return centers

def render_adaptive_video(raw_cut: str, rendered_vid: str, mode: str, clip_start: float, peaks: list):
    cap = cv2.VideoCapture(raw_cut)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_w, out_h = 1080, 1920
    temp_silent = rendered_vid.replace(".mp4", "_silent.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(temp_silent, fourcc, fps, (out_w, out_h))

    ensure_face_models()
    detector = None
    haar_cascade = None

    # 1. Try MediaPipe Tasks CPU delegate
    if os.path.exists(MODEL_PATH):
        try:
            base_options = mp_python.BaseOptions(
                model_asset_path=MODEL_PATH,
                delegate=mp_python.BaseOptions.Delegate.CPU
            )
            options = mp_vision.FaceDetectorOptions(base_options=base_options, min_detection_confidence=0.5)
            detector = mp_vision.FaceDetector.create_from_options(options)
            print("Face tracker: MediaPipe BlazeFace (CPU) active.", flush=True)
        except Exception as e:
            print(f"⚠️ MediaPipe CPU init failed ({e}), trying Haar Cascade...", flush=True)

    # 2. Fallback to OpenCV Haar Cascade (verified non-empty)
    if detector is None:
        for xml_candidate in [HAAR_PATH, getattr(cv2.data, "haarcascades", "") + "haarcascade_frontalface_default.xml"]:
            if xml_candidate and os.path.exists(xml_candidate):
                clf = cv2.CascadeClassifier(xml_candidate)
                if not clf.empty():
                    haar_cascade = clf
                    print(f"Face tracker: OpenCV Haar Cascade loaded from {xml_candidate}.", flush=True)
                    break

    if detector is None and haar_cascade is None:
        print("⚠️ No face detector available; using smooth center framing.", flush=True)

    rel_peaks = [p - clip_start for p in peaks if p >= clip_start]
    smooth_cx = 0.5
    left_cx, right_cx = 0.28, 0.72
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t_sec = frame_idx / fps

        zoom = 1.0
        for pk in rel_peaks:
            dt = t_sec - pk
            if 0.0 <= dt <= 0.45:
                zoom = 1.0 + 0.14 * math.sin((dt / 0.45) * math.pi)
                break

        if frame_idx % 5 == 0 and mode in ("podcast_single", "podcast_dual"):
            centers = []
            if detector is not None:
                centers = detect_faces_tasks(detector, frame)
            elif haar_cascade is not None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                rects = haar_cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5, minSize=(40, 40))
                centers = [((x + w / 2.0) / src_w, (y + h / 2.0) / src_h) for (x, y, w, h) in rects]

            if centers:
                if mode == "podcast_single":
                    target_cx = centers[0][0]
                    smooth_cx = 0.85 * smooth_cx + 0.15 * target_cx
                elif mode == "podcast_dual":
                    left_faces = [c[0] for c in centers if c[0] < 0.5]
                    right_faces = [c[0] for c in centers if c[0] >= 0.5]
                    if left_faces:
                        left_cx = 0.85 * left_cx + 0.15 * left_faces[0]
                    if right_faces:
                        right_cx = 0.85 * right_cx + 0.15 * right_faces[0]

        if mode == "podcast_dual":
            half_h = out_h // 2
            crop_w = int(src_h * (out_w / half_h) / zoom)
            crop_h = int(src_h / zoom)
            crop_w = min(src_w, max(64, crop_w))
            crop_h = min(src_h, max(64, crop_h))
            y1 = max(0, (src_h - crop_h) // 2)

            x1_top = int(np.clip(left_cx * src_w - crop_w / 2, 0, max(0, src_w - crop_w)))
            x1_bot = int(np.clip(right_cx * src_w - crop_w / 2, 0, max(0, src_w - crop_w)))

            top_crop = cv2.resize(frame[y1 : y1 + crop_h, x1_top : x1_top + crop_w], (out_w, half_h), interpolation=cv2.INTER_LANCZOS4)
            bot_crop = cv2.resize(frame[y1 : y1 + crop_h, x1_bot : x1_bot + crop_w], (out_w, half_h), interpolation=cv2.INTER_LANCZOS4)
            canvas = np.vstack([top_crop, bot_crop])

        elif mode == "gaming_action":
            bg = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
            bg = cv2.GaussianBlur(bg, (51, 51), 0)
            fg_w = out_w
            fg_h = int(out_w * (src_h / src_w))
            fg = cv2.resize(frame, (fg_w, fg_h), interpolation=cv2.INTER_LANCZOS4)
            y_offset = (out_h - fg_h) // 2
            canvas = bg
            canvas[y_offset : y_offset + fg_h, 0:out_w] = fg

        else:
            crop_h = int(src_h / zoom)
            crop_w = int(crop_h * (out_w / out_h))
            crop_w = min(src_w, max(64, crop_w))
            crop_h = min(src_h, max(64, crop_h))
            x1 = int(np.clip(smooth_cx * src_w - crop_w / 2, 0, max(0, src_w - crop_w)))
            y1 = max(0, (src_h - crop_h) // 2)
            cropped = frame[y1 : y1 + crop_h, x1 : x1 + crop_w]
            canvas = cv2.resize(cropped, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)

        writer.write(canvas)
        frame_idx += 1

    cap.release()
    writer.release()
    if detector is not None:
        detector.close()

    subprocess.run([
        "ffmpeg", "-nostdin", "-y",
        "-i", temp_silent, "-i", raw_cut,
        "-map", "0:v:0", "-map", "1:a:0?",
        "-c:v", "libx264", "-crf", "19", "-preset", "veryfast",
        "-c:a", "aac", "-b:a", "192k",
        rendered_vid
    ], check=True)

    if os.path.exists(temp_silent):
        os.remove(temp_silent)

def main():
    os.makedirs("final_reels", exist_ok=True)
    with open("work/blueprint.json", "r", encoding="utf-8") as f:
        blueprint = json.load(f)
    with open("work/words.json", "r", encoding="utf-8") as f:
        words = json.load(f)

    master_video = "work/master.mp4"
    clips = blueprint.get("clips", [])

    for idx, clip in enumerate(clips, start=1):
        start = float(clip["start_time"])
        end = float(clip["end_time"])
        duration = max(5.0, end - start)
        mode = clip.get("content_mode", "podcast_single")
        theme = clip.get("caption_theme", "editorial_podcast")
        title = clip.get("title", f"CLIP {idx}")
        peaks = clip.get("peak_moments", [])

        print(f"=== Building Reel {idx} [{mode.upper()}]: {title} ({start}s - {end}s) ===", flush=True)

        raw_cut = f"work/cut_{idx}.mp4"
        rendered_vid = f"work/tracked_{idx}.mp4"
        ass_path = f"work/subs_{idx}.ass"
        final_out = f"final_reels/reel_{idx}_{mode}.mp4"

        subprocess.run([
            "ffmpeg", "-nostdin", "-y",
            "-ss", str(start), "-i", master_video, "-t", str(duration),
            "-c:v", "libx264", "-crf", "17", "-preset", "veryfast",
            "-c:a", "aac", "-b:a", "192k",
            raw_cut
        ], check=True)

        render_adaptive_video(raw_cut, rendered_vid, mode, start, peaks)
        generate_ass_subtitles(words, start, end, title, theme, ass_path)

        subprocess.run([
            "ffmpeg", "-nostdin", "-y",
            "-i", rendered_vid,
            "-vf", f"ass={ass_path}:fontsdir=fonts",
            "-af", "loudnorm=I=-14:TP=-1.5:LRA=11",
            "-c:v", "libx264", "-crf", "18", "-preset", "fast",
            "-c:a", "aac", "-b:a", "192k",
            final_out
        ], check=True)

        print(f"✅ Finished {final_out}", flush=True)

if __name__ == "__main__":
    main()
