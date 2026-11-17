import os
import json
import subprocess
import cv2
import numpy as np
import mediapipe as mp
import pysubs2

def build_dynamic_captions(words, clip_start, clip_end, theme, hook_title, ass_path):
    subs = pysubs2.SSAFile()
    subs.info["PlayResX"] = 1080
    subs.info["PlayResY"] = 1920

    # Theme configurations (Colors in BGR format for ASS)
    themes = {
        "hype_gaming": {
            "primary": pysubs2.Color(255, 255, 255, 0),
            "highlight": r"\c&H00FF00&",   # Neon Green
            "fontsize": 86,
            "marginv": 340,
            "chunk_size": 2
        },
        "editorial_podcast": {
            "primary": pysubs2.Color(255, 255, 255, 0),
            "highlight": r"\c&H00D7FF&",   # Warm Gold/Yellow
            "fontsize": 76,
            "marginv": 380,
            "chunk_size": 3
        },
        "bold_explainer": {
            "primary": pysubs2.Color(255, 255, 255, 0),
            "highlight": r"\c&HFF9900&",   # Electric Cyan/Blue
            "fontsize": 80,
            "marginv": 360,
            "chunk_size": 3
        }
    }
    cfg = themes.get(theme, themes["editorial_podcast"])

    # Main Caption Style
    cap_style = pysubs2.SSAStyle()
    cap_style.fontname = "Montserrat ExtraBold"
    cap_style.fontsize = cfg["fontsize"]
    cap_style.primarycolor = cfg["primary"]
    cap_style.outlinecolor = pysubs2.Color(0, 0, 0, 0)
    cap_style.backcolor = pysubs2.Color(0, 0, 0, 140)
    cap_style.bold = True
    cap_style.outline = 7
    cap_style.shadow = 4
    cap_style.alignment = 2
    cap_style.marginv = cfg["marginv"]
    subs.styles["Default"] = cap_style

    # Top Hook Banner Style (visible for first 4.5 seconds)
    hook_style = pysubs2.SSAStyle()
    hook_style.fontname = "Montserrat ExtraBold"
    hook_style.fontsize = 58
    hook_style.primarycolor = pysubs2.Color(0, 0, 0, 0)         # Black text
    hook_style.outlinecolor = pysubs2.Color(255, 255, 255, 0)   # White pill box border
    hook_style.backcolor = pysubs2.Color(255, 255, 255, 0)
    hook_style.borderstyle = 3                                  # Opaque box background
    hook_style.outline = 18
    hook_style.bold = True
    hook_style.alignment = 8                                    # Top-center
    hook_style.marginv = 210
    subs.styles["HookBanner"] = hook_style

    # Add Hook Title Banner for the opening 4.5 seconds
    duration_ms = int((clip_end - clip_start) * 1000)
    subs.events.append(pysubs2.SSAEvent(
        start=0,
        end=min(4500, duration_ms),
        style="HookBanner",
        text=hook_title.upper()
    ))

    clip_words = [
        {
            "word": w["word"].upper(),
            "start": max(0.0, w["start"] - clip_start),
            "end": min(clip_end - clip_start, w["end"] - clip_start)
        }
        for w in words
        if w["end"] > clip_start and w["start"] < clip_end
    ]

    chunk_size = cfg["chunk_size"]
    for i in range(0, len(clip_words), chunk_size):
        chunk = clip_words[i:i + chunk_size]
        for active_idx, active_word in enumerate(chunk):
            start_ms = int(active_word["start"] * 1000)
            end_ms = int(active_word["end"] * 1000)
            if end_ms <= start_ms:
                continue

            parts = []
            for idx, item in enumerate(chunk):
                if idx == active_idx:
                    # Pop-in scale bounce (\fscx115\fscy115 -> \fscx100\fscy100) + color highlight
                    anim = r"{" + cfg["highlight"] + r"\fscx118\fscy118\t(0,90,\fscx100\fscy100)}"
                    reset = r"{\c&HFFFFFF&\fscx100\fscy100}"
                    parts.append(f"{anim}{item['word']}{reset}")
                else:
                    parts.append(item["word"])

            subs.events.append(pysubs2.SSAEvent(
                start=start_ms,
                end=end_ms,
                style="Default",
                text=" ".join(parts)
            ))

    subs.save(ass_path)

def render_adaptive_video(raw_cut, output_video, mode, clip_start, peak_moments):
    cap = cv2.VideoCapture(raw_cut)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video, fourcc, fps, (1080, 1920))

    mp_face = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

    # Convert peak timestamps relative to clip start
    rel_peaks = [p - clip_start for p in peak_moments if p >= clip_start]

    smooth_x = orig_w / 2.0
    left_speaker_x = orig_w * 0.28
    right_speaker_x = orig_w * 0.72
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_sec = frame_idx / fps
        frame_idx += 1

        # Calculate dynamic zoom punch if near a peak moment (0.45s decay)
        zoom_factor = 1.0
        for pk in rel_peaks:
            dt = current_sec - pk
            if 0.0 <= dt <= 0.45:
                zoom_factor = 1.0 + 0.12 * (1.0 - (dt / 0.45))
                break

        # Subtle periodic retention zoom on explainer/single podcast every 7 seconds
        if mode in ("explainer", "podcast_single") and zoom_factor == 1.0:
            cycle = current_sec % 7.0
            if 3.5 <= cycle <= 6.8:
                zoom_factor = 1.05

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_face.process(rgb)
        detections = results.detections if results.detections else []

        # MODE 1: GAMING ACTION (Blurred full-height background + Crisp uncropped 16:9 gameplay + Facecam overlay)
        if mode == "gaming_action":
            # Blurred ambient background
            bg_crop_w = int(orig_h * (9 / 16))
            bg_x1 = max(0, (orig_w - bg_crop_w) // 2)
            bg = cv2.resize(frame[:, bg_x1:bg_x1 + bg_crop_w], (1080, 1920))
            bg = cv2.GaussianBlur(bg, (75, 75), 0)
            bg = cv2.convertScaleAbs(bg, alpha=0.55, beta=0)

            # Apply zoom punch to gameplay center
            zh, zw = int(orig_h / zoom_factor), int(orig_w / zoom_factor)
            zy1, zx1 = (orig_h - zh) // 2, (orig_w - zw) // 2
            zoomed_game = frame[zy1:zy1 + zh, zx1:zx1 + zw]

            fg_h = int(1080 * (orig_h / orig_w))
            fg = cv2.resize(zoomed_game, (1080, fg_h), interpolation=cv2.INTER_LANCZOS4)

            y_offset = (1920 - fg_h) // 2 + 60
            bg[y_offset:y_offset + fg_h, 0:1080] = fg

            # If a facecam is detected in corners, crop & place cleanly above gameplay
            if detections:
                det = detections[0].location_data.relative_bounding_box
                fx = int((det.xmin + det.width / 2) * orig_w)
                fy = int((det.ymin + det.height / 2) * orig_h)
                cam_size = int(orig_h * 0.28)
                cx1 = max(0, min(orig_w - cam_size, fx - cam_size // 2))
                cy1 = max(0, min(orig_h - cam_size, fy - cam_size // 2))
                cam_crop = cv2.resize(frame[cy1:cy1 + cam_size, cx1:cx1 + cam_size], (340, 340))
                # White border around facecam
                cam_bordered = cv2.copyMakeBorder(cam_crop, 6, 6, 6, 6, cv2.BORDER_CONSTANT, value=(255, 255, 255))
                bg[300:652, 364:716] = cam_bordered

            canvas = bg

        # MODE 2: PODCAST DUAL SPLIT-SCREEN (Top Speaker + Bottom Speaker stacked 1080x960 each)
        elif mode == "podcast_dual":
            if len(detections) >= 2:
                xs = sorted([(d.location_data.relative_bounding_box.xmin + d.location_data.relative_bounding_box.width / 2) * orig_w for d in detections[:2]])
                left_speaker_x = 0.85 * left_speaker_x + 0.15 * xs[0]
                right_speaker_x = 0.85 * right_speaker_x + 0.15 * xs[1]

            # Each half needs 1080x960 aspect ratio (1.125:1)
            half_crop_h = orig_h
            half_crop_w = min(orig_w // 2, int(half_crop_h * (1080 / 960)))

            lx1 = int(max(0, min(orig_w - half_crop_w, left_speaker_x - half_crop_w / 2)))
            rx1 = int(max(0, min(orig_w - half_crop_w, right_speaker_x - half_crop_w / 2)))

            top_half = cv2.resize(frame[:, lx1:lx1 + half_crop_w], (1080, 960), interpolation=cv2.INTER_LANCZOS4)
            bot_half = cv2.resize(frame[:, rx1:rx1 + half_crop_w], (1080, 960), interpolation=cv2.INTER_LANCZOS4)

            canvas = np.vstack([top_half, bot_half])
            # Clean 4px divider line between speakers
            cv2.line(canvas, (0, 960), (1080, 960), (255, 255, 255), 4)

        # MODE 3: SINGLE SPEAKER / EXPLAINER (Smooth Face-Tracked 9:16 + Dynamic Zoom)
        else:
            if detections:
                best = max(detections, key=lambda d: d.location_data.relative_bounding_box.width)
                bbox = best.location_data.relative_bounding_box
                target_x = (bbox.xmin + bbox.width / 2.0) * orig_w
                if abs(target_x - smooth_x) > orig_w * 0.04:
                    smooth_x = 0.08 * target_x + 0.92 * smooth_x

            crop_h = int(orig_h / zoom_factor)
            crop_w = int(crop_h * (9 / 16))
            x1 = int(max(0, min(orig_w - crop_w, smooth_x - crop_w / 2)))
            y1 = (orig_h - crop_h) // 2

            cropped = frame[y1:y1 + crop_h, x1:x1 + crop_w]
            canvas = cv2.resize(cropped, (1080, 1920), interpolation=cv2.INTER_LANCZOS4)

        out.write(canvas)

    cap.release()
    out.release()
    mp_face.close()

def main():
    os.makedirs("final_reels", exist_ok=True)
    with open("work/blueprint.json", "r", encoding="utf-8") as f:
        blueprint = json.load(f)
    with open("work/words.json", "r", encoding="utf-8") as f:
        words = json.load(f)

    for idx, clip in enumerate(blueprint["clips"], start=1):
        start = clip["start_time"]
        end = clip["end_time"]
        duration = end - start
        mode = clip["content_mode"]
        theme = clip["caption_theme"]
        title = clip["title"]
        peaks = clip.get("peak_moments", [])

        safe_name = "".join(c if c.isalnum() or c in " _-" else "" for c in title).strip().replace(" ", "_")
        raw_cut = f"work/cut_{idx}.mp4"
        rendered_vid = f"work/vert_{idx}.mp4"
        ass_file = f"work/subs_{idx}.ass"
        final_reel = f"final_reels/Reel_{idx}_{mode}_{safe_name[:25]}.mp4"

        print(f"\n=== Building Reel {idx} [{mode.upper()}]: {title} ({start}s - {end}s) ===")

        # 1. Extract lossless-grade segment
        subprocess.run([
            "ffmpeg", "-nostdin", "-y", "-ss", str(start), "-i", "work/master.mp4",
            "-t", str(duration), "-c:v", "libx264", "-crf", "15", "-preset", "veryfast",
            "-c:a", "aac", "-b:a", "192k", raw_cut
        ], check=True)

        # 2. Adaptive Layout & Dynamic Physics Render
        render_adaptive_video(raw_cut, rendered_vid, mode, start, peaks)

        # 3. Generate Animated Pop-In Captions + Hook Banner
        build_dynamic_captions(words, start, end, theme, title, ass_file)

        # 4. Final Studio Master Encode (CRF 18 + Broadcast Loudnorm)
        subprocess.run([
            "ffmpeg", "-nostdin", "-y",
            "-i", rendered_vid,
            "-i", raw_cut,
            "-vf", f"ass={ass_file}:fontsdir=fonts",
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-c:v", "libx264",
            "-crf", "18",
            "-preset", "medium",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "192k",
            "-ar", "48000",
            "-af", "loudnorm=I=-14:TP=-1.5:LRA=11",
            "-movflags", "+faststart",
            final_reel
        ], check=True)

        print(f"✅ Mastered Reel saved: {final_reel}")

if __name__ == "__main__":
    main()
