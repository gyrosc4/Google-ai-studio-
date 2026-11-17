import os
import time
import json
import subprocess
from typing import Literal
from faster_whisper import WhisperModel
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

VIDEO_URL = os.environ["VIDEO_URL"]
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
MODEL_NAME = "gemini-3.8-flash"

class ClipDirective(BaseModel):
    title: str = Field(description="Punchy 3-5 word viral hook banner text for the top of the Reel")
    start_time: float = Field(description="Exact start timestamp in seconds")
    end_time: float = Field(description="Exact end timestamp in seconds (duration 30 to 60s)")
    content_mode: Literal["podcast_dual", "podcast_single", "gaming_action", "explainer"] = Field(
        description="Best visual layout style based on the video footage"
    )
    caption_theme: Literal["hype_gaming", "editorial_podcast", "bold_explainer"]
    peak_moments: list[float] = Field(
        description="Timestamps (in seconds relative to full video) where hype, punchlines, or key shots happen for zoom punches"
    )
    virality_score: int = Field(description="1-100 retention score")

class DirectorBlueprint(BaseModel):
    overall_genre: str
    clips: list[ClipDirective]

def get_video_duration(video_path: str) -> float:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", video_path
        ],
        capture_output=True, text=True, check=True
    )
    return float(result.stdout.strip())

def main():
    os.makedirs("work", exist_ok=True)
    master_video = "work/master.mp4"
    proxy_video = "work/proxy_gemini.mp4"
    audio_wav = "work/audio.wav"

    print("1. Downloading 1080p master stream via yt-dlp...", flush=True)
    yt_cmd = [
        "yt-dlp",
        "-f", "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--merge-output-format", "mp4",
        "--extractor-args", "youtube:player_client=default,tv_embedded",
        "-o", master_video,
    ]

    yt_cookies = os.environ.get("YT_COOKIES", "").strip()
    if yt_cookies:
        with open("work/cookies.txt", "w", encoding="utf-8") as cf:
            cf.write(yt_cookies)
        yt_cmd.extend(["--cookies", "work/cookies.txt"])

    yt_cmd.append(VIDEO_URL)
    subprocess.run(yt_cmd, check=True)

    duration_sec = get_video_duration(master_video)
    duration_min = duration_sec / 60.0
    print(f"Video duration detected: {duration_min:.1f} minutes ({duration_sec:.0f}s)", flush=True)

    # Dynamically scale proxy FPS so total tokens stay safely under 800k even on 3+ hour videos
    if duration_min <= 40:
        proxy_fps = "2"
    elif duration_min <= 110:
        proxy_fps = "1"
    else:
        proxy_fps = "0.5"  # 1 frame every 2 seconds for 2-4 hour VODs

    print(f"2. Creating token-optimized proxy ({proxy_fps} fps) for {MODEL_NAME}...", flush=True)
    subprocess.run([
        "ffmpeg", "-nostdin", "-y", "-i", master_video,
        "-vf", f"scale=-2:360,fps={proxy_fps}",
        "-c:v", "libx264", "-crf", "33", "-preset", "ultrafast",
        "-c:a", "aac", "-b:a", "32k", "-ac", "1",
        proxy_video
    ], check=True)

    print("3. Extracting 16kHz mono WAV for high-precision Whisper alignment...", flush=True)
    subprocess.run([
        "ffmpeg", "-nostdin", "-y", "-i", master_video,
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        audio_wav
    ], check=True)

    print("4. Running faster-whisper (small.en + Silero VAD)...", flush=True)
    model = WhisperModel("small.en", device="cpu", compute_type="int8")
    segments, _ = model.transcribe(audio_wav, word_timestamps=True, vad_filter=True)

    transcript_lines = []
    word_timeline = []
    for seg in segments:
        transcript_lines.append(f"[{seg.start:.2f}s - {seg.end:.2f}s] {seg.text.strip()}")
        if seg.words:
            for w in seg.words:
                word_timeline.append({
                    "word": w.word.strip(),
                    "start": round(w.start, 3),
                    "end": round(w.end, 3)
                })

    with open("work/words.json", "w", encoding="utf-8") as f:
        json.dump(word_timeline, f, indent=2)

    full_transcript = "\n".join(transcript_lines)
    client = genai.Client(api_key=GEMINI_API_KEY)

    base_prompt = f"""
    You are an elite viral short-form video director (Alex Hormozi / top Twitch & Podcast clipper level).
    Analyze the attached video/audio alongside the exact word-accurate transcript below.

    Your task: Extract the 3 highest-performing, standalone segments (30 to 60 seconds each) for Instagram Reels.

    Selection & Editing Rules:
    1. PODCASTS / INTERVIEWS:
       - Choose controversial takes, emotional stories, or sharp insights.
       - If two speakers are visible in the frame, set content_mode to 'podcast_dual' so the editor creates a top/bottom split screen. If one speaker is framed, use 'podcast_single'.
    2. GAMING STREAMS:
       - Find insane clutches, flick shots, funny fails, or intense reactions.
       - Set content_mode to 'gaming_action' so HUD/crosshair stays uncropped and webcam is preserved.
       - Mark exact timestamps of kills, laughs, or beat drops in 'peak_moments' for dynamic zoom punches.
    3. HOOK & ENDING:
       - Start on the exact second the hook begins (never mid-sentence). End right after the payoff.
       - Create a punchy 3-5 word 'title' that acts as an irresistible top hook banner.

    Transcript:
    {full_transcript}
    """

    uploaded_file = None
    try:
        print(f"5. Uploading proxy video to {MODEL_NAME} for multimodal analysis...", flush=True)
        uploaded_file = client.files.upload(file=proxy_video)

        while uploaded_file.state.name == "PROCESSING":
            time.sleep(3)
            uploaded_file = client.files.get(name=uploaded_file.name)

        if uploaded_file.state.name != "ACTIVE":
            raise RuntimeError(f"Gemini file processing failed: {uploaded_file.state.name}")

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[uploaded_file, base_prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=DirectorBlueprint,
                temperature=0.25
            ),
        )
    except Exception as e:
        print(f"⚠️ Multimodal video upload hit limit or error ({e}). Falling back to transcript + keyframe analysis...", flush=True)
        # Extract 6 evenly spaced keyframes so Gemini still sees visual layout (Podcast vs Gaming)
        keyframe_parts = []
        for idx, pct in enumerate([0.1, 0.25, 0.45, 0.65, 0.8, 0.92]):
            ts = duration_sec * pct
            img_path = f"work/kf_{idx}.jpg"
            subprocess.run([
                "ffmpeg", "-nostdin", "-y", "-ss", str(ts), "-i", master_video,
                "-frames:v", "1", "-vf", "scale=-2:480", img_path
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if os.path.exists(img_path):
                with open(img_path, "rb") as img_f:
                    keyframe_parts.append(types.Part.from_bytes(data=img_f.read(), mime_type="image/jpeg"))

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[*keyframe_parts, base_prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=DirectorBlueprint,
                temperature=0.25
            ),
        )
    finally:
        if uploaded_file:
            try:
                client.files.delete(name=uploaded_file.name)
            except Exception:
                pass

    with open("work/blueprint.json", "w", encoding="utf-8") as f:
        f.write(response.text)

    print("Director Blueprint Generated:\n", response.text, flush=True)

if __name__ == "__main__":
    main()
