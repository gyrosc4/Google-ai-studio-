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

def main():
    os.makedirs("work", exist_ok=True)
    master_video = "work/master.mp4"
    proxy_video = "work/proxy_gemini.mp4"
    audio_wav = "work/audio.wav"

    print("1. Downloading 1080p master stream via yt-dlp...")
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

    print("2. Creating lightweight multimodal proxy for Gemini Vision+Audio analysis...")
    subprocess.run([
        "ffmpeg", "-nostdin", "-y", "-i", master_video,
        "-vf", "scale=-2:360,fps=2",
        "-c:v", "libx264", "-crf", "32", "-preset", "ultrafast",
        "-c:a", "aac", "-b:a", "48k", "-ac", "1",
        proxy_video
    ], check=True)

    print("3. Extracting 16kHz mono WAV for high-precision Whisper alignment...")
    subprocess.run([
        "ffmpeg", "-nostdin", "-y", "-i", master_video,
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        audio_wav
    ], check=True)

    print("4. Running faster-whisper (small.en + Silero VAD)...")
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

    print("5. Uploading proxy video to Gemini 2.5 Pro for multimodal scene & audio analysis...")
    client = genai.Client(api_key=GEMINI_API_KEY)
    uploaded_file = client.files.upload(file=proxy_video)

    while uploaded_file.state.name == "PROCESSING":
        time.sleep(3)
        uploaded_file = client.files.get(name=uploaded_file.name)

    if uploaded_file.state.name != "ACTIVE":
        raise RuntimeError(f"Gemini file processing failed: {uploaded_file.state.name}")

    prompt = f"""
    You are an elite viral short-form video director (Alex Hormozi / top Twitch & Podcast clipper level).
    Watch and listen to the attached video alongside the exact word-accurate transcript below.

    Your task: Extract the 3 highest-performing, standalone segments (30 to 60 seconds each) for Instagram Reels.

    Selection & Editing Rules:
    1. PODCASTS / INTERVIEWS:
       - Choose controversial takes, emotional stories, or sharp insights.
       - If two speakers are visible in the frame, set content_mode to 'podcast_dual' so the editor creates a top/bottom split screen. If one speaker is framed, use 'podcast_single'.
    2. GAMING STREAMS:
       - Watch the visuals and listen to audio spikes to find insane clutches, flick shots, funny fails, or intense reactions.
       - Set content_mode to 'gaming_action' so HUD/crosshair stays uncropped and webcam is preserved.
       - Mark exact timestamps of kills, laughs, or beat drops in 'peak_moments' for dynamic zoom punches.
    3. HOOK & ENDING:
       - Start on the exact second the hook begins (never mid-sentence). End right after the payoff.
       - Create a punchy 3-5 word 'title' that acts as an irresistible top hook banner.

    Transcript:
    {chr(10).join(transcript_lines)}
    """

    response = client.models.generate_content(
        model="gemini-2.5-pro",
        contents=[uploaded_file, prompt],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=DirectorBlueprint,
            temperature=0.25
        ),
    )

    # Clean up remote Gemini file
    client.files.delete(name=uploaded_file.name)

    with open("work/blueprint.json", "w", encoding="utf-8") as f:
        f.write(response.text)

    print("Director Blueprint Generated:\n", response.text)

if __name__ == "__main__":
    main()
