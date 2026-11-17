"""Step 1: download, whisper, audio-energy map, silence gaps, scene keyframes,
then Gemini (free-tier only) as EYES+EARS. Fallback: dense frames -> Gemini
inline -> OpenRouter free VL (Ling / Nemotron). Never 6 blind keyframes again."""
import os
import time
import json
import wave
import subprocess
import numpy as np
import requests
from typing import Literal, Optional
from pydantic import BaseModel, Field
from faster_whisper import WhisperModel, BatchedInferencePipeline
from google import genai
from google.genai import types
from common import run, ffprobe_duration, save_json, extract_json, openrouter_chat, WORK

VIDEO_URL = os.environ["VIDEO_URL"]
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "small")

FREE_MODEL_ORDER = [
    "gemini-3.8-flash", "gemini-3.6-flash", "gemini-4-hydrogen",
    "gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-3-flash-preview",
]
EXCLUDED = ("tts", "live", "image", "nano-banana", "omni", "transcribe",
            "translate", "embedding", "robotics", "deep-research", "pro", "argon", "carbon")

VL_FALLBACK = ["inclusionai/ling-3.2-flash:free", "nvidia/nemotron-3.5-super-vl:free"]


class SpeakerBox(BaseModel):
    label: str = Field(description="Speaker name if readable on screen, else 'left'/'right'/'center'")
    x: float = Field(description="Normalized 0-1 horizontal center of the speaker")
    y: float = Field(description="Normalized 0-1 vertical center")


class FrameNote(BaseModel):
    t: float
    layout: Literal["podcast_dual", "podcast_single", "gaming_action", "explainer"]
    speakers: list[SpeakerBox] = []
    on_screen_text: str = ""
    note: str = ""


class HypeMoment(BaseModel):
    t: float
    kind: str = Field(description="laughter / punchline / kill / beat_drop / shout / reveal")
    note: str = ""


class VisualNotes(BaseModel):
    overall_genre: str
    summary: str
    timeline: list[FrameNote] = []
    hype_moments: list[HypeMoment] = []


def get_model_pool(client):
    try:
        discovered = []
        for m in client.models.list():
            name = m.name.replace("models/", "")
            methods = getattr(m, "supported_actions", None) or \
                getattr(m, "supported_generation_methods", []) or []
            if "generateContent" not in methods or not name.startswith("gemini-"):
                continue
            if any(b in name.lower() for b in EXCLUDED):
                continue
            if not any(k in name for k in ("flash", "lite", "hydrogen")):
                continue
            discovered.append(name)
        ordered = [m for m in FREE_MODEL_ORDER if m in discovered]
        extras = sorted([m for m in discovered if m not in ordered], reverse=True)
        pool = ordered + extras
        print(f"Free-tier Gemini pool: {pool}", flush=True)
        return pool or FREE_MODEL_ORDER
    except Exception as e:
        print(f"⚠️ model list failed ({e}); using built-in free pool", flush=True)
        return FREE_MODEL_ORDER


def gemini_failover(client, pool, contents, prompt):
    last = None
    for name in pool:
        try:
            print(f"Trying {name}...", flush=True)
            return client.models.generate_content(
                model=name, contents=[*contents, prompt],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=VisualNotes, temperature=0.2))
        except Exception as e:
            last = e
            if "INVALID_ARGUMENT" in str(e) and "token" in str(e).lower():
                raise
            print(f"⚠️ {name} busy ({str(e)[:120]}), failing over", flush=True)
            time.sleep(2)
    raise RuntimeError(f"All Gemini models failed: {last}")


def audio_energy(wav_path):
    with wave.open(wav_path, "rb") as w:
        sr = w.getframerate()
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    hop = int(sr * 0.25)
    n = len(pcm) // hop
    curve = []
    for i in range(n):
        seg = pcm[i * hop:(i + 1) * hop]
        rms = float(np.sqrt(np.mean(seg ** 2)) + 1e-6)
        curve.append(round(20 * np.log10(rms), 2))
    curve = np.array(curve)
    med = float(np.median(curve))
    peaks = []
    for i in range(1, n - 1):
        if curve[i] > med + 6 and curve[i] >= curve[i - 1] and curve[i] > curve[i + 1]:
            t = round(i * 0.25, 2)
            if not peaks or t - peaks[-1] > 0.6:
                peaks.append(t)
    return {"hop": 0.25, "curve": curve.tolist(), "peaks": peaks}


def silence_gaps(words, duration):
    gaps = []
    if words and words[0]["start"] > 0.5:
        gaps.append([0.0, words[0]["start"]])
    for a, b in zip(words, words[1:]):
        if b["start"] - a["end"] >= 0.35:
            gaps.append([round(a["end"], 3), round(b["start"], 3)])
    if words and duration - words[-1]["end"] > 0.5:
        gaps.append([words[-1]["end"], duration])
    return gaps


def scene_keyframes(master, duration):
    changes = []
    try:
        from scenedetect import detect, ContentDetector
        scenes = detect(master, ContentDetector(threshold=27.0))
        changes = [round(s[0].get_seconds(), 2) for s in scenes[1:]]
    except Exception as e:
        print(f"⚠️ scene detect failed ({e}); uniform sampling only", flush=True)
    step = max(10.0, duration / 60.0)
    uniform = [round(t, 2) for t in np.arange(step / 2, duration, step)]
    kf = sorted(set(changes + uniform))[:150]
    return {"changes": changes, "keyframes": kf}


def extract_frames(master, timestamps, tag):
    paths = []
    for i, ts in enumerate(timestamps):
        p = f"{WORK}/{tag}_{i}.jpg"
        subprocess.run(["ffmpeg", "-nostdin", "-y", "-ss", str(ts), "-i", master,
                        "-frames:v", "1", "-vf", "scale=-2:480", p],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if os.path.exists(p):
            paths.append((ts, p))
    return paths


def vl_fallback_notes(frames, prompt):
    """Dense frame batches through free OpenRouter vision models."""
    import base64
    timeline, hype, genre, summary = [], [], "unknown", ""
    for chunk_start in range(0, len(frames), 60):
        chunk = frames[chunk_start:chunk_start + 60]
        content = [{"type": "text", "text": prompt + "\nFrame timestamps (seconds): "
                    + json.dumps([t for t, _ in chunk])}]
        for ts, p in chunk:
            with open(p, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            content.append({"type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        txt = openrouter_chat(VL_FALLBACK, [{"role": "user", "content": content}],
                              schema_hint=VisualNotes.model_json_schema())
        d = extract_json(txt)
        timeline += d.get("timeline", [])
        hype += d.get("hype_moments", [])
        genre = d.get("overall_genre", genre)
        summary = d.get("summary", summary) or summary
    return {"overall_genre": genre, "summary": summary,
            "timeline": timeline, "hype_moments": hype}


def main():
    os.makedirs(WORK, exist_ok=True)
    master, proxy, wav = f"{WORK}/master.mp4", f"{WORK}/proxy_gemini.mp4", f"{WORK}/audio.wav"

    print("1. Downloading master via yt-dlp...", flush=True)
    yt = ["yt-dlp", "-f",
          "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
          "--merge-output-format", "mp4",
          "--extractor-args", "youtube:player_client=default,tv_embedded",
          "-o", master]
    cookies = os.environ.get("YT_COOKIES", "").strip()
    if cookies:
        with open(f"{WORK}/cookies.txt", "w") as f:
            f.write(cookies)
        yt += ["--cookies", f"{WORK}/cookies.txt"]
    yt.append(VIDEO_URL)
    run(yt)

    duration = ffprobe_duration(master)
    dmin = duration / 60
    fps = "2" if dmin <= 40 else ("1" if dmin <= 110 else "0.5")
    print(f"2. Proxy {fps}fps for {dmin:.1f}min video...", flush=True)
    run(["ffmpeg", "-nostdin", "-y", "-i", master, "-vf", f"scale=-2:360,fps={fps}",
         "-c:v", "libx264", "-crf", "33", "-preset", "ultrafast",
         "-c:a", "aac", "-b:a", "32k", "-ac", "1", proxy])
    run(["ffmpeg", "-nostdin", "-y", "-i", master, "-vn",
         "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", wav])

    print(f"3. Whisper ({WHISPER_MODEL}) word-level transcript...", flush=True)
    wm = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8", cpu_threads=4)
    segs, _ = BatchedInferencePipeline(model=wm).transcribe(wav, batch_size=8, word_timestamps=True)
    words, lines = [], []
    for s in segs:
        lines.append(f"[{s.start:.2f}s - {s.end:.2f}s] {s.text.strip()}")
        for w in (s.words or []):
            words.append({"word": w.word.strip(), "start": round(w.start, 3), "end": round(w.end, 3)})
    save_json("words.json", words)
    with open(f"{WORK}/transcript.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("4. Audio energy + silence gaps + scene keyframes...", flush=True)
    save_json("energy.json", audio_energy(wav))
    save_json("gaps.json", silence_gaps(words, duration))
    scenes = scene_keyframes(master, duration)
    save_json("scenes.json", scenes)

    transcript = "\n".join(lines)
    prompt = f"""You are the EYES and EARS of a viral reel studio. Watch and LISTEN to this video.
Produce visual/audio notes, NOT clip selections (a separate showrunner picks clips).
Return: overall_genre; a 2-sentence summary; a timeline of <=40 entries spread across the video
(t, layout, speaker boxes with normalized x/y and on-screen names if readable, on_screen_text, note);
and hype_moments (t, kind, note) for laughter, punchlines, kills, beat drops, shouts, reveals —
use what you HEAR as well as what you see. Timestamps in seconds.

Transcript for speaker-name grounding:
{transcript}"""

    client = genai.Client(api_key=GEMINI_API_KEY)
    pool = get_model_pool(client)
    notes, uploaded = None, None
    try:
        print("5. Uploading proxy to Gemini File API...", flush=True)
        uploaded = client.files.upload(file=proxy)
        while uploaded.state.name == "PROCESSING":
            time.sleep(3)
            uploaded = client.files.get(name=uploaded.name)
        if uploaded.state.name != "ACTIVE":
            raise RuntimeError(f"file state {uploaded.state.name}")
        notes = json.loads(gemini_failover(client, pool, [uploaded], prompt).text)
    except Exception as e:
        print(f"⚠️ video path failed ({e}); dense keyframes -> Gemini inline", flush=True)
        frames = extract_frames(master, scenes["keyframes"][:40], "kf")
        try:
            parts = [types.Part.from_bytes(data=open(p, "rb").read(), mime_type="image/jpeg")
                     for _, p in frames]
            notes = json.loads(gemini_failover(client, pool, parts, prompt).text)
        except Exception as e2:
            print(f"⚠️ Gemini inline failed ({e2}); OpenRouter free VL fallback", flush=True)
            try:
                frames = extract_frames(master, scenes["keyframes"], "kf")
                notes = vl_fallback_notes(frames, prompt)
            except Exception as e3:
                print(f"⚠️ ALL vision paths failed ({e3}); showrunner will work transcript-only", flush=True)
                notes = {"overall_genre": "unknown", "summary": "", "timeline": [], "hype_moments": []}
    finally:
        if uploaded:
            try:
                client.files.delete(name=uploaded.name)
            except Exception:
                pass

    save_json("visual_notes.json", notes)
    save_json("duration.json", {"duration": duration})
    print("Visual notes saved.\n", json.dumps(notes)[:1500], flush=True)


if __name__ == "__main__":
    main()
