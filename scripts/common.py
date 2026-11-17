"""Shared helpers: subprocess, json, OpenRouter chat, ASS time, emoji assets."""
import os
import re
import json
import time
import subprocess
import requests

WORK = "work"


def run(cmd, **kw):
    return subprocess.run(cmd, check=True, **kw)


def ffprobe_duration(path: str) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        capture_output=True, text=True, check=True)
    return float(r.stdout.strip())


def load_json(name):
    with open(os.path.join(WORK, name), "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(name, obj):
    os.makedirs(WORK, exist_ok=True)
    with open(os.path.join(WORK, name), "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def extract_json(text: str):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    try:
        return json.loads(text)
    except Exception:
        s, e = text.find("{"), text.rfind("}")
        if s >= 0 and e > s:
            return json.loads(text[s:e + 1])
        raise


def openrouter_chat(model_pool, messages, temperature=0.3, max_tokens=12000,
                    json_mode=True, schema_hint=None):
    """Failover chat across OpenRouter models. Returns raw content text."""
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY missing")
    if schema_hint:
        messages = messages + [{
            "role": "user",
            "content": "Respond with ONLY valid JSON matching this schema:\n"
                       + json.dumps(schema_hint)}]
    last = None
    for model in model_pool:
        body = {"model": model, "messages": messages,
                "temperature": temperature, "max_tokens": max_tokens}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        for attempt in range(3):
            try:
                r = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={"Authorization": f"Bearer {key}",
                             "Content-Type": "application/json"},
                    json=body, timeout=600)
                if r.status_code == 429:
                    wait = float(r.headers.get("Retry-After", 5))
                    print(f"⏳ {model} 429, waiting {wait}s", flush=True)
                    time.sleep(min(wait, 30))
                    continue
                if r.status_code in (402, 404, 400):
                    raise RuntimeError(f"{model} -> {r.status_code} {r.text[:160]}")
                r.raise_for_status()
                data = r.json()
                return data["choices"][0]["message"]["content"]
            except Exception as e:
                last = e
                print(f"⚠️ {model} attempt {attempt+1} failed ({str(e)[:140]})", flush=True)
                time.sleep(3)
        print(f"⚠️ failing over from {model}", flush=True)
    raise RuntimeError(f"All OpenRouter models failed: {last}")


def ass_time(t: float) -> str:
    t = max(0.0, t)
    h = int(t // 3600); m = int((t % 3600) // 60); s = t % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def emoji_png(emoji: str, path: str) -> bool:
    """Download a Twemoji PNG (CC-BY 4.0). Tries with/without FE0F."""
    cps = "-".join(f"{ord(c):x}" for c in emoji if ord(c) != 0xFE0F)
    cps_f = "-".join(f"{ord(c):x}" for c in emoji)
    for cp in (cps_f, cps):
        url = f"https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/72x72/{cp}.png"
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200 and r.content[:4] == b"\x89PNG":
                with open(path, "wb") as f:
                    f.write(r.content)
                return True
        except Exception:
            pass
    return False
