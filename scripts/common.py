"""Shared helpers. Never returns null content; never hangs silently."""
import os
import re
import json
import time
import subprocess
import requests

WORK = "work"
OR_URL = "https://openrouter.ai/api/v1/chat/completions"


def run(cmd, **kw):
    return subprocess.run(cmd, check=True, **kw)


def ffprobe_duration(path):
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


def extract_json(text):
    if not text or not isinstance(text, str):
        raise ValueError("model returned no text")
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    try:
        return json.loads(text)
    except Exception:
        s = text.find("{")
        e = text.rfind("}")
        if s >= 0 and e > s:
            return json.loads(text[s:e + 1])
        raise


def openrouter_chat(model_pool, messages, temperature=0.3, max_tokens=8000,
                    json_mode=True, schema_hint=None):
    """Failover chat across OpenRouter models. Hidden reasoning is disabled for
    JSON calls (prevents null-content token burns). Retries 429, strips an
    unsupported reasoning param on 400, rejects empty content, fails over."""
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY missing")
    if schema_hint:
        messages = messages + [{
            "role": "user",
            "content": "Respond with ONLY valid JSON matching this schema:\n"
                       + json.dumps(schema_hint)}]
    headers = {"Authorization": "Bearer " + key, "Content-Type": "application/json"}
    last = None
    for model in model_pool:
        for attempt in range(4):
            body = {"model": model, "messages": messages,
                    "temperature": temperature, "max_tokens": max_tokens}
            if json_mode:
                body["response_format"] = {"type": "json_object"}
                body["reasoning"] = {"enabled": False}
            try:
                r = requests.post(OR_URL, headers=headers, json=body, timeout=300)
                if r.status_code == 429:
                    wait = float(r.headers.get("Retry-After", 5))
                    print("waiting on 429: " + str(wait) + "s", flush=True)
                    time.sleep(min(wait, 30))
                    continue
                if r.status_code == 400 and "reasoning" in body:
                    body.pop("reasoning")
                    r = requests.post(OR_URL, headers=headers, json=body, timeout=300)
                if r.status_code in (402, 404, 400):
                    raise RuntimeError(model + " -> " + str(r.status_code) + " " + r.text[:160])
                r.raise_for_status()
                data = r.json()
                choices = data.get("choices") or [{}]
                msg = choices[0].get("message") or {}
                content = msg.get("content")
                if not content or not str(content).strip():
                    raise RuntimeError("empty/null content (finish_reason="
                                       + str(choices[0].get("finish_reason")) + ")")
                return content
            except Exception as e:
                last = e
                print("attempt failed: " + str(e)[:140], flush=True)
                time.sleep(2)
        print("failing over from " + model, flush=True)
    raise RuntimeError("All OpenRouter models failed: " + str(last))


def ass_time(t):
    t = max(0.0, t)
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t % 60
    return str(h) + ":" + str(m).zfill(2) + ":" + ("%05.2f" % s)


def emoji_png(emoji, path):
    """Twemoji PNG (CC-BY 4.0). Tries with/without FE0F."""
    cps = "-".join(format(ord(c), "x") for c in emoji if ord(c) != 0xFE0F)
    cps_f = "-".join(format(ord(c), "x") for c in emoji)
    for cp in (cps_f, cps):
        url = "https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/72x72/" + cp + ".png"
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200 and r.content[:4] == b"\x89PNG":
                with open(path, "wb") as f:
                    f.write(r.content)
                return True
        except Exception:
            pass
    return False
