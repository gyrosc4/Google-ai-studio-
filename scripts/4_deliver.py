"""Step 4: Telegram delivery with 50MB guard + real error checking."""
import os
import glob
import subprocess
import requests

TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
CHAT = os.environ["TELEGRAM_CHAT_ID"]
LIMIT = 49 * 1024 * 1024


def send(method, path, caption):
    with open(path, "rb") as f:
        r = requests.post(f"https://api.telegram.org/bot{TOKEN}/{method}",
                          data={"chat_id": CHAT, "caption": caption},
                          files={method.replace("send", "").lower(): f}, timeout=600)
    ok = r.status_code == 200 and r.json().get("ok")
    if not ok:
        print(f"❌ {method} failed: {r.text[:200]}", flush=True)
    return ok


def main():
    failures = 0
    for reel in sorted(glob.glob("final_reels/*.mp4")):
        name = os.path.basename(reel)
        path = reel
        if os.path.getsize(reel) > LIMIT:
            compact = f"work/compact_{name}"
            print(f"⚠️ {name} > 49MB, re-encoding compact copy", flush=True)
            subprocess.run(["ffmpeg", "-nostdin", "-y", "-i", reel,
                            "-vf", "scale=720:-2", "-c:v", "libx264", "-crf", "26",
                            "-preset", "fast", "-c:a", "aac", "-b:a", "128k", compact], check=True)
            path = compact
        caption = f"🎬 {name}"
        if not send("sendVideo", path, caption):
            if not send("sendDocument", path, caption):
                failures += 1
                continue
        print(f"✅ delivered {name}", flush=True)
    if failures:
        raise SystemExit(f"{failures} reel(s) failed delivery (masters still in artifacts)")


if __name__ == "__main__":
    main()
