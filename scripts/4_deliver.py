"""Step 4: Telegram delivery. 49MB guard -> compact re-encode -> sendVideo ->
sendDocument fallback. Real response checking. Never uploads corpses."""
import os
import glob
import subprocess
import requests

TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
CHAT = os.environ["TELEGRAM_CHAT_ID"]
LIMIT = 49 * 1024 * 1024


def send(method, path, caption):
    if method == "sendVideo":
        field = "video"
    else:
        field = "document"
    with open(path, "rb") as f:
        r = requests.post("https://api.telegram.org/bot" + TOKEN + "/" + method,
                          data={"chat_id": CHAT, "caption": caption},
                          files={field: f}, timeout=600)
    ok = r.status_code == 200 and r.json().get("ok")
    if not ok:
        print(method + " failed: " + r.text[:200], flush=True)
    return ok


def main():
    reels = sorted(glob.glob("final_reels/*.mp4"))
    if not reels:
        raise SystemExit("no reels found in final_reels/")
    failures = 0
    for reel in reels:
        name = os.path.basename(reel)
        if os.path.getsize(reel) < 1 * 1024 * 1024:
            print("skipping suspicious tiny file " + name, flush=True)
            continue
        path = reel
        if os.path.getsize(reel) > LIMIT:
            compact = "work/compact_" + name
            print(name + " > 49MB, re-encoding compact copy", flush=True)
            subprocess.run(["ffmpeg", "-nostdin", "-y", "-i", reel,
                            "-vf", "scale=720:-2", "-c:v", "libx264", "-crf", "26",
                            "-preset", "fast", "-c:a", "aac", "-b:a", "128k",
                            compact], check=True)
            path = compact
        caption = "🎬 " + name
        if not send("sendVideo", path, caption):
            if not send("sendDocument", path, caption):
                failures = failures + 1
                continue
        print("delivered " + name, flush=True)
    if failures:
        raise SystemExit(str(failures) + " reel(s) failed delivery (masters still in artifacts)")


if __name__ == "__main__":
    main()
