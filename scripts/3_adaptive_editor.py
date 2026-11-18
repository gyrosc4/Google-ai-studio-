"""Step 3: editor v3. BlazeFace init once; 30fps cap; cheap resamplers; low-res
blur; eased zoom+shake in ALL modes; silence tightening w/ remap; kinetic hook,
captions v2, progress bar, lower-third, emoji stickers; procedural SFX ducked
under voice; single encode pass; hard duration/size guards; ffmpeg stderr logged;
partial files deleted; intermediates cleaned; reel-count assert at the end."""
import os
import math
import time
import glob
import wave
import subprocess
import numpy as np
import cv2
import requests
from scipy.signal import butter, lfilter
from common import run, load_json, ass_time, emoji_png, WORK

TIGHTEN = os.environ.get("TIGHTEN_SILENCE", "1") == "1"
MODEL_PATH = WORK + "/blaze_face_short_range.tflite"
HAAR_PATH = WORK + "/haarcascade_frontalface_default.xml"
OUT_W, OUT_H = 1080, 1920
MAX_REEL_BYTES = 300 * 1024 * 1024

THEMES = {
    "hype_gaming": ("&H0000FFFF", "&H00FFFFFF", "&H00000000"),
    "editorial_podcast": ("&H0000E5FF", "&H00FFFFFF", "&H00101010"),
    "bold_explainer": ("&H0055FF55", "&H00FFFFFF", "&H00000000"),
}


def ensure_face_models():
    if not os.path.exists(MODEL_PATH):
        try:
            r = requests.get("https://storage.googleapis.com/mediapipe-models/face_detector/"
                             "blaze_face_short_range/float16/1/blaze_face_short_range.tflite",
                             timeout=60)
            with open(MODEL_PATH, "wb") as f:
                f.write(r.content)
        except Exception as e:
            print("blaze download failed: " + str(e), flush=True)
    if not os.path.exists(HAAR_PATH):
        try:
            r = requests.get("https://raw.githubusercontent.com/opencv/opencv/master/data/"
                             "haarcascades/haarcascade_frontalface_default.xml", timeout=60)
            with open(HAAR_PATH, "wb") as f:
                f.write(r.content)
        except Exception as e:
            print("haar download failed: " + str(e), flush=True)


def make_detector():
    ensure_face_models()
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
        if os.path.exists(MODEL_PATH):
            opts = mp_vision.FaceDetectorOptions(
                base_options=mp_python.BaseOptions(
                    model_asset_path=MODEL_PATH,
                    delegate=mp_python.BaseOptions.Delegate.CPU),
                min_detection_confidence=0.5)
            d = mp_vision.FaceDetector.create_from_options(opts)
            print("Face tracker: MediaPipe BlazeFace (CPU) active.", flush=True)
            return ("mp", d)
    except Exception as e:
        print("MediaPipe init failed, Haar fallback: " + str(e), flush=True)
    cands = [HAAR_PATH, getattr(cv2.data, "haarcascades", "") + "haarcascade_frontalface_default.xml"]
    for p in cands:
        if p and os.path.exists(p):
            clf = cv2.CascadeClassifier(p)
            if not clf.empty():
                print("Face tracker: Haar from " + p, flush=True)
                return ("haar", clf)
    print("no face detector; center framing", flush=True)
    return ("none", None)


def faces(kind, det, frame):
    h, w, _ = frame.shape
    if kind == "mp":
        import mediapipe as mp
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        out = []
        for d in res.detections:
            bb = d.bounding_box
            out.append(((bb.origin_x + bb.width / 2) / w,
                         (bb.origin_y + bb.height / 2) / h))
        return out
    if kind == "haar":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = det.detectMultiScale(gray, 1.15, 5, minSize=(40, 40))
        return [((x + bw / 2) / w, (y + bh / 2) / h) for (x, y, bw, bh) in rects]
    return []


def keep_segments(gaps, s, e):
    segs = []
    cur = 0.0
    for gs, ge in gaps:
        gs = gs - s
        ge = ge - s
        if ge <= 0 or gs >= e - s or ge - gs <= 1.2:
            continue
        a = max(cur, gs + 0.15)
        b = min(e - s, ge - 0.15)
        if a > cur + 0.05:
            segs.append((cur, a))
        cur = b
    if cur < e - s - 0.05:
        segs.append((cur, e - s))
    if not segs:
        segs = [(0.0, e - s)]
    return segs


def make_remap(segs):
    offs = []
    acc = 0.0
    for a, b in segs:
        offs.append((a, b, acc))
        acc = acc + (b - a)

    def remap(t):
        for a, b, off in offs:
            if t <= b:
                return off + max(0.0, t - a)
        return acc
    return remap, acc


def tighten_cut(raw, cut, segs):
    parts = []
    for a, b in segs:
        parts.append("between(t,%.3f,%.3f)" % (a, b))
    expr = "+".join(parts)
    try:
        run(["ffmpeg", "-nostdin", "-y", "-i", raw,
             "-vf", "select='" + expr + "',setpts=N/FRAME_RATE/TB",
             "-af", "aselect='" + expr + "',asetpts=N/SR/TB",
             "-c:v", "libx264", "-crf", "16", "-preset", "ultrafast",
             "-c:a", "aac", "-b:a", "192k", cut])
        return True
    except Exception as e:
        print("tighten failed, untrimmed cut: " + str(e), flush=True)
        run(["ffmpeg", "-nostdin", "-y", "-i", raw, "-c", "copy", cut])
        return False


def synth_sfx(path, dur, peaks, cues):
    sr = 48000
    n = int(dur * sr)
    buf = np.zeros(n, dtype=np.float32)
    rng = np.random.default_rng(7)

    def add(sig, t0, gain):
        i = int(t0 * sr)
        j = min(n, i + len(sig))
        if i < n:
            buf[i:j] = buf[i:j] + sig[:j - i] * gain

    if cues.get("whoosh", True):
        b, a = butter(2, [600 / (sr / 2), 3200 / (sr / 2)], btype="band")
        for p in peaks:
            L = int(0.35 * sr)
            env = np.sin(np.linspace(0, math.pi, L)) ** 2
            add(lfilter(b, a, rng.standard_normal(L)) * env, max(0, p - 0.12), 0.5)
    if cues.get("riser", True) and dur > 6:
        L = int(2.2 * sr)
        b, a = butter(2, 1400 / (sr / 2), btype="low")
        env = np.linspace(0, 1, L) ** 2.2
        add(lfilter(b, a, rng.standard_normal(L)) * env, dur - 2.4, 0.30)
    L = int(0.09 * sr)
    blip = np.sin(2 * np.pi * 880 * np.arange(L) / sr) * np.exp(-np.arange(L) / (0.02 * sr))
    add(blip, 0.12, 0.35)
    pcm = (np.clip(buf, -1, 1) * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def build_ass(path, words, s, e, remap, dur, clip):
    theme = THEMES.get(clip["caption_theme"], THEMES["editorial_podcast"])
    primary, secondary, outline = theme
    emphasis = set(w.upper() for w in clip.get("emphasis_words", []))
    header = ("[Script Info]\n"
              "ScriptType: v4.00+\n"
              "PlayResX: 1080\n"
              "PlayResY: 1920\n"
              "WrapStyle: 2\n"
              "ScaledBorderAndShadow: yes\n\n"
              "[V4+ Styles]\n"
              "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
              "Style: HookWord,Montserrat ExtraBold,74,&H00FFFFFF,&H00FFFFFF,&H00000000,&H0000FFFF,-1,0,0,0,100,100,0,0,1,8,2,8,60,60,180,1\n"
              "Style: KineticCaption,Montserrat ExtraBold,86," + primary + "," + secondary + "," + outline + ",&H80000000,-1,0,0,0,100,100,0,0,1,6,3,2,60,60,340,1\n"
              "Style: LowerThird,Montserrat ExtraBold,46,&H00FFFFFF,&H00FFFFFF,&H00000000,&HC8000000,-1,0,0,0,100,100,0,0,3,0,0,7,60,60,620,1\n"
              "Style: Progress,Montserrat ExtraBold,10," + primary + "," + primary + "," + primary + ",&H00000000,0,0,0,0,100,100,0,0,1,0,0,7,0,0,0,1\n\n"
              "[Events]\n"
              "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")
    events = []
    title_words = clip["title"].upper().split()
    widths = [len(w) * 46 + 24 for w in title_words]
    x = (OUT_W - sum(widths)) // 2
    for i, w in enumerate(title_words):
        st = min(0.10 * i, max(0.0, dur - 0.5))
        cx = x + widths[i] // 2
        x = x + widths[i]
        style = clip.get("hook_style", "pop")
        if style == "slide":
            tag = "{\\move(%d,420,%d,300,0,220)\\alpha&HFF&\\t(0,220,\\alpha&H00&)}" % (cx, cx)
        elif style == "shake":
            tag = "{\\fscx120\\fscy120\\alpha&HFF&\\t(0,120,\\fscx100\\fscy100\\alpha&H00&)\\t(120,200,\\frz2)\\t(200,280,\\frz0)}"
        else:
            tag = "{\\fscx30\\fscy30\\alpha&HFF&\\t(0,170,\\fscx100\\fscy100\\alpha&H00&)}"
        events.append("Dialogue: 2,%s,%s,HookWord,,0,0,0,,{\\an8\\pos(%d,300)}%s%s" % (
            ass_time(st), ass_time(dur), cx, tag, w))
    events.append("Dialogue: 0,0:00:00.00,%s,Progress,,0,0,0,,"
                  "{\\p1\\clip(0,1906,0,1920)\\t(0,%d,\\clip(0,1906,1080,1920))}"
                  "m 0 0 l 1080 0 1080 14 0 14{\\p0}" % (ass_time(dur), int(dur * 1000)))
    if clip.get("lower_third"):
        events.append("Dialogue: 1,0:00:00.30,%s,LowerThird,,0,0,0,,"
                      "{\\move(-500,700,60,700,0,350)\\fad(0,250)}%s" % (
                          ass_time(min(dur, 5.0)), clip["lower_third"].upper()))
    cw = []
    for w in words:
        if w["end"] > s and w["start"] < e and w["word"].strip():
            cw.append({"word": w["word"].upper(),
                       "start": remap(max(s, w["start"]) - s),
                       "end": remap(min(e, w["end"]) - s)})
    for i in range(0, len(cw), 3):
        chunk = cw[i:i + 3]
        for ai in range(len(chunk)):
            aw = chunk[ai]
            st = min(aw["start"], max(0.0, dur - 0.1))
            if ai + 1 < len(chunk):
                en = chunk[ai + 1]["start"]
            else:
                en = aw["end"]
            en = min(en, dur)
            if en <= st:
                en = min(dur, st + 0.12)
            toks = []
            for j in range(len(chunk)):
                txt = chunk[j]["word"].replace("{", "").replace("}", "")
                if j == ai:
                    pop = 128 if txt in emphasis else 112
                    toks.append("{\\c" + primary + "\\t(0,90,\\fscx%d\\fscy%d)\\t(90,200,\\fscx100\\fscy100)}%s{\\r}" % (pop, pop, txt))
                elif txt in emphasis:
                    toks.append("{\\c" + primary + "\\fscx115\\fscy115}" + txt + "{\\r}")
                else:
                    toks.append("{\\c" + secondary + "}" + txt)
            events.append("Dialogue: 0,%s,%s,KineticCaption,,0,0,0,,%s" % (
                ass_time(st), ass_time(en), " ".join(toks)))
    with open(path, "w", encoding="utf-8") as f:
        f.write(header + "\n".join(events) + "\n")


def render(cut, final, ass, sfx, mode, peaks_rel, dur, detector, priors, emojis):
    cap = cv2.VideoCapture(cut)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = 2 if src_fps > 32 else 1
    fps = src_fps / step
    sw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    sh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    kind, det = detector
    cmd = ["ffmpeg", "-nostdin", "-y",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-s",
           str(OUT_W) + "x" + str(OUT_H), "-r", str(fps), "-i", "pipe:0",
           "-i", cut, "-i", sfx]
    for _, p in emojis:
        cmd = cmd + ["-loop", "1", "-i", p]
    fc = ["[0:v]ass=" + ass + ":fontsdir=fonts[base]"]
    vlabel = "base"
    for i in range(len(emojis)):
        t = emojis[i][0]
        fc.append("[%d:v]format=rgba,scale=96:96,fade=t=in:st=%.2f:d=0.15:alpha=1,"
                  "fade=t=out:st=%.2f:d=0.2:alpha=1[e%d]" % (3 + i, t, t + 1.6, i))
        fc.append("[%s][e%d]overlay=x=W-170:y=1520:enable='between(t,%.2f,%.2f)'[v%d]" % (
            vlabel, i, t, t + 1.8, i))
        vlabel = "v%d" % i
    fc.append("[2:a]volume=0.6[sfxraw]")
    fc.append("[sfxraw][1:a]sidechaincompress=threshold=0.06:ratio=10:attack=15:release=250[sfxd]")
    fc.append("[1:a][sfxd]amix=inputs=2:duration=first:normalize=0,loudnorm=I=-14:TP=-1.5:LRA=11[aout]")
    cmd = cmd + ["-filter_complex", ";".join(fc),
                 "-map", "[" + vlabel + "]", "-map", "[aout]",
                 "-c:v", "libx264", "-crf", "18", "-preset", "veryfast",
                 "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                 "-c:a", "aac", "-b:a", "192k",
                 "-t", "%.3f" % dur, "-fs", "300M", final]
    logp = WORK + "/ffmpeg_" + os.path.basename(final) + ".log"
    logf = open(logp, "wb")
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=logf)

    smooth_cx = priors.get("single", 0.5)
    left_cx = priors.get("left", 0.28)
    right_cx = priors.get("right", 0.72)
    t0 = time.time()
    idx = 0
    out_i = 0
    try:
        while cap.grab():
            if idx % step:
                idx = idx + 1
                continue
            ret, frame = cap.retrieve()
            idx = idx + 1
            if not ret:
                break
            t = out_i / fps
            zoom = 1.0
            shake = 0.0
            for pk in peaks_rel:
                dt = t - pk
                if 0 <= dt <= 0.5:
                    env = math.sin((dt / 0.5) * math.pi)
                    zoom = 1.0 + 0.14 * env
                    shake = env
                    break
            if out_i % 10 == 0 and mode in ("podcast_single", "podcast_dual"):
                cs = faces(kind, det, frame)
                if cs:
                    if mode == "podcast_single":
                        smooth_cx = 0.85 * smooth_cx + 0.15 * cs[0][0]
                    else:
                        lf = [c[0] for c in cs if c[0] < 0.5]
                        rf = [c[0] for c in cs if c[0] >= 0.5]
                        if lf:
                            left_cx = 0.85 * left_cx + 0.15 * lf[0]
                        if rf:
                            right_cx = 0.85 * right_cx + 0.15 * rf[0]
            dx = int(3 * math.sin(t * 47) * shake)
            dy = int(2 * math.cos(t * 53) * shake)

            if mode == "podcast_dual":
                half = OUT_H // 2
                cw = min(sw, max(64, int(sh * (OUT_W / half) / zoom)))
                ch = min(sh, max(64, int(sh / zoom)))
                y1 = max(0, (sh - ch) // 2)
                xt = int(np.clip(left_cx * sw - cw / 2, 0, max(0, sw - cw))) + dx
                xb = int(np.clip(right_cx * sw - cw / 2, 0, max(0, sw - cw))) + dx
                xt = int(np.clip(xt, 0, max(0, sw - cw)))
                xb = int(np.clip(xb, 0, max(0, sw - cw)))
                top = cv2.resize(frame[y1:y1 + ch, xt:xt + cw], (OUT_W, half),
                                 interpolation=cv2.INTER_CUBIC)
                bot = cv2.resize(frame[y1:y1 + ch, xb:xb + cw], (OUT_W, half),
                                 interpolation=cv2.INTER_CUBIC)
                canvas = np.vstack([top, bot])
            elif mode == "gaming_action":
                small = cv2.resize(frame, (270, 480), interpolation=cv2.INTER_AREA)
                small = cv2.GaussianBlur(small, (15, 15), 0)
                bg = cv2.resize(small, (OUT_W, OUT_H), interpolation=cv2.INTER_LINEAR)
                fg_h = int(OUT_W * (sh / sw))
                zfg = cv2.resize(frame, (int(OUT_W * zoom), int(fg_h * zoom)),
                                 interpolation=cv2.INTER_LINEAR)
                zh, zw = zfg.shape[:2]
                x0 = max(0, (zw - OUT_W) // 2 + dx)
                y0 = max(0, (zh - fg_h) // 2 + dy)
                fg = zfg[y0:y0 + fg_h, x0:x0 + OUT_W]
                if fg.shape[0] != fg_h or fg.shape[1] != OUT_W:
                    fg = cv2.resize(fg, (OUT_W, fg_h))
                yo = (OUT_H - fg_h) // 2
                canvas = bg
                canvas[yo:yo + fg_h, 0:OUT_W] = fg
            else:
                ch = min(sh, max(64, int(sh / zoom)))
                cw = min(sw, max(64, int(ch * (OUT_W / OUT_H))))
                x1 = int(np.clip(smooth_cx * sw - cw / 2, 0, max(0, sw - cw))) + dx
                y1 = max(0, (sh - ch) // 2) + dy
                x1 = int(np.clip(x1, 0, max(0, sw - cw)))
                y1 = int(np.clip(y1, 0, max(0, sh - ch)))
                canvas = cv2.resize(frame[y1:y1 + ch, x1:x1 + cw], (OUT_W, OUT_H),
                                    interpolation=cv2.INTER_CUBIC)
            proc.stdin.write(canvas.tobytes())
            out_i = out_i + 1
            if out_i % 300 == 0:
                el = time.time() - t0
                print("  render: %d frames | %.1f fps effective | %.0fs elapsed" % (
                    out_i, out_i / el, el), flush=True)
    except BrokenPipeError:
        print("ffmpeg closed pipe early (see log)", flush=True)
    finally:
        cap.release()
        try:
            proc.stdin.close()
        except Exception:
            pass
        rc = proc.wait()
        logf.close()
    if rc != 0:
        tail = open(logp, "rb").read()[-2000:].decode(errors="replace")
        raise RuntimeError("ffmpeg render failed rc=" + str(rc) + "\n" + tail)
    if out_i == 0:
        raise RuntimeError("render produced zero frames")


def main():
    os.makedirs("final_reels", exist_ok=True)
    bp = load_json("blueprint.json")
    if not bp.get("clips"):
        raise RuntimeError("blueprint has no clips")
    words = load_json("words.json")
    gaps = load_json("gaps.json")
    notes = load_json("visual_notes.json")
    master = WORK + "/master.mp4"
    detector = make_detector()

    for idx, clip in enumerate(bp["clips"], 1):
        s = float(clip["start_time"])
        e = float(clip["end_time"])
        mode = clip["content_mode"]
        print("=== Reel %d [%s] %s (%.1f-%.1f) ===" % (
            idx, mode, clip["title"], s, e), flush=True)
        raw = WORK + "/raw_" + str(idx) + ".mp4"
        cut = WORK + "/cut_" + str(idx) + ".mp4"
        run(["ffmpeg", "-nostdin", "-y", "-ss", str(s), "-i", master,
             "-t", str(e - s), "-c:v", "libx264", "-crf", "16",
             "-preset", "ultrafast", "-c:a", "aac", "-b:a", "192k", raw])
        if TIGHTEN:
            segs = keep_segments(gaps, s, e)
        else:
            segs = [(0.0, e - s)]
        removed = (e - s) - sum(b - a for a, b in segs)
        if TIGHTEN and removed > 0.4:
            if not tighten_cut(raw, cut, segs):
                segs = [(0.0, e - s)]
        else:
            run(["ffmpeg", "-nostdin", "-y", "-i", raw, "-c", "copy", cut])
            segs = [(0.0, e - s)]
        remap, dur = make_remap(segs)
        if dur <= 0:
            raise RuntimeError("reel %d zero duration after tightening" % idx)
        peaks_rel = [remap(p - s) for p in clip.get("peak_moments", []) if s <= p <= e]

        priors = {}
        tl = notes.get("timeline", [])
        if tl:
            mid = (s + e) / 2
            near = min(tl, key=lambda n: abs(n["t"] - mid))
            sp = sorted(near.get("speakers", []), key=lambda b: b["x"])
            if len(sp) >= 2:
                priors = {"left": sp[0]["x"], "right": sp[-1]["x"]}
            elif len(sp) == 1:
                priors = {"single": sp[0]["x"]}

        sfx = WORK + "/sfx_" + str(idx) + ".wav"
        synth_sfx(sfx, dur, peaks_rel, clip.get("sfx", {}))
        ass = WORK + "/subs_" + str(idx) + ".ass"
        build_ass(ass, words, s, e, remap, dur, clip)
        emojis = []
        for j, cue in enumerate(clip.get("emojis", [])[:2]):
            p = WORK + "/emoji_" + str(idx) + "_" + str(j) + ".png"
            if emoji_png(cue["emoji"], p):
                emojis.append((remap(cue["t"] - s), p))
        final = "final_reels/reel_" + str(idx) + "_" + mode + ".mp4"
        try:
            render(cut, final, ass, sfx, mode, peaks_rel, dur, detector, priors, emojis)
        except Exception:
            if os.path.exists(final):
                os.remove(f
