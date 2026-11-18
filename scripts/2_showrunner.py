"""Step 2: ERNIE 5.5 Flash (free, OpenRouter) showrunner + one critic re-pick max.
Grounding snaps cuts to sentence/silence boundaries, validates peaks against
audio energy, dedupes overlaps, sorts by virality. Never dies on a bad response."""
import os
import json
from typing import Literal
from pydantic import BaseModel, Field, ValidationError
from common import load_json, save_json, extract_json, openrouter_chat, WORK

NUM_REELS = max(1, min(5, int(os.environ.get("NUM_REELS", "3"))))
BRAIN_POOL = ["baidu/ernie-5.5-flash-preview:free",
              "inclusionai/ling-3.2-flash:free",
              "nvidia/nemotron-3-ultra-550b-a55b:free"]


class EmojiCue(BaseModel):
    t: float
    emoji: str


class SfxCues(BaseModel):
    whoosh: bool = True
    riser: bool = True


class ClipDirective(BaseModel):
    title: str
    start_time: float
    end_time: float
    content_mode: Literal["podcast_dual", "podcast_single", "gaming_action", "explainer"]
    caption_theme: Literal["hype_gaming", "editorial_podcast", "bold_explainer"]
    virality_score: int
    peak_moments: list[float] = []
    emphasis_words: list[str] = []
    hook_style: Literal["pop", "slide", "shake"] = "pop"
    lower_third: str = ""
    emojis: list[EmojiCue] = []
    sfx: SfxCues = SfxCues()
    reason: str = ""


class Blueprint(BaseModel):
    overall_genre: str
    clips: list[ClipDirective]


class CriticItem(BaseModel):
    index: int
    keep: bool
    hook_strength: int
    payoff_complete: bool
    standalone: bool
    issue: str = ""


class CriticReport(BaseModel):
    items: list[CriticItem]


def snap(t, candidates, window):
    best = t
    bd = window
    for c in candidates:
        d = abs(c - t)
        if d < bd:
            best = c
            bd = d
    return best


def ground(bp, duration, words, gaps, energy_peaks):
    starts = [w["start"] for w in words] + [g[1] for g in gaps]
    ends = [w["end"] for w in words] + [g[0] for g in gaps]
    out = []
    for c in bp.clips:
        c.start_time = max(0.0, min(c.start_time, duration - 5))
        c.end_time = max(c.start_time + 20, min(c.end_time, duration))
        c.start_time = snap(c.start_time, starts, 1.5)
        c.end_time = snap(c.end_time, ends, 2.0)
        if c.end_time - c.start_time < 20:
            c.end_time = min(duration, c.start_time + 35)
        if c.end_time - c.start_time > 75:
            c.end_time = c.start_time + 60
        peaks = []
        for p in c.peak_moments:
            if p < c.start_time or p > c.end_time:
                continue
            near = [e for e in energy_peaks if abs(e - p) <= 0.75]
            if near:
                peaks.append(near[0])
            else:
                peaks.append(p)
        if not peaks:
            peaks = [e for e in energy_peaks
                     if c.start_time + 2 <= e <= c.end_time - 2][:3]
        c.peak_moments = sorted(set(peaks))[:4]
        c.emojis = [e for e in c.emojis if c.start_time <= e.t <= c.end_time][:2]
        out.append(c)
    out.sort(key=lambda c: -c.virality_score)
    kept = []
    for c in out:
        overlap = False
        for k in kept:
            lo = max(c.start_time, k.start_time)
            hi = min(c.end_time, k.end_time)
            shorter = min(c.end_time - c.start_time, k.end_time - k.start_time)
            if hi - lo > 0.3 * shorter:
                overlap = True
        if not overlap:
            kept.append(c)
        if len(kept) >= NUM_REELS:
            break
    bp.clips = kept
    return bp


def clip_transcript(words, s, e):
    return " ".join(w["word"] for w in words if w["end"] > s and w["start"] < e)


def main():
    words = load_json("words.json")
    gaps = load_json("gaps.json")
    energy = load_json("energy.json")
    notes = load_json("visual_notes.json")
    duration = load_json("duration.json")["duration"]
    transcript = open(WORK + "/transcript.txt", encoding="utf-8").read()

    visual_lines = []
    for n in notes.get("timeline", []):
        visual_lines.append("[%ss] %s speakers=%s text=%s %s" % (
            n["t"], n["layout"], n.get("speakers", []),
            n.get("on_screen_text", ""), n.get("note", "")))
    hype_lines = []
    for h in notes.get("hype_moments", []):
        hype_lines.append("[%ss] %s %s" % (h["t"], h["kind"], h.get("note", "")))

    director_prompt = (
        "You are an elite short-form showrunner (Hormozi-level clipper).\n"
        "Pick the " + str(NUM_REELS) + " BEST standalone 30-60s segments for Instagram Reels from this video.\n"
        "You have the exact word-timestamped transcript, visual notes from a vision model that WATCHED\n"
        "the video, and audio-energy peaks (loud/hype moments). Ground every decision in them.\n\n"
        "RULES:\n"
        "- Start exactly where a hook sentence begins; end right after the payoff. Never mid-sentence.\n"
        "- Prefer controversial takes, emotional stories, insane plays, funny fails, sharp insights.\n"
        "- content_mode from the visual timeline layout; caption_theme to match genre.\n"
        "- peak_moments MUST come from the audio-energy peak list or hype_moments (zoom punches).\n"
        "- emphasis_words: 2-5 punchy words per clip that captions should accent (uppercase).\n"
        "- title: 3-5 word irresistible hook. lower_third: speaker name if known.\n"
        "- emojis: max 2 per clip, absolute timestamp where the emotion lands.\n"
        "- virality_score 1-100 honest; reason: one sentence why this clip retains viewers.\n"
        "- Clips must not overlap each other.\n\n"
        "AUDIO ENERGY PEAKS (seconds): " + json.dumps(energy["peaks"][:200]) + "\n"
        "HYPE MOMENTS (seen/heard): " + ("\n".join(hype_lines) or "none") + "\n"
        "VISUAL TIMELINE: " + ("\n".join(visual_lines) or "none") + "\n"
        "GENRE: " + notes.get("overall_genre", "unknown") + " - " + notes.get("summary", "") + "\n\n"
        "TRANSCRIPT:\n" + transcript)

    schema = Blueprint.model_json_schema()
    bp = None
    last_good = None
    feedback = ""
    for iteration in range(2):
        prompt = director_prompt
        if feedback:
            prompt = prompt + ("\nPREVIOUS PICKS REJECTED BY CRITIC, fix these issues and pick "
                               "better/alternative segments:\n" + feedback)
        try:
            txt = openrouter_chat(BRAIN_POOL, [{"role": "user", "content": prompt}],
                                  temperature=0.35, max_tokens=8000, schema_hint=schema)
            bp = Blueprint.model_validate(extract_json(txt))
        except Exception as e:
            print("director attempt failed: " + str(e)[:200], flush=True)
            if last_good is not None:
                print("re-pick failed; keeping last grounded blueprint", flush=True)
                bp = last_good
                break
            feedback = "Previous attempt errored: " + str(e)[:300] + ". Return ONLY valid JSON."
            bp = None
            continue
        bp = ground(bp, duration, words, gaps, energy["peaks"])
        if not bp.clips:
            feedback = "No valid clips survived grounding; pick different segments."
            bp = None
            continue
        last_good = bp

        critic_payload = []
        for i, c in enumerate(bp.clips):
            critic_payload.append("CLIP %d [%.1f-%.1f] hook='%s'\n%s" % (
                i, c.start_time, c.end_time, c.title,
                clip_transcript(words, c.start_time, c.end_time)))
        critic_msg = ("You are a ruthless retention critic. For each clip judge: does the hook land in "
                      "the first 3 seconds, is the payoff complete inside the clip, does it stand alone "
                      "without context? Return items with index, keep (bool), hook_strength 1-10, "
                      "payoff_complete, standalone, issue.\n\n" + "\n\n".join(critic_payload))
        try:
            ctxt = openrouter_chat(BRAIN_POOL, [{"role": "user", "content": critic_msg}],
                                   temperature=0.1, max_tokens=4000,
                                   schema_hint=CriticReport.model_json_schema())
            report = CriticReport.model_validate(extract_json(ctxt))
        except Exception as e:
            print("critic failed, accepting grounded blueprint: " + str(e)[:160], flush=True)
            break
        bad = []
        for it in report.items:
            if (not it.keep) or (not it.payoff_complete) or (not it.standalone):
                bad.append(it)
        if not bad or iteration == 1:
            print("critic approved (or final iteration); locking blueprint", flush=True)
            break
        feedback = "\n".join("clip %d: %s" % (it.index, it.issue or "weak hook/incomplete payoff")
                             for it in bad)
        print("critic rejected " + str(len(bad)) + " clips, re-picking once", flush=True)
        bp = None

    if bp is None or not bp.clips:
        raise RuntimeError("Showrunner could not produce a valid blueprint")
    save_json("blueprint.json", bp.model_dump())
    print("Blueprint:\n" + json.dumps(bp.model_dump(), indent=2)[:2500], flush=True)


if __name__ == "__main__":
    main()
