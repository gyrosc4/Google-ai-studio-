"""Step 2: ERNIE 5.5 Flash (free, OpenRouter) as showrunner.
Transcript + visual notes + audio energy -> grounded blueprint -> critic pass."""
import os
import json
from typing import Literal, Optional
from pydantic import BaseModel, Field, ValidationError
from common import (load_json, save_json, extract_json, openrouter_chat,
                    ffprobe_duration, WORK)

NUM_REELS = int(os.environ.get("NUM_REELS", "3"))
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
    title: str = Field(description="3-5 word viral hook banner")
    start_time: float
    end_time: float
    content_mode: Literal["podcast_dual", "podcast_single", "gaming_action", "explainer"]
    caption_theme: Literal["hype_gaming", "editorial_podcast", "bold_explainer"]
    virality_score: int
    peak_moments: list[float] = []
    emphasis_words: list[str] = Field(default=[], description="Words to accent in captions, uppercase")
    hook_style: Literal["pop", "slide", "shake"] = "pop"
    lower_third: str = Field(default="", description="Speaker name for animated lower-third, or empty")
    emojis: list[EmojiCue] = Field(default=[], description="Max 2 emoji stickers with absolute timestamps")
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
    best, bd = t, window
    for c in candidates:
        d = abs(c - t)
        if d < bd:
            best, bd = c, d
    return best


def ground(bp: Blueprint, duration, words, gaps, energy_peaks):
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
            if not (c.start_time <= p <= c.end_time):
                continue
            near = [e for e in energy_peaks if abs(e - p) <= 0.75]
            peaks.append(near[0] if near else p)
        if not peaks:
            peaks = [e for e in energy_peaks if c.start_time + 2 <= e <= c.end_time - 2][:3]
        c.peak_moments = sorted(set(peaks))[:4]
        c.emojis = [e for e in c.emojis if c.start_time <= e.t <= c.end_time][:2]
        out.append(c)
    out.sort(key=lambda c: -c.virality_score)
    kept = []
    for c in out:
        overlap = False
        for k in kept:
            lo = max(c.start_time, k.start_time); hi = min(c.end_time, k.end_time)
            if hi - lo > 0.3 * min(c.end_time - c.start_time, k.end_time - k.start_time):
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
    transcript = open(f"{WORK}/transcript.txt", encoding="utf-8").read()

    visual_lines = "\n".join(
        f"[{n['t']}s] {n['layout']} speakers={n.get('speakers', [])} text={n.get('on_screen_text','')} {n.get('note','')}"
        for n in notes.get("timeline", []))
    hype_lines = "\n".join(f"[{h['t']}s] {h['kind']} {h.get('note','')}"
                           for h in notes.get("hype_moments", []))

    director_prompt = f"""You are an elite short-form showrunner (Hormozi-level clipper).
Pick the {NUM_REELS} BEST standalone 30-60s segments for Instagram Reels from this video.
You have the exact word-timestamped transcript, visual notes from a vision model that WATCHED
the video, and audio-energy peaks (loud/hype moments). Ground every decision in them.

RULES:
- Start exactly where a hook sentence begins; end right after the payoff. Never mid-sentence.
- Prefer controversial takes, emotional stories, insane plays, funny fails, sharp insights.
- content_mode from the visual timeline layout; caption_theme to match genre.
- peak_moments MUST come from the audio-energy peak list or hype_moments (zoom punches).
- emphasis_words: 2-5 punchy words per clip that captions should accent (uppercase).
- title: 3-5 word irresistible hook. lower_third: speaker name if known.
- emojis: max 2 per clip, absolute timestamp where the emotion lands.
- virality_score 1-100 honest; reason: one sentence why this clip retains viewers.
- Clips must not overlap each other.

AUDIO ENERGY PEAKS (seconds): {json.dumps(energy['peaks'][:200])}
HYPE MOMENTS (seen/heard): {hype_lines or 'none'}
VISUAL TIMELINE: {visual_lines or 'none'}
GENRE: {notes.get('overall_genre','unknown')} — {notes.get('summary','')}

TRANSCRIPT:
{transcript}"""

    schema = Blueprint.model_json_schema()
    bp = None
    last_good = None
    feedback = ""
    for iteration in range(3):
        prompt = director_prompt + (
            f"\nPREVIOUS PICKS REJECTED BY CRITIC, fix these issues and pick better/alternative segments:\n{feedback}"
            if feedback else "")
        try:
            txt = openrouter_chat(BRAIN_POOL, [{"role": "user", "content": prompt}],
                                  temperature=0.35, schema_hint=schema)
            bp = Blueprint.model_validate(extract_json(txt))
        except Exception as e:
            print(f"⚠️ director attempt failed ({str(e)[:200]})", flush=True)
            if last_good is not None:
                print("↩️ re-pick failed; keeping last grounded blueprint", flush=True)
                bp = last_good
                break
            feedback = f"Previous attempt errored: {str(e)[:300]}. Return ONLY valid JSON."
            bp = None
            continue
        bp = ground(bp, duration, words, gaps, energy["peaks"])
        if not bp.clips:
            feedback = "No valid clips survived grounding; pick different segments."
            bp = None
            continue
        last_good = bp

        # ---- CRITIC PASS ----
        critic_payload = "\n\n".join(
            f"CLIP {i} [{c.start_time:.1f}-{c.end_time:.1f}] hook='{c.title}'\n{clip_transcript(words, c.start_time, c.end_time)}"
            for i, c in enumerate(bp.clips))
        try:
            ctxt = openrouter_chat(BRAIN_POOL, [{"role": "user", "content":
                f"You are a ruthless retention critic. For each clip judge: does the hook land in the "
                f"first 3 seconds, is the payoff complete inside the clip, does it stand alone without "
                f"context? Return items with index, keep (bool), hook_strength 1-10, payoff_complete, "
                f"standalone, issue.\n\n{critic_payload}"}],
                temperature=0.1, schema_hint=CriticReport.model_json_schema())
            report = CriticReport.model_validate(extract_json(ctxt))
        except Exception as e:
            print(f"⚠️ critic failed ({str(e)[:160]}); accepting grounded blueprint", flush=True)
            break
        bad = [it for it in report.items
               if (not it.keep) or (not it.payoff_complete) or (not it.standalone)
               or it.hook_strength < 5]
        if not bad or iteration == 2:
            print("✅ critic approved (or final iteration); locking blueprint", flush=True)
            break
        feedback = "\n".join(f"clip {it.index}: {it.issue or 'weak hook/incomplete payoff'}" for it in bad)
        print(f"🔁 critic rejected {len(bad)} clips, re-picking (iter {iteration+1})", flush=True)
        bp = None

    if bp is None or not bp.clips:
        raise RuntimeError("Showrunner could not produce a valid blueprint")
    save_json("blueprint.json", bp.model_dump())
    print("Blueprint:\n", json.dumps(bp.model_dump(), indent=2)[:2500], flush=True)


if __name__ == "__main__":
    main()
