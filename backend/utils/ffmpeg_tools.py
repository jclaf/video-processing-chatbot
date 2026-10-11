"""Outils FFmpeg / ffprobe : contrôle qualité (QA) des clips et assemblage final.

- probe_video / qa_check : ffprobe -> clip lisible ? flux vidéo ? durée cohérente ?
- assemble               : normalise chaque clip (taille, fps, audio), ajoute la voix off si fournie,
                           puis concatène (coupes franches) ou enchaîne avec des transitions (xfade).
"""
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import httpx

from utils.media import MEDIA_DIR

FFMPEG_TIMEOUT = float(os.getenv("ASSEMBLY_TIMEOUT_S", "600"))
OUTPUT_FPS = int(os.getenv("OUTPUT_FPS", "30"))
TRANSITIONS = ("cut", "fade", "fadeblack", "dissolve", "wipeleft", "slideleft")   # noms xfade (hors « cut »)


def have_ffmpeg() -> bool:
    return bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


# ---------- fichiers locaux ----------
def local_path(url: str | None) -> Path | None:
    """Fichier local d'une URL /media/... servie par ce back (None sinon)."""
    if url and "/media/" in url:
        p = MEDIA_DIR / url.rsplit("/media/", 1)[-1]
        return p if p.exists() else None
    return None


def ensure_local(url: str | None) -> Path | None:
    """Chemin local d'un média ; les URL distantes (ex. médias d'exemple du mode mock) sont téléchargées une fois."""
    p = local_path(url)
    if p:
        return p
    if url and url.startswith(("http://", "https://")):
        ext = Path(url.split("?")[0]).suffix or ".bin"
        p = MEDIA_DIR / f"dl_{hashlib.sha1(url.encode()).hexdigest()[:12]}{ext}"
        if p.exists():
            return p
        try:
            r = httpx.get(url, follow_redirects=True, timeout=60)
            r.raise_for_status()
            p.write_bytes(r.content)
            return p
        except httpx.HTTPError:
            return None
    return None


def _run(cmd: list[str]) -> None:
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=FFMPEG_TIMEOUT)
    if r.returncode != 0:
        raise RuntimeError((r.stderr or "erreur ffmpeg").strip()[-400:])


# ---------- QA ----------
def probe_video(path: str | Path) -> dict | None:
    """Infos ffprobe d'un fichier, ou None s'il est illisible."""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
            capture_output=True, text=True, timeout=30,
        )
        if r.returncode != 0:
            return None
        data = json.loads(r.stdout)
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return None
    streams = data.get("streams", [])
    v = next((s for s in streams if s.get("codec_type") == "video"), None)
    a = next((s for s in streams if s.get("codec_type") == "audio"), None)
    fmt = data.get("format", {})
    return {
        "has_video": v is not None,
        "has_audio": a is not None,
        "duration": float(fmt.get("duration") or (v or {}).get("duration") or 0),
        "width": int((v or {}).get("width") or 0),
        "height": int((v or {}).get("height") or 0),
    }


def qa_check(path: str | Path | None, expected_duration: float | None = None,
             tolerance: float = 1.5, strict_duration: bool = False) -> dict:
    """Contrôle d'un clip. Retourne {"ok": clip exploitable, "issues": [...], "warnings": [...], "info": {...}}.

    Bloquants (issues) : fichier absent / illisible, pas de flux vidéo, durée quasi nulle.
    Durée différente de celle demandée : avertissement, bloquant seulement si strict_duration.
    """
    if not path or not Path(path).exists():
        return {"ok": False, "issues": ["fichier introuvable"], "warnings": [], "info": {}}
    info = probe_video(path)
    if info is None:
        return {"ok": False, "issues": ["fichier illisible (ffprobe)"], "warnings": [], "info": {}}
    issues, warnings = [], []
    if not info["has_video"]:
        issues.append("aucun flux vidéo")
    if info["duration"] < 0.5:
        issues.append(f"durée nulle ou trop courte ({info['duration']:.1f} s)")
    if expected_duration and abs(info["duration"] - expected_duration) > tolerance:
        (issues if strict_duration else warnings).append(
            f"durée {info['duration']:.1f} s au lieu de {expected_duration:g} s")
    return {"ok": not issues, "issues": issues, "warnings": warnings, "info": info}


# ---------- assemblage ----------
_AFMT = "aresample=44100,aformat=sample_fmts=fltp:channel_layouts=stereo"


def normalize_scene(src: Path, dst: Path, size: tuple[int, int], fps: int,
                    narration: Path | None = None, duck: float = 0.25) -> list[str]:
    """Ré-encode un clip aux paramètres communs (taille, fps, audio stéréo 44,1 kHz) ; ajoute la voix off.

    Le son d'origine passe à `duck` (0-1) sous la voix off. Retourne la liste des avertissements.
    """
    info = probe_video(src) or {}
    d = info.get("duration") or 0
    if d <= 0:
        raise RuntimeError(f"clip illisible : {src.name}")
    w, h = size
    warnings: list[str] = []

    cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(src)]
    nxt = 1
    if info.get("has_audio"):
        base = "0:a"
    else:   # pas de piste audio : on en crée une silencieuse pour que tous les clips soient identiques
        cmd += ["-f", "lavfi", "-t", f"{d:.3f}", "-i", "anullsrc=r=44100:cl=stereo"]
        base, nxt = f"{nxt}:a", nxt + 1

    fc = [f"[0:v]scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,"
          f"setsar=1,fps={fps},format=yuv420p[v]"]
    if narration:
        nd = (probe_video(narration) or {}).get("duration") or 0
        cmd += ["-i", str(narration)]
        tempo = ""
        if nd > d:
            f = min(1.35, nd / d)
            tempo = f",atempo={f:.4f}"
            warnings.append(
                f"voix off de {nd:.1f} s trop longue pour une scène de {d:.1f} s : " +
                ("accélérée" if nd / f <= d + 0.05 else "accélérée puis coupée"))
        fc.append(f"[{base}]{_AFMT},volume={duck}[b]")
        fc.append(f"[{nxt}:a]{_AFMT}{tempo},apad[n]")
        fc.append("[b][n]amix=inputs=2:duration=first:dropout_transition=0,volume=2[a]")
    else:
        fc.append(f"[{base}]{_AFMT}[a]")

    cmd += ["-filter_complex", ";".join(fc), "-map", "[v]", "-map", "[a]", "-t", f"{d:.3f}",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "160k", "-ar", "44100", "-ac", "2", "-movflags", "+faststart", str(dst)]
    _run(cmd)
    return warnings


def _concat_copy(clips: list[Path], dst: Path, tmp: str) -> None:
    listing = Path(tmp) / "list.txt"
    listing.write_text("".join(f"file '{str(c).replace(chr(39), chr(39) + chr(92) + chr(39) + chr(39))}'\n" for c in clips))
    _run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(listing),
          "-c", "copy", "-movflags", "+faststart", str(dst)])


def _xfade(clips: list[Path], durs: list[float], transitions: list[str], td: float, dst: Path) -> None:
    """Enchaîne les clips avec des transitions (vidéo xfade + audio acrossfade). Pas de « cut » ici."""
    cmd = ["ffmpeg", "-y", "-v", "error"]
    for c in clips:
        cmd += ["-i", str(c)]
    fc, v_prev, a_prev, length = [], "0:v", "0:a", durs[0]
    for k in range(1, len(clips)):
        dk = min(td, 0.5 * min(durs[k - 1], durs[k]))
        fc.append(f"[{v_prev}][{k}:v]xfade=transition={transitions[k - 1]}:duration={dk:.3f}:offset={max(length - dk, 0):.3f}[v{k}]")
        fc.append(f"[{a_prev}][{k}:a]acrossfade=d={dk:.3f}[a{k}]")
        length += durs[k] - dk
        v_prev, a_prev = f"v{k}", f"a{k}"
    cmd += ["-filter_complex", ";".join(fc), "-map", f"[{v_prev}]", "-map", f"[{a_prev}]",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", str(dst)]
    _run(cmd)


def assemble(scenes: list[dict], transitions: list[str], td: float, duck: float, dst: Path,
             fps: int = OUTPUT_FPS) -> list[str]:
    """Assemble les scènes (dans l'ordre) en un MP4. scenes = [{"path": Path, "narration": Path | None}].

    transitions : un par raccord (len(scenes) - 1), parmi TRANSITIONS. Retourne les avertissements.
    """
    if not scenes:
        raise RuntimeError("aucune scène à assembler")
    first = probe_video(scenes[0]["path"]) or {}
    w, h = (first.get("width") or 1280), (first.get("height") or 720)
    size = (w - w % 2, h - h % 2)
    warnings: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        norm: list[Path] = []
        for i, sc in enumerate(scenes):
            out = Path(tmp) / f"n{i}.mp4"
            warnings += normalize_scene(Path(sc["path"]), out, size, fps, sc.get("narration"), duck)
            norm.append(out)
        if len(norm) == 1:
            shutil.copy(norm[0], dst)
        elif all(t == "cut" for t in transitions[: len(norm) - 1]):
            _concat_copy(norm, dst, tmp)
        else:
            # Les « cut » consécutifs sont d'abord fusionnés en segments (vraies coupes franches, sans réencodage) ;
            # les transitions s'appliquent ensuite entre segments.
            groups, seg_trans = [[norm[0]]], []
            for k, t in enumerate(transitions[: len(norm) - 1]):
                if t == "cut":
                    groups[-1].append(norm[k + 1])
                else:
                    groups.append([norm[k + 1]])
                    seg_trans.append(t)
            segments = []
            for j, group in enumerate(groups):
                if len(group) == 1:
                    segments.append(group[0])
                else:
                    seg = Path(tmp) / f"seg{j}.mp4"
                    _concat_copy(group, seg, tmp)
                    segments.append(seg)
            durs = [(probe_video(p) or {}).get("duration") or 0 for p in segments]
            _xfade(segments, durs, seg_trans, td, dst)
    return warnings
