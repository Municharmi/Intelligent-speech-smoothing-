"""
data_pipeline/step1_build_dataset.py
======================================
Builds training dataset from 3 sources:
  1. Your student recordings + Excel transcriptions
  2. UASpeech dataset (CM/CF = mild control, M/F = moderate/severe dysarthric)
  3. YouTube speech therapy videos

UASpeech speaker severity (from original paper):
  Control speakers (CM*, CF*) → mild  (normal speech, reference)
  Dysarthric speakers:
    M04, M12, F03, F04         → mild      (intelligibility >75%)
    M05, M09, M10, M14, F01   → moderate  (intelligibility 25–75%)
    M01, M02, M03, M06, M07,
    M08, M11, M16, F02, F05   → severe    (intelligibility <25%)

Run from speech_project_complete folder:
  python data_pipeline/step1_build_dataset.py \
      --student_audio   recordings \
      --student_excel   Transcript.xlsx \
      --uaspeech_dir    "C:/Users/YourName/Downloads/UASpeech Dataset/UASpeech/audio/noisereduce" \
      --output          data/dataset.csv
"""

import os, sys, csv, argparse, shutil, subprocess, tempfile
from pathlib import Path
from collections import Counter

# ── UASpeech severity labels (from published intelligibility scores) ──
UASPEECH_SEVERITY = {
    # Control speakers → mild (normal reference speech)
    "CM01":"mild","CM04":"mild","CM05":"mild","CM06":"mild",
    "CM08":"mild","CM09":"mild","CM10":"mild","CM12":"mild","CM13":"mild",
    "CF02":"mild","CF03":"mild","CF04":"mild","CF05":"mild",

    # Dysarthric — mild (intelligibility >75%)
    "M04":"mild","M12":"mild","F03":"mild","F04":"mild",

    # Dysarthric — moderate (intelligibility 25–75%)
    "M05":"moderate","M09":"moderate","M10":"moderate","M14":"moderate","F01":"moderate",

    # Dysarthric — severe (intelligibility <25%)
    "M01":"severe","M02":"severe","M03":"severe","M06":"severe","M07":"severe",
    "M08":"severe","M11":"severe","M16":"severe","F02":"severe","F05":"severe",
}

# ── YouTube videos with speech therapy sessions ──────────────────────
YOUTUBE_VIDEOS = [
    {"url":"https://www.youtube.com/watch?v=0MgsQMnf6jg", "severity":"mild",     "desc":"Speech therapy mild ID"},
    {"url":"https://www.youtube.com/watch?v=EQy3fJCM9eo", "severity":"moderate", "desc":"AAC therapy moderate ID"},
    {"url":"https://www.youtube.com/watch?v=xpPiyBxiXoM", "severity":"moderate", "desc":"Down syndrome speech therapy"},
    {"url":"https://www.youtube.com/watch?v=2vT0Ov1e8dU", "severity":"moderate", "desc":"Autism communication therapy"},
    {"url":"https://www.youtube.com/watch?v=pVlMsOyJJVE", "severity":"severe",   "desc":"Severe ID AAC communication"},
    {"url":"https://www.youtube.com/watch?v=nzp6yILzF-Q", "severity":"moderate", "desc":"Dysarthria speech therapy"},
    {"url":"https://www.youtube.com/watch?v=KZ6YWJP2zLk", "severity":"mild",     "desc":"ID communication training"},
    {"url":"https://www.youtube.com/watch?v=4YOuRBGpPHc", "severity":"mild",     "desc":"Speech delay functional phrases"},
    {"url":"https://www.youtube.com/watch?v=pVlMsOyJJVE", "severity":"severe",   "desc":"Severe communication impairment"},
    {"url":"https://www.youtube.com/watch?v=TghgCHOeLk0", "severity":"moderate", "desc":"Intellectual disability speech"},
]


def preprocess_audio(src_path: str, dst_path: str) -> bool:
    """Convert any audio to 16kHz mono WAV. Returns True on success."""
    try:
        from pydub import AudioSegment
        from pydub.effects import normalize
        seg = AudioSegment.from_file(src_path)
        seg = seg.set_channels(1).set_frame_rate(16000)
        seg = normalize(seg)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        seg.export(dst_path, format="wav")
        return True
    except Exception as e:
        print(f"    SKIP {os.path.basename(src_path)}: {e}")
        return False


# ══════════════════════════════════════════════════════════════════════
# SOURCE 1 — Student recordings + Excel transcriptions
# ══════════════════════════════════════════════════════════════════════
def load_students(audio_dir: str, excel_path: str, out_audio_dir: Path) -> list:
    print("\n[1/3] Loading student recordings...")
    rows = []

    # Parse Excel
    trans_map = {}
    if excel_path and os.path.exists(excel_path):
        try:
            import openpyxl
            wb = openpyxl.load_workbook(excel_path, read_only=True)
            ws = wb.active
            data = list(ws.iter_rows(values_only=True))
            # Auto-detect columns
            for row in data:
                vals = [str(v).strip() if v is not None else "" for v in row]
                vals = [v for v in vals if v]
                if len(vals) >= 2:
                    fname = vals[0]
                    text  = vals[1]
                    if text.lower() not in ("transcription","transcript","text",""):
                        trans_map[fname.lower()] = text
                        # Store with extension variants
                        for ext in [".m4a.mp4",".mp4",".m4a",".mp3",".wav"]:
                            trans_map[(fname + ext).lower()] = text
        except ImportError:
            try:
                import pandas as pd
                df = pd.read_excel(excel_path, header=None)
                for _, row in df.iterrows():
                    vals = [str(v).strip() for v in row if str(v).strip() not in ("nan","")]
                    if len(vals) >= 2:
                        fname = vals[0]
                        text  = vals[1]
                        trans_map[fname.lower()] = text
                        for ext in [".m4a.mp4",".mp4",".m4a",".mp3",".wav"]:
                            trans_map[(fname+ext).lower()] = text
            except Exception as e:
                print(f"  WARNING: Could not read Excel: {e}")
        print(f"  Loaded {len(trans_map)//6} transcriptions from {excel_path}")
    else:
        print("  WARNING: No Excel file found — student transcriptions will be empty")

    # Process audio files
    if not os.path.exists(audio_dir):
        print(f"  WARNING: {audio_dir} not found")
        return rows

    audio_out = out_audio_dir / "student"
    audio_out.mkdir(parents=True, exist_ok=True)

    exts = {".mp4",".m4a",".mp3",".wav",".ogg",".flac",".webm",".aac"}
    files = [f for f in Path(audio_dir).iterdir() if f.suffix.lower() in exts]

    count = 0
    for fpath in sorted(files):
        stem = fpath.name
        # Look up transcript
        transcript = (trans_map.get(fpath.name.lower()) or
                      trans_map.get(fpath.stem.lower()) or "")

        out_wav = audio_out / f"{fpath.stem.replace(' ','_')}.wav"
        if preprocess_audio(str(fpath), str(out_wav)):
            rows.append({
                "source":     "student",
                "filename":   str(out_wav),
                "transcript": transcript,
                "severity":   "",   # to be labelled manually
                "speaker_id": fpath.stem,
                "duration_s": "",
            })
            count += 1

    print(f"  Added {count} student recordings")
    print(f"  Transcription match rate: {sum(1 for r in rows if r['transcript'])}/{count}")
    return rows


# ══════════════════════════════════════════════════════════════════════
# SOURCE 2 — UASpeech dataset
# ══════════════════════════════════════════════════════════════════════
def load_uaspeech(uaspeech_dir: str, out_audio_dir: Path,
                  max_per_speaker: int = 50) -> list:
    print("\n[2/3] Loading UASpeech dataset...")
    rows = []

    if not uaspeech_dir or not os.path.exists(uaspeech_dir):
        print(f"  UASpeech not found at: {uaspeech_dir}")
        print("  Set --uaspeech_dir to your UASpeech noisereduce folder path")
        return rows

    audio_out = out_audio_dir / "uaspeech"
    audio_out.mkdir(parents=True, exist_ok=True)

    ua_path = Path(uaspeech_dir)
    speakers_found = []

    for spk_dir in sorted(ua_path.iterdir()):
        if not spk_dir.is_dir():
            continue
        spk = spk_dir.name  # e.g. CM08, M04, F02
        severity = UASPEECH_SEVERITY.get(spk)
        if not severity:
            print(f"  SKIP unknown speaker: {spk}")
            continue

        speakers_found.append(spk)
        wav_files = sorted(spk_dir.glob("*.wav"))[:max_per_speaker]
        count = 0

        for wf in wav_files:
            # Parse word from filename: CM08_B1_CW6_M8.wav → CW6
            parts = wf.stem.split("_")
            word = ""
            for p in parts:
                if p.startswith("CW") or p.startswith("UW") or p.startswith("D") or p.startswith("LB") or p.startswith("LC") or p.startswith("LH"):
                    word = p
                    break

            out_wav = audio_out / f"{spk}_{wf.stem}.wav"
            if preprocess_audio(str(wf), str(out_wav)):
                rows.append({
                    "source":     "uaspeech",
                    "filename":   str(out_wav),
                    "transcript": word,
                    "severity":   severity,
                    "speaker_id": spk,
                    "duration_s": "",
                })
                count += 1

        print(f"  {spk:6s} [{severity:8s}]: {count} files")

    sev_counts = Counter(r["severity"] for r in rows)
    print(f"\n  Total UASpeech: {len(rows)} files from {len(speakers_found)} speakers")
    print(f"  By severity: mild={sev_counts.get('mild',0)}  moderate={sev_counts.get('moderate',0)}  severe={sev_counts.get('severe',0)}")
    return rows


# ══════════════════════════════════════════════════════════════════════
# SOURCE 3 — YouTube videos
# ══════════════════════════════════════════════════════════════════════
def load_youtube(out_audio_dir: Path, max_segments: int = 40) -> list:
    print("\n[3/3] Downloading YouTube speech therapy videos...")
    rows = []

    try:
        import yt_dlp
    except ImportError:
        print("  Installing yt-dlp...")
        subprocess.run([sys.executable,"-m","pip","install","yt-dlp","-q"], check=False)
        try:
            import yt_dlp
        except ImportError:
            print("  SKIP: Could not install yt-dlp")
            return rows

    from pydub import AudioSegment
    from pydub.effects import normalize

    yt_out = out_audio_dir / "youtube"
    yt_out.mkdir(parents=True, exist_ok=True)

    for i, vid in enumerate(YOUTUBE_VIDEOS):
        url      = vid["url"]
        severity = vid["severity"]
        desc     = vid["desc"]
        print(f"  [{i+1}/{len(YOUTUBE_VIDEOS)}] {desc[:45]} [{severity}]", end=" ... ", flush=True)

        tmp = Path(tempfile.mkdtemp())
        try:
            opts = {
                "format":         "bestaudio/best",
                "outtmpl":        str(tmp/"audio.%(ext)s"),
                "postprocessors": [{"key":"FFmpegExtractAudio","preferredcodec":"wav"}],
                "quiet":          True, "no_warnings":True,
                "ignoreerrors":   True, "socket_timeout":30,
            }
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=True)
            if not info:
                print("unavailable")
                continue

            wavs = list(tmp.glob("*.wav"))
            if not wavs:
                print("no audio")
                continue

            full = AudioSegment.from_wav(str(wavs[0]))
            full = full.set_channels(1).set_frame_rate(16000)
            full = normalize(full)

            vid_id  = info.get("id", f"yt{i}")
            seg_ms  = 8000
            n_segs  = len(full) // seg_ms
            count   = 0

            for si in range(min(n_segs, max_segments)):
                chunk = full[si*seg_ms:(si+1)*seg_ms]
                if chunk.rms < 80:
                    continue
                out = yt_out / f"{vid_id}_s{si:03d}.wav"
                chunk.export(str(out), format="wav")
                rows.append({
                    "source":     "youtube",
                    "filename":   str(out),
                    "transcript": "",
                    "severity":   severity,
                    "speaker_id": f"yt_{vid_id}",
                    "duration_s": "8",
                })
                count += 1
            print(f"{count} segments")

        except Exception as e:
            print(f"error: {str(e)[:50]}")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    sev = Counter(r["severity"] for r in rows)
    print(f"  Total YouTube: {len(rows)} segments  mild={sev.get('mild',0)}  moderate={sev.get('moderate',0)}  severe={sev.get('severe',0)}")
    return rows


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Build training dataset from all sources")
    parser.add_argument("--student_audio",   default="recordings",       help="Path to your student recordings folder")
    parser.add_argument("--student_excel",   default="Transcript.xlsx",  help="Path to your Excel transcription file")
    parser.add_argument("--uaspeech_dir",    default="",                 help="Path to UASpeech noisereduce folder")
    parser.add_argument("--output",          default="data/dataset.csv", help="Output CSV path")
    parser.add_argument("--skip_youtube",    action="store_true",        help="Skip YouTube download")
    parser.add_argument("--max_uaspeech",    type=int, default=50,       help="Max files per UASpeech speaker")
    parser.add_argument("--max_yt_segments", type=int, default=30,       help="Max segments per YouTube video")
    args = parser.parse_args()

    print("="*60)
    print("  Dataset Builder — Speech Smoothing System v2")
    print("="*60)

    out_audio_dir = Path("data/audio_processed")
    out_audio_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    all_rows += load_students(args.student_audio, args.student_excel, out_audio_dir)
    all_rows += load_uaspeech(args.uaspeech_dir, out_audio_dir, args.max_uaspeech)
    if not args.skip_youtube:
        all_rows += load_youtube(out_audio_dir, args.max_yt_segments)

    # Save CSV
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fields = ["source","filename","transcript","severity","speaker_id","duration_s"]
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_rows)

    # Summary
    src = Counter(r["source"]   for r in all_rows)
    sev = Counter(r["severity"] for r in all_rows if r["severity"])
    unlabelled = sum(1 for r in all_rows if not r["severity"])

    print(f"\n{'='*60}")
    print("  DATASET COMPLETE")
    print(f"{'='*60}")
    print(f"  Total samples : {len(all_rows)}")
    print(f"  By source     : student={src.get('student',0)}  uaspeech={src.get('uaspeech',0)}  youtube={src.get('youtube',0)}")
    print(f"  By severity   : mild={sev.get('mild',0)}  moderate={sev.get('moderate',0)}  severe={sev.get('severe',0)}")
    print(f"  Unlabelled    : {unlabelled} (student recordings — label in step 2)")
    print(f"  Output        : {args.output}")
    print()
    print("  Next step — label your student recordings:")
    print("  python data_pipeline/step2_label.py")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
