"""
setup_transcript_lookup.py
===========================
Reads your Excel transcription file and creates
data/transcript_lookup.json so the app uses your
correct transcriptions instead of Whisper.

Run: python setup_transcript_lookup.py --excel Transcript.xlsx
"""
import argparse, json, os
from pathlib import Path

def build_lookup(excel_path: str, output: str = "data/transcript_lookup.json"):
    lookup = {}

    try:
        import openpyxl
        wb = openpyxl.load_workbook(excel_path, read_only=True)
        ws = wb.active
        for row in ws.iter_rows(values_only=True):
            vals = [str(v).strip() if v is not None else "" for v in row]
            vals = [v for v in vals if v and v.lower() not in ("filename","transcription","transcript","text","nan")]
            if len(vals) >= 2:
                fname = vals[0]
                text  = vals[1]
                for ext in ["", ".m4a.mp4", ".mp4", ".m4a", ".mp3", ".wav"]:
                    key = (fname + ext).lower()
                    lookup[key] = text
    except ImportError:
        import pandas as pd
        df = pd.read_excel(excel_path, header=None)
        for _, row in df.iterrows():
            vals = [str(v).strip() for v in row if str(v).strip() not in ("nan","")]
            if len(vals) >= 2:
                fname, text = vals[0], vals[1]
                if text.lower() in ("transcription","transcript","text"): continue
                for ext in ["", ".m4a.mp4", ".mp4", ".m4a", ".mp3", ".wav"]:
                    lookup[(fname+ext).lower()] = text

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(lookup, f, indent=2, ensure_ascii=False)

    unique = len(lookup) // 6
    print(f"Created lookup with {unique} transcriptions → {output}")
    print("The app will now use your Excel transcriptions as ground truth.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--excel",  default="Transcript.xlsx")
    parser.add_argument("--output", default="data/transcript_lookup.json")
    args = parser.parse_args()

    if not os.path.exists(args.excel):
        print(f"Excel file not found: {args.excel}")
        exit(1)

    build_lookup(args.excel, args.output)
