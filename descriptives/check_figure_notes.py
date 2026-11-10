"""Check that every figure the papers use has an up-to-date note.

Each figure's footer is written to paper/figures/notes/<stem>.txt when the figure is saved
(_style.py), not drawn on the image. This lists every figure that paper/paper_skeleton.tex and
paper/paper_outline_short.tex include and flags any whose note is missing, older than the
figure, or empty ("(no note)"). Exit status 1 if any note is missing or stale.

    .venv/bin/python descriptives/check_figure_notes.py
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"; FIG = PAPER / "figures"; NOTES = FIG / "notes"
TEX = ["paper_skeleton.tex", "paper_outline_short.tex"]
SLACK = 120   # seconds: the note is written just before the figure is saved

stems = sorted({Path(m).stem if Path(m).suffix in (".png", ".pdf", ".jpg") else Path(m).name
                for t in TEX if (PAPER / t).exists()
                for m in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}", (PAPER / t).read_text())})
bad = 0
for s in stems:
    figs = [f for f in (FIG / f"{s}{ext}" for ext in (".png", ".pdf", ".jpg")) if f.exists()]
    note = NOTES / f"{s}.txt"
    if not figs:
        print(f"MISSING FIGURE  {s}"); bad += 1; continue
    if not note.exists():
        print(f"MISSING NOTE    {s}"); bad += 1; continue
    newest = max(f.stat().st_mtime for f in figs)
    if note.stat().st_mtime < newest - SLACK:
        print(f"STALE NOTE      {s}  (figure saved after its note: regenerate it with the current _style.py)"); bad += 1; continue
    if "(no note)" in note.read_text():
        print(f"empty note      {s}")
print(f"{len(stems)} figures checked, {bad} missing or stale")
sys.exit(1 if bad else 0)
