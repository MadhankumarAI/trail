"""Inject the Phase 3 geometry payload into the template."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
tpl = (ROOT / "web" / "template_03.html").read_text(encoding="utf-8")
data = (ROOT / "web" / "phase3.json").read_text(encoding="utf-8")

out = tpl.replace("__P3__", data)
dst = ROOT / "phases" / "03-how-models-work.html"
dst.write_text(out, encoding="utf-8")
print(f"wrote {dst.relative_to(ROOT)}  {len(out)/1e6:.2f} MB")
