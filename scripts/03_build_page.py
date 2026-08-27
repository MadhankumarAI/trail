"""Inject the exported scan payload into the Phase 2 template."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
tpl = (ROOT / "web" / "template_02.html").read_text(encoding="utf-8")

meta = (ROOT / "web" / "meta.json").read_text()
img = (ROOT / "web" / "frame.jpg.txt").read_text()
pay = (ROOT / "web" / "payload.json.txt").read_text()

out = (tpl.replace("__META__", meta)
          .replace("__IMG__", img)
          .replace("__PAYLOAD__", pay))

dst = ROOT / "phases" / "02-point-anatomy.html"
dst.write_text(out, encoding="utf-8")
print(f"wrote {dst.relative_to(ROOT)}  {len(out)/1e6:.2f} MB")
