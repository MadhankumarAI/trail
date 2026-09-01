"""Inject the terrain payload into the viewer template."""
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
tpl = (ROOT / "web" / "template_terrain.html").read_text(encoding="utf-8")
data = (ROOT / "web" / "terrain.json").read_text(encoding="utf-8")
dst = ROOT / "reports" / "terrain.html"
dst.parent.mkdir(exist_ok=True)
assert "__TERRAIN__" in tpl, "placeholder missing"
dst.write_text(tpl.replace("__TERRAIN__", data), encoding="utf-8")
print(f"wrote {dst.relative_to(ROOT)}  {dst.stat().st_size/1e6:.2f} MB")
