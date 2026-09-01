"""Inject the emitted grid_map message into the layer viewer."""
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
tpl = (ROOT / "web" / "template_gridmap.html").read_text(encoding="utf-8")
data = (ROOT / "web" / "gridmap_sample.json").read_text(encoding="utf-8")
assert "__GRIDMAP__" in tpl, "placeholder missing"
dst = ROOT / "reports" / "gridmap.html"
dst.parent.mkdir(exist_ok=True)
dst.write_text(tpl.replace("__GRIDMAP__", data), encoding="utf-8")
print(f"wrote {dst.relative_to(ROOT)}  {dst.stat().st_size/1e6:.2f} MB")
