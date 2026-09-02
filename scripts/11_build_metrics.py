"""Inject the measured metrics into the results dashboard."""
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
tpl = (ROOT / "web" / "template_metrics.html").read_text(encoding="utf-8")
data = (ROOT / "web" / "metrics.json").read_text(encoding="utf-8")
assert "__METRICS__" in tpl, "placeholder missing"
dst = ROOT / "reports" / "results.html"
dst.parent.mkdir(exist_ok=True)
dst.write_text(tpl.replace("__METRICS__", data), encoding="utf-8")
print(f"wrote {dst.relative_to(ROOT)}  {dst.stat().st_size/1024:.0f} KB")
