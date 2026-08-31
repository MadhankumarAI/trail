"""Inject the simulator payload into the viewer template."""
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
tpl = (ROOT / "web" / "template_sim.html").read_text(encoding="utf-8")
data = (ROOT / "web" / "sim.json").read_text(encoding="utf-8")
dst = ROOT / "reports" / "simulator.html"
dst.parent.mkdir(exist_ok=True)
out = tpl.replace("__SIM__", data)
dst.write_text(out, encoding="utf-8")
print(f"wrote {dst.relative_to(ROOT)}  {len(out)/1e6:.2f} MB")
