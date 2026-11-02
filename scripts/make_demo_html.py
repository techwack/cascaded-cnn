"""Build a self-contained interactive demo page (demo_out/demo.html) to open in any browser.

    python scripts/make_demo_html.py            # 3 real test images per class, run through the trained cascade
    python scripts/make_demo_html.py --per-class 5 --seed 1

Pick a class, press "Run cascade" and the three stages reveal their decisions one by one.
Needs trained checkpoints (see scripts/predict.py).
"""
import argparse
import base64
import io
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from predict import explain, read_any  # noqa: E402
from solarnet.common import load_cascade, load_config, preprocess, to_batch  # noqa: E402
from solarnet.data import load_records  # noqa: E402
from solarnet.taxonomy import Taxonomy  # noqa: E402

PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Cascaded-SolarNet demo</title>
<style>
:root{--bg:#f6f7f9;--card:#fff;--ink:#1b1f24;--mut:#6b7380;--line:#e1e4e8;--acc:#d9480f;--ok:#2b8a3e;--bad:#c92a2a;--bar:#2b3a4a;--barlt:#cfd5dc}
@media (prefers-color-scheme:dark){:root{--bg:#14171b;--card:#1d2127;--ink:#eceff3;--mut:#98a1ad;--line:#2b3138;--bar:#e8ecf1;--barlt:#3a424b}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 system-ui,Segoe UI,Roboto,sans-serif}
.wrap{max-width:1100px;margin:0 auto;padding:20px 16px 40px}
h1{font-size:22px;margin:0 0 2px}.sub{color:var(--mut);margin:0 0 16px}
.chips{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:14px}
.chip{border:1px solid var(--line);background:var(--card);color:var(--ink);padding:6px 11px;border-radius:999px;cursor:pointer;font-size:13px}
.chip.on{background:var(--ink);color:var(--bg);border-color:var(--ink)}
.row{display:grid;grid-template-columns:220px 1fr;gap:16px}
@media(max-width:760px){.row{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px}
#img{width:100%;image-rendering:pixelated;border-radius:6px;display:block}
.meta{margin-top:8px;color:var(--mut);font-size:13px}
.btns{display:flex;gap:8px;margin-top:10px}
button.act{flex:1;border:0;border-radius:8px;padding:9px 10px;font-size:14px;cursor:pointer;background:var(--acc);color:#fff;font-weight:600}
button.sec{background:transparent;color:var(--ink);border:1px solid var(--line);font-weight:500}
.stage{opacity:.25;transition:opacity .5s;margin-bottom:14px}.stage.show{opacity:1}
.stage h3{margin:0 0 6px;font-size:14px;display:flex;gap:8px;align-items:center}
.tag{font-size:11px;background:var(--line);color:var(--mut);padding:2px 7px;border-radius:6px;font-weight:500}
.bar{display:grid;grid-template-columns:120px 1fr 44px;gap:8px;align-items:center;margin:3px 0;font-size:13px}
.track{height:12px;background:var(--line);border-radius:6px;overflow:hidden}
.fill{height:100%;width:0;background:var(--barlt);transition:width .9s ease}.fill.pick{background:var(--bar)}
.skip{color:var(--mut);font-style:italic;font-size:13px}
#res{font-size:18px;font-weight:700;margin-top:4px}
.ok{color:var(--ok)}.bad{color:var(--bad)}
</style></head><body><div class="wrap">
<h1>Cascaded-SolarNet live demo</h1>
<p class="sub">Real test images from InfraredSolarModules, run through the trained 5-CNN cascade. Pick a class, press Run.</p>
<div class="chips" id="chips"></div>
<div class="row">
 <div class="card"><img id="img" alt="IR module"><div class="meta" id="meta"></div>
  <div class="btns"><button class="act" id="run">Run cascade</button><button class="act sec" id="next">Another</button></div></div>
 <div class="card">
  <div class="stage" id="s1"><h3>Stage 1 &middot; triage <span class="tag">CNN-1</span></h3><div id="b1"></div></div>
  <div class="stage" id="s2"><h3>Stage 2 &middot; fault family <span class="tag">CNN-2</span></h3><div id="b2"></div></div>
  <div class="stage" id="s3"><h3 id="h3">Stage 3 &middot; specialist <span class="tag">CNN-3</span></h3><div id="b3"></div></div>
  <div class="stage" id="fin"><div class="meta">Final prediction</div><div id="res"></div></div>
 </div></div></div>
<script>
const D=__DATA__;const classes=Object.keys(D);let cur=classes[0],idx=0,timer=[];
const $=id=>document.getElementById(id);
function bars(el,labels,vals,pick){el.innerHTML=labels.map((l,i)=>`<div class="bar"><span>${l}</span><div class="track"><div class="fill ${i==pick?'pick':''}" data-w="${vals[i]*100}"></div></div><span>${Math.round(vals[i]*100)}%</span></div>`).join('')}
function grow(el){requestAnimationFrame(()=>requestAnimationFrame(()=>el.querySelectorAll('.fill').forEach(f=>f.style.width=f.dataset.w+'%')))}
function reset(){timer.forEach(clearTimeout);timer=[];['s1','s2','s3','fin'].forEach(i=>$(i).classList.remove('show'));['b1','b2','b3'].forEach(i=>$(i).innerHTML='')}
function load(){reset();const e=D[cur][idx];$('img').src=e.img;$('meta').textContent='True class: '+e.truth+'  ·  test image '+(idx+1)+' of '+D[cur].length}
function run(){reset();const e=D[cur][idx];
 bars($('b1'),['No-Anomaly','Defect'],e.p1,e.p1[1]>0.5?1:0);$('s1').classList.add('show');grow($('b1'));
 const fin=()=>{const ok=e.pred===e.truth;$('res').innerHTML=`<span class="${ok?'ok':'bad'}">${e.pred}</span> <span class="meta">${ok?'correct':'(true: '+e.truth+')'}</span>`;$('fin').classList.add('show')};
 if(e.exited){timer.push(setTimeout(()=>{$('b2').innerHTML='<div class="skip">skipped: healthy panels exit early</div>';$('b3').innerHTML='<div class="skip">skipped</div>';$('s2').classList.add('show');$('s3').classList.add('show')},1300));timer.push(setTimeout(fin,2300));return}
 timer.push(setTimeout(()=>{bars($('b2'),e.fams,e.p2,e.fams.indexOf(e.family));$('s2').classList.add('show');grow($('b2'))},1500));
 timer.push(setTimeout(()=>{$('h3').innerHTML='Stage 3 &middot; '+e.family+' specialist <span class="tag">CNN-3</span>';bars($('b3'),e.cls3,e.p3,e.cls3.indexOf(e.pred));$('s3').classList.add('show');grow($('b3'))},3100));
 timer.push(setTimeout(fin,4700))}
$('chips').innerHTML=classes.map(c=>`<button class="chip" data-c="${c}">${c}</button>`).join('');
function sel(){document.querySelectorAll('.chip').forEach(b=>b.classList.toggle('on',b.dataset.c===cur))}
document.querySelectorAll('.chip').forEach(b=>b.onclick=()=>{cur=b.dataset.c;idx=0;sel();load();run()});
$('run').onclick=run;$('next').onclick=()=>{idx=(idx+1)%D[cur].length;load();run()};
sel();load();setTimeout(run,400);
</script></body></html>"""


def png_b64(img: np.ndarray, scale: int = 8) -> str:
    rgba = (matplotlib.colormaps["inferno"](img)[..., :3] * 255).astype(np.uint8)
    im = Image.fromarray(rgba).resize((img.shape[1] * scale, img.shape[0] * scale), Image.NEAREST)
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/improved3.yaml")
    ap.add_argument("--taxonomy", default="configs/taxonomy.yaml")
    ap.add_argument("--per-class", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="demo_out/demo.html")
    args = ap.parse_args()

    cfg, tax = load_config(args.config), Taxonomy.load(args.taxonomy)
    saved = json.load(open(cfg["data"]["splits_file"]))
    cascade = load_cascade(cfg, tax, Path(cfg["train"]["out_dir"]))
    records = {r["id"]: r for r in load_records(cfg["data"]["root"])}
    rng = np.random.default_rng(args.seed)
    data = {}
    for cls in tax.classes:
        pool = [records[i] for i in saved["test"] if records[i]["label"] == cls]
        picks = rng.choice(len(pool), size=min(args.per_class, len(pool)), replace=False)
        data[cls] = []
        for k in picks:
            rec = pool[int(k)]
            img = read_any(rec["path"])
            x = preprocess(to_batch(img[None]), saved["stats"], cfg["data"]["upsample"])
            r = explain(cascade, x, tax)
            data[cls].append({"img": png_b64(img), "truth": cls, "pred": r["pred"], "exited": r["exited"],
                              "p1": r["p_stage1"], "fams": list(tax.families), "p2": r["p_stage2"],
                              "family": r["family"], "cls3": r["classes3"], "p3": r["p_stage3"]})
    out = Path(args.out)
    out.parent.mkdir(exist_ok=True)
    out.write_text(PAGE.replace("__DATA__", json.dumps(data)), encoding="utf-8")
    n = sum(len(v) for v in data.values())
    ok = sum(e["pred"] == e["truth"] for v in data.values() for e in v)
    print(f"wrote {out} ({n} images, {ok} predicted correctly)")


if __name__ == "__main__":
    main()
