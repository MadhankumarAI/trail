/* Execute the terrain viewer's script against a stub DOM.
 *
 * A page with a thrown exception renders blank and looks exactly like a page
 * that works until someone opens it. This runs the real script from the built
 * HTML, so a typo, a missing element id or a bad typed-array offset fails here
 * instead of in front of the user. It also exercises every frame and both
 * colour modes, because the offset arithmetic is per-frame and an odd cell
 * count is what breaks Int16Array.
 */
const fs = require('fs');
const path = require('path');

const html = fs.readFileSync(
  path.join(__dirname, '..', 'reports', 'terrain.html'), 'utf8');
const m = html.match(/<script>([\s\S]*)<\/script>/);
if (!m) { console.error('no script block found'); process.exit(1); }

// --- stub DOM -----------------------------------------------------------
const drawn = { rect: 0, text: 0, quad: 0 };
const ctx = new Proxy({}, {
  get(_, k) {
    if (k === 'fillRect') return () => { drawn.rect++; };
    if (k === 'fillText') return () => { drawn.text++; };
    if (k === 'measureText') return () => ({ width: 10 });
  if (k === 'beginPath' || k === 'moveTo' || k === 'lineTo' ||
      k === 'closePath') return () => {};
  if (k === 'fill') return () => { drawn.quad++; };
    return () => {};
  },
  set() { return true; },
});

const ids = {};
function node(id) {
  if (ids[id]) return ids[id];
  const n = {
    id, checked: id === 'rings' || id === 'relief', value: '0', max: '0',
    textContent: '', innerHTML: '', style: {},
    classList: { add() {}, remove() {} },
    getContext: () => ctx,
    width: 620, height: 620,
    addEventListener() {},
  };
  ids[id] = n;
  return n;
}

global.document = {
  getElementById: node,
  querySelector: () => node('tbody'),
};
global.addEventListener = () => {};
global.setInterval = () => 1;
global.clearInterval = () => {};
global.atob = s => Buffer.from(s, 'base64').toString('binary');

// --- run ----------------------------------------------------------------
let D;
try {
  // capture D so the harness can drive every frame afterwards
  const src = m[1] + '\n;globalThis.__D = D; globalThis.__show = show;'
            + '\n;globalThis.__cells = cells;';
  new Function(src)();
  D = globalThis.__D;
} catch (e) {
  console.error('SCRIPT THREW on load: ' + e.message);
  process.exit(1);
}

let bad = 0;
const tot = { s: 0, a: 0 };

// both view modes, both colour modes, every frame. the 3D path sorts and
// projects per cell, so a bad index there throws only when it is actually run.
const views = [['top-down', 'v_top'], ['3D', 'v_3d']];
for (let k = 0; k < D.frames.length; k++) {
  for (const [vname, vid] of views) {
    try { ids[vid].onclick(); } catch (e) {
      console.error(`  view ${vname} toggle THREW: ${e.message}`); bad++;
    }
    for (const mode of [false, true]) {
      ids['height'].checked = mode;
      try {
        globalThis.__show(k);
      } catch (e) {
        console.error(`  frame ${k} (${vname}, height=${mode}) THREW: ${e.message}`);
        bad++;
      }
    }
  }
  // the alignment check that actually matters: the last element of every
  // typed-array view must be readable, i.e. the block was long enough
  const A = globalThis.__cells('single', k), B = globalThis.__cells('accum', k);
  if (A.n !== D.frames[k].ns || B.n !== D.frames[k].na) {
    console.error(`  frame ${k}: cell count mismatch`); bad++;
  }
  if (A.n && (A.k[A.n - 1] > 3 || B.k[B.n - 1] > 3)) {
    console.error(`  frame ${k}: class byte out of range -> misaligned view`);
    bad++;
  }
  tot.s += A.n; tot.a += B.n;
}

// --- cost of each render path -------------------------------------------
// The claim is that 2.5D relief shading is cheaper than a 3D projection, so
// measure it rather than assert it. Same frames, same cells, both panels.
function timeMode(vid, label) {
  ids[vid].onclick();
  ids['height'].checked = false;
  globalThis.__show(0);                      // warm up
  const t0 = process.hrtime.bigint();
  for (let rep = 0; rep < 5; rep++)
    for (let k = 0; k < D.frames.length; k++) globalThis.__show(k);
  const ms = Number(process.hrtime.bigint() - t0) / 1e6 / (5 * D.frames.length);
  console.log(`  ${label.padEnd(22)} ${ms.toFixed(2)} ms / frame (both panels)`);
  return ms;
}
console.log('render cost:');
ids['relief'].checked = false;
const tFlat = timeMode('v_top', '2.5D flat');
ids['relief'].checked = true;
const tRelief = timeMode('v_top', '2.5D + relief');
const t3d = timeMode('v_3d', '3D projected');
console.log(`  -> relief costs ${(tRelief / tFlat).toFixed(2)}x flat, ` +
            `3D costs ${(t3d / tRelief).toFixed(2)}x relief`);
ids['relief'].checked = true;
ids['v_top'].onclick();
console.log('');

const oddS = D.frames.filter(f => f.ns % 2).length;
const oddA = D.frames.filter(f => f.na % 2).length;
console.log(`frames        : ${D.frames.length}`);
console.log(`odd counts    : ${oddS} single, ${oddA} accumulated ` +
            `(these are the ones that break Int16Array offsets)`);
console.log(`cells total   : ${tot.s.toLocaleString()} single, ` +
            `${tot.a.toLocaleString()} accumulated`);
console.log(`canvas ops    : ${drawn.rect.toLocaleString()} fillRect, ` +
            `${drawn.quad.toLocaleString()} quad fills, ${drawn.text} fillText`);
console.log(`table rows    : ${(ids['tbody'].innerHTML.match(/<tr>/g) || []).length}`);
console.log(`readout       : "${ids['fno'].textContent}"`);
console.log(`thresholds    : "${ids['thr'].textContent}"`);
console.log(bad ? `\nFAILED with ${bad} error(s)` : '\nOK - no exceptions');
process.exit(bad ? 1 : 0);
