/* Minimal DOM shim to run plot.js headlessly: rebuilds the elements from the JSON tree written by
   test_interactive.py, evaluates the embedded script, exercises it, and prints what it changed. */
const fs = require('fs');
const tree = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
class El {
  constructor(tag, attrs) { this.tag = tag; this.attrs = attrs || {}; this.children = []; this.parentElement = null; this.nodeType = 1;
    this.style = {}; this.listeners = {}; this._text = '';
    const self = this; this.classList = { contains: c => self.cls().includes(c), toggle(c, on) { const s = new Set(self.cls()); on ? s.add(c) : s.delete(c); self.attrs['class'] = [...s].join(' '); } }; }
  cls() { return (this.attrs['class'] || '').split(' ').filter(Boolean); }
  setAttribute(k, v) { this.attrs[k] = String(v); } getAttribute(k) { return k in this.attrs ? this.attrs[k] : null; }
  appendChild(c) { c.parentElement = this; this.children.push(c); return c; }
  insertBefore(c, ref) { c.parentElement = this; const i = this.children.indexOf(ref); this.children.splice(i < 0 ? this.children.length : i, 0, c); return c; }
  remove() { const p = this.parentElement; if (p) p.children.splice(p.children.indexOf(this), 1); }
  get childNodes() { return this.children; } get textContent() { return this._text; } set textContent(v) { this._text = String(v); }
  set innerHTML(v) { this._html = v; } get innerHTML() { return this._html; }
  addEventListener(name, fn) { (this.listeners[name] = this.listeners[name] || []).push(fn); }
  removeEventListener() {} fire(name, ev) { (this.listeners[name] || []).forEach(fn => fn(ev)); }
  getBoundingClientRect() { return { left: 0, top: 0, width: +this.attrs.width || 640, height: +this.attrs.height || 640 }; }
  querySelector(sel) { return this.querySelectorAll(sel)[0] || null; }
  querySelectorAll(sel) { const out = []; const walk = e => (e.children || []).forEach(c => { if (c.attrs && sel.split(',').some(s => matches(c, s))) out.push(c); walk(c); }); walk(this); return out; }
}
function matches(e, sel) {
  const last = sel.trim().split(/\s+/).pop();  // descendant combinators are ignored: the last simple selector decides
  const m = last.match(/^([\w-]+)?((?:\.[\w-]+)*)(\[([\w-]+)(?:="([^"]*)")?\])?$/);
  if (!m) throw new Error('unsupported selector ' + sel);
  if (m[1] && e.tag !== m[1]) return false;
  for (const c of m[2].split('.').filter(Boolean)) if (!e.cls().includes(c)) return false;
  if (m[3] && (e.attrs[m[4]] === undefined || (m[5] !== undefined && e.attrs[m[4]] !== m[5]))) return false;
  return true;
}
let script = '', cssText = '';
function build(node) {
  if ('comment' in node) return { nodeType: 8, parentElement: null };
  if ('text' in node) return { nodeType: 3, text: node.text };
  const e = new El(node.tag, node.attrs);
  node.children.forEach(c => { const child = build(c); if (child.nodeType === 3) { if (node.tag === 'script') script += child.text; else if (node.tag === 'style') cssText += child.text; else e._text += child.text; } else e.appendChild(child); });
  return e;
}
const doc = build(tree);
const root = doc.querySelector('[data-auto-theme]');
const vars = {}; (cssText.match(new RegExp('#' + root.attrs.id + '\\{([^}]*)\\}'))[1]).split(';').forEach(kv => { const [k, v] = kv.split(':'); if (k) vars[k.trim()] = v; });
global.document = { getElementById: id => id === root.attrs.id ? root : null, createElementNS: (ns, tag) => new El(tag), documentElement: new El('html'), body: new El('body') };
global.window = { getComputedStyle: () => ({ getPropertyValue: k => vars[k] || '', backgroundColor: 'rgb(255, 255, 255)' }), matchMedia: () => ({ matches: false }) };
eval(script);
const svg = root.querySelector('svg.cv'), view = svg.querySelector('.view');
const addrs = view.querySelectorAll('.addr'), objs = view.querySelectorAll('.obj');
const cls = g => g.attrs['data-tip'].match(/<b>(\w+)/)[1];
svg.fire('wheel', { preventDefault() {}, deltaY: -1, clientX: 320, clientY: 320 });
const zoomed = view.getAttribute('transform'), fixedZoomed = view.querySelector('.fx').getAttribute('transform');
svg.fire('dblclick', {});
const reset = view.getAttribute('transform'), fixedReset = view.querySelector('.fx').getAttribute('transform');
addrs[0].fire('mousemove', { clientX: 10, clientY: 10 });
console.log(JSON.stringify({
  addressFills: addrs.map(g => g.querySelector('circle').getAttribute('fill')),
  busFills: objs.filter(g => cls(g) === 'bus').map(g => g.querySelector('.mk').getAttribute('fill')),
  lineStrokes: objs.filter(g => cls(g) === 'line').map(g => g.querySelector('polyline').getAttribute('stroke')),
  scaleSvgs: root.querySelectorAll('.lg .sc').filter(sc => sc.querySelector('svg')).length,
  transformAfterZoom: zoomed, transformAfterReset: reset, fixedAfterZoom: fixedZoomed, fixedAfterReset: fixedReset,
  tooltipOnHover: root.querySelector('.tip').innerHTML,
}));
