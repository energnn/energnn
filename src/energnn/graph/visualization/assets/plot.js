/* Interactive Graph renderer. Reads the JSON payload next to it, builds the SVG elements
   once, then re-projects them on every rotation, zoom or pan. Mirrors
   energnn.graph.visualization.layout.object_geometries. __UID__ is replaced by the plot id. */
(function(){
var root=document.getElementById('__UID__');var D=JSON.parse(root.querySelector('script[type="application/json"]').textContent);
var S=D.size,PAD=30,N=D.nAddr,P=D.pos,yaw=D.ndim===3?0.6:0,pitch=D.ndim===3?-0.35:0;
var svg=root.querySelector('svg.cv'),tip=root.querySelector('.tip'),NS='http://www.w3.org/2000/svg';
/* theme "auto": follow the notebook, i.e. the first opaque background color above the plot
   (JupyterLab, VS Code and PyCharm themes set it), else the OS preference */
function detectTheme(){var dark=null,e=root.parentElement,guard=0;
 while(e&&dark===null&&guard++<400){var m=(win(e).getComputedStyle(e).backgroundColor||'').match(/rgba?\(([^)]+)\)/);
  if(m){var c=m[1].split(',').map(parseFloat);if(c.length<4||c[3]>0)dark=(0.2126*c[0]+0.7152*c[1]+0.0722*c[2])/255<0.5;}
  var next=e.parentElement;if(!next){try{next=win(e).frameElement;}catch(x){next=null;}}e=next;}
 if(dark===null)dark=!!(window.matchMedia&&window.matchMedia('(prefers-color-scheme: dark)').matches);
 var was=root.classList.contains('dark');root.classList.toggle('dark',dark);return was!==dark;}
if(D.autoTheme){detectTheme();
 /* follow theme switches made after load (Furo docs, JupyterLab, VS Code toggle attributes on html/body) */
 if(window.MutationObserver){var obs=new MutationObserver(function(){if(detectTheme()&&typeof render==='function'){render();scales();}});
  [document.documentElement,document.body].forEach(function(n){if(n)obs.observe(n,{attributes:true});});}}
function win(e){return (e.ownerDocument&&e.ownerDocument.defaultView)||window;}
function css(name){return win(root).getComputedStyle(root).getPropertyValue(name).trim();}
function hex(h){h=h.replace('#','');return [0,2,4].map(function(i){return parseInt(h.substr(i,2),16)/255;});}
function toHex(c){return '#'+c.map(function(v){v=Math.max(0,Math.min(1,v));return ('0'+Math.round(v*255).toString(16)).slice(-2);}).join('');}
function mix(a,b,t){return [a[0]+(b[0]-a[0])*t,a[1]+(b[1]-a[1])*t,a[2]+(b[2]-a[2])*t];}
function seqColor(v){var st=[0,1,2,3].map(function(i){return hex(css('--s'+i));});var x=Math.max(0,Math.min(1,v))*3,lo=Math.min(2,Math.floor(x));return mix(st[lo],st[lo+1],x-lo);}
function bivColor(u,v){var c00=hex(css('--b00')),c10=hex(css('--b10')),c01=hex(css('--b01')),c11=hex(css('--b11'));return mix(mix(c00,c10,u),mix(c01,c11,u),v);}
function chColor(ch){if(ch.length===1)return toHex(seqColor(ch[0]));if(ch.length===2)return toHex(bivColor(ch[0],ch[1]));return toHex(ch);}
function el(tag,attrs,parent){var e=document.createElementNS(NS,tag);for(var k in attrs)e.setAttribute(k,attrs[k]);if(parent)parent.appendChild(e);return e;}
/* view rotation (3D only): yaw around the vertical axis, pitch around the horizontal one */
function rot(p){var cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch);var x=cy*p[0]+sy*p[2],z=-sy*p[0]+cy*p[2];var y=cp*p[1]-sp*z;return [x,y];}
/* 3D layouts are shrunk so that the rotated [-1, 1] cube stays (almost) inside the canvas */
var SCALE=D.ndim===3?0.65:1;
/* the canvas maps [-1-margin, 1+margin] so stubs, loops and fanned edges stay in view; zoom (Z) and
   pan (OX, OY) are applied here, in pixels, so strokes, markers and labels keep their size */
var M=D.margin,W=2+2*M,Z=1,OX=0,OY=0;
function px(p){var q=rot(p);return [OX+Z*(PAD+(q[0]*SCALE+1+M)/W*(S-2*PAD)),OY+Z*(PAD+(1+M-q[1]*SCALE)/W*(S-2*PAD))];}
function pts(list){return list.map(function(p){var q=px(p);return q[0].toFixed(1)+','+q[1].toFixed(1);}).join(' ');}
function add(a,b,k){return [a[0]+b[0]*k,a[1]+b[1]*k,a[2]+b[2]*k];}
function norm(a){return Math.sqrt(a[0]*a[0]+a[1]*a[1]+a[2]*a[2]);}
/* geometry of one object from the node positions P */
function geom(o){var ports=o.ports;
/* stub, loop and degenerate-hub offsets are divided by the zoom so they stay constant on screen,
   like the address radius; their spokes start at the address center, so nothing ever detaches */
 var off=D.stub*D.addrR/Z;
 if(o.kind==='stub'){var A=P[ports[0]],tp=add(A,[o.direction[0],o.direction[1],0],off);return {lines:[[A,tp]],marker:tp,labels:[add(A,tp,1).map(function(v){return v/2;})]};}
 if(o.kind==='loop'){var A=P[ports[0]],mk=add(A,[o.direction[0],o.direction[1],0],off),c1=fanned(A,mk,-0.5),c2=fanned(A,mk,0.5);
  return {lines:[c1,c2],marker:mk,labels:[c1[8],c2[8]]};}
 if(o.kind==='pair'){var curve=fanned(P[ports[0]],P[ports[1]],o.fan);return {lines:[curve],marker:curve[8],labels:[curve[3],curve[13]]};}
 if(o.kind==='hub'){var H=o.direction?add(P[ports[0]],[o.direction[0],o.direction[1],0],off):P[o.hub],counts={},seen={},lines=[],labels=[];
  ports.forEach(function(p){counts[p]=(counts[p]||0)+1;});
  ports.forEach(function(p){var j=seen[p]||0;seen[p]=j+1;
   if(counts[p]===1){lines.push([H,P[p]]);labels.push(add(H,P[p],1).map(function(v){return v/2;}));}
   else{var c=fanned(H,P[p],j-(counts[p]-1)/2);lines.push(c);labels.push(c[8]);}});
  return {lines:lines,marker:H,labels:labels};}
 return null;}
/* Bezier curve from A to B bent by its rank among parallel connections (mirrors layout._fanned_curve) */
function fanned(A,B,fan){var ch=add(B,A,-1),L=Math.max(norm(ch),1e-9),d=ch.map(function(v){return v/L;});
 var n=[-d[1],d[0],0],nl=norm(n);n=nl<1e-9?[1,0,0]:n.map(function(v){return v/nl;});var h=fan*Math.min(0.3*L,D.fanH);
 var ctrl=add(add(A,B,1).map(function(v){return v/2;}),n,2*h),curve=[];
 for(var k=0;k<17;k++){var t=k/16;curve.push(A.map(function(v,i){return (1-t)*(1-t)*v+2*t*(1-t)*ctrl[i]+t*t*B[i];}));}
 return curve;}
/* build the elements once; an object colored by hyper_edge_colors keeps its channels for render() */
var objs=[],addrs=[];
D.classes.forEach(function(c){c.objects.forEach(function(o){if(o.kind==='none')return;
 var g=el('g',{'class':'obj','data-tip':o.tip},svg),color=c.color||'var(--neutral)';
 var nl=o.kind==='hub'?o.ports.length:(o.kind==='loop'?2:1),lines=[],labels=[];
 for(var k=0;k<nl;k++)lines.push(el('polyline',{fill:'none',stroke:color,'stroke-width':D.stroke.toFixed(1),'stroke-opacity':'0.85'},g));
 c.portNames.forEach(function(pn){var t=el('text',{'class':'pl','text-anchor':'middle'},g);t.textContent=pn;labels.push(t);});
 var mk=el('polygon',{'class':'mk',fill:color,stroke:'var(--surface)','stroke-width':'1'},g);
 objs.push({o:o,shape:c.shape,lines:lines,labels:labels,mk:mk,channels:o.color||null});});});
for(var i=0;i<N;i++){var g=el('g',{'class':'addr','data-tip':D.addrTips[i]},svg);
 var c=el('circle',{r:D.rAddr.toFixed(1),fill:'var(--surface)',stroke:'var(--ink)','stroke-width':'1.2'},g);
 if(D.inferred[i])c.setAttribute('stroke-dasharray','3 2');
 var tx=el('text',{'text-anchor':'middle','dominant-baseline':'central',fill:'var(--ink)','font-size':D.fontSize,'pointer-events':'none'},g);tx.textContent=i;
 var notes=[];if(D.inferred[i])notes.push(D.inferredTip);if(D.colors&&!D.colors[i])notes.push(D.noColorTip);
 if(notes.length)g.setAttribute('data-tip',D.addrTips[i]+'<br><i>'+notes.join(', ')+'</i>');addrs.push({c:c,t:tx});}
var MK=D.markers;
function markerPts(shape,x,y){return MK[shape].map(function(d){return (x+d[0]).toFixed(1)+','+(y+d[1]).toFixed(1);}).join(' ');}
function render(){
 objs.forEach(function(ob){var G=geom(ob.o);G.lines.forEach(function(l,k){ob.lines[k].setAttribute('points',pts(l));});
  G.labels.forEach(function(p,k){if(ob.labels[k]){var q=px(p);ob.labels[k].setAttribute('x',q[0].toFixed(1));ob.labels[k].setAttribute('y',(q[1]-3).toFixed(1));}});
  var m=px(G.marker);ob.mk.setAttribute('points',markerPts(ob.shape,m[0],m[1]));
  if(ob.channels){var col=chColor(ob.channels);ob.mk.setAttribute('fill',col);ob.lines.forEach(function(l){l.setAttribute('stroke',col);});}});
 for(var i=0;i<N;i++){var q=px(P[i]);addrs[i].c.setAttribute('cx',q[0].toFixed(1));addrs[i].c.setAttribute('cy',q[1].toFixed(1));
  addrs[i].t.setAttribute('x',q[0].toFixed(1));addrs[i].t.setAttribute('y',q[1].toFixed(1));
  if(D.colors)addrs[i].c.setAttribute('fill',D.colors[i]?chColor(D.colors[i]):'var(--surface)');}}
/* color scales in the legend (addresses, hyper-edges), computed from the active theme's CSS variables */
function scales(){root.querySelectorAll('.lg .sc').forEach(function(legend){var C=+legend.getAttribute('data-ch'),old=legend.querySelector('svg');if(old)old.remove();
 var anchor=null;legend.childNodes.forEach(function(n){if(n.nodeType===8)anchor=n;});if(!anchor)return;
 if(C===1){var bar=el('svg',{width:'90',height:'10'});for(var k=0;k<30;k++)el('rect',{x:k*3,y:0,width:3,height:10,fill:toHex(seqColor(k/29))},bar);legend.insertBefore(bar,anchor);}
 else if(C===2){var sq=el('svg',{width:'40',height:'40'});for(var a=0;a<8;a++)for(var b=0;b<8;b++)el('rect',{x:a*5,y:35-b*5,width:5,height:5,fill:toHex(bivColor(a/7,b/7))},sq);legend.insertBefore(sq,anchor);}});}
/* tooltips */
root.querySelectorAll('[data-tip]').forEach(function(e){
 e.addEventListener('mousemove',function(ev){tip.innerHTML=e.getAttribute('data-tip');tip.style.display='block';var r=root.getBoundingClientRect();tip.style.left=(ev.clientX-r.left+14)+'px';tip.style.top=(ev.clientY-r.top+14)+'px';});
 e.addEventListener('mouseleave',function(){tip.style.display='none';});});
/* view control: the toolbar picks the drag mode (rotate in 3D, pan) and offers zoom in/out and reset;
   the wheel always zooms on the cursor, shift-drag always pans, double-click resets */
var drag=null,mode=D.ndim===3?'rotate':'pan';
function toSvg(e){var r=svg.getBoundingClientRect();return [(e.clientX-r.left)/r.width*S,(e.clientY-r.top)/r.height*S];}
function zoomAt(f,cx,cy){Z*=f;OX=cx-(cx-OX)*f;OY=cy-(cy-OY)*f;render();}
function reset(){Z=1;OX=0;OY=0;yaw=D.ndim===3?0.6:0;pitch=D.ndim===3?-0.35:0;render();}
function setMode(m){mode=m;root.querySelectorAll('.tb [data-mode]').forEach(function(b){b.classList.toggle('on',b.getAttribute('data-mode')===m);});}
root.querySelectorAll('.tb button').forEach(function(b){b.addEventListener('click',function(){
 var m=b.getAttribute('data-mode'),a=b.getAttribute('data-act');
 if(m)setMode(m);else if(a==='zin')zoomAt(1.25,S/2,S/2);else if(a==='zout')zoomAt(0.8,S/2,S/2);else if(a==='reset')reset();});});
setMode(mode);
svg.addEventListener('wheel',function(e){e.preventDefault();var c=toSvg(e);zoomAt(e.deltaY<0?1.25:0.8,c[0],c[1]);},{passive:false});
svg.addEventListener('mousedown',function(e){e.preventDefault();drag={x:e.clientX,y:e.clientY,ox:OX,oy:OY,yaw:yaw,pitch:pitch,rotate:mode==='rotate'&&D.ndim===3&&!e.shiftKey};
 var w=win(svg);function move(ev){if(!drag)return;var r=svg.getBoundingClientRect(),dx=ev.clientX-drag.x,dy=ev.clientY-drag.y;
  if(drag.rotate){yaw=drag.yaw+dx/r.width*Math.PI;pitch=drag.pitch-dy/r.height*Math.PI;}
  else{OX=drag.ox+dx*S/r.width;OY=drag.oy+dy*S/r.height;}
  render();}
 function up(){drag=null;w.removeEventListener('mousemove',move);w.removeEventListener('mouseup',up);}
 w.addEventListener('mousemove',move);w.addEventListener('mouseup',up);});
svg.addEventListener('dblclick',reset);
render();scales();
})();
