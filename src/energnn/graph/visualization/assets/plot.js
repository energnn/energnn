/* Interactive Graph renderer. Reads the JSON payload next to it, builds the SVG elements
   once, then re-projects them on every rotation / frame change. Mirrors
   energnn.graph.visualization.layout.object_geometries. __UID__ is replaced by the plot id. */
(function(){
var root=document.getElementById('__UID__');var D=JSON.parse(root.querySelector('script[type="application/json"]').textContent);
var S=D.size,PAD=30,N=D.nAddr,T=D.frames.length,t=0,yaw=D.ndim===3?0.6:0,pitch=D.ndim===3?-0.35:0;
var svg=root.querySelector('svg.cv'),tip=root.querySelector('.tip'),NS='http://www.w3.org/2000/svg';
/* theme "auto": follow the notebook, i.e. the first opaque background color above the plot
   (JupyterLab, VS Code and PyCharm themes set it), else the OS preference */
if(D.autoTheme){var dark=null,e=root.parentElement;
 while(e&&dark===null){var m=(getComputedStyle(e).backgroundColor||'').match(/rgba?\(([^)]+)\)/);
  if(m){var c=m[1].split(',').map(parseFloat);if(c.length<4||c[3]>0)dark=(0.2126*c[0]+0.7152*c[1]+0.0722*c[2])/255<0.5;}
  e=e.parentElement;}
 if(dark===null)dark=!!(window.matchMedia&&window.matchMedia('(prefers-color-scheme: dark)').matches);
 if(dark)root.classList.add('dark');}
function css(name){return getComputedStyle(root).getPropertyValue(name).trim();}
function hex(h){h=h.replace('#','');return [0,2,4].map(function(i){return parseInt(h.substr(i,2),16)/255;});}
function toHex(c){return '#'+c.map(function(v){v=Math.max(0,Math.min(1,v));return ('0'+Math.round(v*255).toString(16)).slice(-2);}).join('');}
function mix(a,b,t){return [a[0]+(b[0]-a[0])*t,a[1]+(b[1]-a[1])*t,a[2]+(b[2]-a[2])*t];}
function seqColor(v){var st=[0,1,2,3].map(function(i){return hex(css('--s'+i));});var x=Math.max(0,Math.min(1,v))*3,lo=Math.min(2,Math.floor(x));return mix(st[lo],st[lo+1],x-lo);}
function bivColor(u,v){var c00=hex(css('--b00')),c10=hex(css('--b10')),c01=hex(css('--b01')),c11=hex(css('--b11'));return mix(mix(c00,c10,u),mix(c01,c11,u),v);}
function addrColor(ch){if(ch.length===1)return toHex(seqColor(ch[0]));if(ch.length===2)return toHex(bivColor(ch[0],ch[1]));return toHex(ch);}
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
/* geometry of one object from the address positions P of the current frame */
function geom(o,P){var ports=o.ports;
 if(o.kind==='stub'){var A=P[ports[0]],tp=add(A,[o.direction[0],o.direction[1],0],D.stub*D.addrR);return {lines:[[A,tp]],marker:tp,labels:[add(A,tp,1).map(function(v){return v/2;})]};}
 if(o.kind==='loop'){var A=P[ports[0]],u=[o.direction[0],o.direction[1],0],v=[-u[1],u[0],0],r=D.loopR,c=add(A,u,D.addrR+r+0.01),circle=[];
  for(var k=0;k<25;k++){var th=2*Math.PI*k/24;circle.push(add(add(c,u,r*Math.cos(th)),v,r*Math.sin(th)));}
  return {lines:[circle],marker:add(c,u,r),labels:[add(add(c,u,1.6*r*Math.cos(0.9)),v,1.6*r*Math.sin(0.9)),add(add(c,u,1.6*r*Math.cos(0.9)),v,-1.6*r*Math.sin(0.9))]};}
 if(o.kind==='pair'){var curve=fanned(P[ports[0]],P[ports[1]],o.fan);return {lines:[curve],marker:curve[8],labels:[curve[3],curve[13]]};}
 if(o.kind==='hub'){var H=P[o.hub],counts={},seen={},lines=[],labels=[];
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
/* build the elements once */
var objs=[],addrs=[];
D.classes.forEach(function(c){c.objects.forEach(function(o){if(o.kind==='none')return;
 var g=el('g',{'class':'obj','data-tip':o.tip},svg),color=c.color||'var(--neutral)';
 var nl=o.kind==='hub'?o.ports.length:1,lines=[],labels=[];
 for(var k=0;k<nl;k++)lines.push(el('polyline',{fill:'none',stroke:color,'stroke-width':D.stroke.toFixed(1),'stroke-opacity':'0.85'},g));
 c.portNames.forEach(function(pn){var t=el('text',{'class':'pl','text-anchor':'middle'},g);t.textContent=pn;labels.push(t);});
 var mk=el('polygon',{'class':'mk',fill:color,stroke:'var(--surface)','stroke-width':'1'},g);
 objs.push({o:o,shape:c.shape,lines:lines,labels:labels,mk:mk});});});
for(var i=0;i<N;i++){var g=el('g',{'class':'addr','data-tip':D.addrTips[i]},svg);
 var c=el('circle',{r:D.rAddr.toFixed(1),fill:'var(--surface)',stroke:'var(--ink)','stroke-width':'1.2'},g);
 var tx=el('text',{'text-anchor':'middle','dominant-baseline':'central',fill:'var(--ink)','font-size':D.fontSize,'pointer-events':'none'},g);tx.textContent=i;addrs.push({c:c,t:tx});}
var MK=D.markers;
function markerPts(shape,x,y){return MK[shape].map(function(d){return (x+d[0]).toFixed(1)+','+(y+d[1]).toFixed(1);}).join(' ');}
/* frame at a fractional time: positions and color channels are interpolated linearly */
function lerpRows(a,b,k){return a.map(function(row,i){return row.map(function(v,j){return v+(b[i][j]-v)*k;});});}
function at(list){var i0=Math.min(T-1,Math.max(0,Math.floor(t))),i1=Math.min(T-1,i0+1),k=t-i0;return k>0&&i1>i0?lerpRows(list[i0],list[i1],k):list[i0];}
function render(){var P=at(D.frames),C=D.colors?at(D.colors):null;
 objs.forEach(function(ob){var G=geom(ob.o,P);G.lines.forEach(function(l,k){ob.lines[k].setAttribute('points',pts(l));});
  G.labels.forEach(function(p,k){if(ob.labels[k]){var q=px(p);ob.labels[k].setAttribute('x',q[0].toFixed(1));ob.labels[k].setAttribute('y',(q[1]-3).toFixed(1));}});
  var m=px(G.marker);ob.mk.setAttribute('points',markerPts(ob.shape,m[0],m[1]));});
 for(var i=0;i<N;i++){var q=px(P[i]);addrs[i].c.setAttribute('cx',q[0].toFixed(1));addrs[i].c.setAttribute('cy',q[1].toFixed(1));
  addrs[i].t.setAttribute('x',q[0].toFixed(1));addrs[i].t.setAttribute('y',q[1].toFixed(1));
  if(C)addrs[i].c.setAttribute('fill',addrColor(C[i]));}
 var lab=root.querySelector('.tl .fr');if(lab)lab.textContent='t = '+(Math.round(t*10)/10)+' / '+(T-1);}
/* color scale in the legend, computed from the active theme's CSS variables */
var legend=root.querySelector('.lg .sc');if(legend&&D.colors){var C=D.colors[0][0].length,anchor=legend.childNodes[1];
 if(C===1){var bar=el('svg',{width:'90',height:'10'});for(var k=0;k<30;k++)el('rect',{x:k*3,y:0,width:3,height:10,fill:toHex(seqColor(k/29))},bar);legend.insertBefore(bar,anchor);}
 else if(C===2){var sq=el('svg',{width:'40',height:'40'});for(var a=0;a<8;a++)for(var b=0;b<8;b++)el('rect',{x:a*5,y:35-b*5,width:5,height:5,fill:toHex(bivColor(a/7,b/7))},sq);legend.insertBefore(sq,anchor);}}
/* tooltips */
root.querySelectorAll('[data-tip]').forEach(function(e){
 e.addEventListener('mousemove',function(ev){tip.innerHTML=e.getAttribute('data-tip');tip.style.display='block';var r=root.getBoundingClientRect();tip.style.left=(ev.clientX-r.left+14)+'px';tip.style.top=(ev.clientY-r.top+14)+'px';});
 e.addEventListener('mouseleave',function(){tip.style.display='none';});});
/* view control: the toolbar picks the drag mode (rotate in 3D, pan) and offers zoom in/out, reset and
   full screen; the wheel always zooms on the cursor, shift-drag always pans, double-click resets */
var drag=null,mode=D.ndim===3?'rotate':'pan';
function toSvg(e){var r=svg.getBoundingClientRect();return [(e.clientX-r.left)/r.width*S,(e.clientY-r.top)/r.height*S];}
function zoomAt(f,cx,cy){Z*=f;OX=cx-(cx-OX)*f;OY=cy-(cy-OY)*f;render();}
function reset(){Z=1;OX=0;OY=0;yaw=D.ndim===3?0.6:0;pitch=D.ndim===3?-0.35:0;render();}
function setMode(m){mode=m;root.querySelectorAll('.tb [data-mode]').forEach(function(b){b.classList.toggle('on',b.getAttribute('data-mode')===m);});}
/* full screen. In a page of its own (JupyterLab, a saved file): the browser API when available, else a
   fixed overlay filling the window. In an output iframe (PyCharm, VS Code) neither works: the API is
   refused and a fixed overlay collapses the iframe, so the figure is enlarged to the iframe's width
   instead, the host growing the iframe to fit. Esc leaves all of them. */
var fsBtn=root.querySelector('.tb [data-act="fs"]'),inFrame=true;
try{inFrame=window.self!==window.top;}catch(e){}
var fsClass=inFrame?'big':'fs',fsOn=false;
function setFs(on){fsOn=on;root.classList.toggle(fsClass,on);if(fsBtn)fsBtn.textContent=on?'\u2716':'\u26F6';}
function toggleFs(){var on=!fsOn;setFs(on);
 if(inFrame)return;
 if(on&&root.requestFullscreen){var p=root.requestFullscreen();if(p&&p.catch)p.catch(function(){});}
 else if(!on&&document.fullscreenElement===root&&document.exitFullscreen)document.exitFullscreen();}
document.addEventListener('fullscreenchange',function(){if(document.fullscreenElement!==root&&fsOn&&!inFrame)setFs(false);});
document.addEventListener('keydown',function(e){if(e.key==='Escape'&&fsOn)toggleFs();});
root.querySelectorAll('.tb button').forEach(function(b){b.addEventListener('click',function(){
 var m=b.getAttribute('data-mode'),a=b.getAttribute('data-act');
 if(m)setMode(m);else if(a==='zin')zoomAt(1.25,S/2,S/2);else if(a==='zout')zoomAt(0.8,S/2,S/2);else if(a==='reset')reset();else if(a==='fs')toggleFs();});});
setMode(mode);
svg.addEventListener('wheel',function(e){e.preventDefault();var c=toSvg(e);zoomAt(e.deltaY<0?1.25:0.8,c[0],c[1]);},{passive:false});
svg.addEventListener('mousedown',function(e){e.preventDefault();drag={x:e.clientX,y:e.clientY,ox:OX,oy:OY,yaw:yaw,pitch:pitch,rotate:mode==='rotate'&&D.ndim===3&&!e.shiftKey};});
window.addEventListener('mousemove',function(e){if(!drag)return;var r=svg.getBoundingClientRect(),dx=e.clientX-drag.x,dy=e.clientY-drag.y;
 if(drag.rotate){yaw=drag.yaw+dx/r.width*Math.PI;pitch=drag.pitch-dy/r.height*Math.PI;}
 else{OX=drag.ox+dx*S/r.width;OY=drag.oy+dy*S/r.height;}
 render();});
window.addEventListener('mouseup',function(){drag=null;});
svg.addEventListener('dblclick',reset);
/* time slider and play button: playback interpolates between frames (one frame per D.interval ms)
   and pauses D.pause ms at the end before looping */
var slider=root.querySelector('.tl input'),play=root.querySelector('.tl button'),playing=false,last=0,pauseUntil=0;
function stop(){playing=false;play.textContent='▶';}
function step(now){if(!playing)return;
 if(pauseUntil){if(now>=pauseUntil){pauseUntil=0;t=0;last=now;}}
 else{t+=(now-last)/D.interval;last=now;if(t>=T-1){t=T-1;pauseUntil=now+D.pause;}}
 slider.value=Math.round(t);render();requestAnimationFrame(step);}
if(slider){slider.addEventListener('input',function(){stop();t=+slider.value;render();});
 play.addEventListener('click',function(){if(playing){stop();return;}
  if(t>=T-1)t=0;playing=true;pauseUntil=0;play.textContent='■';last=performance.now();requestAnimationFrame(step);});}
render();
})();
