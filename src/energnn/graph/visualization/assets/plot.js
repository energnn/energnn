/* Interaction layer of the interactive plot. Python draws the whole SVG; this script only paints the
   elements colored by a value (their channels are in data-ch, the colormap comes from the theme's CSS
   variables), draws the color scales of the legend, follows the notebook theme, and handles zoom, pan
   and tooltips. __UID__ is replaced by the plot id. */
(function(){
var root=document.getElementById('__UID__'),svg=root.querySelector('svg.cv'),view=svg.querySelector('.view'),tip=root.querySelector('.tip');
var NS='http://www.w3.org/2000/svg',S=+svg.getAttribute('width');
function win(e){return (e.ownerDocument&&e.ownerDocument.defaultView)||window;}
function css(name){return win(root).getComputedStyle(root).getPropertyValue(name).trim();}
function hex(h){h=h.replace('#','');return [0,2,4].map(function(i){return parseInt(h.substr(i,2),16)/255;});}
function toHex(c){return '#'+c.map(function(v){v=Math.max(0,Math.min(1,v));return ('0'+Math.round(v*255).toString(16)).slice(-2);}).join('');}
function mix(a,b,t){return [a[0]+(b[0]-a[0])*t,a[1]+(b[1]-a[1])*t,a[2]+(b[2]-a[2])*t];}
function seqColor(v){var st=[0,1,2,3].map(function(i){return hex(css('--s'+i));});var x=Math.max(0,Math.min(1,v))*3,lo=Math.min(2,Math.floor(x));return mix(st[lo],st[lo+1],x-lo);}
function bivColor(u,v){return mix(mix(hex(css('--b00')),hex(css('--b10')),u),mix(hex(css('--b01')),hex(css('--b11')),u),v);}
function chColor(ch){return toHex(ch.length===1?seqColor(ch[0]):bivColor(ch[0],ch[1]));}
function el(tag,attrs,parent){var e=document.createElementNS(NS,tag);for(var k in attrs)e.setAttribute(k,attrs[k]);parent.appendChild(e);return e;}
/* colors that depend on the theme: value-colored elements and the legend's color scales */
function paint(){
 root.querySelectorAll('[data-ch]').forEach(function(g){var col=chColor(g.getAttribute('data-ch').split(',').map(parseFloat));
  g.querySelectorAll('circle,.mk').forEach(function(e){e.setAttribute('fill',col);});
  g.querySelectorAll('polyline').forEach(function(e){e.setAttribute('stroke',col);});});
 root.querySelectorAll('.lg .sc').forEach(function(sc){var C=+sc.getAttribute('data-ch'),old=sc.querySelector('svg');if(old)old.remove();
  var anchor=null;sc.childNodes.forEach(function(n){if(n.nodeType===8)anchor=n;});if(!anchor)return;
  var box=document.createElementNS(NS,'svg');
  if(C===1){box.setAttribute('width','90');box.setAttribute('height','10');for(var k=0;k<30;k++)el('rect',{x:k*3,y:0,width:3,height:10,fill:toHex(seqColor(k/29))},box);}
  else{box.setAttribute('width','40');box.setAttribute('height','40');for(var a=0;a<8;a++)for(var b=0;b<8;b++)el('rect',{x:a*5,y:35-b*5,width:5,height:5,fill:toHex(bivColor(a/7,b/7))},box);}
  sc.insertBefore(box,anchor);});}
/* theme "auto": follow the notebook, i.e. the first opaque background color above the plot
   (JupyterLab, VS Code and PyCharm themes set it), else the OS preference; repaint on theme switches */
function detectTheme(){var dark=null,e=root.parentElement,guard=0;
 while(e&&dark===null&&guard++<400){var m=(win(e).getComputedStyle(e).backgroundColor||'').match(/rgba?\(([^)]+)\)/);
  if(m){var c=m[1].split(',').map(parseFloat);if(c.length<4||c[3]>0)dark=(0.2126*c[0]+0.7152*c[1]+0.0722*c[2])/255<0.5;}
  var next=e.parentElement;if(!next){try{next=win(e).frameElement;}catch(x){next=null;}}e=next;}
 if(dark===null)dark=!!(window.matchMedia&&window.matchMedia('(prefers-color-scheme: dark)').matches);
 var was=root.classList.contains('dark');root.classList.toggle('dark',dark);return was!==dark;}
if(root.getAttribute('data-auto-theme')==='1'){detectTheme();
 if(window.MutationObserver){var obs=new MutationObserver(function(){if(detectTheme())paint();});
  [document.documentElement,document.body].forEach(function(n){if(n)obs.observe(n,{attributes:true});});}}
/* zoom and pan: a transform on the view group; strokes keep their width (non-scaling-stroke in the CSS) */
var Z=1,OX=0,OY=0,drag=null;
function render(){view.setAttribute('transform','translate('+OX.toFixed(1)+' '+OY.toFixed(1)+') scale('+Z.toFixed(3)+')');}
function toSvg(e){var r=svg.getBoundingClientRect();return [(e.clientX-r.left)/r.width*S,(e.clientY-r.top)/r.height*S];}
function zoomAt(f,cx,cy){Z*=f;OX=cx-(cx-OX)*f;OY=cy-(cy-OY)*f;render();}
function reset(){Z=1;OX=0;OY=0;render();}
root.querySelectorAll('.tb button').forEach(function(b){b.addEventListener('click',function(){var a=b.getAttribute('data-act');
 if(a==='zin')zoomAt(1.25,S/2,S/2);else if(a==='zout')zoomAt(0.8,S/2,S/2);else reset();});});
svg.addEventListener('wheel',function(e){e.preventDefault();var c=toSvg(e);zoomAt(e.deltaY<0?1.25:0.8,c[0],c[1]);},{passive:false});
svg.addEventListener('mousedown',function(e){e.preventDefault();drag={x:e.clientX,y:e.clientY,ox:OX,oy:OY};var w=win(svg);
 function move(ev){if(!drag)return;var r=svg.getBoundingClientRect();OX=drag.ox+(ev.clientX-drag.x)*S/r.width;OY=drag.oy+(ev.clientY-drag.y)*S/r.height;render();}
 function up(){drag=null;w.removeEventListener('mousemove',move);w.removeEventListener('mouseup',up);}
 w.addEventListener('mousemove',move);w.addEventListener('mouseup',up);});
svg.addEventListener('dblclick',reset);
/* tooltips */
root.querySelectorAll('[data-tip]').forEach(function(e){
 e.addEventListener('mousemove',function(ev){tip.innerHTML=e.getAttribute('data-tip');tip.style.display='block';var r=root.getBoundingClientRect();tip.style.left=(ev.clientX-r.left+14)+'px';tip.style.top=(ev.clientY-r.top+14)+'px';});
 e.addEventListener('mouseleave',function(){tip.style.display='none';});});
paint();
})();
