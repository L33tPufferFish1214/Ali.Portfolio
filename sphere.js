/* Gold particle sphere — a from-scratch WebGL2 rebuild of the old Spline scene.
   Every number in S comes from that scene's settings (emitter, curl noise, force
   collider, cursor-follow, orbit limits, camera zoom), so it behaves the same:
   - particles are born on a sphere, drift outward and swirl in curl noise, fading out
   - an invisible ball keeps the shell hollow and follows the cursor, pushing particles
   - drag orbits the view (soft limits, springs back); right-drag pans
   Physics runs on the GPU (transform feedback), so 20k particles cost almost nothing. */

const S = {
    radius: 212.5,          // emitter sphere (425 wide)
    colliderRadius: 210,    // force ball (420 wide)
    rate: 2000,             // births per second
    life: 10,               // seconds
    // Spline's speed/noise numbers use its own internal units, so these were tuned by eye
    // side-by-side against the original (Spline: speed 5, gravity 0.2, noise strength 10, scale 6)
    speed: 0.6,             // outward launch speed
    gravity: 0.025,         // scaled by the same factor as speed (5 -> 0.6)
    noiseStrength: 5.5,     // curl-noise swirl
    noiseFreq: 0.014,       // curl-noise feature size (~70 units)
    radialNoise: 0.3,       // share of the swirl allowed to push in/out of the shell (rest flows along it)
    bounce: 1,
    zoom: 1.108,            // orthographic camera zoom (px per unit at the 980x900 frame)
    frame: [980, 900],
    pointSize: 1.9,         // Spline: 2 with its soft "dot blur" sprite; 1.9 matches its brightness
    color1: [0.8412, 0.5957, 0.2858],
    color2: [0.9528, 0.848, 0.7123],
    followDamping: 4,       // how fast the force ball chases the cursor (Spline: 10; slowed on request)
    resetSpeed: 5,          // how fast it drifts back home when the cursor leaves (Spline: 5)
    hoverArea: 1.15,        // reacts within 1.15x the sphere radius, not the whole canvas
    yawLimit: 0.4363,
    pitchLimit: 0.6109,
    panLimit: 250,
    dragSpeed: 0.006,       // radians per px of drag
};

const NOISE_GLSL = `
vec3 mod289(vec3 x){return x-floor(x*(1.0/289.0))*289.0;}
vec4 mod289(vec4 x){return x-floor(x*(1.0/289.0))*289.0;}
vec4 permute(vec4 x){return mod289(((x*34.0)+1.0)*x);}
vec4 taylorInvSqrt(vec4 r){return 1.79284291400159-0.85373472095314*r;}
float snoise(vec3 v){
  const vec2 C=vec2(1.0/6.0,1.0/3.0); const vec4 D=vec4(0.0,0.5,1.0,2.0);
  vec3 i=floor(v+dot(v,C.yyy)); vec3 x0=v-i+dot(i,C.xxx);
  vec3 g=step(x0.yzx,x0.xyz); vec3 l=1.0-g;
  vec3 i1=min(g.xyz,l.zxy); vec3 i2=max(g.xyz,l.zxy);
  vec3 x1=x0-i1+C.xxx; vec3 x2=x0-i2+C.yyy; vec3 x3=x0-D.yyy;
  i=mod289(i);
  vec4 p=permute(permute(permute(i.z+vec4(0.0,i1.z,i2.z,1.0))+i.y+vec4(0.0,i1.y,i2.y,1.0))+i.x+vec4(0.0,i1.x,i2.x,1.0));
  float n_=0.142857142857; vec3 ns=n_*D.wyz-D.xzx;
  vec4 j=p-49.0*floor(p*ns.z*ns.z);
  vec4 x_=floor(j*ns.z); vec4 y_=floor(j-7.0*x_);
  vec4 x=x_*ns.x+ns.yyyy; vec4 y=y_*ns.x+ns.yyyy; vec4 h=1.0-abs(x)-abs(y);
  vec4 b0=vec4(x.xy,y.xy); vec4 b1=vec4(x.zw,y.zw);
  vec4 s0=floor(b0)*2.0+1.0; vec4 s1=floor(b1)*2.0+1.0; vec4 sh=-step(h,vec4(0.0));
  vec4 a0=b0.xzyw+s0.xzyw*sh.xxyy; vec4 a1=b1.xzyw+s1.xzyw*sh.zzww;
  vec3 p0=vec3(a0.xy,h.x); vec3 p1=vec3(a0.zw,h.y); vec3 p2=vec3(a1.xy,h.z); vec3 p3=vec3(a1.zw,h.w);
  vec4 norm=taylorInvSqrt(vec4(dot(p0,p0),dot(p1,p1),dot(p2,p2),dot(p3,p3)));
  p0*=norm.x; p1*=norm.y; p2*=norm.z; p3*=norm.w;
  vec4 m=max(0.6-vec4(dot(x0,x0),dot(x1,x1),dot(x2,x2),dot(x3,x3)),0.0); m=m*m;
  return 42.0*dot(m*m,vec4(dot(p0,x0),dot(p1,x1),dot(p2,x2),dot(p3,x3)));
}
vec3 potential(vec3 p){
  return vec3(snoise(p), snoise(p+vec3(31.416,-17.2,5.3)), snoise(p+vec3(-9.1,47.7,22.6)));
}
vec3 curlNoise(vec3 p){
  const float e=0.1;
  vec3 dx=vec3(e,0.0,0.0), dy=vec3(0.0,e,0.0), dz=vec3(0.0,0.0,e);
  vec3 x0=potential(p-dx), x1=potential(p+dx);
  vec3 y0=potential(p-dy), y1=potential(p+dy);
  vec3 z0=potential(p-dz), z1=potential(p+dz);
  return vec3((y1.z-y0.z)-(z1.y-z0.y), (z1.x-z0.x)-(x1.z-x0.z), (x1.y-x0.y)-(y1.x-y0.x))/(2.0*e);
}`;

// state per particle: position (3), velocity (3), age (1). age < 0 = not born yet.
const UPDATE_VS = `#version 300 es
precision highp float;
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aVel;
layout(location=2) in float aAge;
out vec3 vPos; out vec3 vVel; out float vAge;
uniform float uDt, uLife, uRadius, uSpeed, uGravity, uNoiseStrength, uNoiseFreq, uColR, uBounce, uRadialNoise;
uniform vec3 uCollider;
uniform uint uFrame;
${NOISE_GLSL}
uint pcg(uint v){ uint s=v*747796405u+2891336453u; uint w=((s>>((s>>28u)+4u))^s)*277803737u; return (w>>22u)^w; }
float rnd(inout uint s){ s=pcg(s); return float(s)/4294967295.0; }
void spawn(inout vec3 p, inout vec3 v){
  uint s=uint(gl_VertexID)*1973u+uFrame*9277u+1u;
  float z=rnd(s)*2.0-1.0, a=rnd(s)*6.2831853, r=sqrt(1.0-z*z);
  vec3 n=vec3(r*cos(a), z, r*sin(a));
  p=n*uRadius; v=n*uSpeed;
}
void main(){
  vec3 p=aPos, v=aVel; float age=aAge;
  if(age<0.0){
    age+=uDt;
    if(age>=0.0){ spawn(p,v); age=0.0; }
  } else {
    age+=uDt;
    if(age>=uLife){ spawn(p,v); age=0.0; }
    v.y-=uGravity*uDt;
    // swirl mostly along the shell (only uRadialNoise of it may push in/out), so the sphere stays
    // a tight shell like the original; flow along the surface also bunches particles into strands
    vec3 swirl=curlNoise(p*uNoiseFreq)*uNoiseStrength;
    vec3 rad=normalize(p+1e-5);
    swirl-=dot(swirl,rad)*rad*(1.0-uRadialNoise);
    p+=(v+swirl)*uDt;
    // force ball: push particles out of it and bounce them off its surface
    vec3 d=p-uCollider; float len=length(d);
    if(len<uColR && len>0.0001){
      vec3 n=d/len; p=uCollider+n*uColR;
      float vn=dot(v,n); if(vn<0.0) v-=(1.0+uBounce)*vn*n;
    }
  }
  vPos=p; vVel=v; vAge=age;
}`;

const UPDATE_FS = `#version 300 es
precision mediump float; out vec4 o; void main(){ o=vec4(0.0); }`;

const RENDER_VS = `#version 300 es
precision highp float;
layout(location=0) in vec3 aPos;
layout(location=2) in float aAge;
uniform mat3 uView; uniform vec2 uScale; uniform vec2 uPan; uniform float uLife, uPointSize;
uniform vec3 uColor1, uColor2;
out vec3 vColor; out float vAlpha;
void main(){
  if(aAge<0.0){ gl_Position=vec4(2.0,2.0,0.0,1.0); gl_PointSize=0.0; return; }
  vec3 q=uView*aPos;
  gl_Position=vec4((q.xy+uPan)*uScale,0.0,1.0);
  gl_PointSize=uPointSize;
  float t=clamp(aAge/uLife,0.0,1.0);
  vColor=mix(uColor1,uColor2,t);
  vAlpha=1.0-t;                       // linear fade-out over life
}`;

const RENDER_FS = `#version 300 es
precision mediump float;
in vec3 vColor; in float vAlpha; out vec4 o;
void main(){
  vec2 d=gl_PointCoord*2.0-1.0; float r2=dot(d,d);
  if(r2>1.0) discard;
  float a=exp(-r2*2.5)*vAlpha;        // soft "dot blur" sprite
  o=vec4(vColor*a,a);                 // premultiplied alpha
}`;

const canvas = document.getElementById('sphere');
const gl = canvas && canvas.getContext('webgl2', { alpha: true, premultipliedAlpha: true, antialias: false });

if (gl) init(); else if (canvas) canvas.remove(); // no WebGL2: page works fine without it

function compile(type, src) {
    const sh = gl.createShader(type);
    gl.shaderSource(sh, src);
    gl.compileShader(sh);
    if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(sh));
    return sh;
}

function program(vs, fs, varyings) {
    const p = gl.createProgram();
    gl.attachShader(p, compile(gl.VERTEX_SHADER, vs));
    gl.attachShader(p, compile(gl.FRAGMENT_SHADER, fs));
    if (varyings) gl.transformFeedbackVaryings(p, varyings, gl.INTERLEAVED_ATTRIBS);
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
    const u = {};
    for (let i = 0, n = gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS); i < n; i++) {
        const name = gl.getActiveUniform(p, i).name;
        u[name] = gl.getUniformLocation(p, name);
    }
    return { p, u };
}

function init() {
    let update, render;
    try {
        update = program(UPDATE_VS, UPDATE_FS, ['vPos', 'vVel', 'vAge']);
        render = program(RENDER_VS, RENDER_FS);
    } catch (e) {
        console.error('sphere: shader error', e);
        canvas.remove();
        return;
    }

    // phones/tablets: 12k particles instead of 20k and a lower render resolution (the sphere
    // is display-only and dimmed there, so the difference isn't visible but the battery is)
    const lowPower = matchMedia('(max-width: 768px), (pointer: coarse)').matches;
    if (lowPower) S.rate = 1200;
    const maxDpr = lowPower ? 1.5 : 2;
    const COUNT = S.rate * S.life; // 20,000 alive at steady state on desktop, same as the original
    const STRIDE = 7 * 4;
    const init = new Float32Array(COUNT * 7);
    // staggered birth times: the sphere fills in over the first 10s, like the original
    for (let i = 0; i < COUNT; i++) init[i * 7 + 6] = -(i / COUNT) * S.life;

    const buffers = [0, 1].map(() => {
        const b = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, b);
        gl.bufferData(gl.ARRAY_BUFFER, init, gl.DYNAMIC_COPY);
        return b;
    });
    const vaos = buffers.map(b => {
        const vao = gl.createVertexArray();
        gl.bindVertexArray(vao);
        gl.bindBuffer(gl.ARRAY_BUFFER, b);
        gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 3, gl.FLOAT, false, STRIDE, 0);
        gl.enableVertexAttribArray(1); gl.vertexAttribPointer(1, 3, gl.FLOAT, false, STRIDE, 12);
        gl.enableVertexAttribArray(2); gl.vertexAttribPointer(2, 1, gl.FLOAT, false, STRIDE, 24);
        return vao;
    });
    gl.bindVertexArray(null);
    gl.bindBuffer(gl.ARRAY_BUFFER, null); // a buffer can't be bound elsewhere while transform feedback writes to it
    const tf = gl.createTransformFeedback();
    let cur = 0, frame = 0;

    // interaction state (units = scene units; collider/pan are relative to the sphere center)
    const collider = [0, 0, 0], target = [0, 0, 0];
    let following = false, yaw = 0, pitch = 0, pan = [0, 0];
    let drag = null;

    let cssW = 1, cssH = 1, ppu = 1;
    function resize() {
        const dpr = Math.min(devicePixelRatio || 1, maxDpr);
        cssW = canvas.clientWidth || 1;
        cssH = canvas.clientHeight || 1;
        canvas.width = Math.round(cssW * dpr);
        canvas.height = Math.round(cssH * dpr);
        ppu = S.zoom * Math.min(cssW / S.frame[0], cssH / S.frame[1]); // css px per scene unit
        if (reduceMotion) draw();
    }

    function viewMatrix() { // pitch * yaw, column-major mat3
        const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
        return new Float32Array([cy, sp * sy, -cp * sy, 0, cp, sp, sy, -sp * cy, cp * cy]);
    }

    function step(dt) {
        // collider eases toward the cursor (or back home when the cursor leaves)
        const k = 1 - Math.exp(-(following ? S.followDamping : S.resetSpeed) * dt);
        for (let i = 0; i < 3; i++) collider[i] += ((following ? target[i] : 0) - collider[i]) * k;
        // soft limits: after a drag, the view springs back inside its limits
        if (!drag) {
            const back = 1 - Math.exp(-8 * dt);
            yaw -= (yaw - clamp(yaw, S.yawLimit)) * back;
            pitch -= (pitch - clamp(pitch, S.pitchLimit)) * back;
            pan = pan.map(v => v - (v - clamp(v, S.panLimit)) * back);
        }

        gl.useProgram(update.p);
        gl.uniform1f(update.u.uDt, dt);
        gl.uniform1f(update.u.uLife, S.life);
        gl.uniform1f(update.u.uRadius, S.radius);
        gl.uniform1f(update.u.uSpeed, S.speed);
        gl.uniform1f(update.u.uGravity, S.gravity);
        gl.uniform1f(update.u.uNoiseStrength, S.noiseStrength);
        gl.uniform1f(update.u.uNoiseFreq, S.noiseFreq);
        gl.uniform1f(update.u.uColR, S.colliderRadius);
        gl.uniform1f(update.u.uBounce, S.bounce);
        gl.uniform1f(update.u.uRadialNoise, S.radialNoise);
        gl.uniform3fv(update.u.uCollider, collider);
        gl.uniform1ui(update.u.uFrame, frame++);
        gl.bindVertexArray(vaos[cur]);
        gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, tf);
        gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, 0, buffers[1 - cur]);
        gl.enable(gl.RASTERIZER_DISCARD);
        gl.beginTransformFeedback(gl.POINTS);
        gl.drawArrays(gl.POINTS, 0, COUNT);
        gl.endTransformFeedback();
        gl.disable(gl.RASTERIZER_DISCARD);
        gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, 0, null);
        gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, null);
        cur = 1 - cur;
    }

    function draw() {
        const dpr = canvas.width / cssW;
        gl.viewport(0, 0, canvas.width, canvas.height);
        gl.clearColor(0, 0, 0, 0);
        gl.clear(gl.COLOR_BUFFER_BIT);
        gl.enable(gl.BLEND);
        gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
        gl.useProgram(render.p);
        gl.uniformMatrix3fv(render.u.uView, false, viewMatrix());
        gl.uniform2f(render.u.uScale, (2 * ppu) / cssW, (2 * ppu) / cssH);
        gl.uniform2f(render.u.uPan, pan[0], pan[1]);
        gl.uniform1f(render.u.uLife, S.life);
        gl.uniform1f(render.u.uPointSize, Math.max(1, S.pointSize * ppu * dpr));
        gl.uniform3fv(render.u.uColor1, S.color1);
        gl.uniform3fv(render.u.uColor2, S.color2);
        gl.bindVertexArray(vaos[cur]);
        gl.drawArrays(gl.POINTS, 0, COUNT);
        gl.bindVertexArray(null);
    }

    const clamp = (v, lim) => Math.max(-lim, Math.min(lim, v));

    // cursor -> point on the plane through the sphere center facing the camera
    function cursorToScene(e) {
        const r = canvas.getBoundingClientRect();
        // getBoundingClientRect includes CSS scale; convert back to the canvas's own css px
        const sx = cssW / r.width, sy = cssH / r.height;
        const vx = ((e.clientX - r.left) * sx - cssW / 2) / ppu - pan[0];
        const vy = (cssH / 2 - (e.clientY - r.top) * sy) / ppu - pan[1];
        const m = viewMatrix(); // view space -> scene space is the transpose
        target[0] = m[0] * vx + m[1] * vy;
        target[1] = m[3] * vx + m[4] * vy;
        target[2] = m[6] * vx + m[7] * vy;
        return Math.hypot(vx, vy); // distance from the sphere center, in scene units
    }

    // only the sphere itself (plus a small margin) reacts, not the whole canvas
    const overSphere = e => cursorToScene(e) <= S.radius * S.hoverArea;

    canvas.addEventListener('pointermove', e => {
        const inside = overSphere(e);
        following = inside || !!drag;
        canvas.style.cursor = inside || drag ? (drag ? 'grabbing' : 'grab') : 'default';
        if (!drag) return;
        const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
        drag.x = e.clientX; drag.y = e.clientY;
        if (drag.pan) {
            pan[0] += dx / ppu; pan[1] -= dy / ppu;
        } else {
            // past a limit the drag gets heavier (soft limit), then springs back on release
            const resist = (v, lim) => (Math.abs(v) > lim ? 0.3 : 1);
            yaw += dx * S.dragSpeed * resist(yaw, S.yawLimit);
            pitch += dy * S.dragSpeed * resist(pitch, S.pitchLimit);
        }
    });
    canvas.addEventListener('pointerdown', e => {
        if (!overSphere(e)) return;
        drag = { x: e.clientX, y: e.clientY, pan: e.button === 2 || e.shiftKey };
        canvas.setPointerCapture(e.pointerId);
    });
    const endDrag = () => { drag = null; };
    canvas.addEventListener('pointerup', endDrag);
    canvas.addEventListener('pointercancel', endDrag);
    canvas.addEventListener('pointerleave', () => { if (!drag) following = false; });
    canvas.addEventListener('contextmenu', e => e.preventDefault()); // right-drag pans, like the original

    const reduceMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;
    new ResizeObserver(resize).observe(canvas);
    resize();

    if (reduceMotion) {
        // no animation: settle the sphere to its full state once and show a still frame
        for (let i = 0; i < S.life * 30; i++) step(1 / 30);
        draw();
        return;
    }

    // only animate while the sphere is on screen (rAF already pauses in hidden tabs)
    let visible = true, last = 0, raf = 0;
    function loop(t) {
        // while an Experience/Projects panel covers it, the sphere is only seen through frosted
        // glass: run it at ~30fps so the browser isn't re-blurring it 60+ times a second
        if (last && t - last < 30 && document.querySelector('dialog[open]')) {
            raf = requestAnimationFrame(loop);
            return;
        }
        const dt = last ? Math.min((t - last) / 1000, 1 / 20) : 1 / 60;
        last = t;
        step(dt);
        draw();
        raf = requestAnimationFrame(loop);
    }
    new IntersectionObserver(([entry]) => {
        visible = entry.isIntersecting;
        if (visible && !raf) { last = 0; raf = requestAnimationFrame(loop); }
        if (!visible && raf) { cancelAnimationFrame(raf); raf = 0; }
    }).observe(canvas);
    raf = requestAnimationFrame(loop);
}
