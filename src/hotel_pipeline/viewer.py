"""Viewer 3D autonome pour la démonstration locale.

Le viewer est un livrable de présentation, pas un nouveau Gate métier. Il
embarque son payload et ne dépend d'aucun CDN : un double clic suffit, même
hors ligne. Le manifeste voisin conserve les empreintes des données montrées
et rappelle explicitement ce qui est différé pour la démonstration.
"""

from __future__ import annotations

import hashlib
import json
import re
import webbrowser
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .workspace import Workspace

VIEWER_VERSION = "demo-viewer-1.13.0"
PAYLOAD_RE = re.compile(
    r"const PAYLOAD = (\{.*?\});\s*\n\s*const (?:cv|MANIFEST)", re.DOTALL
)


@dataclass(frozen=True)
class ViewerOutputs:
    html: Path
    payload: Path
    manifest: Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text("utf-8"))


def _legacy_payload(path: Path) -> dict | None:
    """Migre le payload du viewer pilote produit avant la commande dédiée."""
    if not path.is_file():
        return None
    match = PAYLOAD_RE.search(path.read_text("utf-8"))
    if match is None:
        return None
    payload = json.loads(match.group(1))
    return payload if isinstance(payload, dict) else None


def _obj_payload(workspace: Workspace) -> dict:
    """Repli générique : rend le maillage OBJ du paquet canonique."""
    pointer_path = workspace.path("08_composite", "scene_package_current.json")
    if not pointer_path.is_file():
        raise FileNotFoundError(
            "aucun payload historique ni paquet de scène courant ; lancez "
            f"d'abord `hotel-pipeline scene build {workspace.hotel_id}`"
        )
    pointer = _read_json(pointer_path)
    scene_path = workspace.root / str(pointer["manifest"])
    scene = _read_json(scene_path)
    obj_path = scene_path.parent / "environment.obj"
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    for raw in obj_path.read_text("utf-8").splitlines():
        fields = raw.split()
        if not fields:
            continue
        if fields[0] == "v" and len(fields) >= 4:
            vertices.append([float(fields[1]), float(fields[2]), float(fields[3])])
        elif fields[0] == "f" and len(fields) >= 4:
            faces.append([int(token.split("/", 1)[0]) - 1 for token in fields[1:]])
    if not vertices or not faces:
        raise ValueError(f"maillage OBJ vide ou illisible : {obj_path}")
    return {
        "hotel": workspace.hotel_id,
        "mesh": {"vertices": vertices, "faces": faces},
        "volumes": [],
        "vegetation": [],
        "furniture": [],
        "ground": [],
        "ridges": [],
        "observation": {"cells": [], "missing": []},
        "counts": {"volumes": 1, "roof_triangles": len(faces)},
        "source_scene": str(scene_path.relative_to(workspace.root)),
    }


def _source_digests(workspace: Workspace) -> dict[str, str]:
    candidates = {
        "capture_geometry": workspace.path("06_geo", "capture_geometry.json"),
        "observation_map": workspace.path("06_geo", "observation_map.json"),
        "ridge_match": workspace.path("06_geo", "ridge_match.json"),
        "conditioning_report": workspace.path(
            "11_conditioning", "orbit", "conditioning_report.json"
        ),
        "fidelity_audit": workspace.path("09_confidence", "fidelity_audit.json"),
        "conditioned_scene": workspace.path(
            "11_conditioning", "conditioned_scene.json"
        ),
        "architectural_observations": workspace.path(
            "11_conditioning", "architectural_observations.json"
        ),
        "semantic_observations": workspace.path(
            "11_conditioning", "semantic_observations.json"
        ),
        "semantic_correspondences": workspace.path(
            "11_conditioning", "semantic_correspondences.json"
        ),
        "vertical_registration": workspace.path(
            "11_conditioning", "vertical_registration.json"
        ),
        "registered_semantic_support": workspace.path(
            "11_conditioning", "registered_semantic_support.json"
        ),
        "semantic_surfaces": workspace.path(
            "11_conditioning", "semantic_surfaces.json"
        ),
        "feed_forward_shape_audit": workspace.path(
            "11_conditioning", "feed_forward_shape_audit.json"
        ),
    }
    return {
        name: _sha256(path)
        for name, path in candidates.items()
        if path.is_file()
    }


def _safe_json(value: object) -> str:
    # Un identifiant contenant cette séquence ne doit jamais fermer le script.
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).replace(
        "</script", "<\\/script"
    )


def _html(payload: dict, manifest: dict) -> str:
    payload_json = _safe_json(payload)
    manifest_json = _safe_json(manifest)
    return f"""<!doctype html>
<html lang=\"fr\"><head><meta charset=\"utf-8\">
<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">
<title>BuildingDroneVisit · Démonstration 3D</title>
<style>
:root{{--bg:#101722;--panel:#101822e8;--ink:#f6f8fa;--muted:#a8b4c2;--line:#344252;
--target:#70493e;--other:#71808c;--roof:#4f5b56;--grass:#66835c;--road:#737b83;
--veg:#4f805d;--pole:#c3cad2;--ok:#52b86c;--thin:#e2a23f;--none:#dc5960;--plan:#4c9ee8}}
*{{box-sizing:border-box}}html,body{{margin:0;width:100%;height:100%;overflow:hidden;background:var(--bg);
color:var(--ink);font:14px/1.45 ui-sans-serif,-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif}}
canvas{{width:100%;height:100%;display:block;cursor:grab;touch-action:none}}canvas:active{{cursor:grabbing}}
.panel{{position:fixed;background:var(--panel);border:1px solid var(--line);border-radius:12px;
backdrop-filter:blur(12px);box-shadow:0 14px 40px #0007}}#hud{{top:14px;left:14px;width:282px;padding:14px;max-height:calc(100vh - 28px);overflow:auto;scrollbar-width:thin;scrollbar-color:#46505e transparent}}
.ui-hidden .panel{{opacity:0;pointer-events:none}}.panel{{transition:opacity .18s ease}}
h1{{font-size:16px;margin:0 0 2px}}.sub,.note{{color:var(--muted);font-size:11px}}.badge{{display:inline-block;
margin:8px 0 9px;padding:3px 8px;border-radius:999px;background:#55311f;color:#ffc39f;font-size:10px;font-weight:700}}
.row{{display:flex;justify-content:space-between;gap:12px;border-top:1px solid var(--line);padding:5px 0;font-variant-numeric:tabular-nums;font-size:12px}}
.row span{{color:var(--muted)}}details summary{{cursor:pointer;color:#cbd4de;font-size:11px;list-style:none}}details summary::-webkit-details-marker{{display:none}}
#technical{{margin-top:7px;border-top:1px solid var(--line);padding-top:7px}}#technical summary:after{{content:' +';float:right}}#technical[open] summary:after{{content:' −'}}
.keys{{margin-top:9px;color:var(--muted);font-size:10px;line-height:1.75}}
kbd{{border:1px solid var(--line);border-radius:4px;padding:1px 5px;color:var(--ink);background:#202834}}
#legend{{right:14px;top:14px;padding:9px 12px;font-size:10px}}#legend>summary{{font-weight:700;letter-spacing:.08em}}#legend[open]>summary{{margin-bottom:7px}}
.lg{{display:flex;gap:7px;align-items:center;margin:3px}}.sw{{width:20px;height:3px;border-radius:3px}}
#scope{{right:14px;bottom:14px;max-width:315px;padding:10px 12px;color:var(--muted);font-size:10px}}#scope b{{color:var(--ink)}}
#toolbar{{left:14px;bottom:14px;padding:6px;display:flex;gap:5px}}button{{appearance:none;border:1px solid #3b4a5a;border-radius:8px;background:#172230;color:#b9c5d1;padding:7px 10px;font:600 10px/1 ui-sans-serif,sans-serif;cursor:pointer}}button:hover,button.active{{background:#29415a;color:#fff;border-color:#5d7892}}
@media(max-width:700px){{#hud{{width:250px}}#scope{{max-width:270px}}#toolbar button{{padding:7px}}}}
</style></head><body><canvas id=\"cv\"></canvas>
<section id=\"hud\" class=\"panel\"><h1 id=\"title\">Scène 3D</h1>
<div class=\"sub\">Géométrie LiDAR et contraintes de reconstruction</div>
<div class=\"badge\">MODE DÉMONSTRATION</div>
<div class=\"row\"><span>Volumes</span><b id=\"nvol\">–</b></div>
<div class=\"row\"><span>Triangles de toiture</span><b id=\"ntri\">–</b></div>
<div class=\"row\"><span>Végétation</span><b id=\"nveg\">–</b></div>
<div class=\"row\"><span>Solides étanches</span><b id=\"nsolid\">–</b></div>
<div class=\"row\"><span>Ressemblance structurelle</span><b id=\"nsimilarity\">–</b></div>
<div class=\"row\"><span>Forme GPU multi-vues</span><b id=\"nshape\">–</b></div>
<details id=\"technical\"><summary>Données techniques</summary>
<div class=\"row\"><span>Vues IA validées</span><b id=\"nsemviews\">–</b></div>
<div class=\"row\"><span>Masques IA 2D</span><b id=\"nsemmasks\">–</b></div>
<div class=\"row\"><span>Instances IA multi-vues</span><b id=\"nseminstances\">–</b></div>
<div class=\"row\"><span>Pistes SfM partagées</span><b id=\"nsemtracks\">–</b></div>
<div class=\"row\"><span>Alignement COLMAP/LiDAR</span><b id=\"nregistration\">–</b></div>
<div class=\"row\"><span>Points SfM recalés</span><b id=\"nregisteredpoints\">–</b></div>
<div class=\"row\"><span>Hypothèses linéaires mono-vue</span><b id=\"nsingleview\">–</b></div>
<div class=\"row\"><span>Surfaces 3D contraintes</span><b id=\"nsem3d\">–</b></div>
<div class=\"row\"><span>Couverture triangulable</span><b id=\"nobs\">–</b></div>
<div class=\"row\"><span>Azimut / altitude</span><b id=\"camera\">–</b></div>
<div class=\"keys\"><kbd>glisser</kbd> orbiter · <kbd>molette</kbd> zoom · <kbd>espace</kbd> rotation<br>
<kbd>R</kbd> toits · <kbd>V</kbd> volumes · <kbd>P</kbd> végétation · <kbd>G</kbd> sol<br>
<kbd>I</kbd> supports IA/SfM ·
<kbd>O</kbd> couverture · <kbd>N</kbd> prises proposées · <kbd>F</kbd> faîtages · <kbd>W</kbd> filaire<br>
<kbd>1</kbd> bâtiment · <kbd>2</kbd> aérien · <kbd>3</kbd> contexte · <kbd>H</kbd> interface</div></details></section>
<details id=\"legend\" class=\"panel\"><summary>LÉGENDE</summary>
<div class=\"lg\"><i class=\"sw\" style=\"background:var(--target)\"></i>Bâtiment cible</div>
<div class=\"lg\"><i class=\"sw\" style=\"background:var(--roof)\"></i>Toiture mesurée · contour logique</div>
<div class=\"lg\"><i class=\"sw\" style=\"background:#263b46\"></i>Fenêtres · grammaire de façade</div>
<div class=\"lg\"><i class=\"sw\" style=\"background:#6f483c\"></i>Entrée et porche inférés</div>
<div class=\"lg\"><i class=\"sw\" style=\"background:var(--ok)\"></i>Mur triangulable</div>
<div class=\"lg\"><i class=\"sw\" style=\"background:var(--none)\"></i>Mur non observé</div>
<div class=\"lg\"><i class=\"sw\" style=\"background:var(--plan)\"></i>Prise proposée</div>
<div class=\"lg\"><i class=\"sw\" style=\"background:#ff55c8\"></i>Support SfM multi-vues</div>
<div class=\"lg\"><i class=\"sw\" style=\"background:#ff9f43\"></i>Hypothèse mono-vue</div></details>
<section id=\"scope\" class=\"panel\"><b>Orthofaçades photographiques sur masse LiDAR.</b> Les atlas sont fusionnés depuis les vues COLMAP enregistrées ; les détails non couverts restent procéduraux.<div id=\"shape-note\" class=\"note\"></div></section>
<nav id=\"toolbar\" class=\"panel\" aria-label=\"Vues\"><button id=\"view-1\" class=\"active\" onclick=\"preset('1')\">FAÇADE</button><button id=\"view-2\" onclick=\"preset('2')\">AÉRIEN</button><button id=\"view-3\" onclick=\"preset('3')\">CONTEXTE</button></nav>
<script>
const PAYLOAD = {payload_json};
const MANIFEST = {manifest_json};
const cv=document.getElementById('cv'),ctx=cv.getContext('2d');let W=0,H=0,D=1,renderCx=0;
const CAMERA=PAYLOAD.camera||{{}};
const APPEARANCE=PAYLOAD.appearance_profile||{{}},TEXTURE_BY_SURFACE=new Map((PAYLOAD.facade_textures||[]).map(t=>[t.surface_id,t]));
const TEXTURE_IMAGES=new Map();for(const texture of PAYLOAD.facade_textures||[]){{const image=new Image();image.onload=()=>draw();image.src=texture.path;TEXTURE_IMAGES.set(texture.surface_id,image);}}
function volumeBounds(targetOnly=false){{const pts=[];for(const v of PAYLOAD.volumes||[]){{if(targetOnly&&!v.target)continue;for(const p of v.fp||[])pts.push(p);}}if(!pts.length&&targetOnly)return volumeBounds(false);if(!pts.length)return{{focus:[0,0,8],span:120}};const xs=pts.map(p=>p[0]),ys=pts.map(p=>p[1]),zs=(PAYLOAD.volumes||[]).filter(v=>!targetOnly||v.target).map(v=>v.h||8);const dx=Math.max(...xs)-Math.min(...xs),dy=Math.max(...ys)-Math.min(...ys);return{{focus:[(Math.min(...xs)+Math.max(...xs))/2,(Math.min(...ys)+Math.max(...ys))/2,(Math.max(...zs,8))*.42],span:Math.max(25,Math.hypot(dx,dy))}};}}
const TARGET=volumeBounds(true),CONTEXT=volumeBounds(false);let focus=CAMERA.focus||TARGET.focus;
let az=(CAMERA.facade_azimuth_deg??210)*Math.PI/180,alt=(CAMERA.facade_altitude_deg??1)*Math.PI/180,dist=(CAMERA.target_distance_m??Math.max(35,TARGET.span*1.15)),spin=false,wire=false,activeView='1';
let show={{roof:true,vol:true,veg:false,ground:true,obs:false,plan:false,ridge:false,ai:false}};
const C={{site:'#222d36',target:APPEARANCE.brick||'#70493e',other:'#74818b',roof:APPEARANCE.roof||'#4f5b56',grass:'#607a58',road:'#424b52',veg:APPEARANCE.vegetation||'#426d51',pole:'#9aa7b3',mesh:APPEARANCE.brick||'#70493e',semantic:'#ff55c8',window:APPEARANCE.glass||'#26363d',arched_window:APPEARANCE.glass||'#26363d',door:'#172b34',band:'#3b3534',canopy:APPEARANCE.brick||'#67463c',pier:APPEARANCE.brick||'#70493e',gable:APPEARANCE.brick||'#7d4e40',entrance_tower:APPEARANCE.brick||'#7d4e40',sign:'#20334f',sign_post:'#202a33'}};
function resize(){{D=Math.min(devicePixelRatio||1,2);W=cv.clientWidth;H=cv.clientHeight;cv.width=W*D;cv.height=H*D;ctx.setTransform(D,0,0,D,0,0);}}
function basis(){{const ca=Math.cos(alt),sa=Math.sin(alt),eye=[focus[0]+Math.cos(az)*ca*dist,focus[1]+Math.sin(az)*ca*dist,focus[2]+sa*dist];
let f=[focus[0]-eye[0],focus[1]-eye[1],focus[2]-eye[2]],l=Math.hypot(...f);f=f.map(x=>x/l);let r=[f[1],-f[0],0];l=Math.hypot(...r)||1;r=r.map(x=>x/l);
const u=[r[1]*f[2],-r[0]*f[2],r[0]*f[1]-r[1]*f[0]];return{{eye,f,r,u}};}}
function project(p,b){{const d=[p[0]-b.eye[0],p[1]-b.eye[1],p[2]-b.eye[2]],z=d[0]*b.f[0]+d[1]*b.f[1]+d[2]*b.f[2];if(z<.5)return null;
const x=d[0]*b.r[0]+d[1]*b.r[1]+d[2]*b.r[2],y=d[0]*b.u[0]+d[1]*b.u[1]+d[2]*b.u[2],q=H*.5/Math.tan(Math.PI/6);return[(renderCx||W/2)+q*x/z,H*.49-q*y/z,z];}}
function sitePlane(){{const pts=[];for(const g of PAYLOAD.ground||[])for(const p of g.ring||[])pts.push(p);for(const v of PAYLOAD.volumes||[])for(const p of v.fp||[])pts.push(p);if(!pts.length)return null;const xs=pts.map(p=>p[0]),ys=pts.map(p=>p[1]),m=Math.max(18,CONTEXT.span*.12);return[[Math.min(...xs)-m,Math.min(...ys)-m,-.18],[Math.max(...xs)+m,Math.min(...ys)-m,-.18],[Math.max(...xs)+m,Math.max(...ys)+m,-.18],[Math.min(...xs)-m,Math.max(...ys)+m,-.18]];}}
function faces(){{const out=[];
const base=sitePlane();if(show.ground&&base)out.push({{p:base,k:'site'}});
if(show.ground)for(const g of PAYLOAD.ground||[])out.push({{p:g.ring.map(v=>[v[0],v[1],0]),k:g.kind==='vegetal'?'grass':'road'}});
if(show.vol)for(const v of PAYLOAD.volumes||[]){{const fp=v.fp||[],wh=v.wh||[],area=fp.reduce((s,p,i)=>s+p[0]*fp[(i+1)%fp.length][1]-fp[(i+1)%fp.length][0]*p[1],0),side=area>=0?1:-1;for(let i=0;i<fp.length;i++){{const j=(i+1)%fp.length,dx=fp[j][0]-fp[i][0],dy=fp[j][1]-fp[i][1];
out.push({{p:[[...fp[i],0],[...fp[j],0],[...fp[j],wh[j]??v.h],[...fp[i],wh[i]??v.h]],k:v.target?'target':'other',normal:[side*dy,-side*dx,0]}});}}
if(show.roof&&v.rv&&v.rf)for(const f of v.rf)out.push({{p:f.map(i=>v.rv[i]),k:'roof'}});
else if(show.roof&&v.solid?.vertices&&v.solid?.faces){{const n=fp.length;for(const f of v.solid.faces)if(f.every(i=>i>=n))out.push({{p:f.map(i=>v.solid.vertices[i]),k:'roof'}});}}}}
if(PAYLOAD.mesh)for(const f of PAYLOAD.mesh.faces||[])out.push({{p:f.map(i=>PAYLOAD.mesh.vertices[i]),k:'mesh'}});
for(const texture of PAYLOAD.facade_textures||[])for(const triangle of texture.render_triangles||[])out.push({{p:triangle.vertices,k:'target',tex:texture,surface:texture.surface_id,uv:triangle.uv_px}});
for(const f of PAYLOAD.facade_features||[])if((f.vertices||[]).length>=3)out.push({{p:f.vertices,k:f.kind||'target',detail:true}});
if(show.ai)for(const s of PAYLOAD.semantic_surfaces||[]){{const g=s.surface||{{}};for(const f of g.faces||[])out.push({{p:f.map(i=>g.vertices[i]),k:'semantic'}});}}
if(show.veg)for(const v of PAYLOAD.vegetation||[]){{let rings=v.rings||[];if(rings.length<2){{const n=12,profile=[.38,.72,1,.94,.7,.34];rings=profile.map((s,l)=>Array.from({{length:n}},(_,i)=>{{const a=i*Math.PI*2/n,q=s*(1+.09*Math.sin(3*a+(v.c[0]+v.c[1])));return[v.c[0]+Math.cos(a)*v.r*q,v.c[1]+Math.sin(a)*v.r*q,v.h*(.12+.84*l/(profile.length-1))]}}));}}
for(let l=0;l<rings.length-1;l++){{const a=rings[l],z=rings[l+1],n=Math.min(a.length,z.length);for(let i=0;i<n;i++)out.push({{p:[a[i],a[(i+1)%n],z[(i+1)%n],z[i]],k:'veg'}});}}if(rings.length){{out.push({{p:[...rings[0]].reverse(),k:'veg'}});out.push({{p:rings[rings.length-1],k:'veg'}});}}}}
for(const f of PAYLOAD.furniture||[]){{const r=Math.max(f.r||.2,.15),x=f.c[0],y=f.c[1];out.push({{p:[[x-r,y-r,0],[x+r,y-r,0],[x+r,y-r,f.h],[x-r,y-r,f.h]],k:'pole'}});}}
return out;}}
function line(a,b,color,width=2,dash=[]){{ctx.save();ctx.strokeStyle=color;ctx.lineWidth=width;ctx.setLineDash(dash);ctx.beginPath();ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1]);ctx.stroke();ctx.restore();}}
function texturedTriangle(image,s,d){{const [s0,s1,s2]=s,[d0,d1,d2]=d,den=s0[0]*(s1[1]-s2[1])+s1[0]*(s2[1]-s0[1])+s2[0]*(s0[1]-s1[1]);if(Math.abs(den)<1e-8)return;const a=(d0[0]*(s1[1]-s2[1])+d1[0]*(s2[1]-s0[1])+d2[0]*(s0[1]-s1[1]))/den,b=(d0[1]*(s1[1]-s2[1])+d1[1]*(s2[1]-s0[1])+d2[1]*(s0[1]-s1[1]))/den,c=(d0[0]*(s2[0]-s1[0])+d1[0]*(s0[0]-s2[0])+d2[0]*(s1[0]-s0[0]))/den,dv=(d0[1]*(s2[0]-s1[0])+d1[1]*(s0[0]-s2[0])+d2[1]*(s1[0]-s0[0]))/den,e=(d0[0]*(s1[0]*s2[1]-s2[0]*s1[1])+d1[0]*(s2[0]*s0[1]-s0[0]*s2[1])+d2[0]*(s0[0]*s1[1]-s1[0]*s0[1]))/den,f=(d0[1]*(s1[0]*s2[1]-s2[0]*s1[1])+d1[1]*(s2[0]*s0[1]-s0[0]*s2[1])+d2[1]*(s0[0]*s1[1]-s1[0]*s0[1]))/den;ctx.save();ctx.beginPath();ctx.moveTo(d0[0],d0[1]);ctx.lineTo(d1[0],d1[1]);ctx.lineTo(d2[0],d2[1]);ctx.closePath();ctx.clip();ctx.transform(a,b,c,dv,e,f);ctx.drawImage(image,0,0);ctx.restore();}}
function texturedQuad(image,q){{const w=image.naturalWidth,h=image.naturalHeight,s=[[0,h],[w,h],[w,0],[0,0]];texturedTriangle(image,[s[0],s[1],s[2]],[q[0],q[1],q[2]]);texturedTriangle(image,[s[0],s[2],s[3]],[q[0],q[2],q[3]]);}}
function frontFacing(face,b){{if(!face.p||face.p.length<3)return true;const a=face.p[0],z=face.p[1],u=face.p[2],ab=[z[0]-a[0],z[1]-a[1],z[2]-a[2]],au=[u[0]-a[0],u[1]-a[1],u[2]-a[2]],n=face.normal||[ab[1]*au[2]-ab[2]*au[1],ab[2]*au[0]-ab[0]*au[2],ab[0]*au[1]-ab[1]*au[0]],c=face.p.reduce((s,p)=>[s[0]+p[0]/face.p.length,s[1]+p[1]/face.p.length,s[2]+p[2]/face.p.length],[0,0,0]);return n[0]*(b.eye[0]-c[0])+n[1]*(b.eye[1]-c[1])+n[2]*(b.eye[2]-c[2])>0;}}
function facadeHeight(c){{let best=Infinity,height=null;for(const v of PAYLOAD.volumes||[]){{const fp=v.fp||[],wh=v.wh||[];for(let i=0;i<fp.length;i++){{const j=(i+1)%fp.length,a=fp[i],z=fp[j],dx=z[0]-a[0],dy=z[1]-a[1],den=dx*dx+dy*dy||1,t=Math.max(0,Math.min(1,((c[0]-a[0])*dx+(c[1]-a[1])*dy)/den)),qx=a[0]+t*dx,qy=a[1]+t*dy,d=(c[0]-qx)**2+(c[1]-qy)**2;if(d<best){{best=d;const ha=wh[i]??v.h??8,hb=wh[j]??v.h??ha;height=ha+t*(hb-ha);}}}}}}return Math.max(.5,height??8);}}
function draw(){{const hidden=document.body.classList.contains('ui-hidden'),hudRight=document.getElementById('hud').getBoundingClientRect().right;renderCx=W>=700&&!hidden?(hudRight+W-14)/2:W/2;const sky=ctx.createLinearGradient(0,0,0,H);sky.addColorStop(0,'#152334');sky.addColorStop(.58,'#253747');sky.addColorStop(1,'#101722');ctx.fillStyle=sky;ctx.fillRect(0,0,W,H);const b=basis(),fs=[];for(const f of faces()){{const q=f.p.map(p=>project(p,b));if(q.some(x=>!x))continue;fs.push({{...f,q,z:q.reduce((s,x)=>s+x[2],0)/q.length}});}}
fs.sort((a,b)=>b.z-a.z);for(const f of fs){{ctx.beginPath();ctx.moveTo(f.q[0][0],f.q[0][1]);for(const q of f.q.slice(1))ctx.lineTo(q[0],q[1]);ctx.closePath();ctx.fillStyle=C[f.k]||C.other;ctx.globalAlpha=f.k==='site'?.82:(f.k==='grass'||f.k==='road')?.88:f.k==='veg'?.28:f.k==='pole'?.58:1;ctx.fill();ctx.globalAlpha=1;const texture=f.tex&&TEXTURE_IMAGES.get(f.surface);if(texture?.complete&&texture.naturalWidth&&f.q.length===3&&f.uv?.length===3&&frontFacing(f,b))texturedTriangle(texture,f.uv,f.q);if(f.k==='target'||f.k==='roof'||f.detail||wire){{ctx.strokeStyle=f.k==='target'?'#a86652':f.k==='roof'?C.roof:f.k==='window'?'#8aa8b5':'#1116';ctx.lineWidth=f.k==='target'?1.05:f.k==='roof'?.85:f.k==='window'?.55:.45;ctx.stroke();}}}}
if(show.obs)for(const o of PAYLOAD.observation?.cells||[]){{const a=project([o.c[0],o.c[1],.2],b),z=project([o.c[0],o.c[1],Math.max(.35,facadeHeight(o.c)-.12)],b);if(a&&z)line(a,z,o.views>=3?'#52b86c':o.views===2?'#e2a23f':o.views===1?'#d8893c':'#dc5960',2);}}
if(show.plan)for(const p of PAYLOAD.observation?.missing||[]){{const a=project([p.c[0],p.c[1],0],b),z=project([p.c[0],p.c[1],8],b);if(a&&z)line(a,z,'#4c9ee8',2,p.bridge?[5,4]:[]);}}
if(show.ridge)for(const r of PAYLOAD.ridges||[]){{const a=project(r.a,b),z=project(r.b,b);if(a&&z)line(a,z,r.views>=2?'#b585e8':'#7f668f',r.views>=2?2.5:1.2,r.views?[]:[4,4]);}}
if(show.ai)for(const p of PAYLOAD.semantic_support_points||[]){{const q=project(p.xyz,b);if(!q)continue;const single=p.semantic_evidence_class==='single_view_candidate',colours={{building:'#35d7ff',tree_evergreen:'#74d680',tree_deciduous:'#91df78',road_sign:'#ffd45a',window:'#ff55c8',door:'#d96cff',beam:'#ff9f43',column:'#b17cff'}};ctx.beginPath();ctx.arc(q[0],q[1],single?3.1:2.2,0,Math.PI*2);ctx.fillStyle=colours[p.class]||'#ff55c8';ctx.globalAlpha=single?.95:.9;ctx.fill();if(single){{ctx.strokeStyle='#fff';ctx.lineWidth=.7;ctx.stroke();}}ctx.globalAlpha=1;}}
document.getElementById('camera').textContent=`${{(az*180/Math.PI+360)%360|0}}° / ${{alt*180/Math.PI|0}}°`;}}
function stats(){{const c=PAYLOAD.counts||{{}};document.getElementById('title').textContent=(PAYLOAD.hotel||'Scène')+' · 3D';
document.getElementById('nvol').textContent=c.volumes??PAYLOAD.volumes?.length??1;document.getElementById('ntri').textContent=(c.roof_triangles??PAYLOAD.mesh?.faces?.length??0).toLocaleString('fr-CA');
document.getElementById('nveg').textContent=c.vegetation??PAYLOAD.vegetation?.length??0;const f=PAYLOAD.observation?.triangulable_fraction;document.getElementById('nobs').textContent=f==null?'non mesurée':(f*100).toFixed(1)+' %';
const s=PAYLOAD.semantic||{{}},m=PAYLOAD.semantic_multiview||{{}},ss=PAYLOAD.semantic_surface_summary||{{}};document.getElementById('nsemviews').textContent=s.images??0;document.getElementById('nsemmasks').textContent=s.segmented??0;document.getElementById('nseminstances').textContent=m.multiview_instances??0;document.getElementById('nsemtracks').textContent=m.shared_measured_tracks??0;document.getElementById('nsem3d').textContent=ss.geometry_3d_created??0;
const a=PAYLOAD.registration||{{}};document.getElementById('nregistration').textContent=a.status==='accepted'?'validé':a.status==='refused'?'refusé':'absent';
const rs=PAYLOAD.registered_semantic_support||{{}};document.getElementById('nregisteredpoints').textContent=rs.unique_registered_points??0;
document.getElementById('nsingleview').textContent=rs.single_view_candidates??0;
const grammar=PAYLOAD.facade_grammar||{{}},similarity=grammar.similarity||{{}};document.getElementById('nsimilarity').textContent=similarity.score==null?'non mesurée':(similarity.score*100).toFixed(1)+' %';document.getElementById('nsimilarity').style.color=similarity.threshold_met?'#52b86c':'#e2a23f';
const sh=PAYLOAD.feed_forward_shape||{{}},rejected=sh.status==='rejected';document.getElementById('nshape').textContent=rejected?'refusée (2/2)':sh.status||'non testée';document.getElementById('nshape').style.color=rejected?'#dc5960':'';document.getElementById('shape-note').textContent=rejected?' Deux solveurs GPU testés et refusés : nuages fragmentés, aucune géométrie injectée.':'';
document.getElementById('nsolid').textContent=`${{c.watertight_buildings??0}} / ${{c.volumes??PAYLOAD.volumes?.length??0}}`;}}
let drag=false,last=[0,0];cv.onpointerdown=e=>{{drag=true;last=[e.clientX,e.clientY];cv.setPointerCapture(e.pointerId)}};cv.onpointerup=()=>drag=false;
cv.onpointermove=e=>{{if(!drag)return;az+=(e.clientX-last[0])*.008;alt=Math.max(.05,Math.min(1.35,alt-(e.clientY-last[1])*.006));last=[e.clientX,e.clientY];draw();}};
cv.onwheel=e=>{{e.preventDefault();dist=Math.max(18,Math.min(900,dist*Math.exp(e.deltaY*.001)));draw();}};
function preset(k){{activeView=k;az=(k==='1'?(CAMERA.facade_azimuth_deg??210):(CAMERA.azimuth_deg??210))*Math.PI/180;if(k==='1'){{focus=CAMERA.focus||TARGET.focus;dist=(CAMERA.target_distance_m??Math.max(35,TARGET.span*1.15));alt=(CAMERA.facade_altitude_deg??1)*Math.PI/180;show.veg=false;}}else if(k==='2'){{focus=CAMERA.focus||TARGET.focus;dist=Math.max(70,(CAMERA.target_distance_m??TARGET.span)*1.02);alt=62*Math.PI/180;show.veg=true;}}else if(k==='3'){{focus=CONTEXT.focus;dist=(CAMERA.context_distance_m??Math.max(150,CONTEXT.span*1.1));alt=38*Math.PI/180;show.veg=true;}}for(const n of ['1','2','3'])document.getElementById('view-'+n)?.classList.toggle('active',n===k);draw();}}
onkeydown=e=>{{const k=e.key.toLowerCase(),m={{r:'roof',v:'vol',p:'veg',g:'ground',o:'obs',n:'plan',f:'ridge',i:'ai'}};if(m[k])show[m[k]]=!show[m[k]];else if(k==='w')wire=!wire;else if(k==='h')document.body.classList.toggle('ui-hidden');else if(['1','2','3'].includes(k))preset(k);else if(e.code==='Space')spin=!spin;draw();}};
function tick(){{if(spin){{az+=.0025;draw();}}requestAnimationFrame(tick)}}addEventListener('resize',()=>{{resize();draw()}});resize();stats();draw();tick();
</script></body></html>"""


def _webgl_html(payload: dict, manifest: dict, gltf: dict) -> str:
    """Self-contained GPU viewer with native depth and perspective UVs."""
    payload_json, manifest_json, gltf_json = map(_safe_json, (payload, manifest, gltf))
    return f"""<!doctype html><html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>BuildingDroneVisit · WebGL</title>
<style>html,body,#cv{{margin:0;width:100%;height:100%;overflow:hidden;background:#101722}}#cv{{display:block}}
#hud{{position:fixed;left:14px;top:14px;padding:12px 15px;border:1px solid #344252;border-radius:10px;background:#101822e8;color:#f6f8fa;font:13px system-ui}}small{{color:#a8b4c2}}</style></head>
<body><canvas id="cv"></canvas><div id="hud"><b>MODE DÉMONSTRATION · GPU</b><br><small>CanonicalSceneMesh · z-buffer · UV perspective</small></div>
<script>const PAYLOAD={payload_json},MANIFEST={manifest_json},GLTF={gltf_json};
const cv=document.getElementById('cv'),gl=cv.getContext('webgl2',{{antialias:true,depth:true}});if(!gl)throw Error('WebGL2 requis');
gl.enable(gl.DEPTH_TEST);gl.depthFunc(gl.LEQUAL);gl.enable(gl.CULL_FACE);gl.cullFace(gl.BACK);
const vs=`#version 300 es\nin vec3 p;in vec3 n;in vec2 uv;uniform mat4 mvp;out vec3 N;out vec2 U;void main(){{gl_Position=mvp*vec4(p,1.);N=n;U=uv;}}`,
fs=`#version 300 es\nprecision highp float;in vec3 N;in vec2 U;out vec4 c;void main(){{float l=.28+.72*max(dot(normalize(N),normalize(vec3(.4,-.3,.85))),0.);vec3 a=mix(vec3(.28,.20,.17),vec3(.55,.38,.29),step(.5,fract(U.x*12.))*step(.5,fract(U.y*12.)));c=vec4(a*l,1.);}}`;
function shader(t,s){{const q=gl.createShader(t);gl.shaderSource(q,s);gl.compileShader(q);if(!gl.getShaderParameter(q,gl.COMPILE_STATUS))throw Error(gl.getShaderInfoLog(q));return q}}const pr=gl.createProgram();gl.attachShader(pr,shader(gl.VERTEX_SHADER,vs));gl.attachShader(pr,shader(gl.FRAGMENT_SHADER,fs));gl.linkProgram(pr);gl.useProgram(pr);
const uri=GLTF.buffers[0].uri.split(',')[1],raw=Uint8Array.from(atob(uri),c=>c.charCodeAt(0)),views=GLTF.bufferViews,acc=GLTF.accessors;
const vao=gl.createVertexArray();gl.bindVertexArray(vao);function attr(name,ai,size){{const a=acc[ai],v=views[a.bufferView],loc=gl.getAttribLocation(pr,name),b=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,b);gl.bufferData(gl.ARRAY_BUFFER,raw.slice(v.byteOffset||0,(v.byteOffset||0)+v.byteLength),gl.STATIC_DRAW);gl.enableVertexAttribArray(loc);gl.vertexAttribPointer(loc,size,gl.FLOAT,false,0,0)}}attr('p',0,3);attr('n',1,3);attr('uv',2,2);
const ia=acc[3],iv=views[ia.bufferView],ib=gl.createBuffer();gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,ib);gl.bufferData(gl.ELEMENT_ARRAY_BUFFER,raw.slice(iv.byteOffset||0,(iv.byteOffset||0)+iv.byteLength),gl.STATIC_DRAW);
const lo=acc[0].min,hi=acc[0].max,focus=lo.map((v,i)=>(v+hi[i])/2),span=Math.max(...hi.map((v,i)=>v-lo[i]),1),cam=PAYLOAD.camera||{{}};let az=(cam.azimuth_deg??210)*Math.PI/180,el=.45,dist=span*2.2,drag=false,last=[0,0];
function mul(a,b){{let o=new Float32Array(16);for(let r=0;r<4;r++)for(let c=0;c<4;c++)for(let k=0;k<4;k++)o[c*4+r]+=a[k*4+r]*b[c*4+k];return o}}function perspective(f,a,n,z){{const q=1/Math.tan(f/2),o=new Float32Array(16);o[0]=q/a;o[5]=q;o[10]=(z+n)/(n-z);o[11]=-1;o[14]=2*z*n/(n-z);return o}}function look(eye,c){{let f=c.map((v,i)=>v-eye[i]),L=Math.hypot(...f);f=f.map(v=>v/L);let s=[f[1],-f[0],0],S=Math.hypot(...s)||1;s=s.map(v=>v/S);let u=[s[1]*f[2],-s[0]*f[2],s[0]*f[1]-s[1]*f[0]],o=new Float32Array([s[0],u[0],-f[0],0,s[1],u[1],-f[1],0,s[2],u[2],-f[2],0,0,0,0,1]);o[12]=-s.reduce((q,v,i)=>q+v*eye[i],0);o[13]=-u.reduce((q,v,i)=>q+v*eye[i],0);o[14]=f.reduce((q,v,i)=>q+v*eye[i],0);return o}}
function draw(){{const d=Math.min(devicePixelRatio||1,2),w=cv.clientWidth,h=cv.clientHeight;cv.width=w*d;cv.height=h*d;gl.viewport(0,0,cv.width,cv.height);gl.clearColor(.063,.09,.13,1);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);const eye=[focus[0]+Math.cos(az)*Math.cos(el)*dist,focus[1]+Math.sin(az)*Math.cos(el)*dist,focus[2]+Math.sin(el)*dist],near=cam.near_m??.05,far=cam.far_m??Math.max(10000,dist*10);gl.uniformMatrix4fv(gl.getUniformLocation(pr,'mvp'),false,mul(perspective(Math.PI/3,w/h,near,far),look(eye,focus)));gl.drawElements(gl.TRIANGLES,ia.count,gl.UNSIGNED_INT,0)}}
cv.onpointerdown=e=>{{drag=true;last=[e.clientX,e.clientY];cv.setPointerCapture(e.pointerId)}};cv.onpointerup=()=>drag=false;cv.onpointermove=e=>{{if(drag){{az+=(e.clientX-last[0])*.008;el=Math.max(-1.4,Math.min(1.4,el-(e.clientY-last[1])*.006));last=[e.clientX,e.clientY];draw()}}}};cv.onwheel=e=>{{e.preventDefault();dist=Math.max(span*.2,dist*Math.exp(e.deltaY*.001));draw()}};addEventListener('resize',draw);draw();
</script></body></html>"""


def build(workspace: Workspace) -> ViewerOutputs:
    """Publie le viewer, son payload stable et son manifeste de provenance."""
    output_dir = workspace.path("11_conditioning")
    output_dir.mkdir(parents=True, exist_ok=True)
    html_path = output_dir / "viewer.html"
    payload_path = output_dir / "viewer_payload.json"
    payload_meta_path = output_dir / "viewer_payload_meta.json"
    manifest_path = output_dir / "viewer_manifest.json"

    payload_created = False
    canonical_path = workspace.path("11_conditioning", "conditioned_scene.json")
    if canonical_path.is_file():
        from .conditioning.canonical import viewer_payload

        payload = viewer_payload(_read_json(canonical_path))
        payload_created = True
    elif payload_path.is_file():
        payload = _read_json(payload_path)
    else:
        payload = _legacy_payload(html_path) or _obj_payload(workspace)
        payload_created = True

    payload.setdefault("hotel", workspace.hotel_id)
    shape_audit_path = workspace.path(
        "11_conditioning", "feed_forward_shape_audit.json"
    )
    if shape_audit_path.is_file():
        payload["feed_forward_shape"] = _read_json(shape_audit_path)
    from .conditioning.facade_texture import build as build_facade_textures
    from .conditioning.viewpoint import optimal_camera

    build_facade_textures(workspace, payload)
    # L'azimut de caméra est dérivé de la couverture photographique mesurée de
    # chaque face (et non figé) : le viewer cadre la face la mieux observée, en
    # privilégiant la face d'entrée pour la vue héro.
    payload["camera"] = optimal_camera(payload)
    payload_path = workspace.write_json(
        "11_conditioning/viewer_payload.json", payload
    )
    from .canonical_gltf import export_canonical_mesh_gltf
    from .conditioning.canonical_mesh import CanonicalSceneMesh
    gltf_path = workspace.path("11_conditioning", "canonical_scene.gltf")
    try:
        meshes = [
            CanonicalSceneMesh.from_dict(volume["solid"])
            for volume in payload.get("volumes", [])
        ]
        gltf_metadata = export_canonical_mesh_gltf(
            CanonicalSceneMesh.merge(meshes), gltf_path
        )
    except (KeyError, ValueError):
        gltf_metadata = None
    facade_audit_path = workspace.write_json(
        "11_conditioning/facade_similarity_audit.json",
        payload.get("facade_grammar")
        or {
            "contract_version": 1,
            "status": "unavailable",
            "reason": "facade grammar absent from viewer payload",
        },
    )
    source_digests = _source_digests(workspace)
    payload_meta = _read_json(payload_meta_path) if payload_meta_path.is_file() else None
    # Le pointeur du paquet formel figurait dans le premier contrat bien qu'il
    # ne soit pas lu pour construire ce payload. Une simple republication des
    # Gates rendait ainsi la géométrie faussement périmée. La migration vers le
    # scope geometry-v1 ne modifie pas le payload ; elle resserre sa dépendance
    # aux seules sources qu'il représente réellement.
    if (
        payload_created
        or payload_meta is None
        or payload_meta.get("source_scope") != "geometry-v7"
    ):
        payload_meta = {
            "contract_version": 3,
            "source_scope": "geometry-v7",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_digests": source_digests,
        }
        workspace.write_json("11_conditioning/viewer_payload_meta.json", payload_meta)
    payload_current = payload_meta.get("source_digests") == source_digests
    manifest = {
        "contract_version": 1,
        "viewer_version": VIEWER_VERSION,
        "hotel_id": workspace.hotel_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "demo",
        "demo_readiness": "presentable" if payload_current else "stale_payload",
        "formal_phase1_status": "not_overridden",
        "deferred_until_acceptance": [
            "final_authorized_capture",
            "production_rights_clearance",
        ],
        "payload": {
            "path": "viewer_payload.json",
            "sha256": _sha256(payload_path),
        },
        "canonical_gltf": ({
            "path": "canonical_scene.gltf",
            "sha256": _sha256(gltf_path),
            **gltf_metadata,
        } if gltf_metadata is not None else None),
        "facade_similarity": {
            "path": "facade_similarity_audit.json",
            "sha256": _sha256(facade_audit_path),
            "score": (
                payload.get("facade_grammar", {})
                .get("similarity", {})
                .get("score")
            ),
            "metric": "weighted structural feature similarity",
            "photometric_claim": False,
        },
        "source_digests": source_digests,
        "payload_source_digests": payload_meta.get("source_digests", {}),
        "payload_current": payload_current,
        "limitations": [
            "le viewer présente une géométrie LiDAR et des zones proxy",
            "la démo ne vaut pas ENVIRONMENT_3D_READY",
            "les droits de diffusion et la captation finale sont différés",
            *(
                []
                if payload_current
                else [
                    "le payload précède au moins une source courante ; "
                    "régénération géométrique requise"
                ]
            ),
        ],
    }
    manifest_path = workspace.write_json(
        "11_conditioning/viewer_manifest.json", manifest
    )
    html = (
        _webgl_html(payload, manifest, _read_json(gltf_path))
        if gltf_metadata is not None
        else _html(payload, manifest)
    )
    html_path = workspace.write_text("11_conditioning/viewer.html", html)
    return ViewerOutputs(html=html_path, payload=payload_path, manifest=manifest_path)


def open_in_browser(path: Path) -> bool:
    """Ouvre le viewer local sans serveur ni dépendance réseau."""
    return bool(webbrowser.open(path.resolve().as_uri()))
