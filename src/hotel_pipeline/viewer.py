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

VIEWER_VERSION = "demo-viewer-1.8.0"
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
:root{{--bg:#0d1118;--panel:#151b24e8;--ink:#f4f6f8;--muted:#9ba7b5;--line:#2b3542;
--target:#e9763d;--other:#66717e;--roof:#8eab85;--grass:#6e8d58;--road:#686d73;
--veg:#4f7f55;--pole:#aeb7c3;--ok:#52b86c;--thin:#e2a23f;--none:#dc5960;--plan:#4c9ee8}}
*{{box-sizing:border-box}}html,body{{margin:0;width:100%;height:100%;overflow:hidden;background:var(--bg);
color:var(--ink);font:14px/1.45 ui-sans-serif,-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif}}
canvas{{width:100%;height:100%;display:block;cursor:grab;touch-action:none}}canvas:active{{cursor:grabbing}}
.panel{{position:fixed;background:var(--panel);border:1px solid var(--line);border-radius:12px;
backdrop-filter:blur(12px);box-shadow:0 14px 40px #0007}}#hud{{top:14px;left:14px;width:310px;padding:16px}}
h1{{font-size:16px;margin:0 0 3px}}.sub,.note{{color:var(--muted);font-size:12px}}.badge{{display:inline-block;
margin:10px 0;padding:4px 8px;border-radius:999px;background:#55311f;color:#ffc39f;font-size:11px;font-weight:700}}
.row{{display:flex;justify-content:space-between;border-top:1px solid var(--line);padding:6px 0;font-variant-numeric:tabular-nums}}
.row span{{color:var(--muted)}}.keys{{margin-top:10px;color:var(--muted);font-size:11px;line-height:1.8}}
kbd{{border:1px solid var(--line);border-radius:4px;padding:1px 5px;color:var(--ink);background:#202834}}
#legend{{left:14px;bottom:14px;padding:11px 13px;font-size:11px}}.lg{{display:flex;gap:7px;align-items:center;margin:3px}}
.sw{{width:20px;height:3px;border-radius:3px}}#scope{{right:14px;bottom:14px;max-width:390px;padding:12px 14px;
color:var(--muted);font-size:11px}}#scope b{{color:var(--ink)}}
</style></head><body><canvas id=\"cv\"></canvas>
<section id=\"hud\" class=\"panel\"><h1 id=\"title\">Scène 3D</h1>
<div class=\"sub\">Géométrie LiDAR et contraintes de reconstruction</div>
<div class=\"badge\">MODE DÉMONSTRATION</div>
<div class=\"row\"><span>Volumes</span><b id=\"nvol\">–</b></div>
<div class=\"row\"><span>Triangles de toiture</span><b id=\"ntri\">–</b></div>
<div class=\"row\"><span>Végétation</span><b id=\"nveg\">–</b></div>
<div class=\"row\"><span>Solides étanches</span><b id=\"nsolid\">–</b></div>
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
<kbd>O</kbd> couverture · <kbd>N</kbd> prises proposées · <kbd>F</kbd> faîtages · <kbd>W</kbd> filaire</div></section>
<section id=\"legend\" class=\"panel\">
<div class=\"lg\"><i class=\"sw\" style=\"background:var(--target)\"></i>Bâtiment cible</div>
<div class=\"lg\"><i class=\"sw\" style=\"background:var(--roof)\"></i>Toiture mesurée · contour logique</div>
<div class=\"lg\"><i class=\"sw\" style=\"background:var(--ok)\"></i>Mur triangulable</div>
<div class=\"lg\"><i class=\"sw\" style=\"background:var(--none)\"></i>Mur non observé</div>
<div class=\"lg\"><i class=\"sw\" style=\"background:var(--plan)\"></i>Prise proposée</div>
<div class=\"lg\"><i class=\"sw\" style=\"background:#ff55c8\"></i>Support SfM multi-vues</div>
<div class=\"lg\"><i class=\"sw\" style=\"background:#ff9f43\"></i>Hypothèse mono-vue</div></section>
<section id=\"scope\" class=\"panel\"><b>Portée de la démo.</b> Les droits de diffusion et la captation finale
sont différés jusqu’à acceptation. Ce viewer ne transforme pas le proxy actuel en
<code>ENVIRONMENT_3D_READY</code>.</section>
<script>
const PAYLOAD = {payload_json};
const MANIFEST = {manifest_json};
const cv=document.getElementById('cv'),ctx=cv.getContext('2d');let W=0,H=0,D=1;
let az=210*Math.PI/180,alt=26*Math.PI/180,dist=150,spin=false,wire=false;
let show={{roof:true,vol:true,veg:true,ground:true,obs:false,plan:false,ridge:false,ai:false}};
const C={{target:'#e9763d',other:'#66717e',roof:'#8eab85',grass:'#6e8d58',road:'#686d73',veg:'#4f7f55',pole:'#aeb7c3',mesh:'#e9763d',semantic:'#ff55c8'}};
function resize(){{D=Math.min(devicePixelRatio||1,2);W=cv.clientWidth;H=cv.clientHeight;cv.width=W*D;cv.height=H*D;ctx.setTransform(D,0,0,D,0,0);}}
function basis(){{const ca=Math.cos(alt),sa=Math.sin(alt),eye=[Math.cos(az)*ca*dist,Math.sin(az)*ca*dist,sa*dist+8];
let f=[-eye[0],-eye[1],10-eye[2]],l=Math.hypot(...f);f=f.map(x=>x/l);let r=[f[1],-f[0],0];l=Math.hypot(...r)||1;r=r.map(x=>x/l);
const u=[r[1]*f[2],-r[0]*f[2],r[0]*f[1]-r[1]*f[0]];return{{eye,f,r,u}};}}
function project(p,b){{const d=[p[0]-b.eye[0],p[1]-b.eye[1],p[2]-b.eye[2]],z=d[0]*b.f[0]+d[1]*b.f[1]+d[2]*b.f[2];if(z<.5)return null;
const x=d[0]*b.r[0]+d[1]*b.r[1]+d[2]*b.r[2],y=d[0]*b.u[0]+d[1]*b.u[1]+d[2]*b.u[2],q=H*.5/Math.tan(Math.PI/6);return[W/2+q*x/z,H/2-q*y/z,z];}}
function faces(){{const out=[];
if(show.ground)for(const g of PAYLOAD.ground||[])out.push({{p:g.ring.map(v=>[v[0],v[1],0]),k:g.kind==='vegetal'?'grass':'road'}});
if(show.vol)for(const v of PAYLOAD.volumes||[]){{const fp=v.fp||[],wh=v.wh||[];for(let i=0;i<fp.length;i++){{const j=(i+1)%fp.length;
out.push({{p:[[...fp[i],0],[...fp[j],0],[...fp[j],wh[j]??v.h],[...fp[i],wh[i]??v.h]],k:v.target?'target':'other'}});}}
if(show.roof&&v.solid?.vertices&&v.solid?.faces){{const n=fp.length;for(const f of v.solid.faces)if(f.every(i=>i>=n))out.push({{p:f.map(i=>v.solid.vertices[i]),k:'roof'}});}}
if(show.roof&&v.rv&&v.rf)for(const f of v.rf)out.push({{p:f.map(i=>v.rv[i]),k:'roof'}});}}
if(PAYLOAD.mesh)for(const f of PAYLOAD.mesh.faces||[])out.push({{p:f.map(i=>PAYLOAD.mesh.vertices[i]),k:'mesh'}});
if(show.ai)for(const s of PAYLOAD.semantic_surfaces||[]){{const g=s.surface||{{}};for(const f of g.faces||[])out.push({{p:f.map(i=>g.vertices[i]),k:'semantic'}});}}
if(show.veg)for(const v of PAYLOAD.vegetation||[]){{let rings=v.rings||[];if(rings.length<2){{const n=12,profile=[.38,.72,1,.94,.7,.34];rings=profile.map((s,l)=>Array.from({{length:n}},(_,i)=>{{const a=i*Math.PI*2/n,q=s*(1+.09*Math.sin(3*a+(v.c[0]+v.c[1])));return[v.c[0]+Math.cos(a)*v.r*q,v.c[1]+Math.sin(a)*v.r*q,v.h*(.12+.84*l/(profile.length-1))]}}));}}
for(let l=0;l<rings.length-1;l++){{const a=rings[l],z=rings[l+1],n=Math.min(a.length,z.length);for(let i=0;i<n;i++)out.push({{p:[a[i],a[(i+1)%n],z[(i+1)%n],z[i]],k:'veg'}});}}if(rings.length){{out.push({{p:[...rings[0]].reverse(),k:'veg'}});out.push({{p:rings[rings.length-1],k:'veg'}});}}}}
for(const f of PAYLOAD.furniture||[]){{const r=Math.max(f.r||.2,.15),x=f.c[0],y=f.c[1];out.push({{p:[[x-r,y-r,0],[x+r,y-r,0],[x+r,y-r,f.h],[x-r,y-r,f.h]],k:'pole'}});}}
return out;}}
function line(a,b,color,width=2,dash=[]){{ctx.save();ctx.strokeStyle=color;ctx.lineWidth=width;ctx.setLineDash(dash);ctx.beginPath();ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1]);ctx.stroke();ctx.restore();}}
function facadeHeight(c){{let best=Infinity,height=null;for(const v of PAYLOAD.volumes||[]){{const fp=v.fp||[],wh=v.wh||[];for(let i=0;i<fp.length;i++){{const j=(i+1)%fp.length,a=fp[i],z=fp[j],dx=z[0]-a[0],dy=z[1]-a[1],den=dx*dx+dy*dy||1,t=Math.max(0,Math.min(1,((c[0]-a[0])*dx+(c[1]-a[1])*dy)/den)),qx=a[0]+t*dx,qy=a[1]+t*dy,d=(c[0]-qx)**2+(c[1]-qy)**2;if(d<best){{best=d;const ha=wh[i]??v.h??8,hb=wh[j]??v.h??ha;height=ha+t*(hb-ha);}}}}}}return Math.max(.5,height??8);}}
function draw(){{ctx.clearRect(0,0,W,H);const b=basis(),fs=[];for(const f of faces()){{const q=f.p.map(p=>project(p,b));if(q.some(x=>!x))continue;fs.push({{...f,q,z:q.reduce((s,x)=>s+x[2],0)/q.length}});}}
fs.sort((a,b)=>b.z-a.z);for(const f of fs){{ctx.beginPath();ctx.moveTo(f.q[0][0],f.q[0][1]);for(const q of f.q.slice(1))ctx.lineTo(q[0],q[1]);ctx.closePath();ctx.fillStyle=C[f.k]||C.other;ctx.globalAlpha=(f.k==='grass'||f.k==='road')?.65:f.k==='veg'?.92:1;ctx.fill();ctx.globalAlpha=1;if(wire){{ctx.strokeStyle='#111a';ctx.lineWidth=.5;ctx.stroke();}}}}
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
document.getElementById('nsolid').textContent=`${{c.watertight_buildings??0}} / ${{c.volumes??PAYLOAD.volumes?.length??0}}`;}}
let drag=false,last=[0,0];cv.onpointerdown=e=>{{drag=true;last=[e.clientX,e.clientY];cv.setPointerCapture(e.pointerId)}};cv.onpointerup=()=>drag=false;
cv.onpointermove=e=>{{if(!drag)return;az+=(e.clientX-last[0])*.008;alt=Math.max(.05,Math.min(1.35,alt-(e.clientY-last[1])*.006));last=[e.clientX,e.clientY];draw();}};
cv.onwheel=e=>{{e.preventDefault();dist=Math.max(35,Math.min(420,dist*Math.exp(e.deltaY*.001)));draw();}};
onkeydown=e=>{{const k=e.key.toLowerCase(),m={{r:'roof',v:'vol',p:'veg',g:'ground',o:'obs',n:'plan',f:'ridge',i:'ai'}};if(m[k])show[m[k]]=!show[m[k]];else if(k==='w')wire=!wire;else if(e.code==='Space')spin=!spin;draw();}};
function tick(){{if(spin){{az+=.0025;draw();}}requestAnimationFrame(tick)}}addEventListener('resize',()=>{{resize();draw()}});resize();stats();draw();tick();
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
    payload_path = workspace.write_json(
        "11_conditioning/viewer_payload.json", payload
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
    html_path = workspace.write_text(
        "11_conditioning/viewer.html", _html(payload, manifest)
    )
    return ViewerOutputs(html=html_path, payload=payload_path, manifest=manifest_path)


def open_in_browser(path: Path) -> bool:
    """Ouvre le viewer local sans serveur ni dépendance réseau."""
    return bool(webbrowser.open(path.resolve().as_uri()))
