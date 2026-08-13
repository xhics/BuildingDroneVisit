"""Revue humaine de la visibilité de la cible (Lot 1B §6).

Trois populations distinctes, qu'il ne faut jamais additionner ni confondre :
sans quoi « revue terminée » ne veut rien dire.

```text
en attente          tout asset dont la cascade n'a pas conclu
bloquants           ceux qui, seuls, empêchent une décision de rôle
cohorte de validation  un lot choisi pour éprouver le classifieur
```

Un asset bloquant est en attente ; un membre de la cohorte peut ne l'être pas.
Les trois comptes se rapportent donc toujours au même manifeste, jamais l'un à
l'autre.

La décision humaine, elle, est **append-only**. Corriger une revue ajoute une
entrée qui dit ce qu'elle corrige ; l'ancienne reste lisible. Aucun mécanisme
d'acceptation en masse n'existe ici : une décision non regardée n'est pas une
décision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from html import escape
from pathlib import Path

from pydantic import ValidationError

from .logging import get_logger
from .schemas import (
    Asset,
    GeometryEntry,
    GeometrySuitability,
    ReconstructionRole,
    ReviewDecision,
    ReviewEntry,
    ReviewStatus,
    Subject,
)
from .schemas.assets import DECISION_STATUS, VISIBILITY_OF

log = get_logger("review")

class ReviewRefused(RuntimeError):
    """La décision n'a pas été inscrite, et rien n'a été modifié."""


# --- populations -----------------------------------------------------------


def pending(assets: list[Asset]) -> list[Asset]:
    """Tout ce que la cascade n'a pas tranché."""
    return [a for a in assets if a.review_status is ReviewStatus.NEEDS_REVIEW]


def would_confirm(asset: Asset, policy) -> tuple[bool, str]:  # noqa: ANN001
    """Une revue favorable ferait-elle de cet asset un porteur ?

    La question ne se répond pas en récitant les prédicats de `role_for` : les
    recopier, c'était en oublier — l'arbitrage de grappe et la politique
    temporelle n'y figuraient pas, si bien qu'une vue déjà couverte ou non
    datée passait pour « débloquable ». On simule donc la décision et on
    demande au juge lui-même.

    La revue porte sur **deux** questions depuis qu'elles sont séparées :
    l'identité et l'aptitude géométrique. Ne simuler que la première rendait
    la file vide par construction — plus aucune confirmation seule ne produit
    un porteur — ce qui aurait fait passer un changement de définition pour
    une revue terminée.
    """
    from .roles import role_for

    simulated = asset.model_copy(
        update={
            "target_visibility_decision": ReviewDecision.CONFIRMED,
            "target_building_visible": True,
            "review_status": ReviewStatus.HUMAN_ACCEPTED,
            "geometry_suitability": GeometrySuitability.PRIMARY,
        }
    )
    role, reason = role_for(simulated, policy)
    return role is ReconstructionRole.PHOTO_GEOMETRY, reason


def blocking(assets: list[Asset], policy) -> list[Asset]:  # noqa: ANN001
    """Assets dont la revue, à elle seule, débloque un rôle.

    Un asset n'est bloquant que si **tout le reste** est déjà satisfait : une
    revue favorable — identité confirmée et aptitude établie — le rendrait
    porteur séance tenante. Ceux qu'un second verrou retient — grappe non
    arbitrée, datation exigée, occlusion — restent en attente sans être
    bloquants : les y mêler ferait espérer d'une revue ce qu'elle ne peut pas
    donner.

    Un asset déjà confirmé mais dont l'aptitude reste à apprécier est bloquant
    lui aussi : c'est bien une revue qui manque.
    """
    def awaits_a_decision(asset: Asset) -> bool:
        # Soit la cascade n'a pas tranché l'identité, soit l'identité est
        # acquise et seule l'aptitude manque. Élargir au-delà — « tout ce qui
        # n'est pas apprécié » — ferait entrer les 247 vues d'environnement,
        # dont la revue dirait non : la file mesurerait alors le corpus, pas
        # le travail restant.
        if asset.review_status is ReviewStatus.NEEDS_REVIEW:
            return True
        return asset.target_building_visible is True and not asset.has_been_assessed

    return [a for a in assets if awaits_a_decision(a) and would_confirm(a, policy)[0]]


def cohort(assets: list[Asset], source: str) -> list[Asset]:
    """Cohorte de validation d'une source, bloquante ou non.

    Éprouver le classifieur demande de regarder aussi ce qu'il a classé avec
    assurance : n'examiner que les cas douteux mesurerait le doute, pas la
    justesse.

    La cohorte n'est pas « toute la source » : sur 189 vues Mapillary, 164 ne
    montrent que l'environnement et n'ont rien à valider. Elle retient celles
    où un bâtiment est présent — 25 ici — car c'est là que la distinction
    « un bâtiment » / « le bâtiment cible » se joue, et là qu'elle a déjà
    échoué.
    """
    return [
        a
        for a in assets
        if a.source == source
        and (a.contains_building or Subject.BUILDING in a.subjects)
    ]


QUEUES: dict[str, tuple[str, object]] = {
    "blocking": (
        "assets dont la seule revue humaine empêche l'attribution d'un rôle",
        blocking,
    ),
    "pending": ("tout asset en attente de revue", lambda assets, policy: pending(assets)),
    "mapillary-candidates": (
        "cohorte de validation Mapillary — caps observés, donc probants",
        lambda assets, policy: cohort(assets, "mapillary"),
    ),
}


@dataclass
class QueueCounts:
    """Les trois nombres, séparés et non additionnables."""

    pending: int = 0
    blocking: int = 0
    cohort: int = 0
    pending_by_source: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "pending": self.pending,
            "blocking": self.blocking,
            "cohort": self.cohort,
            "pending_by_source": self.pending_by_source,
            "note": (
                "trois populations distinctes : un bloquant est en attente, "
                "un membre de la cohorte peut ne pas l'être. Ne pas additionner."
            ),
        }


def counts(assets: list[Asset], policy, cohort_source: str = "mapillary") -> QueueCounts:  # noqa: ANN001
    waiting = pending(assets)
    by_source: dict[str, int] = {}
    for asset in waiting:
        by_source[asset.source] = by_source.get(asset.source, 0) + 1
    return QueueCounts(
        pending=len(waiting),
        blocking=len(blocking(assets, policy)),
        cohort=len(cohort(assets, cohort_source)),
        pending_by_source=dict(sorted(by_source.items())),
    )


# --- file de revue ---------------------------------------------------------


@dataclass
class QueueItem:
    asset_id: str
    source: str
    checksum: str
    local_path: str | None
    sector: str
    role: str
    role_reason: str
    review_status: str
    decision: str
    target_evidence: str | None
    occluded_by: str | None
    heading_is_measured: bool
    sees_building: bool | None
    contains_building: bool | None
    subject_scores: dict[str, float]
    reviews: int

    #: Ce que la mesure géométrique dit de cette vue. Sans ces chiffres, le
    #: réviseur jugerait le cadrage à l'œil alors qu'il est calculé — ou non
    #: calculable, ce qui compte tout autant.
    line_of_sight: str | None = None
    clear_fraction: float | None = None
    risk_fraction: float | None = None
    blocked_fraction: float | None = None
    distance_m: float | None = None
    obstacles_at_risk: list[str] = field(default_factory=list)
    framing: str | None = None
    suitability: str | None = None

    def as_dict(self) -> dict:
        return {
            "asset_id": self.asset_id,
            "source": self.source,
            "checksum": self.checksum,
            "local_path": self.local_path,
            "sector": self.sector,
            "role": self.role,
            "role_reason": self.role_reason,
            "review_status": self.review_status,
            "decision": self.decision,
            "target_evidence": self.target_evidence,
            "occluded_by": self.occluded_by,
            "heading_is_measured": self.heading_is_measured,
            "sees_building": self.sees_building,
            "contains_building": self.contains_building,
            "subject_scores": self.subject_scores,
            "previous_reviews": self.reviews,
            "visibility": {
                "line_of_sight": self.line_of_sight,
                "clear_fraction": self.clear_fraction,
                "risk_fraction": self.risk_fraction,
                "blocked_fraction": self.blocked_fraction,
                "distance_m": self.distance_m,
                "obstacles_at_risk": self.obstacles_at_risk,
                "framing": self.framing,
            },
            "geometry_suitability": self.suitability,
        }


@dataclass
class ReviewQueue:
    name: str
    description: str
    built_at: str = ""
    #: Empreinte du manifeste au moment de la construction. Une file décrit un
    #: état ; sans cette empreinte, une file périmée ne se distingue pas d'une
    #: file courante — la première annonçait une cohorte de 189 quand le code
    #: en calculait 25, et rien sur le fichier ne le disait.
    manifest_digest: str = ""
    counts: QueueCounts = field(default_factory=QueueCounts)
    items: list[QueueItem] = field(default_factory=list)

    @property
    def slug(self) -> str:
        """Nom de publication, unique en pratique comme en théorie.

        L'horodatage à la seconde ne suffit pas : deux files construites dans
        la même seconde s'écrasaient. La microseconde et l'empreinte du
        manifeste séparent aussi bien deux exécutions rapprochées que deux
        états différents du corpus.
        """
        stamp = (self.built_at or "").replace(":", "").replace("-", "").replace(".", "")
        return f"{self.name}_{stamp}_{self.manifest_digest[:12]}"

    def as_dict(self) -> dict:
        return {
            "queue": self.name,
            "description": self.description,
            "built_at": self.built_at,
            "manifest_digest": self.manifest_digest,
            "counts": self.counts.as_dict(),
            "items": [i.as_dict() for i in self.items],
        }


def manifest_digest(assets: list[Asset]) -> str:
    """Empreinte de l'état jugé, indépendante de l'ordre de sérialisation."""
    import hashlib

    payload = "\n".join(sorted(a.model_dump_json() for a in assets))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_queue(  # noqa: ANN001
    assets: list[Asset], name: str, policy, visibility: dict | None = None
) -> ReviewQueue:
    """Assemble une file, avec tout ce qu'il faut pour juger sans le code.

    Les mesures de visibilité y sont jointes quand une exécution a été
    appliquée : un réviseur doit voir ce que la géométrie établit — et surtout
    ce qu'elle n'établit pas.
    """
    from .roles import role_for

    if name not in QUEUES:
        raise ReviewRefused(
            f"file inconnue : {name!r} ; disponibles : {sorted(QUEUES)}"
        )
    description, selector = QUEUES[name]
    selected = selector(assets, policy)

    queue = ReviewQueue(
        name=name,
        description=description,
        built_at=datetime.now(timezone.utc).isoformat(),
        manifest_digest=manifest_digest(assets),
        counts=counts(assets, policy),
    )
    measures = visibility or {}
    for asset in selected:
        role, reason = role_for(asset, policy)
        measure = measures.get(asset.id, {})
        queue.items.append(
            QueueItem(
                asset_id=asset.id,
                source=asset.source,
                checksum=asset.checksum,
                local_path=asset.local_path,
                sector=asset.view_sector.value,
                role=role.value,
                role_reason=reason,
                review_status=asset.review_status.value,
                decision=asset.target_visibility_decision.value,
                target_evidence=asset.target_evidence,
                occluded_by=asset.occluded_by,
                heading_is_measured=asset.heading_is_measured,
                sees_building=asset.sees_building,
                contains_building=asset.contains_building,
                subject_scores=asset.subject_scores,
                reviews=len(asset.review_history),
                line_of_sight=asset.line_of_sight_status,
                clear_fraction=measure.get("proven_clear_fraction"),
                risk_fraction=measure.get("risk_unknown_height_fraction"),
                blocked_fraction=measure.get("proven_blocked_fraction"),
                distance_m=measure.get("distance_m"),
                obstacles_at_risk=list(asset.occlusion_risk_by),
                framing=(
                    f"{asset.target_in_frame_fraction:.0%} du cadre"
                    if asset.target_in_frame_fraction is not None
                    else "non calculable"
                ),
                suitability=asset.geometry_suitability.value,
            )
        )
    return queue


def _fractions(item: QueueItem) -> str:
    """Les trois fractions, ou rien si la vue n'a pas été mesurée."""
    if item.clear_fraction is None:
        return ""
    return (
        f"<small> — dégagé {item.clear_fraction:.0%}, risque "
        f"{item.risk_fraction:.0%}, bloqué {item.blocked_fraction:.0%}</small>"
    )


def to_html(queue: ReviewQueue) -> str:
    """Planche de revue : l'image et, à côté, ce que la machine en a dit.

    Les chemins sont relatifs au fichier produit — la planche doit s'ouvrir
    depuis l'espace de travail, sans serveur.
    """
    rows = []
    for item in queue.items:
        image = (
            f'<img src="{escape(item.local_path)}" alt="{escape(item.asset_id)}">'
            if item.local_path
            else '<p class="missing">image absente</p>'
        )
        scores = "".join(
            f"<li>{escape(k)} <b>{v:.4f}</b></li>"
            for k, v in sorted(item.subject_scores.items(), key=lambda kv: -kv[1])[:6]
        )
        rows.append(
            f"""
    <article>
      <div class="shot">{image}</div>
      <div class="facts">
        <h2>{escape(item.asset_id)}</h2>
        <p class="sub">{escape(item.source)} · secteur {escape(item.sector)} ·
           cap {'mesuré' if item.heading_is_measured else 'choisi'}</p>
        <dl>
          <dt>rôle</dt><dd>{escape(item.role)} — {escape(item.role_reason)}</dd>
          <dt>statut</dt><dd>{escape(item.review_status)} ·
              décision {escape(item.decision)} ({item.reviews} revue(s))</dd>
          <dt>preuve</dt><dd>{escape(item.target_evidence or '—')}</dd>
          <dt>ligne de vue</dt><dd>{escape(item.line_of_sight or 'non mesurée')}
              {_fractions(item)}</dd>
          <dt>distance</dt><dd>{f'{item.distance_m:.0f} m' if item.distance_m else '—'}</dd>
          <dt>cadrage</dt><dd>{escape(item.framing or '—')}</dd>
          <dt>aptitude</dt><dd>{escape(item.suitability or '—')}</dd>
          <dt>à risque</dt><dd>{escape(', '.join(item.obstacles_at_risk) or 'aucun')}</dd>
          <dt>occlusion</dt><dd>{escape(item.occluded_by or 'aucune prouvée')}</dd>
          <dt>empreinte</dt><dd><code>{escape(item.checksum[:16])}…</code></dd>
        </dl>
        <ul class="scores">{scores}</ul>
        <pre>hotel-pipeline assets review set &lt;hôtel&gt; {escape(item.asset_id)} \\
  --decision confirmed|rejected|unresolved \\
  --by "&lt;vous&gt;" --rationale "…" --evidence "…"</pre>
      </div>
    </article>"""
        )

    counts_block = queue.counts.as_dict()
    return f"""<!doctype html>
<html lang="fr"><head><meta charset="utf-8">
<title>Revue — {escape(queue.name)}</title>
<style>
 body {{ font: 15px/1.5 system-ui, sans-serif; margin: 2rem; max-width: 1100px; }}
 header p {{ color: #555; }}
 .counts b {{ font-size: 1.4rem; }}
 .counts span {{ display: inline-block; margin-right: 2rem; }}
 article {{ display: flex; gap: 1.5rem; border-top: 1px solid #ddd; padding: 1.5rem 0; }}
 .shot img {{ max-width: 420px; border-radius: 4px; }}
 .facts {{ flex: 1; }}
 h2 {{ font-size: 1rem; margin: 0; font-family: ui-monospace, monospace; }}
 .sub {{ color: #666; margin: .2rem 0 .8rem; }}
 dl {{ display: grid; grid-template-columns: 7rem 1fr; gap: .2rem .8rem; margin: 0; }}
 dt {{ color: #666; }} dd {{ margin: 0; }}
 .scores {{ list-style: none; padding: 0; display: flex; flex-wrap: wrap; gap: .8rem;
            margin: .8rem 0; color: #444; }}
 pre {{ background: #f5f5f5; padding: .6rem; overflow-x: auto; font-size: 12px; }}
 .missing {{ color: #b00; }}
</style></head><body>
<header>
 <h1>Revue humaine — {escape(queue.name)}</h1>
 <p>{escape(queue.description)}</p>
 <p class="counts">
   <span>en attente <b>{counts_block['pending']}</b></span>
   <span>bloquants <b>{counts_block['blocking']}</b></span>
   <span>cohorte <b>{counts_block['cohort']}</b></span>
 </p>
 <p><small>Trois populations distinctes — un bloquant est en attente, un membre
 de la cohorte peut ne pas l'être. Ne pas additionner.</small></p>
 <p><small>Construite le {escape(queue.built_at)} · {len(queue.items)} image(s)</small></p>
</header>
{"".join(rows)}
</body></html>
"""


# --- décision ---------------------------------------------------------------


def _digest(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prepare(
    assets: list[Asset], asset_id: str, by: str, rationale: str,
    evidence: list[str], workspace_root: Path | None,
) -> tuple[Asset, Path, str]:
    """Contrôles communs à tout arbitrage, avant la moindre mutation.

    Identité et aptitude engagent la même chose : une personne, un motif, une
    preuve, et une image dont on sait qu'elle n'a pas changé depuis.
    """
    if not (by or "").strip():
        raise ReviewRefused("décision sans auteur — une revue anonyme n'engage rien")
    if not (rationale or "").strip():
        raise ReviewRefused(
            "décision sans justification — le verdict seul ne s'audite pas"
        )
    if not [e for e in evidence if (e or "").strip()]:
        raise ReviewRefused(
            "décision sans preuve — le motif dit ce qui a été conclu, la preuve "
            "dit sur quoi ; sans elle, la revue ne se rejoue pas"
        )

    index = next((i for i, a in enumerate(assets) if a.id == asset_id), None)
    if index is None:
        raise ReviewRefused(f"asset inconnu : {asset_id!r}")

    before = assets[index]
    if not before.local_path:
        raise ReviewRefused(
            f"{asset_id} n'a pas de fichier local — on ne juge pas une image absente"
        )

    path = Path(before.local_path)
    if workspace_root and not path.is_absolute():
        path = workspace_root / path
    if not path.is_file():
        raise ReviewRefused(f"{asset_id} : fichier introuvable ({path})")

    actual = _digest(path)
    if actual != before.checksum:
        raise ReviewRefused(
            f"{asset_id} : empreinte {actual[:16]}… au lieu de "
            f"{before.checksum[:16]}… — l'image a changé depuis son manifeste ; "
            "la décision porterait sur autre chose que ce qui est déclaré"
        )
    return before, path, actual


def _revalidate(candidate: Asset, asset_id: str) -> Asset:
    """`model_copy(update=...)` ne revalide rien.

    Les invariants du manifeste seraient contournés par la seule voie qui les
    met en jeu. L'échec laisse la liste d'assets intacte.
    """
    try:
        return Asset.model_validate(candidate.model_dump())
    except ValidationError as exc:
        raise ReviewRefused(
            f"{asset_id} : décision incohérente avec le manifeste — {exc}"
        ) from exc


def decide(
    assets: list[Asset],
    asset_id: str,
    decision: ReviewDecision,
    by: str,
    rationale: str,
    evidence: list[str],
    workspace_root: Path | None = None,
) -> tuple[Asset, Asset]:
    """Inscrit une décision, après avoir vérifié qu'elle porte sur la bonne image.

    Rien n'est modifié si un contrôle échoue : une décision partiellement
    appliquée serait pire qu'aucune, puisqu'elle paraîtrait complète.

    Rend l'asset avant et après, pour que l'appelant produise un avant/après
    sans avoir à le reconstituer.
    """
    before, _path, actual = _prepare(
        assets, asset_id, by, rationale, evidence, workspace_root
    )
    index = next(i for i, a in enumerate(assets) if a.id == asset_id)
    kept = [e.strip() for e in evidence if (e or "").strip()]

    entry = ReviewEntry(
        decision=decision,
        decided_by=by.strip(),
        rationale=rationale.strip(),
        evidence=kept,
        reviewed_checksum=actual,
        # Corriger une revue ne remplace jamais l'entrée précédente : elle la
        # désigne. L'historique reste lisible dans l'ordre où il a été écrit.
        supersedes_index=len(before.review_history) - 1 if before.review_history else None,
    )

    candidate = before.model_copy(
        update={
            "review_history": [*before.review_history, entry],
            "target_visibility_decision": decision,
            "review_status": DECISION_STATUS[decision],
            "reviewer": entry.decided_by,
            "reviewed_at": entry.decided_at,
            "review_rationale": entry.rationale,
            "review_evidence": entry.evidence,
            # La visibilité déclarée suit la décision : la laisser telle quelle
            # ferait coexister un rejet humain et une cible réputée visible.
            "target_building_visible": VISIBILITY_OF[decision],
            "target_evidence": f"revue humaine : {entry.rationale}",
        }
    )

    after = _revalidate(candidate, asset_id)
    assets[index] = after
    log.info("%s : %s par %s", asset_id, decision.value, entry.decided_by)
    return before, after


def assessment_fields(
    suitability: GeometrySuitability,
    by: str,
    rationale: str,
    evidence: list[str],
    checksum: str,
    measurements: dict[str, float] | None = None,
    previous: list | None = None,  # noqa: ANN001
) -> dict:
    """Champs d'une appréciation géométrique, historique compris.

    Exposé plutôt que reconstruit chez chaque appelant : l'aptitude et son
    historique ne doivent jamais pouvoir diverger, et un test qui poserait le
    champ seul décrirait un état que le manifeste refuse.
    """
    previous = list(previous or [])
    entry = GeometryEntry(
        suitability=suitability,
        decided_by=by.strip(),
        rationale=rationale.strip(),
        evidence=[e.strip() for e in evidence if e.strip()],
        reviewed_checksum=checksum,
        measurements=dict(measurements or {}),
        supersedes_index=len(previous) - 1 if previous else None,
    )
    return {
        "geometry_suitability": suitability,
        "geometry_history": [*previous, entry],
    }


def assess(
    assets: list[Asset],
    asset_id: str,
    suitability: GeometrySuitability,
    by: str,
    rationale: str,
    evidence: list[str],
    measurements: dict[str, float] | None = None,
    workspace_root: Path | None = None,
) -> tuple[Asset, Asset]:
    """Inscrit une appréciation d'aptitude géométrique.

    Mêmes exigences que pour la visibilité — auteur, motif, preuve, empreinte
    vérifiée — parce que c'est la même nature de décision : une personne
    engage sa lecture d'une image précise.
    """
    before, path, digest = _prepare(assets, asset_id, by, rationale, evidence, workspace_root)
    index = next(i for i, a in enumerate(assets) if a.id == asset_id)

    candidate = before.model_copy(
        update=assessment_fields(
            suitability, by, rationale,
            [e.strip() for e in evidence if e.strip()],
            digest, measurements, previous=before.geometry_history,
        )
    )
    after = _revalidate(candidate, asset_id)
    assets[index] = after
    log.info("%s : aptitude %s par %s", asset_id, suitability.value, by.strip())
    return before, after


# --- effet d'une décision ---------------------------------------------------


@dataclass
class Impact:
    """Ce que la décision change, mesuré et non supposé."""

    asset_id: str
    decision: str
    checksum: str = ""
    entry: dict = field(default_factory=dict)
    role_before: str = ""
    role_after: str = ""
    reason_before: str = ""
    reason_after: str = ""
    cluster_before: str = ""
    cluster_after: str = ""
    roles_before: dict[str, int] = field(default_factory=dict)
    roles_after: dict[str, int] = field(default_factory=dict)
    counts_before: dict = field(default_factory=dict)
    counts_after: dict = field(default_factory=dict)
    viewpoints_before: dict[str, int] = field(default_factory=dict)
    viewpoints_after: dict[str, int] = field(default_factory=dict)

    @property
    def slug(self) -> str:
        """Nom de publication, unique jusqu'à la microseconde."""
        stamp = datetime.now(timezone.utc).isoformat()
        stamp = stamp.replace(":", "").replace("-", "").replace(".", "")
        return f"{self.asset_id}_{stamp}_{self.checksum[:12]}"

    def as_dict(self) -> dict:
        return {
            "asset_id": self.asset_id,
            "decision": self.decision,
            "reviewed_checksum": self.checksum,
            "entry": self.entry,
            "asset_role": {
                "before": f"{self.role_before} — {self.reason_before}",
                "after": f"{self.role_after} — {self.reason_after}",
            },
            # Le rôle de grappe compte autant que le rôle de reconstruction :
            # la revue peut faire d'une vue le représentant de son point de vue.
            "cluster_role": {"before": self.cluster_before, "after": self.cluster_after},
            "corpus_roles": {"before": self.roles_before, "after": self.roles_after},
            "review_counts": {"before": self.counts_before, "after": self.counts_after},
            # Fichiers et points de vue ne se confondent pas : deux vues d'un
            # même point ne font pas deux observations.
            "viewpoints_by_suitability": {
                "before": self.viewpoints_before, "after": self.viewpoints_after
            },
        }


def viewpoints_by_suitability(assets: list[Asset]) -> dict[str, int]:
    """Points de vue **indépendants**, par aptitude, jamais des fichiers.

    Deux photographies prises du même endroit à deux degrés près ne font pas
    deux observations : elles font une seule, photographiée deux fois. Compter
    les fichiers surestimerait la couverture d'autant.
    """
    best: dict[str, GeometrySuitability] = {}
    for asset in assets:
        if asset.target_building_visible is not True:
            continue
        viewpoint = asset.viewpoint_cluster or asset.id
        current = best.get(viewpoint)
        rank = {
            GeometrySuitability.PRIMARY: 0,
            GeometrySuitability.AUXILIARY: 1,
            GeometrySuitability.UNASSESSED: 2,
            GeometrySuitability.INSUFFICIENT: 3,
        }
        if current is None or rank[asset.geometry_suitability] < rank[current]:
            best[viewpoint] = asset.geometry_suitability

    counted: dict[str, int] = {}
    for suitability in best.values():
        counted[suitability.value] = counted.get(suitability.value, 0) + 1
    return dict(sorted(counted.items()))


def recompute(assets: list[Asset], policy):  # noqa: ANN001, ANN201
    """Réarbitre les grappes **puis** les rôles, sans relancer le classifieur.

    Une décision de revue ne change pas ce que le modèle a vu ; relancer
    OpenCLIP ici coûterait cher et changerait des chiffres sans rapport avec la
    décision, rendant l'avant/après illisible.

    Elle change en revanche la clé de choix du représentant d'un point de vue,
    depuis que celle-ci suit la cible. Ne réaffecter que les rôles laissait
    donc le manifeste dans un état transitoire — canonique périmé, rôles
    calculés dessus — jusqu'à une commande `assets dedup` que rien n'imposait.
    Les deux étapes vont ensemble ou pas du tout.
    """
    from .dedup_levels import assign_roles
    from .roles import assign

    assign_roles(assets, max_overlap=policy.dedup.max_overlap_per_cluster)
    # Le rapport complet, motifs compris : un compte de rôles ne dit pas
    # *pourquoi*, et c'est le motif qui rend un avant/après lisible.
    return assign(assets, policy)
