"""CLI du pipeline (plan directeur §18).

Typer fournit la CLI, un Makefile orchestre le PoC. Chaque commande est
rejouable, détecte un résultat existant, et n'expose aucun secret.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import requests
import typer

from . import __version__, logging as pipeline_logging
from .config import check_providers, load_env
from .capabilities import Capability
from .context import PipelineContext
from .providers.cache import OfflineError
from .providers.geocode import GeocodingError
from .providers.overpass import OverpassError
from .schemas import ProjectManifest, StepRecord
from .steps import STEP_ORDER, STEPS, StepBlocked, StepNotImplemented, run_step
from .workspace import SUBDIRS, Workspace

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Reconstruction d'environnements 3D d'hôtels — Phase 1.",
)

OK = "✓"
KO = "✗"


@app.callback()
def main(verbose: bool = typer.Option(False, "--verbose", "-v", help="Journal détaillé.")) -> None:
    load_env()
    pipeline_logging.configure(verbose=verbose)


@app.command()
def version() -> None:
    """Affiche la version du pipeline."""
    typer.echo(f"hotel-pipeline {__version__}")


@app.command(name="provider-check")
def provider_check() -> None:
    """Contrôle la configuration et la santé des fournisseurs (§6)."""
    statuses = check_providers()
    width = max(len(s.provider.name) for s in statuses)
    blocking = []

    for status in statuses:
        mark = OK if status.configured else KO
        typer.echo(f"{status.provider.name:<{width}}  {mark} {status.label}")
        if status.blocking:
            blocking.append(status.provider.name)

    if blocking:
        typer.echo("")
        typer.secho(
            f"Source obligatoire indisponible : {', '.join(blocking)}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)


@app.command()
def init(
    hotel_id: str = typer.Argument(..., help="Identifiant court, ex. welcominns-boucherville."),
    address: str = typer.Option(..., "--address", help="Adresse officielle complète."),
    lat: float | None = typer.Option(None, "--lat", help="Latitude connue, court-circuite le géocodage."),
    lon: float | None = typer.Option(None, "--lon", help="Longitude connue."),
    radius_m: int = typer.Option(300, "--radius", help="Rayon de collecte, en mètres."),
    place_query: str | None = typer.Option(None, "--place", help="Requête Places."),
    website_url: str | None = typer.Option(None, "--website", help="Site officiel."),
    official_name: str | None = typer.Option(
        None, "--name", help="Nom officiel de l'établissement, pour le profil."
    ),
    country: str | None = typer.Option(
        None, "--country", help="Pays ISO 3166-1 alpha-2, ex. CA, FR."
    ),
    subdivision: str | None = typer.Option(
        None, "--subdivision", help="Subdivision ISO 3166-2 sans le pays, ex. QC."
    ),
    tz: str | None = typer.Option(
        None, "--timezone", help="Fuseau IANA, ex. America/Toronto, Europe/Paris."
    ),
    ocr_language: list[str] = typer.Option(
        [], "--ocr-language", help="Langue d'OCR attendue ; répétable."
    ),
    assume_rights: bool = typer.Option(
        False,
        "--assume-rights",
        help="Assumer l'usage des sources aux droits non établis (tracé au manifeste).",
    ),
    force: bool = typer.Option(False, "--force", help="Réécrit un manifeste existant."),
) -> None:
    """Crée l'espace de travail et le manifeste de projet (§18).

    Fournir --lat/--lon supprime la dépendance au géocodeur, dont le code postal
    a divergé de l'officiel sur cette adresse.
    """
    if (lat is None) != (lon is None):
        typer.secho("--lat et --lon vont ensemble", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    workspace = Workspace(hotel_id)

    if workspace.manifest_path.is_file() and not force:
        typer.secho(
            f"{hotel_id} existe déjà — {workspace.manifest_path}. Utilisez --force pour réécrire.",
            fg=typer.colors.YELLOW,
        )
        raise typer.Exit(code=0)

    workspace.create()
    workspace.write_manifest(
        ProjectManifest(
            hotel_id=hotel_id,
            address=address,
            lat=lat,
            lon=lon,
            collect_radius_m=radius_m,
            place_query=place_query,
            website_url=website_url,
            assume_rights=assume_rights,
        )
    )
    # Une politique **complète** dès l'origine : un fichier partiel ferait
    # venir du code des seuils décisionnels, qu'aucun rapport ne pourrait
    # ensuite citer. L'établissement suivant n'aura pas à migrer.
    from .schemas import DEFAULT_POLICY

    policy_path = workspace.path("00_manifest", "pipeline_policy.json")
    if not policy_path.is_file():
        policy_path.write_text(
            json.dumps(
                json.loads(DEFAULT_POLICY.model_dump_json()),
                indent=2, ensure_ascii=False,
            ) + "\n",
            "utf-8",
        )

    typer.echo(f"{OK} espace de travail créé : {workspace.root}")
    typer.echo(f"  {len(SUBDIRS)} répertoires, manifeste et politique initialisés")

    _scaffold_profile(
        hotel_id, address, official_name, country, subdivision, tz,
        list(ocr_language), lat, lon, place_query, website_url,
    )
    if lat is not None:
        typer.echo(f"  position fournie : {lat:.6f}, {lon:.6f} — géocodage court-circuité")


@app.command()
def status(hotel_id: str = typer.Argument(...)) -> None:
    """Affiche l'état d'avancement d'un hôtel."""
    manifest = Workspace(hotel_id).read_manifest()
    done = manifest.completed_steps()

    typer.echo(f"hôtel   : {manifest.hotel_id}")
    typer.echo(f"adresse : {manifest.address}")
    typer.echo(f"statut  : {manifest.status.value if manifest.status else 'en cours'}")
    typer.echo("")

    for name in STEP_ORDER:
        mark = OK if name in done else "·"
        typer.echo(f"  {mark} {name:<12} {STEPS[name].summary}")

    if manifest.blocked:
        typer.echo("")
        typer.secho(
            f"BLOQUÉ sur '{manifest.blocked.step}' : {manifest.blocked.awaiting}",
            fg=typer.colors.YELLOW,
        )
        typer.echo(f"  forme attendue : {manifest.blocked.expected_form}")


def _run_one(hotel_id: str, step_name: str, force: bool) -> None:
    workspace = Workspace(hotel_id)
    manifest = workspace.read_manifest()

    if step_name in manifest.completed_steps() and not force:
        typer.echo(f"{OK} {step_name} déjà exécuté — ignoré (--force pour rejouer)")
        return

    try:
        run_step(step_name, workspace)
    except (GeocodingError, OverpassError, OfflineError, requests.RequestException) as exc:
        # Une source externe indisponible est un incident d'exploitation, pas
        # un défaut du pipeline : message actionnable plutôt que trace brute.
        typer.secho(f"{KO} source externe indisponible : {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=4) from exc
    except StepNotImplemented as exc:
        typer.secho(f"{KO} {exc}", fg=typer.colors.YELLOW)
        raise typer.Exit(code=2) from exc
    except StepBlocked as exc:
        from .schemas import BlockedState

        manifest.blocked = BlockedState(
            step=exc.step, awaiting=exc.awaiting, expected_form=exc.expected_form
        )
        workspace.write_manifest(manifest)
        typer.secho(f"{KO} bloqué : {exc.awaiting}", fg=typer.colors.YELLOW)
        raise typer.Exit(code=3) from exc

    manifest.blocked = None
    manifest.record(StepRecord(name=step_name))
    workspace.write_manifest(manifest)
    typer.echo(f"{OK} {step_name}")


def _step_command(name: str):
    def command(
        hotel_id: str = typer.Argument(...),
        force: bool = typer.Option(False, "--force", help="Rejoue l'étape."),
    ) -> None:
        _run_one(hotel_id, name, force)

    command.__doc__ = STEPS[name].summary
    return command


for _name in STEP_ORDER:
    app.command(name=_name)(_step_command(_name))


@app.command(name="candidates")
def candidates(hotel_id: str = typer.Argument(...)) -> None:
    """Liste les empreintes candidates à BUILDING_MAIN, du mieux classé au moins."""
    spatial = Workspace(hotel_id).read_spatial()
    if spatial is None:
        typer.secho(
            "aucun manifeste spatial — lancez d'abord : hotel-pipeline collect " + hotel_id,
            fg=typer.colors.YELLOW,
        )
        raise typer.Exit(code=1)

    if spatial.geocode:
        typer.echo(
            f"géocodage : {spatial.geocode.lat:.6f}, {spatial.geocode.lon:.6f} "
            f"({spatial.geocode.provider})"
        )
    typer.echo(f"état : {spatial.state.value}   rayon : {spatial.search_radius_m} m")
    typer.echo("")

    for candidate in spatial.ranked():
        mark = "→" if candidate.feature_id == spatial.confirmed_building_id else " "
        typer.echo(
            f"{mark} {candidate.feature_id:<16} score={candidate.score:.2f}  "
            f"{candidate.distance_to_geocode_m:6.0f} m  {candidate.area_m2:8.0f} m²  "
            f"{candidate.tags.get('name', '(sans nom)')}"
        )
        for reason in candidate.score_reasons:
            typer.echo(f"    · {reason}")

    if spatial.assertions:
        typer.echo("")
        for assertion in spatial.assertions:
            typer.echo(f"  {OK if assertion.passed else KO} {assertion.name} — {assertion.detail}")


@app.command(name="confirm-building")
def confirm_building(
    hotel_id: str = typer.Argument(...),
    feature_id: str = typer.Argument(..., help="Ex. way/29382."),
    by: str = typer.Option(..., "--by", help="Auteur de la confirmation."),
    rationale: str = typer.Option(..., "--rationale", help="Justification, conservée en preuve."),
) -> None:
    """Confirme BUILDING_MAIN. Décision humaine, persistée une fois pour toutes."""
    from datetime import datetime, timezone

    workspace = Workspace(hotel_id)
    spatial = workspace.read_spatial()
    if spatial is None:
        typer.secho("aucun manifeste spatial", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    if spatial.candidate(feature_id) is None:
        typer.secho(
            f"{feature_id} n'est pas un candidat connu. "
            f"Voir : hotel-pipeline candidates {hotel_id}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    spatial.confirmed_building_id = feature_id
    spatial.confirmed_by = by
    spatial.confirmed_at = datetime.now(timezone.utc)
    spatial.confirmation_rationale = rationale
    workspace.write_spatial(spatial)

    typer.echo(f"{OK} BUILDING_MAIN = {feature_id} (confirmé par {by})")


def _scaffold_profile(
    hotel_id: str, address: str, official_name: str | None, country: str | None,
    subdivision: str | None, tz: str | None, ocr_languages: list[str],
    lat: float | None, lon: float | None, place_query: str | None,
    website_url: str | None,
) -> None:
    """Écrit le profil de l'établissement, si de quoi le faire a été fourni.

    Rien n'est déduit : ni le pays depuis l'adresse — une chaîne contenant
    « Québec » n'établit pas un territoire — ni le fuseau depuis le pays, ni
    les langues depuis le fuseau. Chaque déduction de ce genre est un repli
    silencieux de plus, et c'est précisément ce que ce lot supprime.

    Un profil incomplet n'est donc pas écrit à moitié : il n'est pas écrit, et
    la commande dit ce qui manque. Les capacités qui l'exigent s'arrêteront
    d'elles-mêmes, avec le même message.
    """
    import os

    from .schemas import PropertyProfile

    required = {
        "--name": official_name, "--country": country, "--timezone": tz,
        "--ocr-language": ocr_languages or None,
    }
    missing = [flag for flag, value in required.items() if not value]

    directory = Path(os.environ.get("HOTEL_PIPELINE_PROFILES", "profiles"))
    path = directory / f"{hotel_id}.json"

    if missing:
        typer.secho(
            f"  · profil non créé — manquent {', '.join(missing)}. "
            f"Les commandes d'identité et de collecte s'arrêteront tant que "
            f"{path} n'existe pas.",
            fg=typer.colors.YELLOW,
        )
        return

    if path.is_file():
        typer.secho(f"  · profil déjà présent : {path}", fg=typer.colors.YELLOW)
        return

    profile = PropertyProfile(
        property_id=hotel_id, address=address, official_name=official_name,
        country_code=country, subdivision_code=subdivision, timezone=tz,
        ocr_languages=ocr_languages, lat=lat, lon=lon,
        place_query=place_query, website_url=website_url,
    )
    directory.mkdir(parents=True, exist_ok=True)
    path.write_text(profile.model_dump_json(indent=2) + "\n", "utf-8")
    typer.echo(f"  profil créé : {path}")


def _context(
    hotel_id: str, capability: Capability | None = None
) -> PipelineContext:
    """Politique et profil de l'établissement, chargés une seule fois.

    La politique vient de l'espace de travail : la chercher dans le répertoire
    courant faisait dépendre le résultat du lieu d'exécution.

    `capability` déclare ce dont la commande a besoin. Sans elle, le contexte
    était chargé en mode permissif et un profil manquant ne coûtait qu'un
    avertissement jaune — ce qui désarmait le verrou d'identité en silence.
    """
    from .capabilities import CapabilityUnavailable, require

    context, warning = PipelineContext.for_workspace(Workspace(hotel_id))

    if capability is None:
        # Toutes les commandes déclarent la leur. En traiter une comme
        # « inspection » par défaut rouvrirait exactement la porte que la
        # matrice ferme : un oubli deviendrait une permission tacite.
        raise RuntimeError(
            "commande sans capacité déclarée : ajoutez-la à l'appel de "
            "`_context`. La matrice de `capabilities` est la seule autorité."
        )

    try:
        check = require(context, capability)
    except CapabilityUnavailable as exc:
        typer.secho(f"{KO} {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    for limitation in check.partial:
        typer.secho(f"  · lecture partielle : {limitation}", fg=typer.colors.YELLOW)
    if warning and not check.partial:
        typer.secho(f"  · {warning}", fg=typer.colors.YELLOW)
    return context


assets_app = typer.Typer(no_args_is_help=True, help="Inventaire et droits des médias (§9).")
app.add_typer(assets_app, name="assets")

preview_app = typer.Typer(
    no_args_is_help=True,
    help="Constats d'aperçu : ce qu'une vue téléchargée établit, besoin par besoin.",
)
assets_app.add_typer(preview_app, name="preview")


@preview_app.command("list")
def preview_list(
    hotel_id: str = typer.Argument(..., help="Établissement concerné."),
) -> None:
    """Couples asset/besoin en attente de constat."""
    from .schemas.preview import PreviewVerdict

    workspace = Workspace(hotel_id)
    manifest = workspace.read_assets()
    if manifest is None:
        typer.secho(f"{KO} aucun manifeste d'assets", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    log = workspace.read_previews()
    rows: list[tuple[str, str, str, str]] = []
    for asset in manifest.assets:
        provenance = getattr(asset, "acquisition", None)
        for demand_id in getattr(provenance, "serves_demands", None) or []:
            latest = log.latest_for(asset.id, demand_id) if log else None
            rows.append((
                asset.id, demand_id,
                (provenance.demand_levels or {}).get(demand_id, "—"),
                latest.verdict.value if latest else "en attente",
            ))

    if not rows:
        typer.echo("  aucun couple rattaché : rien n'a encore été acquis pour un besoin")
        return

    typer.echo(f"  {len(rows)} couple(s) asset/besoin")
    en_attente = 0
    for asset_id, demand_id, level, verdict in sorted(rows):
        if verdict == "en attente":
            en_attente += 1
        typer.echo(
            f"    {asset_id[-24:]:<26} {demand_id:<34} {level[:24]:<26} {verdict}"
        )
    typer.echo("")
    typer.echo(f"  {en_attente} en attente de constat")


@preview_app.command("assess")
def preview_assess(
    hotel_id: str = typer.Argument(..., help="Établissement concerné."),
    asset_id: str = typer.Option(..., "--asset", help="Asset à évaluer."),
    demand_id: str = typer.Option(..., "--demand", help="Besoin évalué."),
    verdict: str = typer.Option(..., "--verdict", help="established, refuted, inconclusive."),
    rationale: str = typer.Option(..., "--rationale", help="Pourquoi ce verdict."),
    assessed_by: str = typer.Option(..., "--by", help="Qui évalue."),
) -> None:
    """Dépose un constat d'aperçu sur un couple asset/besoin."""
    from .schemas.enums import PreviewVerdict
    from .schemas.preview import PreviewAssessmentLog

    try:
        verdict_enum = PreviewVerdict(verdict)
    except ValueError as exc:
        typer.secho(f"{KO} verdict {verdict!r} inconnu", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc

    workspace = Workspace(hotel_id)
    log = workspace.read_previews() or PreviewAssessmentLog(entries=[])
    log.entries.append({
        "asset_id": asset_id,
        "demand_id": demand_id,
        "verdict": verdict_enum.value,
        "rationale": rationale,
        "assessed_by": assessed_by,
        "assessed_at": datetime.now(timezone.utc).isoformat(),
    })
    workspace.write_json("01_sources/preview_assessments.json", json.loads(log.model_dump_json()))
    typer.secho(f"{OK} constat déposé", fg=typer.colors.GREEN)


policy_app = typer.Typer(
    no_args_is_help=True,
    help="Politique de l'établissement : ce sur quoi les décisions se fondent.",
)
app.add_typer(policy_app, name="policy")


@assets_app.command("import")
def assets_import(
    hotel_id: str = typer.Argument(...),
    csv_path: Path = typer.Argument(..., help="Inventaire CSV."),
    images_root: Path | None = typer.Option(None, "--images-root", help="Racine des fichiers."),
) -> None:
    """Importe un inventaire. Les droits sont obligatoires, sans défaut permissif."""
    from .intake import IntakeError, load_csv
    from .schemas import AssetManifest

    workspace = Workspace(hotel_id)
    try:
        loaded = load_csv(csv_path, images_root)
    except IntakeError as exc:
        typer.secho(f"{KO} {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    workspace.write_assets(AssetManifest(hotel_id=hotel_id, assets=loaded))
    typer.echo(f"{OK} {len(loaded)} asset(s) inventorié(s) — aucun éligible production par défaut")


@assets_app.command("gather")
def assets_gather(
    hotel_id: str = typer.Argument(...),
    radius_m: int = typer.Option(300, "--radius", help="Rayon de collecte, en mètres."),
    place_query: str | None = typer.Option(None, "--place", help="Requête Places."),
    assume_rights: bool = typer.Option(
        False,
        "--assume-rights",
        help="Assumer l'usage des sources aux droits non établis (tracé au manifeste).",
    ),
    classify: bool = typer.Option(
        True, "--classify/--no-classify", help="Classification OpenCLIP (couche vision)."
    ),
    force: bool = typer.Option(False, "--force", help="Réécrit le manifeste d'assets."),
    allow_legacy: bool = typer.Option(
        False, "--allow-legacy",
        help="Autoriser la collecte historique sur un projet qui n'en a pas.",
    ),
) -> None:
    """**Historique.** Collecte multi-sources puis tri assisté (§9, §11).

    Cette commande télécharge d'abord et justifie ensuite : elle ignore les
    besoins, ne demande aucun consentement sur le volume, et `--assume-rights`
    y écrit un état de droits sans décision tracée. La chaîne V2 —
    `discover → plan → acquire` — fait tout cela dans l'ordre inverse, qui est
    le bon.

    Elle reste disponible pour le pilote, dont le corpus en provient, et se
    refuse aux projets neufs : un nouvel établissement contournerait sinon
    toute l'architecture.
    """
    from .gather import (
        build_manifest,
        collect_sources,
        download_all,
        summarise,
        triage,
    )

    workspace = Workspace(hotel_id)
    # Avant tout : ce projet a-t-il le droit d'employer la collecte
    # historique ? Le contrôle vient en tête, faute de quoi un projet neuf
    # échouerait sur un motif secondaire et croirait la commande utilisable.
    _refuse_legacy_gather_on_a_new_project(workspace, allow_legacy)

    spatial = workspace.read_spatial()
    if spatial is None or not spatial.confirmed_building_id:
        typer.secho(
            "confirmez d'abord BUILDING_MAIN — la collecte se centre sur le bâtiment",
            fg=typer.colors.YELLOW,
        )
        raise typer.Exit(code=1)

    if workspace.assets_path.is_file() and not force:
        typer.secho("manifeste d'assets déjà présent (--force pour recollecter)", fg=typer.colors.YELLOW)
        raise typer.Exit(code=0)

    building = spatial.candidate(spatial.confirmed_building_id)
    lat, lon = building.centroid_lat, building.centroid_lon
    typer.echo(f"collecte autour de {lat:.6f}, {lon:.6f} (rayon {radius_m} m)")

    context = _context(hotel_id, Capability.TARGETED_COLLECTION)
    images, reports = collect_sources(
        lat, lon, place_query, radius_m, policy=context.policy, building_wkt=building.wkt
    )
    downloaded = download_all(images, workspace, reports)

    for report in reports:
        if report.skipped_reason:
            typer.echo(f"  {KO} {report.name:<12} ignorée — {report.skipped_reason}")
        else:
            typer.echo(
                f"  {OK} {report.name:<12} {report.collected} trouvée(s), "
                f"{report.downloaded} téléchargée(s)"
            )

    manifest = build_manifest(hotel_id, downloaded, assume_rights=assume_rights)

    classifier = None
    if classify:
        try:
            from .triage.classify import Classifier

            classifier = Classifier(policy=context.policy)
        except ImportError:
            typer.secho(
                "  · OpenCLIP absent — classification ignorée "
                "(installer l'extra 'vision' sur la VM GPU)",
                fg=typer.colors.YELLOW,
            )

    gather_report = triage(manifest.assets, classifier=classifier)
    workspace.write_assets(manifest)
    workspace.write_report("01_sources/gather_report.json", gather_report, context)

    typer.echo("")
    for key, value in summarise(manifest).items():
        typer.echo(f"  {key:<18} {value}")
    typer.echo(f"  {'doublons':<18} {gather_report.duplicates}")


@assets_app.command("promote")
def assets_promote(
    hotel_id: str = typer.Argument(...),
    asset_ids: list[str] = typer.Argument(..., help="Identifiants à rendre éligibles."),
) -> None:
    """Rend des assets éligibles production, après revue des droits."""
    from pydantic import ValidationError

    from .intake import IntakeError, promote

    workspace = Workspace(hotel_id)
    manifest = workspace.read_assets()
    if manifest is None:
        typer.secho("aucun manifeste d'assets — importez d'abord", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    try:
        promoted = promote(manifest, asset_ids)
    except (IntakeError, ValidationError) as exc:
        typer.secho(f"{KO} {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    workspace.write_assets(manifest)
    typer.echo(f"{OK} {len(promoted)} asset(s) éligible(s) production")


@assets_app.command("set-entrance-version")
def assets_set_entrance_version(
    hotel_id: str = typer.Argument(...),
    asset_id: str = typer.Argument(...),
    version: str = typer.Argument(..., help="pre_2024 ou post_2024."),
) -> None:
    """Fixe la version de l'entrée. Verrou humain : non déductible visuellement."""
    from .schemas import EntranceVersion

    workspace = Workspace(hotel_id)
    manifest = workspace.read_assets()
    if manifest is None:
        typer.secho("aucun manifeste d'assets", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    try:
        entrance = EntranceVersion(version)
    except ValueError:
        typer.secho(
            f"{KO} version invalide {version!r} ; attendu pre_2024 ou post_2024",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1) from None

    asset = next((a for a in manifest.assets if a.id == asset_id), None)
    if asset is None:
        typer.secho(f"{KO} asset inconnu : {asset_id}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    manifest.assets[manifest.assets.index(asset)] = asset.model_copy(
        update={"entrance_version": entrance}
    )
    workspace.write_assets(manifest)
    typer.echo(f"{OK} {asset_id} → {entrance.value}")


@assets_app.command("migrate")
def assets_migrate(hotel_id: str = typer.Argument(...)) -> None:
    """Migre le manifeste vers la structure du Lot 1B (§13, étape 1)."""
    from .migration import migrate

    workspace = Workspace(hotel_id)
    manifest = workspace.read_assets()
    if manifest is None:
        typer.secho("aucun manifeste d'assets", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    migrated, report = migrate(manifest)
    workspace.write_assets(migrated)
    workspace.write_json("00_manifest/migration_report.json", report.as_dict())

    typer.echo(f"{OK} {report.total} asset(s) migré(s)")
    for key, value in report.as_dict()["derived"].items():
        typer.echo(f"    dérivé   {key:<26} {value}")
    for key, value in report.left_unknown.items():
        typer.echo(f"    inconnu  {key:<26} {value}")
    if report.unmapped_sources:
        typer.secho(
            f"    sources sans famille connue : {report.unmapped_sources}",
            fg=typer.colors.YELLOW,
        )


@assets_app.command("migrate-review-status")
def assets_migrate_review_status(
    hotel_id: str = typer.Argument(...),
    dry_run: bool = typer.Option(False, "--dry-run", help="Compter sans rien écrire."),
) -> None:
    """Convertit les revues non conclusives vers le statut terminal.

    « Examiné sans conclure » cessait d'être « en attente de revue » : les
    manifestes écrits avant ce changement réclamaient éternellement un travail
    déjà fait. Rien d'autre n'est touché — ni décision, ni historique, ni
    empreinte.
    """
    from .migrate_review_status import migrate_file

    workspace = Workspace(hotel_id)
    if not workspace.assets_path.is_file():
        typer.secho("aucun manifeste d'assets", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    migrated, report = migrate_file(workspace.assets_path)

    if report.converted == 0:
        typer.echo(f"{OK} rien à convertir — {report.already_terminal} déjà terminal(es)")
        return

    if dry_run:
        typer.echo(f"    à convertir  {report.converted}")
        for asset_id in report.converted_ids:
            typer.echo(f"      {asset_id}")
        typer.echo("    --dry-run : rien n'a été écrit")
        return

    workspace.write_assets(migrated)
    workspace.write_json("00_manifest/review_status_migration.json", report.as_dict())

    typer.echo(f"{OK} {report.converted} statut(s) converti(s) sur {report.total}")
    for asset_id in report.converted_ids:
        typer.echo(f"    {asset_id}  needs_review → human_unresolved")
    typer.echo(f"    décisions et historiques inchangés : {report.untouched_decisions}")


@assets_app.command("discover")
def assets_discover(
    hotel_id: str = typer.Argument(...),
    radius_m: int | None = typer.Option(
        None, "--radius", help="Rayon d'interrogation ; défaut : politique de collecte."
    ),
) -> None:
    """Interroge les index des sources. **Aucune image n'est téléchargée.**

    Le besoin juge la collecte : sans `CaptureDemand` déclarées, la découverte
    serait un ramassage sans objectif, et le corpus définirait après coup ce
    qu'on cherchait. Le choix vient au plan, le téléchargement à l'acquisition,
    et l'OCR après elle — à ce stade, aucune image n'existe encore.
    """
    from .discover import DiscoveryRefused, discover
    from .provenance import digest_of
    from .schemas.acquisition import CaptureDemandManifest, DiscoveryMode, DiscoveryScope

    context = _context(hotel_id, Capability.TARGETED_COLLECTION)
    workspace = Workspace(hotel_id)

    payload = workspace.read_json("01_sources/capture_demands.json")
    if not payload:
        typer.secho(
            f"{KO} aucun besoin déclaré : écrivez "
            f"01_sources/capture_demands.json avant de découvrir",
            fg=typer.colors.RED, err=True,
        )
        raise typer.Exit(code=1)

    demands = CaptureDemandManifest.model_validate(payload)
    profile = context.profile
    radius = radius_m or context.policy.collection.radius_m

    scope, demands = _discovery_scope(demands, [d.demand_id for d in demands.demands], digest_of(payload))

    typer.echo(f"  besoins     {len(demands.demands)}")
    typer.echo(f"  position    {profile.lat:.6f}, {profile.lon:.6f}")
    typer.echo(f"  rayon       {radius} m")

    # La recherche adaptative n'interroge que les besoins encore ouverts, et
    # ne **recommande** que ce qui les sert : le plan reste seul à décider ce
    # qui sera téléchargé.
    search = _adaptive_context(workspace, context, demands.demands, payload)

    # Le décompte porte sur **cette** commande : un total cumulé entre deux
    # exécutions ne dirait rien de l'une ni de l'autre.
    from .providers.transport import ledger, reset_ledger

    registre = reset_ledger()
    # Ce qu'on s'attend à émettre **au plus**. La pagination interdit un compte
    # exact d'avance ; annoncer un plafond vaut mieux que se taire, et le
    # comparer à l'effectif dit si l'estimation valait.
    registre.planned_max_requests.update(_planned_calls(context, radius))
    queries = _query_sources(
        profile, context, workspace, radius, search.outstanding or demands.demands, scope,
    )
    search.requests_by_source = _requests_by_source(ledger().by_source())

    try:
        manifest, report = discover(
            hotel_id, demands, queries,
            demand_digest=digest_of(payload),
            policy_digest=context.provenance["policy_digest"],
            search=search,
        )
    except DiscoveryRefused as exc:
        typer.secho(f"{KO} {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    # Le registre entier, non un extrait : les échecs, les pages et les refus
    # ne se lisaient nulle part.
    report.transport = ledger().as_dict()

    # Un rejeu sur cache figé n'est pas une découverte : il ne doit jamais
    # devenir le « dernier manifeste » que `assets plan` ramasserait. Il vit
    # donc à part, sous `replays/`, où le tri par date ne va pas le chercher.
    from .providers.transport import NetworkMode, current_mode

    # Une source d'images en échec produit un corpus **partiel**. Le publier
    # comme manifeste courant ferait planifier sur ce qui a répondu, en
    # présentant l'absence de l'autre comme une absence de vues. Il reste
    # écrit — le diagnostic en dépend — mais à part.
    image_sources = {"mapillary", "street_view"}
    failed = sorted(image_sources & set(report.sources_skipped))
    partial = bool(failed)

    replay = current_mode() is NetworkMode.CACHE_ONLY
    if scope.mode is DiscoveryMode.TARGETED:
        prefix = f"01_sources/targeted/{report.run_id}"
    else:
        prefix = "01_sources/replays" if (replay or partial) else "01_sources"
    if partial:
        typer.secho(
            f"  · corpus partiel — {', '.join(failed)} n'a pas répondu : écrit "
            f"sous {prefix}/, il ne devient pas le manifeste courant. "
            "Planifier dessus prendrait une source absente pour une source vide.",
            fg=typer.colors.YELLOW,
        )
    if replay:
        typer.secho(
            f"  · rejeu sur cache figé : écrit sous {prefix}/, il ne "
            "sera pas repris comme manifeste courant",
            fg=typer.colors.YELLOW,
        )

    # La portée est inscrite **au manifeste** : sans elle, deux corpus de
    # contenus incomparables se ressembleraient.
    manifest = manifest.model_copy(update={"scope": scope})

    if scope.mode is DiscoveryMode.TARGETED:
        manifest = _targeted_manifest(workspace, manifest, scope)

    workspace.write_json(
        f"{prefix}/candidates_{report.run_id}.json",
        json.loads(manifest.model_dump_json()),
    )
    workspace.write_report(f"{prefix}/discovery_{report.run_id}.json", report, context, production="CandidateManifest")

    for source in report.sources_queried:
        returned = report.candidates_by_source[source]
        kept = sum(1 for c in manifest.candidates if c.source == source)
        emitted = report.requests_by_source.get(source)
        detail = f"{returned:>5} rendu(s)"
        if emitted is not None:
            # Appels et résultats sont deux choses : une source prolixe n'est
            # pas une source souvent interrogée.
            detail += f" pour {emitted.total} appel(s)"
        if kept != returned:
            # Les deux chiffres, sinon « 2163 » à côté d'un total de 1636 se
            # lit comme une incohérence alors que c'est un regroupement.
            detail += f" → {kept} retenu(s) après regroupement"
        typer.echo(f"    {source:<14} {detail}")
    for source, reason in sorted(report.sources_skipped.items()):
        typer.secho(f"    {source:<14} non interrogée — {reason}", fg=typer.colors.YELLOW)

    counts = report.viewpoint_counts
    if counts:
        typer.echo("")
        typer.echo(f"    cadrages candidats  {counts['framing_candidates']:>6}")
        typer.echo(f"    panoramas distincts {counts['distinct_panoramas']:>6}")
        typer.echo(f"    points de vue       {counts['viewpoints']:>6}")

    if report.search is not None:
        typer.echo("")
        typer.echo("  recommandations par besoin — le plan décide seul :")
        levels = {
            m.candidate_id: m.recommendation_level
            for m in report.search.measures if m.recommendation_level
        }
        for demand_id in sorted(report.search.recommended):
            retained = report.search.recommended[demand_id]
            considered = sum(
                1 for m in report.search.measures
                if m.demand_id == demand_id and m.rejection_reason is None
            )
            full = sum(
                1 for c in retained
                if levels.get(c) and levels[c].value == "eligible_for_full_acquisition"
            )
            typer.echo(
                f"    {demand_id:<36} {len(retained)} retenue(s) "
                f"sur {considered} éligible(s) — {full} acquérable(s)"
            )
        stats = report.search.distance_distribution
        if stats:
            typer.echo("")
            typer.echo(
                "  distances à la cible — le seuil de recommandation "
                "automatique n'est pas calibré :"
            )
            for demand_id in sorted(stats):
                row = stats[demand_id]
                typer.echo(
                    f"    {demand_id:<34} min {row['min_m']:>6.0f} m  "
                    f"méd {row['median_m']:>6.0f} m  "
                    f"{row['within_automatic_range']:>4} sous {row['limit_m']:.0f} m "
                    f"sur {row['measured']}"
                )

        for stage, reason in sorted(report.search.stages_skipped.items()):
            typer.secho(f"    étape non exécutée : {stage} — {reason}",
                        fg=typer.colors.YELLOW)
        for demand_id, reason in sorted(report.search.demands_skipped.items()):
            typer.secho(f"    {demand_id:<36} ignoré — {reason}",
                        fg=typer.colors.YELLOW)

    typer.echo("")
    typer.echo(f"{OK} {len(manifest.candidates)} candidat(s), 0 octet téléchargé")
    typer.echo(f"    doublons écartés : {report.duplicates_dropped}")
    typer.echo("    prochaine étape : assets plan")


def _refuse_legacy_gather_on_a_new_project(workspace, allowed: bool) -> None:  # noqa: ANN001
    """Refuse la collecte historique là où aucun corpus n'en provient.

    Un projet dont le manifeste d'assets est vide n'a rien à récupérer d'un
    ramassage antérieur : lui en offrir un revient à contourner besoins,
    consentement et décisions de droits.
    """
    if allowed:
        typer.secho(
            "  · collecte historique autorisée explicitement — elle télécharge "
            "avant d'évaluer, et n'établit aucun droit",
            fg=typer.colors.YELLOW,
        )
        return

    manifest = workspace.read_assets()
    if manifest is not None and manifest.assets:
        typer.secho(
            "  · collecte historique sur un corpus existant — préférez "
            "assets discover → plan → acquire",
            fg=typer.colors.YELLOW,
        )
        return

    typer.secho(
        f"{KO} collecte historique refusée sur un projet neuf : elle télécharge "
        f"avant d'évaluer, ignore les besoins et ne demande aucun consentement "
        f"sur le volume. Utilisez « assets discover » puis « plan » et "
        f"« acquire ». Pour passer outre : --allow-legacy.",
        fg=typer.colors.RED, err=True,
    )
    raise typer.Exit(code=2)


@dataclass
class AdaptiveContext:
    """Ce que la recherche sait avant d'interroger quoi que ce soit."""

    outstanding: list = field(default_factory=list)
    skipped: dict = field(default_factory=dict)
    anchors: dict = field(default_factory=dict)
    target: tuple | None = None

    #: De quel côté du bâtiment se tient chaque candidat. Sans lui, les besoins
    #: sectoriels ne se distinguent pas les uns des autres.
    sector: object = None

    #: `candidate_id` → point de vue, pour compter les quotas en observations
    #: et non en cadrages.
    viewpoints: dict = field(default_factory=dict)

    #: Distance en deçà de laquelle deux caméras sont un seul point de vue.
    viewpoint_separation_m: float | None = None

    #: Écart de cap en deçà duquel deux cadrages montrent la même chose.
    framing_merge_bearing_deg: float | None = None

    #: Appels émis par source, mesurés — non estimés.
    requests_by_source: dict = field(default_factory=dict)

    policy: object = None
    collection: object = None
    lineage: dict = field(default_factory=dict)


def _adaptive_context(workspace, context, demands, demands_payload):  # noqa: ANN001, ANN201
    """Besoins ouverts, ancres compatibles et filiation, avant toute requête.

    Une cible non résolue figure dans les besoins ignorés **avec son motif** :
    la remplacer en silence par le bâtiment ferait chercher la façade en
    croyant chercher l'entrée.
    """
    from .adaptive_search import anchors_for, open_demands
    from .plan import group_viewpoints
    from .policy_facets import dependency_digests
    from .provenance import digest_of
    from .schemas.acquisition import DemandAssessmentManifest

    assessment_payload = _latest_json(workspace, "demand_assessment_*.json")
    assessment = (
        DemandAssessmentManifest.model_validate(assessment_payload)
        if assessment_payload else None
    )
    if assessment is None:
        typer.secho(
            "  · aucune évaluation des besoins : la recherche ne sait pas quels "
            "secteurs sont déficitaires — lancez « assets demands assess »",
            fg=typer.colors.YELLOW,
        )
        return AdaptiveContext(
            policy=context.policy.adaptive_search,
            collection=context.policy.collection,
        )

    assets = workspace.read_assets()
    corpus = assets.assets if assets else []
    viewpoints = group_viewpoints(
        [_as_viewpoint_subject(asset) for asset in corpus],
        separation_m=context.policy.geometry.viewpoint_separation_m,
    )

    outstanding = open_demands(assessment, demands)
    skipped = {
        demand.demand_id: "besoin déjà satisfait"
        for demand in demands
        if demand not in outstanding
    }

    anchors = {}
    for demand in outstanding:
        found = anchors_for(demand, corpus, viewpoints, {})
        anchors[demand.demand_id] = found

    geometry = _capture_geometry_if_any(workspace, context)
    target = _target_position(workspace)
    sector = _sector_context(workspace, context, outstanding, geometry)

    return AdaptiveContext(
        outstanding=outstanding,
        skipped=skipped,
        anchors=anchors,
        target=target,
        sector=sector,
        viewpoints=viewpoints,
        viewpoint_separation_m=context.policy.geometry.viewpoint_separation_m,
        framing_merge_bearing_deg=(
            context.policy.collection.framing_merge_bearing_deg
        ),
        policy=context.policy.adaptive_search,
        collection=context.policy.collection,
        lineage={
            "demand_digest": digest_of(demands_payload),
            "demand_assessment_digest": digest_of(assessment_payload),
            "asset_manifest_digest": (
                digest_of(json.loads(assets.model_dump_json())) if assets else None
            ),
            "capture_geometry_digest": (
                digest_of(json.loads(geometry.model_dump_json())) if geometry else None
            ),
            "policy_dependency_digests": dependency_digests(
                context.policy, "CandidateManifest"
            ),
        },
    )


def _latest_json(workspace, pattern: str):  # noqa: ANN001, ANN201
    found = sorted(workspace.path("01_sources").glob(pattern))
    usable = [path for path in found if "report" not in path.name]
    if not usable:
        return None
    return json.loads(usable[-1].read_text("utf-8"))


def _target_position(workspace):  # noqa: ANN001, ANN201
    spatial = _safe_read(workspace.read_spatial)
    if spatial is None or not spatial.confirmed_building_id:
        return None
    building = spatial.candidate(spatial.confirmed_building_id)
    return (building.centroid_lat, building.centroid_lon)


def _query_sources(profile, context, workspace, radius_m: int, demands) -> dict:  # noqa: ANN001
    """Interroge les index disponibles, et dit pourquoi les autres ne le sont pas.

    Une source non configurée n'est pas une source vide : le motif remonte au
    manifeste, et le plan saura qu'il ne juge pas un corpus complet.

    Aucune image n'est téléchargée ici, y compris pour Street View : son
    endpoint de métadonnées est gratuit et suffit à savoir où existe un
    panorama. L'endpoint image, facturé, n'intervient qu'à l'acquisition.
    """
    from .collectors import mapillary
    from .discover import candidates_from
    from .providers.cache import OfflineError

    queries: dict = {}

    try:
        images = mapillary.collect(profile.lat, profile.lon, radius_m=radius_m)
    except OfflineError as exc:
        queries["mapillary"] = f"hors ligne : {exc}"
    except (requests.RequestException, RuntimeError, ValueError) as exc:
        queries["mapillary"] = f"interrogation impossible : {exc}"
    else:
        queries["mapillary"] = candidates_from("mapillary", images)

    queries["street_view"] = _street_view_candidates(
        context, workspace, radius_m, demands
    )
    return queries


def _street_view_candidates(context, workspace, radius_m: int, demands):  # noqa: ANN001, ANN201
    """Panoramas des corridors proches, cadrés vers ce que les besoins demandent.

    Le cap est **dirigé**, jamais balayé : tourner l'horizon produirait des
    acquisitions que rien ne réclame, et le consentement porterait sur elles.
    """
    from .collectors.streetview_v2 import (
        candidate_from, discover_panoramas, framings_for_targets,
    )
    from .demand_targets import TargetUnresolved, resolve
    from .geo.geometry_loader import LegacyManifestRefused, load_capture_geometry
    from .providers.cache import OfflineError

    reference = context.spatial_reference
    geometry_path = workspace.path("06_geo", "capture_geometry.json")
    if reference is None or not geometry_path.is_file():
        return (
            "aucune géométrie de capture : les corridors où chercher des "
            "panoramas ne sont pas résolus"
        )

    try:
        manifest, _ = load_capture_geometry(geometry_path, reference)
    except LegacyManifestRefused as exc:
        return f"géométrie illisible : {exc}"

    corridors = _corridor_elements(manifest)
    if not corridors:
        return "aucun corridor résolu autour du site"

    spatial = _safe_read(workspace.read_spatial)
    front = getattr(spatial, "front_azimuth_deg", None) if spatial else None
    targets = []
    for demand in demands:
        try:
            targets.append(
                resolve(demand, manifest, front, _safe_read(workspace.read_site))
            )
        except TargetUnresolved:
            continue
    if not targets:
        return "aucune cible de besoin résolue : rien vers quoi cadrer"

    try:
        panoramas, skipped = discover_panoramas(
            corridors,
            spacing_m=context.policy.collection.sample_spacing_m,
            snap_radius_m=context.policy.collection.snap_radius_m,
        )
    except OfflineError as exc:
        return f"hors ligne : {exc}"
    except (requests.RequestException, RuntimeError, ValueError) as exc:
        return f"interrogation impossible : {exc}"

    if skipped:
        typer.secho(
            f"    street_view · {len(skipped)} point(s) sans réponse",
            fg=typer.colors.YELLOW,
        )

    return [
        candidate_from(panorama, framing)
        for panorama in panoramas
        for framing in framings_for_targets(panorama, targets)
    ]


def _corridor_elements(manifest) -> list[dict]:  # noqa: ANN001
    """Corridors résolus, au format qu'attend l'échantillonnage de voirie."""
    from shapely import wkt as shapely_wkt

    from .schemas.geometry import GeometryResolutionStatus, GeometryRole

    elements = []
    for geometry in manifest.geometries:
        if geometry.role not in (GeometryRole.ROAD_CANDIDATE, GeometryRole.ACCESS_ROAD):
            continue
        if geometry.resolution_status is not GeometryResolutionStatus.RESOLVED:
            continue
        shape = shapely_wkt.loads(geometry.wgs84_wkt)
        coords = (
            list(shape.coords) if shape.geom_type == "LineString"
            else [point for part in shape.geoms for point in part.coords]
        )
        elements.append(
            {"geometry": [{"lat": lat, "lon": lon} for lon, lat in coords]}
        )
    return elements


@assets_app.command("plan")
def assets_plan(
    hotel_id: str = typer.Argument(...),
    candidates_file: Path | None = typer.Option(
        None, "--candidates", help="Manifeste de candidats ; défaut : le plus récent."
    ),
    measure_volumes: bool = typer.Option(
        False, "--measure-volumes",
        help="Interroge les en-têtes pour connaître les tailles. Ne télécharge aucun corps.",
    ),
    consent_bytes: int | None = typer.Option(
        None, "--consent-bytes",
        help="Volume exact accepté, en octets. Rend le plan exécutable.",
    ),
) -> None:
    """Choisit ce qui sera acquis. **Ne télécharge rien.**

    Le plan naît brouillon. Il ne devient exécutable qu'avec `--consent-bytes`,
    et seulement si le volume annoncé est exact : consentir à un total dont une
    part est inconnue serait consentir à ce qu'on n'a pas montré.
    """
    from .plan import PlanRefused, build, consent
    from .provenance import digest_of
    from .schemas.acquisition import (
        CandidateManifest, CaptureDemandManifest, PlanStatus, VolumeStatus,
    )

    context = _context(hotel_id, Capability.TARGETED_COLLECTION)
    workspace = Workspace(hotel_id)

    # Registre propre : un plan hérité des appels de la découverte annoncerait
    # un coût qui n'est pas le sien.
    from .providers.transport import ledger as plan_ledger, reset_ledger

    plan_registry = reset_ledger()

    demands_payload = workspace.read_json("01_sources/capture_demands.json")
    if not demands_payload:
        typer.secho(f"{KO} aucun besoin déclaré", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    path = candidates_file or _latest_candidates(workspace)
    if path is None:
        typer.secho(
            f"{KO} aucun manifeste de candidats — lancez : assets discover",
            fg=typer.colors.RED, err=True,
        )
        raise typer.Exit(code=1)

    candidates_payload = json.loads(Path(path).read_text("utf-8"))
    candidates = CandidateManifest.model_validate(candidates_payload)
    demands = CaptureDemandManifest.model_validate(demands_payload)

    # Liaison des deux manifestes. `--candidates` permet de désigner un fichier
    # arbitraire : sans ces contrôles, un manifeste d'un autre établissement ou
    # produit contre d'anciens besoins entrerait au plan sans un mot.
    from .schemas.acquisition import validate_recommendation_demands

    mismatches = _validate_manifest_pairing(
        hotel_id, candidates, demands, demands_payload
    )
    mismatches += validate_recommendation_demands(candidates, demands)
    if mismatches:
        for problem in mismatches:
            typer.secho(f"{KO} {problem}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    coverage = _check_coverage(workspace, demands.demands)

    digests = _plan_digests(workspace, context, candidates_payload, demands_payload)
    geometries, geometry_report = _candidate_geometries(
        workspace, context, candidates.candidates, demands.demands
    )

    # La mesure vient **après** la sélection : mesurer les 1 636 candidats pour
    # n'en retenir que neuf lancerait des milliers d'appels inutiles, dont la
    # quasi-totalité sur des vues que le plan n'acquerra jamais.
    try:
        plan, evaluations, report = build(
            hotel_id, candidates.candidates, demands.demands, digests,
            geometries=geometries, sizes=None,
            separation_m=context.policy.geometry.viewpoint_separation_m,
            policy=context.policy,
            # Les niveaux prononcés par la recherche **contraignent** le plan :
            # sans eux, les trois listes publiées restaient informatives et une
            # preview entrait en pleine résolution.
            levels=_recommendation_levels(candidates),
        )
    except PlanRefused as exc:
        typer.secho(f"{KO} {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    # Les requêtes résolues : c'est sur elles que porteront la mesure, le
    # téléchargement et le consentement — jamais sur le candidat, dont la
    # résolution est celle qu'y a laissée la découverte.
    from .acquisition_request import RequestUnresolvable, resolve_all

    try:
        acquisition_requests = resolve_all(
            {c.candidate_id: c for c in candidates.candidates}, plan.acquisitions
        )
    except RequestUnresolvable as exc:
        typer.secho(f"{KO} {exc}", fg=typer.colors.RED, err=True)
        typer.secho(
            "  aucun plan écrit : ce qu'il demande ne se traduit pas dans les "
            "termes de la source",
            fg=typer.colors.YELLOW, err=True,
        )
        raise typer.Exit(code=1) from exc

    sizes, volume_report = _measure_volumes(
        [acquisition_requests[a.candidate_id] for a in plan.acquisitions
         if a.candidate_id in acquisition_requests],
        measure_volumes,
    )
    # Chaque acquisition porte désormais ce qui sera demandé pour elle : le
    # consentement verrouille l'empreinte, non le seul candidat.
    plan = plan.model_copy(update={"acquisitions": [
        a.model_copy(update={
            "provider_resolution": (
                acquisition_requests[a.candidate_id].provider_resolution
                if a.candidate_id in acquisition_requests else None
            ),
            "request_digest": (
                acquisition_requests[a.candidate_id].digest
                if a.candidate_id in acquisition_requests else None
            ),
            **({"expected_bytes": sizes.get(a.candidate_id)} if sizes else {}),
        })
        for a in plan.acquisitions
    ]})

    # Un brouillon irréalisable est un brouillon faux. `bind_plan` existait,
    # était testé, et n'était appelé nulle part : la contradiction entre ce que
    # le plan demande et ce que le fournisseur propose n'apparaissait qu'au
    # moment de payer.
    from .schemas.acquisition import bind_plan

    # Confronté aux évaluations **que le plan vient de produire** : celles du
    # manifeste de découverte sont antérieures à la mesure de cadrage, et un
    # candidat peut y être éligible avant d'être écarté par la géométrie.
    # Juger sur elles comparerait le plan à un état périmé.
    measured = candidates.model_copy(update={"evaluations": evaluations})
    broken = bind_plan(plan, measured, demands)
    if broken:
        for problem in broken:
            typer.secho(f"{KO} {problem}", fg=typer.colors.RED, err=True)
        typer.secho(
            "  aucun plan écrit : un brouillon qui ne peut pas s'exécuter "
            "annoncerait une dépense impossible",
            fg=typer.colors.YELLOW, err=True,
        )
        raise typer.Exit(code=1)

    typer.echo(f"  candidats   {report.candidates}")
    if geometry_report is not None:
        typer.echo(
            f"  mesurés     {geometry_report.measured} "
            f"({geometry_report.with_framing} avec cadrage)"
        )
        for reason, count in sorted(geometry_report.without_framing.items()):
            typer.echo(f"    sans cadrage · {reason:<40} {count:>4}")
        if geometry_report.wrong_sector:
            typer.echo(f"    hors secteur  {geometry_report.wrong_sector}")
        for demand_id, reason in sorted(geometry_report.unresolved_targets.items()):
            typer.secho(
                f"    cible non résolue · {demand_id} — {reason[:48]}",
                fg=typer.colors.YELLOW,
            )
    typer.echo(f"  évaluations {report.evaluations}")
    typer.echo(f"  retenues    {report.selected}")
    if report.preview_required:
        typer.echo(f"  à vérifier  {report.preview_required} (miniature d'abord)")
    for reason, count in sorted(report.rejected_by_reason.items()):
        typer.echo(f"    écarté · {reason[:52]:<52} {count:>4}")
    if report.demands_unplanned:
        typer.secho(
            f"  besoins sans acquisition prévue : "
            f"{', '.join(report.demands_unplanned)}",
            fg=typer.colors.YELLOW,
        )
    if report.demands_planned_pending_preview:
        pending = ", ".join(sorted(report.demands_planned_pending_preview))
        typer.secho(
            f"  · prévu mais non établi — vérification par miniature : {pending}",
            fg=typer.colors.YELLOW,
        )

    typer.echo("")
    if volume_report is not None:
        typer.echo(
            f"  tailles mesurées {len(volume_report.measured)} "
            f"({len(volume_report.unmeasured)} inconnue(s))"
        )
        for candidate_id, reason in sorted(volume_report.unmeasured.items())[:3]:
            typer.secho(f"    sans taille · {candidate_id} — {reason[:44]}",
                        fg=typer.colors.YELLOW)

    typer.echo(f"  volume connu    {report.known_bytes:,} octets".replace(",", " "))
    typer.echo(f"  taille inconnue {report.unknown_size_items} acquisition(s)")
    typer.echo(f"  statut          {report.volume_status}")

    if consent_bytes is not None:
        # Un brouillon montre les obligations manquantes ; un plan exécutable
        # ne peut pas les ignorer. Consentir à acquérir un corpus dont on sait
        # qu'il laisse un objet sans couverture, c'est le figer incomplet.
        if not coverage.complete:
            typer.secho(
                f"{KO} consentement refusé : {len(coverage.unmet)} obligation(s) "
                f"sans demande ni dispense — {', '.join(coverage.unmet)}. "
                f"Déclarez une demande, ou une dispense motivée.",
                fg=typer.colors.RED, err=True,
            )
            raise typer.Exit(code=2)
        if plan.volume_status is not VolumeStatus.EXACT:
            typer.secho(
                f"{KO} consentement refusé : le volume est « "
                f"{plan.volume_status.value} ». Consentir à un total dont une "
                f"part est inconnue serait consentir à ce qui n'a pas été montré.",
                fg=typer.colors.RED, err=True,
            )
            raise typer.Exit(code=2)
        if consent_bytes != plan.known_bytes:
            typer.secho(
                f"{KO} consentement refusé : {consent_bytes} octets acceptés, "
                f"{plan.known_bytes} annoncés. Le consentement porte sur le "
                f"volume exact, non sur un ordre de grandeur.",
                fg=typer.colors.RED, err=True,
            )
            raise typer.Exit(code=2)
        try:
            plan = consent(
                plan, digests,
                measured_from=plan.plan_id,
                # Ce que « télécharger » garantit fait partie de ce qu'on
                # accepte : un volume consenti sous d'autres garanties n'est
                # plus le même engagement.
                download_contract_version=(
                    context.policy.collection.download_contract_version
                ),
            )
        except PlanRefused as exc:
            typer.secho(f"{KO} {exc}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=2) from exc

    workspace.write_json(
        f"01_sources/acquisition_plan_{plan.plan_id}.json",
        json.loads(plan.model_dump_json()),
    )
    report.transport = plan_ledger().as_dict()
    workspace.write_report(
        f"01_sources/plan_report_{plan.plan_id}.json", report, context,
        production="AcquisitionPlan",
    )

    # Les évaluations étaient calculées puis jetées : « pourquoi ce candidat
    # a-t-il été écarté » n'était lisible nulle part, et la recherche suivante
    # ne pouvait rien en apprendre.
    workspace.write_report(
        f"01_sources/candidate_evaluations_{plan.plan_id}.json",
        {
            "plan_id": plan.plan_id,
            "note": (
                "deux lignes d'un même panorama ne sont pas un doublon : ce "
                "sont deux cadrages, et le cadrage change le verdict. Le "
                "couple (candidate_id, demand_id) reste unique."
            ),
            "evaluations": _readable_evaluations(
                evaluations, candidates.candidates,
                context.policy.geometry.viewpoint_separation_m,
            ),
        },
        context, production="CandidateEvaluation",
    )

    typer.echo("")
    if plan.status is PlanStatus.EXECUTABLE:
        typer.echo(f"{OK} plan {plan.plan_id} — exécutable, {plan.known_bytes} octets consentis")
        typer.echo("    prochaine étape : assets acquire")
    else:
        typer.echo(f"{OK} plan {plan.plan_id} — brouillon, rien ne sera téléchargé")
        plan_path = workspace.path(f"01_sources/acquisition_plan_{plan.plan_id}.json")
        if plan.unknown_size_items:
            typer.echo(
                f"    pour l'exécuter : assets measure-plan {hotel_id} "
                f"--plan {plan_path}"
            )
        else:
            typer.echo(
                f"    pour l'exécuter : assets consent-plan {hotel_id} "
                f"--plan {plan_path} --consent-bytes {plan.known_bytes}"
            )


def _candidate_geometries(workspace, context, candidates, demands):  # noqa: ANN001, ANN201
    """Mesure ce que la géométrie permet, ou dit pourquoi elle ne le permet pas.

    Transmet **tout** ce dont la mesure a besoin : le manifeste géométrique —
    et non une empreinte unique, qui ferait mesurer chaque besoin contre le
    bâtiment — l'orientation de façade sans laquelle un secteur ne veut rien
    dire, le manifeste de site pour les références indirectes, les obstacles,
    et le seuil de secteur de la politique matérialisée.
    """
    from .candidate_geometry import measure_all
    from .geo.geometry_loader import LegacyManifestRefused, load_capture_geometry
    from .geo.projection import ProjectionRefused, ProjectionService

    reference = context.spatial_reference
    geometry_path = workspace.path("06_geo", "capture_geometry.json")
    if reference is None or not geometry_path.is_file():
        typer.secho(
            "  · aucune mesure de cadrage : contexte spatial ou géométrie de "
            "capture absents — les candidats resteront à vérifier",
            fg=typer.colors.YELLOW,
        )
        return {}, None

    spatial = _safe_read(workspace.read_spatial)
    front = getattr(spatial, "front_azimuth_deg", None) if spatial else None
    if front is None:
        typer.secho(
            "  · orientation de façade inconnue : les besoins de secteur ne "
            "seront pas mesurés — « avant » et « arrière » ne se distinguent pas",
            fg=typer.colors.YELLOW,
        )

    try:
        manifest, _ = load_capture_geometry(geometry_path, reference)
        measured, report = measure_all(
            candidates, manifest, ProjectionService(reference),
            context.policy.visibility, demands,
            obstacles=_obstacle_shapes(manifest),
            front_azimuth_deg=front,
            site=_safe_read(workspace.read_site),
            half_width_deg=context.policy.geometry.sector_observer_half_width_deg,
            forbidden_zones=_forbidden_zones(manifest),
        )
    except (LegacyManifestRefused, ProjectionRefused, ValueError) as exc:
        typer.secho(f"  · mesure impossible : {exc}", fg=typer.colors.YELLOW)
        return {}, None

    return measured, report


def _forbidden_zones(manifest) -> dict:  # noqa: ANN001
    """Zones interdites nommées par le manifeste géométrique, par référence."""
    from shapely import wkt as shapely_wkt

    from .schemas.geometry import GeometryResolutionStatus

    return {
        geometry.feature_id: shapely_wkt.loads(geometry.projected_wkt)
        for geometry in manifest.geometries
        if geometry.resolution_status is GeometryResolutionStatus.RESOLVED
    }


def _obstacle_shapes(manifest) -> list:  # noqa: ANN001
    from .geo.visibility_run import _obstacles

    return _obstacles(manifest, {})


def _check_coverage(workspace, demands):  # noqa: ANN001, ANN201
    """Confronte les besoins aux obligations du gabarit, avant tout plan.

    Un manifeste peut omettre la façade arrière et paraître complet : ses
    besoins seraient tous satisfaits, sans que rien ne dise qu'un objet n'en a
    jamais eu. Le plan reste possible — planifier ce qu'on a demandé est
    légitime — mais l'oubli est nommé.
    """
    from .coverage_obligations import assess, missing_demands

    report = assess(demands, _read_waivers(workspace))
    if report.complete:
        typer.echo(f"  obligations couvertes ({len(report.demands_by_object)})")
    else:
        typer.secho(
            f"  · {len(report.unmet)} obligation(s) sans demande ni dispense :",
            fg=typer.colors.YELLOW,
        )
        for obligation in missing_demands(report):
            typer.echo(
                f"      {obligation.object_id:<24} "
                f"{obligation.target_kind.value}/{obligation.expected_target_ref}"
            )
        typer.secho(
            "    le plan ne couvrira pas ces objets ; déclarez une demande, ou "
            "une dispense motivée dans 01_sources/coverage_waivers.json",
            fg=typer.colors.YELLOW,
        )

    for demand_id in report.orphan_demands:
        typer.secho(
            f"  · besoin hors gabarit : {demand_id} — vérifiez sa cible",
            fg=typer.colors.YELLOW,
        )
    return report


def _measure_volumes(candidates, requested: bool):  # noqa: ANN001, ANN201
    """Mesure les tailles, si l'opérateur le demande.

    Ce n'est pas gratuit — une requête par candidat — et surtout ce n'est pas
    anodin : interroger un service facturé à l'appel doit rester un geste
    explicite. Sans mesure, le volume reste inconnu et le consentement sera
    refusé, ce qui est la bonne réponse plutôt qu'un échec.
    """
    if not requested:
        typer.secho(
            "  · volumes non mesurés : le consentement exige un total exact, "
            "relancez avec --measure-volumes",
            fg=typer.colors.YELLOW,
        )
        return None, None

    from .volumes import measure

    report = measure(candidates)
    return report.measured, report


def _latest_candidates(workspace) -> Path | None:  # noqa: ANN001
    """Dernier manifeste **de production**.

    `glob` ne descend pas dans les sous-dossiers : les rejeux sur cache figé,
    écrits sous `replays/`, ne peuvent pas devenir le manifeste courant. Un
    rejeu ramassé par le plan ferait acquérir sur un corpus qu'aucune
    interrogation récente n'a produit.
    """
    found = sorted(workspace.path("01_sources").glob("candidates_*.json"))
    return found[-1] if found else None


def _plan_digests(workspace, context, candidates_payload, demands_payload) -> dict:  # noqa: ANN001
    """Toutes les empreintes qu'un plan exécutable doit porter.

    Une absente n'est pas comblée : le plan restera brouillon, et le dira.
    """
    from .provenance import digest_of

    site = _safe_read(workspace.read_site)
    spatial = _safe_read(workspace.read_spatial)
    assets = _safe_read(workspace.read_assets)
    geometry = workspace.read_json("06_geo/capture_geometry.json") or {}

    roads = [g for g in geometry.get("geometries", []) if g.get("role") == "road_candidate"]
    obstacles = [
        g for g in geometry.get("geometries", []) if g.get("role") == "obstacle_building"
    ]

    return {
        "candidate_manifest_digest": digest_of(candidates_payload),
        "demand_digest": digest_of(demands_payload),
        "policy_digest": context.provenance["policy_digest"],
        "site_manifest_digest": digest_of(json.loads(site.model_dump_json())) if site else None,
        "spatial_manifest_digest": (
            digest_of(json.loads(spatial.model_dump_json())) if spatial else None
        ),
        "corpus_digest": digest_of(json.loads(assets.model_dump_json())) if assets else None,
        "road_geometry_digest": digest_of(roads) if roads else None,
        "obstacle_geometry_digest": digest_of(obstacles) if obstacles else None,
    }


def _safe_read(reader):  # noqa: ANN001, ANN201
    try:
        return reader()
    except (FileNotFoundError, OSError, ValueError):
        return None


@assets_app.command("acquire")
def assets_acquire(
    hotel_id: str = typer.Argument(...),
    plan_file: Path | None = typer.Option(
        None, "--plan", help="Plan à exécuter ; défaut : le plus récent."
    ),
) -> None:
    """Exécute un plan **exécutable**. Seul point du pipeline qui télécharge.

    Un brouillon est refusé, un plan périmé aussi : dans les deux cas, les
    images auraient été choisies pour un état que le consentement n'a pas vu.

    Les fichiers acquis sont `public_uncleared` : l'acquisition constate d'où
    ils viennent, elle ne tranche pas leurs droits. L'autorisation est une
    décision distincte — `assets rights clear` — et l'acceptation du risque en
    est une autre, qui ne les améliore pas.
    """
    from .acquire import AcquisitionRefused, run
    from .acquisition import merge, run_directory, verify_acquired
    from .provenance import digest_of
    from .schemas import AssetManifest
    from .schemas.acquisition import AcquisitionPlan, CandidateManifest

    context = _context(hotel_id, Capability.TARGETED_COLLECTION)
    workspace = Workspace(hotel_id)

    path = plan_file or _latest_plan(workspace)
    if path is None:
        typer.secho(
            f"{KO} aucun plan en circulation — lancez : assets plan",
            fg=typer.colors.RED, err=True,
        )
        raise typer.Exit(code=1)

    # Avant toute création de répertoire, lecture de cache ou appel réseau.
    _refuse_invalidated(workspace, path)

    plan_payload = json.loads(Path(path).read_text("utf-8"))
    plan = AcquisitionPlan.model_validate(plan_payload)

    candidates_path = _latest_candidates(workspace)
    if candidates_path is None:
        typer.secho(f"{KO} aucun manifeste de candidats", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    candidates_payload = json.loads(candidates_path.read_text("utf-8"))
    candidates = {
        c.candidate_id: c
        for c in CandidateManifest.model_validate(candidates_payload).candidates
    }

    demands_payload = workspace.read_json("01_sources/capture_demands.json") or {}
    digests = _plan_digests(workspace, context, candidates_payload, demands_payload)

    typer.echo(f"  plan        {plan.plan_id} ({plan.status.value})")
    typer.echo(f"  consenti    {plan.known_bytes} octets, {len(plan.acquisitions)} vue(s)")

    run_id = _new_run_id()
    destination = run_directory(workspace, run_id)

    try:
        acquired, report = run(
            plan, candidates, destination, digests,
            plan_digest=digest_of(plan_payload), run_id=run_id,
            policy=context.policy,
        )
    except AcquisitionRefused as exc:
        typer.secho(f"{KO} {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc

    problems = verify_acquired(acquired, workspace.root)
    if problems:
        for problem in problems:
            typer.secho(f"{KO} {problem}", fg=typer.colors.RED, err=True)
        typer.secho(
            "  aucun asset n'est inscrit au manifeste : un fichier dont "
            "l'empreinte ne correspond pas n'est pas celui qu'on a mesuré",
            fg=typer.colors.RED, err=True,
        )
        raise typer.Exit(code=3)

    manifest = workspace.read_assets() or AssetManifest(hotel_id=hotel_id, assets=[])
    merged = merge(manifest.assets, acquired)
    manifest.assets = merged.assets
    workspace.write_assets(manifest)
    workspace.write_report(f"01_sources/acquisition_{run_id}.json", report, context, production="AcquiredImage")

    for candidate_id, reason in sorted(report.failed.items()):
        typer.secho(f"    échec · {candidate_id} — {reason[:60]}", fg=typer.colors.YELLOW)

    typer.echo("")
    typer.echo(f"{OK} {report.acquired}/{report.planned} fichier(s) acquis")
    typer.echo(f"    {report.bytes_downloaded} octets téléchargés "
               f"sur {report.bytes_consented} consentis")
    typer.echo(f"    répertoire : {destination}")
    typer.echo("    droits     : public_uncleared — aucun droit n'est établi ici")
    typer.echo("    prochaine étape : OCR, puis assets rights clear ou assume-risk")


def _latest_plan(workspace) -> Path | None:  # noqa: ANN001
    """Dernier plan **encore en circulation**.

    Un plan invalidé n'est pas supprimé — ce qu'il disait reste lisible — mais
    il ne doit plus être choisi. Sans ce filtre, invalider le dernier ferait
    simplement remonter l'avant-dernier, qui porte le même défaut.

    Si tous sont invalidés, `None` : il n'y a pas de repli historique. Se
    rabattre sur un plan retiré ferait exécuter ce qu'on venait d'écarter.
    """
    from .plan_invalidation import invalidated_plan_ids

    sources = workspace.path("01_sources")
    retired = invalidated_plan_ids(sources)
    found = [
        path for path in sorted(sources.glob("acquisition_plan_*.json"))
        if path.name[len("acquisition_plan_"):-len(".json")] not in retired
    ]
    return found[-1] if found else None


def _refuse_invalidated(workspace, path: Path) -> None:  # noqa: ANN001
    """Refuse un plan retiré, y compris désigné explicitement.

    `--plan` court-circuite la sélection : sans ce contrôle, il suffisait de
    nommer le fichier pour exécuter ce qu'une invalidation avait écarté.
    """
    from .plan_invalidation import invalidated_plan_ids

    plan_id = Path(path).name[len("acquisition_plan_"):-len(".json")]
    if plan_id not in invalidated_plan_ids(workspace.path("01_sources")):
        return

    typer.secho(
        f"{KO} plan {plan_id} invalidé : il a été retiré de la circulation. "
        "Le fichier reste lisible, mais il ne s'exécute pas.",
        fg=typer.colors.RED, err=True,
    )
    raise typer.Exit(code=2)


def _new_run_id() -> str:
    from .acquisition import new_run_id

    return new_run_id()


@assets_app.command("ocr")
def assets_ocr(
    hotel_id: str = typer.Argument(...),
    run_id: str | None = typer.Option(
        None, "--run", help="N'lire que les fichiers d'une exécution d'acquisition."
    ),
) -> None:
    """Lit les enseignes des fichiers **acquis**, et trace chaque lecture.

    L'OCR vient après l'acquisition : à la découverte, aucune image n'existe.
    Les langues viennent du profil ; les supposer reviendrait à lire
    l'établissement comme s'il était ailleurs.
    """
    from .ocr import OcrRefused, run as run_ocr
    from .triage.sign_ocr import get_reader

    context = _context(hotel_id, Capability.IDENTITY_CLASSIFICATION)
    workspace = Workspace(hotel_id)

    manifest = workspace.read_assets()
    if manifest is None:
        typer.secho(f"{KO} aucun manifeste d'assets", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    languages = context.ocr_languages()
    if not languages:
        typer.secho(
            f"{KO} aucune langue d'OCR déclarée au profil de {hotel_id} — "
            f"elles ne se déduisent ni du pays ni du fuseau",
            fg=typer.colors.RED, err=True,
        )
        raise typer.Exit(code=1)

    targets = [
        asset for asset in manifest.assets
        if asset.local_path
        and (run_id is None or (asset.acquisition and asset.acquisition.run_id == run_id))
    ]
    if not targets:
        typer.secho(
            f"{KO} aucun fichier acquis à lire — lancez : assets acquire",
            fg=typer.colors.RED, err=True,
        )
        raise typer.Exit(code=1)

    typer.echo(f"  fichiers    {len(targets)}")
    typer.echo(f"  langues     {', '.join(languages)}")

    try:
        reader = get_reader()
    except (ImportError, RuntimeError) as exc:
        typer.secho(f"{KO} aucun moteur d'OCR disponible : {exc}",
                    fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    engine, version = _ocr_engine(reader)
    typer.echo(f"  moteur      {engine} {version}")

    try:
        updated, readings, report = run_ocr(
            targets, reader, languages,
            expected=context.identity_terms(), excluded=context.excluded_terms(),
            engine=engine, engine_version=version, workspace_root=workspace.root,
        )
    except OcrRefused as exc:
        typer.secho(f"{KO} {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    by_id = {asset.id: asset for asset in updated}
    manifest.assets = [by_id.get(asset.id, asset) for asset in manifest.assets]
    workspace.write_assets(manifest)

    workspace.write_json(
        f"01_sources/ocr_readings_{report.run_id}.json",
        {"run_id": report.run_id, "readings": [r.as_dict() for r in readings]},
    )
    workspace.write_report(f"01_sources/ocr_report_{report.run_id}.json", report, context)

    typer.echo("")
    for status, count in sorted(report.by_status.items()):
        typer.echo(f"    {status:<12} {count:>5}")
    for term, count in sorted(report.matched_terms.items()):
        typer.echo(f"    terme lu · {term[:40]:<40} {count:>4}")
    for asset_id, reason in sorted(report.skipped.items())[:5]:
        typer.secho(f"    ignoré · {asset_id} — {reason[:52]}", fg=typer.colors.YELLOW)

    typer.echo("")
    typer.echo(f"{OK} {report.read} lecture(s), {len(report.skipped)} ignorée(s)")
    typer.echo("    une enseigne établit une appartenance, jamais une visibilité")


def _ocr_engine(reader) -> tuple[str, str]:  # noqa: ANN001
    """Nom et version du moteur réellement employé.

    Inscrits à chaque lecture : deux versions d'EasyOCR ne lisent pas la même
    chose, et une lecture qu'on ne peut pas rattacher à son moteur ne se rejoue
    pas.
    """
    name = type(reader).__name__
    if name == "LocalReader":
        try:
            import easyocr

            return "easyocr", getattr(easyocr, "__version__", "inconnue")
        except ImportError:  # pragma: no cover — le lecteur existe donc easyocr aussi
            return "easyocr", "inconnue"
    return name.lower(), "inconnue"


rights_app = typer.Typer(
    no_args_is_help=True,
    help="Décisions de droits — distinctes de l'acquisition, qui n'en prend aucune.",
)
assets_app.add_typer(rights_app, name="rights")


@rights_app.command("clear")
def rights_clear(
    hotel_id: str = typer.Argument(...),
    asset_id: str = typer.Argument(...),
    granted: str = typer.Option(..., "--granted", help="licensed, open_data ou owned."),
    scope: str = typer.Option(..., "--scope", help="Ce que l'autorisation couvre."),
    by: str = typer.Option(..., "--by", help="Qui décide."),
    rationale: str = typer.Option(..., "--rationale", help="Sur quoi elle repose."),
    evidence: list[str] = typer.Option(
        [], "--evidence", help="Preuve de l'autorisation ; répétable, obligatoire."
    ),
) -> None:
    """Enregistre une autorisation **prouvée**.

    Une autorisation sans preuve est une affirmation : `--evidence` est
    obligatoire. La portée aussi — « usage interne » et « diffusion publique »
    ne sont pas la même permission.
    """
    from .schemas.rights import RightsAction, RightsDecision

    _record_rights_decision(
        hotel_id, asset_id,
        action=RightsAction.CLEAR, granted=granted, scope=scope, by=by,
        rationale=rationale, evidence=list(evidence),
    )


@rights_app.command("assume-risk")
def rights_assume_risk(
    hotel_id: str = typer.Argument(...),
    asset_id: str = typer.Argument(...),
    scope: str = typer.Option(..., "--scope", help="Ce que l'usage envisagé couvre."),
    by: str = typer.Option(..., "--by", help="Qui accepte le risque."),
    rationale: str = typer.Option(..., "--rationale", help="Pourquoi on avance."),
    evidence: list[str] = typer.Option(
        [], "--evidence", help="Ce qui a été examiné avant d'accepter ; répétable."
    ),
) -> None:
    """Accepte le risque **sans** améliorer les droits.

    L'état juridique reste `public_uncleared` : accepter un risque n'accorde
    rien. Falsifier l'état pour se donner le droit de continuer rendrait le
    manifeste inutilisable comme preuve de diligence.
    """
    from .schemas.rights import RightsAction

    _record_rights_decision(
        hotel_id, asset_id,
        action=RightsAction.ASSUME_RISK, granted=None, scope=scope, by=by,
        rationale=rationale, evidence=list(evidence),
    )


def _record_rights_decision(  # noqa: ANN001
    hotel_id, asset_id, *, action, granted, scope, by, rationale, evidence
) -> None:
    """Contrôles, puis inscription append-only. Rien n'est écrasé."""
    from .rights import apply
    from .schemas import Rights
    from .schemas.rights import RightsAction, RightsDecision

    context = _context(hotel_id, Capability.INSPECTION)
    workspace = Workspace(hotel_id)

    manifest = workspace.read_assets()
    if manifest is None:
        typer.secho(f"{KO} aucun manifeste d'assets", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    index = next(
        (i for i, asset in enumerate(manifest.assets) if asset.id == asset_id), None
    )
    if index is None:
        typer.secho(f"{KO} asset {asset_id!r} inconnu", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    asset = manifest.assets[index]
    try:
        decision = RightsDecision(
            action=action,
            granted_rights=Rights(granted) if granted else None,
            decided_by=by, rationale=rationale, scope=scope,
            evidence=evidence, reviewed_checksum=asset.checksum,
            supersedes_index=len(asset.rights_history) - 1
            if asset.rights_history else None,
        )
        manifest.assets[index] = apply(asset, decision)
    except ValueError as exc:
        typer.secho(f"{KO} {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc

    workspace.write_assets(manifest)
    updated = manifest.assets[index]

    typer.echo(f"{OK} {asset_id} — {action.value}")
    typer.echo(f"    droits     {updated.rights.value}")
    typer.echo(f"    grevés     {updated.rights_encumbered}")
    typer.echo(f"    portée     {scope}")
    typer.echo(f"    décisions  {len(updated.rights_history)} au total")
    if action is RightsAction.ASSUME_RISK:
        typer.secho(
            "    l'état juridique est inchangé : accepter un risque n'accorde rien",
            fg=typer.colors.YELLOW,
        )


demands_app = typer.Typer(
    no_args_is_help=True,
    help="Besoins de capture — instanciés depuis les obligations du gabarit.",
)
assets_app.add_typer(demands_app, name="demands")


@demands_app.command("build")
def demands_build(
    hotel_id: str = typer.Argument(...),
    force: bool = typer.Option(False, "--force", help="Réécrit la copie canonique."),
) -> None:
    """Instancie les besoins depuis les obligations. **Aucun appel réseau.**

    Traduit ce que le gabarit exige en besoins que le reste du pipeline sait
    juger. Les demandes écrites par l'opérateur sont préservées : le générateur
    n'est pas propriétaire du manifeste, il y ajoute ce qui est dû.

    Un objet non résolu produit un besoin **non ciblable**, jamais une
    dispense : le déclarer sans objet ferait disparaître un manque.
    """
    from .demands_build import DemandsRefused, build, validate_targets
    from .provenance import digest_of
    from .schemas.acquisition import CaptureDemandManifest

    context = _context(hotel_id, Capability.INSPECTION)
    workspace = Workspace(hotel_id)

    site = _safe_read(workspace.read_site)
    if site is None:
        typer.secho(
            f"{KO} aucun manifeste de site — lancez : site build",
            fg=typer.colors.RED, err=True,
        )
        raise typer.Exit(code=1)

    spatial = _safe_read(workspace.read_spatial)
    existing_payload = workspace.read_json("01_sources/capture_demands.json")
    existing = (
        CaptureDemandManifest.model_validate(existing_payload)
        if existing_payload else None
    )
    waivers = _read_waivers(workspace)

    site_payload = json.loads(site.model_dump_json())
    digests = {
        "site_manifest_digest": digest_of(site_payload),
        "spatial_manifest_digest": (
            digest_of(json.loads(spatial.model_dump_json())) if spatial else None
        ),
        "policy_digest": context.provenance["policy_digest"],
    }

    try:
        manifest, report = build(
            hotel_id, site, context.policy.coverage, existing, waivers, digests
        )
    except DemandsRefused as exc:
        typer.secho(f"{KO} {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    geometry = _capture_geometry_if_any(workspace, context)
    problems = validate_targets(manifest, site, geometry)
    if problems:
        for problem in problems:
            typer.secho(f"{KO} {problem}", fg=typer.colors.RED, err=True)
        typer.secho(
            "  aucun besoin n'est écrit : une demande qui ne vise rien resterait "
            "ouverte indéfiniment, et compterait comme un manque réel",
            fg=typer.colors.RED, err=True,
        )
        raise typer.Exit(code=2)

    payload = json.loads(manifest.model_dump_json())
    digest = digest_of(payload)

    workspace.write_json(f"01_sources/capture_demands_{digest}.json", payload)
    workspace.write_report(
        f"01_sources/capture_demands_build_{digest}.json", report, context,
        production="CaptureDemandManifest",
    )

    canonical = workspace.path("01_sources", "capture_demands.json")
    if canonical.is_file() and not force:
        typer.secho(
            f"  · copie canonique conservée — {canonical.name} existe déjà ; "
            f"--force pour l'activer",
            fg=typer.colors.YELLOW,
        )
    else:
        workspace.write_json("01_sources/capture_demands.json", payload)

    typer.echo(f"  générés     {len(report.generated_from_obligation)}")
    typer.echo(f"  conservés   {len(report.operator_defined)} (opérateur)")
    for object_id, reason in sorted(report.unresolved_target.items()):
        typer.secho(f"    non ciblable · {object_id} — {reason[:52]}",
                    fg=typer.colors.YELLOW)
    for object_id in sorted({**report.waived, **report.not_applicable}):
        typer.echo(f"    dispensé · {object_id}")

    typer.echo("")
    typer.echo(f"{OK} {len(manifest.demands)} besoin(s) — empreinte {digest}")
    typer.echo("    prochaine étape : assets demands assess")


@demands_app.command("assess")
def demands_assess(hotel_id: str = typer.Argument(...)) -> None:
    """Évalue les besoins sur le corpus **existant**. Aucune collecte.

    Répond, besoin par besoin : combien de points de vue indépendants le
    servent, lesquels, et ce qui manque. C'est ce rapport qui dira ensuite à la
    recherche adaptative quels secteurs sont déficitaires — sans lui, le
    collecteur devrait redéfinir les objectifs de couverture, et deux sources
    d'autorité finiraient par diverger.
    """
    from .demands_assess import assess
    from .plan import group_viewpoints
    from .provenance import digest_of
    from .schemas.acquisition import CaptureDemandManifest

    context = _context(hotel_id, Capability.INSPECTION)
    workspace = Workspace(hotel_id)

    payload = workspace.read_json("01_sources/capture_demands.json")
    if not payload:
        typer.secho(
            f"{KO} aucun besoin — lancez : assets demands build",
            fg=typer.colors.RED, err=True,
        )
        raise typer.Exit(code=1)

    manifest = workspace.read_assets()
    if manifest is None:
        typer.secho(f"{KO} aucun manifeste d'assets", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    demands = CaptureDemandManifest.model_validate(payload)
    corpus = json.loads(manifest.model_dump_json())

    # Les points de vue viennent des positions, jamais du nombre de fichiers :
    # neuf fichiers à six positions font six observations, pas neuf.
    viewpoints = group_viewpoints(
        [_as_viewpoint_subject(asset) for asset in manifest.assets],
        separation_m=context.policy.geometry.viewpoint_separation_m,
    )

    assessment, report = assess(
        hotel_id, demands.demands, manifest.assets,
        corpus_digest=digest_of(corpus),
        viewpoints=viewpoints,
        demand_digest=digest_of(payload),
    )

    workspace.write_json(
        f"01_sources/demand_assessment_{report.corpus_digest}.json",
        json.loads(assessment.model_dump_json()),
    )
    workspace.write_report(
        f"01_sources/demand_assessment_report_{report.corpus_digest}.json",
        report, context, production="DemandAssessmentManifest",
    )

    typer.echo(f"  assets      {report.assets_considered}")
    for status, identifiers in sorted(report.by_status.items()):
        typer.echo(f"    {status:<16} {len(identifiers):>3}")
    for demand_id, viewpoint_ids in sorted(report.viewpoints_by_demand.items()):
        if viewpoint_ids:
            typer.echo(f"    {demand_id} · {len(viewpoint_ids)} point(s) de vue")

    typer.echo("")
    typer.echo(f"{OK} {len(assessment.assessments)} besoin(s) évalué(s)")
    if report.open_demands:
        typer.echo(f"    encore ouverts : {', '.join(report.open_demands)}")
    typer.echo("    prochaine étape : assets discover (recherche ciblée)")


def _as_viewpoint_subject(asset):  # noqa: ANN001, ANN201
    """Adapte un asset au regroupement par point de vue.

    `group_viewpoints` attend des candidats ; un asset porte les mêmes faits
    sous d'autres noms. L'adapter ici évite de dupliquer la règle de
    regroupement, qui doit rester unique.
    """
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class Subject:
        candidate_id: str
        camera_lat: float | None
        camera_lon: float | None
        panorama_id: str | None

    return Subject(
        candidate_id=asset.id,
        camera_lat=asset.camera_lat,
        camera_lon=asset.camera_lon,
        panorama_id=(
            asset.acquisition.panorama_id if asset.acquisition else None
        ),
    )


def _capture_geometry_if_any(workspace, context):  # noqa: ANN001, ANN201
    """Géométrie de capture, si elle est résolue. `None` sinon — jamais vide.

    Un registre absent et un registre vide ne disent pas la même chose : le
    premier qu'on ne peut pas valider, le second que toute référence est fausse.
    """
    from .geo.geometry_loader import LegacyManifestRefused, load_capture_geometry

    path = workspace.path("06_geo", "capture_geometry.json")
    if context.spatial_reference is None or not path.is_file():
        return None
    try:
        manifest, _ = load_capture_geometry(path, context.spatial_reference)
    except (LegacyManifestRefused, ValueError):
        return None
    return manifest


def _read_waivers(workspace) -> list:  # noqa: ANN001
    from .coverage_obligations import ObligationWaiver

    payload = workspace.read_json("01_sources/coverage_waivers.json") or {}
    return [ObligationWaiver.model_validate(row) for row in payload.get("waivers", [])]


@assets_app.command("dedup")
def assets_dedup(hotel_id: str = typer.Argument(...)) -> None:
    """Déduplication à quatre niveaux (Lot 1B §5)."""
    from . import dedup_levels

    workspace = Workspace(hotel_id)
    manifest = workspace.read_assets()
    spatial = workspace.read_spatial()
    if manifest is None or spatial is None or not spatial.confirmed_building_id:
        typer.secho(
            "manifeste d'assets et bâtiment confirmé requis", fg=typer.colors.RED, err=True
        )
        raise typer.Exit(code=1)

    context = _context(hotel_id, Capability.INSPECTION)
    building = spatial.candidate(spatial.confirmed_building_id)
    report = dedup_levels.run(
        manifest.assets, building.centroid_lat, building.centroid_lon, policy=context.policy
    )

    # L'arbitrage de grappe est un prédicat de rôle : le changer sans
    # réaffecter laissait le manifeste porter des rôles que `role_for` ne
    # produirait plus, et un rapport de rôles décrivant l'état d'avant.
    from .roles import assign

    roles = assign(manifest.assets, context.policy)
    workspace.write_assets(manifest)
    workspace.write_report("01_sources/duplicate_report.json", report, context, production="DuplicateReport")
    workspace.write_report("01_sources/roles_report.json", roles, context)

    typer.echo(f"  fichiers                  {report.files}")
    typer.echo(f"  photographies uniques     {report.perceptual_groups}")
    typer.echo(f"  points de vue indépendants {report.viewpoints}")
    typer.echo(
        f"  rôles : {report.canonical} canonique(s), "
        f"{report.overlap} recouvrement, {report.inactive} inactif(s)"
    )
    typer.echo(f"  rôles : {roles.counts}")
    typer.echo("")
    for family, counts in sorted(report.by_source_family.items()):
        typer.echo(
            f"  {family:<22} {counts['files']:>4} fichiers  "
            f"{counts['photographs']:>4} photos  {counts['viewpoints']:>4} points de vue"
        )


@assets_app.command("classify")
def assets_classify(
    hotel_id: str = typer.Argument(...),
    use_model: bool = typer.Option(True, "--model/--no-model", help="Étape 4 OpenCLIP."),
) -> None:
    """Cascade de catégorisation multidimensionnelle (Lot 1B §6)."""
    from .classify_cascade import classify
    from .sectors import resolve_front
    from .steps import ELEMENTS_FILE

    # La capacité se vérifie avant toute lecture : un projet sans profil doit
    # apprendre qu'il lui manque une identité, pas qu'il lui manque un fichier.
    context = _context(hotel_id, Capability.IDENTITY_CLASSIFICATION)

    workspace = Workspace(hotel_id)
    manifest = workspace.read_assets()
    spatial = workspace.read_spatial()
    if manifest is None or spatial is None:
        typer.secho("manifeste d'assets et manifeste spatial requis", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    elements = workspace.read_json(ELEMENTS_FILE) or []
    front = resolve_front(spatial, elements)
    if front is None:
        typer.secho(
            "  · aucune preuve d'orientation de façade — secteurs laissés inconnus",
            fg=typer.colors.YELLOW,
        )
    else:
        typer.echo(f"  façade avant : {front.degrees:.0f}° ({front.method})")
        spatial.front_azimuth_deg = front.degrees
        spatial.front_azimuth_method = front.method
        workspace.write_spatial(spatial)

    classifier = None
    if use_model:
        try:
            from .triage.classify import Classifier

            classifier = Classifier(policy=context.policy)
        except ImportError:
            typer.secho("  · OpenCLIP absent — étape 4 ignorée", fg=typer.colors.YELLOW)

    report = classify(
        manifest.assets,
        classifier=classifier,
        front_azimuth=front.degrees if front else None,
        policy=context.policy,
    )
    workspace.write_assets(manifest)
    workspace.write_report("01_sources/classification_report.json", report, context, production="ClassificationReport")

    typer.echo("")
    typer.echo(f"  {report.total} asset(s), {report.needs_review} en revue")
    typer.echo(f"  sujets  : {report.subjects_assigned}")
    typer.echo(f"  secteurs: {report.sectors_assigned}")


review_app = typer.Typer(no_args_is_help=True, help="Revue humaine de visibilité (Lot 1B §6).")
assets_app.add_typer(review_app, name="review")


@review_app.command("queue")
def review_queue(
    hotel_id: str = typer.Argument(...),
    queue: str = typer.Option("blocking", "--queue", help="blocking, pending ou mapillary-candidates."),
    mode: str = typer.Option(
        "analysis", "--mode",
        help="analysis (diagnostic complet) ou blind (étiquetage de vérité terrain).",
    ),
    pending_only: bool = typer.Option(
        False, "--pending-only", help="N'inclure que les vues non encore examinées."
    ),
    reference_id: str = typer.Option(
        None, "--reference",
        help="Asset servant de référence visuelle sur la planche aveugle.",
    ),
) -> None:
    """Produit une file versionnée et une planche HTML à examiner.

    Deux modes, pour deux usages incompatibles : la planche d'analyse montre ce
    que le système conclut, la planche aveugle ne montre que l'image. Étiqueter
    depuis la première produirait des étiquettes qui héritent du verdict
    qu'elles doivent juger.
    """
    from datetime import datetime, timezone

    from . import review as review_module

    workspace = Workspace(hotel_id)
    manifest = workspace.read_assets()
    if manifest is None:
        typer.secho("aucun manifeste d'assets", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    context = _context(hotel_id, Capability.INSPECTION)

    # Les mesures de la dernière exécution appliquée accompagnent la file : le
    # réviseur doit voir ce que la géométrie établit, et ce qu'elle n'établit
    # pas.
    measures: dict[str, dict] = {}
    applied = {a.visibility_run_id for a in manifest.assets if a.visibility_run_id}
    for identifier in sorted(applied):
        raw = workspace.read_json(f"06_geo/visibility_run_{identifier}.json") or {}
        for assessment in raw.get("assessments", []):
            measures[assessment["subject_ref"]] = assessment

    if mode not in ("analysis", "blind"):
        typer.secho(f"{KO} mode inconnu : {mode!r}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    try:
        built = review_module.build_queue(
            manifest.assets, queue, context.policy, visibility=measures
        )
    except review_module.ReviewRefused as exc:
        typer.secho(f"{KO} {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    if pending_only:
        built.items = [i for i in built.items if not i.reviews]

    protocol = None
    if mode == "blind":
        from . import cohort as cohort_module

        by_id = {a.id: a for a in manifest.assets}
        selected = [by_id[i.asset_id] for i in built.items if i.asset_id in by_id]

        # L'ordre dérive de l'empreinte de **cohorte**, celle que la planche
        # annonce — non de celle du manifeste, qui change à chaque décision.
        cohort_members = cohort_module.members(manifest.assets)
        cohort_key = cohort_module.cohort_digest(cohort_members)
        order = [a.id for a in cohort_module.blind_order(selected, cohort_key)]
        built.items.sort(key=lambda item: order.index(item.asset_id))
        built.manifest_digest = cohort_key

        reference = _blind_reference(manifest, reference_id)
        protocol = cohort_module.build_protocol(
            selected, hotel_id, cohort_key=cohort_key, reference=reference,
            predictions_digest=_digest_of(
                workspace.path("01_sources", f"cohort_predictions_{cohort_key}.json")
            ),
            sequence_register_digest=_digest_of(
                workspace.path("01_sources", f"sequence_register_{cohort_key}.json")
            ),
        )
        protocol_path = workspace.path(
            "01_sources", f"review_protocol_{protocol.protocol_id}.json"
        )
        try:
            outcome = cohort_module.publish(protocol, protocol_path)
        except cohort_module.ProtocolConflict as exc:
            typer.secho(f"{KO} {exc}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=4) from exc

        built.protocol_id = protocol.protocol_id
        built.protocol_digest = _digest_of(protocol_path)
        typer.echo(f"  protocole {protocol.protocol_id} ({outcome})")

    # Le nom porte la date **et** l'empreinte du manifeste : une file décrit
    # un état, et deux exécutions dans la même seconde ne doivent pas s'écraser.
    json_path = workspace.write_report(
        f"01_sources/review_queue_{mode}_{built.slug}.json",
        built.as_dict(blind=mode == "blind"),
        context,
    )
    board = workspace.path("01_sources", f"review_board_{mode}_{built.slug}.html")

    if mode == "blind":
        board.write_text(
            review_module.to_blind_html(built, protocol.reference, protocol.protocol_id),
            encoding="utf-8",
        )
    else:
        board.write_text(review_module.to_html(built), encoding="utf-8")

    numbers = built.counts
    typer.echo("")
    typer.secho(f"  en attente  {numbers.pending:>4}", fg=typer.colors.YELLOW)
    typer.secho(f"  bloquants   {numbers.blocking:>4}", fg=typer.colors.YELLOW)
    typer.secho(f"  cohorte     {numbers.cohort:>4}", fg=typer.colors.YELLOW)
    # Dire à la fois « la revue est close » et « cette image demeure indécise ».
    typer.secho(
        f"  indécises   {numbers.reviewed_unresolved:>4}  examinées, revue close",
        fg=typer.colors.YELLOW,
    )
    typer.echo("  populations distinctes — ne pas additionner")
    typer.echo("")
    for source, total in numbers.pending_by_source.items():
        typer.echo(f"    en attente · {source:<14} {total:>4}")
    typer.echo("")
    typer.echo(f"  file « {queue} » : {len(built.items)} image(s)")
    typer.echo(f"    {json_path}")
    typer.echo(f"    {board}")


@review_app.command("set")
def review_set(
    hotel_id: str = typer.Argument(...),
    asset_id: str = typer.Argument(...),
    decision: str = typer.Option(..., "--decision", help="confirmed, rejected ou unresolved."),
    by: str = typer.Option(..., "--by", help="Auteur de la décision."),
    rationale: str = typer.Option(..., "--rationale", help="Motif, obligatoire."),
    evidence: list[str] = typer.Option(
        ..., "--evidence",
        help="Preuve(s) à l'appui, au moins une. Répétable.",
    ),
    blinding: str = typer.Option(
        "unblinded", "--blinding",
        help="blind si la décision a été prise sans voir la sortie du système.",
    ),
    protocol_id: str = typer.Option(
        None, "--protocol", help="Protocole d'étiquetage aveugle suivi."
    ),
) -> None:
    """Inscrit **une** décision humaine. Aucune acceptation en masse."""
    from . import review as review_module
    from .schemas import ReviewDecision

    workspace = Workspace(hotel_id)
    manifest = workspace.read_assets()
    if manifest is None:
        typer.secho("aucun manifeste d'assets", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    try:
        verdict = ReviewDecision(decision)
    except ValueError as exc:
        typer.secho(
            f"{KO} décision inconnue : {decision!r} ; attendu "
            f"{[d.value for d in ReviewDecision]}",
            fg=typer.colors.RED, err=True,
        )
        raise typer.Exit(code=1) from exc

    context = _context(hotel_id, Capability.IDENTITY_CLASSIFICATION)
    before_roles = review_module.recompute(manifest.assets, context.policy).counts
    before_counts = review_module.counts(manifest.assets, context.policy).as_dict()
    before_viewpoints = review_module.viewpoints_by_suitability(manifest.assets)

    try:
        protocol, protocol_digest = _load_protocol(workspace, protocol_id)
        before, after = review_module.decide(
            manifest.assets, asset_id, verdict, by, rationale, list(evidence),
            workspace_root=workspace.root, blinding=blinding,
            protocol=protocol, protocol_digest=protocol_digest,
        )
    except review_module.ReviewRefused as exc:
        # Rien n'a été écrit : le manifeste sur disque est intact.
        typer.secho(f"{KO} {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    from .roles import role_for

    role_before, reason_before = role_for(before, context.policy)
    after_roles = review_module.recompute(manifest.assets, context.policy).counts
    updated = next(a for a in manifest.assets if a.id == asset_id)
    role_after, reason_after = role_for(updated, context.policy)

    impact = review_module.Impact(
        asset_id=asset_id,
        decision=verdict.value,
        checksum=after.review_history[-1].reviewed_checksum,
        entry=json.loads(after.review_history[-1].model_dump_json()),
        role_before=role_before.value,
        role_after=role_after.value,
        reason_before=reason_before,
        reason_after=reason_after,
        cluster_before=before.cluster_role.value if before.cluster_role else "—",
        cluster_after=updated.cluster_role.value if updated.cluster_role else "—",
        roles_before=before_roles,
        roles_after=after_roles,
        counts_before=before_counts,
        counts_after=review_module.counts(manifest.assets, context.policy).as_dict(),
        viewpoints_before=before_viewpoints,
        viewpoints_after=review_module.viewpoints_by_suitability(manifest.assets),
    )

    workspace.write_assets(manifest)
    workspace.write_report(f"01_sources/review_decision_{impact.slug}.json", impact, context)

    typer.echo("")
    typer.secho(
        f"  {asset_id} · {verdict.value} · par {after.reviewer}", fg=typer.colors.GREEN
    )
    typer.echo(f"  revue n° {len(after.review_history)} — les précédentes sont conservées")
    typer.echo(f"  statut  : {before.review_status.value} → {after.review_status.value}")
    typer.echo(f"  rôle    : {role_before.value} ({reason_before})")
    typer.echo(f"            {role_after.value} ({reason_after})")
    typer.echo("")
    for role in sorted(set(before_roles) | set(after_roles)):
        was, now = before_roles.get(role, 0), after_roles.get(role, 0)
        marker = "  " if was == now else " ←"
        typer.echo(f"    {role:<20} {was:>4} → {now:>4}{marker}")


@review_app.command("geometry")
def review_geometry(
    hotel_id: str = typer.Argument(...),
    asset_id: str = typer.Argument(...),
    suitability: str = typer.Option(
        ..., "--suitability", help="primary, auxiliary ou insufficient."
    ),
    by: str = typer.Option(..., "--by", help="Auteur de l'appréciation."),
    rationale: str = typer.Option(..., "--rationale", help="Motif, obligatoire."),
    evidence: list[str] = typer.Option(..., "--evidence", help="Preuve(s), au moins une."),
    measure: list[str] = typer.Option(
        [], "--measure", help="Mesure à l'appui, forme clé=valeur. Répétable."
    ),
    blinding: str = typer.Option(
        "unblinded", "--blinding",
        help="blind si l'appréciation a été rendue sans voir la sortie du système.",
    ),
    protocol_id: str = typer.Option(
        None, "--protocol", help="Protocole d'étiquetage aveugle suivi."
    ),
) -> None:
    """Apprécie ce que l'image apporte à la **structure**, non son identité."""
    from . import review as review_module
    from .schemas import GeometrySuitability

    workspace = Workspace(hotel_id)
    manifest = workspace.read_assets()
    if manifest is None:
        typer.secho("aucun manifeste d'assets", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    try:
        verdict = GeometrySuitability(suitability)
    except ValueError as exc:
        typer.secho(
            f"{KO} aptitude inconnue : {suitability!r} ; attendu "
            f"{[s.value for s in GeometrySuitability]}",
            fg=typer.colors.RED, err=True,
        )
        raise typer.Exit(code=1) from exc

    if verdict is GeometrySuitability.UNASSESSED:
        typer.secho(
            f"{KO} « unassessed » est l'état initial, pas une appréciation",
            fg=typer.colors.RED, err=True,
        )
        raise typer.Exit(code=1)

    measurements: dict[str, float] = {}
    for item in measure:
        key, _, value = item.partition("=")
        try:
            measurements[key.strip()] = float(value)
        except ValueError as exc:
            typer.secho(f"{KO} mesure illisible : {item!r} (attendu clé=valeur)",
                        fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1) from exc

    context = _context(hotel_id, Capability.INSPECTION)
    before_roles = review_module.recompute(manifest.assets, context.policy).counts
    before_counts = review_module.counts(manifest.assets, context.policy).as_dict()
    before_viewpoints = review_module.viewpoints_by_suitability(manifest.assets)

    try:
        protocol, protocol_digest = _load_protocol(workspace, protocol_id)
        before, after = review_module.assess(
            manifest.assets, asset_id, verdict, by, rationale, list(evidence),
            measurements, workspace_root=workspace.root, blinding=blinding,
            protocol=protocol, protocol_digest=protocol_digest,
        )
    except review_module.ReviewRefused as exc:
        typer.secho(f"{KO} {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    from .roles import role_for

    role_before, reason_before = role_for(before, context.policy)
    after_roles = review_module.recompute(manifest.assets, context.policy).counts
    updated = next(a for a in manifest.assets if a.id == asset_id)
    role_after, reason_after = role_for(updated, context.policy)

    impact = review_module.Impact(
        asset_id=asset_id,
        decision=verdict.value,
        checksum=after.geometry_history[-1].reviewed_checksum,
        entry=json.loads(after.geometry_history[-1].model_dump_json()),
        role_before=role_before.value,
        role_after=role_after.value,
        reason_before=reason_before,
        reason_after=reason_after,
        cluster_before=before.cluster_role.value if before.cluster_role else "—",
        cluster_after=updated.cluster_role.value if updated.cluster_role else "—",
        roles_before=before_roles,
        roles_after=after_roles,
        counts_before=before_counts,
        counts_after=review_module.counts(manifest.assets, context.policy).as_dict(),
        viewpoints_before=before_viewpoints,
        viewpoints_after=review_module.viewpoints_by_suitability(manifest.assets),
    )

    workspace.write_assets(manifest)
    workspace.write_report(f"01_sources/geometry_decision_{impact.slug}.json", impact, context)

    typer.echo("")
    typer.secho(f"  {asset_id} · {verdict.value} · par {after.geometry_history[-1].decided_by}",
                fg=typer.colors.GREEN)
    typer.echo(f"  appréciation n° {len(after.geometry_history)}")
    typer.echo(f"  rôle    : {role_before.value} ({reason_before})")
    typer.echo(f"            {role_after.value} ({reason_after})")
    typer.echo("")
    for role in sorted(set(before_roles) | set(after_roles)):
        was, now = before_roles.get(role, 0), after_roles.get(role, 0)
        typer.echo(f"    {role:<20} {was:>4} → {now:>4}{'' if was == now else ' ←'}")
    typer.echo("")
    typer.echo("  points de vue indépendants, par aptitude :")
    for name, total in review_module.viewpoints_by_suitability(manifest.assets).items():
        typer.echo(f"    {name:<14} {total:>4}")


def _load_protocol(workspace, protocol_id: str | None):  # noqa: ANN001
    """Charge un protocole d'étiquetage, et vérifie qu'il dit ce qu'il est."""
    from . import cohort as cohort_module

    if not protocol_id:
        return None, None
    path = workspace.path("01_sources", f"review_protocol_{protocol_id}.json")
    if not path.is_file():
        raise typer.BadParameter(f"protocole introuvable : {path.name}")

    payload = json.loads(path.read_text("utf-8"))
    protocol = cohort_module.ReviewProtocol(
        protocol_id=payload["protocol_id"], hotel_id=payload["hotel_id"],
        cohort_digest=payload["cohort_digest"], blinding=payload["blinding"],
        created_at=payload.get("created_at", ""), members=payload.get("members", []),
        reference=payload.get("reference"),
        predictions_digest=payload.get("predictions_digest"),
        sequence_register_digest=payload.get("sequence_register_digest"),
        order=payload.get("presentation_order", []),
    )
    if not protocol.matches_its_id():
        raise typer.BadParameter(
            f"{path.name} ne correspond pas à son identifiant : contenu "
            f"{protocol.content_digest()}"
        )
    return protocol, _digest_of(path)


def _digest_of(path: Path) -> str | None:
    from .intake import sha256_file

    return sha256_file(path)[:16] if path.is_file() else None


def _blind_reference(manifest, reference_id: str | None) -> dict | None:  # noqa: ANN001
    """Référence visuelle de la planche aveugle, choisie **explicitement**.

    Prendre « le premier asset confirmé » laissait l'ordre du manifeste
    désigner une image, qui se trouvait appartenir à l'une des séquences
    évaluées. Le choix est donc demandé, et son appartenance à la cohorte
    inscrite : une aide intra-séquence doit se déclarer.
    """
    from . import cohort as cohort_module

    if not reference_id:
        return None
    asset = next((a for a in manifest.assets if a.id == reference_id), None)
    if asset is None:
        raise typer.BadParameter(f"référence inconnue : {reference_id}")
    if asset.target_visibility_decision.value != "confirmed":
        raise typer.BadParameter(
            f"{reference_id} n'est pas une vue confirmée : une référence non "
            "établie induirait le réviseur en erreur"
        )
    cohort_ids = {a.id for a in cohort_module.members(manifest.assets)}
    return {
        "asset_id": asset.id,
        "local_path": asset.local_path,
        "checksum": asset.checksum,
        "rationale": asset.review_rationale or "",
        "in_cohort": asset.id in cohort_ids,
    }


@review_app.command("cohort")
def review_cohort(
    hotel_id: str = typer.Argument(...),
    source: str = typer.Option("mapillary", "--source"),
) -> None:
    """Fige la cohorte : séquences réelles et prédictions avant étiquetage."""
    from datetime import datetime, timezone

    from . import cohort as cohort_module
    from .collectors.mapillary import image_metadata

    workspace = Workspace(hotel_id)
    manifest = workspace.read_assets()
    if manifest is None:
        typer.secho("aucun manifeste d'assets", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    context = _context(hotel_id, Capability.IDENTITY_CLASSIFICATION)
    members = cohort_module.members(manifest.assets, source)
    digest_value = cohort_module.cohort_digest(members)

    # Les séquences se demandent à la source : les assets historiques n'en
    # portent aucune, et une proximité géographique n'est pas une séquence.
    register = cohort_module.build_register(members, hotel_id, image_metadata)
    workspace.write_json(
        f"01_sources/sequence_register_{digest_value}.json", register.as_dict()
    )

    measures: dict[str, dict] = {}
    for identifier in sorted({a.visibility_run_id for a in manifest.assets if a.visibility_run_id}):
        raw = workspace.read_json(f"06_geo/visibility_run_{identifier}.json") or {}
        for assessment in raw.get("assessments", []):
            measures[assessment["subject_ref"]] = assessment

    snapshot = cohort_module.predictions(members, context.policy, measures)
    snapshot["sequence_correlation"] = register.correlation
    workspace.write_report(
        f"01_sources/cohort_predictions_{digest_value}.json", snapshot, context
    )

    typer.echo("")
    typer.echo(f"  cohorte {source} : {len(members)} membre(s) · empreinte {digest_value}")
    typer.echo(f"  corrélation de séquence : {register.correlation}")
    for sequence, ids in register.by_sequence().items():
        typer.echo(f"    {sequence:<26} {len(ids):>3} image(s)")
    reviewed = [a for a in members if a.review_history]
    typer.echo("")
    typer.echo(f"  déjà examinées : {len(reviewed)} · restantes : {len(members) - len(reviewed)}")
    typer.secho(
        "  portée : précision parmi les candidats détectés ; le rappel n'est pas "
        "mesurable — les faux négatifs sont exclus par construction",
        fg=typer.colors.YELLOW,
    )


@review_app.command("export")
def review_export(
    hotel_id: str = typer.Argument(...),
    root: Path = typer.Option(None, "--root", help="Racine du registre (défaut : decisions/)."),
) -> None:
    """Extrait les décisions humaines dans un registre versionnable."""
    from . import decisions as register_module

    workspace = Workspace(hotel_id)
    manifest = workspace.read_assets()
    if manifest is None:
        typer.secho("aucun manifeste d'assets", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    register = register_module.export(manifest.assets, hotel_id)
    target = register_module.path_for(hotel_id, root)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(register.as_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    visibility = sum(len(d.review_history) for d in register.decisions)
    aptitude = sum(len(d.geometry_history) for d in register.decisions)
    typer.echo("")
    typer.echo(f"  {len(register.decisions)} asset(s) portant une décision")
    typer.echo(f"    visibilité  {visibility:>4} entrée(s)")
    typer.echo(f"    aptitude    {aptitude:>4} entrée(s)")
    typer.echo(f"  {target}")
    typer.secho(
        "  les images ne sont pas versionnées ; leurs empreintes le sont",
        fg=typer.colors.YELLOW,
    )


@review_app.command("import")
def review_import(
    hotel_id: str = typer.Argument(...),
    root: Path = typer.Option(None, "--root", help="Racine du registre (défaut : decisions/)."),
) -> None:
    """Réapplique un registre au manifeste, empreintes vérifiées."""
    from . import decisions as register_module
    from . import review as review_module

    workspace = Workspace(hotel_id)
    manifest = workspace.read_assets()
    if manifest is None:
        typer.secho("aucun manifeste d'assets", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    source = register_module.path_for(hotel_id, root)
    if not source.is_file():
        typer.secho(f"{KO} registre introuvable : {source}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    from datetime import datetime, timezone

    from .intake import sha256_file

    context = _context(hotel_id, Capability.IDENTITY_CLASSIFICATION)
    before_roles = review_module.recompute(manifest.assets, context.policy).counts
    before_counts = review_module.counts(manifest.assets, context.policy).as_dict()
    before_viewpoints = review_module.viewpoints_by_suitability(manifest.assets)

    raw = source.read_text("utf-8")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        typer.secho(f"{KO} registre illisible : {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=4) from exc

    try:
        result = register_module.apply(manifest.assets, payload, hotel_id=hotel_id)
    except register_module.RegisterRefused as exc:
        typer.secho(f"{KO} registre refusé — rien n'a été modifié :", fg=typer.colors.RED, err=True)
        typer.secho(f"    {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=4) from exc

    # L'accord registre/manifeste ne confronte que deux déclarations : le
    # fichier sur disque, lui, a pu changer depuis la revue.
    altered = register_module.verify_files(manifest.assets, workspace.root)
    if altered:
        typer.secho(f"{KO} images jugées modifiées depuis leur revue :",
                    fg=typer.colors.RED, err=True)
        for problem in altered:
            typer.secho(f"    {problem}", fg=typer.colors.RED, err=True)
        typer.secho("  manifeste laissé intact", fg=typer.colors.YELLOW, err=True)
        raise typer.Exit(code=4)

    role_report = review_module.recompute(manifest.assets, context.policy)
    roles = role_report.counts
    workspace.write_assets(manifest)

    stamp = datetime.now(timezone.utc).isoformat().replace(":", "").replace("-", "").replace(".", "")
    digest = sha256_file(source)[:16]
    report = {
        "register": str(source),
        "register_digest": digest,
        "applied": result["applied"],
        "unknown": result["unknown"],
        "roles": {"before": before_roles, "after": roles},
        "review_counts": {
            "before": before_counts,
            "after": review_module.counts(manifest.assets, context.policy).as_dict(),
        },
        "viewpoints_by_suitability": {
            "before": before_viewpoints,
            "after": review_module.viewpoints_by_suitability(manifest.assets),
        },
    }
    workspace.write_report(f"01_sources/review_import_{stamp}_{digest}.json", report, context)
    workspace.write_report("01_sources/roles_report.json", role_report, context)

    typer.echo("")
    typer.echo(f"  {result['applied']} asset(s) mis à jour depuis {source}")
    typer.echo(f"  empreinte du registre : {digest}")
    typer.echo(f"  rôles : {roles}")
    typer.echo(
        f"  points de vue : {review_module.viewpoints_by_suitability(manifest.assets)}"
    )


geo_app = typer.Typer(no_args_is_help=True, help="Sources géospatiales (Lot 1B §9).")
app.add_typer(geo_app, name="geo")


@geo_app.command("route")
def geo_route(hotel_id: str = typer.Argument(...)) -> None:
    """Sources territorialement admissibles — pas encore leur couverture."""
    from .geo import route

    spatial = Workspace(hotel_id).read_spatial()
    if spatial is None or not spatial.confirmed_building_id:
        typer.secho("bâtiment confirmé requis", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    building = spatial.candidate(spatial.confirmed_building_id)
    routing = route(building.centroid_lat, building.centroid_lon)

    typer.echo(f"  territoires : {sorted(routing.territories)}")
    for source in routing.territorial_candidates:
        automated = "" if source.acquisition_automated else "  (acquisition non automatisée)"
        typer.echo(
            f"  ~ {source.source_id:<20} {routing.state_of(source.source_id).value}{automated}"
        )
    for source_id, reason in routing.rejected.items():
        typer.echo(f"  {KO} {source_id:<20} {reason}")


@geo_app.command("reference")
def geo_reference(
    hotel_id: str = typer.Argument(...),
    force: bool = typer.Option(False, "--force", help="Réécrit un contexte existant."),
) -> None:
    """Résout territoire et référentiels du site, et les fige.

    Le référentiel de calcul était une constante de module — `EPSG:2950`, le
    fuseau MTM 8 du Québec — appliquée partout. Il devient un fait spatial du
    site, résolu depuis sa position et opposable aux calculs.
    """
    from .geo import territory
    from .schemas.spatial_reference import TerritoryState

    context = _context(hotel_id, Capability.INSPECTION)
    workspace = Workspace(hotel_id)

    position = _reference_position(context, workspace)
    if position is None:
        typer.secho(
            f"{KO} aucune position de référence : renseignez lat/lon au profil "
            f"ou au manifeste de projet",
            fg=typer.colors.RED, err=True,
        )
        raise typer.Exit(code=1)

    path = workspace.path("00_manifest", "spatial_reference.json")
    if path.is_file() and not force:
        typer.secho(f"  contexte spatial déjà résolu : {path}", fg=typer.colors.YELLOW)
        raise typer.Exit(code=0)

    lat, lon = position
    # Le référentiel vertical vient des données déjà acquises, s'il y en a.
    acquisition = workspace.read_json("06_geo/acquisition_report.json")
    resolved = territory.resolve(
        hotel_id, lat, lon,
        vertical=territory.vertical_from_acquisition(acquisition),
    )

    typer.echo(f"  position    {lat:.6f}, {lon:.6f}")
    typer.echo(f"  territoire  {resolved.territory_state.value}")
    for code in resolved.jurisdictions:
        typer.echo(f"    · {code}")

    if resolved.territory_state is not TerritoryState.RESOLVED:
        typer.secho(
            f"  · aucun référentiel de travail — le géospatial restera "
            f"indisponible pour ce site",
            fg=typer.colors.YELLOW,
        )
    else:
        typer.echo(f"  référentiel {resolved.working_crs} ({resolved.working_unit})")
        typer.echo(f"    emprise   {resolved.working_area_of_use}")
        typer.echo(f"    choix     {resolved.selection_method}")

    vertical = resolved.vertical
    typer.echo(
        f"  vertical    {vertical.crs or 'inconnu'} — "
        f"qualification {'possible' if resolved.vertical_is_usable else 'impossible'}"
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(resolved.model_dump_json(indent=2) + "\n", "utf-8")
    typer.echo(f"{OK} {path}")


def _reference_position(context, workspace) -> tuple[float, float] | None:  # noqa: ANN001
    """Position servant à résoudre le territoire, profil d'abord."""
    profile = context.profile
    if profile is not None and profile.lat is not None and profile.lon is not None:
        return profile.lat, profile.lon
    try:
        project = workspace.read_manifest()
    except FileNotFoundError:
        return None
    if project.lat is None or project.lon is None:
        return None
    return project.lat, project.lon


@geo_app.command("discover")
def geo_discover(
    hotel_id: str = typer.Argument(...),
    no_sizes: bool = typer.Option(False, "--no-sizes", help="Ne pas mesurer les volumes."),
) -> None:
    """Découvre les tuiles LiDAR couvrant l'empreinte. **Ne télécharge aucun LAZ.**"""
    from .geo import CoverageState, route
    from .geo.adapters import elevation_adapter

    workspace = Workspace(hotel_id)
    spatial = workspace.read_spatial()
    if spatial is None or not spatial.confirmed_building_id:
        typer.secho("bâtiment confirmé requis", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    context = _context(hotel_id, Capability.GEOSPATIAL)
    building = spatial.candidate(spatial.confirmed_building_id)

    # Le routage décide **quoi** interroger ; l'adaptateur décide si on sait le
    # faire. Sans adaptateur, aucune requête n'est émise : interroger un service
    # québécois pour un site lyonnais faisait passer son silence pour une
    # absence de couverture.
    routing = route(building.centroid_lat, building.centroid_lon)
    adapter, reasons = elevation_adapter(routing)
    if adapter is None:
        typer.secho(f"{KO} découverte non prise en charge ici", fg=typer.colors.YELLOW)
        for reason in reasons:
            typer.echo(f"    · {reason}")
        typer.echo("    aucune requête émise — état « unsupported », non « non couvert »")
        workspace.write_json(
            "06_geo/lidar_discovery.json",
            {
                "coverage": "unsupported",
                "territories": sorted(routing.territories),
                "reasons": reasons,
                "queries_issued": 0,
                "provenance": context.provenance,
            },
        )
        raise typer.Exit(code=3)

    typer.echo(f"  source      {adapter.source_id}")
    result = adapter(building.wkt, measure_sizes=not no_sizes)
    workspace.write_report("06_geo/lidar_discovery.json", result, context)

    if result.state is CoverageState.DISCOVERY_ERROR:
        typer.secho(f"{KO} découverte impossible : {result.error}", fg=typer.colors.RED, err=True)
        typer.secho(
            "  ce n'est pas une absence de couverture — l'index n'a pas répondu",
            fg=typer.colors.YELLOW,
        )
        raise typer.Exit(code=4)

    if result.state is CoverageState.NOT_COVERED:
        typer.secho(
            f"{KO} aucune tuile n'intersecte l'empreinte "
            f"({result.considered} examinée(s))",
            fg=typer.colors.YELLOW,
        )
        raise typer.Exit(code=3)

    typer.echo(f"{OK} couverture confirmée — {len(result.tiles)} tuile(s)")
    for tile in result.tiles:
        typer.echo(f"    {tile.tile_id}  {tile.project or ''}  {tile.acquired_on or ''}")
        typer.echo(
            f"      densité {tile.point_density_per_m2 or '?'} pts/m²  "
            f"{tile.crs_horizontal or '?'} / {tile.crs_vertical or '?'}"
        )
    typer.echo("")
    typer.secho(
        f"  volume exact à télécharger : {result.total_bytes:,} octets".replace(",", " "),
        fg=typer.colors.YELLOW,
    )
    typer.echo("  aucun LAZ téléchargé — accord requis avant acquisition")


@geo_app.command("acquire")
def geo_acquire(
    hotel_id: str = typer.Argument(...),
    max_bytes: int = typer.Option(
        ..., "--max-bytes", help="Volume autorisé, en octets. Exigé explicitement."
    ),
) -> None:
    """Télécharge les tuiles découvertes. Le volume autorisé est obligatoire."""
    from pathlib import Path

    from .geo.acquire import download_tile, provenance_from
    from .geo.lidar import TileCandidate

    workspace = Workspace(hotel_id)
    discovery = workspace.read_json("06_geo/lidar_discovery.json")
    if not discovery or discovery.get("coverage") != "covered":
        typer.secho(
            "aucune couverture confirmée — lancez d'abord : geo discover",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    tiles = [TileCandidate(**{k: v for k, v in t.items() if k in TileCandidate.__annotations__})
             for t in discovery["tiles"]]
    total = discovery["total_bytes"]

    if total != max_bytes:
        typer.secho(
            f"{KO} volume découvert {total} ≠ volume autorisé {max_bytes} — "
            "acquisition refusée",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    context = _context(hotel_id, Capability.GEOSPATIAL)
    results = []
    for raw, tile in zip(discovery["tiles"], tiles):
        tile.acquired_on = None
        from datetime import date as _date

        if raw.get("acquired_on"):
            tile.acquired_on = _date.fromisoformat(raw["acquired_on"])
        target = workspace.path("06_geo", "lidar_raw", f"{tile.tile_id}.LAZ")
        typer.echo(f"  téléchargement {tile.tile_id} — {raw['exact_size_bytes']:,} octets"
                   .replace(",", " "))
        result = download_tile(tile.url, target, raw["exact_size_bytes"])
        results.append((result, tile))

    payload = {"acquisitions": [r.as_dict() for r, _ in results]}
    failed = [r for r, _ in results if not r.succeeded]

    if not failed:
        payload["sources"] = [
            provenance_from(r, t).model_dump(mode="json") for r, t in results
        ]

    workspace.write_report("06_geo/acquisition_report.json", payload, context, production="AcquiredLaz")

    for result, tile in results:
        if result.succeeded:
            typer.echo(f"  {OK} {tile.tile_id}  sha256 {result.sha256[:16]}…")
        else:
            typer.secho(f"  {KO} {tile.tile_id} — {result.error}", fg=typer.colors.RED)

    if failed:
        typer.secho(
            "  aucune source citable produite — aucun objet ne peut en dériver",
            fg=typer.colors.YELLOW,
        )
        raise typer.Exit(code=4)


@geo_app.command("preflight")
def geo_preflight(hotel_id: str = typer.Argument(...)) -> None:
    """Mesure ce que la tuile porte réellement. **Ne dérive aucun objet.**"""
    from .geo.preflight import BUILDING, GROUND, run

    workspace = Workspace(hotel_id)
    spatial = workspace.read_spatial()
    acquisition = workspace.read_json("06_geo/acquisition_report.json")
    if spatial is None or not acquisition or not acquisition.get("sources"):
        typer.secho(
            "tuile acquise et bâtiment confirmé requis", fg=typer.colors.RED, err=True
        )
        raise typer.Exit(code=1)

    context = _context(hotel_id, Capability.GEOSPATIAL)
    building = spatial.candidate(spatial.confirmed_building_id)
    source = acquisition["sources"][0]
    laz = Path(acquisition["acquisitions"][0]["path"])

    report = run(laz, building.wkt, source["crs_horizontal"])
    workspace.write_report("06_geo/laz_preflight.json", report, context)

    typer.echo(f"  fichier      {report.file}")
    typer.echo(f"  LAS {report.las_version}, format {report.point_format}, "
               f"{report.point_count:,} points".replace(",", " "))
    typer.echo(f"  CRS déclaré  {report.declared_crs}")
    typer.echo(f"  empreinte    {report.footprint_area_m2:.0f} m²")
    typer.echo("")
    typer.echo("  classe            empreinte   pourtour   z médian")
    for code, stats in sorted(report.footprint_classes.items()):
        margin = report.margin_classes[code]
        median = f"{stats.z_median:.1f}" if stats.z_median is not None else "—"
        typer.echo(
            f"  {code} {stats.name:<12} {stats.count:>9} {margin.count:>10} {median:>10}"
        )
    typer.echo("")
    typer.echo(f"  densité sol       {report.ground_density_per_m2} pts/m²")
    typer.echo(f"  densité bâtiment  {report.building_density_per_m2} pts/m²")
    typer.secho(
        f"  couverture toiture {(report.roof_cell_coverage or 0) * 100:.1f} % des cellules",
        fg=typer.colors.GREEN if (report.roof_cell_coverage or 0) >= 0.5 else typer.colors.YELLOW,
    )
    typer.echo(f"  couverture sol     {(report.ground_cell_coverage or 0) * 100:.1f} %")

    for warning in report.warnings:
        typer.secho(f"  ! {warning}", fg=typer.colors.YELLOW)
    if not report.warnings:
        typer.echo("")
        typer.echo("  aucun avertissement — la méthode de dérivation peut être arrêtée")


@geo_app.command("derive")
def geo_derive(hotel_id: str = typer.Argument(...)) -> None:
    """Produit les rasters dérivés. **Ne qualifie aucun objet.**"""
    import shutil

    from pyproj import Transformer
    from shapely import wkt as shapely_wkt
    from shapely.ops import transform as shapely_transform

    from .geo.derive import (
        derive,
        supersede_missing,
        supersede_previous,
        verify_digests,
        verify_publication,
        verify_written,
    )
    from .geo.raster import GridSpec

    workspace = Workspace(hotel_id)
    spatial = workspace.read_spatial()
    acquisition = workspace.read_json("06_geo/acquisition_report.json")
    if spatial is None or not acquisition or not acquisition.get("sources"):
        typer.secho("tuile acquise et bâtiment confirmé requis", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    context = _context(hotel_id, Capability.GEOSPATIAL)
    building = spatial.candidate(spatial.confirmed_building_id)
    source = acquisition["sources"][0]
    laz = Path(acquisition["acquisitions"][0]["path"])

    transformer = Transformer.from_crs("EPSG:4326", source["crs_horizontal"], always_xy=True)
    footprint = shapely_transform(
        lambda xs, ys, zs=None: transformer.transform(xs, ys),
        shapely_wkt.loads(building.wkt),
    )

    from datetime import datetime, timezone

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    staging = workspace.path("06_geo", f"_staging_{run_id}")
    if staging.exists():
        shutil.rmtree(staging)

    preflight = workspace.read_json("06_geo/laz_preflight.json") or {}
    typer.echo(f"  exécution {run_id} — production dans le répertoire de transit…")
    result = derive(
        laz, footprint, staging,
        crs=source["crs_horizontal"], crs_vertical=source["crs_vertical"],
        source_id=source["source_id"], laz_bounds=preflight.get("bounds"),
        policy=context.policy,
    )

    grid = GridSpec(**result.grid)
    problems = verify_written(result, grid, result.expected_layers)
    if problems:
        typer.secho(f"{KO} contrôle des couches échoué :", fg=typer.colors.RED, err=True)
        for problem in problems:
            typer.secho(f"    {problem}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=4)
    typer.echo(f"  {OK} {len(result.layers)} couche(s) relue(s) et contrôlée(s)")

    # Publication non destructive : chaque exécution a son répertoire, et les
    # productions antérieures restent consultables.
    final = workspace.path("06_geo", "derived", run_id)
    final.parent.mkdir(parents=True, exist_ok=True)
    staging.replace(final)
    for name in result.layers:
        result.layers[name] = str(final / f"{name}.tif")
    for artifact in result.artifacts:
        artifact.artifact_id = f"{artifact.artifact_id}@{run_id}"
        artifact.path = str(final / f"{artifact.path.rsplit('/', 1)[-1]}")
        artifact.derived_from_artifacts = [
            f"{parent}@{run_id}" for parent in artifact.derived_from_artifacts
        ]

    site = workspace.read_site()
    if site is not None:
        from .schemas import GeoSourceProvenance

        # Fusion par identifiant : rien n'est effacé, seul le même identifiant
        # est remplacé.
        sources = {s.source_id: s for s in site.geo_sources}
        sources[source["source_id"]] = GeoSourceProvenance.model_validate(source)
        site.geo_sources = list(sources.values())

        # Une série remplace la précédente : sans supersession, deux
        # exécutions resteraient actives et la qualification serait ambiguë.
        replaced = supersede_previous(site, result.artifacts)
        if replaced:
            typer.echo(f"  · {replaced} artefact(s) de la série précédente remplacé(s)")

        # La supersession prive de support les objets qui citaient ces
        # artefacts. Ne pas le dire ici laisserait un manifeste où un objet
        # « inferred » repose sur une production écartée — invalide à la
        # relecture, et faux entre-temps.
        from .geo.qualify import mark_stale

        for object_id in mark_stale(site):
            typer.secho(
                f"  · {object_id} repassé en 'stale' — sa dérivation a été remplacée ; "
                "relancez : geo qualify",
                fg=typer.colors.YELLOW,
            )

        artifacts = {a.artifact_id: a for a in site.artifacts}
        artifacts.update({a.artifact_id: a for a in result.artifacts})
        site.artifacts = list(artifacts.values())

        # Les productions antérieures restent au manifeste, mais celles dont le
        # fichier a disparu ne peuvent plus passer pour courantes.
        stale = supersede_missing(
            site,
            "fichier absent après réorganisation des publications ; produit "
            "avant la correction du rabattement des indices d'agrégation",
        )
        if stale:
            typer.secho(f"  · {stale} artefact(s) invalidé(s) — fichier absent",
                        fg=typer.colors.YELLOW)

        dead = verify_publication(site) + verify_digests(site)
        if dead:
            for problem in dead:
                typer.secho(f"{KO} {problem}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=4)

        workspace.write_site(site)

    # Matérialiser la politique effective à côté des résultats : le rapport
    # porte son empreinte, mais une empreinte ne se relit pas.
    if not workspace.policy_path.is_file():
        workspace.policy_path.parent.mkdir(parents=True, exist_ok=True)
        workspace.policy_path.write_text(
            context.policy.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        typer.echo(f"  · politique effective écrite dans {workspace.policy_path.name}")

    workspace.write_report(f"06_geo/derivation_report_{run_id}.json", result, context, production="DerivedRaster")

    metrics = result.metrics
    typer.echo("")
    typer.echo(f"  grille {grid.width} × {grid.height} à {grid.cell_m} m, {grid.crs}")
    typer.echo(f"  empreinte : {metrics['footprint_cells']} cellules")
    for key, value in metrics["coverage"].items():
        typer.echo(f"    {key:<22} {value * 100:5.1f} %")
    typer.echo("")
    typer.echo(f"  masques      {metrics['definition_masks']}")
    typer.echo(f"  TIN vs IDW   {metrics['tin_vs_idw']}")
    typer.echo(f"  refus extrap {metrics['extrapolation_rejected']}")
    typer.echo(f"  distance sol {metrics['support_distance_in_footprint']}")
    typer.echo(f"  bloc CV      {metrics['block_validation']}")
    pseudo = metrics["pseudo_footprint_validation"]
    typer.echo(
        f"  pseudo-empr. {len(pseudo['trials'])} essai(s), "
        f"{len(pseudo['rejected_candidates'])} refusé(s), "
        f"zone dans la tuile : {pseudo['search_area_within_tile']}"
    )
    for trial in pseudo["trials"]:
        typer.echo(
            f"      RMSE {trial['rmse_m']} m, p95 {trial['p95_m']} m, "
            f"appui max {trial['support_distance_max_m']} m, "
            f"{trial['reconstructed_points']}/{trial['masked_points']} reconstruits"
        )
    typer.echo(f"  hauteurs     {metrics['height_statistics']}")
    for flag in result.qa_flags:
        typer.secho(f"  ! {flag}", fg=typer.colors.YELLOW)
    typer.echo("")
    typer.secho(
        "  aucun objet qualifié — TERRAIN_MAIN et ROOFLINE_MAIN restent en l'état",
        fg=typer.colors.YELLOW,
    )


@geo_app.command("resolve")
def geo_resolve(
    hotel_id: str = typer.Argument(...),
    radius_m: int = typer.Option(350, "--radius", help="Périmètre du réseau routier."),
) -> None:
    """Résout les géométries de capture. **Ne modifie aucun objet du site.**"""
    import hashlib

    from .geo import capture_geometry as cg
    from .geo.projection import ProjectionService
    from .geo.resolve_geometry import resolve
    from .providers import overpass
    from .providers.cache import OfflineError
    from .steps import ELEMENTS_FILE

    workspace = Workspace(hotel_id)
    spatial = workspace.read_spatial()
    site = workspace.read_site()
    if spatial is None or not spatial.confirmed_building_id:
        typer.secho("bâtiment confirmé requis", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    context = _context(hotel_id, Capability.GEOSPATIAL)
    building = spatial.candidate(spatial.confirmed_building_id)
    elements = workspace.read_json(ELEMENTS_FILE) or []
    elements_digest = hashlib.sha256(
        json.dumps(elements, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:16]

    def _ref(kind: str) -> str | None:
        if site is None:
            return None
        return next(
            (o.source_ref for o in site.objects if o.kind == kind and o.source_ref), None
        )

    access_ref = _ref("ACCESS_ROAD_MAIN")
    parking_ref = _ref("PARKING_HOTEL")

    # Deux interrogations distinctes : le cache de collecte ne contient ni
    # route ni voie d'accès, et y chercher l'une aurait rendu une absence
    # inventée.
    roads = roads_error = None
    try:
        roads = overpass.roads_around(
            building.centroid_lat, building.centroid_lon, radius_m=radius_m
        )
    except (OverpassError, OfflineError, requests.RequestException) as exc:
        roads_error = str(exc)
        typer.secho(f"  ! réseau routier indisponible : {exc}", fg=typer.colors.YELLOW)

    access_element = access_error = None
    if access_ref:
        try:
            access_element = overpass.way_by_id(int(access_ref.split("/")[-1]))
        except (OverpassError, OfflineError, requests.RequestException, ValueError) as exc:
            access_error = str(exc)
            typer.secho(f"  ! voie d'accès indisponible : {exc}", fg=typer.colors.YELLOW)

    manifest, report = resolve(
        hotel_id=hotel_id,
        building_wkt=building.wkt,
        access_road_ref=access_ref,
        elements=elements,
        elements_digest=elements_digest,
        roads=roads,
        roads_error=roads_error,
        access_element=access_element,
        access_error=access_error,
        radius_m=float(radius_m),
        parking_ref=parking_ref,
        policy_digest=context.provenance["policy_digest"],
        adjacency_max_m=context.policy.geometry.adjacency_max_m,
        projection_service=ProjectionService(context.spatial_reference),
    )

    workspace.write_json(
        "06_geo/capture_geometry.json", json.loads(manifest.model_dump_json())
    )
    workspace.write_report("06_geo/geometry_resolution_report.json", report, context, production="CaptureGeometryManifest")

    typer.echo("")
    for snap in manifest.snapshots:
        mark = OK if snap.status.value == "success" else KO
        colour = typer.colors.GREEN if snap.status.value == "success" else typer.colors.YELLOW
        typer.secho(
            f"  {mark} {snap.snapshot_id:<26} {snap.status.value:<16} "
            f"{snap.element_count:>4} élément(s)",
            fg=colour,
        )
    typer.echo("")
    typer.echo(f"  résolues     {report.resolved}")
    for missing in report.unresolved:
        typer.secho(
            f"  · {missing.get('feature_id')} non résolu — {missing.get('reason')}",
            fg=typer.colors.YELLOW,
        )
    typer.echo("")
    typer.echo(f"  corridors    {report.corridors}")
    typer.echo(f"  empreintes   routes {report.road_geometry_digest} · "
               f"obstacles {report.obstacle_geometry_digest}")
    if report.crs_problems:
        typer.secho(f"{KO} incohérences de référentiel :", fg=typer.colors.RED, err=True)
        for problem in report.crs_problems:
            typer.secho(f"    {problem}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=4)
    typer.secho(
        "  aucun objet du site modifié : une géométrie absente ne conteste pas "
        "l'existence de l'objet",
        fg=typer.colors.YELLOW,
    )


@geo_app.command("qualify")
def geo_qualify(hotel_id: str = typer.Argument(...)) -> None:
    """Confronte la dernière dérivation aux seuils. N'écrit jamais `confirmed`."""
    from .geo import qualify as qualification
    from .intake import sha256_file

    workspace = Workspace(hotel_id)
    site = workspace.read_site()
    if site is None:
        typer.secho("aucun manifeste de site", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    reports = sorted(workspace.path("06_geo").glob("derivation_report_*.json"))
    if not reports:
        typer.secho(
            "aucun rapport de dérivation — lancez d'abord : geo derive",
            fg=typer.colors.RED, err=True,
        )
        raise typer.Exit(code=1)

    # La plus récente dérivation, et elle seule : qualifier sur une série
    # antérieure reviendrait à juger des rasters que la supersession a écartés.
    latest = reports[-1]
    derivation = workspace.read_json(f"06_geo/{latest.name}")
    context = _context(hotel_id, Capability.GEOSPATIAL_QUALIFICATION)

    # Qualifier sur des seuils que la politique du projet ne porte pas
    # reviendrait à calibrer en silence depuis le code — à n'importe quelle
    # profondeur : un seuil ajouté dans une section déjà présente passerait
    # tout aussi inaperçu qu'une section entière absente.
    implicit = context.implicit_under("qualification")
    if implicit:
        typer.secho(
            f"{KO} la politique effective ({context.policy.version}) ne porte pas "
            f"ces valeurs, qui viendraient du code : {', '.join(implicit)}",
            fg=typer.colors.RED, err=True,
        )
        typer.secho(
            f"  révisez puis remplacez {workspace.policy_path}",
            fg=typer.colors.YELLOW, err=True,
        )
        raise typer.Exit(code=1)

    # Tout est contrôlé avant la moindre mutation : les fichiers cités existent,
    # leur contenu réel correspond à l'empreinte déclarée, et la série active
    # est bien celle que le rapport décrit.
    #
    # `check_series` confronte deux **déclarations** — l'empreinte du manifeste
    # et celle du rapport. Toutes deux peuvent concorder pendant que le GeoTIFF,
    # lui, a été réécrit depuis ; seule la relecture du contenu le dit.
    from .geo.derive import verify_digests, verify_publication

    run_id = latest.stem.removeprefix("derivation_report_")
    mismatches = (
        verify_publication(site)
        + verify_digests(site)
        + qualification.check_series(site, derivation, run_id)
    )
    if mismatches:
        typer.secho(
            f"{KO} la série active n'est pas en état d'être jugée :",
            fg=typer.colors.RED, err=True,
        )
        for problem in mismatches:
            typer.secho(f"    {problem}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=4)

    # Les artefacts remplacés emportent les décisions qu'ils fondaient.
    for object_id in qualification.mark_stale(site):
        typer.secho(
            f"  · {object_id} repassé en 'stale' — artefact cité remplacé",
            fg=typer.colors.YELLOW,
        )

    mapping = qualification.select_artifacts(site)
    report = qualification.report(
        derivation["metrics"],
        context.policy,
        digest=sha256_file(latest)[:16],
        artifacts=sorted({a for ids in mapping.values() for a in ids}),
        run_id=run_id,
    )

    # Le rapport est publié avant d'être cité : un objet doit pouvoir renvoyer
    # à l'empreinte d'un fichier qui existe.
    published = workspace.write_report(
        f"06_geo/{report.name}", report, context,
        production="QualificationReport",
    )
    qualified = qualification.apply(
        site, report, mapping, report_digest=sha256_file(published)[:16]
    )
    workspace.write_site(site)

    typer.echo("")
    typer.echo(f"  rapport publié   : {report.name}")
    typer.echo(f"  dérivation jugée : {latest.name} ({report.qualified_derivation_digest})")
    typer.echo(
        f"  politique {report.policy_version} · {report.qualification_status} · "
        f"{report.intended_use} · calibrée sur {report.calibrated_on_sites} site(s)"
    )
    for kind, verdict in report.verdicts.items():
        typer.echo("")
        colour = typer.colors.GREEN if verdict.passed else typer.colors.YELLOW
        # `qualified` porte des identifiants d'objet, jamais des types : les
        # comparer directement afficherait « non qualifié » sur un objet qui
        # vient d'être inscrit comme inféré.
        inscribed = [o.object_id for o in site.objects if o.kind == kind and o.object_id in qualified]
        state = "inferred" if inscribed else "non qualifié"
        typer.secho(f"  {kind:<14} {state} · confiance {verdict.confidence}", fg=colour)
        for criterion in verdict.criteria:
            mark = OK if criterion.passed else KO
            typer.echo(
                f"      {mark} {criterion.name:<26} {criterion.threshold:<12} "
                f"mesuré {criterion.measured}"
            )
        for reservation in verdict.reservations:
            typer.secho(f"      réserve : {reservation}", fg=typer.colors.YELLOW)

    typer.echo("")
    typer.secho(
        "  aucun objet confirmé : une dérivation reste une inférence",
        fg=typer.colors.YELLOW,
    )


visibility_app = typer.Typer(
    no_args_is_help=True, help="Visibilité multi-rayons (Lot 1B V2 §3)."
)
app.add_typer(visibility_app, name="visibility")


@visibility_app.command("assess")
def visibility_assess(hotel_id: str = typer.Argument(...)) -> None:
    """Mesure la visibilité géométrique. **Ne modifie aucun asset.**"""
    import hashlib
    from datetime import datetime, timezone

    from .geo.visibility_run import base_manifest_digest, digest, run_assessment
    from .intake import sha256_file
    from .schemas.geometry import CaptureGeometryManifest

    workspace = Workspace(hotel_id)
    context = _context(hotel_id, Capability.GEOSPATIAL)

    # Les réglages du moteur doivent figurer dans la politique du projet :
    # calculés depuis les valeurs du code, ils ne se rejoueraient pas.
    implicit = context.implicit_under("visibility")
    if implicit:
        typer.secho(
            f"{KO} la politique effective ({context.policy.version}) ne porte pas "
            f"ces réglages, qui viendraient du code : {', '.join(implicit)}",
            fg=typer.colors.RED, err=True,
        )
        raise typer.Exit(code=1)

    manifest = _capture_geometry_if_any(workspace, context)
    if manifest is None:
        typer.secho(
            f"{KO} aucun manifeste géométrique — lancez d'abord : geo resolve",
            fg=typer.colors.RED, err=True,
        )
        raise typer.Exit(code=1)

    assets_manifest = workspace.read_assets()
    site = workspace.read_site()
    if assets_manifest is None:
        typer.secho("aucun manifeste d'assets", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    from .schemas.geometry import GeometryRole

    digests = {
        "capture_geometry": digest(raw),
        "policy": context.provenance["policy_digest"],
        "site_manifest": digest(json.loads(site.model_dump_json())) if site else "sans-site",
        # Deux empreintes : les fichiers, et le manifeste entier. Une revue
        # humaine ou un cap corrigé ne changent pas les images.
        "asset_files": digest(sorted(f"{a.id}:{a.checksum}" for a in assets_manifest.assets)),
        # Empreinte **de base** : les champs que `visibility apply` écrira en
        # sont exclus, sans quoi l'application périmerait son propre run.
        "asset_manifest": base_manifest_digest(assets_manifest),
        "obstacles": digest(
            sorted(
                g.geometry_digest or ""
                for g in manifest.by_role(GeometryRole.OBSTACLE_BUILDING)
            )
        ),
        "roads": digest(
            sorted(
                g.geometry_digest or ""
                for g in manifest.geometries
                if g.role.value in ("access_road", "road_candidate")
            )
        ),
    }

    # --- verticales mesurées -------------------------------------------------
    # Six données manquaient à chaque rayon à risque ; trois sont dérivables de
    # ce qui est déjà acquis. La quatrième — la hauteur d'œil — ne l'est pas.
    from .geo.elevation import CloudSampler, RasterSampler
    from .geo.visibility_engine import TargetVertical

    target_vertical = None
    camera_ground = None
    obstacle_heights: dict[str, dict] = {}
    enrichment: dict[str, str] = {}

    # Les rasters sont choisis parmi les artefacts **actifs** du manifeste de
    # site, non par « dernier répertoire trié » : un répertoire plus récent
    # peut porter une dérivation invalidée.
    from .schemas.visibility import ElevationSource

    elevation_sources: list[ElevationSource] = []
    active = {a.role: a for a in (site.artifacts if site else []) if a.is_active}
    dtm_artifact, roof_artifact = active.get("dtm"), active.get("dsm_roof")

    # Le rapport de qualification vient des objets qualifiés eux-mêmes, non du
    # dernier fichier d'un glob : c'est `TERRAIN_MAIN` et `ROOFLINE_MAIN` qui
    # disent sur quelle décision ils reposent, et quels artefacts ils citent.
    qualified = {
        o.kind: o for o in (site.objects if site else []) if o.qualification_report
    }
    terrain, roofline = qualified.get("TERRAIN_MAIN"), qualified.get("ROOFLINE_MAIN")

    if dtm_artifact and roof_artifact and terrain and roofline:
        cited = set(terrain.artifact_ids) | set(roofline.artifact_ids)
        uncited = [
            artifact.artifact_id
            for artifact in (dtm_artifact, roof_artifact)
            if artifact.artifact_id not in cited
        ]
        if uncited:
            typer.secho(
                f"{KO} artefacts non cités par les objets qualifiés : {uncited} — "
                "un raster prétendument qualifié doit l'être par la décision",
                fg=typer.colors.RED, err=True,
            )
            raise typer.Exit(code=4)

        reports = {
            name: workspace.path("06_geo", name)
            for name in {terrain.qualification_report, roofline.qualification_report}
        }
        missing = [name for name, path in reports.items() if not path.is_file()]
        if missing:
            typer.secho(
                f"{KO} rapport(s) de qualification introuvable(s) : {missing}",
                fg=typer.colors.RED, err=True,
            )
            raise typer.Exit(code=4)

        for name, path in reports.items():
            recorded = (
                terrain.qualification_report_digest
                if name == terrain.qualification_report
                else roofline.qualification_report_digest
            )
            if recorded and sha256_file(path)[:16] != recorded:
                typer.secho(
                    f"{KO} {name} : empreinte différente de celle inscrite à l'objet",
                    fg=typer.colors.RED, err=True,
                )
                raise typer.Exit(code=4)

        qualification = next(iter(reports.values()))
        latest_derivation = Path(dtm_artifact.path).parent
        sampler = RasterSampler(
            Path(dtm_artifact.path), Path(roof_artifact.path),
            f"artefacts {dtm_artifact.artifact_id} / {roof_artifact.artifact_id}",
        )
        for artifact, role in ((dtm_artifact, "target_ground"), (roof_artifact, "target_top")):
            elevation_sources.append(
                ElevationSource(
                    kind="raster", role=role, artifact_id=artifact.artifact_id,
                    path=artifact.path, sha256=artifact.sha256,
                    horizontal_crs=artifact.crs_horizontal,
                    vertical_crs=artifact.crs_vertical,
                    sampling_method="rasterio.sample au point visé, puis 0,5/1,5/3 m à l'intérieur",
                    qualification_report=qualification.name,
                    qualification_digest=sha256_file(qualification)[:16],
                )
            )

        def _sample(point):  # noqa: ANN001
            ground, roof = sampler.at(point[0], point[1])
            return (
                ground.value_m if ground else None,
                roof.value_m if roof else None,
            )

        target_vertical = TargetVertical(
            sampler=_sample, provenance=f"TERRAIN_MAIN+ROOFLINE_MAIN@{latest_derivation.name}"
        )
        enrichment["target"] = f"rasters qualifiés {latest_derivation.name}"
    else:
        enrichment["target"] = (
            "aucun couple artefact actif / objet qualifié pour la cible"
        )

    acquisition = workspace.read_json("06_geo/acquisition_report.json") or {}
    entry = (acquisition.get("acquisitions") or [{}])[0]
    # La jointure se fait par empreinte, non par position : les deux listes
    # n'ont aucune raison d'être dans le même ordre, et les métadonnées de
    # tuile — identifiant, référentiel vertical — vivent du côté `sources`.
    tile = next(
        (
            source
            for source in acquisition.get("sources", [])
            if source.get("file_digest") == entry.get("sha256")
        ),
        {},
    )
    laz = entry.get("path")
    if laz and Path(laz).is_file():
        # Le nuage est relu et son empreinte recalculée : une source verticale
        # dont on n'a pas vérifié le contenu ne prouve rien.
        laz_digest = sha256_file(Path(laz))
        declared = entry.get("sha256")
        if declared and declared != laz_digest:
            typer.secho(
                f"{KO} le nuage {Path(laz).name} diffère de son empreinte déclarée",
                fg=typer.colors.RED, err=True,
            )
            raise typer.Exit(code=4)
        if not tile:
            typer.secho(
                f"{KO} aucune source ne correspond à l'empreinte du nuage acquis — "
                "la provenance verticale serait invérifiable",
                fg=typer.colors.RED, err=True,
            )
            raise typer.Exit(code=4)
        cloud = CloudSampler.load(Path(laz), f"laz:{tile['tile_id']}")
        from shapely import wkt as shapely_wkt

        from .schemas.geometry import GeometryResolutionStatus

        for geometry in manifest.by_role(GeometryRole.OBSTACLE_BUILDING):
            if geometry.resolution_status is not GeometryResolutionStatus.RESOLVED:
                continue
            ground, top = cloud.within(shapely_wkt.loads(geometry.projected_wkt))
            if ground and top:
                obstacle_heights[geometry.feature_id] = {
                    "ground_m": ground.value_m,
                    "height_m": max(top.value_m - ground.value_m, 0.0),
                    "provenance": top.provenance,
                }
        enrichment["obstacles"] = (
            f"{len(obstacle_heights)}/{len(manifest.by_role(GeometryRole.OBSTACLE_BUILDING))} "
            "mesurés dans le nuage"
        )
        camera_ground = cloud.ground_near
        enrichment["camera_ground"] = "classe 2 au voisinage de la position"

        obstacles_total = len(manifest.by_role(GeometryRole.OBSTACLE_BUILDING))
        for role, method in (
            ("obstacle_height", "médiane classe 2 et p95 classe 6 sous l'emprise"),
            ("camera_ground", "médiane classe 2 dans un rayon de 8 m"),
        ):
            elevation_sources.append(
                ElevationSource(
                    kind="point_cloud", role=role,
                    tile_id=tile["tile_id"], path=str(laz), sha256=laz_digest,
                    horizontal_crs=tile["crs_horizontal"],
                    vertical_crs=tile["crs_vertical"],
                    sampling_method=method,
                    measured=len(obstacle_heights) if role == "obstacle_height" else None,
                    attempted=obstacles_total if role == "obstacle_height" else None,
                )
            )

    enrichment["camera_height"] = (
        "inconnue : aucune source ne publie la hauteur de capteur"
    )

    spatial = workspace.read_spatial()
    front = spatial.front_azimuth_deg if spatial else None

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    if context.spatial_reference is None:
        typer.secho(
            f"{KO} aucun contexte spatial résolu — lancez d'abord : geo reference",
            fg=typer.colors.RED, err=True,
        )
        raise typer.Exit(code=1)

    try:
        run, report = run_assessment(
            run_id, hotel_id, assets_manifest.assets, manifest, context.policy, digests,
            front_azimuth_deg=front, target_vertical=target_vertical,
            camera_ground=lambda origin: camera_ground(*origin) if camera_ground else None,
            obstacle_heights=obstacle_heights,
            elevation_sources=elevation_sources,
            spatial_reference=context.spatial_reference,
        )
    except ValueError as exc:
        typer.secho(f"{KO} {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    report.enrichment = enrichment

    workspace.write_json(
        f"06_geo/visibility_run_{run_id}.json", json.loads(run.model_dump_json())
    )
    workspace.write_report(f"06_geo/visibility_report_{run_id}.json", report, context, production="VisibilityRun")

    typer.echo("")
    typer.echo(f"  exécution {run_id} · moteur {run.engine_version}")
    for key, value in enrichment.items():
        typer.echo(f"    verticale · {key:<14} {value}")
    typer.echo(f"  {report.assets_assessed} asset(s) situé(s) évalué(s)")
    for status, total in sorted(report.by_status.items()):
        typer.echo(f"    {status:<20} {total:>4}")
    typer.echo("")
    typer.secho(
        f"  occultations déclarées auparavant : {report.previously_occluded} → "
        f"{report.previously_occluded_now}",
        fg=typer.colors.YELLOW,
    )
    typer.echo(f"  blocages prouvés : {report.proven_blocked}")
    typer.echo("")
    typer.echo("  données verticales manquantes :")
    for missing, total in sorted(
        report.missing_vertical_counts.items(), key=lambda item: -item[1]
    )[:6]:
        typer.echo(f"    {missing:<44} {total:>4} asset(s)")
    typer.echo("")
    typer.echo(f"  cadrages calculables : {report.framing_computable}")
    for reason, total in sorted(report.framing_not_computable.items()):
        typer.echo(f"    non calculable · {reason:<36} {total:>4}")
    typer.echo("")
    typer.echo(f"  corridors évalués : {report.corridors_assessed} · "
               f"utilité {report.corridors_useful}")
    typer.echo("")
    typer.secho(
        "  aucun asset modifié : la projection est une commande distincte",
        fg=typer.colors.YELLOW,
    )


@visibility_app.command("apply")
def visibility_apply(
    hotel_id: str = typer.Argument(...),
    run_id: str = typer.Argument(..., help="Exécution exacte à projeter."),
    supersede: str = typer.Option(
        None, "--supersede", help="Exécution appliquée que celle-ci remplace."
    ),
) -> None:
    """Projette une exécution vers les assets. Atomique, et sans promotion."""
    from .geo import visibility_apply as projection
    from .geo.visibility_run import digest
    from .intake import sha256_file
    from .schemas.geometry import GeometryRole
    from .schemas.visibility import VisibilityRun

    workspace = Workspace(hotel_id)
    context = _context(hotel_id, Capability.GEOSPATIAL)

    # Le `run_id` est exigé : « le dernier rapport » appliquerait ce qui vient
    # d'être mesuré sans que personne ne l'ait lu.
    path = workspace.path("06_geo", f"visibility_run_{run_id}.json")
    if not path.is_file():
        typer.secho(f"{KO} exécution introuvable : {path.name}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    run = VisibilityRun.model_validate(json.loads(path.read_text("utf-8")))
    run_digest = sha256_file(path)[:16]

    manifest = workspace.read_assets()
    site = workspace.read_site()
    geometry = _capture_geometry_if_any(workspace, context)
    if manifest is None or geometry is None:
        typer.secho(
            "manifeste d'assets et manifeste géométrique requis",
            fg=typer.colors.RED, err=True,
        )
        raise typer.Exit(code=1)

    # Les images sont relues : un checksum déclaré ne prouve pas le contenu.
    altered, absent = [], []
    for asset in manifest.assets:
        if not asset.local_path:
            continue
        path_ = Path(asset.local_path)
        if not path_.is_file():
            # Un fichier déclaré mais absent n'est pas « inchangé » : rien ne
            # dit ce qu'il contenait quand la mesure a été prise.
            absent.append(asset.id)
        elif sha256_file(path_) != asset.checksum:
            altered.append(asset.id)
    if altered or absent:
        typer.secho(
            f"{KO} {len(altered)} image(s) modifiée(s) et {len(absent)} absente(s) : "
            f"{(altered + absent)[:5]}",
            fg=typer.colors.RED, err=True,
        )
        raise typer.Exit(code=4)

    for source in run.elevation_sources:
        source_path = Path(source.path)
        if not source_path.is_file():
            typer.secho(f"{KO} source d'élévation absente : {source.path}",
                        fg=typer.colors.RED, err=True)
            raise typer.Exit(code=4)
        if sha256_file(source_path) != source.sha256:
            typer.secho(
                f"{KO} {source_path.name} : contenu différent de l'empreinte citée "
                f"par l'exécution ({source.role})",
                fg=typer.colors.RED, err=True,
            )
            raise typer.Exit(code=4)

    current = {
        "policy": context.provenance["policy_digest"],
        "capture_geometry": digest(raw_geometry),
        "site_manifest": digest(json.loads(site.model_dump_json())) if site else "sans-site",
        "asset_files": digest(sorted(f"{a.id}:{a.checksum}" for a in manifest.assets)),
        "obstacles": digest(
            sorted(
                g.geometry_digest or ""
                for g in geometry.by_role(GeometryRole.OBSTACLE_BUILDING)
            )
        ),
        "roads": digest(
            sorted(
                g.geometry_digest or ""
                for g in geometry.geometries
                if g.role.value in ("access_road", "road_candidate")
            )
        ),
        "target": next(
            (
                g.geometry_digest
                for g in geometry.by_role(GeometryRole.TARGET_BUILDING)
                if g.geometry_digest
            ),
            None,
        ),
    }

    # Le rapport de qualification cité par les sources d'élévation doit encore
    # exister, porter la même empreinte, et rester celui des objets qualifiés.
    qualified = {
        o.kind: o for o in (site.objects if site else []) if o.qualification_report
    }
    for source in run.elevation_sources:
        if not source.qualification_report:
            continue
        report_path = workspace.path("06_geo", source.qualification_report)
        if not report_path.is_file() or sha256_file(report_path)[:16] != source.qualification_digest:
            typer.secho(
                f"{KO} rapport de qualification {source.qualification_report} absent "
                "ou d'empreinte différente",
                fg=typer.colors.RED, err=True,
            )
            raise typer.Exit(code=4)
        cited = {o.qualification_report for o in qualified.values()}
        if source.qualification_report not in cited:
            typer.secho(
                f"{KO} {source.qualification_report} n'est plus le rapport cité par "
                "les objets qualifiés",
                fg=typer.colors.RED, err=True,
            )
            raise typer.Exit(code=4)

    receipt = workspace.path("06_geo", projection.receipt_name(run_id, run_digest))

    # La vérification précède la détection d'idempotence : sinon une politique
    # ou une géométrie modifiée depuis laisserait déclarer « déjà appliqué » un
    # run devenu invalide.
    problems = projection.verify(
        run, manifest, hotel_id, current, context.spatial_reference
    )
    if problems:
        typer.secho(f"{KO} exécution non applicable :", fg=typer.colors.RED, err=True)
        for problem in problems:
            typer.secho(f"    {problem}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=4)

    idempotent, divergent = projection.already_applied(manifest, run)
    if idempotent:
        # Une commande interrompue après l'écriture du manifeste doit pouvoir
        # se rejouer : le reçu manquant se reconstruit sans rien remuter.
        typer.secho(f"  {OK} exécution déjà appliquée — assets inchangés",
                    fg=typer.colors.GREEN)
        if not receipt.is_file():
            report, _ = projection.project(manifest, run, run_digest, context.policy)
            report.status = "already_applied"
            workspace.write_report(f"06_geo/{receipt.name}", report, context)
            typer.echo(f"  reçu manquant reconstruit : {receipt.name}")
        raise typer.Exit(code=0)
    if divergent and not supersede:
        typer.secho(f"{KO} application refusée :", fg=typer.colors.RED, err=True)
        for problem in divergent[:8]:
            typer.secho(f"    {problem}", fg=typer.colors.RED, err=True)
        typer.secho(
            "  une revue humaine impose une nouvelle mesure : relancez "
            "`visibility assess`, puis appliquez-la avec --supersede",
            fg=typer.colors.YELLOW, err=True,
        )
        raise typer.Exit(code=4)

    if supersede:
        stale = projection.supersedes(run, supersede, manifest)
        if stale:
            typer.secho(f"{KO} remplacement refusé :", fg=typer.colors.RED, err=True)
            for problem in stale:
                typer.secho(f"    {problem}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=4)

    try:
        report, projected = projection.project(
            manifest, run, run_digest, context.policy, superseded=supersede
        )
    except projection.ApplicationRefused as exc:
        typer.secho(f"{KO} {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=4) from exc

    # Dernière écriture, et seule mutation : le manifeste d'abord, le reçu
    # ensuite — un reçu sans manifeste décrirait une application qui n'a pas eu
    # lieu.
    workspace.write_assets(projected)
    workspace.write_report(f"06_geo/{receipt.name}", report, context)

    typer.echo("")
    typer.echo(f"  exécution {run_id} ({run_digest}) appliquée")
    typer.echo(f"  {report.assets_updated} asset(s) mis à jour")
    typer.echo(f"  champs écrits : {', '.join(report.fields_written)}")
    typer.echo("")
    typer.secho(
        f"  {len(report.former_occlusions)} occultation(s) déclarée(s) retirée(s) "
        f"faute de preuve · {len(report.occluded_by_kept)} conservée(s)",
        fg=typer.colors.YELLOW,
    )
    typer.echo("")
    if supersede:
        typer.echo(f"  remplace l'exécution {supersede}, conservée avec son reçu")
    typer.echo(f"  rôles avant : {report.roles_before}")
    typer.echo(f"  rôles après : {report.roles_after}")
    for demotion in report.demotions:
        typer.secho(
            f"    ↓ {demotion['asset_id']} {demotion['from']} → {demotion['to']} "
            f"({demotion['reason']})",
            fg=typer.colors.YELLOW,
        )
    typer.secho(
        f"  {len(report.demotions)} rétrogradation(s) prouvée(s), aucune promotion, "
        "aucune décision humaine touchée",
        fg=typer.colors.GREEN,
    )


site_app = typer.Typer(no_args_is_help=True, help="Instances du site (Lot 1B §4).")
app.add_typer(site_app, name="site")


@site_app.command("build")
def site_build(hotel_id: str = typer.Argument(...)) -> None:
    """Instancie les objets du site depuis le gabarit générique."""
    from . import site as site_builder
    from .steps import ELEMENTS_FILE

    workspace = Workspace(hotel_id)
    spatial = workspace.read_spatial()
    if spatial is None:
        typer.secho("aucun manifeste spatial", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    context = _context(hotel_id, Capability.GEOSPATIAL)
    elements = workspace.read_json(ELEMENTS_FILE) or []
    roads = workspace.read_json("01_sources/road_network.json") or []
    manifest = workspace.read_assets()

    site, report = site_builder.build(
        hotel_id, spatial, elements, roads, manifest.assets if manifest else None
    )
    workspace.write_site(site)
    workspace.write_report("01_sources/site_report.json", report, context)

    for key, value in site.summary().items():
        typer.echo(f"  {key:<20} {value}")
    if site.missing_required():
        typer.secho(f"  types non instanciés : {site.missing_required()}", fg=typer.colors.RED)


@site_app.command("show")
def site_show(hotel_id: str = typer.Argument(...)) -> None:
    """Liste les instances du site, leur état et leurs relations."""
    site = Workspace(hotel_id).read_site()
    if site is None:
        typer.secho("aucun manifeste de site — lancez : site build", fg=typer.colors.YELLOW)
        raise typer.Exit(code=1)

    for obj in sorted(site.objects, key=lambda o: (o.state.value, o.kind)):
        mark = {"confirmed": OK, "inferred": "~", "conflicted": "!", "unresolved": "·"}[
            obj.state.value
        ]
        source = obj.source_ref or ""
        typer.echo(f"  {mark} {obj.kind:<24} {obj.state.value:<11} {source}")
        if obj.unresolved_reason:
            typer.echo(f"      motif : {obj.unresolved_reason}")
        for relation in obj.relations:
            typer.echo(f"      {relation.predicate} → {relation.target_id.split(':')[-1]}")


temporal_app = typer.Typer(
    no_args_is_help=True, help="Datation par portée (entrance, facade, roof, signage)."
)
app.add_typer(temporal_app, name="temporal")


@temporal_app.command("assess")
def temporal_assess(hotel_id: str = typer.Argument(...)) -> None:
    """Dérive la datation par portée, sans écraser les décisions humaines."""
    from .temporal import assess

    workspace = Workspace(hotel_id)
    manifest = workspace.read_assets()
    if manifest is None:
        typer.secho("aucun manifeste d'assets", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    context = _context(hotel_id, Capability.IDENTITY_CLASSIFICATION)
    report = assess(manifest.assets, context.profile, context.policy)
    workspace.write_assets(manifest)
    workspace.write_report("01_sources/temporal_report.json", report, context, production="TemporalReport")

    for scope, counts in sorted(report.by_scope.items()):
        typer.echo(f"  {scope:<12} {counts}")
    typer.echo(f"  décisions humaines            {report.human_decisions}")
    typer.echo(f"  portées sensibles indéterminées {report.sensitive_unknown}")


@temporal_app.command("set")
def temporal_set(
    hotel_id: str = typer.Argument(...),
    asset_id: str = typer.Argument(...),
    scope: str = typer.Argument(..., help="entrance, facade, roof, signage..."),
    status: str = typer.Argument(
        ..., help="current_confirmed, before_event, after_event, historical, unknown"
    ),
    by: str = typer.Option(..., "--by", help="Auteur de la décision."),
    rationale: str = typer.Option(..., "--rationale", help="Justification, conservée."),
    evidence: list[str] = typer.Option([], "--evidence", help="Preuves, répétable."),
) -> None:
    """Enregistre une décision humaine de datation, prioritaire et durable."""
    from .schemas import TemporalDecision, TemporalStatus

    workspace = Workspace(hotel_id)
    manifest = workspace.read_assets()
    if manifest is None:
        typer.secho("aucun manifeste d'assets", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    try:
        parsed = TemporalStatus(status)
    except ValueError:
        allowed = ", ".join(m.value for m in TemporalStatus)
        typer.secho(f"{KO} statut invalide ; attendu : {allowed}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from None

    asset = next((a for a in manifest.assets if a.id == asset_id), None)
    if asset is None:
        typer.secho(f"{KO} asset inconnu : {asset_id}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    decision = TemporalDecision(
        scope=scope, status=parsed, decided_by=by, rationale=rationale,
        evidence=list(evidence),
    )
    kept = [d for d in asset.temporal_decisions if d.scope != scope]
    manifest.assets[manifest.assets.index(asset)] = asset.model_copy(
        update={
            "temporal_decisions": [*kept, decision],
            "temporal_by_scope": {**asset.temporal_by_scope, scope: parsed},
        }
    )
    workspace.write_assets(manifest)
    typer.echo(f"{OK} {asset_id} — {scope} : {parsed.value} (par {by})")


reconstruction_app = typer.Typer(
    no_args_is_help=True, help="Lot 2 — Reconstruction 3D (SfM, alignement, densification)."
)
app.add_typer(reconstruction_app, name="reconstruction")


@reconstruction_app.command("prepare-input")
def reconstruction_prepare_input(
    hotel_id: str = typer.Argument(..., help="Identifiant de l'hôtel."),
    backend: list[str] = typer.Option(
        ["colmap_incremental"],
        "--backend",
        help="Backend(s) autorisé(s) pour cette reconstruction.",
    ),
    mask_set_digest: str | None = typer.Option(
        None,
        "--mask-set-digest",
        help="Empreinte SHA-256 du jeu de masques SfM, si applicable.",
    ),
) -> None:
    """Crée un ReconstructionInputManifest immuable pour le Lot 2.

    Ce snapshot gèle la sélection d'assets, les empreintes des manifestes
    sources et les droits avant toute exécution de solveur. Tous les
    backends (COLMAP, GLUEMAP, MP-SfM, MapAnything, VGGT) reçoivent
    exactement les mêmes données.
    """
    from .reconstruction_input import prepare_input, publish_input

    try:
        manifest, _ = prepare_input(
            hotel_id,
            allowed_backends=backend,
            mask_set_digest=mask_set_digest,
        )
    except (FileNotFoundError, ValueError, ValidationError) as exc:
        typer.secho(f"✗ {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    workspace = Workspace(hotel_id)
    output_path = publish_input(manifest, workspace)
    typer.echo(
        f"{OK} ReconstructionInputManifest créé : {output_path}\n"
        f"    ID   : {manifest.reconstruction_input_id}\n"
        f"    Assets sélectionnés : {len(manifest.selected_asset_ids)}\n"
        f"    Assets exclus      : {len(manifest.excluded_asset_ids)}\n"
        f"    Backends autorisés : {', '.join(manifest.allowed_backends)}"
    )


@reconstruction_app.command("preprocess")
def reconstruction_preprocess(
    hotel_id: str = typer.Argument(..., help="Identifiant de l'hôtel."),
    mask_classes: list[str] = typer.Option(
        ["sky", "people", "cars", "water"],
        "--mask-class",
        help="Classes à masquer (répétable).",
    ),
) -> None:
    """Génère les masques SfM et les images normalisées pour le Lot 2.

    Les masques sont stockés comme `DerivedArtifact` et ne modifient
    jamais les images originales.
    """
    from .reconstruction_input import prepare_input
    from .reconstruction_preprocess import generate_mask_set, publish_mask_set

    try:
        input_manifest, _ = prepare_input(hotel_id)
    except (FileNotFoundError, ValueError, ValidationError) as exc:
        typer.secho(f"✗ {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    workspace = Workspace(hotel_id)
    mask_digest = generate_mask_set(workspace, input_manifest, mask_classes=mask_classes)

    input_path = workspace.path("07_reconstruction", f"reconstruction_input_{input_manifest.reconstruction_input_id}.json")
    data = json.loads(input_path.read_text("utf-8"))
    data["mask_set_digest"] = mask_digest
    input_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")

    typer.echo(
        f"{OK} Masques générés : {mask_digest[:16]}...\n"
        f"    Classes masquées : {', '.join(mask_classes)}\n"
        f"    Entrée mise à jour : {input_path.name}"
    )


@reconstruction_app.command("view-graph")
def reconstruction_view_graph(
    hotel_id: str = typer.Argument(..., help="Identifiant de l'hôtel."),
) -> None:
    """Construit le graphe de vue (ViewGraphManifest) depuis l'entrée de reconstruction."""
    from .reconstruction_input import prepare_input
    from .view_graph import ViewGraphBuilder

    try:
        input_manifest, _ = prepare_input(hotel_id)
    except (FileNotFoundError, ValueError, ValidationError) as exc:
        typer.secho(f"✗ {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    workspace = Workspace(hotel_id)
    builder = ViewGraphBuilder(workspace)
    try:
        view_graph = builder.build(input_manifest)
    except Exception as exc:
        typer.secho(f"✗ construction du graphe échouée : {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    output_dir = workspace.path("07_reconstruction", "view_graphs")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{view_graph.view_graph_id}.json"
    output_path.write_text(json.dumps(view_graph.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n")

    report = view_graph.report
    typer.echo(
        f"{OK} ViewGraphManifest créé : {output_path}\n"
        f"    ID   : {view_graph.view_graph_id}\n"
        f"    Nœuds     : {report.images_selected}\n"
        f"    Paires    : {report.valid_pairs}/{report.pairs_tested}\n"
        f"    Composante : {report.largest_component}\n"
        f"    Répétition : {report.repetitive_risk}\n"
        f"    Intrinsèques : {report.intrinsics_quality}"
    )


@reconstruction_app.command("plan")
def reconstruction_plan(
    hotel_id: str = typer.Argument(..., help="Identifiant de l'hôtel."),
) -> None:
    """Sélectionne les backends de reconstruction selon le ViewGraphReport."""
    from .reconstruction_input import prepare_input
    from .reconstruction_plan import ReconstructionPlanner, publish_plan

    try:
        input_manifest, _ = prepare_input(hotel_id)
    except (FileNotFoundError, ValueError, ValidationError) as exc:
        typer.secho(f"✗ {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    workspace = Workspace(hotel_id)
    vg_dir = workspace.path("07_reconstruction", "view_graphs")
    if not vg_dir.exists():
        typer.secho("✗ aucun ViewGraphManifest trouvé", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    view_graphs = sorted(vg_dir.glob("vg-*.json"))
    if not view_graphs:
        typer.secho("✗ aucun ViewGraphManifest trouvé", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    latest = view_graphs[-1]
    view_graph = ViewGraphManifest.model_validate_json(latest.read_text("utf-8"))

    planner = ReconstructionPlanner(workspace)
    plan = planner.plan(input_manifest, view_graph)
    output_path = publish_plan(plan, workspace)

    typer.echo(
        f"{OK} ReconstructionPlan créé : {output_path}\n"
        f"    Backends  : {', '.join(plan.selected_backends)}\n"
        f"    Rationale : {plan.rationale}"
    )


@reconstruction_app.command("run")
def reconstruction_run(
    hotel_id: str = typer.Argument(..., help="Identifiant de l'hôtel."),
    backend: str = typer.Option("colmap_incremental", "--backend", help="Backend à exécuter."),
) -> None:
    """Exécute une reconstruction SfM sur l'entrée gelée."""
    from .reconstruction_input import prepare_input
    from .reconstruction_run import ReconstructionRunner, publish_run

    try:
        input_manifest, _ = prepare_input(hotel_id)
    except (FileNotFoundError, ValueError, ValidationError) as exc:
        typer.secho(f"✗ {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    workspace = Workspace(hotel_id)
    runner = ReconstructionRunner(workspace)

    try:
        from .schemas.reconstruction import ReconstructionBackend
        backend_enum = ReconstructionBackend(backend)
    except ValueError:
        typer.secho(f"✗ backend inconnu : {backend}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    run = runner.run(input_manifest, backend=backend_enum)
    output_path = publish_run(run, workspace)

    status_mark = f"{OK} {run.status}" if run.status == "completed" else f"✗ {run.status}"
    typer.echo(
        f"{status_mark} Run créé : {output_path}\n"
        f"    Run ID  : {run.run_id}\n"
        f"    Backend : {run.backend}\n"
        f"    Métriques: {run.metrics or 'aucune'}"
    )
    if run.error:
        typer.secho(f"    Erreur : {run.error}", fg=typer.colors.YELLOW, err=True)


@reconstruction_app.command("run-all")
def reconstruction_run_all(
    hotel_id: str = typer.Argument(..., help="Identifiant de l'hôtel."),
) -> None:
    """Exécute tous les backends du ReconstructionPlan."""
    from .reconstruction_input import prepare_input
    from .reconstruction_plan import ReconstructionPlanner, publish_plan
    from .reconstruction_run import ReconstructionRunner, publish_run
    from .reconstruction_consensus import ConsensusBuilder, publish_consensus

    try:
        input_manifest, _ = prepare_input(hotel_id)
    except (FileNotFoundError, ValueError, ValidationError) as exc:
        typer.secho(f"✗ {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    workspace = Workspace(hotel_id)
    plans_dir = workspace.path("07_reconstruction", "plans")
    plan_files = sorted(plans_dir.glob("*.json")) if plans_dir.is_dir() else []
    if not plan_files:
        typer.secho("✗ aucun ReconstructionPlan publié — exécutez d'abord reconstruction plan", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    plan_data = json.loads(plan_files[-1].read_text("utf-8"))
    selected_backends = plan_data.get("selected_backends", ["colmap_incremental"])
    temporal_strategy = plan_data.get("temporal_strategy", "current_only")

    from .schemas.reconstruction import ReconstructionBackend
    runner = ReconstructionRunner(workspace)
    completed_runs = []

    for backend_name in selected_backends:
        try:
            be = ReconstructionBackend(backend_name)
        except ValueError:
            typer.secho(f"  · backend inconnu : {backend_name}", fg=typer.colors.YELLOW, err=True)
            continue

        typer.echo(f"  → {backend_name}...")
        run = runner.run(input_manifest, backend=be)
        publish_run(run, workspace)

        status_mark = f"{OK}" if run.status == "completed" else f"✗"
        typer.echo(f"    {status_mark} {run.run_id} : {run.status}")
        if run.error:
            typer.secho(f"      Erreur : {run.error}", fg=typer.colors.YELLOW, err=True)
        if run.status == "completed":
            completed_runs.append(run.run_id)

    if len(completed_runs) >= 2:
        try:
            builder = ConsensusBuilder(workspace)
            consensus = builder.build(completed_runs)
            publish_consensus(consensus, workspace)
            typer.echo(f"\n{OK} Consensus prêt : run sélectionné = {consensus.selected_run_id}")
        except Exception as exc:
            typer.secho(f"\n  · consensus impossible : {exc}", fg=typer.colors.YELLOW, err=True)


@reconstruction_app.command("consensus")
def reconstruction_consensus(
    hotel_id: str = typer.Argument(..., help="Identifiant de l'hôtel."),
    run_ids: list[str] = typer.Option(
        ...,
        "--run-id",
        help="IDs des runs à comparer (au moins deux).",
    ),
) -> None:
    """Compare plusieurs reconstructions et sélectionne la meilleure."""
    from .reconstruction_consensus import ConsensusBuilder, publish_consensus

    if len(run_ids) < 2:
        typer.secho("✗ au moins deux run_ids sont requis", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    workspace = Workspace(hotel_id)
    builder = ConsensusBuilder(workspace)
    try:
        report = builder.build(run_ids)
    except Exception as exc:
        typer.secho(f"✗ consensus échoué : {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    output_path = publish_consensus(report, workspace)
    typer.echo(
        f"{OK} Consensus publié : {output_path}\n"
        f"    Runs comparés   : {', '.join(report.run_ids)}\n"
        f"    Run sélectionné : {report.selected_run_id}\n"
        f"    Rationale       : {report.selection_rationale}"
    )


@reconstruction_app.command("align")
def reconstruction_align(
    hotel_id: str = typer.Argument(..., help="Identifiant de l'hôtel."),
    run_id: str = typer.Option(..., "--run-id", help="ID du run à aligner."),
    anchor: list[str] = typer.Option(
        ["footprint", "lidar_roof"],
        "--anchor",
        help="Ancres d'alignement (répétable).",
    ),
) -> None:
    """Aligne la reconstruction sur les données géospatiales."""
    from .geo_alignment import GeoAligner, publish_alignment
    from .schemas.reconstruction import AlignmentAnchor

    try:
        anchors = [AlignmentAnchor(a) for a in anchor]
    except ValueError:
        typer.secho(f"✗ ancre inconnue : {anchor}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    workspace = Workspace(hotel_id)
    aligner = GeoAligner(workspace)
    manifest = aligner.align(run_id, anchors=anchors)
    output_path = publish_alignment(manifest, workspace)

    typer.echo(
        f"{OK} Alignement publié : {output_path}\n"
        f"    ID      : {manifest.alignment_id}\n"
        f"    RMSE    : {manifest.alignment_rmse_m:.3f} m\n"
        f"    Empreinte: {manifest.footprint_error_m:.3f} m\n"
        f"    Toiture : {manifest.roof_height_error_m:.3f} m"
    )


@reconstruction_app.command("dense")
def reconstruction_dense(
    hotel_id: str = typer.Argument(..., help="Identifiant de l'hôtel."),
    run_id: str = typer.Option(..., "--run-id", help="ID du run sparse sélectionné."),
    backend: str = typer.Option("brush", "--backend", help="Backend dense (brush, gsplat)."),
) -> None:
    """Exécute la reconstruction dense (placeholder P5)."""
    from .dense_reconstruction import run_dense_reconstruction, publish_dense_result
    from .schemas.reconstruction import ReconstructionBackend

    try:
        backend_enum = ReconstructionBackend(backend)
    except ValueError:
        typer.secho(f"✗ backend dense inconnu : {backend}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    workspace = Workspace(hotel_id)
    result = run_dense_reconstruction(workspace, run_id, backend=backend_enum)
    output_path = publish_dense_result(result, workspace)

    status_mark = f"{OK} {result.status}" if result.status == "completed" else f"✗ {result.status}"
    typer.echo(
        f"{status_mark} Résultat dense : {output_path}\n"
        f"    Result ID : {result.result_id}\n"
        f"    Backend   : {result.backend}"
    )
    if result.error:
        typer.secho(f"    Note : {result.error}", fg=typer.colors.YELLOW, err=True)


@reconstruction_app.command("lidar-report")
def reconstruction_lidar_report(
    hotel_id: str = typer.Argument(..., help="Identifiant de l'hôtel."),
) -> None:
    """Génère un rapport de support LiDAR pour la reconstruction."""
    from .reconstruction_input import prepare_input
    from .lidar_support import LiDARSupportAnalyzer, publish_lidar_report

    try:
        input_manifest, _ = prepare_input(hotel_id)
    except (FileNotFoundError, ValueError, ValidationError) as exc:
        typer.secho(f"✗ {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    workspace = Workspace(hotel_id)
    analyzer = LiDARSupportAnalyzer(workspace)
    report = analyzer.analyze(input_manifest)
    output_path = publish_lidar_report(report, workspace)

    typer.echo(
        f"{OK} Rapport LiDAR publié : {output_path}\n"
        f"    ID      : {report.report_id}\n"
        f"    Viable LiDGS : {'Oui' if report.viable_for_lidgs else 'Non'}\n"
        f"    Classification : {report.classification}"
    )


@reconstruction_app.command("validate")
def reconstruction_validate(
    hotel_id: str = typer.Argument(..., help="Identifiant de l'hôtel."),
    run_id: str = typer.Option(..., "--run-id", help="ID du run à valider."),
) -> None:
    """Exécute les validations held-out, stability et cross-solver."""
    from .reconstruction_validation import build_validation_report, publish_validation_report

    workspace = Workspace(hotel_id)
    report = build_validation_report(workspace, run_id)
    output_path = publish_validation_report(report, workspace)

    typer.echo(
        f"{OK} Validation publiée : {output_path}\n"
        f"    ID      : {report.validation_id}\n"
        f"    Statut  : {report.overall_status}"
    )
    if report.held_out and report.held_out.error:
        typer.secho(f"    Note : {report.held_out.error}", fg=typer.colors.YELLOW, err=True)


@reconstruction_app.command("camera-feasibility")
def reconstruction_camera_feasibility(
    hotel_id: str = typer.Argument(..., help="Identifiant de l'hôtel."),
    run_id: str = typer.Option(..., "--run-id", help="ID du run de reconstruction."),
) -> None:
    """Évalue la faisabilité des poses de caméra sur la reconstruction."""
    from .camera_feasibility import CameraFeasibilityEvaluator, build_validated_camera_path, publish_validated_path

    workspace = Workspace(hotel_id)
    path = build_validated_camera_path(workspace, run_id)
    output_path = publish_validated_path(path, workspace)

    high = sum(1 for p in path.poses if p.overall_score >= 0.7)
    medium = sum(1 for p in path.poses if 0.4 <= p.overall_score < 0.7)
    low = sum(1 for p in path.poses if p.overall_score < 0.4)

    typer.echo(
        f"{OK} Trajectoire validée : {output_path}\n"
        f"    Poses : {len(path.poses)}\n"
        f"    Haute confiance : {high}\n"
        f"    Moyenne        : {medium}\n"
        f"    Basse          : {low}"
    )


@reconstruction_app.command("gate")
def reconstruction_gate(
    hotel_id: str = typer.Argument(..., help="Identifiant de l'hôtel."),
) -> None:
    """Gate final : évalue si ENVIRONMENT_3D_READY peut être déclaré."""
    from .schemas.reconstruction import ReconstructionGateStatus
    from .workspace import Workspace

    workspace = Workspace(hotel_id)

    consensus_dir = workspace.path("07_reconstruction", "consensus")
    alignment_dir = workspace.path("07_reconstruction", "alignment")
    confidence_dir = workspace.path("07_reconstruction", "confidence")
    validation_dir = workspace.path("07_reconstruction", "validation")
    camera_path_dir = workspace.path("07_reconstruction", "validated_camera_paths")

    checks = []

    consensus_files = sorted(consensus_dir.glob("*.json")) if consensus_dir.is_dir() else []
    if consensus_files:
        checks.append(("consensus", True, consensus_files[-1].name))
    else:
        checks.append(("consensus", False, "aucun consensus publié"))

    alignment_files = sorted(alignment_dir.glob("*.json")) if alignment_dir.is_dir() else []
    if alignment_files:
        checks.append(("alignement", True, alignment_files[-1].name))
    else:
        checks.append(("alignement", False, "aucun alignement publié"))

    confidence_files = sorted(confidence_dir.glob("*.geojson")) if confidence_dir.is_dir() else []
    if confidence_files:
        checks.append(("confiance surface", True, confidence_files[-1].name))
    else:
        checks.append(("confiance surface", False, "aucune confiance publiée"))

    validation_files = sorted(validation_dir.glob("*.json")) if validation_dir.is_dir() else []
    if validation_files:
        checks.append(("validation", True, validation_files[-1].name))
    else:
        checks.append(("validation", False, "aucune validation publiée"))

    camera_path_files = sorted(camera_path_dir.glob("*.json")) if camera_path_dir.is_dir() else []
    if camera_path_files:
        checks.append(("trajectoire validée", True, camera_path_files[-1].name))
    else:
        checks.append(("trajectoire validée", False, "aucune trajectoire validée"))

    all_passed = all(passed for _, passed, _ in checks)
    status = ReconstructionGateStatus.ENVIRONMENT_3D_READY if all_passed else ReconstructionGateStatus.NEEDS_AUTHORIZED_CAPTURE

    typer.echo(f"Statut : {status.value}")
    for name, passed, detail in checks:
        mark = "✓" if passed else "✗"
        typer.echo(f"  {mark} {name}: {detail}")
def assets_coverage(hotel_id: str = typer.Argument(...)) -> None:
    """Compte ce qui conditionne la suite du pipeline."""
    from .intake import coverage

    manifest = Workspace(hotel_id).read_assets()
    if manifest is None:
        typer.secho("aucun manifeste d'assets", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    for key, value in coverage(manifest).items():
        typer.echo(f"  {key:<26} {value}")


@app.command(name="run-phase1")
def run_phase1(
    hotel_id: str = typer.Argument(...),
    force: bool = typer.Option(False, "--force", help="Rejoue toutes les étapes."),
) -> None:
    """Enchaîne les étapes de la Phase 1 jusqu'au premier arrêt."""
    for name in STEP_ORDER:
        _run_one(hotel_id, name, force)


@app.command()
def smoke() -> None:
    """Smoke test : valide le socle sans réseau ni GPU (acceptation Lot 0)."""
    import tempfile
    from pathlib import Path

    from .schemas import Asset, AssetCategory, AssetManifest, Rights

    typer.echo(f"{OK} paquet importable — version {__version__}")

    check_providers()
    typer.echo(f"{OK} contrôle des fournisseurs exécutable")

    asset = Asset(
        id="smoke-001",
        source="smoke",
        source_url_or_id="local",
        rights=Rights.OWNED,
        ai_eligible=False,
        confidence=1.0,
        category=AssetCategory.FACADE,
        checksum="0" * 64,
        production_eligible=True,
    )
    AssetManifest(hotel_id="smoke", assets=[asset])
    typer.echo(f"{OK} schémas Pydantic valides")

    with tempfile.TemporaryDirectory() as tmp:
        workspace = Workspace("smoke", root=Path(tmp))
        workspace.create()
        workspace.write_manifest(ProjectManifest(hotel_id="smoke", address="—"))
        workspace.read_manifest()
        missing = [d for d in SUBDIRS if not workspace.path(d).is_dir()]
        if missing:
            typer.secho(f"{KO} répertoires manquants : {missing}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1)
    typer.echo(f"{OK} espace de travail créé et relu")

    typer.echo("")
    typer.secho("smoke test réussi", fg=typer.colors.GREEN)


if __name__ == "__main__":
    app()


def _sector_context(workspace, context, demands, geometry):  # noqa: ANN001, ANN201
    """Cibles résolues et projection, par le **même** chemin que la géométrie.

    Aucune définition sectorielle propre à la recherche : une troisième règle
    aurait divergé des deux existantes sans que rien ne le signale, et le plan
    aurait acheté des vues que `demands assess` aurait refusé de compter.
    """
    from .adaptive_search import SectorContext
    from .demand_targets import TargetUnresolved, resolve
    from .geo.projection import ProjectionRefused, ProjectionService

    reference = context.spatial_reference
    if reference is None or geometry is None:
        typer.secho(
            "  · aucun secteur mesurable : contexte spatial ou géométrie de "
            "capture absents — les besoins de secteur ne se distingueront pas",
            fg=typer.colors.YELLOW,
        )
        return None

    spatial = _safe_read(workspace.read_spatial)
    front = getattr(spatial, "front_azimuth_deg", None) if spatial else None
    if front is None:
        typer.secho(
            "  · orientation de façade inconnue : « de face » et « de coin » ne "
            "se départageront pas",
            fg=typer.colors.YELLOW,
        )

    try:
        projection = ProjectionService(reference)
    except ProjectionRefused as exc:
        typer.secho(f"  · projection refusée : {exc}", fg=typer.colors.YELLOW)
        return None

    site = _safe_read(workspace.read_site)
    half_width = context.policy.geometry.sector_observer_half_width_deg

    proxies = _declared_proxies(workspace)
    targets, unresolved, proxy_targets = {}, {}, {}
    for demand in demands:
        try:
            targets[demand.demand_id] = resolve(
                demand, geometry, front, site, half_width
            )
        except TargetUnresolved as exc:
            # La cible manque : le secteur reste inconnu, et le dire vaut mieux
            # que de replier sur le bâtiment — ce qui ferait chercher la façade
            # en croyant chercher l'entrée.
            unresolved[demand.demand_id] = str(exc).split(" : ", 1)[-1]

            # Un proxy déclaré dit **où chercher**, sans établir la cible. La
            # vue trouvée ainsi restera preview-only jusqu'à vérification.
            proxy_ref = proxies.get(demand.demand_id)
            if proxy_ref:
                resolved_proxy = _resolve_proxy(
                    proxy_ref, demand, geometry, front, site, half_width
                )
                if resolved_proxy is not None:
                    proxy_targets[demand.demand_id] = resolved_proxy

    return SectorContext(
        targets=targets,
        projection=projection,
        front_azimuth_deg=front,
        unresolved=unresolved,
        proxies={k: v for k, v in proxies.items() if k in proxy_targets},
        proxy_targets=proxy_targets,
        heading_tolerance_deg=context.policy.adaptive_search.heading_tolerance_deg,
    )


#: Types de façade et secteur qu'ils désignent. Un proxy `FACADE_PRIMARY` dit
#: « cherche du côté avant », non « cherche l'objet FACADE_PRIMARY » — qui n'a
#: pas de géométrie propre au manifeste.
FACADE_SECTORS: dict[str, str] = {
    "FACADE_PRIMARY": "front",
    "FACADE_LEFT": "left",
    "FACADE_RIGHT": "right",
    "FACADE_REAR": "rear",
}


def _declared_proxies(workspace) -> dict:  # noqa: ANN001
    """Proxies de recherche déclarés à la construction des besoins.

    Ils viennent des obligations, non de la recherche : où chercher tant qu'une
    cible manque est une décision de gabarit, pas une improvisation du
    collecteur.
    """
    payload = _latest_json(workspace, "capture_demands_build_*.json")
    return dict((payload or {}).get("search_proxies") or {})


def _resolve_proxy(proxy_ref, demand, geometry, front, site, half_width):  # noqa: ANN001, ANN201
    """Résout le repère approchant, par le même chemin que les vraies cibles."""
    from .demand_targets import TargetUnresolved, resolve
    from .schemas.acquisition import TargetKind

    from .demand_targets import SECTOR_BEARINGS

    # Un proxy de façade nomme un **secteur** (`FACADE_PRIMARY` → `front`) ;
    # le traiter comme un objet de site cherchait une géométrie qui n'existe
    # pas sous ce nom, et le proxy restait silencieusement inactif.
    sector_ref = FACADE_SECTORS.get(proxy_ref)
    if sector_ref is None and proxy_ref in SECTOR_BEARINGS:
        sector_ref = proxy_ref

    stand_in = demand.model_copy(
        update={
            "target_ref": sector_ref or proxy_ref,
            "target_kind": (
                TargetKind.VIEW_SECTOR if sector_ref else TargetKind.SITE_OBJECT
            ),
        }
    )
    try:
        return resolve(stand_in, geometry, front, site, half_width)
    except TargetUnresolved:
        # Le proxy lui-même n'est pas résolu : on ne cherche pas autour d'un
        # repère qu'on ne sait pas placer.
        return None


def _recommendation_levels(candidates):  # noqa: ANN001, ANN201
    """Niveau prononcé par la recherche, **par couple candidat/besoin**.

    Rend `None` quand aucune recherche adaptative n'a produit ce manifeste :
    c'est la compatibilité avec les espaces de travail antérieurs, et elle se
    reconnaît à l'absence de `adaptive_search_run_id` — non à un registre vide,
    qui signifie au contraire « cherché, rien recommandé ».
    """
    if not getattr(candidates, "adaptive_search_run_id", None):
        return None
    return {
        (entry.candidate_id, entry.demand_id): entry.level
        for entry in getattr(candidates, "recommendations", [])
    }


def _validate_manifest_pairing(hotel_id, candidates, demands, demands_payload):  # noqa: ANN001, ANN201
    """Les deux manifestes parlent-ils du même établissement, et des mêmes besoins ?

    Le brouillon vérifié l'a été sur un couple concordant. Rien ne garantissait
    qu'il le reste : `--candidates` désigne un fichier arbitraire, et un
    manifeste produit contre d'anciens besoins aurait planifié des vues pour des
    exigences qui ont changé depuis.
    """
    from .provenance import digest_of

    problems: list[str] = []

    for label, value in (
        ("manifeste de candidats", candidates.hotel_id),
        ("manifeste de besoins", demands.hotel_id),
    ):
        if value != hotel_id:
            problems.append(
                f"{label} : hotel_id {value!r} au lieu de {hotel_id!r}"
            )

    # L'empreinte que la découverte a inscrite doit être celle des besoins
    # d'aujourd'hui. Sinon le plan répond à une question qu'on ne pose plus.
    current = digest_of(demands_payload)
    declared = getattr(candidates, "demand_digest", None)
    if declared and declared != current:
        problems.append(
            f"empreinte des besoins {declared} au manifeste de candidats, "
            f"{current} aujourd'hui : les besoins ont changé depuis la "
            "découverte — relancez « assets discover »"
        )

    # Une découverte ciblée ne juge que les besoins qu'elle a cherchés. Planifier
    # dessus contre les besoins canoniques ferait lire l'absence de vues pour les
    # six autres comme un constat, alors qu'aucune n'a été cherchée.
    scope = getattr(candidates, "scope", None)
    if scope is not None and scope.demand_ids:
        hors_portée = sorted(
            {d.demand_id for d in demands.demands} - set(scope.demand_ids)
        )
        if hors_portée:
            problems.append(
                "manifeste de candidats ciblé sur "
                f"{', '.join(scope.demand_ids)}, mais le plan porte aussi sur "
                f"{', '.join(hors_portée)} : ces besoins n'ont pas été "
                "cherchés, et leur absence de vues n'est pas un constat"
            )
    return problems


def _readable_evaluations(evaluations, candidates, separation_m):  # noqa: ANN001, ANN201
    """Le cadrage à côté du verdict, pour que deux lignes se distinguent.

    Deux évaluations d'un même panorama portaient des verdicts différents sans
    que rien n'explique pourquoi : le cap et le point de vue vivaient dans un
    autre fichier. Les rapprocher ne change aucun verdict — il rend lisible ce
    qui l'était déjà, à condition de joindre deux documents à la main.
    """
    from .plan import group_viewpoints

    by_id = {candidate.candidate_id: candidate for candidate in candidates}
    viewpoints = group_viewpoints(list(by_id.values()), separation_m)

    readable = []
    for evaluation in evaluations:
        row = json.loads(evaluation.model_dump_json())
        candidate = by_id.get(evaluation.candidate_id)
        if candidate is not None:
            row["framing"] = {
                "panorama_id": candidate.panorama_id,
                "viewpoint_id": viewpoints.get(evaluation.candidate_id),
                "requested_heading_deg": candidate.requested_heading_deg,
                "requested_fov_deg": candidate.requested_fov_deg,
                "requested_pitch_deg": candidate.requested_pitch_deg,
            }
        readable.append(row)
    return readable


#: Préfixe de clé de cache → source du manifeste. Les noms diffèrent — le cache
#: distingue `streetview-meta` de `streetview`, le manifeste ne connaît que
#: `street_view` — et les confondre attribuerait des appels à la mauvaise source.
_CACHE_SOURCES: dict[str, str] = {
    "mapillary": "mapillary",
    "streetview-meta": "street_view",
    "streetview": "street_view",
}


def _requests_by_source(by_source: dict) -> dict:
    """Appels **réellement émis** par source du manifeste.

    Vient du registre de transport, où chaque tentative est inscrite avant
    l'appel : la pagination et les échecs y figurent, ce que le décompte des
    cache misses ne voyait pas.

    Une source absente du tableau n'a émis aucun appel — ce qui est une
    information, à distinguer d'un zéro par défaut : les sources non
    interrogées portent déjà leur motif ailleurs.
    """
    from .schemas.acquisition import SourceRequestCounts

    merged: dict[str, dict[str, int]] = {}
    for prefix, row in by_source.items():
        source = _CACHE_SOURCES.get(prefix)
        if source is None:
            # Une clé qui n'appartient à aucune source d'images — géométrie,
            # réseau routier. La compter fausserait le coût de la collecte.
            continue
        stages = merged.setdefault(source, {})
        for stage, count in row.get("by_stage", {}).items():
            stages[stage] = stages.get(stage, 0) + count

    return {
        source: SourceRequestCounts(
            coarse_search=stages.get("coarse_search", 0),
            metadata_enrichment=stages.get("metadata_enrichment", 0),
            sequence_expansion=stages.get("sequence_expansion", 0),
        )
        for source, stages in sorted(merged.items())
    }


def _planned_calls(context, radius_m: int) -> dict:  # noqa: ANN001
    """Plafond d'appels annoncé **avant** d'interroger.

    Mapillary pagine : on ne sait pas d'avance combien de pages viendront, mais
    on connaît le plafond que le collecteur s'impose. Street View interroge un
    point de corridor à la fois, ce qui se compte.

    Un plafond n'est pas une prédiction : c'est ce qu'on s'autorise. Le
    comparer à l'effectif dit si l'estimation valait.
    """
    from .collectors.mapillary import MAX_IMAGES, PAGE_SIZE

    collection = context.policy.collection
    spacing = max(collection.sample_spacing_m, 1.0)
    # Deux fois le rayon divisé par l'espacement : ordre de grandeur du nombre
    # de points d'un réseau routier couvrant la zone.
    corridor_points = int((2 * collection.road_radius_m) / spacing) ** 2 // 4

    # Les clés sont celles du **registre** — `streetview-meta`, non
    # `street_view` : un plafond annoncé sous un autre nom que l'effectif ne se
    # confronte à rien.
    return {
        "mapillary": -(-MAX_IMAGES // PAGE_SIZE),
        "streetview-meta": max(corridor_points, 1),
    }


@policy_app.command("materialise")
def policy_materialise_command(
    hotel_id: str = typer.Argument(..., help="Établissement dont la politique est à rendre explicite."),
) -> None:
    """Inscrit au fichier les seuils que le code comblait, sans rien changer.

    Migration de **représentation** : les valeurs effectives sont celles déjà
    appliquées. Le reçu le prouve — empreinte identique avant et après — plutôt
    que d'en donner l'assurance.
    """
    from datetime import datetime, timezone

    from .policy_materialise import MaterialisationRefused, materialise

    workspace = Workspace(hotel_id)
    policy_path = workspace.path("00_manifest", "pipeline_policy.json")
    if not policy_path.is_file():
        typer.secho(
            f"{KO} aucune politique à {policy_path}", fg=typer.colors.RED, err=True
        )
        raise typer.Exit(code=1)

    # Une transaction laissée sans reçu se reprend **avant** d'en ouvrir une
    # autre : la relancer à l'aveugle empilerait deux mutations sur un état
    # qu'on n'a pas tranché.
    _resume_pending(workspace, policy_path)

    def publish_prepared(payload: dict) -> None:
        # Jamais modifié ensuite : c'est le reçu qui atteste, et son absence
        # qui signale une transaction à reprendre.
        workspace.write_json(
            f"00_manifest/policy_materialisation_{payload['transaction_id']}"
            "_prepared.json",
            payload,
        )

    def publish_committed(receipt) -> None:  # noqa: ANN001
        workspace.write_json(
            f"00_manifest/policy_materialisation_{receipt.transaction_id}"
            "_committed.json",
            receipt.as_dict(),
        )

    try:
        receipt = materialise(
            policy_path,
            publish_receipt=publish_committed,
            publish_prepared=publish_prepared,
        )
    except MaterialisationRefused as exc:
        typer.secho(f"{KO} {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc

    if not receipt.materialised:
        typer.echo(f"{OK} politique déjà complète — rien à matérialiser")
        return

    typer.echo(f"{OK} {len(receipt.materialised)} chemin(s) rendus explicites")
    typer.echo(f"  empreinte    {receipt.digest_before} → {receipt.digest_after}")
    typer.echo(f"  version      {receipt.version_before} (inchangée)")
    typer.echo(f"  fichier      {receipt.sha_before} → {receipt.sha_after}")
    typer.echo("  aucune valeur modifiée ni disparue")


def _resume_pending(workspace, policy_path) -> None:  # noqa: ANN001
    """Tranche les transactions préparées dont le reçu manque.

    L'empreinte du fichier fait foi : elle dit ce qui s'est **passé**, là où une
    date ou un ordre d'écriture ne diraient que ce qu'on espérait. Une reprise
    ne réécrit jamais le manifeste préparé — elle ajoute son constat à côté.
    """
    from .transaction import TransactionConflict, pending, recover

    directory = workspace.path("00_manifest")
    if not directory.is_dir():
        return

    for prepared in pending(directory, "policy_materialisation"):
        try:
            resolution = recover(prepared, policy_path)
        except TransactionConflict as exc:
            typer.secho(f"{KO} {exc}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=2) from exc

        workspace.write_json(
            f"00_manifest/policy_materialisation_"
            f"{prepared['transaction_id']}_committed.json",
            resolution,
        )
        typer.secho(
            f"  · transaction {prepared['transaction_id']} reprise : "
            f"{resolution['state']} — {resolution['resolution']}",
            fg=typer.colors.YELLOW,
        )


@assets_app.command("invalidate-plans")
def assets_invalidate_plans(
    hotel_id: str = typer.Argument(..., help="Établissement concerné."),
    plan_id: list[str] = typer.Option(
        ..., "--plan-id",
        help="Identifiant à invalider. Répétable ; aucun motif générique.",
    ),
    reason: str = typer.Option(
        ..., "--reason", help="Motif structuré, parmi ceux déclarés.",
    ),
    rationale: str = typer.Option(
        ..., "--rationale", help="Ce qu'un relecteur doit comprendre.",
    ),
) -> None:
    """Retire des plans de la circulation sans effacer ce qu'ils disaient.

    Les fichiers restent intacts : l'événement publié les nomme, avec leur
    empreinte, et c'est lui que la sélection consulte.
    """
    from .plan_invalidation import (
        InvalidationReason,
        InvalidationRefused,
        build,
    )
    from .transaction import commit, prepare

    workspace = Workspace(hotel_id)
    sources = workspace.path("01_sources")

    try:
        structured = InvalidationReason(reason)
    except ValueError as exc:
        typer.secho(
            f"{KO} motif {reason!r} inconnu ; déclarés : "
            f"{[r.value for r in InvalidationReason]}",
            fg=typer.colors.RED, err=True,
        )
        raise typer.Exit(code=1) from exc

    paths = [sources / f"acquisition_plan_{identifier}.json" for identifier in plan_id]

    try:
        event = build(paths, structured, rationale)
    except InvalidationRefused as exc:
        typer.secho(f"{KO} {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    # L'événement est le fichier muté : `prepared` décrit ce qu'on s'apprête à
    # publier, `committed` atteste. Un manifeste préparé seul n'invalide rien.
    target = sources / f"plan_invalidation_{event.invalidation_id}_committed.json"
    payload = json.dumps(event.as_dict(state="committed"), indent=2, ensure_ascii=False) + "\n"

    transaction = prepare(
        target, payload, kind="plan_invalidation",
        intent={
            "reason": structured.value,
            "plan_ids": [plan.plan_id for plan in event.plans],
        },
    )

    commit(
        transaction, payload,
        publish_prepared=lambda data: workspace.write_json(
            f"01_sources/plan_invalidation_{event.invalidation_id}_prepared.json",
            data,
        ),
        publish_committed=lambda _data: None,
    )

    typer.echo(f"{OK} {len(event.plans)} plan(s) retirés de la circulation")
    for plan in event.plans:
        typer.echo(f"    {plan.plan_id}  sha={plan.sha256}")
    typer.echo(f"  motif    {structured.value}")
    typer.echo("  fichiers intacts : l'événement les nomme, il ne les efface pas")


@assets_app.command("measure-plan")
def assets_measure_plan(
    hotel_id: str = typer.Argument(..., help="Établissement concerné."),
    plan_file: Path = typer.Option(
        ..., "--plan", help="Plan **existant** à mesurer. Aucune reconstruction.",
    ),
) -> None:
    """Mesure les acquisitions d'un plan déjà arrêté, sans le refaire.

    `assets plan --measure-volumes` reconstruit la sélection : les appels
    autorisés pourraient alors porter sur d'autres candidats que ceux examinés.
    Ici la sélection n'est **jamais** rappelée — on mesure ce que le plan dit,
    ou l'on refuse.

    Le brouillon n'est pas modifié : un plan qui change après examen ne se
    relit plus comme celui qui a été approuvé. Un plan mesuré est publié à
    côté, et le registre de transport avec lui — y compris sur échec partiel,
    car une mesure interrompue a coûté des appels.
    """
    from .acquisition_request import RequestUnresolvable, resolve_all
    from .plan_invalidation import invalidated_plan_ids
    from .providers.transport import ledger, reset_ledger
    from .schemas.acquisition import (
        AcquisitionPlan,
        CandidateManifest,
        PlanStatus,
        VolumeStatus,
    )
    from .volumes import measure

    context = _context(hotel_id, Capability.TARGETED_COLLECTION)
    workspace = Workspace(hotel_id)
    sources = workspace.path("01_sources")

    path = Path(plan_file)
    if not path.is_file():
        typer.secho(f"{KO} plan introuvable : {path}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    plan = AcquisitionPlan.model_validate(json.loads(path.read_text("utf-8")))

    # --- ce qu'on refuse **avant** tout appel ------------------------------
    problems: list[str] = []
    if plan.hotel_id != hotel_id:
        problems.append(f"plan de {plan.hotel_id!r}, non de {hotel_id!r}")
    if plan.plan_id in invalidated_plan_ids(sources):
        problems.append("plan invalidé : il a été retiré de la circulation")
    if plan.status is not PlanStatus.DRAFT:
        problems.append(
            f"statut « {plan.status.value} » : seul un brouillon se mesure — "
            "un plan consenti porte déjà son volume"
        )

    current = _latest_plan(workspace)
    if current is None or current.resolve() != path.resolve():
        problems.append(
            "ce n'est pas le plan courant : mesurer un plan écarté dépenserait "
            "des appels sur ce qui ne sera pas acquis"
        )

    if problems:
        for problem in problems:
            typer.secho(f"{KO} {problem}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)

    try:
        candidates_path, candidates_payload = _candidate_manifest_for_plan(
            workspace, plan
        )
    except (FileNotFoundError, ValueError) as exc:
        typer.secho(f"{KO} {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc
    candidates = {
        c.candidate_id: c
        for c in CandidateManifest.model_validate(
            candidates_payload
        ).candidates
    }

    # Les requêtes sont **recalculées** puis confrontées à celles du plan : si
    # elles diffèrent, le plan décrit autre chose que ce qu'on s'apprête à
    # mesurer, et mesurer d'abord pour s'en apercevoir ensuite serait payer
    # pour rien.
    try:
        requests_by_candidate = resolve_all(candidates, plan.acquisitions)
    except RequestUnresolvable as exc:
        typer.secho(f"{KO} {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc

    diverged = [
        acquisition.candidate_id
        for acquisition in plan.acquisitions
        if acquisition.request_digest
        and acquisition.candidate_id in requests_by_candidate
        and requests_by_candidate[acquisition.candidate_id].digest
        != acquisition.request_digest
    ]
    if diverged:
        typer.secho(
            f"{KO} requête(s) différentes de celles du plan : {sorted(diverged)} — "
            "aucun appel émis",
            fg=typer.colors.RED, err=True,
        )
        raise typer.Exit(code=2)

    ordered = [
        requests_by_candidate[a.candidate_id]
        for a in plan.acquisitions
        if a.candidate_id in requests_by_candidate
    ]
    typer.echo(f"  {len(ordered)} acquisition(s) à mesurer, sélection inchangée")

    registry = reset_ledger()
    # Le plafond est calculé **avant** le premier appel, depuis les requêtes
    # résolues : l'annoncer après l'exécution ne serait pas un budget mais un
    # constat. Mapillary coûte deux opérations par mesure — résolution
    # d'adresse, puis CDN — Street View une seule.
    registry.planned_max_requests.update(_head_budget(ordered))
    typer.echo(
        "    budget annoncé : "
        + ", ".join(
            f"{source} {count}"
            for source, count in sorted(registry.planned_max_requests.items())
        )
    )
    report = None
    try:
        report = measure(ordered)
    finally:
        # Publié **même sur échec** : une mesure interrompue a coûté des
        # appels, et les taire donnerait à croire qu'elle n'a rien consommé.
        stamp = _new_run_id()
        workspace.write_json(
            f"01_sources/volume_measure_{plan.plan_id}_{stamp}.json",
            {
                "plan_id": plan.plan_id,
                "measured_at": stamp,
                "requests": [request.as_dict() for request in ordered],
                "volumes": report.as_dict() if report else None,
                "transport": registry.as_dict(),
                "note": (
                    "mesure d'un plan **existant** : la sélection n'a pas été "
                    "rejouée, et le brouillon n'est pas modifié"
                ),
            },
        )

    for source, counts in sorted(registry.by_source().items()):
        typer.echo(
            f"    {source:<16} {counts['logical_operations']} opération(s) "
            f"logique(s), {counts['attempted']} échange(s) HTTP"
        )

    if report.unmeasured:
        typer.secho(f"  {len(report.unmeasured)} taille(s) inconnue(s) :",
                    fg=typer.colors.YELLOW)
        for candidate_id, reason in sorted(report.unmeasured.items()):
            typer.secho(f"    {candidate_id[-22:]} — {reason[:60]}",
                        fg=typer.colors.YELLOW)

    # Un plan **mesuré**, à côté du brouillon : celui-ci n'est pas modifié.
    acquisitions = [
        a.model_copy(update={"expected_bytes": report.measured.get(a.candidate_id)})
        for a in plan.acquisitions
    ]
    unknown = [a for a in acquisitions if a.expected_bytes is None]
    total = sum(a.expected_bytes for a in acquisitions if a.expected_bytes is not None)

    # Inscrits, non seulement calculés : le consentement doit reposer sur un
    # artefact autonome, lisible sans rejouer l'addition.
    measured = plan.model_copy(update={
        "plan_id": f"{plan.plan_id}-measured-{stamp}",
        "acquisitions": acquisitions,
        "published_known_bytes": total,
        "published_unknown_size_items": len(unknown),
        "published_volume_status": (
            VolumeStatus.EXACT if not unknown
            else (
                VolumeStatus.UNKNOWN if len(unknown) == len(acquisitions)
                else VolumeStatus.PARTIAL
            )
        ),
    })
    workspace.write_json(
        f"01_sources/acquisition_plan_{measured.plan_id}.json",
        json.loads(measured.model_dump_json()),
    )

    typer.echo("")
    typer.echo(
        f"{OK} {len(report.measured)} taille(s) mesurée(s), "
        f"{report.known_bytes} octets — statut {measured.volume_status.value}"
    )
    typer.echo(f"    plan mesuré : {measured.plan_id}")
    typer.echo(f"    brouillon d'origine intact : {plan.plan_id}")


def _head_budget(requests) -> dict:  # noqa: ANN001
    """Opérations logiques qu'un `HEAD` coûtera, source par source.

    Mapillary ne publie pas d'URL durable : mesurer une de ses vues demande
    d'abord une résolution d'adresse, puis le `HEAD` sur le CDN — deux
    opérations. Street View sert l'adresse directement.

    Calculé depuis les requêtes **résolues**, avant tout appel : un plafond
    annoncé après coup ne serait pas un budget mais un constat.
    """
    from .collectors.mapillary import name as mapillary_name

    budget: dict[str, int] = {}
    for request in requests:
        cost = 2 if request.source == mapillary_name else 1
        budget[request.source] = budget.get(request.source, 0) + cost
    return budget


# --- fonctions de découverte ciblée (restaurées pour les tests) -----------------


def _discovery_scope(demands, selected: list[str], demand_digest: str):  # noqa: ANN001, ANN201
    """Arrête ce que cette découverte couvrira, **avant** d'interroger quoi que ce soit.

    Valider après un appel réseau aurait déjà émis la requête, dépensé le quota
    et écrit une trace pour un besoin qui n'existe pas. Un identifiant inconnu
    doit donc être refusé ici, sans cache ni réseau.

    Le manifeste rendu ne porte que les besoins retenus : filtrer plus loin
    laisserait les autres être interrogés « pour rien », et le rapport dirait
    qu'on les a cherchés.
    """
    from .schemas.acquisition import (
        DiscoveryMode,
        DiscoveryScope,
        TargetKind,
    )

    if not selected:
        return DiscoveryScope(demand_manifest_digest=demand_digest), demands

    connus = {d.demand_id for d in demands.demands}
    inconnus = sorted(set(selected) - connus)
    if inconnus:
        typer.secho(
            f"{KO} besoin(s) inconnu(s) du manifeste canonique : "
            f"{', '.join(inconnus)}",
            fg=typer.colors.RED, err=True,
        )
        typer.secho(
            "  aucune source n'a été interrogée : chercher pour un besoin qui "
            "n'existe pas dépenserait un quota et écrirait une trace que rien "
            "ne justifie",
            fg=typer.colors.RED, err=True,
        )
        raise typer.Exit(code=2)

    retenus = [d for d in demands.demands if d.demand_id in set(selected)]
    corridors = sorted({
        d.target_ref for d in retenus
        if d.target_kind is TargetKind.CONTEXT_CORRIDOR
    })

    scope = DiscoveryScope(
        mode=DiscoveryMode.TARGETED,
        demand_ids=tuple(sorted(d.demand_id for d in retenus)),
        demand_manifest_digest=demand_digest,
        corridor_ref=corridors[0] if len(corridors) == 1 else "",
    )
    restreint = demands.model_copy(update={"demands": retenus})
    return scope, restreint


def _targeted_manifest(workspace, manifest, scope):  # noqa: ANN001, ANN201
    """Écarte ce qu'un aperçu a déjà réfuté, et borne le reste à l'aperçu."""
    from .schemas.preview import PreviewAssessmentLog

    payload = workspace.read_json("01_sources/preview_assessments.json") or {}
    log = PreviewAssessmentLog.model_validate(payload) if payload else None

    réfutés: set[str] = set()
    if log is not None:
        for demand_id in scope.demand_ids:
            réfutés |= log.refuted_for(demand_id)

    gardés = [c for c in manifest.candidates if c.candidate_id not in réfutés]
    restants = {c.candidate_id for c in gardés}
    return manifest.model_copy(update={
        "candidates": gardés,
        "evaluations": [
            e for e in manifest.evaluations if e.candidate_id in restants
        ],
        "recommendations": [
            r for r in manifest.recommendations if r.candidate_id in restants
        ],
        "recommended_for_enrichment": [
            c for c in manifest.recommended_for_enrichment if c in restants
        ],
        "recommended_for_preview": [
            c for c in manifest.recommended_for_preview if c in restants
        ],
        "eligible_for_full_acquisition": [],
    })


def _query_sources(profile, context, workspace, radius_m: int, demands, scope=None) -> dict:  # noqa: ANN001
    """Interroge les index disponibles, et dit pourquoi les autres ne le sont pas."""
    from .collectors import mapillary
    from .discover import candidates_from
    from .providers.cache import OfflineError

    queries: dict = {}

    try:
        images = mapillary.collect(profile.lat, profile.lon, radius_m=radius_m)
    except OfflineError as exc:
        queries["mapillary"] = f"hors ligne : {exc}"
    except (requests.RequestException, RuntimeError, ValueError) as exc:
        queries["mapillary"] = f"interrogation impossible : {exc}"
    else:
        queries["mapillary"] = candidates_from("mapillary", images)

    queries["street_view"] = _street_view_candidates(
        context, workspace, radius_m, demands, scope
    )
    return queries


def _street_view_candidates(context, workspace, radius_m: int, demands, scope=None):  # noqa: ANN001, ANN201
    """Panoramas des corridors proches, cadrés vers ce que les besoins demandent."""
    from .geo.geometry_loader import LegacyManifestRefused, load_capture_geometry

    reference = context.spatial_reference
    geometry_path = workspace.path("06_geo", "capture_geometry.json")
    if reference is None or not geometry_path.is_file():
        return (
            "aucune géométrie de capture : les corridors où chercher des "
            "panoramas ne sont pas résolus"
        )

    try:
        manifest, _ = load_capture_geometry(geometry_path, reference)
    except LegacyManifestRefused as exc:
        return f"géométrie illisible : {exc}"

    corridor_ref = getattr(scope, "corridor_ref", "") if scope else ""
    corridors = _corridor_elements(manifest, corridor_ref)
    if not corridors:
        return "aucun corridor résolu autour du site"

    if corridor_ref:
        typer.secho(
            f"    street_view · échantillonnage borné au corridor "
            f"{corridor_ref}",
            fg=typer.colors.DIM,
        )

    from .demand_targets import TargetUnresolved, resolve

    spatial = _safe_read(workspace.read_spatial)
    front = getattr(spatial, "front_azimuth_deg", None) if spatial else None
    targets = []
    for demand in demands:
        try:
            targets.append(
                resolve(demand, manifest, front, _safe_read(workspace.read_site))
            )
        except TargetUnresolved:
            continue
    if not targets:
        return "aucune cible de besoin résolue : rien vers quoi cadrer"

    from .collectors.streetview_v2 import (
        candidate_from, discover_panoramas, framings_for_targets,
    )
    try:
        panoramas, skipped = discover_panoramas(
            corridors,
            spacing_m=context.policy.collection.sample_spacing_m,
            snap_radius_m=context.policy.collection.snap_radius_m,
        )
    except CorridorScopeRefused as exc:
        return f"portée de corridor refusée : {exc}"

    targets_wkt = [t.wkt for t in targets if getattr(t, "wkt", None)]
    framings = framings_for_targets(panoramas, targets_wkt, front)
    return [candidate_from(p, f) for p, f in zip(panoramas, framings)]


class CorridorScopeRefused(RuntimeError):
    """La référence de corridor ne désigne pas exactement une géométrie."""


def _corridor_elements(manifest, corridor_ref: str = "") -> list[dict]:  # noqa: ANN001
    """Corridors résolus, éventuellement bornés à une référence."""
    if not corridor_ref:
        return manifest.corridors
    return [c for c in manifest.corridors if c.get("feature_id") == corridor_ref]


# --- CLI commands restored ----------------------------------------------------


@site_app.command("resolve")
def site_resolve(
    hotel_id: str = typer.Argument(..., help="Établissement concerné."),
    kind: str = typer.Option(..., "--kind", help="Type d'objet à résoudre."),
    state: str = typer.Option(
        "confirmed", "--state", help="confirmed ou inferred.",
    ),
    demand_id: str = typer.Option(
        None, "--demand",
        help="Besoin dont les constats établis fondent la résolution.",
    ),
    rationale: str = typer.Option(..., "--rationale", help="Ce que la preuve établit."),
    decided_by: str = typer.Option(..., "--by", help="Qui décide."),
    evidence: list[str] = typer.Option(
        None, "--evidence", help="Pièce à l'appui. Répétable.",
    ),
) -> None:
    """Résout un objet de site quand une preuve l'établit."""
    from datetime import datetime, timezone

    from .schemas.enums import ObjectState

    workspace = Workspace(hotel_id)
    site = workspace.read_site()
    if site is None:
        typer.secho(f"{KO} aucun manifeste de site", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    cible = [obj for obj in site.objects if obj.kind == kind]
    if not cible:
        typer.secho(f"{KO} aucun objet de type {kind!r}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    try:
        voulu = ObjectState(state)
    except ValueError as exc:
        typer.secho(f"{KO} état {state!r} inconnu", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc
    if voulu not in (ObjectState.CONFIRMED, ObjectState.INFERRED):
        typer.secho(
            f"{KO} seuls 'confirmed' et 'inferred' se décident ici ; "
            "'unresolved' relève de « site unresolve »",
            fg=typer.colors.RED, err=True,
        )
        raise typer.Exit(code=2)

    établis: list[str] = []
    if voulu is ObjectState.CONFIRMED:
        besoin = demand_id or f"obligation:{kind}"
        log = workspace.read_previews()
        établis = sorted(log.established_for(besoin)) if log else []
        if not établis:
            typer.secho(
                f"{KO} aucun constat établi sur {besoin} : « confirmé » sans "
                "preuve d'aperçu serait une conviction, pas un fait",
                fg=typer.colors.RED, err=True,
            )
            raise typer.Exit(code=2)

    stamp = datetime.now(timezone.utc).isoformat()
    pièces = list(evidence or []) + [f"preview:{a}" for a in établis]
    touchés: list[str] = []
    enrichis: list[str] = []
    for obj in cible:
        déjà = obj.state is voulu
        nouvelles = sorted(set(pièces) - set(obj.evidence))
        if déjà and not nouvelles:
            continue
        obj.evidence = sorted({*obj.evidence, *pièces})
        if déjà:
            enrichis.append(obj.object_id)
            continue
        obj.state = voulu
        obj.unresolved_reason = None
        if voulu is ObjectState.CONFIRMED:
            obj.confirmed_by = decided_by
            obj.confirmed_at = datetime.now(timezone.utc)
            obj.confirmation_rationale = rationale
        touchés.append(obj.object_id)

    if not touchés and not enrichis:
        typer.echo(f"{OK} {kind} déjà {voulu.value} — rien à faire")
        return

    if enrichis and not touchés:
        workspace.write_site(site)
        workspace.write_json(
            f"00_manifest/site_evidence_{kind}_{_new_run_id()}.json",
            {
                "kind": kind,
                "objects": enrichis,
                "state": voulu.value,
                "evidence_added": pièces,
                "rationale": rationale,
                "decided_by": decided_by,
                "decided_at": stamp,
                "note": "preuve ajoutée sans changement d'état",
            },
        )
        typer.secho(f"{OK} {kind} : preuve ajoutée, état inchangé", fg=typer.colors.GREEN)
        for identifiant in enrichis:
            typer.echo(f"    {identifiant}")
        return

    workspace.write_site(site)
    workspace.write_json(
        f"00_manifest/site_resolve_{kind}_{_new_run_id()}.json",
        {
            "kind": kind,
            "objects": touchés,
            "state": voulu.value,
            "established_previews": établis,
            "evidence": pièces,
            "rationale": rationale,
            "decided_by": decided_by,
            "decided_at": stamp,
            "note": (
                "résolution fondée sur des constats d'aperçu établis ; aucune "
                "géométrie n'a été inventée ici"
            ),
        },
    )
    typer.secho(
        f"{OK} {len(touchés)} objet(s) résolu(s), {len(enrichis)} enrichi(s)",
        fg=typer.colors.GREEN,
    )


@site_app.command("unresolve")
def site_unresolve(
    hotel_id: str = typer.Argument(..., help="Établissement concerné."),
    kind: str = typer.Option(..., "--kind", help="Type d'objet à dé-résoudre."),
    rationale: str = typer.Option(..., "--rationale", help="Pourquoi on retire cette résolution."),
    decided_by: str = typer.Option(..., "--by", help="Qui décide."),
) -> None:
    """Dé-résout un objet quand une preuve nouvelle contredit la résolution antérieure."""
    from datetime import datetime, timezone

    from .demand_targets import OBJECT_KIND_ROLES

    workspace = Workspace(hotel_id)
    site = workspace.read_site()
    if site is None:
        typer.secho(f"{KO} aucun manifeste de site", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    cible = [obj for obj in site.objects if obj.kind == kind]
    if not cible:
        typer.secho(f"{KO} aucun objet de type {kind!r}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    for obj in cible:
        if obj.state.value in ("confirmed", "inferred"):
            obj.state = ObjectState.UNRESOLVED
            obj.unresolved_reason = rationale
            obj.confirmed_by = None
            obj.confirmed_at = None
            obj.confirmation_rationale = None

    _stale_geometry_for(workspace, kind, rationale)

    workspace.write_site(site)
    typer.secho(
        f"{OK} objet(s) dé-résolu(s) : "
        "ce qui en dérivait est périmé",
        fg=typer.colors.GREEN,
    )


def _stale_geometry_for(workspace, kind: str, rationale: str) -> list[str]:  # noqa: ANN001
    """Marque `stale` la géométrie qui porte l'objet dé-résolu."""
    from .demand_targets import OBJECT_KIND_ROLES

    role = OBJECT_KIND_ROLES.get(kind)
    if role is None:
        return []

    payload = workspace.read_json("06_geo/capture_geometry.json")
    if not payload:
        return []

    touched: list[str] = []
    for geometry in payload.get("geometries", []):
        if geometry.get("role") != role.value:
            continue
        if geometry.get("resolution_status") == "stale":
            continue
        geometry["resolution_status"] = "stale"
        geometry["stale_reason"] = rationale
        geometry["unresolved_reason"] = rationale
        for shape in ("wgs84_wkt", "projected_wkt", "source_wkt", "wkt"):
            geometry.pop(shape, None)
        touched.append(geometry.get("feature_id", "?"))

    if touched:
        workspace.write_json("06_geo/capture_geometry.json", payload)
    return touched


preview_app = typer.Typer(
    no_args_is_help=True,
    help="Constats d'aperçu : ce qu'une vue téléchargée établit, besoin par besoin.",
)
assets_app.add_typer(preview_app, name="preview")


@preview_app.command("list")
def preview_list(
    hotel_id: str = typer.Argument(..., help="Établissement concerné."),
) -> None:
    """Couples asset/besoin en attente de constat."""
    from .schemas.preview import PreviewVerdict

    workspace = Workspace(hotel_id)
    manifest = workspace.read_assets()
    if manifest is None:
        typer.secho(f"{KO} aucun manifeste d'assets", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    log = workspace.read_previews()
    rows: list[tuple[str, str, str, str]] = []
    for asset in manifest.assets:
        provenance = getattr(asset, "acquisition", None)
        for demand_id in getattr(provenance, "serves_demands", None) or []:
            latest = log.latest_for(asset.id, demand_id) if log else None
            rows.append((
                asset.id, demand_id,
                (provenance.demand_levels or {}).get(demand_id, "—"),
                latest.verdict.value if latest else "en attente",
            ))

    if not rows:
        typer.echo("  aucun couple rattaché : rien n'a encore été acquis pour un besoin")
        return

    typer.echo(f"  {len(rows)} couple(s) asset/besoin")
    en_attente = 0
    for asset_id, demand_id, level, verdict in sorted(rows):
        if verdict == "en attente":
            en_attente += 1
        typer.echo(
            f"    {asset_id[-24:]:<26} {demand_id:<34} {level[:24]:<26} {verdict}"
        )
    typer.echo("")
    typer.echo(f"  {en_attente} en attente de constat")


@preview_app.command("assess")
def preview_assess(
    hotel_id: str = typer.Argument(..., help="Établissement concerné."),
    asset_id: str = typer.Option(..., "--asset", help="Asset à évaluer."),
    demand_id: str = typer.Option(..., "--demand", help="Besoin évalué."),
    verdict: str = typer.Option(..., "--verdict", help="established, refuted, inconclusive."),
    rationale: str = typer.Option(..., "--rationale", help="Pourquoi ce verdict."),
    assessed_by: str = typer.Option(..., "--by", help="Qui évalue."),
) -> None:
    """Dépose un constat d'aperçu sur un couple asset/besoin."""
    from .schemas.enums import PreviewVerdict
    from .schemas.preview import PreviewAssessment

    try:
        verdict_enum = PreviewVerdict(verdict)
    except ValueError as exc:
        typer.secho(f"{KO} verdict {verdict!r} inconnu", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc

    workspace = Workspace(hotel_id)
    log = workspace.read_previews() or PreviewAssessmentLog(entries=[])
    log.entries.append(PreviewAssessment(
        asset_id=asset_id,
        demand_id=demand_id,
        verdict=verdict_enum,
        rationale=rationale,
        assessed_by=assessed_by,
    ))
    workspace.write_json("01_sources/preview_assessments.json", json.loads(log.model_dump_json()))
    typer.secho(f"{OK} constat déposé", fg=typer.colors.GREEN)


@preview_app.command("assess-corpus")
def preview_assess_corpus(
    hotel_id: str = typer.Argument(...),
    asset_id: str = typer.Option(..., "--asset", help="Asset du corpus examiné."),
    demand_id: str = typer.Option(..., "--demand", help="Besoin **précis** jugé."),
    verdict: str = typer.Option(
        ..., "--verdict", help="established | refuted | inconclusive",
    ),
    rationale: str = typer.Option(
        ..., "--rationale", help="Ce qu'un relecteur doit comprendre.",
    ),
    assessed_by: str = typer.Option(..., "--by", help="Qui prononce ce constat."),
    in_frame: float | None = typer.Option(None, "--in-frame"),
    projected_width: float | None = typer.Option(None, "--projected-width"),
    visible: float | None = typer.Option(None, "--visible"),
    unmeasured: list[str] = typer.Option(None, "--unmeasured"),
    evidence: list[str] = typer.Option(None, "--evidence"),
) -> None:
    """Inscrit un constat sur un asset **déjà présent au corpus**."""
    from .schemas.preview import PreviewAssessment, PreviewVerdict

    workspace = Workspace(hotel_id)
    manifest = workspace.read_assets()
    asset = next(
        (a for a in (manifest.assets if manifest else []) if a.id == asset_id), None
    )
    if asset is None:
        typer.secho(f"{KO} asset {asset_id!r} absent du manifeste",
                    fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    if getattr(asset, "acquisition", None) is not None:
        typer.secho(
            f"{KO} {asset_id} vient d'un plan ciblé — utilisez « preview assess », "
            "qui vérifie le besoin réellement servi",
            fg=typer.colors.RED, err=True,
        )
        raise typer.Exit(code=2)

    if not asset.checksum:
        typer.secho(
            f"{KO} {asset_id} sans empreinte : le constat ne pourrait pas être "
            "rattaché au fichier examiné",
            fg=typer.colors.RED, err=True,
        )
        raise typer.Exit(code=2)

    try:
        assessment = PreviewAssessment(
            asset_id=asset_id,
            demand_id=demand_id,
            plan_id=f"corpus:{asset.source}",
            request_digest=f"corpus:{asset.checksum[:16]}",
            checksum=asset.checksum,
            target_ref=demand_id.split(":", 1)[-1],
            in_frame_fraction=in_frame,
            projected_width_fraction=projected_width,
            visible_fraction=visible,
            verdict=PreviewVerdict(verdict),
            unmeasured=list(unmeasured or []),
            rationale=rationale,
            assessed_by=assessed_by,
            evidence=list(evidence or []),
        )
    except ValueError as exc:
        typer.secho(f"{KO} {str(exc).split('Value error, ')[-1].split(' [type')[0]}",
                    fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc

    workspace.append_preview(assessment)
    typer.echo(f"{OK} constat corpus inscrit : {asset_id[-22:]} / {demand_id}")
    typer.echo(f"    verdict     {assessment.verdict.value}")
    typer.echo(f"    filiation   {assessment.plan_id} (non commandé pour ce besoin)")
    if assessment.unmeasured:
        typer.echo(f"    inconnu     {', '.join(assessment.unmeasured)}")
    typer.echo("    ce constat ne promeut rien : demands assess le lira")


coverage_app = typer.Typer(
    no_args_is_help=True,
    help="Contexte, contraintes caméra et couverture finale du Lot 1B.",
)
app.add_typer(coverage_app, name="coverage")


@coverage_app.command("build")
def coverage_build(hotel_id: str = typer.Argument(...)) -> None:
    """Produit les livrables finaux sans muter les vérités amont."""
    from .lot1b_coverage import build

    workspace = Workspace(hotel_id)
    try:
        outputs = build(workspace)
    except (FileNotFoundError, ValueError) as exc:
        typer.secho(f"{KO} {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    typer.secho(f"{OK} couverture Lot 1B publiée", fg=typer.colors.GREEN)
    for name, path in outputs.items():
        typer.echo(f"  {name:<20} {path}")


sources_app = typer.Typer(
    no_args_is_help=True,
    help="Registre des sources : ce qui a été interrogé, et pourquoi les autres ne le sont pas.",
)
app.add_typer(sources_app, name="sources")


@sources_app.command("registry")
def sources_registry(hotel_id: str = typer.Argument(...)) -> None:
    """Affiche le registre des sources pour cet hôtel."""
    from .source_registry import SourceRegistry, build as build_registry

    workspace = Workspace(hotel_id)
    try:
        path = build_registry(workspace)
    except FileNotFoundError as exc:
        typer.secho(f"{KO} corpus incomplet : {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    registry = SourceRegistry.model_validate_json(path.read_text("utf-8"))
    typer.echo(f"  familles requises : {registry.required_families}")
    typer.echo(f"  familles closes : {registry.closed_families}")
    for family in registry.families:
        status = "✓" if family.closed else "✗"
        typer.echo(f"    {status} {family.name} : {family.status}")


@sources_app.command("unavailable")
def sources_unavailable(
    hotel_id: str = typer.Argument(...),
    family: str = typer.Argument(..., help="Famille de sources indisponible."),
    reason: str = typer.Option(..., "--reason", help="Pourquoi cette famille est absente."),
    by: str = typer.Option(..., "--by", help="Qui enregistre cette indisponibilité."),
) -> None:
    """Enregistre qu'une famille de sources est indisponible."""
    workspace = Workspace(hotel_id)
    payload = workspace.read_json("01_sources/source_registry.json") or {}
    families = payload.get("families", [])
    for fam in families:
        if fam.get("name") == family:
            fam["status"] = "unavailable"
            fam["unavailable_reason"] = reason
            fam["unavailable_by"] = by
            break
    else:
        families.append({
            "name": family,
            "status": "unavailable",
            "unavailable_reason": reason,
            "unavailable_by": by,
        })
    payload["families"] = families
    workspace.write_json("01_sources/source_registry.json", payload)
    typer.secho(f"{OK} famille {family} marquée indisponible", fg=typer.colors.GREEN)


@sources_app.command("queried")
def sources_queried(
    hotel_id: str = typer.Argument(...),
) -> None:
    """Liste les sources déjà interrogées pour cet hôtel."""
    workspace = Workspace(hotel_id)
    ledger_path = workspace.path("01_sources", "query_ledger.json")
    if not ledger_path.is_file():
        typer.echo("  aucune source interrogée")
        return
    ledger = json.loads(ledger_path.read_text("utf-8"))
    for source, count in sorted(ledger.get("by_source", {}).items()):
        typer.echo(f"    {source}: {count} requête(s)")


@sources_app.command("reopen")
def sources_reopen(
    hotel_id: str = typer.Argument(...),
    family: str = typer.Argument(..., help="Famille à rouvrir."),
) -> None:
    """Marque une famille de sources comme disponible à nouveau."""
    workspace = Workspace(hotel_id)
    payload = workspace.read_json("01_sources/source_registry.json") or {}
    families = payload.get("families", [])
    for fam in families:
        if fam.get("name") == family:
            fam["status"] = "open"
            fam.pop("unavailable_reason", None)
            fam.pop("unavailable_by", None)
            break
    payload["families"] = families
    workspace.write_json("01_sources/source_registry.json", payload)
    typer.secho(f"{OK} famille {family} rouverte", fg=typer.colors.GREEN)


scene_app = typer.Typer(
    no_args_is_help=True,
    help="Paquet de scène 3D : construction et inspection.",
)
app.add_typer(scene_app, name="scene")


@scene_app.command("build")
def scene_build(hotel_id: str = typer.Argument(...)) -> None:
    """Produit le volume, les rasters, le verdict et le paquet de prompts."""
    from .scene_package import build

    workspace = Workspace(hotel_id)
    try:
        outputs = build(workspace)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        typer.secho(f"{KO} {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    typer.secho(f"{OK} paquet 3D hybride publié", fg=typer.colors.GREEN)
    for name, path in outputs.items():
        typer.echo(f"  {name:<12} {path}")
    typer.secho(
        "  verdict Phase 1 : NEEDS_AUTHORIZED_CAPTURE — ce paquet n'est pas "
        "ENVIRONMENT_3D_READY",
        fg=typer.colors.YELLOW,
    )


def _candidate_manifest_reference(workspace, path: Path) -> str:  # noqa: ANN001
    """Référence portable et confinée du manifeste réellement planifié."""
    resolved = Path(path).resolve()
    try:
        relative = resolved.relative_to(workspace.root.resolve())
    except ValueError as exc:
        raise ValueError(
            "le manifeste de candidats doit vivre dans le workspace pour être "
            "rejouable ; copiez-le sous 01_sources/ avant de planifier"
        ) from exc
    reference = relative.as_posix()
    from .schemas.acquisition import AcquisitionPlan

    probe = AcquisitionPlan(
        plan_id="candidate-reference-check",
        hotel_id=workspace.hotel_id,
        candidate_manifest_ref=reference,
    )
    return str(probe.candidate_manifest_ref)


def _candidate_manifest_for_plan(workspace, plan):  # noqa: ANN001, ANN201
    """Relit le manifeste exact d'un plan et confronte son empreinte."""
    from .provenance import digest_of

    if plan.candidate_manifest_ref:
        path = (workspace.root / plan.candidate_manifest_ref).resolve()
        try:
            path.relative_to(workspace.root.resolve())
        except ValueError as exc:
            raise ValueError("référence de candidats hors workspace") from exc
    else:
        path = _latest_candidates(workspace)
        if path is None:
            raise FileNotFoundError("aucun manifeste de candidats")

    if not path.is_file():
        raise FileNotFoundError(f"manifeste de candidats du plan absent : {path}")
    payload = json.loads(path.read_text("utf-8"))
    actual = digest_of(payload)
    if plan.candidate_manifest_digest and actual != plan.candidate_manifest_digest:
        raise ValueError(
            "empreinte du manifeste de candidats divergente : "
            f"{actual} != {plan.candidate_manifest_digest} — aucun appel émis"
        )
    return path, payload


def _activate_latest_demands(workspace) -> str:  # noqa: ANN001
    """Fait du dernier manifeste de besoins le manifeste **canonique**."""
    found = sorted(
        (path for path in workspace.path("01_sources").glob("capture_demands_*.json")
         if "build" not in path.name),
        key=lambda path: path.stat().st_mtime,
    )
    if not found:
        return "aucun manifeste horodaté — canonique inchangé"

    latest = found[-1]
    canonical = workspace.path("01_sources", "capture_demands.json")
    payload = json.loads(latest.read_text("utf-8"))
    canonical.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", "utf-8"
    )
    return f"{len(payload.get('demands', []))} besoin(s) depuis {latest.name}"


def _plan_demands_for_scope(candidates, demands):  # noqa: ANN001, ANN201
    """Applique au plan la portée déjà validée par la découverte."""
    scope = getattr(candidates, "scope", None)
    demand_ids = tuple(getattr(scope, "demand_ids", ()) or ())
    if not demand_ids:
        return demands

    by_id = {row.demand_id: row for row in demands.demands}
    unknown = sorted(set(demand_ids) - set(by_id))
    if unknown:
        raise ValueError(
            "portée de candidats vers des besoins absents du manifeste canonique : "
            + ", ".join(unknown)
        )
    selected = [by_id[demand_id] for demand_id in demand_ids]
    return demands.model_copy(update={"demands": selected})
