"""CLI du pipeline (plan directeur §18).

Typer fournit la CLI, un Makefile orchestre le PoC. Chaque commande est
rejouable, détecte un résultat existant, et n'expose aucun secret.
"""

from __future__ import annotations

from pathlib import Path

import requests
import typer

from . import __version__, logging as pipeline_logging
from .config import check_providers, load_env
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
    typer.echo(f"{OK} espace de travail créé : {workspace.root}")
    typer.echo(f"  {len(SUBDIRS)} répertoires, manifeste initialisé")
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


def _context(hotel_id: str) -> PipelineContext:
    """Politique et profil de l'établissement, chargés une seule fois.

    La politique vient de l'espace de travail : la chercher dans le répertoire
    courant faisait dépendre le résultat du lieu d'exécution.
    """
    context, warning = PipelineContext.for_workspace(Workspace(hotel_id))
    if warning:
        typer.secho(f"  · {warning}", fg=typer.colors.YELLOW)
    return context


assets_app = typer.Typer(no_args_is_help=True, help="Inventaire et droits des médias (§9).")
app.add_typer(assets_app, name="assets")


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
) -> None:
    """Collecte multi-sources puis tri assisté (§9, §11)."""
    from .gather import (
        build_manifest,
        collect_sources,
        download_all,
        summarise,
        triage,
    )

    workspace = Workspace(hotel_id)
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

    context = _context(hotel_id)
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

    context = _context(hotel_id)
    building = spatial.candidate(spatial.confirmed_building_id)
    report = dedup_levels.run(
        manifest.assets, building.centroid_lat, building.centroid_lon, policy=context.policy
    )
    workspace.write_assets(manifest)
    workspace.write_report("01_sources/duplicate_report.json", report, context)

    typer.echo(f"  fichiers                  {report.files}")
    typer.echo(f"  photographies uniques     {report.perceptual_groups}")
    typer.echo(f"  points de vue indépendants {report.viewpoints}")
    typer.echo(
        f"  rôles : {report.canonical} canonique(s), "
        f"{report.overlap} recouvrement, {report.inactive} inactif(s)"
    )
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

    workspace = Workspace(hotel_id)
    manifest = workspace.read_assets()
    spatial = workspace.read_spatial()
    if manifest is None or spatial is None:
        typer.secho("manifeste d'assets et manifeste spatial requis", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    context = _context(hotel_id)
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
    workspace.write_report("01_sources/classification_report.json", report, context)

    typer.echo("")
    typer.echo(f"  {report.total} asset(s), {report.needs_review} en revue")
    typer.echo(f"  sujets  : {report.subjects_assigned}")
    typer.echo(f"  secteurs: {report.sectors_assigned}")


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


@geo_app.command("discover")
def geo_discover(
    hotel_id: str = typer.Argument(...),
    no_sizes: bool = typer.Option(False, "--no-sizes", help="Ne pas mesurer les volumes."),
) -> None:
    """Découvre les tuiles LiDAR couvrant l'empreinte. **Ne télécharge aucun LAZ.**"""
    from .geo import CoverageState, discover

    workspace = Workspace(hotel_id)
    spatial = workspace.read_spatial()
    if spatial is None or not spatial.confirmed_building_id:
        typer.secho("bâtiment confirmé requis", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    context = _context(hotel_id)
    building = spatial.candidate(spatial.confirmed_building_id)
    result = discover(building.wkt, measure_sizes=not no_sizes)
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

    context = _context(hotel_id)
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

    workspace.write_report("06_geo/acquisition_report.json", payload, context)

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

    context = _context(hotel_id)
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

    from .geo.derive import derive, verify_written
    from .geo.raster import GridSpec

    workspace = Workspace(hotel_id)
    spatial = workspace.read_spatial()
    acquisition = workspace.read_json("06_geo/acquisition_report.json")
    if spatial is None or not acquisition or not acquisition.get("sources"):
        typer.secho("tuile acquise et bâtiment confirmé requis", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    context = _context(hotel_id)
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
    )

    grid = GridSpec(**result.grid)
    problems = verify_written(result, grid)
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

        artifacts = {a.artifact_id: a for a in site.artifacts}
        artifacts.update({a.artifact_id: a for a in result.artifacts})
        site.artifacts = list(artifacts.values())
        workspace.write_site(site)

    workspace.write_report(f"06_geo/derivation_report_{run_id}.json", result, context)

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

    context = _context(hotel_id)
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

    context = _context(hotel_id)
    report = assess(manifest.assets, context.profile, context.policy)
    workspace.write_assets(manifest)
    workspace.write_report("01_sources/temporal_report.json", report, context)

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


@assets_app.command("coverage")
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
