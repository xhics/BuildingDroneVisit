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

    images, reports = collect_sources(lat, lon, place_query, radius_m)
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

            classifier = Classifier()
        except ImportError:
            typer.secho(
                "  · OpenCLIP absent — classification ignorée "
                "(installer l'extra 'vision' sur la VM GPU)",
                fg=typer.colors.YELLOW,
            )

    gather_report = triage(manifest.assets, classifier=classifier)
    workspace.write_assets(manifest)
    workspace.write_json("01_sources/gather_report.json", gather_report.as_dict())

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
