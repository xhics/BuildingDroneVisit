"""La résolution planifiée atteint la requête fournisseur (collecte V2).

Le défaut fermé ici : le plan annonçait `256`, `request_spec` conservait
`thumb_2048`, et `fetch()` ne recevait que le candidat. Trois conséquences
silencieuses — mesurer un fichier, en télécharger un autre, publier une
provenance décrivant le premier.
"""

from __future__ import annotations

import pytest

from hotel_pipeline.acquisition_request import (
    RequestUnresolvable,
    resolve,
    resolve_all,
)
from hotel_pipeline.schemas.acquisition import (
    CaptureCandidate,
    CaptureIntent,
    PlannedAcquisition,
)


def _candidate(candidate_id="mly-1", source="mapillary", **overrides):
    fields = dict(
        candidate_id=candidate_id, source=source, provider_id="1",
        camera_lat=45.5, camera_lon=-73.4,
        request_spec={"provider_id": "1", "resolution": "thumb_2048"},
        available_resolutions=["thumb_256", "thumb_2048", "256", "2048"],
    )
    fields.update(overrides)
    return CaptureCandidate(**fields)


def _acquisition(candidate_id="mly-1", resolution="256"):
    return PlannedAcquisition(
        candidate_id=candidate_id,
        intents=[CaptureIntent.BUILDING_CAPTURE],
        resolution=resolution,
        serves_demands=["obligation:front"],
        selection_rationale="retenu pour l'essai",
    )


# --- la traduction : vocabulaire du plan → vocabulaire du fournisseur --------


def test_a_preview_becomes_a_real_thumbnail_request() -> None:
    """« 256 » au plan devait devenir `thumb_256` à l'API.

    Sans traduction, `request_spec` gardait `thumb_2048` : le plan promettait
    un aperçu et la requête demandait l'image entière.
    """
    request = resolve(_candidate(), _acquisition(resolution="256"))

    assert request.semantic_resolution == "256"
    assert request.provider_resolution == "thumb_256"
    assert request.request_spec["resolution"] == "thumb_256", (
        "c'est ce champ que lit le résolveur d'adresse"
    )


def test_a_full_acquisition_keeps_the_full_resolution() -> None:
    request = resolve(_candidate(), _acquisition(resolution="2048"))

    assert request.provider_resolution == "thumb_2048"
    assert request.request_spec["resolution"] == "thumb_2048"


def test_street_view_translates_to_a_size() -> None:
    """Street View nomme ses résolutions par dimensions."""
    candidate = _candidate(
        "sv-1", source="street_view",
        request_spec={"pano_id": "A", "size": "640x640"},
        available_resolutions=["640x640", "256", "2048", "256x256"],
    )
    request = resolve(candidate, _acquisition("sv-1", resolution="256"))

    assert request.provider_resolution == "256x256"
    assert request.request_spec["size"] == "256x256"
    assert request.width_px == 256 and request.height_px == 256


def test_the_provider_vocabulary_is_accepted_when_declared() -> None:
    """Un cadrage Street View nomme déjà sa taille : la refuser serait absurde."""
    candidate = _candidate(
        "sv-1", source="street_view",
        request_spec={"pano_id": "A", "size": "640x640"},
        available_resolutions=["640x640"],
    )
    request = resolve(candidate, _acquisition("sv-1", resolution="640x640"))

    assert request.provider_resolution == "640x640"


def test_an_unservable_resolution_is_refused_not_guessed() -> None:
    """Servir autre chose téléchargerait un fichier que personne n'a mesuré."""
    candidate = _candidate(available_resolutions=["thumb_2048"])

    with pytest.raises(RequestUnresolvable, match="absent des résolutions"):
        resolve(candidate, _acquisition(resolution="256"))


def test_an_unknown_source_is_refused() -> None:
    candidate = _candidate("x-1", source="source-inconnue")

    with pytest.raises(RequestUnresolvable, match="sans table de résolutions"):
        resolve(candidate, _acquisition("x-1"))


# --- l'empreinte : le consentement porte sur une demande précise -------------


def test_changing_the_resolution_changes_the_identity() -> None:
    """Sans quoi le consentement porterait sur un candidat dont on
    redéfinirait le contenu après coup."""
    preview = resolve(_candidate(), _acquisition(resolution="256"))
    full = resolve(_candidate(), _acquisition(resolution="2048"))

    assert preview.digest != full.digest


def test_the_same_request_yields_the_same_identity() -> None:
    """Deux exécutions doivent produire la même empreinte, sinon rien ne se
    verrouille."""
    first = resolve(_candidate(), _acquisition(resolution="256"))
    second = resolve(_candidate(), _acquisition(resolution="256"))

    assert first.digest == second.digest


def test_a_resolved_request_cannot_be_altered() -> None:
    """Une requête modifiable après consentement n'est plus celle consentie."""
    request = resolve(_candidate(), _acquisition())

    with pytest.raises((AttributeError, TypeError)):
        request.provider_resolution = "thumb_2048"  # type: ignore[misc]


# --- le lot : rien de partiel -------------------------------------------------


def test_one_unresolvable_acquisition_refuses_the_whole_batch() -> None:
    """Un plan dont une acquisition ne se traduit pas annoncerait un volume
    qu'il ne saurait pas obtenir."""
    good = _candidate("mly-1")
    bad = _candidate("mly-2", available_resolutions=["thumb_2048"])

    with pytest.raises(RequestUnresolvable, match="mly-2"):
        resolve_all(
            {"mly-1": good, "mly-2": bad},
            [_acquisition("mly-1"), _acquisition("mly-2")],
        )


def test_a_missing_candidate_does_not_block_the_others() -> None:
    """Une vue disparue se rapporte ; elle n'annule pas les autres."""
    resolved = resolve_all(
        {"mly-1": _candidate("mly-1")},
        [_acquisition("mly-1"), _acquisition("mly-absent")],
    )

    assert set(resolved) == {"mly-1"}
