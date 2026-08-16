"""Comptabilité des appels et modes réseau (collecte V2).

Le compteur précédent vivait dans `cached_call` et comptait les **cache
misses** : deux pages Mapillary produisaient deux `GET` et un seul décompte,
les échecs n'étaient pas comptés, et plusieurs appels contournaient le cache.

Ce que ces tests protègent : un registre qui ne compte que les succès ne mesure
pas un coût, il mesure une chance.
"""

from __future__ import annotations

import pytest

from hotel_pipeline.providers.transport import (
    NetworkMode,
    NetworkRefused,
    Outcome,
    Stage,
    ledger,
    request,
    reset_ledger,
    set_mode,
)


class FakeResponse:
    def __init__(self, status_code=200, length=None):
        self.status_code = status_code
        self.headers = {"Content-Length": str(length)} if length else {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


@pytest.fixture
def online():
    token = set_mode(NetworkMode.ONLINE)
    registre = reset_ledger()
    yield registre
    set_mode(None)


# --- pagination : une page, une requête ---------------------------------------


def test_two_api_pages_count_as_two_attempts(online) -> None:
    """Le compteur précédent en voyait une seule : il comptait les cache
    misses, et deux pages tiennent derrière un unique appel caché."""
    for page in (1, 2):
        request(
            "mapillary", Stage.COARSE_SEARCH, "GET",
            lambda: FakeResponse(200), page=page,
        )

    row = online.by_source()["mapillary"]
    assert row["attempted"] == 2
    assert row["succeeded"] == 2
    assert [a.page for a in online.attempts] == [1, 2]


def test_a_failing_second_page_is_still_counted(online) -> None:
    """Une erreur réseau emporterait l'exception, et avec elle la trace.

    D'où l'inscription **avant** l'appel : un registre qui ne compte que les
    succès mesure une chance, pas un coût.
    """
    request("mapillary", Stage.COARSE_SEARCH, "GET", lambda: FakeResponse(200), page=1)

    def boom():
        raise ConnectionError("connexion perdue")

    with pytest.raises(ConnectionError):
        request("mapillary", Stage.COARSE_SEARCH, "GET", boom, page=2)

    row = online.by_source()["mapillary"]
    assert row["attempted"] == 2, "la tentative échouée compte"
    assert row["succeeded"] == 1
    assert row["network_errors"] == 1
    assert online.attempts[1].error.startswith("ConnectionError")


def test_an_http_error_is_distinguished_from_a_network_error(online) -> None:
    """Un 429 et une coupure ne se corrigent pas de la même façon."""
    request("mapillary", Stage.COARSE_SEARCH, "GET", lambda: FakeResponse(429))

    row = online.by_source()["mapillary"]
    assert row["http_errors"] == 1
    assert row["network_errors"] == 0
    assert online.attempts[0].status_code == 429


def test_a_partial_pagination_leaves_nothing_in_the_cache() -> None:
    """Une pagination interrompue ne doit pas figer une réponse tronquée."""
    from hotel_pipeline.providers.cache import cached_call, get_cache

    set_mode(NetworkMode.ONLINE)
    reset_ledger()
    key = "essai-pagination::partielle"
    get_cache().delete(key)

    def producer():
        raise ConnectionError("deuxième page perdue")

    with pytest.raises(ConnectionError):
        cached_call(key, producer)

    assert get_cache().get(key, default=None) is None, (
        "une réponse partielle mise en cache serait rejouée comme complète"
    )
    set_mode(None)


# --- cache : servi sans réseau -------------------------------------------------


def test_a_cache_hit_emits_no_network_attempt() -> None:
    from hotel_pipeline.providers.cache import cached_call, get_cache

    set_mode(NetworkMode.ONLINE)
    key = "essai-hit::a"
    get_cache().delete(key)
    cached_call(key, lambda: {"valeur": 1})

    registre = reset_ledger()
    assert cached_call(key, lambda: {"valeur": 2}) == {"valeur": 1}

    row = registre.by_source()["essai-hit"]
    assert row["cache_hits"] == 1
    assert row["attempted"] == 0, "zéro requête réseau"
    set_mode(None)


# --- modes : trois états, non deux --------------------------------------------


def test_cache_only_refuses_a_miss_without_calling_the_producer() -> None:
    """Le producteur ne doit **pas** être appelé : refuser après avoir tenté
    sa chance n'est pas un mode fermé."""
    from hotel_pipeline.providers.cache import cached_call, get_cache

    called: list[int] = []
    key = "essai-cache-only::absent"
    get_cache().delete(key)

    set_mode(NetworkMode.CACHE_ONLY)
    reset_ledger()
    try:
        with pytest.raises(NetworkRefused, match="cache_only"):
            cached_call(key, lambda: called.append(1) or {"valeur": 1})
    finally:
        set_mode(None)

    assert called == [], "le producteur n'a pas été appelé"


def test_cache_only_serves_what_is_already_there() -> None:
    """Sans quoi le mode fermé ne servirait à rien."""
    from hotel_pipeline.providers.cache import cached_call, get_cache

    key = "essai-cache-only::present"
    get_cache().delete(key)
    set_mode(NetworkMode.ONLINE)
    cached_call(key, lambda: {"valeur": 7})

    set_mode(NetworkMode.CACHE_ONLY)
    try:
        assert cached_call(key, lambda: {"valeur": 9}) == {"valeur": 7}
    finally:
        set_mode(None)


def test_a_refusal_is_recorded_not_silent() -> None:
    """Une exécution en mode fermé présenterait sinon les mêmes compteurs
    qu'une exécution sans besoin réseau."""
    set_mode(NetworkMode.FORBIDDEN)
    registre = reset_ledger()
    try:
        with pytest.raises(NetworkRefused):
            request("mapillary", Stage.DOWNLOAD, "GET", lambda: FakeResponse(200))
    finally:
        set_mode(None)

    row = registre.by_source()["mapillary"]
    assert row["refused"] == 1
    assert row["attempted"] == 0, "aucun paquet n'est parti"


# --- deux appels pour un HEAD Mapillary ---------------------------------------


def test_a_mapillary_head_costs_two_calls(online) -> None:
    """Résolution Graph puis HEAD CDN : les confondre ferait passer un HEAD
    pour une requête là où il en coûte deux."""
    request("mapillary", Stage.URL_RESOLUTION, "GET", lambda: FakeResponse(200))
    request("mapillary", Stage.VOLUME_PROBE, "HEAD", lambda: FakeResponse(200, 4096))

    row = online.by_source()["mapillary"]
    assert row["attempted"] == 2
    assert row["by_stage"] == {"url_resolution": 1, "volume_probe": 1}
    assert row["bytes_received"] == 4096


# --- isolation entre exécutions ------------------------------------------------


def test_two_runs_do_not_add_up() -> None:
    """Un total cumulé entre deux exécutions ne dirait rien de l'une ni de
    l'autre."""
    set_mode(NetworkMode.ONLINE)
    first = reset_ledger()
    request("mapillary", Stage.COARSE_SEARCH, "GET", lambda: FakeResponse(200))
    assert first.by_source()["mapillary"]["attempted"] == 1

    second = reset_ledger()
    assert second.attempts == []
    request("street_view", Stage.COARSE_SEARCH, "GET", lambda: FakeResponse(200))
    assert "mapillary" not in second.by_source()
    set_mode(None)


# --- aucun secret dans le registre --------------------------------------------


def test_the_ledger_carries_no_url_or_token(online) -> None:
    """Un rapport versionné ne doit pas rendre un secret lisible, et une URL
    signée en est un."""
    request(
        "mapillary", Stage.DOWNLOAD, "GET", lambda: FakeResponse(200, 1024),
        request_digest="abcdef0123456789",
    )

    published = repr(online.as_dict()) + repr(
        [a.as_dict() for a in online.attempts]
    )
    for secret in ("http://", "https://", "token", "Authorization", "OAuth", "key="):
        assert secret not in published, f"{secret!r} ne doit pas figurer au registre"


def test_planned_and_actual_are_published_side_by_side(online) -> None:
    """La pagination interdit un nombre exact d'avance : annoncer un plafond
    vaut mieux que se taire, et le comparer dit si l'estimation valait."""
    online.planned_max_requests["mapillary"] = 8
    for page in (1, 2, 3):
        request(
            "mapillary", Stage.COARSE_SEARCH, "GET",
            lambda: FakeResponse(200), page=page,
        )

    published = online.as_dict()
    assert published["planned_max_requests"]["mapillary"] == 8
    assert published["actual_requests"]["mapillary"] == 3
