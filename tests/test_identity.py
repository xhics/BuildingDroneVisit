"""L'identité du bâtiment se juge sur des modèles, pas sur des métadonnées."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hotel_pipeline.identity.anchors import Anchor, AnchorSet
from hotel_pipeline.identity.sign_match import evaluate as sign_evaluate
from hotel_pipeline.identity.sign_match import find_term
from hotel_pipeline.identity.verdict import (
    IdentityStatus,
    calibrate_threshold,
    judge,
)


def _unit(*values: float) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float32)
    return vector / np.linalg.norm(vector)


class _FakeIndex:
    """Index d'embeddings sans modèle : les tests portent sur la décision."""

    def __init__(self, vectors: dict[Path, np.ndarray]) -> None:
        self._vectors = vectors
        self.saved = False

    def vector_of(self, path: Path) -> np.ndarray:
        return self._vectors[path]

    def save(self) -> None:
        self.saved = True


# --- calibration ------------------------------------------------------------


def test_le_seuil_se_place_entre_deux_populations() -> None:
    """Deux groupes francs : la coupure doit tomber entre les deux."""
    rng = np.random.default_rng(0)
    low = list(rng.normal(0.45, 0.03, 60))
    high = list(rng.normal(0.78, 0.03, 20))
    threshold, reason = calibrate_threshold(low + high)
    assert 0.55 <= threshold <= 0.72
    assert "Otsu" in reason


def test_un_corpus_trop_petit_ne_se_calibre_pas() -> None:
    threshold, reason = calibrate_threshold([0.4, 0.5, 0.9])
    assert threshold == pytest.approx(0.55)
    assert "trop petit" in reason


def test_le_seuil_reste_dans_ses_bornes() -> None:
    """Un corpus sans aucun vrai positif ne doit pas faire fondre le seuil."""
    rng = np.random.default_rng(1)
    threshold, _ = calibrate_threshold(list(rng.normal(0.30, 0.02, 80)))
    assert 0.55 <= threshold <= 0.80


def test_la_calibration_survit_a_une_distribution_continue() -> None:
    """Le cas réel : pas de trou franc, une queue continue.

    C'est la forme qui avait mis en défaut une coupure « au plus grand écart »,
    laquelle tombait dans le bruit de la queue haute.
    """
    rng = np.random.default_rng(2)
    scores = list(rng.uniform(0.35, 0.85, 200))
    threshold, _ = calibrate_threshold(scores)
    retained = sum(1 for s in scores if s >= threshold)
    assert 0 < retained < len(scores)


# --- verdict ----------------------------------------------------------------


def test_sans_ancre_rien_n_est_decidable() -> None:
    verdict = judge("a", 0.9, 0.6, None, anchor_count=0, anchor_coherence=1.0)
    assert verdict.status is IdentityStatus.UNDECIDABLE
    assert "aucune ancre" in verdict.reason


def test_des_ancres_incoherentes_bloquent_le_verdict() -> None:
    """Si les ancres ne se ressemblent pas, l'une d'elles est fausse."""
    verdict = judge("a", 0.9, 0.6, "x", anchor_count=3, anchor_coherence=0.30)
    assert verdict.status is IdentityStatus.UNDECIDABLE
    assert "incohérentes" in verdict.reason


def test_la_zone_d_indecision_demande_une_revue() -> None:
    verdict = judge("a", 0.61, 0.60, "x", anchor_count=2, anchor_coherence=0.9)
    assert verdict.status is IdentityStatus.UNCERTAIN
    assert "revue humaine" in verdict.reason


def test_les_extremes_sont_tranches() -> None:
    high = judge("a", 0.85, 0.60, "x", anchor_count=2, anchor_coherence=0.9)
    low = judge("b", 0.35, 0.60, "x", anchor_count=2, anchor_coherence=0.9)
    assert high.status is IdentityStatus.MATCH
    assert low.status is IdentityStatus.MISMATCH


# --- ancres -----------------------------------------------------------------


def test_une_ancre_confirmee_pese_plus_qu_une_ancre_ocr() -> None:
    confirmed = Anchor("a", Path("a.jpg"), "operator_confirmed", "")
    read = Anchor("b", Path("b.jpg"), "sign_ocr", "")
    assert confirmed.weight > read.weight


def test_la_similarite_retient_le_maximum_pondere() -> None:
    """Une vue d'arrière ne ressemble à aucune vue de face, et compte quand même."""
    front, rear = _unit(1, 0, 0), _unit(0, 1, 0)
    anchors = AnchorSet(
        "h",
        [
            Anchor("front", Path("f.jpg"), "operator_confirmed", ""),
            Anchor("rear", Path("r.jpg"), "operator_confirmed", ""),
        ],
    )
    anchors._vectors = [front, rear]
    score, nearest = anchors.similarity(rear)
    assert nearest == "rear"
    assert score == pytest.approx(1.0, abs=1e-5)


def test_des_ancres_opposees_ont_une_coherence_faible() -> None:
    anchors = AnchorSet("h", [])
    anchors._vectors = [_unit(1, 0), _unit(0, 1)]
    assert anchors.coherence() < 0.2


def test_sans_ancre_la_similarite_est_nulle() -> None:
    assert AnchorSet("h", []).similarity(_unit(1, 0)) == (0.0, None)


# --- lecture d'enseigne -----------------------------------------------------


def test_un_numero_civique_n_empeche_pas_de_lire_une_enseigne() -> None:
    """Cas réel : l'OCR rend « TETRA 1205 TECH » pour l'immeuble voisin."""
    hit = find_term("pere TETRA 1205 TECH", "tetra tech")
    assert hit.matched
    assert hit.method == "subsequence"


def test_une_lettre_mal_lue_n_empeche_pas_de_reconnaitre_un_nom() -> None:
    """Cas réel : « Isomed » est rendu « Tsomed »."""
    hit = find_term("NI 1201 Tsomed 514", "isomed")
    assert hit.matched
    assert hit.method == "fuzzy"


def test_deux_noms_distincts_ne_sont_jamais_confondus() -> None:
    """La tolérance ne doit pas rapprocher deux établissements différents."""
    assert not find_term("HOTEL WELCOMINNS", "tetra tech").matched
    assert not find_term("TETRA TECH", "welcominns").matched


def test_un_mot_court_n_est_pas_apparie_approximativement() -> None:
    """« inn » ne doit pas se déclencher sur « in », « inc » ou « ino »."""
    assert not find_term("ino sport", "inn").matched


def test_le_nom_attendu_l_emporte_sur_l_exclusion() -> None:
    status, term, _ = sign_evaluate(
        "HOTEL WELCOMINNS restaurant tetra tech",
        ["welcominns"],
        ["tetra tech"],
    )
    assert status == "match"
    assert term == "welcominns"


def test_une_enseigne_concurrente_dement_l_appartenance() -> None:
    status, term, _ = sign_evaluate("pere TETRA 1205 TECH", ["welcominns"], ["tetra tech"])
    assert status == "mismatch"
    assert term == "tetra tech"


def test_un_texte_muet_reste_indecis() -> None:
    """Ne pas trancher vaut mieux que trancher faux."""
    status, term, _ = sign_evaluate("Google Gocgle", ["welcominns"], ["tetra tech"])
    assert status == "uncertain"
    assert term is None


# --- dépistage --------------------------------------------------------------


def test_le_depistage_trie_le_corpus_et_sauve_l_index(tmp_path: Path) -> None:
    from hotel_pipeline.identity.screen import screen_assets

    anchor_image = tmp_path / "anchor.jpg"
    anchor_image.write_bytes(b"x")
    same = tmp_path / "same.jpg"
    same.write_bytes(b"y")
    other = tmp_path / "other.jpg"
    other.write_bytes(b"z")

    vectors = {
        anchor_image: _unit(1, 0, 0),
        same: _unit(0.97, 0.24, 0),
        other: _unit(0, 0, 1),
    }
    index = _FakeIndex(vectors)
    anchors = AnchorSet(
        "h", [Anchor("anchor", anchor_image, "operator_confirmed", "vérifiée")]
    )

    result = screen_assets(
        "h",
        [("same", same), ("other", other)],
        anchors,
        index,
        with_attributes=False,
    )

    statuses = {a.asset_id: a.verdict.status for a in result.assets}
    assert statuses["same"] is IdentityStatus.MATCH
    assert statuses["other"] is IdentityStatus.MISMATCH
    assert index.saved


def test_une_image_rejetee_ne_peut_pas_servir_de_reference(tmp_path: Path) -> None:
    """Une belle photographie du mauvais bâtiment ne vaut rien."""
    from hotel_pipeline.identity.screen import ScreenedAsset
    from hotel_pipeline.identity.verdict import IdentityVerdict

    asset = ScreenedAsset(
        "x",
        tmp_path / "x.jpg",
        IdentityVerdict("x", IdentityStatus.MISMATCH, 0.9, 0.6, "a", ""),
        {"close_framing": 1.0, "facade_visible": 1.0, "unobstructed": 1.0},
    )
    assert asset.reference_score == 0.0


def test_une_facade_illisible_effondre_le_score_de_reference(tmp_path: Path) -> None:
    """Cas réel : l'immeuble de bureaux voisin scorait 0,83 en ressemblance."""
    from hotel_pipeline.identity.screen import ScreenedAsset
    from hotel_pipeline.identity.verdict import IdentityVerdict

    def build(facade: float) -> ScreenedAsset:
        return ScreenedAsset(
            "x",
            tmp_path / "x.jpg",
            IdentityVerdict("x", IdentityStatus.MATCH, 0.83, 0.6, "a", ""),
            {"close_framing": 0.9, "facade_visible": facade, "unobstructed": 0.9},
        )

    assert build(0.11).reference_score < build(0.75).reference_score * 0.6


def test_le_rapport_porte_ses_reserves(tmp_path: Path) -> None:
    from hotel_pipeline.identity.screen import screen_assets

    anchor_image = tmp_path / "a.jpg"
    anchor_image.write_bytes(b"a")
    other = tmp_path / "b.jpg"
    other.write_bytes(b"b")
    index = _FakeIndex({anchor_image: _unit(1, 0), other: _unit(0, 1)})
    anchors = AnchorSet("h", [Anchor("a", anchor_image, "operator_confirmed", "")])

    payload = screen_assets(
        "h", [("b", other)], anchors, index, with_attributes=False
    ).as_dict()

    assert "counts" in payload
    assert payload["threshold_reason"]
    joined = " ".join(payload["caveats"])
    assert "ancre fausse" in joined
    assert "revue humaine" in joined


# --- découverte d'ancres ----------------------------------------------------


class _FakeReader:
    """Lecteur d'enseigne scripté : les tests portent sur la décision."""

    def __init__(self, texts: dict[str, str]) -> None:
        self.texts = texts
        self.read_paths: list[Path] = []

    def read(self, path: Path) -> str:
        self.read_paths.append(path)
        return self.texts.get(path.name, "")


@pytest.fixture()
def fake_reader(monkeypatch):
    def install(texts: dict[str, str]) -> _FakeReader:
        from hotel_pipeline.identity import screen

        reader = _FakeReader(texts)
        monkeypatch.setattr(screen, "_sign_reader", lambda: reader)
        return reader

    return install


def _images(tmp_path: Path, names: list[str]) -> list[tuple[str, Path]]:
    out = []
    for name in names:
        path = tmp_path / name
        path.write_bytes(name.encode())
        out.append((path.stem, path))
    return out


def test_une_enseigne_lue_devient_une_ancre(tmp_path: Path, fake_reader) -> None:
    from hotel_pipeline.identity.screen import discover_anchors_by_sign

    images = _images(tmp_path, ["a.jpg", "b.jpg"])
    fake_reader({"a.jpg": "route vide", "b.jpg": "HOTEL WELCOMINNS"})

    found = discover_anchors_by_sign(images, ["welcominns"])

    assert [a.path.name for a in found] == ["b.jpg"]
    assert found[0].origin == "sign_ocr"
    assert "welcominns" in found[0].evidence.lower()


def test_une_enseigne_degradee_est_reconnue(tmp_path: Path, fake_reader) -> None:
    """L'OCR de rue rend un texte sale ; l'appariement doit rester tolérant."""
    from hotel_pipeline.identity.screen import discover_anchors_by_sign

    images = _images(tmp_path, ["a.jpg"])
    fake_reader({"a.jpg": '9 8 "WelCOMINNS'})

    assert len(discover_anchors_by_sign(images, ["Hôtel WelcomINNS"])) == 1


def test_une_enseigne_concurrente_n_est_pas_une_ancre(
    tmp_path: Path, fake_reader
) -> None:
    from hotel_pipeline.identity.screen import discover_anchors_by_sign

    images = _images(tmp_path, ["a.jpg"])
    fake_reader({"a.jpg": "pere TETRA 1205 TECH"})

    found = discover_anchors_by_sign(images, ["welcominns"], ["tetra tech"])
    assert found == []


def test_un_logo_ne_peut_pas_servir_d_ancre(tmp_path: Path, fake_reader) -> None:
    """Cas réel : un wordmark du site officiel arrivait en tête des propositions.

    Une ancre calibre tout le tri par sa ressemblance visuelle : elle doit
    montrer le bâtiment, pas son identité graphique.
    """
    from hotel_pipeline.identity.screen import discover_anchors_by_sign

    images = _images(tmp_path, ["logo.jpg", "photo.jpg"])
    fake_reader({"logo.jpg": "CLUB ELITE WELCOMINNS", "photo.jpg": "HOTEL WELCOMINNS"})

    found = discover_anchors_by_sign(
        images,
        ["welcominns"],
        photo_scores={"logo": 0.05, "photo": 0.93},
    )

    assert [a.path.name for a in found] == ["photo.jpg"]


def test_l_ocr_ne_lit_que_le_budget_alloue(tmp_path: Path, fake_reader) -> None:
    """Lire un corpus entier prendrait des heures : le budget est un garde-fou."""
    from hotel_pipeline.identity.screen import discover_anchors_by_sign

    images = _images(tmp_path, [f"{i:03d}.jpg" for i in range(50)])
    reader = fake_reader({})

    discover_anchors_by_sign(images, ["welcominns"], budget=7)

    assert len(reader.read_paths) == 7


def test_la_recherche_s_arrete_au_nombre_demande(
    tmp_path: Path, fake_reader
) -> None:
    from hotel_pipeline.identity.screen import discover_anchors_by_sign

    images = _images(tmp_path, [f"{i:02d}.jpg" for i in range(10)])
    fake_reader({f"{i:02d}.jpg": "HOTEL WELCOMINNS" for i in range(10)})

    assert len(discover_anchors_by_sign(images, ["welcominns"], limit=3)) == 3


def test_sans_enseigne_lisible_aucune_ancre_n_est_inventee(
    tmp_path: Path, fake_reader
) -> None:
    """Ne rien proposer vaut mieux que proposer une ancre fausse."""
    from hotel_pipeline.identity.screen import discover_anchors_by_sign

    images = _images(tmp_path, ["a.jpg", "b.jpg"])
    fake_reader({"a.jpg": "Google", "b.jpg": ""})

    assert discover_anchors_by_sign(images, ["welcominns"]) == []


# --- droits et numéro civique -----------------------------------------------


def test_les_candidats_viennent_du_manifeste(tmp_path: Path) -> None:
    """Le dépistage parcourait le disque et ignorait les droits."""
    from hotel_pipeline.identity.candidates import collect

    image = tmp_path / "a.jpg"
    image.write_bytes(b"x")
    manifest = tmp_path / "m.json"
    manifest.write_text(
        json.dumps(
            {
                "assets": [
                    {
                        "id": "a",
                        "local_path": str(image),
                        "rights": "open_data",
                        "bearing_from_building_deg": 120.0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    found = collect(manifest)

    assert len(found) == 1
    candidate = found.candidates[0]
    assert candidate.rights == "open_data"
    assert candidate.rights_cleared is True
    assert candidate.bearing_deg == pytest.approx(120.0)


def test_un_recadrage_herite_des_droits_de_sa_source(tmp_path: Path) -> None:
    from hotel_pipeline.identity.candidates import collect

    source = tmp_path / "src.jpg"
    source.write_bytes(b"x")
    crops = tmp_path / "recrops"
    crops.mkdir()
    crop = crops / "SECT225_zj6pG6EOemMZ7d_54h.jpg"
    crop.write_bytes(b"y")

    manifest = tmp_path / "m.json"
    manifest.write_text(
        json.dumps(
            {
                "assets": [
                    {
                        "id": "street_view-zj6pG6EOemMZ7dPlDXJeMA",
                        "local_path": str(source),
                        "rights": "open_data",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    found = collect(manifest, extra_folders=[crops])
    derived = [c for c in found.candidates if not c.in_manifest]

    assert len(derived) == 1
    assert derived[0].rights == "open_data"
    assert derived[0].source_asset_id == "street_view-zj6pG6EOemMZ7dPlDXJeMA"


def test_un_recadrage_orphelin_ne_prend_pas_de_droits(tmp_path: Path) -> None:
    """Ne pas savoir n'est pas une autorisation."""
    from hotel_pipeline.identity.candidates import collect

    crops = tmp_path / "recrops"
    crops.mkdir()
    (crops / "inconnu.jpg").write_bytes(b"y")
    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps({"assets": []}), encoding="utf-8")

    found = collect(manifest, extra_folders=[crops])

    assert found.candidates[0].rights == "unknown"
    assert found.candidates[0].rights_cleared is False


def test_le_classement_n_arbitre_pas_les_droits(tmp_path: Path) -> None:
    """Le statut est transporté, jamais utilisé pour trier : la décision
    d'usage se prend ailleurs, et un classement qui la préjuge masquerait des
    images visuellement meilleures."""
    from hotel_pipeline.identity.screen import ScreenedAsset
    from hotel_pipeline.identity.verdict import IdentityVerdict

    def build(rights: str) -> ScreenedAsset:
        return ScreenedAsset(
            "x",
            tmp_path / "x.jpg",
            IdentityVerdict("x", IdentityStatus.MATCH, 0.8, 0.6, "a", ""),
            {"close_framing": 0.9, "facade_visible": 0.9, "unobstructed": 0.9},
            rights=rights,
        )

    assert build("unknown").reference_score == build("open_data").reference_score
    # Le statut reste lisible dans la sortie.
    assert build("unknown").as_dict()["rights"] == "unknown"


def test_un_numero_civique_voisin_dement_l_appartenance() -> None:
    """Cas réel : l'immeuble du 1205 portait son numéro en grand sur sa façade."""
    from hotel_pipeline.identity.sign_match import contradicts_civic

    assert contradicts_civic("Ampere ALOUER 1205 866 3333", "1195") == "1205"
    assert contradicts_civic("TETRA 1205 TECH", "1195") == "1205"


def test_le_bon_numero_ne_dement_rien() -> None:
    from hotel_pipeline.identity.sign_match import contradicts_civic

    assert contradicts_civic("1195 rue Ampere hotel", "1195") is None


def test_un_numero_lointain_ne_prouve_rien() -> None:
    """Un téléphone ou une rue voisine ne dit rien de l'établissement."""
    from hotel_pipeline.identity.sign_match import contradicts_civic

    assert contradicts_civic("A LOUER 514 866 3333", "1195") is None
    assert contradicts_civic("9 8 WelCOMINNS", "1195") is None


def test_le_numero_civique_se_lit_dans_l_adresse() -> None:
    from hotel_pipeline.identity.sign_match import civic_number

    assert civic_number("1195 rue Ampère, Boucherville") == "1195"
    assert civic_number("rue sans numéro") is None


def test_un_seuil_cale_sur_le_plancher_bascule_en_quantile() -> None:
    """Otsu calé sur une borne n'a rien séparé.

    Mesuré sur ce pilote : élargir le corpus faisait tomber le seuil au
    plancher, et vingt pour cent des images ressortaient `match` — dont des
    bâtiments de brique sans rapport avec le site.
    """
    import numpy as np

    from hotel_pipeline.identity.verdict import THRESHOLD_FLOOR, calibrate_threshold

    # Population unique, sans frontière interne : Otsu se cale sur la borne.
    rng = np.random.default_rng(9)
    scores = list(rng.normal(0.50, 0.09, 400))

    threshold, reason = calibrate_threshold(scores)
    retained = sum(1 for s in scores if s >= threshold)

    assert threshold > THRESHOLD_FLOOR
    assert "quantile" in reason
    # Une part restreinte, non le cinquième du corpus.
    assert retained < len(scores) * 0.15


def test_un_corpus_franchement_separe_garde_otsu() -> None:
    """Le repli ne doit pas remplacer une frontière que la mesure établit."""
    import numpy as np

    from hotel_pipeline.identity.verdict import calibrate_threshold

    rng = np.random.default_rng(10)
    scores = list(rng.normal(0.42, 0.04, 200)) + list(rng.normal(0.80, 0.04, 40))

    _, reason = calibrate_threshold(scores)

    assert "Otsu" in reason
    assert "quantile" not in reason


# --- nature du bâti ---------------------------------------------------------


class _FakeIndexWithText(_FakeIndex):
    """Index dont l'encodeur de texte rend des vecteurs contrôlés."""

    def __init__(self, vectors, text_vectors):
        super().__init__(vectors)
        self._text = text_vectors

        class _Embedder:
            def __init__(self, text):
                self._text = text

            def encode_text(self, phrases):
                return self._text

        self.embedder = _Embedder(text_vectors)


def _screened(asset_id: str, path: Path, status: IdentityStatus):
    from hotel_pipeline.identity.screen import ScreenedAsset
    from hotel_pipeline.identity.verdict import IdentityVerdict

    return ScreenedAsset(
        asset_id, path, IdentityVerdict(asset_id, status, 0.62, 0.63, "a", "")
    )


def test_une_indecise_qui_ne_montre_aucun_bati_est_ecartee(tmp_path: Path) -> None:
    """Une rue résidentielle tombe dans la bande du seuil sans être ambiguë."""
    from hotel_pipeline.identity.screen import _resolve_uncertain

    # Le premier vecteur de texte décrit le bâti attendu ; l'image ressemble
    # nettement à une alternative.
    text = np.stack([_unit(1, 0, 0), _unit(0, 1, 0), _unit(0, 0, 1)])
    vectors = {"rue": _unit(0, 1, 0)}
    assets = [_screened("rue", tmp_path / "rue.jpg", IdentityStatus.UNCERTAIN)]

    resolved = _resolve_uncertain(assets, _FakeIndexWithText(vectors, text), vectors)

    assert resolved == 1
    assert assets[0].verdict.status is IdentityStatus.MISMATCH
    assert "nature du bâti" in assets[0].verdict.reason


def test_la_forme_du_bati_ne_promeut_jamais(tmp_path: Path) -> None:
    """L'immeuble de bureaux voisin correspond parfaitement à la description.

    Mesuré sur ce pilote : promu sur ce seul critère, il ressortait `match` à
    1,00 sans qu'aucune enseigne ne l'ait démenti.
    """
    from hotel_pipeline.identity.screen import _resolve_uncertain

    text = np.stack([_unit(1, 0, 0), _unit(0, 1, 0), _unit(0, 0, 1)])
    vectors = {"voisin": _unit(1, 0, 0)}
    assets = [_screened("voisin", tmp_path / "v.jpg", IdentityStatus.UNCERTAIN)]

    _resolve_uncertain(assets, _FakeIndexWithText(vectors, text), vectors)

    # Le doute subsiste : la forme dit ce qu'une image n'est pas, jamais
    # qu'elle est le bon bâtiment.
    assert assets[0].verdict.status is IdentityStatus.UNCERTAIN
    assert assets[0].built_form is not None


def test_les_verdicts_deja_tranches_ne_bougent_pas(tmp_path: Path) -> None:
    from hotel_pipeline.identity.screen import _resolve_uncertain

    text = np.stack([_unit(1, 0, 0), _unit(0, 1, 0)])
    vectors = {"ok": _unit(0, 1, 0)}
    assets = [_screened("ok", tmp_path / "ok.jpg", IdentityStatus.MATCH)]

    assert _resolve_uncertain(assets, _FakeIndexWithText(vectors, text), vectors) == 0
    assert assets[0].verdict.status is IdentityStatus.MATCH
