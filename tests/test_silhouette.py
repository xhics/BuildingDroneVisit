"""Les images au sol donnent l'allure que le relevé aérien ne peut pas voir."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hotel_pipeline.conditioning.silhouette import (
    CLASS_PROMPTS,
    PROFILE_SHAPES,
    SilhouetteMap,
    SilhouetteReading,
    infer_shape,
    shape_hints,
)


def _map(labels: np.ndarray, classes: list[str], asset_id: str = "a") -> SilhouetteMap:
    return SilhouetteMap(asset_id=asset_id, labels=labels, classes=classes, tile_px=32)


CLASSES = list(CLASS_PROMPTS) + ["indetermine"]


def _index(name: str) -> int:
    return CLASSES.index(name)


# --- lecture d'une carte ----------------------------------------------------


def test_la_fraction_compte_les_tuiles_d_une_nature() -> None:
    labels = np.full((4, 4), _index("ciel"))
    labels[2:, :] = _index("sol")

    silhouette = _map(labels, CLASSES)

    assert silhouette.fraction("ciel") == pytest.approx(0.5)
    assert silhouette.fraction("sol") == pytest.approx(0.5)
    assert silhouette.fraction("vehicule") == 0.0


def test_une_nature_absente_du_vocabulaire_vaut_zero() -> None:
    assert _map(np.zeros((2, 2), dtype=int), CLASSES).fraction("licorne") == 0.0


def test_le_profil_vertical_suit_la_repartition_en_hauteur() -> None:
    """C'est précisément ce qu'un relevé aérien ne peut pas donner."""
    labels = np.full((4, 4), _index("ciel"))
    labels[3, :] = _index("vegetation")  # végétation en bas seulement

    profile = _map(labels, CLASSES).vegetation_profile()

    assert profile[0] == pytest.approx(0.0)
    assert profile[-1] == pytest.approx(1.0)


def test_le_conifere_et_l_herbe_comptent_comme_vegetation() -> None:
    labels = np.array([[_index("conifere"), _index("herbe")]])
    assert _map(labels, CLASSES).vegetation_profile()[0] == pytest.approx(1.0)


# --- allure -----------------------------------------------------------------


def test_un_profil_effile_est_reconnu_conique() -> None:
    name, score = infer_shape(np.asarray(PROFILE_SHAPES["conique"]))
    assert name == "conique"
    assert score > 0.9


def test_un_profil_etale_est_reconnu() -> None:
    name, _ = infer_shape(np.asarray(PROFILE_SHAPES["etale"]))
    assert name == "etale"


def test_un_profil_trop_court_reste_indetermine() -> None:
    """Deux niveaux ne décrivent aucune allure."""
    name, score = infer_shape(np.array([0.0, 0.5, 0.0, 0.0]))
    assert name == "indetermine"
    assert score == 0.0


def test_un_profil_vide_reste_indetermine() -> None:
    assert infer_shape(np.zeros(10)) == ("indetermine", 0.0)


def test_l_allure_ne_depend_pas_de_la_hauteur_apparente() -> None:
    """Un arbre photographié de loin a la même allure que de près."""
    reference = np.asarray(PROFILE_SHAPES["conique"])
    scaled = reference * 0.3

    assert infer_shape(reference)[0] == infer_shape(scaled)[0]


# --- synthèse ---------------------------------------------------------------


def test_l_allure_dominante_ressort_de_plusieurs_vues() -> None:
    conical = np.asarray(PROFILE_SHAPES["conique"])
    spread = np.asarray(PROFILE_SHAPES["etale"])

    class _Fake(SilhouetteMap):
        def __init__(self, profile: np.ndarray) -> None:
            super().__init__("x", np.zeros((2, 2), dtype=int), CLASSES, 32)
            self._profile = profile

        def vegetation_profile(self) -> np.ndarray:
            return self._profile

    reading = SilhouetteReading(
        hotel_id="h", maps=[_Fake(conical), _Fake(conical), _Fake(spread)]
    )
    hints = shape_hints(reading)

    assert hints["dominant_shape"] == "conique"
    assert hints["by_shape"]["conique"] == 2
    assert hints["views_used"] == 3


def test_sans_vue_aucune_allure_n_est_avancee() -> None:
    hints = shape_hints(SilhouetteReading(hotel_id="h"))
    assert hints["dominant_shape"] == "indetermine"
    assert hints["views_used"] == 0


def test_le_rapport_porte_ses_reserves() -> None:
    payload = SilhouetteReading(hotel_id="h", maps=[]).as_dict()
    joined = " ".join(payload["caveats"])

    assert "indetermine" in joined
    assert "ne le remplace pas" in joined
    assert "hiver" in joined


def test_l_allure_est_declaree_comme_hypothese() -> None:
    assert "hypothèse" in shape_hints(SilhouetteReading(hotel_id="h"))["caveat"]


# --- rendu ------------------------------------------------------------------


def test_le_volume_rendu_suit_l_allure() -> None:
    """Un conifère effilé et un érable étalé ne peuvent pas être le même cylindre."""
    from hotel_pipeline.conditioning.environment import VegetationPatch
    from hotel_pipeline.conditioning.render import _vegetation_faces

    def top_width(shape: str) -> float:
        patch = VegetationPatch("arbres_matures", (0.0, 0.0), 4.0, 10.0, 100)
        patch.shape = shape
        vertices = np.concatenate(_vegetation_faces(patch))
        high = vertices[vertices[:, 2] > 8.0]
        return float(np.hypot(high[:, 0], high[:, 1]).max())

    assert top_width("conique") < top_width("etale")


def test_une_allure_inconnue_retombe_sur_un_cylindre() -> None:
    from hotel_pipeline.conditioning.environment import VegetationPatch
    from hotel_pipeline.conditioning.render import _vegetation_faces

    patch = VegetationPatch("arbustes", (0.0, 0.0), 3.0, 5.0, 50)
    assert patch.shape is None
    assert len(_vegetation_faces(patch)) > 0
