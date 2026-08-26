"""Client Sogni — soumission du story-board à un générateur vidéo.

**État de vérification du schéma — lire avant d'utiliser.**

Confirmé par un appel réel à l'API (``probe()``, voir historique du projet)
et par le code source officiel de Sogni (dépôts publics
``Sogni-AI/sogni-creative-agent-skill`` et ``Sogni-AI/sogni-client``, pas
seulement leur documentation marketing) :

- ``POST https://api.sogni.ai/v1/creative-agent/workflows``, auth
  ``Authorization: Bearer <clé>``.
- Structure : ``{"input": {"title": ..., "steps": [{"id", "toolName",
  "arguments"}]}}``. Confirmé en direct : ``toolName: "generate_video"``
  avec ``arguments: {"prompt": "test"}`` **suffit à démarrer une vraie
  génération** (HTTP 201, ``status: "running"``) — il n'existe pas de mode
  bac-à-sable sur cet endpoint. Tout payload structurellement valide
  s'exécute pour de vrai ; seul un payload délibérément incomplet renvoie
  une erreur de validation exploitable pour sonder le schéma sans risque.
- ``arguments.prompt`` est le seul champ strictement requis.
- ``arguments.referenceImageUrls: string[]`` — confirmé verbatim dans
  l'interface TypeScript du SDK officiel (``@sogni-ai/sogni-client``,
  ``docs/media/llms-full.txt``) : *« Seedance and HappyHorse only »*, la
  famille de modèles visée ici. Seedance 2.5 (``seedance2-5``) accepte
  jusqu'à 30 images de référence par appel (docs.sogni.ai/models/seedance-2-5/,
  cité verbatim).
- Suivi : ``GET .../workflows/{workflowId}``. Annulation :
  ``POST .../workflows/{workflowId}/cancel`` — confirmés en direct.

**Ce qui reste réellement ouvert :** ``referenceImageUrls`` exige des URL
**HTTPS déjà publiques**. Le mécanisme qui permet à Sogni d'héberger un
fichier local ("the normal S3 upload path", selon le SDK) est interne à leur
client JavaScript/TypeScript à base de WebSocket — il n'est pas exposé comme
un endpoint REST documenté que ce module (Python, appels REST directs)
puisse appeler. Aucun SDK Python officiel n'a pu être confirmé (le paquet
qu'un résumé de recherche a un temps laissé croire existant n'a pas été
retrouvé sur PyPI). Ce module ne résout donc **pas** de lui-même le passage
« chemin local -> URL publique » : ``build_workflow_payload()`` exige des
URL déjà publiques dans ``storyboard.Keyframe.reference_image`` et lève une
erreur claire sinon, plutôt que d'envoyer silencieusement des chemins qui
échoueraient côté serveur.
"""

from __future__ import annotations

import requests

from .storyboard import ContinuousStoryboard

BASE_URL = "https://api.sogni.ai/v1"
WORKFLOWS_ENDPOINT = f"{BASE_URL}/creative-agent/workflows"
TIMEOUT = 60

#: Modèle acceptant jusqu'à 30 images de référence en un seul appel — c'est
#: ce qui permet une vidéo continue en une seule génération plutôt qu'en
#: clips séparés à recoller. Voir docs.sogni.ai/models/seedance-2-5/.
DEFAULT_VIDEO_MODEL = "seedance2-5"

#: Nombre maximal d'images de référence accepté par ``DEFAULT_VIDEO_MODEL``.
#: `storyboard.MAX_KEYFRAMES` reste en-dessous, par marge de sécurité.
MAX_REFERENCE_IMAGES = 30

#: Le nom des champs et la structure sont confirmés (voir docstring) ; ce qui
#: reste non résolu est l'hébergement public des images locales, pas le
#: schéma de la requête elle-même.
BEST_EFFORT_SCHEMA = False


class SogniError(RuntimeError):
    """Un appel à l'API Sogni a échoué ou a renvoyé une erreur."""


def _headers(api_key: str) -> dict:
    return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}


def _safe_json(resp: requests.Response):
    try:
        return resp.json()
    except ValueError:
        return resp.text[:2000]


def probe(api_key: str) -> dict:
    """Appel minimal pour vérifier la clé et observer la vraie forme des réponses.

    Envoie volontairement un corps vide : la réponse (probablement une
    erreur 400 listant les champs requis) est la source la plus fiable pour
    confirmer le schéma exact — bien plus que la documentation publique, qui
    s'est révélée incomplète lors de la conception de ce module. N'entraîne
    a priori aucun coût de génération.
    """
    try:
        resp = requests.post(WORKFLOWS_ENDPOINT, headers=_headers(api_key), json={}, timeout=TIMEOUT)
    except requests.RequestException as exc:
        raise SogniError(f"connexion à l'API Sogni échouée : {exc}") from exc
    return {"status_code": resp.status_code, "body": _safe_json(resp)}


def build_workflow_payload(
    storyboard: ContinuousStoryboard, *, video_model: str = DEFAULT_VIDEO_MODEL
) -> dict:
    """Construit **une seule** requête ``creative-agent/workflows`` — champs confirmés.

    Un seul step ``generate_video``, avec toutes les images de référence du
    story-board (ordonnées) et le prompt unique décrivant le survol complet.
    Volontairement **pas** un step par figure : plusieurs appels produiraient
    des clips séparés à recoller au montage, ce qui recrée exactement les
    coupures qu'une vidéo continue doit éviter. La continuité vient d'ici —
    un seul appel — et de la trajectoire déjà chaînée en amont
    (``maneuvers.chain_maneuvers``), pas d'un assemblage après coup.

    Lève :class:`SogniError` si un ``reference_image`` de ``storyboard``
    n'est pas déjà une URL HTTPS publique (ex : un chemin local non hébergé)
    — mieux vaut échouer tôt et clairement que soumettre une requête qui
    échouera côté serveur, ou pire, qui s'exécute quand même sur une
    référence inutilisable (voir docstring du module : cet endpoint n'a
    aucun mode bac-à-sable, un payload valide s'exécute pour de vrai).
    """
    if len(storyboard.keyframes) > MAX_REFERENCE_IMAGES:
        raise SogniError(
            f"{len(storyboard.keyframes)} images de référence, au-delà des "
            f"{MAX_REFERENCE_IMAGES} acceptées par {video_model} — réduire "
            f"storyboard.MAX_KEYFRAMES ou --max-keyframes."
        )

    local_paths = [k.reference_image for k in storyboard.keyframes if not _is_public_https_url(k.reference_image)]
    if local_paths:
        raise SogniError(
            f"{len(local_paths)} image(s) de référence ne sont pas des URL HTTPS "
            f"publiques (ex: {local_paths[0]!r}) — referenceImageUrls exige des URL "
            f"déjà hébergées ; voir la section « ce qui reste réellement ouvert » du "
            f"docstring du module pour les options d'hébergement."
        )

    arguments = {
        "prompt": storyboard.master_prompt_fr,
        "videoModel": video_model,
        "duration": round(storyboard.total_duration_s),
        "referenceImageUrls": [k.reference_image for k in storyboard.keyframes],
    }
    steps = [{"id": "survol_continu", "toolName": "generate_video", "arguments": arguments}]

    return {
        "input": {"title": f"Survol drone continu — {storyboard.address}", "steps": steps},
        "token_type": "auto",
        "confirm_cost": False,
    }


def _is_public_https_url(value: str) -> bool:
    lowered = value.lower()
    if not lowered.startswith("https://"):
        return False
    return "localhost" not in lowered and "127.0.0.1" not in lowered


__all__ = [
    "BEST_EFFORT_SCHEMA",
    "MAX_REFERENCE_IMAGES",
    "SogniError",
    "build_workflow_payload",
    "probe",
]
