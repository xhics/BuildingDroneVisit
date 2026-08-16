# État d'implémentation — BuildingDroneVisit

**Photographie vérifiée :** 16 août 2026  
**Jalon stable observé :** `85a1e48`  
**Pilote :** WelcomINNS Boucherville

Ce document répond à la question « qu'est-ce qui fonctionne réellement
aujourd'hui ? ». Il ne remplace ni la destination décrite par le plan directeur,
ni les preuves stockées dans `work/<hotel>/`.

Statuts :

- **livré** : chemin exécutable, contrat et tests présents ;
- **partiel** : mécanisme réel, mais raccord ou résultat requis manquant ;
- **en cours** : modifications non stabilisées au moment de la photographie ;
- **absent** : prévu dans l'architecture, pas de chemin exécutable ;
- **hors lot courant** : volontairement repoussé après le Lot 1B.

---

## 1. Verdict global

| Périmètre | État | Verdict |
|---|---|---|
| Socle reproductible | livré | CLI, schémas, profils, politique, workspace, smoke |
| Portabilité second site | livré | territoire et CRS dynamiques, refus hors emprise |
| Vérité du site | partiel | structure solide, neuf objets encore non résolus |
| Revue et qualification | livré | décisions et protocoles append-only |
| Collecte ciblée V2 | partiel | boucle réelle jusqu'à l'aperçu ; expansion Mapillary non raccordée |
| LiDAR, terrain, toiture | livré | dérivations qualifiées comme inférences |
| Orthophoto et cadastre | partiel | catalogue présent, acquisition absente |
| Orientation des façades | partiel | décision à 227,89°, aval non recalculé |
| Contexte et contraintes caméra | absent | livrables finaux Lot 1B non produits |
| Router | absent | enum seulement, aucune décision exécutable |
| SfM et reconstruction | hors lot courant | Lot 2 non commencé |
| Orchestrateur Phase 1 | partiel | seules les sous-commandes modernes sont réelles |

**Conclusion :** les mécanismes du Lot 1B sont avancés, mais sa définition de
DONE n'est pas atteinte. Les tests verts ne compensent pas l'absence de vérité
sur l'entrée, le stationnement, la parcelle et la couverture finale.

---

## 2. État factuel du pilote

### 2.1 Corpus

| Mesure | Valeur |
|---|---:|
| Assets | 335 |
| `context_lock` | 295 |
| `photo_geometry` | 9 |
| `reference_only` | 28 |
| `identity_evidence` | 3 |
| Droits `open_data` | 189 |
| Droits `public_uncleared` | 134 |
| Droits `unknown` | 12 |
| Assets positionnés | 313 |

Les six acquisitions ciblées les plus récentes ont été publiées atomiquement,
puis évaluées. Elles restent des références : aucune n'a été promue par le seul
fait d'avoir été téléchargée.

### 2.2 Objets du site

| Objet | État courant | Lecture |
|---|---|---|
| `BUILDING_MAIN` | confirmed | empreinte OSM confirmée |
| `ACCESS_ROAD_MAIN` | inferred | accès conditionnel, géométrie disponible |
| `ROOFLINE_MAIN` | inferred | toiture LiDAR qualifiée |
| `TERRAIN_MAIN` | inferred | terrain interpolé et qualifié |
| `PROPERTY_SIGN` | inferred | existence plausible, géométrie non établie |
| `PARKING_HOTEL` | unresolved | association précédente réfutée |
| `ENTRANCE_MAIN_CURRENT` | unresolved | emplacement et état courant non établis |
| `PROPERTY_PARCEL` | unresolved | cadastre non acquis |
| `FACADE_PRIMARY/LEFT/RIGHT/REAR` | unresolved | dépendaient de l'orientation à réappliquer |
| `DRIVEWAY_MAIN` | unresolved | existence non établie |
| `PARK_AND_RIDE` | unresolved | distinction attendue, absence non vérifiée |

Total : 14 objets, dont 1 confirmé, 4 inférés et 9 non résolus.

### 2.3 Besoins et aperçus

| Mesure | Valeur |
|---|---:|
| Besoins | 8 |
| Besoins ouverts | 8 |
| Constats d'aperçu établis | 0 |
| Constats réfutés | 9 |
| Constats indécis | 0 |

`refuted` qualifie l'élément de preuve, pas le besoin. Un besoin dont tous les
aperçus sont réfutés reste ouvert.

### 2.4 Géospatial et visibilité

| Élément | Résultat |
|---|---|
| LiDAR | tuile réelle acquise et hachée |
| DTM | 100 % défini sur l'emprise, interpolé sous le bâtiment |
| Toiture | environ 96,9 % observée |
| nDSM | environ 96,9 % valide |
| Hauteur médiane | environ 10,26 m |
| Qualification | `inferred`, confiance moyenne, provisoire |
| Assets évalués en visibilité | 307 |
| Directions dégagées en plan | 215 |
| Partiels | 49 |
| À risque exclusif | 43 |
| Blocages verticaux prouvés | 0 |

« Dégagé » signifie qu'une direction vers l'empreinte n'est pas masquée en
plan. Cela n'établit ni l'identité, ni le cadrage, ni l'utilité géométrique.

---

## 3. Gate du Lot 1

Le plan directeur exige que le bon bâtiment, son entrée actuelle et son
stationnement soient confirmés par des preuves enregistrées.

| Critère | État | Verdict |
|---|---|---|
| Bon bâtiment | `BUILDING_MAIN confirmed` | atteint |
| Entrée actuelle | unresolved | échec |
| Stationnement hôtel | unresolved | échec |
| Corpus minimal autorisé | droits encore non clarifiés sur 146 assets | partiel |
| Séparation temporelle de l'entrée | achèvement des travaux inconnu | échec |

**Lot 1 : non accepté.**

---

## 4. Avancement du Lot 1B

| Étape du plan | État | Ce qui manque pour fermer |
|---|---|---|
| 1. Structure de vérité | livré | aucun raccord critique connu |
| 2. Déduplication | partiel | embedding recadrage/filigrane |
| 3. Catégorisation | livré | élargir la validation à plusieurs sites |
| 4. Street View multi-position | livré | requalifier après nouvelle orientation |
| 5. Sources photographiques | partiel | expansion Mapillary et familles prioritaires non raccordées |
| 6. LiDAR/orthophoto | partiel | orthophoto et cadastre absents |
| 7. Contexte et couverture | partiel | manifeste de contexte et contraintes caméra |
| 8. Capture complémentaire | absent | dépend du Router et du rapport de couverture |

### Livrables finaux encore absents

- `coverage/coverage_report.json` canonique ;
- `coverage/zone_confidence.geojson` ;
- `coverage/camera_constraints.json` ;
- `coverage/context_manifest.json` ;
- `coverage/capture_brief.md`, si nécessaire ;
- `LOT_1B_REPORT.md` ;
- `router_decision.json`.

---

## 5. Blocages prioritaires

### P0 — Recalculer tout ce qui dépend de l'orientation

L'orientation courante vaut `227,89°`, mais les 313 assets positionnés portent
encore un secteur calculé depuis l'ancienne orientation. La confrontation
directe rend 313 divergences sur 313.

À fermer atomiquement :

1. valider et publier la décision d'orientation par une commande officielle ;
2. recalculer les secteurs des assets ;
3. périmer visibilité, candidats, recherches, évaluations et plans dépendants ;
4. reconstruire les besoins ciblables ;
5. relancer `demands assess`.

### P0 — Fermer la décision d'orientation

Les tests algorithmiques existent, mais la preuve persistée doit aussi vérifier
que les checksums, positions, segments et normales correspondent aux assets et
à l'empreinte courante. Le calcul du segment observé ne doit pas dépendre d'un
indice fourni sans confrontation.

### P1 — Ne plus reproposer un aperçu réfuté

Le raccord est livré dans les derniers commits : un couple asset/besoin réfuté
est retiré avant la mesure et ne consomme plus le budget. Il doit être vérifié
sur la prochaine découverte réelle après recalcul des secteurs.

### P1 — Implémenter le Router

Entrées minimales :

- besoins courants et preuves ;
- état des objets critiques ;
- couverture photo et géométrique ;
- zones proxy et lacunes ;
- droits ;
- possibilité de capture complémentaire.

Sortie : une décision versionnée et motivée, jamais une conclusion reconstruite
à la main depuis plusieurs rapports.

### P1 — Fermer contexte, orthophoto et cadastre

Sans ces éléments, le système sait mesurer le volume mais pas encore verrouiller
proprement la parcelle et tout l'environnement visible.

### P2 — Remplacer l'orchestrateur historique

`run-phase1` doit appeler les mêmes contrats que les sous-commandes modernes.
Il ne doit pas créer une troisième variante de collecte ou de péremption.

---

## 6. Phase 1 après le Lot 1B

Les éléments suivants restent volontairement hors du lot courant :

- hloc et LightGlue ;
- pycolmap et rapport G5 ;
- reconstruction photo-first ou hybride ;
- Brush ou autre représentation 3D ;
- alignement et composite ;
- carte de confiance finale ;
- validation `ENVIRONMENT_3D_READY`.

Les répertoires `03_preflight`, `04_masks`, `05_colmap`, `07_reconstruction`,
`08_composite`, `09_confidence` et `10_validation` ne contiennent encore aucun
livrable de production.

---

## 7. Dette documentaire fermée par les documents actuels

Avant cette photographie, les plans ne décrivaient pas les mécanismes pourtant
implémentés suivants :

- profils, politique et facettes ;
- manifeste de site et artefacts dérivés ;
- revue aveugle et aptitude géométrique ;
- besoins et recommandations par couple ;
- recherche adaptative ;
- plan, mesure, consentement et invalidation ;
- transport HTTP et transaction d'acquisition ;
- constats d'aperçu ;
- visibilité multi-rayons ;
- référentiels verticaux ;
- orientation depuis les segments de façade.

Ils sont désormais décrits dans `ARCHITECTURE_ACTUELLE.md` et justifiés dans
`DECISIONS_ARCHITECTURE.md`.

---

## 8. Vérification standard

La validation technique à rejouer avant de déplacer un Gate :

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m compileall -q src tests
.venv/bin/python -m pip check
.venv/bin/hotel-pipeline smoke
git diff --check
```

Le résultat doit être rapporté séparément du verdict métier.
