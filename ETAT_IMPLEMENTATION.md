# État d'implémentation — BuildingDroneVisit

**Photographie vérifiée :** 17 août 2026
**Jalon stable observé :** `49b5f0f` + livrables locaux non commités en validation
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
| Vérité du site | partiel | structure solide, cinq objets encore non résolus |
| Revue et qualification | livré | décisions et protocoles append-only |
| Collecte ciblée V2 | partiel | boucle réelle jusqu'à l'aperçu ; expansion Mapillary non raccordée |
| Registre des sources | livré, campagne incomplète | 2 familles requises sur 15 closes ; aucune absence masquée |
| LiDAR, terrain, toiture | livré | dérivations qualifiées comme inférences |
| Orthophoto et cadastre | partiel | état de couverture publié ; orthophoto non acquise, cadastre manuel |
| Orientation des façades | livré | décision à 227,89° propagée et vérifiée indépendamment |
| Contexte et contraintes caméra | livré | manifestes canoniques produits sous `coverage/` |
| Router | livré pour B/D | décision Path D publiée ; Path A/C restent hors contrat |
| Paquet 3D provider-agnostic | livré comme proxy | OBJ, rasters, orbite virtuelle, prompts et verdict Phase 1 |
| SfM et reconstruction | hors lot courant | Lot 2 non commencé |
| Orchestrateur Phase 1 | livré pour le Lot 1B | étape `lot1b` déléguant aux contrats modernes |

**Conclusion :** les mécanismes du Lot 1B sont avancés, mais sa définition de
DONE n'est pas atteinte. Les tests verts ne compensent pas l'absence de vérité
sur l'entrée, le stationnement et la parcelle, ni la preuve photo de l'accès.

---

## 2. État factuel du pilote

### 2.1 Corpus

| Mesure | Valeur |
|---|---:|
| Assets | 335 |
| `context_lock` | 304 |
| `photo_geometry` | 9 |
| `reference_only` | 16 |
| `identity_evidence` | 5 |
| `reject` | 1 |
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
| `FACADE_PRIMARY/LEFT/RIGHT/REAR` | inferred | segments d'empreinte réinstanciés depuis l'orientation |
| `DRIVEWAY_MAIN` | unresolved | existence non établie |
| `PARK_AND_RIDE` | unresolved | distinction attendue, absence non vérifiée |

Total : 14 objets, dont 1 confirmé, 8 inférés et 5 non résolus.

### 2.3 Besoins et aperçus

| Mesure | Valeur |
|---|---:|
| Besoins | 7 |
| Besoins ouverts | 6 |
| Besoins partiellement couverts | 1 |
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
| Run courant | 313 assets évalués, appliqué et empreint |
| Visibilité | 219 clear, 50 partial, 44 at_risk, 0 blocked |
| Corridors | 129, dont 100 utiles et 29 inconnus |

Les anciens nombres de visibilité restent auditables, mais le run appliqué
ci-dessus est le seul courant. « Dégagé » signifie seulement qu'une direction
vers l'empreinte n'est pas masquée en plan ; cela n'établit ni l'identité, ni le
cadrage, ni l'utilité géométrique.

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
| 2. Déduplication | livré | 335/335 courants, 269 photos, 209 points de vue ; hash robuste appliqué et régressions recadrage/filigrane/distincte passées |
| 3. Catégorisation | livré | élargir la validation à plusieurs sites |
| 4. Street View multi-position | livré | visibilité recalculée et appliquée après orientation |
| 5. Sources photographiques | partiel | registre canonique : 2/15 familles requises closes |
| 6. LiDAR/orthophoto | partiel | orthophoto CMM couverte mais non acquise ; cadastre manuel |
| 7. Contexte et couverture | livré | rapports canoniques produits, cinq objets restent non résolus |
| 8. Capture complémentaire | en cours | besoin ACCESS_ROAD_MAIN ciblé ; brouillon `20260817T031056701995Z`, passage réseau non engagé |

### Livrables finaux produits

- `coverage/coverage_report.json` canonique ;
- `coverage/zone_confidence.geojson` ;
- `coverage/camera_constraints.json` ;
- `coverage/context_manifest.json` ;
- `coverage/capture_brief.md`, si nécessaire ;
- `work/<hotel>/LOT_1B_REPORT.md` et la synthèse pilote `LOT_1B_REPORT.md` ;
- décision Router versionnée sous `10_validation/`.
- `00_manifest/source_registry.json`, avec état, preuve et motif par famille.

---

## 5. Blocages prioritaires

### P0 fermé — Propager l'orientation

L'orientation courante vaut `227,89°`. `orientation apply` a :

- vérifié l'empreinte du bâtiment et les SHA-256 des deux preuves ;
- réinstancié les quatre façades ;
- recalculé les 313 secteurs avec `bearing_deg` ;
- périmé 20 productions de visibilité, sans toucher au LiDAR ni à sa
  qualification ;
- activé le manifeste canonique à 7 besoins ;
- permis une évaluation à 6 besoins ouverts et 1 partiellement couvert ;
- promu le manifeste courant à 916 candidats sans appel réseau.

Contrôle indépendant : `pyproj.Geod.inv`, distinct du code d'application,
retrouve zéro divergence sur les 313 assets positionnés.

### P1 — Ne plus reproposer un aperçu réfuté

Le raccord est livré dans les derniers commits : un couple asset/besoin réfuté
est retiré avant la mesure et ne consomme plus le budget. La découverte courante
à 916 candidats confirme qu'aucun des neuf couples réfutés n'a été reproposé.

### P1 fermé — Router

Entrées minimales :

- besoins courants et preuves ;
- état des objets critiques ;
- couverture photo et géométrique ;
- zones proxy et lacunes ;
- droits ;
- possibilité de capture complémentaire.

Sortie courante : `PATH_D_HYBRID / CAPTURE_REQUIRED`. Les façades sont couvertes
par les proxies qualifiés ; `ACCESS_ROAD_MAIN`, ciblable mais sans preuve photo
ni proxy routier, porte seul la demande de capture.

### P1 partiel — Contexte, orthophoto et cadastre

Le contexte et les contraintes caméra sont publiés. L'orthophoto CMM 2023 est
déclarée couverte pour le territoire à 5 m et reste non acquise : elle verrouille
le contexte, jamais le toit ou la parcelle. Infolot demande une acquisition
manuelle ; `PROPERTY_PARCEL` reste donc `unresolved`.

### P2 fermé — Orchestrateur raccordé aux contrats modernes

`run-phase1` traverse désormais une étape `lot1b`, placée entre `collect` et
`preflight`. Elle appelle les **mêmes** fonctions que `sources registry`,
`coverage build` et `scene build` : aucune collecte ni péremption n'est
réécrite, et aucune troisième variante n'a été introduite.

La décision du Router n'est jamais rejouée par l'étape : elle est lue. Sans
décision publiée, la traversée s'arrête plutôt que de citer une route
inexistante. Un prérequis absent produit un arrêt documenté, pas une trace
brute.

Sur le corpus pilote, la traversée republie les livrables canoniques puis
s'arrête sur le gate réel : campagne de sources incomplète, 2/15 familles
requises closes. Le verdict et la route restent inchangés.

### P2 bis — Reçus d'indisponibilité de sources

L'état `unavailable_documented` existait au contrat sans qu'aucun chemin ne
puisse le produire : une famille sans collecteur restait indéfiniment ouverte.
Deux commandes append-only comblent ce vide :

```bash
hotel-pipeline sources unavailable <hotel> <famille> --reason <motif> --by <auteur>
hotel-pipeline sources reopen <hotel> <famille> --reason <motif> --by <auteur>
```

Un reçu documente une indisponibilité **observée**. Une interrogation courante
prime toujours sur un reçu périmé, afin qu'un constat ancien ne masque jamais
une preuve réelle. Le retrait conserve l'historique.

Émettre les reçus reste une décision humaine : le mécanisme est livré, la
campagne n'est pas close pour autant.

---

## 6. Verdict Phase 1 et paquet hybride

La commande locale suivante produit un paquet adressé par les empreintes de
ses entrées :

```bash
hotel-pipeline scene build welcominns-boucherville
```

Le pointeur canonique est
`08_composite/scene_package_current.json`. Le paquet courant contient :

- un volume OBJ extrudé depuis l'empreinte confirmée, classé `proxy` ;
- les rasters actifs DTM, DSM de toiture et nDSM, relus par SHA-256 ;
- 12 poses d'une orbite **virtuelle** dérivée du FOV de la politique ;
- les contraintes caméra, la carte de confiance et les claims interdits ;
- un contrat de prompts sans appel à un fournisseur vidéo réel ;
- un script d'import Blender et un verdict Phase 1 typé.

Ce résultat ne vaut pas reconstruction photo-réaliste. Le verdict courant est
`NEEDS_AUTHORIZED_CAPTURE`, pour quatre raisons prouvées : entrée et
stationnement non établis, un seul point de vue indépendant, aucune mesure SfM
et aucune approbation humaine finale.

## 7. Phase 1 restant à exécuter

Les éléments suivants restent non exécutés :

- hloc et LightGlue ;
- pycolmap et rapport G5 ;
- reconstruction photo-first ou hybride ;
- Brush ou autre représentation 3D ;
- alignement SfM et composite photoréaliste ;
- carte de confiance finale ;
- validation `ENVIRONMENT_3D_READY`.

`08_composite` contient désormais un paquet hybride/proxy inspectable. Les
répertoires `05_colmap` et `07_reconstruction` ne portent toujours aucun
résultat SfM ou Brush ; cette absence est un gate, pas un détail d'affichage.

---

## 8. Dette documentaire fermée par les documents actuels

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

## 9. Vérification standard

La validation technique à rejouer avant de déplacer un Gate :

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m compileall -q src tests
.venv/bin/python -m pip check
.venv/bin/hotel-pipeline smoke
git diff --check
```

Le résultat doit être rapporté séparément du verdict métier.
