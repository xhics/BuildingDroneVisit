# Plan général directeur — WelcomINNS Boucherville

## Phase 1 — Reconstruction d’un environnement 3D inspectable

**Version recadrée selon le document d’accompagnement V3.2**  
**Hôtel pilote :** Hôtel WelcomINNS, 1195 rue Ampère, Boucherville, Québec J4B 7M6

---

## 1. Objet du plan

Ce plan organise la Phase 1 du pipeline pour le WelcomINNS de Boucherville.

L’objectif n’est pas encore de produire une vidéo. Il est de démontrer qu’à partir d’une adresse, de sources autorisées et de clés de fournisseurs, le système peut produire un environnement 3D fidèle, traçable, inspectable et suffisamment fiable pour devenir plus tard la base d’un système de tournage virtuel.

```text
ADRESSE + CONFIGURATION + CLÉS API
                ↓
COLLECTE ET QUALIFICATION DES DONNÉES
                ↓
RECONSTRUCTION / FUSION 3D
                ↓
VALIDATION + CARTE DE CONFIANCE
                ↓
ENVIRONMENT_3D_READY
                ╳
          ARRÊT DE LA PHASE 1
```

Le succès du pilote est donc un environnement 3D reconnaissable du véritable WelcomINNS, accompagné de ses preuves, de ses limites et de ses scores de confiance.

## 2. Frontière fonctionnelle

### Inclus dans la Phase 1

- résolution de la propriété et du bâtiment exacts ;
- contrôle de la configuration et de la disponibilité des fournisseurs ;
- collecte de photographies, métadonnées et données géographiques ;
- qualification des droits et de l’éligibilité des assets ;
- déduplication, classification, qualité et analyse de couverture ;
- test SfM réel ;
- choix explicite d’une route de reconstruction ;
- reconstruction photo-first, geo-first ou hybride ;
- géoréférencement, alignement et fusion ;
- identification des objets critiques ;
- environnement 3D composite inspectable ;
- carte de confiance par zone ;
- validation algorithmique, visuelle et humaine ;
- rapports, tests et artefacts reproductibles ;
- verdict `ENVIRONMENT_3D_READY`, `NEEDS_CAPTURE` ou `REJECTED`.

### Explicitement hors Phase 1

- scénario de visite ;
- nombre, durée et découpage des plans ;
- trajectoires finales de caméra ;
- tournage virtuel intérieur ou extérieur ;
- transitions extérieur/intérieur ;
- génération vidéo par IA ;
- sélection ou consommation d’un fournisseur vidéo ;
- ComfyUI vidéo de production ;
- conditionnement depth/canny pour la vidéo finale ;
- continuité entre les plans ;
- montage, musique, voix, titres et livraison vidéo ;
- contrôle qualité post-génération vidéo.

Ces éléments devront faire l’objet d’un plan Phase 2 distinct. Ils ne constituent ni des livrables ni des critères de succès du pilote actuel.

## 3. Contexte spécifique au WelcomINNS

Le site officiel présente un établissement de 116 chambres, avec stationnement gratuit et piscine intérieure. La piscine ne doit donc pas être interprétée comme un objet extérieur du terrain. [Site officiel WelcomINNS](https://hotelwelcominns.com/fr/)

### Atouts du site pilote

- adresse officielle précise ;
- bâtiment et accès visibles depuis l’espace public ;
- stationnement et réseau routier fournissant du contexte spatial ;
- photographies publiques associées à l’établissement ;
- empreintes de bâtiments et données routières disponibles dans la zone.

### Risques à lever avant reconstruction

- l’empreinte cartographique du bâtiment n’est pas clairement nommée comme hôtel ;
- un grand stationnement incitatif voisin ne doit pas être confondu avec celui de l’hôtel ;
- le code postal retourné par certaines données cartographiques diffère du code postal officiel ;
- les galeries publiques mélangent des intérieurs et des extérieurs ;
- la diversité réelle des vues extérieures demeure à mesurer ;
- la Ville a approuvé une rénovation de l’entrée principale en 2024 ; les images d’avant et d’après rénovation ne doivent pas être fusionnées sans contrôle. [Décision municipale](https://www.boucherville.ca/wp-content/uploads/2024/09/PV_seance_240916.pdf)

La première vérité à établir est donc l’association exacte entre l’adresse, la parcelle, le bâtiment, l’entrée actuelle et le stationnement privé de l’hôtel.

## 4. Invariants et objets critiques

### Objets à identifier et valider

```text
PROPERTY_WELCOMINNS
BUILDING_MAIN
ENTRANCE_MAIN_CURRENT
ROOFLINE_MAIN
FACADE_PRIMARY
FACADE_LEFT
FACADE_RIGHT
FACADE_REAR
HOTEL_SIGN
DRIVEWAY_MAIN
PARKING_HOTEL
RUE_AMPERE
TERRAIN_MAIN
```

### Objets à distinguer ou à exclure

```text
INDOOR_POOL
PARK_AND_RIDE_DE_MORTAGNE
NEIGHBOURING_HOTELS
UNRELATED_COMMERCIAL_BUILDINGS
AUTOROUTE_20_CONTEXT
```

`AUTOROUTE_20_CONTEXT` peut servir de contexte géographique simplifié, mais ne fait pas partie de la propriété à reconstruire avec un niveau de détail élevé.

Chaque objet critique doit contenir :

- identifiant stable ;
- catégorie ;
- géométrie ou localisation ;
- sources d’évidence ;
- date ou période probable ;
- état `confirmed`, `inferred`, `conflicted` ou `unresolved` ;
- score de confiance ;
- relations spatiales qualitatives ;
- corrections humaines et justification.

## 5. Architecture d’exécution du PoC

Le poste local sert au développement et au contrôle. La vision, le SfM et la reconstruction sont exécutés dans une seule VM GPU cloud, avec une image Docker reproductible.

```text
MAC / POSTE DE CONTRÔLE
        ↓ Git + CLI
VM GPU CLOUD — CIBLE INITIALE : RTX 4090 24 Go OU ÉQUIVALENT
        ├─ collecte et normalisation
        ├─ OpenCLIP / IQA / clustering
        ├─ Grounded-SAM2
        ├─ hloc + LightGlue + pycolmap
        ├─ Brush
        ├─ VoxCity / géospatial
        ├─ Blender headless (bpy)
        ├─ tests
        └─ rapports et artefacts
        ↓
work/welcominns-boucherville/...
```

Pour ce PoC, ne pas introduire Kubernetes, Prefect, une architecture distribuée ou du serverless. Une VM GPU, Docker et une CLI rejouable suffisent. L’allocation et l’arrêt automatiques de la VM viendront après la preuve technique.

## 6. Configuration, secrets et santé des fournisseurs

Les secrets sont injectés à l’exécution et ne sont jamais committés.

```text
.env                 # secret et ignoré par Git
.env.example         # noms de variables, sans valeur

GOOGLE_MAPS_API_KEY=
GOOGLE_PLACES_API_KEY=
MAPILLARY_TOKEN=
VISION_API_KEY=
```

Chaque adaptateur doit exposer un contrôle de configuration et de santé :

```text
hotel-pipeline provider-check

Google Places     ✓ configured
Street View       ✓ configured
Mapillary         ✗ missing token
OSM               ✓ no key required
Overture          ✓ no key required
```

L’absence d’une source optionnelle ne doit pas provoquer un échec global. Elle réduit la couverture disponible et influence le Preflight et le Router. Une source obligatoire indisponible doit produire une erreur explicite et actionnable.

## 7. Architecture fonctionnelle consolidée

```text
ADDRESS
  ↓
CONFIG + PROVIDER HEALTH
  ↓
COLLECTION
  ├─ photographies, métadonnées et sources
  ├─ OSM, Overture, DEM et données ouvertes
  └─ références autorisées
  ↓
NORMALIZATION + RIGHTS MANIFEST
  ↓
REFERENCE REASONER
  ↓
PREFLIGHT CASCADE G0 → G5
  ↓
ROUTER
  ├─ PATH A — 3D détaillée ouverte ou licenciée
  ├─ PATH B — photo-first : hloc / pycolmap / Brush
  ├─ PATH C — geo-first : VoxCity
  ├─ PATH D — hybride avec correction humaine
  └─ REJECT
  ↓
GEOREFERENCING + ALIGNMENT
  ↓
CRITICAL OBJECTS
  ↓
COMPOSITE 3D ENVIRONMENT
  ↓
CONFIDENCE MAP
  ↓
ALGORITHMIC + AI-ASSISTED + HUMAN VALIDATION
  ↓
ENVIRONMENT_3D_READY
```

## 8. Ordre réel de construction

Le pipeline complet ne doit pas être construit avant d’avoir démontré que le noyau de reconstruction fonctionne sur le WelcomINNS.

### Expérience 1 — Preuve de reconstruction

Chaîne minimale :

```text
PHOTOS AUTORISÉES ET RETENUES
        ↓
HLOC + LIGHTGLUE
        ↓
PYCOLMAP
        ↓
BRUSH
        ↓
SPLAT INSPECTABLE
```

Question : peut-on obtenir un splat dans lequel le véritable WelcomINNS est reconnaissable et navigable ?

Critères :

- bâtiment principal identifiable ;
- façade principale et entrée actuelle cohérentes ;
- taux d’enregistrement mesuré ;
- composante principale mesurée ;
- résultat consultable dans un viewer ;
- artefacts et zones absentes documentés.

Si la réponse est non, ne pas construire prématurément toute l’automatisation. Déterminer d’abord si le problème vient des données, des droits, de la connectivité visuelle, du matching ou de la méthode de reconstruction.

### Expérience 2 — Inspection 3D

Chaîne :

```text
SPLAT
  ↓
NETTOYAGE MINIMAL
  ↓
ENVIRONNEMENT INSPECTABLE
```

Question : les zones commerciales importantes — bâtiment, entrée, enseigne, accès et stationnement — sont-elles représentées sans ambiguïté majeure ?

### Expérience 3 — Préparation du futur master

Importer l’environnement dans Blender headless et vérifier seulement :

- chargement reproductible ;
- unités et axes ;
- géoréférencement ou transformation d’alignement ;
- visibilité des objets critiques ;
- possibilité technique d’un futur filming contrôlé.

Cette expérience ne doit pas créer de scénario, de plans caméra finaux ni de vidéo.

### Consolidation après la preuve

Les adaptateurs de collecte, les Gates complets, le Router, le Reference Reasoner et la carte de confiance sont construits après la preuve de reconstruction, ou en parallèle seulement s’ils accélèrent directement cette preuve.

## 9. Collecte et manifeste de données

Créer deux espaces strictement séparés :

```text
reference_only/
production_eligible/
```

Sources à inventorier :

- médias récents fournis ou autorisés par le WelcomINNS ;
- site officiel et brochures officielles ;
- médias sociaux officiels ;
- photographies publiques indexées ;
- données OSM et Overture ;
- DEM, orthophotos et données municipales ouvertes ;
- capture complémentaire autorisée, si nécessaire.

Une image publique reste `reference_only` tant que ses droits ne permettent pas son utilisation dans la reconstruction ou un traitement IA.

Chaque asset doit être validé par un modèle Pydantic équivalent à :

```text
AssetManifest
  id
  source
  source_url_or_id
  rights
  ai_eligible
  confidence
  category
  capture_year
  season
  device
  gps_if_available
  checksum
  derived_from[]
```

Champs additionnels utiles au pilote :

```text
exterior_or_interior
entrance_version
property_match_status
duplicate_group
production_eligible
```

Un champ obligatoire absent ou mal typé doit produire une erreur explicite. Aucun asset ne doit être routé silencieusement avec des métadonnées invalides.

## 10. Reference Reasoner

Le `Reference Reasoner` analyse les références sans produire de géométrie.

```text
ALL REFERENCES
      ↓
REFERENCE REASONER
      ├─ catégories : façade, entrée, aérien, stationnement, intérieur
      ├─ objets : BUILDING_MAIN, ENTRANCE_MAIN_CURRENT, PARKING_HOTEL
      ├─ relations spatiales qualitatives
      ├─ saison et année probables
      ├─ anomalies : ancienne entrée, hiver, nuit, rendu 3D
      └─ vues manquantes
      ↓
SPATIAL KNOWLEDGE GRAPH
```

Exemple :

```json
{
  "subject": "PARKING_HOTEL",
  "relation": "adjacent_to",
  "object": "BUILDING_MAIN",
  "confidence": 0.88,
  "evidence": ["img_014", "osm_feature_29382"]
}
```

Ce graphe contient des hypothèses qualitatives, jamais une vérité métrique. Les données géographiques et la reconstruction doivent confirmer ou rejeter ses relations.

## 11. Cascade Preflight G0 à G5

Les contrôles sont exécutés par coût croissant.

| Gate | But | Outil principal | Décision initiale |
|---|---|---|---|
| G0 | Inventaire brut | Comptage et métadonnées | Candidats suffisants |
| G1 | Déduplication perceptuelle | `imagededup` | Images uniques suffisantes |
| G2 | Extérieur / intérieur | OpenCLIP ou classifieur | Au moins 15 extérieurs exploitables |
| G3 | Qualité | `pyiqa` + règles | Flou, compression et faible utilité écartés |
| G4 | Diversité et connectivité probable | CLIP + graphe de retrieval | Vues diverses et graphe visuel plausible |
| G5 | SfM sparse réel | hloc + LightGlue + pycolmap | Reconstruction mesurée |

Seuils initiaux de G5 :

- `registration_rate >= 0.60` ;
- `main_component_ratio >= 0.70` des images enregistrées ;
- façade principale et entrée dans la composante principale ;
- aucune reconstruction double causée par le mélange avant/après rénovation ;
- échelle et orientation récupérables ou alignables.

CLIP ne doit pas être interprété comme un estimateur d’angle caméra. Il mesure surtout une diversité sémantique. Le graphe de retrieval complète ce signal avant le test métrique G5.

Ces seuils sont des valeurs de départ à recalibrer après environ cinq hôtels, sans réécrire les résultats historiques.

## 12. Router de reconstruction

Le Router décide à partir de scores et de règles. L’IA peut expliquer la décision, mais ne doit pas la prendre seule.

### Path A — 3D ouverte ou licenciée

À utiliser seulement si une donnée 3D suffisamment détaillée, autorisée et exportable existe réellement pour la propriété.

### Path B — Photo-first

Route privilégiée lorsque G5 confirme une couverture et un enregistrement suffisants.

```text
PHOTOS RETENUES
  ↓
MASQUES DYNAMIQUES ET CIEL, SI UTILES
  ↓
HLOC
  ├─ extraction de features compatible et licenciée
  └─ LightGlue matching
  ↓
PYCOLMAP MAPPER
  ↓
POSES + SPARSE MODEL
  ↓
BRUSH
  ↓
GAUSSIAN SPLAT
  ↓
NETTOYAGE + INSPECTION
```

Brush est l’option initiale. `gsplat` ou Nerfstudio restent des fallbacks documentés si une limite de qualité ou de fonctionnalité est démontrée.

### Path C — Geo-first

À utiliser lorsque les photos sont insuffisantes, mais que l’empreinte, la hauteur, le terrain et les données géographiques permettent un environnement proxy fiable via VoxCity et des sources ouvertes.

Cette route ne doit pas inventer des détails architecturaux non observés.

### Path D — Hybride

Assembler par zones :

| Zone | Source privilégiée |
|---|---|
| Entrée actuelle | Reconstruction photo si couverture récente suffisante |
| Façade principale | Photo-first |
| Façades latérales | Photo-first ou proxy selon la confiance |
| Façade arrière | Photo-first si fiable, sinon geo proxy explicite |
| Toiture | Données aériennes et géométrie contrôlée |
| Stationnement hôtel | Données cartographiques confirmées |
| Terrain | DEM |
| Autoroute et voisinage | Contexte géographique simplifié |

### Reject / Needs Capture

Refuser ou demander une capture autorisée lorsque :

- la propriété demeure ambiguë ;
- les droits de production sont insuffisants ;
- G5 échoue sans route geo-first crédible ;
- l’entrée actuelle ne peut être distinguée de l’ancienne ;
- le bâtiment est fusionné avec un voisin ;
- les objets commerciaux critiques ne peuvent pas être validés.

## 13. Segmentation

SAM2 n’est pas utilisé seul comme classifieur sémantique.

```text
GROUNDING / DÉTECTEUR VISION-LANGAGE
             ↓
            SAM2
             ↓
MASKS
  ├─ sky
  ├─ people
  ├─ vehicles
  ├─ selected_reflections
  ├─ entrance
  ├─ hotel_sign
  └─ parking_boundary
```

Les fenêtres, baies vitrées et surfaces réfléchissantes ne doivent pas être masquées aveuglément : elles peuvent porter une information architecturale essentielle. Le traitement est configurable par classe et par cas.

## 14. Géoréférencement, alignement et environnement composite

Les coordonnées, poses et transformations sont produites par des outils déterministes : pycolmap, ICP, PDAL, Open3D et traitements géographiques appropriés.

Sorties attendues :

```text
property_boundary.geojson
building_main.geojson
hotel_parking.geojson
access_network.geojson
critical_objects.json
spatial_manifest.json
colmap/
splat/
geo/
composite_environment/
alignment_report.json
```

L’environnement composite doit :

- isoler le bon bâtiment et le stationnement de l’hôtel ;
- conserver la volumétrie et la toiture observables ;
- représenter l’entrée actuelle sans mélange temporel ;
- séparer l’hôtel des bâtiments commerciaux voisins ;
- distinguer les zones reconstruites des proxies ;
- enregistrer la provenance de chaque zone ;
- être inspectable et chargeable de façon reproductible.

## 15. Rôle exact de l’IA générative

Principe : l’IA interprète, classe, relie, diagnostique et propose. Elle ne remplace jamais le noyau métrique ou géométrique.

| Étape | Rôle autorisé de l’IA | Garde-fou |
|---|---|---|
| Résolution de propriété | Lever les ambiguïtés de nom et d’adresse | Ne jamais inventer de coordonnées |
| Collecte | Prioriser les pages et catégoriser les sources | Respecter les droits et fournisseurs |
| Métadonnées | Détecter des incohérences | Décision finale par règles et manifeste |
| Déduplication | Aucun | Algorithme spécialisé |
| Extérieur / intérieur | Classification vision-langage | Conserver le score |
| Qualité | Juger l’utilité pour reconstruction | IQA déterministe prioritaire |
| Diversité | Décrire les façades ou vues manquantes | Ne pas assimiler CLIP à une pose |
| Segmentation | Détection sémantique et proposition de masques | Masques contrôlés |
| SfM / matching | Aucun | hloc, LightGlue et pycolmap |
| Router | Expliquer la route et ses risques | Route décidée par scores et règles |
| Reconstruction | Aucun dans le noyau | Outils déterministes |
| Objets critiques | Identifier et relier les preuves | Géométrie confirmée par sources |
| Alignement | Aucun | Outils géométriques déterministes |
| Fusion | Proposer une source prioritaire par zone | Aucune transformation libre |
| Validation | Comparer captures 3D et références | Conserver les scores algorithmiques |
| Rapport | Synthétiser risques et zones faibles | Citer l’évidence interne |

## 16. Carte de confiance

La sortie Phase 1 comprend une carte de confiance par zone ou secteur.

```text
CONFIANCE > 70 %    zone forte
CONFIANCE 40–70 %  zone moyenne, prudence
CONFIANCE < 40 %    zone faible, correction ou exclusion future
```

La confiance combine notamment :

- couverture des références ;
- densité de reconstruction ;
- erreur de reprojection ;
- taux d’enregistrement local ;
- cohérence avec les images de validation ;
- confiance des objets critiques ;
- qualité de l’alignement ;
- statut reconstruit ou proxy ;
- conflit temporel éventuel.

Même si les règles de caméra appartiennent à la Phase 2, cette carte est produite dès maintenant afin qu’un futur moteur de filming sache quelles zones sont fiables.

## 17. Validation et tests obligatoires

### Validation du WelcomINNS

- inspection visuelle depuis plusieurs points de vue ;
- comparaison aux références non utilisées pour la reconstruction ;
- contrôle de la silhouette, de la toiture et des façades ;
- contrôle de l’entrée actuelle et de l’enseigne ;
- contrôle de la séparation bâtiment / stationnement ;
- contrôle de la séparation avec les voisins ;
- contrôle du géoréférencement et des unités ;
- documentation des zones faibles et corrections manuelles.

### Stratégie de tests

| Niveau | Cible | Principe |
|---|---|---|
| Unit tests | Scoring, Router, Pydantic, Gates, calculs géographiques | Rapides et sans réseau |
| Adapter tests | Fournisseurs vers le modèle interne | Fixtures enregistrées |
| Integration tests | Petit corpus vers hloc/pycolmap et modèle produit | CPU/GPU selon le test |
| Golden hotel tests | Cas easy, medium et bad data | Détection des régressions |
| Provider health tests | Clés, configuration et quotas de base | Aucun secret dans les logs |

Assertions typiques :

```text
expected_path
registration_rate_min
main_component_ratio_min
critical_objects_expected
output_3d_exists
expected_reject
```

Structure :

```text
tests/golden/
  hotel_easy/
  hotel_medium/
  hotel_bad_data/
  welcominns_boucherville/
```

## 18. CLI, reprise et arborescence

Chaque étape doit être rejouable indépendamment. Typer fournit la CLI, un Makefile orchestre le PoC et `diskcache` évite les appels externes répétés.

```text
hotel-pipeline provider-check
hotel-pipeline collect welcominns-boucherville
hotel-pipeline preflight welcominns-boucherville
hotel-pipeline reconstruct welcominns-boucherville
hotel-pipeline align welcominns-boucherville
hotel-pipeline validate welcominns-boucherville
hotel-pipeline run-phase1 welcominns-boucherville
```

```text
work/welcominns-boucherville/
  00_manifest/
  01_sources/
  02_images/
    reference_only/
    production_eligible/
  03_preflight/
  04_masks/
  05_colmap/
  06_geo/
  07_reconstruction/
  08_composite/
  09_confidence/
  10_validation/
  report.json
```

Chaque commande doit :

- lire des entrées versionnées ;
- produire des sorties versionnées ;
- enregistrer paramètres et versions des outils ;
- être idempotente ou détecter proprement un résultat existant ;
- permettre `--force` uniquement de manière explicite ;
- ne jamais exposer de secret dans les journaux.

## 19. Livrables de la Phase 1

```text
project_manifest.json
asset_manifest.json
rights_manifest.json
provider_health.json
spatial_manifest.json
critical_objects.json
spatial_knowledge_graph.json
preflight_report.json
router_decision.json
colmap/
splat/
geo/
composite_environment/
alignment_report.json
confidence_map.json
validation_report.json
manual_corrections.json
cost_report.json
human_time_report.json
lessons_learned.md
report.json
```

Aucun fichier vidéo n’est un livrable de cette phase.

## 20. Définition de DONE — `ENVIRONMENT_3D_READY`

La Phase 1 est terminée seulement si :

- l’environnement 3D est inspectable ;
- il est géoréférencé ou aligné avec une précision documentée ;
- le bâtiment principal et les objets critiques sont identifiés ;
- l’entrée actuelle est distinguée de toute version antérieure ;
- les sources et droits de chaque asset sont tracés ;
- les scores des Gates et la route réellement suivie sont enregistrés ;
- une carte de confiance par zone existe ;
- les zones faibles, anomalies et corrections manuelles sont documentées ;
- le chargement dans Blender est reproductible ;
- aucune donnée Google Photorealistic 3D Tiles n’a servi à produire le résultat ;
- les tests obligatoires passent ;
- une revue humaine approuve explicitement le statut final.

Exemple de rapport :

```json
{
  "hotel_id": "welcominns-boucherville",
  "status": "ENVIRONMENT_3D_READY",
  "path": "PHOTO_FIRST",
  "registration_rate": 0.78,
  "main_component_ratio": 0.84,
  "building_confidence": 0.91,
  "entrance_confidence": 0.93,
  "parking_confidence": 0.82,
  "terrain_confidence": 0.89,
  "front_facade_confidence": 0.94,
  "rear_facade_confidence": 0.37,
  "manual_review_required": true
}
```

Les valeurs ci-dessus illustrent le format ; elles ne sont pas des mesures actuelles du WelcomINNS.

## 21. Branche Google Photorealistic 3D Tiles

Cette branche est extérieure au pipeline de production. Elle peut uniquement servir de benchmark visuel humain lorsque la couverture existe.

```text
GOOGLE PHOTOREALISTIC 3D TILES
              │
              ▼
       HUMAN BENCHMARK
              ↕
    ENVIRONMENT_3D_READY
```

Usages interdits :

```text
Google 3D → reconstruction
Google 3D → extraction automatique
Google 3D → référence ou entraînement IA
Google 3D → asset de production
```

## 22. Lots d’implémentation

### Lot 0 — Socle reproductible

- dépôt, Docker, CLI Typer et Makefile ;
- schémas Pydantic ;
- gestion `.env` et `provider-check` ;
- arborescence `work/<hotel>` ;
- journalisation sans secrets.

**Acceptation :** la VM GPU peut exécuter un smoke test depuis un clone propre.

### Lot 1 — Dataset minimal WelcomINNS

- résolution de propriété ;
- manifeste spatial ;
- inventaire et droits des photos ;
- séparation avant/après rénovation ;
- corpus minimal autorisé.

**Acceptation :** le bon bâtiment, son entrée actuelle et son stationnement sont confirmés par des preuves enregistrées.

### Lot 2 — Expérience de reconstruction

- hloc + LightGlue + pycolmap ;
- rapport G5 ;
- Brush ;
- viewer et inspection.

**Acceptation :** résultat reconnaissable et navigable, ou diagnostic documenté conduisant à `NEEDS_CAPTURE` ou à une autre route.

### Lot 3 — Inspection et Blender

- nettoyage minimal ;
- import headless ;
- unités, axes et transformation ;
- rapport d’exploitabilité future.

**Acceptation :** environnement chargeable de façon reproductible sans concevoir de tournage.

### Lot 4 — Collecte et qualification consolidées

- adaptateurs ;
- cache ;
- manifeste de droits ;
- Gates G0 à G4 ;
- Reference Reasoner.

**Acceptation :** collecte rejouable et chaque rejet expliqué.

### Lot 5 — Router et routes alternatives

- décisions déterministes ;
- path photo-first ;
- path geo-first ;
- path hybride ;
- rejet et demande de capture.

**Acceptation :** la même entrée et les mêmes scores produisent la même route.

### Lot 6 — Alignement, fusion et objets critiques

- géoréférencement ;
- environnement composite ;
- provenance par zone ;
- registre des objets critiques.

**Acceptation :** aucune confusion entre l’hôtel, le stationnement incitatif et les bâtiments voisins.

### Lot 7 — Confiance, validation et tests golden

- carte de confiance ;
- comparaisons aux références ;
- tests unitaires, adaptateurs, intégration et golden ;
- rapport humain.

**Acceptation :** les scores sont reproductibles et les zones faibles sont visibles, non dissimulées.

### Lot 8 — Verdict Phase 1

- rapport final ;
- coûts et temps humain ;
- leçons apprises ;
- décision de passage ou non à la Phase 2.

**Acceptation :** statut final explicite et justifié.

## 23. Décisions finales possibles

```text
ENVIRONMENT_3D_READY
NEEDS_AUTHORIZED_CAPTURE
NEEDS_MANUAL_CORRECTION
GEO_FIRST_PROXY_ONLY
REJECTED_PROPERTY_AMBIGUOUS
REJECTED_RIGHTS_INSUFFICIENT
REJECTED_DATA_INSUFFICIENT
```

Un `ENVIRONMENT_3D_READY` ne signifie pas que toutes les zones sont parfaites. Il signifie que les forces, limites et zones faibles sont mesurées, visibles et compatibles avec une décision responsable pour la Phase 2.

## 24. Ordre directeur obligatoire

1. créer le socle Docker/CLI minimal sur la VM GPU ;
2. confirmer la propriété, le bâtiment, l’entrée actuelle et le stationnement ;
3. obtenir un corpus extérieur actuel et juridiquement exploitable ;
4. tester immédiatement hloc, LightGlue et pycolmap sur ce corpus ;
5. produire et inspecter un premier splat Brush ;
6. vérifier son chargement et son alignement dans Blender ;
7. décider si le noyau de reconstruction est viable ;
8. consolider ensuite collecte, Gates, Router et Reference Reasoner ;
9. construire les routes geo-first et hybrides seulement selon les besoins observés ;
10. produire l’environnement composite, la carte de confiance et les tests ;
11. rendre le verdict Phase 1 ;
12. ouvrir un document Phase 2 séparé uniquement après `ENVIRONMENT_3D_READY`.

Le risque principal du pilote WelcomINNS n’est pas de produire rapidement une image séduisante. Il est de démontrer, preuves à l’appui, que le système a reconstruit le bon hôtel, dans sa version actuelle, sans confondre sa propriété avec le stationnement ou les bâtiments voisins, et sans masquer ses zones d’incertitude.
