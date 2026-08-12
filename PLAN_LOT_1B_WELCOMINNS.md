# Lot 1B — Couverture étendue et vérité visuelle WelcomINNS

Document d'exécution consolidé, sans duplication, de tout le travail restant avant le Lot 2.

Il complète `PLAN_DIRECTEUR_WELCOMINNS.md`, qui demeure la référence d'architecture. Le Lot 1B ne lance ni hloc, ni LightGlue, ni pycolmap, ni Brush. Il doit d'abord produire un corpus réellement diversifié, un paquet géospatial déterministe et des limites visibles qui empêchent toute invention ultérieure.

---

## 1. Résultat recherché

À partir de l'adresse du WelcomINNS Boucherville, produire un paquet d'entrée traçable qui établit :

1. l'identité du bon bâtiment, de l'entrée actuelle et du stationnement de l'hôtel ;
2. les vues photographiques réellement indépendantes de chaque secteur observable ;
3. le volume, le terrain et la toiture issus de données géographiques ;
4. le contexte environnant qui devra rester fidèle ;
5. les zones fiables, les zones proxy et les zones non observées ;
6. les vues complémentaires exactes à demander à l'hôtel seulement si elles demeurent nécessaires.

Le principe directeur est simple : **observer, mesurer ou masquer — jamais inventer**.

---

## 2. Périmètre et décisions déjà prises

### Inclus dans le Lot 1B

- extension de la collecte photographique ;
- exploration Street View depuis plusieurs positions ;
- déduplication exacte, perceptuelle et géométrique ;
- catégorisation multidimensionnelle ;
- validation d'appartenance et de période ;
- téléchargement du LiDAR, des orthophotos et des empreintes ;
- production des entrées déterministes nécessaires au Path D hybride ;
- qualification du contexte environnant ;
- carte de couverture et contraintes futures de caméra ;
- brief de capture complémentaire, si les sources distantes ne suffisent pas.

### Explicitement hors périmètre

- hloc, LightGlue et pycolmap ;
- entraînement ou rendu de Gaussian Splat ;
- Router final des chemins A à D ;
- environnement composite final dans Blender ;
- conception d'un mouvement de caméra ou d'une vidéo ;
- génération visuelle destinée à remplacer une zone inconnue.

### Politique de sources retenue pour le pilote

Toutes les sources publiques ou accessibles sont considérées utilisables à la demande de l'opérateur. Les droits ne bloquent donc pas la collecte ni la qualification. La provenance, les conditions et l'éventuel caractère `rights_encumbered` restent néanmoins enregistrés dans le manifeste afin que cette décision ne disparaisse pas.

---

## 3. État vérifié au début du Lot 1B

### Déjà accompli

- `BUILDING_MAIN` confirmé comme `way/54581348` par inspection visuelle ;
- propriété distinguée de l'Hôtel Mortagne ;
- tests de séparation géométrique présents ;
- collecteurs Mapillary, Street View Static, Google Places et site officiel opérationnels ;
- collecteurs Wikimedia Commons, Flickr et TripAdvisor présents dans le code ;
- pHash, contrôle de qualité de base, OpenCLIP, visibilité géométrique et OCR présents ;
- logique OCR corrigée : le terme attendu prime sur une exclusion, les recherches utilisent les limites de mots et le texte réel « De Mortagne » est couvert par un test de régression ;
- provenance, droits assumés et version d'entrée représentés dans le manifeste.

### Corpus actuellement enregistré

| Mesure | Valeur actuelle |
|---|---:|
| Images totales | 219 |
| Mapillary | 189 |
| Street View | 8 |
| Google Places | 10 |
| Site officiel | 12 |
| Images avec position | 197 |
| Images extérieures | 203 |
| Images cadrant réellement le bâtiment | 20 |
| Vues de façade principale | 17 |
| Vues latérales éparses | 3 |
| Arrière | 0 |
| Toiture exploitable au sol | 0 |
| Images sans position | 22 |
| OCR lisible parmi ces 22 | 6 |
| OCR confirmant explicitement WelcomINNS | 1 |

Le seuil quantitatif de 20 extérieurs est atteint, mais pas la diversité. Les 219 fichiers représentent principalement une seule façade vue depuis la rue Ampère.

### Limites déjà mesurées et à ne pas réexaminer en boucle

- Wikimedia Commons n'a livré aucune image pertinente du WelcomINNS dans la zone testée ;
- Panoramax ne possède aucune image jusqu'à environ 1 km de l'adresse lors de la vérification ;
- KartaView n'a pas répondu aux requêtes de zone ;
- Expedia, Hotels.com, Momondo et Kayak republient principalement une même famille de médias ;
- l'OCR seul ne peut pas lever l'appartenance de la majorité des images promotionnelles.

Ces constats restent dans le registre des sources. Une nouvelle interrogation n'est justifiée qu'après changement de date, d'API ou de couverture annoncé.

---

## 4. Modèle de données à compléter avant toute nouvelle collecte massive

Le manifeste actuel ne permet pas de distinguer correctement doublon, vue similaire, secteur visible et rôle de reconstruction. Ajouter les champs suivants sans remplacer les champs existants :

```text
Asset
  source_family
  exact_duplicate_group
  perceptual_duplicate_group
  viewpoint_cluster
  subjects[]
  view_sector
  capture_type
  reconstruction_role
  temporal_status
  classification_confidence
  classification_method
  review_status
```

### Valeurs contrôlées

`source_family`
: Origine véritable du média. Exemple : Expedia, Hotels.com, Momondo et Kayak peuvent tous pointer vers `expedia_media`; The Vendry peut pointer vers `iceportal_51113`.

`subjects[]`
: Liste multi-étiquette parmi `building`, `entrance`, `sign`, `parking`, `roof`, `grounds`, `road`, `neighbour`, `interior`, `other`.

`view_sector`
: `front`, `left`, `right`, `rear`, `roof`, `front_left_corner`, `front_right_corner`, `rear_left_corner`, `rear_right_corner`, `transition`, `context`, `unknown`.

`capture_type`
: `street_imagery`, `traveler`, `promotional`, `social`, `aerial_oblique`, `orthophoto`, `lidar`, `municipal_document`, `hotel_capture`, `unknown`.

`reconstruction_role`
: `photo_geometry`, `texture_reference`, `geo_geometry`, `context_lock`, `identity_evidence`, `reference_only`, `reject`.

`temporal_status`
: `pre_2024`, `post_2024`, `current_confirmed`, `historical`, `unknown`.

`review_status`
: `automatic_accepted`, `human_accepted`, `needs_review`, `rejected`.

Une image peut montrer simultanément le bâtiment, le stationnement et l'enseigne. La catégorie unique actuelle reste disponible pour compatibilité, mais elle ne doit plus porter toute la décision.

---

## 5. Déduplication sans perdre le recouvrement utile

La déduplication doit distinguer quatre situations au lieu de simplement supprimer les fichiers proches.

### Niveau 1 — Fichier identique

- calculer SHA-256 sur le contenu ;
- regrouper les checksums identiques dans `exact_duplicate_group` ;
- conserver un fichier canonique et toutes les provenances.

### Niveau 2 — Même photographie republiée

- conserver le pHash existant ;
- ajouter un embedding visuel robuste aux recadrages, changements d'exposition, watermarks et redimensionnements ;
- regrouper dans `perceptual_duplicate_group` ;
- sélectionner comme canonique la meilleure résolution, la compression la plus faible et la meilleure provenance.

### Niveau 3 — Même point de vue

Pour les médias géolocalisés, construire `viewpoint_cluster` à partir de :

- position caméra ;
- direction vers le bâtiment ;
- distance à l'empreinte ;
- secteur du bâtiment visible ;
- similarité d'image.

Deux fichiers différents pris pratiquement au même endroit comptent comme un seul point de vue pour la couverture.

### Niveau 4 — Recouvrement utile

Ne pas éliminer les images successives qui créent une continuité spatiale exploitable. Dans chaque `viewpoint_cluster`, conserver :

- la meilleure vue canonique ;
- jusqu'à deux vues supplémentaires seulement si elles apportent un déplacement ou un recouvrement utile ;
- les autres comme références inactives, sans les supprimer du registre.

### Validation requise

- tests pour copie recompressée, recadrage, watermark et changement d'exposition ;
- test garantissant que deux positions réellement différentes ne fusionnent pas ;
- rapport avant/après par `source_family`, groupe perceptuel et point de vue ;
- les Gates comptent les `viewpoint_cluster`, jamais les fichiers bruts.

---

## 6. Catégorisation fiable et explicable

Le classifieur OpenCLIP actuel force la meilleure des six classes, même lorsque toutes les probabilités sont faibles. Le remplacer comme décideur unique par une cascade :

1. métadonnées déterministes de la source ;
2. position, cap et visibilité géométrique ;
3. OCR et nom de propriété ;
4. classifieur multi-étiquette ;
5. revue humaine uniquement pour les cas décisifs et ambigus.

### Règles de classification

- une confiance insuffisante produit `unknown` ou `needs_review` ;
- aucune image n'est déclarée façade uniquement parce qu'elle est extérieure ;
- `sees_building` reste déterminé par géométrie lorsque position et cap existent ;
- une image sans position peut être confirmée par OCR, métadonnées de fiche ou comparaison visuelle, avec méthode enregistrée ;
- l'appartenance et la catégorie sont deux décisions indépendantes ;
- la version de l'entrée ne doit jamais être inférée sans preuve datée ;
- les catégories générées automatiquement conservent leur score et leur méthode.

### Jeu de validation minimal

Créer un échantillon humain équilibré comprenant :

- façade principale ;
- côtés ;
- arrière ou absence d'arrière ;
- entrée pré/post-2024 ;
- stationnement hôtel et parc-o-bus ;
- bâtiments voisins ;
- intérieur ;
- enseigne seule ;
- photos promotionnelles ambiguës.

Le classifieur n'est accepté que si les erreurs sont visibles dans une matrice de confusion et si `unknown` empêche les faux positifs les plus dangereux.

---

## 7. Street View multi-position, pas multi-rotation

Le collecteur actuel retourne huit orientations du panorama le plus proche. Il doit devenir un collecteur de positions indépendantes.

### Procédure

1. récupérer le réseau routier et les accès dans un rayon initial de 350 m autour de `BUILDING_MAIN` ;
2. inclure rue Ampère, boulevard de Mortagne, voies de desserte et accès publics au stationnement ;
3. échantillonner les routes tous les 10 à 20 m, avec une densité accrue près des coins visibles ;
4. rechercher les panorama IDs par lots pouvant aller jusqu'à 100 positions ;
5. dédupliquer immédiatement les panorama IDs et leurs positions ;
6. suivre les liens vers les panoramas adjacents lorsqu'ils révèlent une position nouvelle ;
7. calculer le cap vers le point le plus proche de l'empreinte, plutôt qu'utiliser huit caps fixes ;
8. demander une vue principale cadrant l'hôtel, puis une variante plus large uniquement si elle révèle la transition route–entrée–stationnement ;
9. conserver date, position réelle, cap, champ de vision, pitch, copyright et identifiant de session ;
10. appliquer visibilité et `viewpoint_cluster` avant téléchargement en pleine résolution.

### Contraintes

- une rotation différente depuis le même panorama ne compte pas comme un nouvel angle du bâtiment ;
- un panorama ne cadrant pas l'empreinte est du contexte, pas une vue du bâtiment ;
- l'historique peut servir à comprendre une transformation, mais ne doit pas être fusionné avec l'entrée actuelle ;
- l'absence de voie derrière l'hôtel reste une absence de couverture, pas un échec du collecteur.

Référence technique : [Street View Tiles API](https://developers.google.com/maps/documentation/tile/streetview), qui accepte la recherche groupée de panorama IDs et expose les panoramas adjacents.

---

## 8. Registre unique des sources photographiques

Chaque source est interrogée une fois par campagne et rattachée à une famille véritable. Une source n'est retenue que si elle apporte un média nouveau ou une provenance supérieure.

### Priorité A — Forte probabilité d'angles nouveaux

#### TripAdvisor voyageurs

- fiche connue : `locationId=183189` ;
- galerie publique observée : 98 photos, dont 68 voyageurs ;
- corriger le collecteur pour demander explicitement `Traveler`, gérer `offset`, la pagination et les limites réelles du niveau d'API ;
- si l'API standard reste plafonnée à cinq médias, utiliser Terra, un accès supérieur ou la galerie publique complète ;
- conserver auteur, date, légende, album, dimensions et URL originale.

Référence : [Location Photos API](https://tripadvisor-content-api.readme.io/reference/getlocationphotos).

#### Facebook et Instagram officiels

- collecter albums, publications et carrousels de l'hôtel ;
- extraire des images-clés des Reels et vidéos montrant l'extérieur ;
- conserver date de publication, texte, identifiant du post et relation vidéo–image ;
- privilégier les déplacements dans l'entrée ou le stationnement plutôt que les visuels de chambre.

#### Hôtel, architecte, entrepreneur et dossier municipal

- demander le dossier PIIA `2024-70127` : photos existantes, implantation, élévations, plans soumis, architecte et entrepreneur ;
- demander au WelcomINNS son dossier média original, ses photos de travaux et les exports non compressés ;
- retrouver l'intervenant de la rénovation et demander ses photos avant/pendant/après chantier ;
- enregistrer ces documents comme preuve temporelle de l'entrée actuelle.

Référence : [procès-verbal municipal confirmant le dossier](https://www.boucherville.ca/wp-content/uploads/2024/09/PV_seance_240916.pdf).

### Priorité B — Catalogues structurés

#### ICEPortal/Shiji

- utiliser l'identifiant de catalogue observé `51113` ;
- demander les assets originaux via l'API OAuth ;
- rattacher les republications The Vendry et autres à `iceportal_51113` ;
- ne compter qu'une fois chaque photo republiée.

#### Booking.com

- récupérer les photos et leurs métadonnées par Demand API si l'accès est disponible ;
- sinon inventorier la galerie publique ;
- comparer le stock au site officiel et à ICEPortal avant téléchargement complet.

Référence : [Booking Demand API — accommodation details](https://developers.booking.com/demand/docs/accommodations/about-accommodation).

#### Expedia Media

- collecter une seule fois le catalogue source Expedia ;
- rattacher Hotels.com, Kayak, Momondo et autres republications à la même famille ;
- ne garder d'une republication qu'une meilleure résolution ou une métadonnée absente de la source canonique.

#### Foursquare

- rechercher l'établissement par nom et adresse ;
- demander jusqu'à 50 photos par page ;
- filtrer en priorité `outdoor_building_exterior`, `outdoor_building_and_grounds`, `outdoor_grounds` et `outdoor_or_storefront` ;
- classer par plus récentes puis populaires.

Référence : [Foursquare Place Photos](https://docs.foursquare.com/fsq-developers-places/reference/place-photos).

### Priorité C — Découverte complémentaire

- Flickr : élargir le collecteur actuel au stock public pertinent, puisque les licences ne bloquent plus ce pilote ; conserver la licence comme provenance ;
- Google Places : conserver les dix images déjà collectées et ne relancer que si la fiche change ;
- site officiel : parcourir les pages internes, brochures PDF, médias de CMS et versions d'images originales ;
- Québec Vacances, Tourisme Montérégie, Cvent, Eventective, Travel Weekly, Reserving et annuaires similaires : découverte uniquement, puis rattachement à la famille source réelle ;
- Yelp et Apple/Bing Places : sonder une fois et conserver uniquement les photos réellement indépendantes.

### Sources de rue ouvertes

- Mapillary reste actif, mais ses séquences doivent être regroupées en points de vue et non en fichiers ;
- KartaView peut être retenté une fois avec une requête corrigée ou son nouveau service, puis marqué indisponible si l'appel échoue encore ;
- Panoramax et Commons restent en veille, sans interrogations répétitives dans cette campagne.

---

## 9. LiDAR, orthophoto et paquet géospatial automatique

L'objectif n'est pas de faire croire qu'une orthophoto est une photographie de façade. Elle sert à produire le volume, la toiture, le terrain et le contexte du Path D.

### Données à récupérer

- empreinte confirmée de `BUILDING_MAIN` ;
- limites du stationnement de l'hôtel ;
- réseau routier et accès ;
- nuage de points LiDAR LAZ couvrant la propriété et une marge de contexte ;
- MNT pour le terrain nu ;
- MNS/DSM produit depuis le LiDAR brut lorsque nécessaire ;
- orthophoto la plus récente disponible, plus sa date et sa résolution ;
- orthophoto historique seulement pour expliquer une transformation.

Sources : [Données LiDAR du Québec](https://www.donneesquebec.ca/recherche/dataset/donnees-lidar-du-quebec) et [imagerie orthorectifiée du Québec](https://www.donneesquebec.ca/recherche/fr/dataset/imagerie-orthorectifiee-du-quebec).

### Traitement déterministe

1. résoudre automatiquement l'empreinte vers les tuiles d'index LiDAR et orthophoto ;
2. télécharger uniquement les tuiles intersectant l'emprise avec marge ;
3. reprojeter dans un système métrique commun ;
4. découper le nuage au bâtiment et au contexte immédiat ;
5. séparer sol et points de surface ;
6. calculer hauteur au-dessus du terrain, enveloppe de toiture et pentes principales ;
7. comparer la silhouette de toiture à l'orthophoto ;
8. produire un volume proxy aligné sur l'empreinte sans inventer portes ni fenêtres ;
9. produire terrain, stationnement et contexte cartographique ;
10. enregistrer date, résolution, système de coordonnées, tuiles sources et transformations.

### Précision importante

Le MNT seul ne contient pas la volumétrie du bâtiment. La toiture et la hauteur doivent venir du nuage LiDAR brut ou d'un MNS/DSM, puis être comparées au MNT.

### Sorties géospatiales

```text
geo/
  source_indexes/
  lidar_raw/
  ortho_raw/
  building_points.laz
  terrain_dtm.tif
  surface_dsm.tif
  latest_orthophoto.tif
  building_footprint.geojson
  hotel_parking.geojson
  access_network.geojson
  roof_proxy.geojson
  building_volume_proxy.*
  geo_provenance.json
  geo_quality_report.json
```

---

## 10. Verrouiller l'environnement réel

Le bâtiment est la cible, mais son environnement ne doit pas être réinventé lors d'une génération future.

### Séparer trois couches

`stable_context`
: Routes, trottoirs, accès, stationnement, bâtiments voisins, limites végétales importantes, enseignes et lampadaires structurants.

`dynamic_context`
: Voitures, personnes, neige, ombres temporaires, mobilier mobile et végétation saisonnière fine.

`unknown_context`
: Zone masquée ou non observée dont la géométrie et l'apparence ne sont pas suffisamment établies.

### Règles de fidélité contextuelle

- la position et la géométrie du contexte stable viennent de données déterministes ;
- les photos réelles servent à confirmer son apparence et sa période ;
- l'IA ne peut ni déplacer une route, ni changer un voisin, ni créer une autre entrée ;
- un élément dynamique peut être retiré ou remplacé plus tard sans modifier la structure ;
- une zone inconnue reste inconnue, proxy ou invisible ; elle n'est jamais promue silencieusement en vérité.

### Sortie attendue

```text
context_manifest.json
  stable_objects[]
  dynamic_classes[]
  unknown_zones[]
  source_evidence[]
  temporal_status
```

---

## 11. Couverture, confiance et futures contraintes de caméra

Le Lot 1B ne conçoit pas le tournage, mais il doit fournir les contraintes qui empêcheront une caméra future d'exposer une zone non fiable.

### États par zone

`trusted`
: Couverture photo récente, plusieurs points de vue indépendants ou données géométriques solides. Une future caméra pourra s'en approcher selon le niveau de détail mesuré.

`proxy`
: Volume et position fiables, apparence partiellement observée. Visible seulement à une distance minimale calculée.

`unobserved`
: Apparence ou géométrie insuffisante. Zone interdite à une future caméra jusqu'à capture complémentaire.

### Seuils de couverture photographique visés

Les nombres suivants portent sur des `viewpoint_cluster` indépendants après déduplication :

| Secteur | Cible | Minimum pour être `trusted` |
|---|---:|---:|
| Façade principale actuelle | 12 | 8 |
| Côté gauche | 8 | 5 |
| Côté droit | 8 | 5 |
| Arrière | 5 | 3 |
| Coins et transitions | 6 | 4 |
| Toiture/aérien | 4 références | LiDAR + orthophoto suffisent pour la géométrie proxy |

Ces seuils ne sont pas un concours de volume. Une zone n'est `trusted` que si les vues sont actuelles, attribuées au bon bâtiment, suffisamment détaillées et spatialement diversifiées.

### Sorties de couverture

```text
coverage_report.json
zone_confidence.geojson
camera_constraints.json
```

`camera_constraints.json` doit indiquer par zone :

- état de confiance ;
- distance minimale future ;
- angles autorisés ou interdits ;
- niveau de détail maximal ;
- cause de la restriction ;
- preuve nécessaire pour lever la restriction.

---

## 12. Capture complémentaire par l'hôtel, seulement après épuisement des sources

Si un secteur requis reste `unobserved`, générer un brief précis plutôt qu'une demande vague de « plus de photos ».

### Contenu du brief

- plan annoté des positions de capture ;
- secteur à filmer ;
- sens de déplacement ;
- hauteur et orientation du téléphone ;
- distance recommandée ;
- recouvrement demandé entre images ;
- consigne de ne pas zoomer numériquement ;
- parcours continu reliant façade, coins, côtés, arrière et stationnement ;
- liste des zones déjà couvertes à ne pas refaire.

### Capture minimale souhaitée

- vidéo lente ou rafale géolocalisée autour des seules zones manquantes ;
- images-clés extraites sans remplacer le fichier original ;
- 60 à 80 % de recouvrement visuel ;
- lumière homogène et absence de pluie si possible ;
- aucune exigence de matériel spécialisé.

La capture complémentaire ne devient obligatoire que si une demande future exige de montrer une zone actuellement interdite.

---

## 13. Ordre d'implémentation obligatoire

### Étape 1 — Corriger la structure de vérité

- étendre les schémas du manifeste ;
- migrer sans perte les 219 assets existants ;
- ajouter les enums et validations ;
- préserver l'historique et la provenance.

**Acceptation :** les anciens manifestes se chargent ou migrent explicitement, et aucune catégorie ambiguë n'est transformée silencieusement en certitude.

### Étape 2 — Déduplication à trois niveaux

- checksum exact ;
- pHash + embedding ;
- regroupement des points de vue ;
- choix canonique sans suppression destructive.

**Acceptation :** le rapport distingue fichiers, photographies uniques et points de vue indépendants.

### Étape 3 — Catégorisation multidimensionnelle

- cascade déterministe puis IA ;
- catégories multi-étiquettes ;
- seuils `unknown` ;
- petit jeu de validation humaine.

**Acceptation :** chaque décision porte méthode, confiance et statut de revue.

### Étape 4 — Street View multi-position

- échantillonnage du réseau ;
- recherche groupée de panoramas ;
- suivi d'adjacence ;
- cadrage vers le bâtiment ;
- déduplication avant téléchargement.

**Acceptation :** le rapport énumère les positions Street View uniques et les secteurs réellement visibles.

### Étape 5 — Étendre les sources photographiques

Ordre : TripAdvisor voyageurs → Meta → PIIA/hôtel/intervenants → ICEPortal → Booking → Foursquare → autres annuaires et plateformes.

**Acceptation :** chaque nouvelle image est reliée à une `source_family`; aucune republication ne gonfle le nombre de vues.

### Étape 6 — Produire le paquet LiDAR/orthophoto

- résolution et téléchargement des tuiles ;
- terrain, surface, toiture et volume proxy ;
- provenance et qualité.

**Acceptation :** la toiture et le volume sont dérivés automatiquement de données 3D, pas du MNT seul ni d'une génération IA.

### Étape 7 — Qualifier le contexte et la couverture

- stable/dynamique/inconnu ;
- trusted/proxy/unobserved ;
- contraintes futures de caméra ;
- rapport par secteur.

**Acceptation :** toute zone faible est visible et justifiée ; aucune zone inconnue n'est présentée comme fidèle.

### Étape 8 — Décider s'il faut contacter l'hôtel

- comparer les résultats aux seuils ;
- produire le brief uniquement pour les secteurs manquants ;
- ne pas demander de refaire la façade déjà couverte.

**Acceptation :** la demande est localisée, minimale et directement exécutable par une personne sur place.

---

## 14. Tests obligatoires du Lot 1B

### Schémas et migration

- migration des 219 assets sans perte ;
- enums invalides rejetés explicitement ;
- provenance multiple préservée ;
- droits assumés tracés mais non bloquants.

### Déduplication

- même fichier ;
- même photo recompressée ;
- recadrage et watermark ;
- même position avec images différentes ;
- positions différentes non fusionnées ;
- recouvrement utile conservé.

### Classification

- cas faible donnant `unknown` ;
- multi-étiquette façade + stationnement + enseigne ;
- voisin rejeté malgré une catégorie extérieure plausible ;
- entrée pré-2024 non fusionnée avec l'entrée actuelle ;
- régression OCR « De Mortagne » conservée.

### Street View

- plusieurs points d'échantillonnage ramenant le même panorama ;
- panoramas uniques conservés ;
- cap calculé vers l'empreinte ;
- panorama hors champ classé contexte ;
- aucune fuite de clé dans le manifeste ;
- reprise sur cache sans appels inutiles.

### Géodonnées

- résolution empreinte → tuiles ;
- reprojection métrique ;
- MNS/DSM distinct du MNT ;
- hauteur positive et plausible ;
- provenance de chaque raster et transformation ;
- absence de données produisant `unobserved`, jamais une géométrie inventée.

### Golden report

Un test golden fixe la structure du rapport WelcomINNS : totaux bruts, photos uniques, points de vue, secteurs, périodes, confiance, proxies et manques.

---

## 15. Livrables finaux du Lot 1B

```text
00_manifest/
  asset_manifest.json
  spatial_manifest.json
  source_registry.json
  migration_report.json

01_sources/
  source_reports/
  duplicate_report.json
  classification_report.json
  streetview_coverage.json
  source_family_map.json

geo/
  ... paquet défini au §9

coverage/
  coverage_report.json
  zone_confidence.geojson
  camera_constraints.json
  context_manifest.json
  capture_brief.md          # seulement si nécessaire

LOT_1B_REPORT.md
```

`LOT_1B_REPORT.md` doit répondre sans ambiguïté à cinq questions :

1. combien de photographies uniques ont été obtenues ?
2. combien de points de vue indépendants couvrent chaque secteur ?
3. quelles zones sont photographiques, géométriques proxy ou inconnues ?
4. quelles parties de l'environnement sont verrouillées par des preuves ?
5. quelles captures précises manquent encore, le cas échéant ?

---

## 16. Définition de DONE

Le Lot 1B est terminé lorsque :

- toutes les familles de sources prioritaires ont été interrogées ou portent une raison d'indisponibilité ;
- les republications ne comptent plus comme nouvelles photos ;
- les Gates comptent des points de vue indépendants par secteur ;
- la catégorisation peut répondre `unknown` et conserve ses preuves ;
- Street View a exploré plusieurs positions et non un seul panorama tourné huit fois ;
- le LiDAR brut ou le MNS/DSM, le MNT et l'orthophoto couvrant la propriété sont acquis et qualifiés ;
- la toiture et le volume proxy sont dérivés automatiquement ;
- le contexte stable est identifié et protégé contre une modification générative ;
- chaque zone est `trusted`, `proxy` ou `unobserved` ;
- les contraintes futures de caméra empêchent de montrer les zones insuffisantes ;
- un brief minimal de capture hôtel existe seulement si des zones nécessaires restent non observées ;
- le rapport final permet de décider de la suite sans commencer le Lot 2.

Le résultat attendu n'est pas « un bâtiment complet à tout prix ». C'est une représentation documentée de ce qui est connu : **façade photographique, volume et toiture mesurés, contexte verrouillé, zones manquantes explicitement évitées ou capturées plus tard**.
