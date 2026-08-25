# Texture de façade : passer de « le mur se projette ici » à « ce pixel montre ce mur »

HEAD de référence : `5aebeea7937f019b0a6b1134e03b47126ee249e9`.

## Principe directeur

On n'extrait et on ne projette **que l'information du bâtiment lui-même**, prouvée
pixel par pixel. Tout le reste demeure non observé. **Un atlas troué est le
résultat attendu**, pas un échec : le proxy mesuré reprend sa place sous les
trous, et aucune couleur n'est fabriquée pour les combler.

Corollaire immédiat : la chaîne actuelle qui rejette un texel puis le reconstruit
(`orthofacade` met le texel à zéro → `_fill_texture` dilate 13×13 et Telea-inpaint,
`facade_texture.py:262-282`) est supprimée, pas ajustée.

## Décisions arrêtées (avec l'utilisateur)

1. **Preuve de visibilité** = masque sémantique obligatoire **+** z-buffer du
   proxy 3D (premier objet touché) **+** profondeur LiDAR. Une vue sans masque
   ne contribue à rien.
2. **Occlusion LiDAR** construite sur les classes végétation ASPRS **3/4/5
   seulement**. Le sol (2) est exclu — sous une vue de rue, les points de sol
   rasants rejetteraient à tort le bas des murs. Le bâti (6) est exclu — avec
   ~1 m d'erreur de recalage, les points de la cible se rejetteraient eux-mêmes,
   et l'auto-occlusion est déjà traitée par le z-buffer proxy.
3. **Gate de pose** = écart mesuré, en pixels, entre la silhouette projetée du
   proxy et la frontière du masque `building`, converti en mètres par la GSD
   locale, évalué **par (façade, vue)**. Aucune correction du plan n'est
   appliquée : on mesure, on garde ou on refuse, on publie. Le `p90_m` 3D
   redevient un pré-requis grossier lu sur le **holdout**, et cesse d'être
   présenté comme une « erreur de reprojection ».
4. **Masques** : le code lit des rasters à trous quand ils existent, et dégrade
   explicitement sinon (`mask_fidelity`). Le re-run modèle est une tâche
   d'exécution séparée.
5. **Chaîne sémantique** : les masques de texture vont dans un artefact dédié
   (`11_conditioning/texture_view_masks.json` + rasters), produit par le même
   code via `--purpose texture`. La chaîne
   identité → correspondances → registration reste **gelée et validée**.
6. **Enseigne** : pas de renommage de classe. Une enseigne incluse dans le masque
   `building` et à la profondeur du mur est une enseigne de façade → conservée.
   Une enseigne devant le bâtiment → occulteur. Décision tracée par région.
7. **Périmètre** : Lot A + Lot B. Multi-bâtiment et dé-spécialisation de la
   grammaire hors périmètre, mais la simplification doit être **déclarée**.

## Constats vérifiés dans le code et le workspace pilote

| Défaut | Preuve |
| --- | --- |
| Vues sans masque texturées via un simple quadrilatère | `facade_texture.py:370` (`vis = facade_mask` si `base_mask is None`) ; pilote : 10 vues recalées, **6** avec masque |
| Masques existants sans occulteurs mobiles | prompt du run pilote (`semantic_observations.json:9`) : ni `car`, ni `person`, ni `hedge`, ni `fence` |
| Aucun test de premier impact | `_facade_polygon_mask` (`facade_texture.py:334-360`) ne fait que `fillPoly` de 4 coins ; cible à **11 sommets** (angles rentrants) + 1 volume voisin dans le payload |
| Gate « précis » à 2,5 m sur le mauvais indicateur | `facade_texture.py:31` et `:311` lisent `metrics.fit.p90_m` ; pilote : `fit.p90 = 2.151`, `holdout.p90 = 1.987`, métrique = distance 3D plus-proche-voisin COLMAP↔LiDAR |
| Test qui valide 2,15 m | `tests/test_facade_texture.py:53-58` (c'est littéralement la valeur du pilote) |
| Trous SAM perdus | `semantic_detection.py:249` `RETR_EXTERNAL` + `:253` plus grand contour seulement |
| Enseigne d'hôtel effacée | `semantic_detection.py:142` (`sign`/`logo` → `sign`) + `facade_texture.py:176` (`sign` dans les occulteurs) |
| `wh` moyenné | `facade_texture.py:321-331` ; pilote : `wh = [9.21, 9.6, 9.39, 8.94, 4.25, ...]` → arête 3 = 8.94 → 4.25 m, texturée à 6.6 m rectangulaire, affichée en trapèze (`viewer.py:233`) |
| Désaccord → tout jeté | `orthofacade.py:292-300` (std sur toutes les vues, canvas remis à 0) |
| Texel rejeté ré-inpainté | `facade_texture.py:271-282` (dilate 13×13, Telea, alpha dilaté + flouté) |
| Azimut incompatible | `viewpoint.py:53` `atan2(outward[0], outward[1])` (boussole) vs `viewer.py:223` et `facade_preview.py:51` (`cos(az)`→x, 0° = est) → 90° d'écart ; `viewpoint` est le seul en désaccord |
| Tests d'azimut permissifs | `tests/test_conditioning_viewpoint.py:40,51` tolèrent 90° |
| `texel_m` faux dans l'audit | `facade_texture.py:522` passe `0.12`, `orthofacade.py:151` écrit `TEXEL_M = 0.05` |
| Cache incomplet | `facade_texture.py:402-414` : ni le modèle COLMAP chargé, ni `selection_path`, tous deux lus **après** le test de cache (`:445-450`) |
| Fenêtres procédurales par-dessus les photos | `viewer.py:237` et `facade_preview.py:98` conditionnent à `supplemental_reference`, que `_supplemental_facade` retourne toujours `None` (`facade_texture.py:297`) |
| Distances divergentes | `viewer.py:219` `×0.56`, `facade_preview.py:144` `×0.62`, vs `target_distance_m` optimisé |
| Multi-bâtiment | `scene.py:136-137` `max(area)` et `:167` `exterior` seul |

## Lot A — rendre la visibilité prouvable

### 1. `src/hotel_pipeline/geo/facade_visibility.py` (nouveau)

Objet central, tel que proposé dans la revue :

```
FacadeTexelCandidate = {
  source_view, col, row, u_m, v_norm, pixel_xy,
  wall_depth_m, proxy_first_hit_depth_m, proxy_first_hit_face_id,
  lidar_depth_m, semantic_visible, semantic_source,
  pose_error_px, pose_error_m, local_gsd_m, incidence_deg, sharpness,
  colour_rgb, rejections: tuple[str, ...]
}
```

- `ProxyDepth.render(camera, triangles, face_ids, width, height)` : z-buffer numpy
  (barycentrique par boîte englobante, sans dépendance nouvelle). Rendu possible à
  résolution réduite (côté max ~1600 px) avec échantillonnage au plus proche.
  Triangles = **tous** les volumes du payload (murs `fp`/`wh` par sommet, toits
  `rv`/`rf` ou `solid`), pas seulement la cible.
- `LidarOcclusion` : lit la fenêtre LAZ via `conditioning/laz_cache.read_window`,
  filtre les classes 3/4/5, passe en repère local
  (`− scene_origin_projected_xyz`), projette avec la **même** caméra, splatte
  (`r_px = clip(ceil(f × 0.5 / depth), 1, 6)`), garde la profondeur minimale.
- `measure_facade_alignment(camera, plane, proxy_faceid_map, building_mask)` :
  écart médian, en pixels, entre l'arête haute projetée du mur (et ses arêtes
  verticales de coin, quand elles sont des arêtes de silhouette) et la frontière
  du masque `building`, restreint aux colonnes où le sommet de silhouette
  appartient bien à cette face. Retourne `(error_px, error_m, columns_used)`.
- `admit(candidate, policy)` : point unique de décision, avec codes de refus
  ordonnés — `semantic_absent`, `semantic_not_building`, `occluded_by_proxy`,
  `occluded_by_lidar`, `pose_error`, `resolution`, `incidence`, `behind_camera`,
  `outside_image`. C'est ce qui rend l'audit dénombrable par cause.

### 2. `src/hotel_pipeline/conditioning/texture_masks.py` (nouveau)

- Lecture prioritaire de `11_conditioning/texture_view_masks.json` (+ rasters
  PNG/NPZ), repli sur `semantic_observations.json`.
- Sortie par vue : `building` (bool), `occluders` (bool), `fidelity`
  (`raster_sam` | `raster_grabcut` | `polygon_no_holes`), `classes_present`,
  `sign_regions` (non tranchées à ce stade).
- Occulteurs : `tree_evergreen`, `tree_deciduous`, `bush`, `car`, `truck`, `bus`,
  `person`, `bicycle`, `fence`, `pole`, `lamp_post`, `road_sign`, `mobiliary`,
  `hvac_unit`, `flower_pot`.
- **Enseignes** : chaque région `sign` est classée une fois les cartes de
  profondeur disponibles — façade (≥ 70 % de ses pixels ont pour premier impact
  une face de la cible **et** aucun retour LiDAR plus proche) → conservée ;
  sinon → occulteur. La décision par région part dans l'audit.

### 3. `semantic_detection.py` : rasters et intention

- Persister le masque raster de chaque observation
  (`11_conditioning/texture_view_masks/<run_id>/<asset_id>/<observation_id>.png`),
  pour SAM 2 comme pour GrabCut. Le polygone reste un aperçu.
- `mask_to_polygon` : passer en `RETR_CCOMP` **uniquement pour détecter** la
  présence de trous et écrire `has_holes: true` — le polygone conserve le contour
  externe, mais la perte est désormais déclarée, plus silencieuse.
- Nouveau `--purpose texture` : écrit `texture_view_masks.json` (contrat propre,
  digests propres) sans toucher `semantic_observations.json`. Prompt étendu avec
  `car. truck. bus. person. bicycle. bush. hedge. fence. pole. planter.`
- `select_validated_images` : en mode `texture`, sélectionner **toutes** les vues
  présentes dans le modèle COLMAP d'ancrage, pas un `limit` de 2.

### 4. `geo/orthofacade.py` : candidats, statuts, fusion robuste, `top_z(u)`

- `FacadePlane` gagne `top_z_start_m` / `top_z_end_m` et `top_z(u)` (interpolation
  linéaire). `point(u, v_norm)` devient **normalisé en hauteur** :
  `z = v_norm × top_z(u)`. C'est exactement ce que consomment `viewer.py:245`
  (`texturedQuad` sur le trapèze) et `facade_preview.py:217` (homographie sur les
  4 coins) — l'atlas cesse d'être étiré sur une hauteur qu'il n'a pas mesurée.
  `height_m` reste exposé, comme `max(top_z)`, pour les rapports.
- `rectify(plane, views, texel_m, *, occlusion, policy)` produit des
  `FacadeTexelCandidate`, puis fusionne par texel :
  1. conversion Lab ;
  2. médiane ;
  3. rejet d'outliers par MAD (`k = 2.5`) — le cas `brique / brique / feuille`
      doit garder les deux briques, **pas** jeter les trois ;
  4. si `inliers ≥ 2` et dispersion des inliers `≤ MAX_INLIER_SPREAD_DE` →
      `OBSERVED_CONSENSUS`, couleur = moyenne pondérée des inliers
      (poids `cos(incidence)^p / gsd`, modulé par la netteté) ;
  5. si un seul candidat → `OBSERVED_SINGLE` ;
  6. si aucun consensus atteignable → `REJECTED_DISAGREEMENT`.
- Statuts finaux : `OBSERVED_CONSENSUS`, `OBSERVED_SINGLE`,
  `REJECTED_DISAGREEMENT`, `REJECTED_OCCLUDED`, `REJECTED_SEMANTIC`,
  `REJECTED_POSE`, `REJECTED_RESOLUTION`, `UNOBSERVED`. `is_observed` vrai pour
  les deux `OBSERVED_*` seulement.
- `as_dict()` publie le `texel_m` **réellement utilisé** (fin de la régression
  0,05 vs 0,12), `top_z_start_m`, `top_z_end_m`, les comptes par statut et les
  refus par cause.
- L'incidence par texel (déjà présente) est conservée ; la GSD devient elle aussi
  par texel, et `MIN_PIXELS_PER_M` s'applique au texel, plus à la vue entière.

### 5. `conditioning/facade_texture.py` : réordonner et ne rien fabriquer

Nouvel ordre d'exécution (le cache ne peut plus être testé avant d'avoir résolu
ses propres entrées) :

1. résoudre correspondances → manifeste d'ancrage → `selection_path` → modèle
   COLMAP ;
2. calculer le digest d'entrée **incluant** les fichiers du modèle
   (`cameras/images/points3D`), `selection_path`, le manifeste d'ancrage, les
   masques de texture, `wh`/`fp`, `texel_m` et **tous** les seuils de politique ;
3. tester le cache ;
4. construire les caméras ; par vue : `ProxyDepth` + `LidarOcclusion` ;
5. construire les masques (dont le tri des enseignes) ;
6. pré-requis grossier : `registration.status == accepted` **et**
   `metrics.holdout.p90_m ≤ REGISTRATION_HOLDOUT_MAX_P90_M` — commentaire corrigé,
   plus aucune mention d'« erreur de reprojection » ;
7. par façade : `top_z` depuis `wh[i]`/`wh[i+1]`, gate de pose par vue, candidats,
   fusion, composition ;
8. composer l'atlas : `alpha = 255` sur les `OBSERVED_*`, `alpha = 0` partout
   ailleurs. **Aucune** dilatation, **aucun** flou, **aucun** Telea, aucun
   remplissage brique. Le proxy se voit à travers, c'est voulu.
9. écrire `edge_XX.png`, `edge_XX_status.png` (carte de statut colorée) et des
   planches de diagnostic par vue (quad projeté, masque, causes de rejet).
- Supprimer `_supplemental_facade` et le champ `supplemental_reference` (stub mort
  qui, en plus, casse la logique `covered` des deux rendus).
- Remplacer `_edge_height` par la lecture directe de `wh[i]`, `wh[i+1]`.

### 6. `conditioning/viewpoint.py` : convention d'azimut

- `bearing = degrees(atan2(outward[1], outward[0])) % 360` et une docstring qui
  **nomme** la convention : 0° = +X (est), sens antihoraire, identique à
  `viewer.js` et `facade_preview`.
- Supprimer l'affirmation « aucune valeur magique » : `MIN_FACADE_EDGE_M`,
  `ENTRANCE_WEIGHT`, `SWEEP_STEP_DEG`, bornes 35/220 m et 14–32° sont des
  **heuristiques à calibrer**, déclarées comme telles.
- Corriger l'annotation de retour de `_geometry` (déclare 4 valeurs, en renvoie 5).

## Lot B — cohérence du rendu et de l'audit

### 7. Détails procéduraux : la photo mesurée gagne

- Après composition, pour chaque entrée de `facade_features` portant un
  `edge_index` : recalculer sa boîte `(u, v_norm)` sur le plan, échantillonner la
  carte de statut, écrire `texture_coverage` (fraction observée) et
  `covered_by_photo = texture_coverage ≥ 0.6`.
- `viewer.py:237` et `facade_preview.py:98` masquent les détails **si et seulement
  si** `covered_by_photo`, pour **tous** les genres (plus de liste
  `['window','band']`, plus de `supplemental_reference`).

### 8. Distance de caméra unique

- `optimal_camera` calcule `target_distance_m` depuis le demi-FOV réel des deux
  rendus (`q = H×0.5/tan(π/6)` → 30°) et l'encombrement projeté du bâtiment, avec
  marge explicite.
- `viewer.py:219,268` et `facade_preview.py:144` consomment `target_distance_m`
  **tel quel** : suppression de `×0.56` et de `×0.62`.
- Replis `?? 210` remplacés par une valeur dérivée des bornes du payload.

### 9. Rang de preuve dans le choix de vue

- `optimal_camera` renvoie `evidence_rank` ∈
  `measured_coverage` > `semantically_constrained` > `inferred_grammar`.
- `ENTRANCE_WEIGHT` ne s'applique **qu'en départage** : quand une couverture
  mesurée existe, une inférence de grammaire ne peut plus déplacer la caméra.

### 10. Simplification multi-bâtiment déclarée (sans la corriger)

- `scene.py` : lorsque `MultiPolygon → max(area)` s'applique, ou qu'un anneau
  intérieur est ignoré, écrire un enregistrement explicite
  (`geometry_simplification: {rule, parts_dropped, holes_dropped, area_dropped_m2}`)
  repris dans l'audit de scène et le manifeste du viewer. Le défaut reste, mais il
  cesse d'être invisible.

## Seuils, déclarés comme heuristiques à calibrer

| Constante | Valeur | Note |
| --- | --- | --- |
| `TEXEL_M_FACADE` | 0.12 | source unique, publiée dans l'audit |
| `MIN_PIXELS_PER_M` | 2.0 | par texel désormais |
| `MAX_INCIDENCE_DEG` | 65 | inchangé |
| `PROXY_DEPTH_TOLERANCE_M` | 0.25 | tolérance numérique ; le test principal est l'identité de face |
| `LIDAR_CLASSES` | (3, 4, 5) | végétation seule (cf. décision 2) |
| `LIDAR_OCCLUSION_MARGIN_M` | `max(1.5, holdout.p90_m)` | pilote → 1.99 m : seuls les occulteurs à ≥ 2 m devant sont vus |
| `POSE_MAX_ERROR_M` / `_PX` | 0.5 / 12 | gate par (façade, vue) |
| `REGISTRATION_HOLDOUT_MAX_P90_M` | 3.0 | pré-requis grossier, jamais suffisant |
| `MAD_OUTLIER_K` | 2.5 | rejet d'outlier en Lab |
| `MIN_INLIERS_FOR_CONSENSUS` | 2 | |
| `MAX_INLIER_SPREAD_DE` | 12 | ΔE entre inliers |
| `COVERED_BY_PHOTO_FRACTION` | 0.6 | masquage des détails procéduraux |

## Résultat attendu sur le pilote (à assumer, pas à contourner)

Avec les artefacts actuels : 4 vues sur 10 sans masque → exclues ; 6 vues en
`polygon_no_holes` sans classe voiture ; gate de pose à 0,5 m probablement non
tenu sur la plupart des façades. Sortie attendue : `status: unavailable` avec les
causes dénombrées, atlas de diagnostic écrits, viewer affichant le proxy. C'est
l'état honnête, et c'est le point de départ mesurable du re-run.

## Tests

`tests/test_facade_visibility.py` (nouveau)
- un mur avant occultant un mur arrière du **même** bâtiment → arrière rejeté
  `occluded_by_proxy`, avant admis ;
- volume voisin entre caméra et cible → tous les candidats rejetés ;
- point LiDAR d'arbre 3 m devant → `occluded_by_lidar` ; points sur le mur à ±1 m
  → admis (marge) ;
- **vue sans masque sémantique → zéro candidat admis** (`semantic_absent`) et
  atlas vide : c'est le P0 nº 2 verrouillé par un test ;
- masque raster **avec trou** → le trou est respecté ; masque polygone sans trou →
  `mask_fidelity: polygon_no_holes` dans l'audit ;
- enseigne à la profondeur du mur et incluse dans `building` → conservée ;
  enseigne devant → retirée.

`tests/test_facade_pose_gate.py` (nouveau)
- masque décalé de N px → erreur mesurée ≈ N px et ≈ N × GSD mètres ;
- au-delà du seuil → refus ; aligné → admission ;
- `fit.p90 = 2.15` seul n'admet plus rien (inversion explicite de
  `tests/test_facade_texture.py:53-58`) ; le pré-requis se lit sur `holdout`.

`tests/test_orthofacade.py` (mise à jour)
- `wh = [8, 12]` → `top_z(0) == 8`, `top_z(L) == 12`, et `v_norm = 1` atterrit sur
  la hauteur **locale** aux deux extrémités ;
- 3 échantillons `brique / brique / feuille` → couleur ≈ brique,
  `OBSERVED_CONSENSUS`, outlier compté ;
- 2 échantillons en désaccord franc → `REJECTED_DISAGREEMENT` ;
- **alpha exactement 0** sur les texels rejetés et non observés après composition
  (aucun Telea, aucune dilatation) ;
- `as_dict()["texel_m"]` égal au `texel_m` transmis.

`tests/test_conditioning_viewpoint.py` (réécriture)
- assertion géométrique : reconstruire l'œil avec la formule de `viewer.js`
  (`eye = focus + [cos az, sin az]·d`) et exiger
  `dot(normalize(eye − centre_façade), outward) > 0.95` — un décalage de 90° doit
  échouer ;
- couverture mesurée contre entrée inférée en conflit → la couverture gagne,
  `evidence_rank == "measured_coverage"` ;
- `target_distance_m` cadre le bâtiment sous le demi-FOV de 30° avec marge.

`tests/test_facade_texture.py` (mise à jour) et `tests/test_facade_preview.py`
- détail procédural sur zone couverte → non émis ; sur zone non couverte → émis ;
- aucune lecture de `supplemental_reference` ;
- cache : modifier un fichier du modèle COLMAP ou `selection_path` invalide le
  digest.

`tests/test_conditioning.py` / audit de scène
- une cible `MultiPolygon` ou à trous produit un enregistrement
  `geometry_simplification` non vide.

## Exécution après implémentation (runbook, machine GPU)

1. `conditioning semantic-detect <hotel> --purpose texture --limit <toutes les
   vues du modèle d'ancrage> --device cuda --segmentation sam2
   --sam2-checkpoint <chemin>` → `texture_view_masks.json` + rasters.
2. Régénérer le viewer (`facade_texture` + `viewpoint` + `viewer`).
3. Comparer l'audit avant/après : couverture par façade, refus par cause,
   `pose_error_m` par (façade, vue).
4. `semantic_observations.json`, `semantic_correspondences.json` et
   `vertical_registration.json` **ne doivent pas changer** — c'est le critère qui
   prouve que la chaîne validée est restée gelée.

## Risques

- **Le pilote peut rester à zéro façade texturée.** C'est accepté, mais il faut
  que l'audit dise précisément *quelle* barrière tombe en premier, sinon on ne
  saura pas quoi améliorer ensuite.
- **Coût de calcul** : z-buffer + LiDAR par vue. Rendre à résolution réduite,
  mémoïser par vue (pas par façade), et mesurer le temps sur le pilote.
- **Marge LiDAR grossière** (≈ 2 m sur ce site) : un auvent ou un arbre proche du
  mur passera. À déclarer, pas à masquer par un seuil flatteur.
- **Deux jeux de masques** (`semantic_observations` vs `texture_view_masks`) :
  noms, digests et provenance distincts obligatoires pour éviter la confusion.
- **`top_z(u)` change la sémantique de l'atlas** (hauteur normalisée) : les deux
  rendus doivent être vérifiés ensemble, sinon on remplace un décalage par un autre.

## Hors périmètre (assumé, et désormais déclaré)

- Cible multi-parties, trous d'emprise, plusieurs prismes cibles
  (`scene.py:136-167`) : seule la **déclaration** de la simplification est faite.
- Dé-spécialisation de la grammaire (`3.65 m`, canopy `7.2×3.8`, tour, pignon) et
  sortie des constantes vers un profil calibrable.
- Recalage par façade (estimer puis appliquer un décalage du plan) : le gate
  mesure et publie l'écart, ce qui prépare ce travail sans l'engager.
- Profondeur MVS dense et points COLMAP comme source d'occlusion supplémentaire.
