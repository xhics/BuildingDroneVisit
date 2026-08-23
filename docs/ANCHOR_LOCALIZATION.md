# Localisation automatique guidée par ancres

Cette voie ajoute des images à une base SfM fiable sans intervention humaine et
sans transformer une approximation en mesure.

## Contrat de preuve

Une image compte dans le taux validé seulement si elle est :

1. une ancre du noyau, compatible avec les priors GPS/cap et reconstruite de
   façon stable lors d'au moins deux exécutions indépendantes ; ou
2. localisée par PnP contre au moins trois images de référence, avec tous les
   seuils de reprojection, profondeur positive, GPS/cap et stabilité satisfaits.

Une sortie virtuelle ou feed-forward reste `view_inferred`. Elle peut proposer
une vue, mais ne compte jamais dans G5. Une valeur absente produit
`insufficient_evidence`, jamais un succès par défaut.

## Exécution

Le pipeline complet est exposé par une seule commande :

```bash
hotel-pipeline reconstruction anchor-localize HOTEL_ID \
  --input-manifest work/HOTEL_ID/07_reconstruction/reconstruction_input_ID.json \
  --source-model PATH/TO/COLMAP_MODEL \
  --database PATH/TO/database.db \
  --image-dir PATH/TO/images \
  --features PATH/TO/features.h5 \
  --matches PATH/TO/matches.h5
```

Elle publie des artefacts append-only sous :

- `07_reconstruction/anchors/` : sélection et modèle du noyau ;
- `07_reconstruction/localization/` : verdicts et tentatives par image ;
- `07_reconstruction/localization/variants/` : corrections dérivées,
  adressées par empreinte.

## Ordre automatique des tentatives

Pour chaque image hors noyau :

1. ALIKED/LightGlue déjà calculé, avec toutes les ancres ;
2. même PnP avec les six puis les quatre ancres GPS les plus proches ;
3. correction photométrique CLAHE puis rematching ORB vers les points 3D ;
4. undistorsion déterministe puis rematching ORB vers les points 3D.

Les corrections conservent l'empreinte de l'original et de l'image dérivée.
L'undistorsion n'est tentée que pour un modèle de caméra explicitement pris
en charge. Aucun niveau ne peut promouvoir une vue si un seuil manque.

L'orchestrateur transmet les poses acceptées au tour suivant et borne la
propagation par `max_rounds`, `max_hop` et `max_attempts_per_level`. Le backend
H5 fourni garde volontairement une carte 3D figée sur le noyau : tant qu'une
pose acceptée n'a pas triangulé de nouveaux points, il ne prétend pas l'utiliser
comme nouvelle référence 3D. Les poses rejetées ou inférées ne sont jamais
promues.

## Décision G5

`SparseConsensusGate.registration_rate` lit désormais le taux validé du
`LocalizationManifest`. Le taux brut COLMAP reste disponible uniquement dans
`raw_registration_rate`. Sans manifeste de localisation et noyau `ready`, le
taux validé vaut zéro.

Le seuil du pilote reste 60 %. Si le pipeline reste nettement sous ce seuil,
la suite automatique est de produire une demande d'acquisition de vues
adjacentes autorisées ; changer silencieusement de seuil ou fabriquer des poses
n'est pas une option.
