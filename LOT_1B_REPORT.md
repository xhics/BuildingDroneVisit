# Rapport Lot 1B — WelcomINNS Boucherville

**État vérifié :** 16 août 2026  
**Rapport canonique machine :** `work/welcominns-boucherville/coverage/coverage_report.json`

## Verdict

Le Lot 1B n'est pas clos. La route canonique est **path_d_hybrid** avec le
statut **capture_required**. Le volume, la toiture et le terrain sont
exploitables comme proxies qualifiés ; leur apparence ne l'est pas.

## Couverture

- 335 assets, dont 9 porteurs de géométrie ;
- 269 photographies uniques et 209 points de vue au rapport de déduplication courant ;
- 1 point de vue indépendant ;
- besoins satisfaits : 0, partiels : 1, ouverts : 4 ;
- capture complémentaire : `obligation:ACCESS_ROAD_MAIN` seulement, depuis son corridor résolu et avec aperçu préalable ;
- droits : 146 assets non clarifiés.

## Contexte, orthophoto et cadastre

- LiDAR Québec : `covered / acquired`, par intersection réelle de la tuile ;
- orthophoto CMM 2023 à 5 m : `covered / available_not_acquired`, usage contexte seulement ;
- cadastre Infolot : `manual_acquisition_required`, donc `PROPERTY_PARCEL` reste `unresolved`.

Le registre photographique canonique clôt 2 familles requises sur 15 :
Mapillary et Street View. Places et le site officiel possèdent des assets sans
reçu de campagne courant ; les autres familles restent non interrogées,
manuelles, non implémentées ou sans indisponibilité documentée.

## Objets réexaminés

Les cinq objets demandés restent `unresolved` : PARKING_HOTEL,
ENTRANCE_MAIN_CURRENT, PROPERTY_PARCEL, DRIVEWAY_MAIN et PARK_AND_RIDE. Cette
relecture n'a apporté aucune preuve nouvelle et ne les promeut donc pas.

## Condition de fermeture

Obtenir et juger une vue de l'accès ; acquérir et qualifier l'orthophoto ;
acquérir l'extrait cadastral ; établir l'entrée actuelle et l'association du
stationnement ; et clôturer les 13 familles photographiques encore
ouvertes par interrogation ou indisponibilité documentée. Régénérer ensuite ce
rapport et republier le Router si ses entrées changent.

La déduplication robuste n'est plus une réserve : elle couvre 335/335 assets,
examine 30 891 couples plausibles et exécute les régressions recadrage,
filigrane et image distincte avec l'algorithme de production. G1 est `passed`.

Les 146 droits non clarifiés bornent l'usage des images mais ne sont pas
présentés comme la cause du manque de couverture.

## Export hybride post-Router

Un paquet provider-agnostic est publié sous `08_composite/` : volume OBJ proxy,
DTM/DSM/nDSM, contraintes caméra, carte de confiance, orbite virtuelle et
prompts. Son verdict est `NEEDS_AUTHORIZED_CAPTURE`, jamais
`ENVIRONMENT_3D_READY`. Aucun SfM, splat Brush ni appel vidéo réel n'a été
exécuté.
