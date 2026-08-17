# Rapport Lot 1B — WelcomINNS Boucherville

**État vérifié :** 17 août 2026
**Rapport canonique machine :** `work/welcominns-boucherville/coverage/coverage_report.json`

## Verdict

Le Lot 1B n'est pas clos. La route canonique est **path_d_hybrid** avec le
statut **capture_required**. Le volume, la toiture et le terrain sont
exploitables comme proxies qualifiés ; leur apparence ne l'est pas.

## Couverture

- 338 assets, dont 15 porteurs de géométrie ;
- 272 photographies uniques et 210 points de vue au rapport de déduplication courant ;
- 3 points de vue indépendants ;
- besoins satisfaits : 0, partiels : 3, ouverts : 4 ;
- capture complémentaire : `obligation:ACCESS_ROAD_MAIN` seulement, depuis son corridor résolu et avec aperçu préalable ;
- droits : 146 assets non clarifiés.

## Contexte, orthophoto et cadastre

- LiDAR Québec : `covered / acquired`, par intersection réelle de la tuile ;
- orthophoto CMM 2023 à 5 m : `covered / available_not_acquired`, usage contexte seulement ;
- cadastre Infolot : `manual_acquisition_required`, donc `PROPERTY_PARCEL` reste `unresolved`.

Le registre photographique canonique clôt **14 familles requises sur 15**.
Mapillary et Street View par découverte ciblée ; Places, le site officiel et
Wikimedia Commons par campagne consignée — Commons rendant les 3 seules images
du corpus sous licence explicitement réutilisable (CC BY-SA 4.0). Onze familles
portent un reçu d'indisponibilité observée : absence de clé pour TripAdvisor et
Flickr, absence de collecteur et d'accès public pour les autres.

Reste `hotel_project_team` : une demande directe à l'hôtel, au dossier
municipal ou aux intervenants. Aucun mécanisme ne s'y substitue.

## Objets réexaminés

Quatre des cinq objets sont sortis de `unresolved`, par examen direct des images
déjà présentes au corpus et sans acquisition nouvelle :

- `ENTRANCE_MAIN_CURRENT` → **confirmed** : porte-cochère, portes vitrées et
  allée identiques sur trois prises séparées, dont une portant le numéro civique
  1195 et une vue Street View courante ;
- `PROPERTY_SIGN` → **confirmed** : enseigne sur pylône « HÔTEL WELCOMINNS »
  lisible depuis deux points de vue indépendants ;
- `DRIVEWAY_MAIN` → **confirmed** : chaussée marquée reliant la voie publique à
  la porte-cochère, sur deux points de vue ;
- `PARK_AND_RIDE` → **inferred** : terminus de transport collectif observé —
  abribus, quais, signalisation — nettement distinct du stationnement hôtelier ;
- `PARKING_HOTEL` → **inferred** : existence observée sur deux points de vue.
  L'objet n'est **pas** confirmé : l'association `way/1467386732` reste réfutée,
  ce polygone couvrant le 1205 et non le 1195. Son `source_ref` a été retiré
  pour qu'aucun lecteur ne le prenne pour une preuve.

`PROPERTY_PARCEL` demeure `unresolved` : le cadastre n'est pas acquis, et rien
d'observable sur une photographie ne tient lieu d'emprise cadastrale.

## Condition de fermeture

Obtenir et juger une vue de l'accès ; acquérir et qualifier l'orthophoto ;
acquérir l'extrait cadastral ; et obtenir la réponse de l'hôtel, du dossier
municipal ou des intervenants — seule famille photographique encore ouverte.
Régénérer ensuite ce rapport et republier le Router si ses entrées changent.

L'entrée actuelle n'y figure plus : elle est confirmée. L'association du
stationnement reste attendue, mais elle demande une emprise cadastrale, non une
photographie de plus.

Trois façades — gauche, arrière, droite — ne sont **pas** couvrables par ce
corpus, et c'est un constat mesuré : les vues des secteurs arrière montrent des
pavillons unifamiliaux d'une rue résidentielle voisine, jamais l'hôtel. Le
secteur calculé décrit la position de la caméra, non ce que l'image montre.

La déduplication robuste n'est plus une réserve : elle couvre 338/338 assets,
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
