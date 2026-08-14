# Inventaire de portabilité — hypothèses à extraire avant le second site

Établi en lecture seule sur `75507d6`. Aucune écriture hors ce fichier, aucun appel réseau.

L'objet n'est pas la liste des noms codés en dur : elle est courte et la plupart des occurrences de « WelcomINNS » sont des justifications en commentaire, légitimes. L'objet est la liste des **hypothèses comportementales** — ce que le code tient pour acquis sans le demander, et ce qu'il fait quand la réponse manque.

Le fil conducteur des trois familles les plus graves :

```text
un repli silencieux vaut mieux qu'une erreur   → faux, il déplace l'erreur plus loin
une absence de réponse vaut « la réponse du pilote » → territoire QC, EPSG:2950, calibrations
une classe métier se lit dans du texte libre    → la taxonomie de validation
```

---

## A. Territoire et référentiels

### A1 — Tout point de la Terre appartient au Québec

| Champ | Contenu |
|---|---|
| Emplacement | [catalog.py:195](src/hotel_pipeline/geo/catalog.py#L195) |
| Hypothèse | `territories = {"QC"}` est le point de départ inconditionnel ; les deux boîtes suivantes n'ajoutent que `QC-CMM` et `QC-MONTEREGIE` |
| Portée réelle | territoire |
| Repli actuel | `QC`, jamais absent, jamais `unknown` |
| Effet sur un autre site | **fausse réussite.** Lyon rend `{"QC"}` et se voit proposer `lidar-quebec` et `cadastre-quebec`. Le routage réussit, l'acquisition échouera plus tard sans que la cause soit lisible |
| Destination | adaptateur territorial — résolution par intersection avec des limites administratives, avec un état `unknown` de plein droit |
| Test exigé | un point hors Québec ne rend jamais `QC` ; un point sans référentiel territorial connu rend `unknown` et **aucune** source candidate |

### A2 — Le routage ne sait pas dire « je ne sais pas »

| Champ | Contenu |
|---|---|
| Emplacement | [catalog.py:209-232](src/hotel_pipeline/geo/catalog.py#L209-L232) |
| Hypothèse | toute position produit un `Routing` exploitable ; l'absence de source candidate n'est pas un état distinct |
| Portée réelle | générique |
| Repli actuel | liste vide de candidats, sans état signalant que le territoire lui-même est inconnu |
| Effet sur un autre site | **fausse réussite silencieuse** — « aucune source » se lit comme « rien à faire ici », pas comme « je ne sais pas où je suis » |
| Destination | adaptateur territorial ; état `unsupported_territory` explicite |
| Test exigé | territoire inconnu → état explicite, distinct d'un territoire connu sans source |

### A3 — `EPSG:2950` est le référentiel de calcul, partout

| Champ | Contenu |
|---|---|
| Emplacement | [geometry.py:103](src/hotel_pipeline/schemas/geometry.py#L103), [visibility.py:187](src/hotel_pipeline/schemas/visibility.py#L187), [capture_geometry.py:95](src/hotel_pipeline/geo/capture_geometry.py#L95) |
| Hypothèse | une constante de module suffit ; le référentiel projeté ne dépend pas du site |
| Portée réelle | territoire (NAD83 / MTM 8 : Québec, 75° à 72° ouest) |
| Repli actuel | `EPSG:2950`, sans alternative possible |
| Effet sur un autre site | **blocage** en écriture — [geometry.py:253](src/hotel_pipeline/schemas/geometry.py#L253) refuse tout autre référentiel. C'est le bon comportement par accident : le refus est franc, mais il interdit le second site au lieu de le servir |
| Destination | **manifeste spatial.** Le CRS résolu est un fait spatial du site, non un seuil configurable : il n'a rien à faire dans une politique, où il deviendrait réglable. Il vit dans un `SpatialReferenceContext` versionné, que les calculs citent au lieu de le supposer |
| Test exigé | chaque site utilise un CRS projeté **déclaré**, adapté à sa position, et dont l'emprise contient **toutes** les géométries calculées — cible, obstacles, corridors et positions caméra comprises |

> Le test naïf — « deux sites, deux CRS différents » — serait faux : deux établissements d'un même fuseau MTM partagent légitimement `EPSG:2950`. Ce qui doit être vrai n'est pas la différence, c'est l'adéquation, et elle se vérifie sur l'emprise réellement occupée par les calculs, pas sur le seul point central.

### A4 — Le moteur de visibilité reprojette sans vérifier l'emprise

| Champ | Contenu |
|---|---|
| Emplacement | [visibility_run.py:205](src/hotel_pipeline/geo/visibility_run.py#L205) |
| Hypothèse | `Transformer.from_crs("EPSG:4326", "EPSG:2950")` en littéral, sans contrôle de domaine |
| Portée réelle | territoire |
| Repli actuel | aucun — pyproj projette hors emprise sans lever |
| Effet sur un autre site | **fausse réussite, la plus dangereuse de l'inventaire.** Lyon projette en `x=5 637 219 m, y=8 760 910 m` sans erreur. Distances, azimuts, largeurs angulaires et occlusions sont alors calculés sur une géométrie déformée de milliers de kilomètres — et le rapport a l'air normal |
| Destination | même résolution dynamique qu'en A3, plus le contrôle d'emprise déjà écrit en [capture_geometry.py:158](src/hotel_pipeline/geo/capture_geometry.py#L158) |
| Test exigé | une position hors emprise du CRS projeté fait échouer le run avant tout calcul, avec un message nommant le CRS et l'emprise |

> `capture_geometry` fait déjà ce contrôle et `visibility_run` ne le fait pas. C'est la même opération, écrite deux fois, dont une seule est protégée.

### A5 — Le référentiel vertical n'existe pas dans le calcul

| Champ | Contenu |
|---|---|
| Emplacement | [visibility_engine.py:81-120](src/hotel_pipeline/geo/visibility_engine.py#L81-L120), [visibility_engine.py:224](src/hotel_pipeline/geo/visibility_engine.py#L224) |
| Hypothèse | `CameraVertical.ground_m`, `TargetVertical.height_m` et `Obstacle.ground_m` sont comparables entre eux ; seule une `provenance` en texte libre les accompagne |
| Portée réelle | générique |
| Repli actuel | aucun référentiel déclaré ; les altitudes se soustraient telles quelles |
| Effet sur un autre site | **fausse réussite.** Deux sources verticales de référentiels différents — orthométrique et ellipsoïdal — diffèrent de plusieurs dizaines de mètres. `vertical_verdict` rendrait alors un blocage « prouvé » qui n'existe pas. Sur le pilote tout venait d'une tuile unique, donc le défaut n'a jamais pu se manifester |
| Destination | schéma — chaque mesure verticale porte son référentiel, et la comparaison n'est permise qu'à référentiel identique **ou** via une transformation verticale explicite |
| Test exigé | deux référentiels sans transformation déclarée rendent `unknown` ; avec transformation vérifiable, la comparaison est permise et la transformation figure dans la preuve ; un référentiel inconnu ne produit jamais un blocage prouvé |

Deux référentiels différents ne sont pas nécessairement incompatibles : ils le sont tant que rien ne dit comment passer de l'un à l'autre. Une transformation verticale est donc admise, à condition d'être **déclarée et vérifiable**, et elle porte :

```text
référentiel source            ex. NAD83(CSRS) hauteurs ellipsoïdales
référentiel destination       ex. CGVD2013 hauteurs orthométriques
type de hauteur               ellipsoïdale | orthométrique | au-dessus du sol
unité                         mètre, pied international, pied US
opération de transformation   identifiant de l'opération appliquée
modèle de géoïde              le cas échéant, avec sa version
précision annoncée            celle de l'opération, non celle espérée
provenance                    d'où vient cette déclaration
```

Sans ces éléments, la comparaison est refusée et le verdict vertical vaut `unknown`. Une transformation supposée serait pire que l'absence actuelle : elle donnerait à un écart de référentiel l'apparence d'une mesure.

> Conforme à votre critère : le CRS vertical peut rester inconnu. Aujourd'hui il n'est pas inconnu, il est **absent** — ce qui revient à supposer qu'il est partout le même, ce qui est la transformation identité appliquée sans l'avoir déclarée.

### A6 — La lecture du référentiel vertical est propre au fournisseur québécois

| Champ | Contenu |
|---|---|
| Emplacement | [lidar.py:207](src/hotel_pipeline/geo/lidar.py#L207) |
| Hypothèse | le référentiel vertical se lit dans `SYSREF_ALTIMETRIQUE`, `referentiel_vertical` ou `vertical` |
| Portée réelle | fournisseur |
| Repli actuel | `None` — correct, rien n'est inventé |
| Effet sur un autre site | **blocage doux** : le champ reste vide, la provenance verticale devient invérifiable, l'enrichissement se dégrade sans erreur |
| Destination | adaptateur fournisseur |
| Test exigé | un fournisseur dont les champs sont inconnus rend `crs_vertical=None` **et** le signale, au lieu de le taire |

---

## B. Calibrations

### B1 — Trois identifiants de calibration du pilote sont les valeurs par défaut

| Champ | Contenu |
|---|---|
| Emplacement | [policy.py:34](src/hotel_pipeline/schemas/policy.py#L34), [policy.py:147](src/hotel_pipeline/schemas/policy.py#L147), [policy.py:209](src/hotel_pipeline/schemas/policy.py#L209) |
| Hypothèse | `welcominns-2026-08-36-images` et `welcominns-pilot-qualification-v1` sont les valeurs de départ de toute politique |
| Portée réelle | établissement |
| Repli actuel | ces identifiants exactement, avec `calibrated_on_sites=1` |
| Effet sur un autre site | **fausse réussite.** Un `PipelinePolicy()` neuf à Lyon se déclare calibré sur WelcomINNS. La provenance des rapports le recopie, et une calibration d'un site devient l'autorité d'un autre |
| Destination | politique de site — le champ reste, la valeur par défaut devient `non-calibré` avec `calibrated_on_sites=0` |
| Test exigé | une politique par défaut ne nomme aucun établissement ; appliquer une calibration nommée exige de déclarer les sites la portant |

> `terrain.calibration_id` est déjà correct : `« non-calibré — valeurs initiales, un seul site »`. C'est le modèle à généraliser aux deux autres.

### B2 — Les seuils du modèle sont des littéraux mesurés sur 36 images

| Champ | Contenu |
|---|---|
| Emplacement | [policy.py:26-34](src/hotel_pipeline/schemas/policy.py#L26-L34) |
| Hypothèse | `subject_accept=0.50`, `subject_reject=0.20`, `review_confidence_floor=0.60` valent partout |
| Portée réelle | établissement, jusqu'à validation multi-sites |
| Repli actuel | ces valeurs, appliquées sans mention de leur origine au moment de la décision |
| Effet sur un autre site | **fausse réussite** — les décisions automatiques héritent d'un réglage dont rien ne dit qu'il vaut ailleurs |
| Destination | politique ; inchangés en valeur, mais leur usage doit rester tracé comme provisoire |
| Test exigé | tout rapport citant un seuil cite aussi son `calibration_id` et le nombre de sites |

---

## C. Taxonomie de validation — le défaut que vous avez identifié

### C1 — La classe métier se déduit d'une analyse de texte libre

| Champ | Contenu |
|---|---|
| Emplacement | [validation.py:213-232](src/hotel_pipeline/validation.py#L213-L232) |
| Hypothèse | la nature d'une confusion se lit dans le motif de rejet, par recherche de sous-chaînes : `mortagne`, `1205`, `1201`, `tetra`, `isomed`, `bureaux`, `stationnement`… |
| Portée réelle | établissement **et** langue |
| Repli actuel | `other` |
| Effet sur un autre site | **erreur des deux côtés à la fois.** Mesuré sur un corpus lyonnais fictif : |
| Destination | manifeste d'annotations de validation, **distinct** de l'historique de revue ; `validation.py` n'agrège plus que des valeurs déclarées |
| Test exigé | la taxonomie ne contient aucun nom propre ; un rejet non annoté rend `unannotated`, jamais une classe devinée |

```text
motif de rejet                                    classe rendue
« enseigne Ibis Budget : concurrent »          →  other                  ← le cas visé, manqué
« immeuble de bureaux, plaque Groupe Bertrand »→  neighbouring_office     ← juste, par chance
« quai désert et platanes, aucun bâtiment »    →  other                  ← manqué
« enseigne Hôtel Mercure : concurrent »        →  competitor_same_kind    ← juste, par chance
« office building next door, not the target »  →  other                  ← toute langue non française
```

Les deux justes le sont parce que le motif contient un mot français figurant dans la liste. Pire, le marqueur `hôtel` classe en concurrent **tout** rejet mentionnant le mot — y compris un motif portant sur la cible elle-même.

**L'annotation ne touche pas `ReviewEntry`.** Classer une confusion est une analyse postérieure, pas une décision de visibilité : l'ajouter à l'historique de revue laisserait croire que l'image a été rejugée, et une entrée `unblinded` postérieure viendrait polluer une première passe aveugle que le rapport doit garder intacte. Les deux objets ne répondent pas à la même question et n'ont pas la même autorité.

Un manifeste séparé, `ValidationAnnotationManifest`, porte donc :

```text
asset_id                  l'image annotée
review_entry_ref          index et empreinte de l'entrée de revue visée
reviewed_checksum         l'image telle qu'elle a été jugée
taxonomy_version          la taxonomie appliquée, versionnée
confusion_class           valeur générique, sans nom propre
annotated_by / at         auteur et date
rationale / evidence      motif et preuves
blinding                  unblinded_posthoc — toujours, par construction
supersedes_index          correction append-only d'une annotation antérieure
```

`blinding=unblinded_posthoc` est une constante et non un choix : une annotation posée après coup ne peut pas prétendre à l'aveuglement. La nommer distinctement de `unblinded` évite qu'elle soit confondue avec une revue de seconde passe, qui, elle, portait sur l'identité.

Dans l'ordre :

1. le manifeste et sa taxonomie versionnée, sans aucun nom propre ;
2. `validation.confusions()` agrège ces valeurs et ne lit plus aucun texte ; un rejet sans annotation rend `unannotated` ;
3. les noms réels restent dans le profil (`competitor_names` existe déjà), les décisions et [tests/data/blind_rejects_welcominns.json](tests/data/blind_rejects_welcominns.json) ;
4. les huit rejets réels sont annotés — leur historique de revue aveugle n'est ni touché ni relu, et le rapport `blind_first_pass` reste bit à bit identique.

Le rapport devra montrer les deux populations séparément : ce que la passe aveugle a établi, et ce que l'annotation postérieure en dit.

### C2 — Les limites publiées citent les nombres du pilote

| Champ | Contenu |
|---|---|
| Emplacement | [validation.py:30-38](src/hotel_pipeline/validation.py#L30-L38), [cohort.py:342](src/hotel_pipeline/cohort.py#L342) |
| Hypothèse | « les 189 vues Mapillary », « deux séquences », « confusions WelcomINNS / Tetra Tech / Toyota » sont des constantes de texte |
| Portée réelle | établissement |
| Repli actuel | ces phrases, quel que soit le corpus |
| Effet sur un autre site | **fausse réussite** — un rapport lyonnais affirmerait porter sur 189 vues Mapillary et sur Tetra Tech |
| Destination | rapport — les nombres viennent du corpus mesuré, la formulation reste générique |
| Test exigé | aucun rapport de validation ne contient de nom d'établissement absent du profil du site |

---

## D. Profil et politique

### D1 — Dix-neuf commandes acceptent un profil absent

| Champ | Contenu |
|---|---|
| Emplacement | [cli.py:263-273](src/hotel_pipeline/cli.py#L263-L273), consommé 19 fois |
| Hypothèse | `_context()` avertit en jaune puis continue |
| Portée réelle | générique |
| Repli actuel | contexte sans profil : `identity_terms=[]`, `excluded_terms=[]`, `ocr_languages=["fr","en"]` |
| Effet sur un autre site | **fausse réussite.** Vérifié : sans profil, une enseigne lue « HOTEL MERCURE LYON PART-DIEU » rend `uncertain` au lieu de `mismatch`. Le verrou anti-confusion — le risque n° 1 du plan directeur — est désarmé par une ligne d'avertissement |
| Destination | **matrice de capacités centralisée** — chaque commande déclare ses prérequis, le contexte les valide avant toute mutation |
| Test exigé | la matrice couvre toutes les commandes — une commande non déclarée est une erreur de construction, non un passe-droit ; sans profil, chaque commande exigeant l'identité s'arrête sur une erreur typée |

Toutes les commandes ne doivent pas exiger un profil : `init` doit précisément tourner avant qu'il existe. Dix-neuf gardes recopiés dans la CLI seraient par ailleurs dix-neuf occasions d'en oublier un — c'est exactement ainsi que `blocking()` avait divergé de `role_for`. Les prérequis se déclarent donc une fois, par capacité :

```text
capacité                        prérequis
──────────────────────────────────────────────────────────────────
bootstrap                       aucun — init, profile create
inspection                      lecture partielle autorisée, et signalée comme telle
identity_classification         profil obligatoire
targeted_collection             profil + position + politique
geospatial                      contexte spatial obligatoire
qualification                   politique matérialisée obligatoire
```

Deux conséquences à tenir :

- `load_lenient` cesse d'être la voie d'accès par défaut : seules les commandes déclarant la capacité `bootstrap` ou `inspection` y ont droit ;
- « lecture partielle autorisée » n'est pas « silencieuse » — une inspection sans profil doit dire ce qu'elle ne peut pas établir, sans quoi elle redevient un faux succès.

L'erreur d'une capacité absente doit être **typée et actionnable** : quelle capacité manque, quel élément l'aurait satisfaite, quelle commande le crée.

### D2 — La langue d'OCR retombe sur le français et l'anglais

| Champ | Contenu |
|---|---|
| Emplacement | [context.py:213](src/hotel_pipeline/context.py#L213), [sign_ocr.py:88](src/hotel_pipeline/triage/sign_ocr.py#L88) |
| Hypothèse | un établissement sans profil est francophone ou anglophone |
| Portée réelle | territoire |
| Repli actuel | `["fr", "en"]` aux deux endroits |
| Effet sur un autre site | **erreur** hors de ces langues : l'OCR lit mal, `property_match_status` reste `uncertain`, et l'échec ressemble à une image illisible |
| Destination | profil, avec dérivation possible du territoire |
| Test exigé | aucune langue par défaut hors profil ; un profil sans langue déclarée empêche l'OCR au lieu d'en supposer une |

### D3 — Les indices de tri sont des mots français

| Champ | Contenu |
|---|---|
| Emplacement | [resolve.py:43-45](src/hotel_pipeline/resolve.py#L43-L45) |
| Hypothèse | un parc-o-bus se reconnaît à « incitatif » ; un hébergement aux étiquettes `hotel`/`motel` |
| Portée réelle | territoire et fournisseur (conventions OSM locales) |
| Repli actuel | ces listes |
| Effet sur un autre site | **erreur silencieuse** — un parc relais lyonnais n'est pas reconnu et devient un stationnement candidat de l'hôtel |
| Destination | politique ou adaptateur territorial |
| Test exigé | les listes viennent de la configuration ; une liste vide n'affirme rien plutôt que de tout accepter |

### D4 — Le géocodeur québécois est tenté pour toute adresse

| Champ | Contenu |
|---|---|
| Emplacement | [geocode.py:90-100](src/hotel_pipeline/providers/geocode.py#L90-L100) |
| Hypothèse | si `GEOCODER_QC_URL` est défini, il vaut pour l'adresse courante, quelle qu'elle soit |
| Portée réelle | territoire |
| Repli actuel | Nominatim en secours après échec |
| Effet sur un autre site | **fausse réussite possible** — une adresse française interrogée sur un géocodeur québécois peut rendre une correspondance approximative plutôt qu'une erreur. Le repli n'a lieu qu'en cas d'exception, pas en cas de mauvaise réponse |
| Destination | adaptateur territorial — le géocodeur se choisit sur le territoire, pas sur une variable d'environnement |
| Test exigé | une adresse hors du territoire d'un géocodeur ne lui est pas soumise |

---

## E. Temporel

### E1 — L'année de capture est calculée en UTC, comparée à des dates civiles

| Champ | Contenu |
|---|---|
| Emplacement | [mapillary.py:150](src/hotel_pipeline/collectors/mapillary.py#L150), comparé en [temporal.py:85-89](src/hotel_pipeline/temporal.py#L85-L89) |
| Hypothèse | l'année UTC d'un horodatage vaut l'année locale de la prise de vue |
| Portée réelle | territoire (décalage au fuseau) |
| Repli actuel | UTC |
| Effet sur un autre site | **erreur de bord**, rare mais réelle : une capture du 31 décembre au soir à Boucherville tombe en janvier UTC. Comparée à `started_on`/`completed_on`, qui sont des dates civiles locales, elle peut basculer un `before_event` en `after_event` |
| Destination | manifeste — conserver l'horodatage complet et le fuseau du site, et ne réduire à l'année qu'au moment de la comparaison |
| Test exigé | une capture proche du changement d'année rend la même année civile locale que celle du site |

> Portée limitée — la granularité annuelle absorbe presque tout. Je le note parce que c'est une hypothèse territoriale non déclarée, pas parce qu'elle a produit une erreur ici.

### E2 — Le reste du temporel est déjà générique

`RenovationEvent` a remplacé `pre_2024`/`post_2024`, [temporal.py:75-93](src/hotel_pipeline/temporal.py#L75-L93) rend `unknown` sans profil et refuse de conclure sur une approbation seule. Rien à extraire. Le seul nom propre restant est un exemple en docstring.

---

## F. Fournisseurs

### F1 — Tailles, caps et pagination sont des littéraux de module

| Champ | Contenu |
|---|---|
| Emplacement | [mapillary.py:24-31](src/hotel_pipeline/collectors/mapillary.py#L24-L31), [commons.py:40](src/hotel_pipeline/collectors/commons.py#L40), [places.py:75](src/hotel_pipeline/collectors/places.py#L75) |
| Hypothèse | `thumb_2048_url`, `iiurlwidth=2048`, `maxWidthPx`, `PAGE_SIZE=200`, `MAX_IMAGES=1500` conviennent partout |
| Portée réelle | fournisseur |
| Repli actuel | ces valeurs |
| Effet sur un autre site | **fausse réussite** en zone dense : `MAX_IMAGES=1500` tronque sans le dire, et le corpus paraît complet |
| Destination | politique de collecte ; le plafond atteint doit être un fait rapporté, pas un silence |
| Test exigé | atteindre le plafond produit un état explicite dans le manifeste de collecte |

### F2 — La convention d'axes du WFS québécois est écrite dans le code

| Champ | Contenu |
|---|---|
| Emplacement | [lidar.py:115-127](src/hotel_pipeline/geo/lidar.py#L115-L127) |
| Hypothèse | ce GeoServer attend `lon,lat` en dépit de la norme WFS 2.0 |
| Portée réelle | fournisseur |
| Repli actuel | `lon,lat` |
| Effet sur un autre site | **fausse réussite** — un service conforme rendrait zéro entité, c'est-à-dire « non couvert ». Le commentaire dit déjà que c'est le pire mode de défaillance, et qu'il s'est produit |
| Destination | adaptateur fournisseur, avec la convention déclarée par source |
| Test exigé | zéro entité en réponse ne peut pas produire `not_covered` sans contrôle indépendant de la convention |

### F3 — Le catalogue est propre au Québec, sa structure ne l'est pas

`GeoSource` déclare emprise, résolution, licence et portée ; `serves()` route sur des identifiants libres. La structure est portable. Ce qui ne l'est pas est le **contenu** — quatre sources québécoises — et le fait qu'aucun mécanisme ne dise « ce territoire n'a pas de catalogue ». À traiter avec A2, pas séparément.

---

## G. Identifiants, chemins et exemples

Peu de choses, et rien de comportemental.

| Emplacement | Nature | Destination |
|---|---|---|
| [cli.py:72](src/hotel_pipeline/cli.py#L72) | `help="ex. welcominns-boucherville"` | exemple documentaire — à rendre neutre |
| [cohort.py:342](src/hotel_pipeline/cohort.py#L342) | chaîne publiée dans le rapport | voir C2 |
| commentaires de `assets.py`, `geometry.py`, `visibility_engine.py`, `resolve.py`, `sign_ocr.py` | justifications citant le cas réel | **à conserver**, sous condition ci-dessous |

`work/` et `profiles/` contiennent les données du pilote, à leur place.

### G1 — Où un nom propre reste permis, et à quelle condition

Un nom propre est permis en commentaire **s'il est présenté comme l'origine empirique d'une règle provisoire** — la mesure qui a produit le seuil, et qui cessera de le justifier dès qu'un second site parlera. Il est interdit partout où il agit :

```text
interdit    valeurs par défaut
interdit    branches conditionnelles
interdit    rapports génériques
interdit    taxonomies exécutables
permis      commentaire déclarant une origine empirique provisoire
permis      profils, décisions, fixtures de validation
```

Le test correspondant se formule sur ces quatre familles, pas sur le fichier entier : chercher les noms propres partout condamnerait les justifications, qui sont ce qui rend les seuils auditables.

Pour les explications longues, un lien vers une fixture ou un ADR vaut mieux qu'une accumulation de cas dans le code. [sign_ocr.py:61](src/hotel_pipeline/triage/sign_ocr.py#L61) en est déjà à trois noms de salles de réunion pour justifier une règle sur les jetons courts : la règle mérite un ADR, le code une phrase.

---

## H. Smoke test — second site hors Québec, lecture seule

Hôtel fictif à Lyon (45,7640 N ; 4,8357 E), aucun profil créé, aucun réseau, aucune écriture. Script : `smoke_second_site.py` (scratchpad, non versionné).

| Critère attendu | Résultat obtenu | Verdict |
|---|---|---|
| aucun profil WelcomINNS chargé | profil absent, `ProfileNotFound` levé en mode strict | ✅ |
| territoire ≠ QC | `["QC"]` | ❌ |
| aucune source Québec proposée | `lidar-quebec`, `cadastre-quebec` | ❌ |
| aucun EPSG:2950 choisi | `EPSG:2950`, emprise `lat 44,98..62,53 / lon -75..-72`, Lyon hors emprise, **projection acceptée sans erreur** en `x=5 637 219 / y=8 760 910` | ❌ |
| aucune calibration WelcomINNS appliquée | `welcominns-2026-08-36-images`, `welcominns-pilot-qualification-v1` | ❌ |
| `unsupported`/`unresolved` explicite si donnée manquante | mode permissif : contexte rendu, `identity_terms=[]`, OCR `fr/en`, enseigne concurrente lue → `uncertain` | ❌ |
| aucun téléchargement | aucun | ✅ |

Deux critères sur sept. C'est l'échec révélateur attendu, et il est net : **rien n'a planté**. Le socle actuel exécuterait un second site jusqu'au bout en produisant des chiffres d'apparence normale, calculés dans un référentiel faux, avec le calibrage d'un autre établissement et sans verrou d'identité.

Le seul refus franc vient de [geometry.py:253](src/hotel_pipeline/schemas/geometry.py#L253), qui n'accepte pas d'autre CRS projeté que 2950 — un blocage, pas une protection.

---

## Séquence autorisée

**Commit 0 — inventaire et décisions architecturales.** Ce document, corrigé des quatre arbitrages : `ValidationAnnotationManifest` séparé de l'historique de revue (C1), noms propres permis en commentaire sous condition d'origine empirique déclarée (G1), CRS résolu dans le manifeste spatial et non dans la politique (A3), transformation verticale explicite plutôt que refus systématique (A5), matrice de capacités plutôt que gardes recopiés (D1).

**Commit 1 — bootstrap générique profil/politique.** Ferme D1, D2, B1, B2, et la part de C2 vivant dans la politique.

```text
tout nouveau projet naît « non-calibré », zéro site
aucune valeur WelcomINNS dans un défaut
le profil déclare position, pays, langues OCR, fuseau et identité
load_lenient réservé aux commandes déclarant bootstrap ou inspection
capacité absente → erreur typée et actionnable
la politique WelcomINNS existante se charge inchangée, empreinte inchangée
les textes de C2 se calculent depuis le corpus, jamais constants
```

Le sixième point est le plus contraignant : rendre les défauts génériques ne doit pas déplacer d'un octet le fichier de politique du pilote. Il sera vérifié par comparaison d'empreinte avant/après, pas par lecture.

**Commit 2 — territoire et référentiels dynamiques.** Ferme A1 à A6, D3, D4, F2, F3. Le `SpatialReferenceContext` versionné y naît, avec le contrôle d'emprise sur toutes les géométries calculées et la transformation verticale déclarée.

**Commit 3 — smoke test du second site.** Les sept critères du tableau H, sur `init → profile → site manifest → source routing → capture geometry → demands → discover --dry-run`, sans charger un seul fichier du WelcomINNS — et le projet WelcomINNS conserve décisions, artefacts et empreintes.

**C1** reste hors de cette séquence, en commit distinct à votre arbitrage. Elle ne touche plus l'historique de revue, mais elle crée un manifeste et une taxonomie versionnée : c'est un objet à part entière, pas un raccord de portabilité.

`assets discover` vient après, construit une seule fois sur le socle portable.
