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
| Vérité du site | partiel | 4 confirmés, 9 inférés ; seule la parcelle reste non résolue |
| Preuve tirée du corpus | livré | constat, lecture et résolution raccordés ; 0 octet téléchargé |
| Revue et qualification | livré | décisions et protocoles append-only |
| Collecte ciblée V2 | partiel | boucle réelle jusqu'à l'aperçu ; expansion Mapillary non raccordée |
| Registre des sources | livré, campagne quasi close | 14 familles requises sur 15 ; seule la demande à l'hôtel reste |
| LiDAR, terrain, toiture | livré | dérivations qualifiées comme inférences |
| Orthophoto et cadastre | partiel | état de couverture publié ; orthophoto non acquise, cadastre manuel |
| Orientation des façades | livré | décision à 227,89° propagée et vérifiée indépendamment |
| Contexte et contraintes caméra | livré | manifestes canoniques produits sous `coverage/` |
| Router | livré pour B/D | décision Path D publiée ; Path A/C restent hors contrat |
| Paquet 3D provider-agnostic | livré comme proxy | OBJ, rasters, orbite virtuelle, prompts et verdict Phase 1 |
| SfM et reconstruction | hors lot courant | Lot 2 non commencé |
| Orchestrateur Phase 1 | livré pour le Lot 1B | étape `lot1b` déléguant aux contrats modernes |

**Conclusion :** les mécanismes du Lot 1B sont avancés, mais sa définition de
DONE n'est pas atteinte. Ce qui bloque a changé de nature : l'entrée, l'enseigne
et l'allée sont désormais établies sur le corpus déjà collecté, sans acquisition
nouvelle. Restent la parcelle cadastrale, l'orthophoto, la preuve photo de
l'accès et la campagne de sources — dont aucun n'est un défaut de code.

---

## 2. État factuel du pilote

### 2.1 Corpus

| Mesure | Valeur |
|---|---:|
| Assets | 338 |
| `context_lock` | 301 |
| `photo_geometry` | 15 |
| `reference_only` | 16 |
| `identity_evidence` | 5 |
| `reject` | 1 |
| Droits `open_data` | 189 |
| Droits `public_uncleared` | 134 |
| Droits `unknown` | 12 |
| Assets positionnés | 313 |
| Fichiers sur disque | 609, pour 123 Mo |
| Images ≥ 640 px | 596 |
| Images ≥ 1920 px | 449 |

Les six acquisitions ciblées les plus récentes ont été publiées atomiquement,
puis évaluées. Elles restent des références : aucune n'a été promue par le seul
fait d'avoir été téléchargée.

`photo_geometry` passe de 9 à 15 : six vues du corpus ont été examinées, jugées
aptes et arbitrées, sans qu'aucun fichier ne soit téléchargé.

### 2.2 Objets du site

| Objet | État courant | Lecture |
|---|---|---|
| `BUILDING_MAIN` | confirmed | empreinte OSM confirmée |
| `ENTRANCE_MAIN_CURRENT` | confirmed | porte-cochère établie sur deux points de vue, corroborée 2024 → courant |
| `PROPERTY_SIGN` | confirmed | enseigne HÔTEL WELCOMINNS lisible sur deux points de vue |
| `DRIVEWAY_MAIN` | confirmed | allée marquée reliant la voie publique à la porte-cochère |
| `ACCESS_ROAD_MAIN` | inferred | accès conditionnel, géométrie disponible |
| `ROOFLINE_MAIN` | inferred | toiture LiDAR qualifiée |
| `TERRAIN_MAIN` | inferred | terrain interpolé et qualifié |
| `PARKING_HOTEL` | inferred | existence observée ; association cadastrale toujours réfutée |
| `PARK_AND_RIDE` | inferred | terminus de transport observé, distinct du stationnement hôtelier |
| `FACADE_PRIMARY/LEFT/RIGHT/REAR` | inferred | segments d'empreinte réinstanciés depuis l'orientation |
| `PROPERTY_PARCEL` | unresolved | cadastre non acquis |

Total : 14 objets, dont 4 confirmés, 9 inférés et 1 non résolu.

Ces résolutions ne viennent d'aucune acquisition nouvelle : elles sortent des
596 images en pleine résolution déjà présentes au corpus, qu'aucun mécanisme ne
savait convertir en preuve. Voir §5, « Preuve tirée du corpus ».

### 2.3 Besoins et aperçus

| Mesure | Valeur |
|---|---:|
| Besoins | 7 |
| Besoins ouverts | 4 |
| Besoins partiellement couverts | 3 |
| Constats d'aperçu établis | 9 |
| Constats réfutés | 12 |
| Constats indécis | 1 |
| Points de vue indépendants | 3 |

`refuted` qualifie l'élément de preuve, pas le besoin. Un besoin dont tous les
aperçus sont réfutés reste ouvert.

Les douze constats réfutés portent tous sur des vignettes de 256 px : c'est la
résolution, non le site, qui empêche de conclure. Les neuf constats établis
portent sur des images de 640 à 2048 px déjà présentes au corpus.

Le constat indécis est la vue `ACCESS_ROAD_MAIN` acquise au Gate G3 : 640 px,
voie réelle, mais bâtiment cible absent du cadre. Indécis n'est pas réfuté —
la vue montre quelque chose, sans qu'on puisse le rattacher à l'empreinte.

`ENTRANCE_MAIN_CURRENT`, `PROPERTY_SIGN` et `FACADE_PRIMARY` atteignent
désormais les deux points de vue exigés. Leur seul manque restant est la
**continuité**, qui ne se mesure jamais sur un corpus existant : elle demande le
recouvrement entre images, donc le Lot 2.

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
| Entrée actuelle | `ENTRANCE_MAIN_CURRENT confirmed` | atteint |
| Stationnement hôtel | `PARKING_HOTEL inferred` | partiel |
| Corpus minimal autorisé | droits encore non clarifiés sur 146 assets | partiel |
| Séparation temporelle de l'entrée | trois prises concordantes, dates exactes inconnues | partiel |

**Lot 1 : non accepté, mais plus pour les mêmes raisons.**

Deux critères en échec sont devenus l'un atteint, l'autre partiel. L'entrée
actuelle est confirmée : la porte-cochère, ses portes vitrées et l'allée qui y
mène sont identiques sur trois prises séparées — la vue Places portant le numéro
civique 1195, une vue Mapillary de l'hiver 2024, et une vue Street View courante.
Elle n'a donc été ni déplacée ni reconstruite dans cet intervalle.

Le stationnement reste `inferred` et non `confirmed`, délibérément : son
existence est observée sur deux points de vue, mais l'association cadastrale
précédente — `way/1467386732`, qui couvre le 1205 et non le 1195 — demeure
réfutée et n'a pas été rétablie. Une existence photographique ne vaut pas une
emprise.

La séparation temporelle reste partielle : les trois prises concordent, mais
aucune ne porte de date exacte au manifeste, et l'achèvement des travaux
demeure inconnu.

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

### P1 — Campagne de sources : 2/15 → 14/15

Trois voies ont été essayées, et le registre distingue désormais ce qui est
interrogé, ce qui est indisponible et ce qui attend une décision humaine.

**Ce qui a été interrogé.** Trois collecteurs existaient sans avoir jamais
tourné. Wikimedia Commons ne demande aucune clé : la géorecherche rend
**3 images CC BY-SA 4.0** dans un rayon de 500 m — deux du Terminus de
Mortagne, une de la rue Nobel. Ce sont les premières images du corpus à porter
une licence explicitement réutilisable, contre 146 assets `public_uncleared`.
Places et le site officiel ont été rejoués : 10 et 12 résultats, identiques au
corpus — la campagne est donc complète, non lacunaire.

**Ce qui est indisponible, et pourquoi.** Onze reçus documentent une
indisponibilité **observée**, jamais un renoncement : TripAdvisor et Flickr ont
un collecteur mais aucune clé (`provider-check` le confirme) ; Booking, Expedia,
ICE Portal, Foursquare, Yelp/Apple/Bing, les annuaires, KartaView, Panoramax et
les réseaux sociaux officiels n'ont ni collecteur ni accès public — pour
plusieurs, l'accès suppose un contrat commercial ou l'accord de l'exploitant.

**Ce qui reste.** `hotel_project_team` seulement : une demande directe à
l'hôtel, au dossier municipal ou aux intervenants. Aucun mécanisme ne peut s'y
substituer, et rien n'a été inventé pour la clore.

Un verrou de registre a dû être levé au passage. Le registre ne lisait que les
manifestes de candidats comme preuve d'interrogation, or **seule la découverte
ciblée en produit** : un collecteur exécuté directement ne laissait aucune
trace, et sa famille restait ouverte alors qu'elle avait répondu.
`sources queried` consigne cette interrogation, avec sa requête, son nombre de
résultats et sa trace. `--returned 0` est admis : c'est l'interrogation qui
ferme la campagne, pas la moisson — confondre les deux rouvrirait toute source
pauvre. Un manifeste courant continue de primer sur un reçu de campagne.

### P1 — Orthophoto et cadastre : les pistes automatiques sont épuisées

Plutôt que de répéter « acquisition manuelle », les routes plausibles ont été
sondées. Le résultat est négatif et désormais mesuré :

- **GéoMont 2023**, 20 cm, couvre la Montérégie mais **exclut explicitement le
  territoire de la CMM** — et Boucherville en relève. Exclusion déjà inscrite au
  catalogue, vérifiée ;
- le **portail géospatial du gouvernement** (`geoegl.msp.gouv.qc.ca`) expose
  140 couches WFS et 6 716 couches WMS : **aucune couche cadastrale**. Les
  couches `images2024/2025/2026` sont des images **satellitaires** dont
  l'emprise (-75,09 à -74,02) ne contient pas le site (-73,44) ;
- **Données Québec** ne publie pas le cadastre : les « unités d'évaluation
  foncière » existent en GeoJSON, mais par municipalité — Montréal, Rimouski,
  Rouyn-Noranda. Rien pour Boucherville ;
- les services **WFS du MERN** hébergeant l'index LiDAR n'exposent aucun index
  d'imagerie ou d'orthophoto (404 sur les points d'entrée attendus) ;
- les hôtes **CMM** testés ne résolvent pas ou refusent (403).

Conclusion : `manual_acquisition_required` est exact pour le cadastre, et
l'orthophoto CMM à 5 m reste la seule couverture déclarée. Ce n'est plus une
hypothèse du catalogue, c'est un constat.

### P1 fermé — Gates réseau exécutés, `ACCESS_ROAD_MAIN` toujours ouvert

Les trois Gates ont été franchis dans l'ordre, chacun avec son propre
consentement. Le résultat est négatif, et c'est un résultat.

| Gate | Coût réel | Résultat |
|---|---|---|
| G1 découverte ciblée | **0 appel réseau** — 1 hit cache Mapillary, 17 hits Street View | 195 candidats, 40 éligibles, **0 sous 250 m** |
| G2 mesure des volumes | 7 opérations logiques, aucun corps d'image | 134 405 octets, statut `exact` |
| G3 acquisition | 134 394 octets sur 134 405 consentis | 5/5 fichiers, atomique |

G1 n'a rien coûté : le plafond de 25 opérations était planifié, le cache a tout
servi. Le mode hors ligne strict l'avait refusé plus tôt, mais c'était le mode,
non l'absence de données.

Le brouillon `20260817T031056701995Z` prévu au plan a été écarté avant toute
mesure : son `corpus_digest` valait `fbfcfbde8…` quand le corpus courant vaut
`a3ad184b…`. Le mesurer aurait dépensé des appels sur une sélection faite
contre un corpus disparu.

**Ce que les cinq fichiers acquis montrent :**

- la vue `ACCESS_ROAD_MAIN` à 640 px — 67 % du volume — montre bien une voie
  asphaltée à cases numérotées, clôture et lampadaires, mais **le bâtiment
  cible n'y apparaît pas**. La géométrie annonçait 31,6 % de cible dans le
  cadre à 30,2 m ; l'image ne le confirme pas. Il s'agit d'un panorama
  contributeur (© Marc Durand - Panosphere360), dont l'orientation réelle
  diffère du cap demandé de 133,2°. Verdict : **indécis**, avec deux inconnues
  nommées — le raccord à l'empreinte, et l'identité de la voie ;
- les quatre vignettes 256 px sont **réfutées** : une prise de nuit depuis un
  véhicule en mouvement, une autoroute, des pavillons unifamiliaux, et un hall
  de concessionnaire automobile.

`ACCESS_ROAD_MAIN` reste donc ouvert, et le Router maintient
`path_d_hybrid / capture_required`. La demande de capture ne repose plus sur
une absence de recherche : elle repose sur une recherche menée, mesurée et
jugée.

Coût cumulé des trois Gates : **0,13 Mo**, contre 123 Mo déjà présents.

### P0 fermé — Preuve tirée du corpus déjà collecté

Le corpus contenait 604 fichiers pour 123 Mo, dont **596 images de 640 px ou
plus** et 449 de 1920 px ou plus. Aucune ne pouvait servir de preuve : trois
verrous indépendants s'y opposaient, et aucun n'était un manque de données.

1. **`preview assess` exigeait une acquisition ciblée.** Un constat ne pouvait
   porter que sur un fichier commandé *pour* le besoin jugé. La règle est juste
   — une mesure prise pour autre chose ne doit pas créditer un besoin qu'elle
   n'a jamais visé — mais elle rendait inexploitables 329 assets sur 335. Seules
   6 vignettes de 256 px étaient éligibles, et c'est précisément leur résolution
   qui a fait réfuter les neuf constats antérieurs.
2. **`demands assess` ne lisait pas les constats.** Le CLI appelait `assess()`
   sans lui passer `previews` : le paramètre retombait sur `None` et tout
   verdict établi était ignoré en silence. Le mécanisme existait des deux côtés,
   le fil manquait entre les deux.
3. **Aucune commande ne savait résoudre un objet de site.** `site unresolve`
   savait démentir une association ; rien ne savait en établir une. Un objet
   restait `unresolved` même après qu'un relecteur eut vu la chose.

Trois ajouts lèvent ces verrous sans affaiblir la règle qui les motivait :

- `assets preview assess-corpus` constate sur un asset du corpus, avec une
  filiation explicite `corpus:<source>` au lieu d'un `plan_id` inventé. Un
  relecteur voit immédiatement que la vue n'a pas été commandée pour ce besoin
  et peut la contester sur ce motif. Un fichier issu d'un plan ciblé est
  refusé et renvoyé à `preview assess` ;
- `demands assess` reçoit désormais le journal des constats ;
- `site resolve` établit un objet. `confirmed` exige des constats établis sur
  le besoin correspondant ; `inferred` s'en dispense mais réclame la même
  justification écrite. Aucune géométrie n'est produite : le contour reste le
  travail des sources géospatiales.

Résultat, sans un octet téléchargé : objets non résolus **5 → 1**, points de
vue indépendants **1 → 3**, constats établis **0 → 9**, revendications
interdites par le Router **5 → 1**.

### P0 fermé — Ce que le corpus ne montre pas

Trois façades restent ouvertes, et l'examen direct des images explique pourquoi
plutôt que de le supposer. Les vues des secteurs `rear`, `rear_left_corner` et
`rear_right_corner` — pourtant à 65-70 m et en ligne de vue « dégagée » —
montrent des **pavillons unifamiliaux d'une rue résidentielle située derrière
la propriété**. L'hôtel n'y apparaît pas.

Le secteur calculé décrit la position de la caméra par rapport à l'empreinte,
non ce que l'image montre. Trois de ces vues sont désormais rejetées en revue
avec ce motif, ce qui les retire des propositions futures.

`FACADE_LEFT`, `FACADE_REAR` et `FACADE_RIGHT` ne sont donc pas couvrables par
ce corpus. C'est un constat mesuré, et il fonde la demande de capture du Router
au lieu de la laisser reposer sur un décompte.

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
