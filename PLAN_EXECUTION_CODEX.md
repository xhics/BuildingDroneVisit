# Plan d'exécution Codex — fermeture du Lot 1B et paquet 3D

**Créé le :** 17 août 2026  
**Pilote :** `welcominns-boucherville`  
**Branche de départ observée :** `49b5f0f`  
**État du plan :** actif  
**Responsable d'architecture :** OpenAI Codex  
**Principe :** observer, mesurer ou masquer — jamais inventer.

Ce document est le plan opérationnel persistant du goal Codex. Il ne remplace
pas `PLAN_DIRECTEUR_WELCOMINNS.md`, `PLAN_LOT_1B_WELCOMINNS.md` ni
`ARCHITECTURE_ACTUELLE.md`. Il relie ces documents aux actions réellement
exécutées, aux artefacts produits et aux conditions de reprise.

## 1. Objectif du goal

1. Fermer le Lot 1B avec un verdict factuel et reproductible.
2. Atteindre `ENVIRONMENT_3D_READY` si les preuves le permettent ; sinon
   publier un verdict motivé qui interdit de présenter un proxy comme une
   reconstruction validée.
3. Produire un paquet de scène 3D provider-agnostic, local, traçable et
   réutilisable, sans lancer le Lot 2 ni un fournisseur vidéo réel.
4. Valider les schémas, les empreintes, les tests, les artefacts et la
   documentation avant la remise finale.

## 2. État de départ vérifié

- Router : `PATH_D_HYBRID / CAPTURE_REQUIRED`.
- Cause de capture : `obligation:ACCESS_ROAD_MAIN`, ciblable mais sans preuve
  photographique ni proxy routier qualifié.
- Corpus : 335 assets, 9 porteurs de géométrie, 1 point de vue indépendant.
- Géospatial : terrain et toiture qualifiés comme inférences ; LiDAR acquis.
- Objets encore non résolus : `PROPERTY_PARCEL` seulement. Les quatre autres ont
  été établis depuis le corpus déjà collecté, sans acquisition nouvelle.
- Orthophoto CMM : couverture connue, non acquise, contexte seulement.
- Cadastre : acquisition manuelle requise.
- Lot 2 : non commencé et hors périmètre de ce goal.

## 3. Séquence d'exécution

### E1 — Stabiliser les livrables canoniques du Lot 1B

**État : terminé et republié sur le corpus courant.**

Sorties :

- `coverage/coverage_report.json` ;
- `coverage/zone_confidence.geojson` ;
- `coverage/camera_constraints.json` ;
- `coverage/context_manifest.json` ;
- `coverage/capture_brief.md` ;
- `LOT_1B_REPORT.md` racine et copie sous le workspace.

Critères de sortie :

- schémas relus sans défaut ;
- empreintes et références cohérentes ;
- cinq objets non résolus conservés comme tels ;
- aucune source ne prétend établir parcelle ou toiture hors de ses capacités.

### E2 — Statuer honnêtement sur `ENVIRONMENT_3D_READY`

**État : terminé.**

Verdict courant : `NEEDS_AUTHORIZED_CAPTURE`.

Motifs :

- entrée actuelle non établie ;
- stationnement non établi ;
- un seul point de vue indépendant ;
- aucune mesure SfM ;
- aucune approbation humaine finale du Gate.

Règle : le verdict ne passe à `ENVIRONMENT_3D_READY` que si tous les Gates
requis sont `passed`, sans blocage et avec approbation humaine explicite.

### E3 — Produire le paquet de scène 3D provider-agnostic

**État : terminé et republié sur le corpus courant.**

Sorties sous `08_composite/` :

- volume `environment.obj` et matériau ;
- DTM, DSM toiture et nDSM actifs, copiés et vérifiés par SHA-256 ;
- carte de confiance et contraintes caméra ;
- orbite virtuelle de 12 poses, explicitement `simulation_only` ;
- contrat de prompts sans appel à un fournisseur réel ;
- script d'import Blender ;
- `phase1_verdict.json`, `scene.json` et pointeur canonique.

Limite : le paquet est un `hybrid_proxy_package`, jamais un splat, un SfM ou
une reconstruction photoréaliste.

### E4 — Fermer la boucle ciblée `ACCESS_ROAD_MAIN`

**État : en cours, brouillon ciblé produit.**

Travail local déjà effectué :

- découverte ciblée exacte sur un seul besoin ;
- cache-only, zéro appel réseau ;
- 17 positions Street View pertinentes au lieu de 1 215 ;
- un candidat Mapillary recommandé en aperçu, mais distant de 345 m ;
- aucun candidat sous le seuil automatique de 250 m ;
- plan brouillon courant `20260817T031056701995Z`, limité au couple
  `mapillary-817552220025789 / obligation:ACCESS_ROAD_MAIN` ;
- résolution `thumb_256`, niveau `recommended_for_preview`, volume encore
  inconnu ; aucun HEAD, GET ou consentement exécuté ;
- filiation explicite vers
  `01_sources/targeted/20260817T031050597389Z/candidates_20260817T031050597389Z.json` ;
  les brouillons antérieurs sont invalidés append-only et leurs fichiers
  restent intacts. Le plus récent a été retiré pour `stale_corpus` après
  l'ajout des signatures robustes.

Travail immédiat :

1. obtenir le Gate pour mesurer le brouillon : deux opérations logiques
   Mapillary au maximum (résolution de l'URL puis mesure du volume), aucun corps
   d'image ;
2. publier le plan mesuré sans reconstruire la sélection ;
3. soumettre séparément le nombre exact d'octets au consentement ;
4. acquérir atomiquement l'aperçu seulement après ce consentement ;
5. soumettre l'aperçu à `PreviewAssessment` ;
6. republier besoins, couverture et Router seulement si une preuve est établie.

Gate réseau prévisible si le cache ne suffit pas : au plus 17 opérations de
métadonnées Street View pour ce seul besoin, puis un budget d'image séparé et
mesuré. Le consentement à l'un ne vaut pas consentement à l'autre.

### E5 — Fermer les prérequis non photographiques du Lot 1B

**État : non terminé.**

La définition de DONE de `PLAN_LOT_1B_WELCOMINNS.md` exige davantage que le
seul besoin porté par le Router :

1. acquérir et qualifier l'orthophoto couvrant la propriété ; la déclaration
   territoriale CMM à 5 m ne vaut pas acquisition ;
2. acquérir manuellement l'extrait cadastral ou conserver explicitement
   `PROPERTY_PARCEL` non résolu avec le Gate externe correspondant ;
3. terminer ou motiver l'indisponibilité des familles de sources prioritaires ;
   le registre canonique est publié et établit actuellement que 2 familles
   requises sur 15 seulement sont closes. Le **mécanisme** de reçu est
   désormais livré (`sources unavailable` / `sources reopen`, append-only) :
   ce qui manque n'est plus du code, mais le constat humain famille par
   famille ;
4. **fermé** — la déduplication robuste est appliquée aux 335 assets : 30 891
   couples plausibles examinés, aucune fusion robuste sur le corpus réel, et
   régressions recadrage, filigrane et image distincte exécutées par la
   commande de production. G1 est `passed` ;
5. **fermé pour quatre objets sur cinq** — le réexamen a bien porté sur des
   preuves, mais elles étaient déjà au corpus : 596 images en pleine résolution
   qu'aucun mécanisme ne savait convertir en constat. `ENTRANCE_MAIN_CURRENT`,
   `PROPERTY_SIGN` et `DRIVEWAY_MAIN` sont confirmés ; `PARK_AND_RIDE` et
   `PARKING_HOTEL` sont inférés. `PROPERTY_PARCEL` reste non résolu et le
   restera tant que le cadastre n'est pas acquis.

Ces actions ne sont pas remplacées par le paquet proxy. Toute opération réseau,
manuelle ou susceptible d'engager un coût reçoit son propre Gate et son propre
budget avant exécution.

### E6 — Validation finale et documentation

**État : en cours.**

À exécuter après E4/E5 ou pendant l'attente d'une autorisation :

1. tests ciblés des modules modifiés ;
2. suite complète `pytest` ;
3. `compileall` et `pip check` ;
4. `git diff --check` ;
5. relecture de chaque fichier du paquet et de son SHA-256 ;
6. validation des JSON par leurs modèles Pydantic ;
7. mise à jour cohérente de `ETAT_IMPLEMENTATION.md`,
   `ARCHITECTURE_ACTUELLE.md` et `LOT_1B_REPORT.md` ;
8. rapport final distinguant : livré, non atteint, preuve manquante et prochaine
   action exacte.

## 4. Gates et autorisations

| Gate | Condition | Action sans autorisation |
|---|---|---|
| Réseau métadonnées | appels externes nécessaires | arrêter avant l'appel et annoncer le plafond exact |
| Téléchargement | taille et requêtes résolues | mesurer d'abord ; demander un consentement lié aux empreintes |
| Source payante | coût possible | ne jamais engager automatiquement |
| Décision humaine | identité, aptitude, Gate final | écrire un état en attente, ne pas simuler l'approbation |
| Lot 2 / GPU | SfM, LightGlue, pycolmap, splat | hors périmètre ; ne pas lancer |

## 5. Définition de DONE

Le goal est terminé uniquement lorsque :

- le Lot 1B possède des artefacts canoniques cohérents et un verdict final ;
- le Router a été republié sur les entrées courantes ;
- le statut Phase 1 est soit `ENVIRONMENT_3D_READY` réellement démontré, soit
  un refus motivé et versionné qui nomme les preuves manquantes ;
- le paquet 3D provider-agnostic est vérifié et reproductible ;
- tous les tests passent, `git diff --check` est muet et la documentation est
  alignée ;
- aucune donnée n'a été inventée, aucun appel payant n'a été lancé sans accord.

Un arrêt devant un Gate d'autorisation ne vaut pas DONE : le plan reste actif
et reprend à la première action locale ou autorisée suivante.

### Audit de complétude du Lot 1B

| Exigence normative | État courant | Preuve ou manque |
|---|---|---|
| Familles prioritaires interrogées ou indisponibilité motivée | échec mesuré | `source_registry.json` : 2/15 familles requises closes |
| Republications neutralisées | livré | 335/335 ; pHash et hash robuste, 30 891 couples plausibles, trois régressions de production, G1 `passed` |
| Points de vue indépendants par secteur | livré | 3 points de vue ; besoins et Router les comptent, pas les fichiers |
| Objets critiques établis sur preuve | livré sauf parcelle | 4 confirmés, 9 inférés, 1 non résolu ; 0 octet téléchargé |
| `unknown` et preuves conservées | livré | revue et classification append-only |
| Street View multi-position | livré | positions de corridor, panorama distinct du cadrage |
| LiDAR, MNT et orthophoto acquis et qualifiés | échec | LiDAR/MNT présents ; orthophoto CMM non acquise |
| Toiture et volume proxy automatiques | livré | rasters actifs et paquet `hybrid-1.2.1` |
| Contexte protégé | livré | `context_manifest.json` et règles de préservation |
| Zones `trusted/proxy/unobserved` | livré | `zone_confidence.geojson` |
| Contraintes caméra | livré | `camera_constraints.json` |
| Brief minimal de capture | livré | seulement `ACCESS_ROAD_MAIN` |
| Rapport permettant de décider sans Lot 2 | livré avec verdict négatif | `incomplete_capture_required` |

Conséquence : le Lot 1B possède son rapport final mais n'atteint pas encore sa
définition de DONE. La fermeture ne sera déclarée qu'après résolution des
échecs ci-dessus ou modification explicite du document normatif par l'opérateur.

## 6. Point de reprise courant

**E4 est terminé.** Les trois Gates ont été franchis le 17 août 2026, chacun
avec son consentement propre :

1. **G1** — redécouverte ciblée sur le corpus courant. Le brouillon
   `20260817T031056701995Z` a été écarté sans être mesuré : son `corpus_digest`
   ne correspondait plus. Coût réel : **0 appel réseau**, le cache ayant servi
   les 18 opérations planifiées. Résultat : 195 candidats, 40 éligibles,
   **0 sous 250 m** — le plus proche à 345 m.
2. **G2** — mesure des volumes : 7 opérations logiques, aucun corps d'image,
   **134 405 octets** en statut `exact`.
3. **G3** — acquisition atomique après consentement au total exact :
   **134 394 octets sur 134 405**, 5/5 fichiers.

**Résultat : `ACCESS_ROAD_MAIN` reste ouvert.** La vue à 640 px montre une voie
asphaltée réelle, mais sans le bâtiment cible dans le cadre — verdict
**indécis**, avec deux inconnues nommées. Les quatre vignettes 256 px sont
réfutées : nuit en mouvement, autoroute, pavillons, hall de concessionnaire.

Le Router maintient `path_d_hybrid / capture_required`. La demande de capture
ne repose plus sur une absence de recherche, mais sur une recherche menée,
mesurée et jugée.

**Point de reprise courant :** aucun Gate réseau ne reste utile pour ce besoin.
Élargir la recherche au-delà du corridor résolu serait un nouveau Gate, à
demander explicitement — la recherche ne s'élargit jamais implicitement. Les
prochains blocages sont non photographiques : orthophoto, cadastre, et la
campagne de sources.
