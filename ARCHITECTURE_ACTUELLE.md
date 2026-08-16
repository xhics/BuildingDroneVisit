# Architecture actuelle — BuildingDroneVisit

**État vérifié :** 16 août 2026  
**Périmètre :** pipeline générique et pilote WelcomINNS Boucherville  
**Jalon stable observé :** `2977969`

Ce document décrit le système réellement implémenté. Il complète les documents
de destination suivants sans les remplacer :

- `PLAN_DIRECTEUR_WELCOMINNS.md` définit la cible de la Phase 1 ;
- `PLAN_LOT_1B_WELCOMINNS.md` définit le résultat attendu avant le Lot 2 ;
- `PLAN_IMPLEMENTATION_WELCOMINNS.md` décrit les contraintes d'exécution ;
- `ETAT_IMPLEMENTATION.md` confronte cette cible à l'état courant ;
- `DECISIONS_ARCHITECTURE.md` explique les choix structurants.

En cas de contradiction sur le **comportement actuel**, le code, ses schémas et
les artefacts vérifiés priment sur les anciens exemples de commandes. En cas de
contradiction sur la **destination du produit**, le plan directeur reste la
référence.

---

## 1. Frontière fonctionnelle

La Phase 1 s'arrête à `ENVIRONMENT_3D_READY` : un environnement 3D
reproductible, inspectable et accompagné d'une carte de confiance. Elle ne
produit ni tournage, ni montage, ni vidéo finale.

Le dépôt possède aujourd'hui deux niveaux d'exécution :

1. un pipeline Lot 1B moderne, exposé par les sous-commandes `assets`, `geo`,
   `visibility`, `site`, `policy` et `temporal` ;
2. l'orchestrateur historique `run-phase1`, dont seule l'étape `collect` est
   construite et dont les étapes suivantes lèvent `StepNotImplemented`.

Les sous-commandes modernes sont la voie de travail réelle. `run-phase1` ne
devient canonique que lorsqu'il orchestre ces mêmes contrats sans les doubler.

---

## 2. Invariants d'architecture

### 2.1 Rien n'est inventé

Une absence de preuve produit `unknown`, `unresolved`, `stale`,
`preview_required` ou `unreachable` avec motif. Elle ne devient jamais une
valeur du pilote par défaut.

### 2.2 L'objectif précède la collecte

`CaptureDemand` décrit ce qu'il faut obtenir avant d'interroger les sources.
Le corpus ne redéfinit donc pas silencieusement le niveau de couverture attendu.

### 2.3 Les décisions portent sur le bon couple

Une photographie peut convenir à un besoin et être rejetée pour un autre. Les
évaluations, recommandations et constats d'aperçu sont adressés par
`(candidate_id ou asset_id, demand_id)`.

### 2.4 Un fait, une décision et une autorisation sont distincts

- l'acquisition constate l'origine et le contenu d'un fichier ;
- la revue établit ou réfute l'identité et l'aptitude ;
- une décision de droits autorise éventuellement un usage ;
- aucun de ces actes ne vaut automatiquement pour les deux autres.

### 2.5 Toute mutation importante est traçable

Les revues, constats, invalidations, migrations, consentements et changements
de vérité sont append-only ou portent une filiation explicite. Un résultat
périmé est conservé, mais ne reste pas courant.

### 2.6 La provenance ne vaut pas dépendance

Un rapport peut citer un profil ou une politique pour être lisible sans que
tous leurs champs aient influencé son calcul. Les empreintes de dépendance sont
donc séparées de la provenance générale.

### 2.7 Aucun réseau implicite

Le transport possède trois modes fermés :

- `online` : les appels autorisés sont enregistrés ;
- `cache_only` : un cache incompatible ou absent est un refus explicite ;
- `forbidden` : aucune lecture de cache ne contourne l'interdiction.

---

## 3. Sources de vérité

| Vérité | Artefact ou schéma | Responsabilité |
|---|---|---|
| Projet | `00_manifest/project.json` | identité de l'espace de travail et profil choisi |
| Politique | `00_manifest/pipeline_policy.json` | seuils décisionnels matérialisés |
| Profil | `profiles/<property_id>.json` | noms, langues, travaux et indices propres au site |
| Référentiel spatial | `00_manifest/spatial_reference.json` | territoire, CRS horizontal et vertical |
| Bâtiment candidat | `00_manifest/spatial_manifest.json` | empreinte confirmée et orientation courante |
| Objets du site | `00_manifest/site_manifest.json` | instances, états, relations et preuves |
| Médias acquis | `00_manifest/asset_manifest.json` | fichiers, provenance, droits et décisions humaines |
| Géométrie de capture | `00_manifest/capture_geometry.json` | cible, routes, corridors et obstacles |
| Besoins | manifeste `CaptureDemandManifest` | objectifs immuables de couverture |
| État des besoins | `DemandAssessmentManifest` | mesure contre un corpus précis |
| Candidats | `CandidateManifest` | métadonnées, évaluations et recommandations |
| Dépense proposée | `AcquisitionPlan` | requêtes résolues, volumes et niveaux par besoin |
| Aperçus jugés | `PreviewAssessmentLog` | établi, réfuté ou indécis par couple |
| Géodonnées dérivées | `SiteManifest.derived_artifacts` et `06_geo/` | rasters, masques, filiation et qualification |
| Visibilité | `VisibilityRun` et reçus d'application | fractions multi-rayons et risques d'occlusion |

Un rapport secondaire ne doit pas devenir une nouvelle source de vérité. Il
doit citer l'empreinte des manifestes qu'il résume.

---

## 4. Contexte d'exécution générique

### 4.1 Profil et politique

`PropertyProfile` contient les faits propres à l'établissement. `PipelinePolicy`
contient les règles génériques et les seuils calibrés. Une politique n'est pas
personnalisée pour faire réussir un site.

Les capacités déclarent leurs prérequis : une commande qui classe une identité
exige un profil ; une commande géospatiale exige un contexte spatial ; une
qualification exige les seuils matérialisés.

### 4.2 Facettes de politique

La politique est divisée selon les productions qu'un changement doit périmer :

- découverte de collecte ;
- taille d'acquisition ;
- géométrie des candidats ;
- objectifs de couverture ;
- préférences de recherche ;
- visibilité ;
- résolution du bâtiment ;
- déduplication ;
- classification ;
- temporalité ;
- dérivation et qualification géospatiales.

Une nouvelle valeur de politique doit appartenir à une facette ou être déclarée
hors facette avec justification. Un champ orphelin est une erreur de contrat.

### 4.3 Portabilité spatiale

Le territoire et le CRS sont résolus depuis la position, jamais depuis le nom
du pilote. Toute géométrie calculée doit rester dans l'emprise d'usage du CRS.
Deux hauteurs de référentiels différents ne sont comparables qu'au moyen d'une
transformation verticale déclarée et vérifiable.

Le smoke test de portabilité couvre notamment le Québec, Lyon en `EPSG:2154` et
un territoire non pris en charge.

---

## 5. Structure de vérité du site

Le gabarit décrit des types génériques comme `BUILDING_MAIN`,
`ENTRANCE_MAIN_CURRENT`, `PARKING_HOTEL` et `PROPERTY_PARCEL`. Le manifeste de
site instancie ces types pour un établissement précis.

États :

- `confirmed` : preuve directe et vérifiable ;
- `inferred` : dérivation ou association motivée ;
- `unresolved` : objet attendu mais non établi ;
- `stale` : résultat autrefois utilisable, périmé par une vérité nouvelle.

Une exclusion est elle aussi une instance. Un parc-o-bus ou un immeuble voisin
n'est pas seulement un mot à bannir : c'est un objet distinct relié au site.

Les rasters 2,5D vivent dans `DerivedArtifact`, pas dans un WKT. Ils portent
résolution, domaine de couverture, fractions mesurées/interpolées, filiation,
CRS, empreinte et statut.

---

## 6. Qualification des médias

### 6.1 Déduplication

La chaîne distingue :

1. fichier exact par checksum ;
2. photographie republiée par pHash ;
3. point de vue par position, panorama ou séquence ;
4. recouvrement utile au sein d'une grappe.

Les membres ne sont pas supprimés. Un canonique est choisi et les autres
restent auditables. L'embedding robuste aux recadrages et filigranes n'est pas
encore implémenté.

### 6.2 Trois questions indépendantes

Pour chaque média :

1. montre-t-il un bâtiment ?
2. montre-t-il **le bâtiment cible** ?
3. est-il géométriquement `primary`, `auxiliary` ou `insufficient` ?

Une réponse positive à la première ne répond jamais aux deux suivantes.

### 6.3 Revue humaine

Les historiques de visibilité et d'aptitude sont append-only. Une correction
supersède une entrée antérieure sans la réécrire. Une revue aveugle porte son
protocole, l'empreinte de la cohorte, l'ordre présenté et la référence utilisée.

`human_unresolved` est terminal : il signifie que l'image a été examinée et que
les preuves ne permettent pas de conclure. Une nouvelle décision exige une
nouvelle entrée.

### 6.4 Temporalité et droits

La temporalité est évaluée par portée : entrée, façade, toiture, enseigne. Une
approbation municipale ne vaut ni début ni achèvement de travaux.

Les droits sont évalués séparément. `public_uncleared` reste une réserve. Un
usage assumé doit être explicitement autorisé et tracé ; l'acquisition ne prend
jamais cette décision.

---

## 7. Collecte ciblée V2

```text
SiteManifest + CaptureGeometry + politique
                    ↓
           CaptureDemandManifest
                    ↓
             demands assess
                    ↓
     discover metadata-only par corridors
                    ↓
 CandidateEvaluation par candidat et besoin
                    ↓
  recherche adaptative, secteurs, séquences,
      parallaxe potentielle et continuité
                    ↓
       recommandation full ou preview
                    ↓
          AcquisitionPlan draft
                    ↓
      mesure HEAD du plan déjà arrêté
                    ↓
   consentement lié aux digests et au plafond
                    ↓
       acquisition bornée et atomique
                    ↓
 PreviewAssessment par asset et besoin
                    ↓
          nouvelle évaluation des besoins
```

### 7.1 Découverte

`CaptureCandidate` ne porte que des métadonnées. La découverte ne télécharge
aucune image. Les candidats rejetés restent publiés avec leur motif afin qu'un
zéro ne masque pas une source non interrogée.

Street View part des corridors et produit des cadrages vers les cibles.
Mapillary utilise les secteurs déficitaires et les métadonnées de séquence
rendues par sa recherche. Le moteur d'expansion bornée d'une séquence existe,
mais sa seconde passe n'est pas encore raccordée au fournisseur : le rapport
doit donc la déclarer `skipped` plutôt que publier un zéro ambigu. Même connue,
la continuité dans une séquence reste `potential` avant mesure visuelle et
enregistrement géométrique.

### 7.2 Plan et consentement

Une recommandation appartient au couple candidat/besoin. Si le même fichier est
`full` pour un besoin et `preview` pour un autre, le niveau le plus prudent
gouverne le fichier, tandis que les deux décisions restent visibles.

La résolution sémantique du plan est traduite en résolution fournisseur par
`ResolvedAcquisitionRequest`. Son `request_digest` traverse HEAD, consentement,
téléchargement et provenance.

### 7.3 Transport et publication

Le registre de transport sépare :

- opérations logiques prévues ;
- opérations logiques exécutées ;
- échanges HTTP réels ;
- lectures de cache ;
- octets déclarés, reçus, stagés et publiés.

Le corps est écrit dans un staging borné par le consentement et la limite par
fichier. Format, dimensions et checksum sont vérifiés avant publication. Une
acquisition multiple publie tous ses fichiers ou aucun.

### 7.4 Boucle d'aperçu

`PreviewAssessment` rend :

- `established` : toutes les métriques exigées sont mesurées ;
- `refuted` : cet élément ne répond pas à ce besoin ;
- `inconclusive` : il reste des métriques explicitement non mesurées.

Un aperçu réfuté ne ferme pas le besoin et ne doit pas être racheté pour le
même couple lors de la recherche suivante.

---

## 8. Géospatial et visibilité

### 8.1 Routage des sources

Une source territorialement admissible n'est pas déclarée couverte avant
intersection avec son index. `covered`, `not_covered`, `discovery_error`,
`manual_acquisition_required` et `unknown` restent distincts.

Le LiDAR du Québec possède une découverte et une acquisition réelles. Les
orthophotos CMM et le cadastre sont catalogués avec leurs limites, mais ne
possèdent pas encore le même raccord d'acquisition.

### 8.2 Terrain et toiture

Le terrain est interpolé depuis la classe sol autour de l'emprise. La toiture
est mesurée depuis la classe bâtiment. Le nDSM n'existe que là où terrain et
toiture sont tous deux définis.

La qualification distingue une toiture largement mesurée d'un terrain
interpolé sous le bâtiment. Une dérivation reste `inferred`, jamais
`confirmed`, même si ses seuils sont franchis.

### 8.3 Visibilité multi-rayons

La silhouette de la cible est échantillonnée en plusieurs cellules angulaires.
Chaque cellule appartient exactement à une catégorie géométrique. Le rapport
publie une borne inférieure de visibilité prouvée et une borne supérieure qui
inclut les obstacles de hauteur inconnue.

Un obstacle sans données verticales suffisantes produit un risque, jamais un
blocage prouvé. Les cadrages de caméra sont calculés séparément de la visibilité
géométrique.

---

## 9. Orientation des façades

L'orientation ne doit plus être déduite d'un stationnement supposé. La méthode
courante :

1. segmenter l'anneau extérieur de l'empreinte ;
2. calculer les normales extérieures ;
3. regrouper les segments colinéaires ;
4. confronter des vues d'identité confirmée aux segments qu'elles documentent ;
5. accepter seulement des preuves convergeant vers le même groupe de façade.

Une contradiction laisse l'orientation `unresolved`; elle n'est jamais résolue
par une moyenne entre des murs opposés.

Une nouvelle orientation doit périmer et recalculer les secteurs des assets,
les évaluations de candidats, la visibilité, les recherches adaptatives, les
plans et les évaluations de besoins qui en dépendent.

Ce raccord est fermé sur le pilote : `orientation apply` vérifie l'empreinte du
bâtiment et les SHA-256 des preuves, réinstancie les quatre façades, recalcule
les secteurs avec le relèvement géodésique canonique, périme les productions de
visibilité et active les besoins dérivés. Une vérification indépendante avec
`pyproj.Geod.inv` retrouve zéro divergence sur les 313 assets positionnés.

---

## 10. Router et reconstruction

Le Router prévu doit transformer les preuves en une décision explicite :

- photo-first ;
- geo-first ;
- hybride ;
- capture complémentaire ;
- arrêt motivé.

Il doit juger la complétude sur les `CaptureDemand`, puis expliquer cette
décision par les objets du site, les zones de confiance et les contraintes de
caméra. Les objets ne remplacent pas les besoins : ils disent **quoi** existe ;
les besoins disent **quelle observation** est nécessaire.

Le Router n'est pas encore implémenté. Les étapes SfM, reconstruction,
alignement, composite et validation finale restent également hors de l'état
courant.

---

## 11. Règles pour toute nouvelle contribution

1. Ne pas ajouter de repli propre à un établissement dans un module générique.
2. Matérialiser tout seuil décisionnel dans la politique et sa facette.
3. Produire un état explicite lorsque la donnée manque.
4. Fermer les relations entre manifestes au point de liaison.
5. Valider entièrement en mémoire avant la première mutation.
6. Enregistrer la tentative réseau avant l'appel et son issue après.
7. Ne jamais publier de secret, URL signée ou jeton dans un rapport.
8. Conserver les résultats remplacés et leur motif de péremption.
9. Ajouter un contrôle négatif qui prouve que la garde nouvelle mord.
10. Vérifier sur une réponse authentique lorsque le contrat dépend d'un service.
11. Distinguer tests verts, qualité des composants et Gate métier.
12. Mettre à jour `ETAT_IMPLEMENTATION.md` quand un Gate change d'état.
