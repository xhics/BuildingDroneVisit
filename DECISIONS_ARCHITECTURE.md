# Registre des décisions d'architecture

**État :** 16 août 2026

Ce registre conserve les choix qui gouvernent déjà le code mais n'étaient pas
réunis dans les plans. Il ne remplace pas les preuves propres au pilote.

Statuts :

- **adoptée** : décision implémentée et normative ;
- **provisoire** : implémentée, mais calibrée sur trop peu de sites ;
- **à fermer** : direction retenue, raccord incomplet.

---

## D-001 — Exprimer les besoins avant la collecte

**Statut : adoptée**

**Décision.** Les objectifs vivent dans `CaptureDemandManifest`. La découverte
et l'acquisition répondent à ces objectifs ; elles ne les déduisent pas du
corpus déjà disponible.

**Conséquence.** Un corpus volumineux peut rester insuffisant. La couverture se
compte en points de vue indépendants satisfaisant un besoin, pas en fichiers.

---

## D-002 — Conserver l'inconnu comme état de plein droit

**Statut : adoptée**

**Décision.** Une valeur manquante produit `unknown`, `unresolved`, `stale`,
`preview_required` ou un refus explicite selon sa nature.

**Conséquence.** Il est interdit de remplir l'absence avec une valeur du pilote,
une date supposée, un CRS historique ou une géométrie proche.

---

## D-003 — Séparer profil d'établissement et politique de pipeline

**Statut : adoptée**

**Décision.** Le profil contient les faits du site. La politique contient les
seuils génériques. Les calibrations déclarent les sites qui les ont éprouvés.

**Conséquence.** Un hôtel ne peut pas modifier ses seuils afin de réussir le
Gate. Un changement réel de profil déplace son empreinte.

---

## D-004 — Périmer par dépendance, pas par provenance générale

**Statut : à fermer**

**Décision.** Les facettes de politique et de profil déterminent ce qu'un
changement invalide réellement.

**Conséquence.** Modifier un nom ne périme pas un DTM. Modifier un seuil de
cadrage périme les évaluations et plans, pas le LAZ acquis.

**Reste.** La péremption de politique est appliquée. Celle des facettes de
profil et le registre de calibration doivent être davantage raccordés aux
commandes de production.

---

## D-005 — Instancier les objets critiques, y compris les exclusions

**Statut : adoptée**

**Décision.** `SiteManifest` porte des instances réelles et leurs relations. Un
objet attendu reste `unresolved` s'il n'est pas établi.

**Conséquence.** Un voisin ou parc-o-bus n'est pas seulement un terme négatif :
sa distinction avec le bâtiment cible peut être vérifiée.

---

## D-006 — Ne jamais réécrire une décision humaine

**Statut : adoptée**

**Décision.** Revues, aptitudes, droits, constats et corrections sont
append-only. Une correction référence ce qu'elle supersède.

**Conséquence.** Attribution, aveuglement, preuves et erreurs antérieures restent
auditables.

---

## D-007 — Séparer identité, visibilité et aptitude géométrique

**Statut : adoptée**

**Décision.** Voir un bâtiment ne signifie pas voir la cible ; voir la cible ne
signifie pas disposer d'une bonne vue de reconstruction.

**Conséquence.** `GeometrySuitability` possède son propre historique. Une vue
lointaine peut confirmer l'identité et rester `auxiliary` ou `insufficient`.

---

## D-008 — Évaluer et recommander par couple candidat/besoin

**Statut : adoptée**

**Décision.** `CandidateEvaluation` et `DemandRecommendation` sont adressés par
le couple, pas seulement par le candidat.

**Conséquence.** Une autorisation full pour la voie d'accès ne contamine pas la
façade arrière servie en preview par le même cadrage.

---

## D-009 — Rechercher par géométrie utile avant téléchargement

**Statut : adoptée, seuils provisoires**

**Décision.** Les candidats sont classés depuis les corridors, la cible propre
au besoin, le secteur, l'orientation, la visibilité en plan, la distance, la
parallaxe et la continuité potentielle.

**Conséquence.** Une rue non adjacente ne devient pas utile par sa seule présence
dans un rayon. Les seuils encore calibrés sur un site restent déclarés comme
provisoires.

---

## D-010 — Distinguer preview et acquisition complète

**Statut : adoptée**

**Décision.** Une métrique requise mais inconnue force une miniature. Seul un
`PreviewAssessment established` peut rendre le couple promouvable.

**Conséquence.** Une impossibilité de calcul n'est pas traitée comme un succès.
Un aperçu réfuté ne ferme pas le besoin et ne doit pas être reproposé.

---

## D-011 — Résoudre la requête fournisseur avant le consentement

**Statut : adoptée**

**Décision.** Le plan porte `provider_resolution`, `request_spec` et
`request_digest`. HEAD et GET consomment la même requête résolue.

**Conséquence.** Une résolution sémantique comme `full_available` ne peut pas se
transformer après consentement en requête fournisseur différente.

---

## D-012 — Un seul transport, comptable et sans secrets

**Statut : adoptée**

**Décision.** Tout accès fournisseur traverse le transport commun. Chaque
tentative, redirection et issue est comptée, sans publier l'URL signée.

**Conséquence.** Opérations logiques, échanges HTTP, cache et octets sont
distincts. `forbidden` précède toute lecture du cache.

---

## D-013 — Consentement borné et publication atomique

**Statut : adoptée**

**Décision.** Le consentement porte sur les requêtes exactes, le volume et la
version du contrat de téléchargement. Les fichiers transitent par staging.

**Conséquence.** Le chunk qui dépasserait la borne n'est pas écrit. Format,
dimensions et SHA-256 sont vérifiés. Une acquisition multiple publie tous les
fichiers ou aucun.

---

## D-014 — Conserver les plans et artefacts remplacés

**Statut : adoptée**

**Décision.** Un plan ou artefact périmé est invalidé ou supersédé dans un
registre append-only ; son fichier n'est ni réécrit ni supprimé.

**Conséquence.** `_latest_plan` ne considère que les plans courants et engagés,
sans repli silencieux vers un brouillon ancien.

---

## D-015 — Le LiDAR mesure le volume, pas l'apparence ni la parcelle

**Statut : adoptée**

**Décision.** Le LiDAR peut établir terrain et toiture par dérivation. Il ne
fonde jamais une limite cadastrale ni l'apparence actuelle d'une façade.

**Conséquence.** `TERRAIN_MAIN` et `ROOFLINE_MAIN` restent `inferred`. La
parcelle attend le cadastre et l'apparence attend une preuve photographique.

---

## D-016 — Représenter l'incertitude d'occlusion par deux bornes

**Statut : adoptée**

**Décision.** La visibilité multi-rayons publie une borne inférieure prouvée et
une borne supérieure incluant les obstacles dont la hauteur est inconnue.

**Conséquence.** Une géométrie 2D ne produit jamais à elle seule un blocage
vertical prouvé. Un corridor utile géométriquement reste séparé de son statut
d'accès.

---

## D-017 — Orienter le bâtiment depuis les façades documentées

**Statut : à fermer**

**Décision.** L'orientation principale vient des normales de segments
colinéaires que des images d'identité confirmée désignent comme façade avant.
Le stationnement et la voie d'accès peuvent corroborer, jamais décider seuls.

**Conséquence.** Des preuves désignant des façades opposées laissent
l'orientation non résolue. Une nouvelle orientation périme tous les secteurs et
calculs dérivés.

**Reste.** Fermer la validation persistante de la preuve et la propagation
atomique vers les 313 assets positionnés et leurs productions aval.

---

## D-018 — Le Router juge des besoins et explique avec les objets

**Statut : à fermer**

**Décision.** La complétude se juge sur `CaptureDemand`. Les objets du site,
artefacts, lacunes et droits expliquent la décision et la route possible.

**Conséquence.** Le Router ne se contente pas de compter des types d'objets, et
un besoin ne disparaît pas parce que sa cible reste non résolue.

---

## D-019 — Préserver l'environnement au moyen de preuves et contraintes

**Statut : à fermer**

**Décision.** Les zones environnantes doivent être `trusted`, `proxy` ou
`unobserved`. Les caméras futures évitent les zones proxy ou inconnues dans les
plans rapprochés.

**Conséquence.** L'IA ne régénère pas librement le contexte. Une capture sur
site n'est demandée que pour une zone nécessaire que les sources distantes ne
peuvent établir.

---

## D-020 — Arrêter la Phase 1 avant le tournage

**Statut : adoptée**

**Décision.** La Phase 1 se termine à `ENVIRONMENT_3D_READY`.

**Conséquence.** Trajectoires créatives, génération vidéo, montage et QA de la
vidéo appartiennent à une Phase 2 séparée.

