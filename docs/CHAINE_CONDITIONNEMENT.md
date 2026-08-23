# Chaîne de conditionnement — de l'adresse aux cartes qui contraignent

## L'enchaînement

```bash
# 1. ancres : lues sur les enseignes, sans confirmation humaine
python -m hotel_pipeline.cli identity discover <hotel> --budget 60 --write

# 2. tri du corpus par identité du bâtiment
python -m hotel_pipeline.cli identity screen <hotel> --top 8

# 3. tuiles LiDAR à l'échelle de la scène, obstacles compris
python -m hotel_pipeline.cli geo discover <hotel> --scene

# 4. cartes de conditionnement le long d'une orbite
python -m hotel_pipeline.cli conditioning render <hotel> --frames 96
```

Chaque étape est facultative : sans dépistage, l'appui photographique n'est pas
évalué ; sans LiDAR, les hauteurs restent des hypothèses **déclarées**.

---

## Les deux appuis, mesurés séparément

Une frame peut être géométriquement irréprochable sans qu'aucune photographie
ne montre cette face du bâtiment. Le générateur y inventerait l'apparence.

| `guidance_mode` | Signification | Conduite |
|---|---|---|
| `geometry_strong` | géométrie attestée, angle photographié | conditionner fort |
| `unreferenced` | **géométrie solide, aucune photographie de cette face** | ne pas contraindre l'apparence |
| `geometry_weak` | > 60 % de pixels en hauteur supposée | silhouette seule |
| `prefer_ungrounded` | cible < 2 % de l'image | ne pas conditionner |

Verdict d'ensemble : `condition_strongly`, `condition_partially`,
`unreferenced_arc`, `prefer_ungrounded`.

L'appui décroît avec l'écart angulaire — plein jusqu'à 25°, nul au-delà de 70° —
plutôt que de basculer d'un coup : une façade reste reconnaissable sur une
trentaine de degrés de rotation.

---

## Hauteurs : trois sources, par ordre de force

1. **nDSM dérivé** — qualifié, et porte une surface de toit triangulée ;
2. **nuage LAZ brut** — pour tout ce que le raster ne couvre pas ;
3. **hypothèse déclarée** — 12 m cible, 8 m voisins, avec son motif.

Sur le pilote : 1 volume par nDSM, 9 complétés au nuage, 18 encore supposés.

Une hauteur non mesurée dit **pourquoi** :

- *hors de la tuile obtenue* → `geo discover --scene` élargit l'emprise ;
- *trop peu de points classés bâtiment* → la donnée manque réellement.

« Hors couverture » est un motif ; « pas mesuré » n'en est pas un.

---

## Ce que la mise au point sur le corpus réel a appris

### Le cap de la caméra n'est pas l'azimut de vue

Un premier diagnostic annonçait un trou angulaire de 161° sur la trajectoire.
Il était faux : le suffixe `_54h_` du nom des recadrages est le **cap de la
caméra**, non l'azimut sous lequel la vue voit le bâtiment — les deux diffèrent
de plus de cent degrés sur ce corpus. Les recadrages sont désormais résolus
vers leur asset source, qui porte `bearing_from_building_deg` mesuré.

### Une bande d'indécision fixe produit un volume arbitraire

±0,06 autour du seuil mettait 32 % du corpus en revue humaine — non parce que
ces images étaient ambiguës, mais parce que la distribution était resserrée. La
bande est maintenant dérivée de la dispersion (0,35 σ, borné) : **19 %**, sans
que le classement des références change.

### La propagation par quasi-doublons ne rendait rien

Approche essayée puis **retirée** : lever les indécisions en propageant le
verdict d'images quasi identiques. Rendement mesuré sur le pilote : 0 sur 51 à
quorum 2, quel que soit le seuil — les indécis n'ont presque jamais deux
voisins tranchés.

### Interroger l'index à l'échelle de la cible seule

La découverte LiDAR portait sur l'empreinte du bâtiment — 72 × 77 m — alors que
la scène rendue s'étend sur plus d'un kilomètre. Vingt volumes sur vingt-sept
tombaient hors des tuiles obtenues. `--scene` interroge l'enveloppe des volumes
réellement rendus.

---

## Le sol : classer ne suffit pas, il faut de la continuité

La nature du sol vient de l'intensité du retour LiDAR — une surface végétale
renvoie davantage qu'un enrobé, l'écart mesuré ici atteignant cinq mille
unités. Mais une classification **cellule par cellule** ignore le voisinage, et
un terrain réel ne ressemble pas à un damier.

Mesuré sur ce pilote avant correction :

| | Avant | Après |
|---|---|---|
| Cellules contredisant leur voisinage | 19 % | **6 %** |
| Plages d'une ou deux cellules | 60 % | **0 %** |
| Nombre de plages | 67 | 21 |

Deux traitements, appliqués après la classification :

1. **vote majoritaire** sur les huit voisins, deux passes. Le seuil est réglé
   haut (62 %) pour qu'une frontière franche — le bord d'une allée — survive au
   lissage au lieu d'être arrondie ;
2. **absorption des plages de moins de trois cellules** dans la nature qui les
   entoure : quelques cellules perdues au milieu d'un stationnement ne
   décrivent pas une pelouse.

Une cellule `indetermine` participe au vote sans jamais l'emporter : elle
marque un doute, pas une nature.

### Un défaut d'indexation qui annulait tout

Le lissage n'a d'abord **rien changé**. Les centres de cellules valent
`(colonne + 0,5) × maille` : arrondir `x / maille` faisait tomber deux cellules
voisines sur le même index, et chaque cellule se retrouvait sans voisin. Le
plancher (`floor`) rétablit la correspondance.

---

## Précision de la toiture, et ce que le LiDAR ne verra jamais

Le nDSM est à cinquante centimètres, mais l'échantillonnage n'en retenait
qu'une cellule sur deux : les décrochements fins — le porche d'entrée, ses
avancées — s'arrondissaient avant d'être rendus. Le pas est passé à un, soit la
pleine résolution du relevé.

| | Pas de 2 | Pas de 1 |
|---|---|---|
| Triangles du toit | 3 370 | **14 032** |
| Séquence de 96 frames | 4,4 min | **6,7 min** |

**Ce qui reste hors de portée.** Le porche réel repose sur des colonnes de
brique, avec un toit en pente et un oculus. À vingt-sept points par mètre carré
vus du ciel, un pilier de quarante centimètres reçoit deux à quatre retours :
il est indétectable. Et un relevé aérien voit le **dessus** du porche, jamais
ce qui le soutient.

Cette forme est dans les photographies au sol, pas dans le LiDAR. C'est
précisément ce qu'une reconstruction feed-forward — MapAnything, VGGT —
apporterait, et c'est l'objet du lot `runpod/`.

---

## Le sol : lisser sans effacer

Un premier réglage rendait quinze pour cent des plages **très dentelées** — un
périmètre trois fois celui d'un disque de même aire. Le sol prenait un aspect
fractal que rien dans le terrain ne justifie : une pelouse a un bord franc, et
ses digitations d'un mètre sont du bruit de classification.

| | Avant | Après |
|---|---|---|
| Plages | 179 | **89** |
| Sommets de contour | 2 844 | **999** |
| Plages très dentelées | 15 % | **10 %** |
| Aire médiane | 62 m² | **214 m²** |

Trois réglages, tous mesurés : lissage porté de 1,2 à 2,6 cellules,
simplification de 0,8 à 1,6, et aire minimale de 12 à 40 m² — en deçà, une
plage est un îlot de quelques cellules qui multiplie les fragments sans
décrire de surface reconnaissable.

---

## Réserves

- **Le dépistage parcourt le système de fichiers, non le manifeste.** Les
  références retenues portent donc un statut de droits `INCONNU` — à vérifier
  avant toute diffusion.
- **18 volumes restent supposés** sur le pilote, faute de tuiles couvrant leur
  emprise.
- **Aucune source au sol n'atteste un toit** : sans nDSM, ces faces sont
  déclassées dans la carte de confiance.
