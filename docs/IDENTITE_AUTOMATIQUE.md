# Tri automatique par identité du bâtiment

## Le problème résolu

Un lot nommé « façade » du workspace pilote contenait des **maisons
résidentielles voisines**, et un tri manuel les avait laissées passer. Aucune
règle sur les métadonnées ne pouvait les attraper : la distinction est
purement visuelle, et tous les champs du manifeste — coordonnées, azimut,
année — étaient corrects.

Le jugement est donc confié à des **modèles**, pas à des seuils écrits à la
main sur des champs.

```bash
# 1. amorçage sans intervention : lit les enseignes, propose des ancres
python -m hotel_pipeline.cli identity discover <hotel> --budget 60 --write

# 2. tri du corpus contre ces ancres
python -m hotel_pipeline.cli identity screen <hotel> --top 8
```

L'étape 1 est facultative : `identity anchor <hotel> <image> --evidence "..."`
déclare une ancre à la main quand aucune enseigne n'est lisible.

### L'amorçage automatique

Le nom d'un établissement, écrit sur son propre panneau, est la seule preuve
d'identité **intrinsèque à l'image**. Toute autre amorce — proximité au
centroïde, azimut — présumerait ce qu'elle prétend établir.

La commande enchaîne deux modèles :

1. **classement par vraisemblance d'enseigne** — un score CLIP trie le corpus
   avant toute lecture, car l'OCR coûte plusieurs secondes par image et en
   ouvrir 700 prendrait des heures ;
2. **lecture et appariement tolérant** sur les mieux classées.

Les noms viennent du profil d'établissement : `official_name` fournit les
termes attendus, `competitor_names` les termes qui disqualifient.

Vérifié sur le pilote, ancres effacées : sur **721 images**, le système
retrouve seul l'image que la confirmation manuelle avait retenue, plus une
seconde vue de façade que le tri manuel avait manquée.

---

## Les trois modèles enchaînés

| Étage | Modèle | Rôle |
|---|---|---|
| **Ressemblance** | OpenCLIP ViT-B-32 | distance aux ancres confirmées |
| **Attributs** | OpenCLIP, prompts naturels | cadrage, façade lisible, occultation, saison |
| **Enseigne** | EasyOCR + appariement tolérant | preuve d'identité **discriminante** |

Aucun entraînement, aucun jeu annoté : les attributs sont décrits en langage
naturel dans `ATTRIBUTE_PROMPTS` et se règlent en les relisant.

---

## Ce que la mise au point sur le corpus réel a appris

Quatre défauts sont apparus à l'exécution, chacun corrigé dans le code et
couvert par un test.

### 1. Un embedding mesure une scène, pas une identité

L'immeuble de bureaux du **1205** (Tetra Tech) obtenait **0,80** contre les
ancres de l'hôtel du **1195** — plus que la plupart des vraies vues. Même rue,
même neige, même lumière grise : le vecteur encode surtout la scène.

**Correction** : `facade_visible` devient un **facteur multiplicatif**, non un
terme additif, et l'OCR d'enseigne tranche en dernier ressort. Les deux
voisins — Tetra Tech et Isomed — sont désormais démentis automatiquement.

### 2. Un OCR de rue ne rend pas un texte propre

L'OCR lit `TETRA 1205 TECH` (numéro civique inséré) et `Tsomed` pour Isomed.
Une recherche sur limites de mots échouait sur les deux.

**Correction** : [sign_match.py](../src/hotel_pipeline/identity/sign_match.py)
apparie en sous-séquence et en approximation, sans jamais rapprocher deux noms
réellement distincts.

### 3. Une coupure « au plus grand écart » suppose une forme que le réel n'a pas

Premier essai : couper à l'écart le plus large entre deux scores triés. Sur un
corpus de rue, la distribution est **continue** — du pavillon voisin à la
façade de face — et la coupure tombait dans le bruit de la queue haute :
**zéro** image retenue sur 199.

**Correction** : critère d'Otsu, qui maximise la séparation inter-classes.

### 4. Un logo porte le nom aussi bien qu'une façade

La première exécution de `identity discover` proposait en tête un **wordmark**
du site officiel — un aplat de couleur portant « CLUB ÉLITE WELCOMINNS ».
L'OCR y lisait le bon nom, mais une ancre calibre tout le tri par sa
ressemblance visuelle : celle-ci aurait aligné le corpus sur un logo.

**Correction** : un attribut `is_photograph` écarte les visuels graphiques —
logos, plans, captures d'écran — avant qu'ils ne deviennent des ancres.

### 5. Otsu coupe toujours en deux, même une population unique

Sur le lot ne contenant **que** des maisons, Otsu produisait une coupure au
plancher et promouvait en « référence » le pavillon le moins dissemblable.

**Correction** : un garde-fou exige un **écart des moyennes** ≥ 0,12 avant de
croire à deux groupes. Le critère porte sur l'écart et non sur la variance
d'Otsu, car celle-ci s'effondre quand le groupe recherché est petit — dix
vraies vues parmi cent cinquante donnent 0,004, contre 0,021 pour deux groupes
comparables. Un seuil posé sur la variance aurait rejeté justement les corpus
où les bonnes images sont rares.

---

## Les garde-fous

- **Sans ancre, aucun verdict.** `undecidable`, jamais un score par défaut.
- **Ancres incohérentes entre elles → `undecidable`.** Si elles ne se
  ressemblent pas, l'une est fausse et tout le tri s'inverserait proprement.
- **`uncertain` est un résultat**, pas un échec : ces images vont en revue
  humaine, pas en production.
- **Résolution minimale** : une vignette de 200 px peut montrer le bon
  bâtiment sans pouvoir servir de référence — pénalité progressive.
- **Dédoublonnage par contenu**, pas par chemin.

---

## Résultat sur le pilote

Corpus de 200 images (163 après dédoublonnage) :

| Verdict | Nombre |
|---|---|
| `match` | 38 |
| `mismatch` | 56 |
| `uncertain` | 69 |

Seuil auto-calibré à **0,587**. Les deux bâtiments voisins sont démentis par
leur enseigne. Le classement est mené par `TGT3` et `TGT1` — deux vues d'été,
ciel bleu, enseigne nette — **qu'un tri manuel avait manquées**.

---

## Réserve

Tout repose sur les ancres. Une ancre fausse ne dégrade pas le tri : elle
l'**inverse** proprement, en validant le voisin et en rejetant la cible.

L'amorçage automatique lève l'obligation de confirmation initiale, sans lever
cette réserve : une ancre lue pèse moins qu'une ancre confirmée
(`ORIGIN_WEIGHT` : 0,85 contre 1,0), et `identity discover` ne publie rien
sans `--write`. Sur un site dont aucune enseigne n'est lisible, la commande
échoue franchement plutôt que de proposer une ancre faible — et la
confirmation manuelle reste alors nécessaire.
