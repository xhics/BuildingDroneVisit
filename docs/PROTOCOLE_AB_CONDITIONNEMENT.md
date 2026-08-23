# Protocole A/B — l'étape 3D vaut-elle son coût ?

**Question tranchée par ce protocole**

Les deux options partagent entrées et sortie. Seule diffère une étape :

- **A** — carte/chemin caméra + images internet → génération vidéo IA
- **B** — carte/chemin caméra + images internet → **génération 3D** → génération vidéo IA

B ne se justifie que si l'étape 3D achète quelque chose que A ne peut pas
obtenir. Ce protocole mesure quoi, plutôt que d'en débattre.

---

## 1. Ce que le harnais fournit

`hotel_pipeline.conditioning` produit, le long d'une orbite, les cartes qu'un
générateur conditionné consomme :

| Canal | Rôle |
|---|---|
| `depth/` | profondeur par frame — impose la parallaxe |
| `normal/` | normales RGB — impose l'orientation des surfaces |
| `silhouette/` | cible en blanc, obstacles en gris — impose l'occultation |
| `confidence/` | **vert attesté, rouge supposé, noir libre** |

Plus `conditioning_report.json`, qui porte le verdict et, frame par frame, un
`guidance_mode`.

### La commande

```bash
python -m hotel_pipeline.cli conditioning render welcominns-boucherville \
    --frames 90 --width 768 --height 432
```

Sortie dans `work/<hotel>/11_conditioning/orbit/`.

### Le point qui fait tout le protocole

La 3D **n'a pas besoin d'être belle**. Elle sert de squelette, pas de rendu.
Des prismes bien placés avec la bonne trajectoire battent un maillage détaillé
mal aligné. C'est ce qui abaisse la barre de réussite de l'étape 3D — et ce qui
rend le test réalisable aujourd'hui, avec la géométrie déjà disponible.

---

## 2. Le repli, décidé par frame

Chaque frame porte son `guidance_mode` :

| Mode | Signification | Conduite |
|---|---|---|
| `geometry_strong` | cible dominante, géométrie attestée | conditionner fort |
| `geometry_weak` | > 60 % de pixels en hauteur supposée | conditionner faible, ou silhouette seule |
| `prefer_ungrounded` | cible < 2 % de l'image | **ne pas conditionner** |

Et un verdict d'ensemble : `condition_strongly`, `condition_partially`,
`prefer_ungrounded`.

> Une géométrie fausse contraint **pire** qu'aucune géométrie. A échoue
> joliment, B mal calibré échoue rigidement — il force le générateur à peindre
> proprement une erreur. C'est le seul risque sérieux de B, et le
> `guidance_mode` existe pour l'éviter.

---

## 3. Exécution

### Étape 1 — la géométrie (faite)

```bash
python -m hotel_pipeline.cli conditioning render welcominns-boucherville --frames 90
```

État du pilote WelcomINNS : **90 frames, verdict `condition_strongly`**, 100 %
des frames fortement contraignables. Bâtiment cible résolu (`way/54581348`),
27 obstacles, EPSG:2950.

### Étape 2 — les images de référence

Prendre les mêmes dans les deux branches, sinon le test ne mesure rien. Le
dépôt les qualifie déjà (`assets`, droits, cohortes temporelles) :

```bash
python -m hotel_pipeline.cli assets review list welcominns-boucherville
```

Cohorte : `current_confirmed`, pour ne pas mélanger avant/après rénovation.

### Étape 3 — les deux générations

**Hors de ce dépôt** : aucun modèle vidéo n'est accessible depuis le pipeline.
À lancer où tu veux (Runway, Kling, Wan, ComfyUI local).

- **Branche A** — prompt + images de référence, sans conditionnement.
- **Branche B** — mêmes prompt et références, **plus** `depth/` (ou `normal/`)
  en ControlNet vidéo, frame à frame. Là où `guidance_mode` vaut
  `prefer_ungrounded`, retirer le conditionnement.

Même seed, même durée, même résolution.

### Étape 4 — la comparaison

| Critère | Mesure | Ce qu'il tranche |
|---|---|---|
| **Cohérence du bâtiment** | frame 1 vs frame N : étages, fenêtres, couleur | la dérive temporelle |
| **Parallaxe** | l'avant-plan glisse-t-il par rapport à la façade ? | l'effet « carte postale » |
| **Reproductibilité** | refaire un 2ᵉ plan, autre azimut — même bâtiment ? | **le critère décisif** |
| **Fidélité** | comparer aux photos de référence | l'écart au réel |
| **Coût humain** | temps par plan | le passage à l'échelle |

---

## 4. Lecture des résultats

**Si B ne gagne que sur la reproductibilité** — c'est déjà le résultat
attendu, et il suffit dès que le volume dépasse quelques bâtiments. Le coût
humain de A est quasi constant par bâtiment ; celui de B s'amortit.

**Si B ne gagne sur rien de visible** — l'objection A l'emporte, et le
périmètre 3D doit fondre. Garder alors la couche de qualification des sources
et des droits, qui reste utile aux deux branches.

**Si B gagne nettement** — la question devient *quelle qualité* de 3D viser.
Probablement bien moins que ce que le plan directeur exige aujourd'hui :
`ENVIRONMENT_3D_READY` vise un environnement **inspectable**, plus exigeant que
ce dont le conditionnement vidéo a besoin.

---

## 5. Réserves, qui font partie du résultat

- **Aucune hauteur n'est mesurée** au pilote : les 28 volumes sortent en
  `height_assumed`. L'emprise au sol est attestée, la silhouette verticale est
  une hypothèse (12 m cible, 8 m obstacles). Le LiDAR de `06_geo/lidar_raw`
  lèverait cette réserve.
- **Le toit n'est attesté par aucune source au sol** : ses pixels sont
  déclassés à 35 % du crédit. C'est la faiblesse structurelle du plan aérien
  — et l'endroit où A garde l'avantage, parce qu'elle invente le toit sans que
  personne ne vérifie.
- **La qualification du pilote est `provisional`**, calibrée sur un seul site,
  `visual_proxy_not_survey`. Le harnais ne produit pas une mesure.
