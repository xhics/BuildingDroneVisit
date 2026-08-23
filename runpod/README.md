# RunPod — reconstruction de forme, sans gaspiller

## La réponse courte sur le coût

**Oui, tu ne paies que ce que tu allumes.** Trois choses à retenir :

1. Un pod **arrêté** ne facture plus le GPU — seulement son disque, quelques
   centimes par Go et par mois.
2. Un pod **supprimé** ne facture plus rien du tout.
3. Le script fourni **arrête le pod tout seul** à la fin du calcul, succès ou
   échec. C'est le garde-fou principal : une session oubliée coûte bien plus
   cher que l'inférence.

Le piège classique n'est pas le prix horaire, c'est la machine laissée allumée
la nuit. `run_shape.sh` est écrit pour que ça n'arrive pas.

---

## Ce qu'il te faut

| Choix | Recommandation | Pourquoi |
|---|---|---|
| Type | **Pod GPU à la demande** | facturé à la seconde, arrêtable |
| GPU | **RTX 4090 ou A5000, 24 Go** | VGGT-1B tient largement ; inutile de payer un A100 |
| Image | `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` | torch et CUDA déjà là |
| Disque conteneur | 30 Go | les poids VGGT font ~5 Go |
| Volume persistant | **aucun**, au début | il facture même pod éteint |

Évite « Spot / Interruptible » pour un premier essai : c'est moins cher mais la
machine peut être coupée en cours de route.

---

## Le déroulé

### 1. Chez toi — préparer le paquet

```bash
python runpod/prepare_bundle.py welcominns-boucherville
```

Produit `runpod/bundle/` : **15 images, 2,6 Mo**. Le workspace complet pèse
3 Go et n'a rien à faire sur le GPU — le modèle ne consomme que les images.

### 2. Sur RunPod — créer le pod

Deploy → GPU à la demande → l'image ci-dessus → Deploy. Puis « Connect » →
« Web Terminal ».

### 3. Téléverser

Depuis le terminal du pod, le plus simple sans clé SSH :

```bash
mkdir -p /workspace/bundle
```

Puis glisse-dépose `bundle/images/`, `runpod/infer_shape.py` et
`runpod/run_shape.sh` via l'onglet fichiers de l'interface. Alternative si tu
as `runpodctl` installé localement :

```bash
runpodctl send runpod/bundle
```

### 4. Lancer — et oublier

```bash
cd /workspace
chmod +x run_shape.sh
BACKEND=vggt ./run_shape.sh 2>&1 | tee run.log
```

Le pod s'arrête seul à la fin. Compte **10 à 20 minutes** au total, dont
l'essentiel en téléchargement des poids.

### 5. Récupérer avant de supprimer

`/workspace/out/shape.ply` et `shape_run.json`. Redémarre le pod le temps de
les télécharger, puis **supprime-le** — un pod arrêté facture encore son
disque.

---

## Garde-fous

- `STOP_WHEN_DONE=1` par défaut : arrêt via `runpodctl stop pod`, armé par un
  `trap EXIT` **avant** toute installation, pour qu'une erreur de dépendance
  n'immobilise pas la machine.
- Pour observer sans arrêt automatique : `STOP_WHEN_DONE=0 ./run_shape.sh`.
- Mets une **limite de dépense** dans les réglages de facturation RunPod. C'est
  le seul filet qui ne dépend pas d'un script.

---

## Coût attendu

Un premier essai complet tient sous **1 $**, à titre indicatif : quelques
dizaines de minutes sur une carte à environ 0,40 $/h. Vérifie le tarif affiché,
il varie selon la région et la demande.

---

## Ce que tu récupères, et ce que ça ne dit pas

`shape.ply` — un nuage de points de la façade, inféré depuis 15 vues.

Deux réserves qui font partie du résultat :

- **Repère arbitraire.** Ni échelle métrique, ni orientation. Le recalage sur
  l'emprise géoréférencée se fait ensuite en local, avec `geometry_align`
  (Umeyama sim3) qui est déjà écrit.
- **Le nuage mêle le bâtiment et son environnement** — chaussée, arbres, ciel.
  Le script écarte déjà la moitié la moins fiable des points ; isoler la façade
  reste une étape à part.

Si le nuage est inexploitable, l'information reste utile : elle dira que quinze
vues Street View peu convergentes ne suffisent pas, et que la voie du
conditionnement 2,5D est la bonne.
