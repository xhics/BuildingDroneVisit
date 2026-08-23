# Coût d'exécution, et où un GPU change quelque chose

Mesuré sur le poste de développement — Mac Intel, sans CUDA ni Metal — pour le
pilote WelcomINNS : 28 volumes, 69 massifs, 868 dalles de sol, 96 frames.

## Le bilan par étape

| Étape | Durée | Accéléré par un GPU ? |
|---|---|---|
| Chargement de la scène | < 1 s | sans objet |
| Hauteurs nDSM | 0,3 s | non |
| Hauteurs nuage LAZ | 8 s | non |
| Environnement + sol | 13 s | non |
| **Rendu, 96 frames** | **6 min** | oui, mais inutile |
| **Silhouettes CLIP, 6 vues** | **3 min 20** | **oui, nettement** |
| Dépistage d'identité, 200 images | ~5 min | **oui, nettement** |

## La réponse courte

**Non pour la géométrie, oui pour les modèles.**

Tout ce qui lit le LiDAR — hauteurs, végétation, classification du sol — reste
sous la quinzaine de secondes et n'a aucun intérêt à passer sur GPU : ce sont
des parcours de nuage et des agrégations, pas de l'algèbre dense.

Le rendu est le poste le plus lourd en absolu, mais c'est un faux problème : il
produit des cartes de profondeur qu'on ne régénère qu'au changement de
trajectoire. Six minutes pour une séquence complète est un coût acceptable, et
le porter sur GPU demanderait de réécrire le rasteriseur pour un gain sans
usage.

**Les modèles sont le vrai sujet.** CLIP coûte 33 secondes par image sur ce
CPU. Dépister deux cents images ou lire vingt silhouettes se compte en
minutes ; sur une carte récente, en secondes. Et l'inférence de forme —
MapAnything, VGGT — n'est **pas exécutable** ici : les paquets ciblent des
versions de torch plus récentes et supposent CUDA.

## Ce qui a été optimisé, et ce qui restait à prendre

Le rendu était deux fois plus lent que nécessaire, pour deux raisons trouvées
au profileur :

1. **`camera.basis()` recalculé par triangle** — quinze mille fois par image,
   pour un repère qui ne dépend que de la pose. Mémorisé.
2. **`np.cross` sur des vecteurs de trois composantes** — son mécanisme
   générique pesait un tiers du temps. Écrit à la main.

Résultat : **8,9 → 4,7 secondes par frame**, sans rien changer aux sorties.

Le reste — deux secondes par image dans `_rasterise` — est le coût irréductible
de quinze mille appels Python. Le réduire encore demanderait de vectoriser la
boucle sur les triangles, ce qui change la nature du code pour un gain qui ne
débloque rien.

## Quand louer un GPU

- **Inférence de forme** (MapAnything/VGGT) : indispensable, voir `runpod/`.
- **Dépistage d'un parc** : à partir de quelques centaines d'images par site,
  multiplié par le nombre de sites.
- **Un site isolé** : inutile. La chaîne complète tient en une quinzaine de
  minutes sur un portable.
