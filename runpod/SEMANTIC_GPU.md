# RunPod GPU — détection sémantique WelcomINNS

Configuration vérifiée le 24 août 2026 sur une NVIDIA A40 :

- image : `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` ;
- Python 3.12.3, Torch 2.8.0+cu128, Torchvision 0.23.0+cu128 ;
- volume persistant monté sur `/workspace` ;
- projet : `/workspace/BuildingDroneVisit` ;
- environnement : `/workspace/.venvs/buildingdrone` ;
- caches : `/workspace/.cache`.

## Recréer l'environnement

Depuis un terminal du Pod :

```bash
cd /workspace/BuildingDroneVisit
bash runpod/setup_semantic_gpu.sh
```

Le script installe Grounding DINO via Transformers et SAM 2 depuis le dépôt
officiel Meta. Il ne remplace pas le PyTorch CUDA fourni par l'image RunPod.

## Reprendre une session

```bash
source /workspace/.venvs/buildingdrone/bin/activate
export HOTEL_PIPELINE_WORK=/workspace/BuildingDroneVisit/work
export HF_HOME=/workspace/.cache/huggingface
export TORCH_HOME=/workspace/.cache/torch
export SAM2_CHECKPOINT=/workspace/.cache/sam2/sam2.1_hiera_large.pt
cd /workspace/BuildingDroneVisit
```

## Détection de qualité validée

```bash
hotel-pipeline conditioning semantic-detect welcominns-boucherville \
  --limit 6 \
  --device cuda \
  --cache-only \
  --segmentation sam2 \
  --model-id IDEA-Research/grounding-dino-base \
  --threshold 0.35 \
  --text-threshold 0.30
```

Le run vérifié a produit 38 masques sur 6 vues.

Pour la passe de détail de façade utilisée par le viewer 1.6.0 :

```bash
hotel-pipeline conditioning semantic-detect welcominns-boucherville \
  --limit 6 \
  --device cuda \
  --cache-only \
  --segmentation sam2 \
  --model-id IDEA-Research/grounding-dino-base \
  --threshold 0.22 \
  --text-threshold 0.20 \
  --prompt "hotel building. hotel entrance. entrance door. entrance canopy. brick column. structural column. horizontal structural beam. window. hotel sign. road sign. lamp post. rooftop air conditioning unit. gutter. balcony. evergreen tree. deciduous tree."
```

Cette seconde passe a produit 159 masques : notamment 39 fenêtres, 10
colonnes, 3 poutres et 26 portes candidates en 2D. Ces détections ne deviennent
pas automatiquement de la géométrie.

## Association multi-vues

Cette étape relit les pistes du noyau COLMAP. Elle ne nécessite plus le GPU,
mais l'extra `sfm` et le modèle d'ancrage doivent être disponibles :

```bash
hotel-pipeline conditioning semantic-link welcominns-boucherville
```

Le run de détail du 24 août a trouvé 13 instances multi-vues — bâtiment,
portes, arbres, panneaux et fenêtres — soutenues par 303 pistes COLMAP mesurées.
Les poutres et colonnes sont restées des candidats 2D : aucune n'avait un
recoupement multi-vues suffisant. Une ressemblance visuelle seule n'est jamais
suffisante pour relier deux instances.

## Audit COLMAP/LiDAR

À exécuter sur la machine qui possède les tuiles LAZ :

```bash
hotel-pipeline conditioning register-colmap-lidar welcominns-boucherville
```

Le premier audit a estimé une translation candidate, puis l'a refusée sur les
points de contrôle. Le chemin spécialisé ajouté ensuite utilise seulement les
pistes sémantiques du bâtiment et les retours LiDAR de classe bâtiment. Il a
validé la translation sur holdout et contrôles négatifs, sans modifier la scène.

Les supports peuvent alors être projetés dans le repère local :

```bash
hotel-pipeline conditioning semantic-register welcominns-boucherville
hotel-pipeline conditioning semantic-surface welcominns-boucherville
```

Le run de détail a recalé 303 points uniques. L'audit a examiné 8 instances
planes candidates : une surface de panneau soutenue par 18 points a été
acceptée; les 7 fenêtres/portes/panneaux restants ont été refusés pour support
insuffisant, résidus trop élevés ou orientation incohérente. La surface acceptée
s'arrête au convexe mesuré : aucune épaisseur ou continuation occultée n'est
inventée.

Pour republier le viewer après ces étapes :

```bash
hotel-pipeline conditioning scene-build welcominns-boucherville
```

## Arrêt

Une fois les résultats rapatriés, arrêter le Pod dans la console RunPod. Le
GPU ne doit pas rester allumé; le volume `/workspace` conserve l'environnement,
les poids et le projet tant que le volume n'est pas supprimé.
