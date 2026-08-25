# Lot GPU WelcomINNS

Ce lot regroupe toute l'inférence GPU utile dans une session RunPod, sans
attente humaine pendant la facturation. Il valide les entrées avant de charger
les modèles, reprend les étapes déjà terminées, contrôle les sorties, produit
une archive avec empreintes SHA-256 puis arrête le pod.

## Machine

- RTX 4090 24 Go ou A40 48 Go ;
- image PyTorch/CUDA officielle ;
- volume persistant monté sous `/workspace`.

Les images sont des références destinées à la démonstration expérimentale.
Le résultat n'est pas éligible à la production et ne change aucun Gate.

## Transfert du paquet unique

Téléverser `runpod/welcominns-runpod-upload.tar.gz` et son fichier `.sha256`
dans `/workspace` sur le pod, puis extraire le paquet autonome :

```bash
cd /workspace
sha256sum -c welcominns-runpod-upload.tar.gz.sha256
mkdir -p /workspace/BuildingDroneVisit
cd /workspace/BuildingDroneVisit
tar -xzf /workspace/welcominns-runpod-upload.tar.gz
```

Le fichier `runpod/welcominns-runpod-upload.tar.gz.sha256` permet de vérifier
le transfert avec `sha256sum -c` avant l'extraction.

## Simulation sans GPU

```bash
cd /workspace/BuildingDroneVisit
ACK_REFERENCE_ONLY=1 DRY_RUN=1 STOP_WHEN_DONE=0 \
  bash runpod/run_welcominns_batch.sh
```

## Exécution canonique

```bash
cd /workspace/BuildingDroneVisit
ACK_REFERENCE_ONLY=1 bash runpod/run_welcominns_batch.sh
```

VGGT est le backend par défaut. Pour une comparaison plus coûteuse :

```bash
ACK_REFERENCE_ONLY=1 BACKENDS=vggt,mapanything \
  bash runpod/run_welcominns_batch.sh
```

La détection Grounding DINO/SAM 2 actuelle est déjà publiée. Pour la rejouer
explicitement avec les mêmes paramètres, le dépôt complet et son workspace
doivent en revanche être présents sous `/workspace/BuildingDroneVisit` :

```bash
ACK_REFERENCE_ONLY=1 RUN_SEMANTIC=1 \
  bash runpod/run_welcominns_batch.sh
```

Le chemin de l'archive finale est imprimé sous la forme `RESULTAT=...`. Si
`runpodctl` ou `RUNPOD_POD_ID` est absent, le script demande explicitement
l'arrêt manuel au lieu de prétendre l'avoir fait.
