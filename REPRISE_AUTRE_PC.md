# Reprendre le projet sur une autre machine

## 1. Code et environnement Python

```bash
git clone https://github.com/xhics/BuildingDroneVisit.git
cd BuildingDroneVisit
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
python -m pip install -e .
```

La virtualenv locale n'est volontairement pas transférée : elle contient des
binaires liés au système et doit être reconstruite sur la nouvelle machine.

## 2. Secrets

```bash
cp .env.example .env
```

Transférer ensuite les valeurs de l'ancien `.env` par un canal privé
(gestionnaire de mots de passe, AirDrop ou support chiffré). Les clés API et les
clés SSH ne sont jamais publiées dans le dépôt ou dans une release publique.

## 3. Workspace WelcomINNS

Le snapshot portable est joint à la release GitHub `workspace-20260824`. Il
contient `work/`, sauf les virtualenvs, caches Python et dépôts Git imbriqués
reproductibles.

```bash
gh release download workspace-20260824 --pattern 'welcominns-workspace.tar.gz.part-*'
cat welcominns-workspace.tar.gz.part-* | tar -xzf -
```

Vérifier ensuite :

```bash
make demo-status
make viewer
```

Le viewer autonome reste directement accessible dans
`work/welcominns-boucherville/11_conditioning/viewer.html`.
