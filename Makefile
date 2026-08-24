.DEFAULT_GOAL := help
HOTEL ?= welcominns-boucherville
ADDRESS ?= 1195 rue Ampère, Boucherville, Québec J4B 7M6
HOTEL_PIPELINE ?= $(if $(wildcard .venv/bin/hotel-pipeline),.venv/bin/hotel-pipeline,hotel-pipeline)
PYTEST ?= $(if $(wildcard .venv/bin/pytest),.venv/bin/pytest,pytest)

.PHONY: help install test smoke provider-check init status geometry demo demo-status viewer phase1 docker-base clean

help: ## Affiche cette aide
	@grep -E '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) | awk -F':.*?## ' '{printf "  %-16s %s\n", $$1, $$2}'

install: ## Installe le paquet et ses dépendances de développement
	python -m pip install -e ".[dev]"

test: ## Lance la suite de tests (rapide, sans réseau)
	$(PYTEST) -q

smoke: ## Smoke test du socle — acceptation du Lot 0
	$(HOTEL_PIPELINE) smoke

provider-check: ## Contrôle la configuration des fournisseurs
	$(HOTEL_PIPELINE) provider-check

init: ## Crée l'espace de travail de $(HOTEL)
	$(HOTEL_PIPELINE) init $(HOTEL) --address "$(ADDRESS)"

status: ## État d'avancement de $(HOTEL)
	$(HOTEL_PIPELINE) status $(HOTEL)

geometry: ## Reconstruit la scène canonique et son audit topologique
	$(HOTEL_PIPELINE) conditioning scene-build $(HOTEL)

demo: ## Prépare et ouvre la démonstration 3D de $(HOTEL)
	$(HOTEL_PIPELINE) conditioning scene-build $(HOTEL)
	$(HOTEL_PIPELINE) demo prepare $(HOTEL) --launch

demo-status: ## Vérifie la démonstration sans ouvrir le navigateur
	$(HOTEL_PIPELINE) demo status $(HOTEL)

viewer: ## Ouvre le viewer 3D courant de $(HOTEL)
	$(HOTEL_PIPELINE) viewer open $(HOTEL)

phase1: ## Enchaîne la Phase 1 pour $(HOTEL)
	$(HOTEL_PIPELINE) run-phase1 $(HOTEL)

docker-base: ## Construit l'image socle (Lot 0)
	docker build -f docker/Dockerfile.base -t hotel-pipeline:base .

clean: ## Supprime les artefacts Python locaux
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache *.egg-info src/*.egg-info
