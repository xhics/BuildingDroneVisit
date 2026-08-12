.DEFAULT_GOAL := help
HOTEL ?= welcominns-boucherville
ADDRESS ?= 1195 rue Ampère, Boucherville, Québec J4B 7M6

.PHONY: help install test smoke provider-check init status phase1 docker-base clean

help: ## Affiche cette aide
	@grep -E '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) | awk -F':.*?## ' '{printf "  %-16s %s\n", $$1, $$2}'

install: ## Installe le paquet et ses dépendances de développement
	python -m pip install -e ".[dev]"

test: ## Lance la suite de tests (rapide, sans réseau)
	pytest -q

smoke: ## Smoke test du socle — acceptation du Lot 0
	hotel-pipeline smoke

provider-check: ## Contrôle la configuration des fournisseurs
	hotel-pipeline provider-check

init: ## Crée l'espace de travail de $(HOTEL)
	hotel-pipeline init $(HOTEL) --address "$(ADDRESS)"

status: ## État d'avancement de $(HOTEL)
	hotel-pipeline status $(HOTEL)

phase1: ## Enchaîne la Phase 1 pour $(HOTEL)
	hotel-pipeline run-phase1 $(HOTEL)

docker-base: ## Construit l'image socle (Lot 0)
	docker build -f docker/Dockerfile.base -t hotel-pipeline:base .

clean: ## Supprime les artefacts Python locaux
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache *.egg-info src/*.egg-info
