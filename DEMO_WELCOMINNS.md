# Démonstration WelcomINNS — parcours opérateur

**État vérifié :** 24 août 2026
**Portée :** démonstration locale avant acceptation
**Commande canonique :** `make demo`

## Ce que la démonstration établit

La démonstration présente un prototype 3D inspectable construit depuis la
géométrie LiDAR et les contraintes mesurées du corpus actuel. Elle montre :

- le volume du bâtiment et ses toitures mesurées ;
- deux coques bâties logiquement fermées, étanches et supportées ;
- le terrain, 76 couronnes LiDAR non coniques et le mobilier détectés ;
- la couverture photographique des murs ;
- les faîtages retrouvés dans les images ;
- les positions que demanderait une captation ultérieure ;
- le diagnostic SfM réel, y compris son refus G5 actuel.

La démo n'exige pas maintenant une nouvelle captation ni la clearance finale
des droits. Ces deux chantiers sont différés jusqu'à acceptation. Cette portée
ne remplace pas le verdict Phase 1 : le paquet reste
`NEEDS_AUTHORIZED_CAPTURE`, et ne doit pas être présenté comme
`ENVIRONMENT_3D_READY` ou comme une reconstruction photoréaliste complète.

## Préparer et ouvrir

Depuis la racine du dépôt :

```bash
make demo
```

La commande republie le paquet canonique, le viewer autonome et le manifeste
de démonstration, puis ouvre le navigateur.

Équivalent détaillé :

```bash
.venv/bin/hotel-pipeline conditioning scene-build welcominns-boucherville
.venv/bin/hotel-pipeline demo prepare welcominns-boucherville --launch
```

Inspection sans ouvrir de fenêtre :

```bash
.venv/bin/hotel-pipeline demo status welcominns-boucherville
.venv/bin/hotel-pipeline viewer open welcominns-boucherville --no-launch
```

Le viewer est autonome et fonctionne hors ligne :

```text
work/welcominns-boucherville/11_conditioning/viewer.html
```

## Déroulé conseillé, cinq minutes

1. Présenter le problème : produire une scène inspectable à partir d'une
   adresse et de sources hétérogènes.
2. Faire tourner la scène et afficher successivement volumes, toits, sol et
   végétation.
3. Activer la couverture des murs avec `O` : vert signifie triangulable,
   rouge signifie non observé.
4. Activer les faîtages avec `F`, puis les prises proposées avec `N`.
5. Expliquer le diagnostic : la géométrie LiDAR est solide, mais les images
   publiques ne forment pas encore un réseau SfM globalement cohérent.
6. Conclure par la séparation des étapes : le prototype démontre le pipeline ;
   après acceptation, captation et droits permettent la version de production.

## Contrôles du viewer

| Contrôle | Action |
|---|---|
| glisser | orbiter autour du bâtiment |
| molette | zoomer |
| espace | activer la rotation automatique |
| `R` | afficher/masquer les toitures |
| `V` | afficher/masquer les volumes |
| `P` | afficher/masquer la végétation |
| `G` | afficher/masquer le sol |
| `O` | couverture photographique des murs |
| `N` | prises proposées |
| `F` | faîtages et arêtes de toiture |
| `W` | mode filaire |

## État mesuré présenté

Le manifeste de démonstration courant porte notamment :

- statut `DEMO_READY` ;
- 96 frames de conditionnement ;
- verdict `condition_partially` ;
- fraction fortement conditionnée : 0,781 ;
- fraction sans référence d'apparence : 0,219 ;
- score de fidélité géométrique : 0,9028 ;
- G1 : passé sur 349 assets, avec 35 417 paires robustes évaluées et les
  régressions recadrage/filigrane validées ;
- G5 : refusé, 25,5 % d'enregistrement géographiquement validé pour un seuil
  de 60 %.

Ces mesures sont relues depuis les artefacts. Le viewer publie un payload
stable et `viewer_manifest.json` enregistre son empreinte ainsi que celles de
ses sources.

La scène conditionnée ajoute les mesures de contrôle suivantes :

- 2 bâtiments sur 2 ont zéro arête ouverte, zéro arête non-manifold et un
  volume positif ;
- les 76 volumes végétaux portent chacun six anneaux issus des retours LiDAR,
  au lieu d'un cône reconstruit depuis un seul rayon ;
- 792 observations architecturales 2D sont enregistrées avec leur pose et
  leur provenance : 544 arêtes de toiture, 240 membres linéaires candidats,
  6 panneaux/ouvertures candidats et 2 enseignes ; 75 sont éligibles à la
  triangulation, mais **0** géométrie
  3D n'est produite automatiquement à ce stade ;
- le détecteur OpenCV hors ligne produit les candidats ci-dessus ; Grounding
  DINO et SAM 2 ne sont pas encore installés, donc leur classe sémantique reste
  `UNKNOWN` et n'est jamais promue en objet 3D.

## Fichiers de contrôle

```text
11_conditioning/viewer.html
11_conditioning/viewer_payload.json
11_conditioning/viewer_manifest.json
11_conditioning/conditioned_scene.json
11_conditioning/topology_audit.json
11_conditioning/geometry_benchmark.json
11_conditioning/architectural_observations.json
11_conditioning/demo_manifest.json
08_composite/scene_package_current.json
```

Le code refuse désormais de prendre un run synthétique ou un run COLMAP échoué
pour un G5 réussi. Une décision G5 mesurée est nécessaire ; sa simple présence
dans `07_reconstruction/runs/` ne suffit plus.
