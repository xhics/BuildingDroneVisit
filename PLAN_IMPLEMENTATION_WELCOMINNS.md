# Complément d'implémentation — Phase 1 WelcomINNS

Document **complémentaire** à `PLAN_DIRECTEUR_WELCOMINNS.md` (V3.2, 24 sections). Le plan directeur fait foi sur tout ce qu'il traite : périmètre, architecture fonctionnelle, Gates G0–G5, Router, arborescence `work/<hotel>/`, CLI `hotel-pipeline`, Lots 0–8, définition de DONE.

Ce document ne couvre **que ce que le plan directeur ne traite pas** : contraintes matérielles constatées, exécution concrète sur VM GPU louée, sources géospatiales québécoises précises, qualité de reconstruction, et seuils économiques. En cas de recouvrement, le plan directeur l'emporte sans discussion.

---

## 1. Contraintes matérielles constatées

Le §5 du plan directeur pose le principe « poste local = développement et contrôle, VM GPU cloud = traitement ». Relevé du poste local :

```text
Intel i7-4770HQ (2014) · Iris Pro 5200 · macOS 12.6 · x86_64
Python 3.14 (trop récent pour torch et pycolmap) · python3.12 disponible
Ni COLMAP, ni GDAL · ffmpeg présent
48 Go d'espace disque libre
```

Conséquences :

- pas de CUDA : OpenCLIP, Grounded-SAM2, hloc/LightGlue, pycolmap et Brush ne sont pas exécutables localement, conformément au partage prévu par le §5 ;
- **Python 3.11 est épinglé** pour le projet, indépendamment des versions présentes localement ;
- l'architecture `x86_64` permet de construire l'image Docker nativement, sans émulation ;
- les 48 Go libres sont le point de vigilance réel du stockage — voir §3.3.

## 2. VM GPU — fournisseur et cycle de vie

Le §5 du plan directeur fixe la cible matérielle (RTX 4090 24 Go ou équivalent) et interdit explicitement Kubernetes, Prefect, le distribué et le serverless pour ce PoC. Il précise également que **« l'allocation et l'arrêt automatiques de la VM viendront après la preuve technique »**.

### 2.1 Fournisseur

**RunPod**, conforme à la cible du §5.

```text
GPU            : RTX 4090 24 Go
Container Disk : 100 Go
Secret         : RUNPOD_API_KEY, hors dépôt, conformément au §6
```

### 2.2 Pas d'orchestration automatique au Lot 0

Le §5 étant explicite, **aucun `launch.py` d'allocation/destruction automatique n'est écrit avant la preuve technique** (Lots 0 à 3). La VM est provisionnée et détruite manuellement, ou par un script shell minimal sans logique de reprise. Le Makefile du §18 orchestre les étapes *sur* la VM, il ne pilote pas la VM.

L'automatisation du cycle de vie devient pertinente au Lot 4 ou après, quand la collecte rejouable justifie des exécutions répétées.

Quand elle arrivera, deux points sont à anticiper dès maintenant :

- le SDK Python officiel `runpod` couvre le provisioning, le statut et la destruction, **mais pas le transfert de fichiers** — la synchronisation poste ↔ VM est une brique distincte (SSH/rsync avec attente de disponibilité SSH, qui n'est pas immédiate après le passage à `RUNNING`) ;
- la destruction doit être placée dans un `try/finally` : le poste de coût le plus fréquent sur RunPod est une VM oubliée allumée, pas le calcul.

### 2.3 Image Docker en deux couches

Une image contenant CUDA, COLMAP, torch, hloc, Grounded-SAM2, Brush et Blender pèse plusieurs gigaoctets et se construit lentement sur un i7 de 2014.

- **Lot 0** — image minimale (base RunPod PyTorch, Python 3.11, paquet du projet, Typer). Suffit à l'acceptation du Lot 0 : « la VM GPU peut exécuter un smoke test depuis un clone propre ».
- **Lot 2** — couche vision (hloc, LightGlue, pycolmap, Brush), construite **au-dessus de l'image PyTorch officielle RunPod** plutôt que localement de zéro.
- **Lot 3** — couche Blender headless (`bpy`).

## 3. Persistance et stockage

### 3.1 Le dépôt local est le système de référence

Le Container Disk meurt avec la VM. Or le pipeline s'étale sur plusieurs jours et plusieurs vies de VM, avec des verrous humains entre les deux (§4). Les manifestes, rapports et décisions vivent donc dans le dépôt local, rapatriés après chaque session de travail. La VM reste jetable.

### 3.2 `diskcache` — emplacement à trancher

Le §18 prévoit `diskcache` pour éviter les appels externes répétés. **Placé sur le Container Disk, ce cache ne survit à rien et n'a aucun effet.** Il doit vivre côté poste local, ou sur un volume persistant si la collecte migre un jour sur la VM. Décision retenue : **cache local**, cohérent avec le fait que la collecte et la qualification des droits sont largement manuelles.

### 3.3 Volumétrie

Avec 48 Go libres localement, l'accumulation photos + COLMAP + splat + rapports par run peut saturer le poste avant la fin de la Phase 1. À surveiller dès le Lot 2, pas à découvrir au Lot 7. Si le seuil est atteint, un Network Volume RunPod (0,07 $/Go) est la réponse — la cible de synchronisation doit donc rester un paramètre de configuration.

## 4. Verrous humains et VM facturée

Le plan directeur prévoit correction humaine et revue humaine (§4, §17, §20), mais ne dit pas ce que fait la machine pendant ce temps.

Règle : **aucune attente interactive sur une VM facturée.** Une étape qui requiert une décision humaine — confirmation de `BUILDING_MAIN`, arbitrage de droits, version d'entrée pré/post-rénovation — écrit un état bloquant explicite (ce qu'elle attend, sous quelle forme), libère la VM, et se reprend à la session suivante.

Deux verrous sont identifiés comme non automatisables pour ce pilote :

1. **Confirmation du bâtiment.** Le §3 signale que l'empreinte n'est pas nommée « hôtel » et qu'un parc-o-bus voisin prête à confusion. Une requête Overpass par rayon renverra plusieurs empreintes sans étiquette univoque. Le choix humain initial doit être **persisté dans le manifeste spatial**, pas redemandé à chaque exécution — sinon l'objectif multi-hôtels du §1 reste manuel en pratique.
2. **Version de l'entrée (pré/post-rénovation 2024).** Elle ne peut pas être déduite visuellement sans référence datée. OpenCLIP (G2) présélectionne les extérieurs ; un humain tranche la version sur ce sous-ensemble réduit. À traiter comme verrou, pas comme classification automatique.

## 5. Sources géospatiales québécoises

Le §9 du plan directeur cite OSM, Overture, DEM et données municipales ouvertes de façon générique. Précisions pour ce site :

| Besoin | Source retenue | Remarque |
|---|---|---|
| Géocodage adresse | **Adresses Québec** (officiel) en primaire, Nominatim en secours | Nominatim résout parfois sur le centroïde de rue plutôt que sur le bâtiment |
| Empreinte bâtiment | OSM Overpass, empreintes CMM | Ambiguïté attendue, voir §4 |
| Hauteur et toiture | **LiDAR Québec, MNS − MNT** | Alimente `ROOFLINE_MAIN`, mal couvert par la photo au sol |
| Parcelle | Cadastre du Québec, rôle d'évaluation | Sert `PROPERTY_WELCOMINNS` |
| Orthophoto | CMM ou Données Québec | Licence à vérifier avant usage en production |
| Terrain | MNT Québec | `TERRAIN_MAIN` |

Deux points d'exécution non triviaux :

- **le LiDAR québécois est distribué par feuillet**, pas par requête ponctuelle : une sous-étape de résolution adresse → feuillet est nécessaire avant toute soustraction MNS − MNT ;
- **l'instance publique Overpass est limitée en débit et souvent congestionnée.** Acceptable pour un hôtel, fragile pour l'automatisation multi-hôtels visée au §1 — prévoir un extrait OSM régional local (`pyrosm`/`osmium`) comme repli avant de passer à l'échelle.

Bibliothèques : `geopandas` et `shapely` pour les assertions géométriques (contiguïté `PARKING_HOTEL` ↔ `BUILDING_MAIN`, disjonction avec `PARK_AND_RIDE_DE_MORTAGNE`), `rasterio` pour MNS/MNT.

## 6. Qualité de reconstruction

L'Expérience 2 du §8 exige un environnement « sans ambiguïté majeure » et le §17 impose un contrôle de silhouette, toiture et façades. Quatre points affectant directement ce résultat ne sont traités nulle part dans le plan directeur.

### 6.1 Intrinsèques caméra hétérogènes

Les photos proviennent d'appareils et d'années différents (§3, §9). pycolmap peut estimer des intrinsèques par image quand l'EXIF fournit une focale exploitable. À expliciter : **intrinsèques par image dérivées de l'EXIF**, avec groupe séparé ou rejet pour les images sans métadonnées fiables — un partage d'intrinsèques imposé à un corpus hétérogène biaise la géométrie.

### 6.2 Harmonisation colorimétrique

Des photos prises à des saisons, heures et appareils différents produisent, une fois fusionnées dans un même splat, des variations locales de teinte visibles selon la vue dominante à cet endroit. Une normalisation inter-images avant entraînement (correspondance d'histogramme ou correction de balance des blancs par image sur un référentiel commun) est le levier de qualité perçue le moins coûteux disponible.

### 6.3 Hautes lumières saturées

Les JPEG publics auront des ciels et des reflets de vitrage cramés. Ces pixels corrompent silencieusement l'entraînement sur des surfaces que le §13 identifie précisément comme sensibles. À détecter et pondérer, pas à ignorer.

Corollaire au §13 : la distinction utile est **masquage pour le matching** contre **masquage pour le rendu**. Les vitrages portent de l'information architecturale à conserver pour la texture finale, mais leurs réflexions cassent l'hypothèse de constance photométrique de LightGlue. Le même pixel peut donc être exclu de l'extraction de features tout en étant conservé au rendu.

### 6.4 Floaters et nettoyage post-entraînement

Sur un corpus de 20 à 80 vues, un splat produit des gaussiennes flottantes dans les zones peu observées (arrière du bâtiment, angles rares). Deux lignes de défense : réglage des seuils d'élagage pendant l'entraînement — les valeurs par défaut visent des captures denses de plusieurs centaines de vues — puis passe de nettoyage post-hoc (filtrage d'outliers statistiques, seuils d'opacité et de taille) avant inspection.

### 6.5 Brush — position maintenue

Le §12 pose Brush comme option initiale, `gsplat` et Nerfstudio comme fallbacks documentés « si une limite de qualité ou de fonctionnalité est démontrée ». **Cette position est la bonne et n'est pas remise en cause ici** : une substitution avant preuve d'une limite réelle serait un choix non fondé. Le point de bascule à surveiller est le besoin d'une perte masquée personnalisée dans la boucle d'entraînement — modifiable trivialement dans `gsplat` (Python/PyTorch), nettement plus coûteux dans Brush (Rust/WGPU). Ce besoin relève de la Phase 2 et ne se pose pas encore.

## 7. Économie du pilote

Le §19 exige `cost_report.json` et `human_time_report.json`, mais aucun seuil ne leur donne de sens : un rapport de coût sans cible est un rapport sans conclusion.

Le §1 du plan directeur énonce déjà la thèse testée — produire l'environnement **à partir d'une adresse**, de sources autorisées et de clés API — donc l'automatisabilité fait partie du succès, pas seulement la fidélité.

Ligne de base à battre, à terme :

```text
opérateur de drone local, une visite   ≈ 300 $
photographies existantes de l'hôtel    déjà disponibles
```

Ordres de grandeur indicatifs, **non bloquants pour la Phase 1** :

```text
coût machine par hôtel, en régime automatisé   ≈ 50 $
temps humain par hôtel, en régime automatisé   ≈ 4 h
```

Aucun seuil formel n'est arrêté à ce stade : le pilote sert d'abord à produire la mesure, pas à la juger. Il suffit que `cost_report.json` et `human_time_report.json` soient réellement alimentés au fil des Lots, afin que la décision de passage en Phase 2 dispose de chiffres. Le pilote peut réussir techniquement (`ENVIRONMENT_3D_READY`) et se révéler économiquement non viable ; les deux verdicts se rendront séparément, le moment venu.

Note : au premier hôtel, ce pipeline coûtera davantage que la ligne de base. Son avantage revendiqué est l'automatisation à N hôtels, pas le coût unitaire — ce qui rend le §7 indissociable du §4 (les verrous humains persistés ou non déterminent s'il existe un N).

## 8. Points ouverts

- **Maturité de VoxCity pour les sources québécoises** (LiDAR provincial, cadastre, CMM) non vérifiée. Path C étant un fallback et non la route principale, ce n'est pas bloquant au Lot 0, mais doit être testé avant le Lot 5 plutôt que supposé.
- **Ordre G1/G2** — le §11 fixe l'ordre par coût croissant, déduplication avant tri extérieur/intérieur. Si le corpus est majoritairement intérieur, comme le §3 le laisse attendre, hacher perceptuellement des intérieurs qui seront écartés en G2 est du calcul perdu. À recalibrer empiriquement sur le corpus réel plutôt qu'à figer.
- **G4 face à G5** — sur un corpus de quelques dizaines d'images, le matching exhaustif de hloc est peu coûteux et donne la connectivité géométrique réelle. Le graphe de retrieval CLIP garde sa valeur comme signal préalable bon marché, mais son utilité marginale doit être mesurée sur ce pilote avant d'être reconduite pour les hôtels à gros corpus.
- **Automatisation du cycle de vie de la VM** — à ouvrir au Lot 4, pas avant (§2.2).
