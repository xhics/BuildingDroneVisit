# simple_mode

Mode simple, autonome : **une adresse → une image satellite annotée d'un
trajet caméra drone + un texte expliquant chaque figure de vol.**

Ce module ne dépend pas de `src/hotel_pipeline`. Il ne lit ni géométrie
mesurée du bâtiment, ni panoramas, ni LiDAR — les trajectoires qu'il dessine
sont des **gabarits stylisés** (orbite, spirale, travelling, passage rasant),
centrés sur l'adresse géocodée et mis à l'échelle de l'image satellite. C'est
un aperçu pédagogique/illustratif, pas un plan de vol mesuré.

Pour un trajet ancré sur des vantages réellement vérifiés (imagerie de rue,
LiDAR), voir `src/hotel_pipeline/camera_path_real.py` dans le pipeline
principal — un système différent, avec des garanties différentes.

## Usage

Réutilise le venv du projet (`requests`, `Pillow`, `python-dotenv` y sont
déjà installés) :

```bash
python -m simple_mode "123 rue Principale, Boucherville, QC"
```

Nécessite `GOOGLE_MAPS_API_KEY` dans `.env` à la racine du repo (Geocoding
API + Static Maps API doivent être activées sur cette clé côté Google Cloud).

Sorties, dans le répertoire courant :
- `out.png` — image satellite avec les trajectoires, une légende, une
  échelle et une flèche du nord
- `out.md` — texte expliquant chaque figure : rôle dans la vidéo, rayon,
  altitude, longueur du tracé, technique de pilotage requise

Options :

```bash
python -m simple_mode "adresse" --zoom 20 --size 800 --out output/mon_hotel
```

- `--zoom` : niveau de zoom Google Static Maps (défaut 19 — bâtiment de
  taille hôtel de centre-ville ; monter à 20-21 pour un bâtiment plus petit)
- `--size` : taille de l'image en pixels avant échelle (défaut 640)
- `--scale` : facteur d'échelle Google Static Maps, 1 ou 2 (défaut 2)
- `--out` : préfixe des fichiers de sortie

### Option IA (OpenAI)

Nécessite `OPENAI_API_KEY` dans `.env` et le package `openai`
(`pip install -e .[simple-ai]`, ou directement `pip install openai>=1.40`
dans le venv existant).

```bash
python -m simple_mode "adresse" --ai-plan --ai-illustration
```

- `--ai-plan` : au lieu du gabarit fixe de `maneuvers.py`, demande à OpenAI
  de **concevoir** les figures (combien, lesquelles, rayons, altitudes,
  rôle, technique de pilotage) pour cette adresse précise. Le modèle ne
  renvoie que des paramètres ; le tracé exact (chaque point de la
  trajectoire) est toujours calculé localement par les mêmes fonctions
  géométriques que le mode par défaut — `out.png` reste pixel-exact par
  rapport aux paramètres choisis, seul le choix des figures devient
  non-déterministe d'un appel à l'autre.
- `--ai-illustration` : génère en plus `out.illustration.png`, une image
  stylisée/cinématique du trajet via `gpt-image-1` (édition de la photo
  satellite). C'est un visuel marketing indicatif, **pas** une source de
  vérité géométrique — le tracé n'y est pas garanti pixel-exact. `out.png`
  (PIL) reste la référence précise.
- `--openai-model` / `--openai-image-model` (ou `OPENAI_MODEL` /
  `OPENAI_IMAGE_MODEL` dans `.env`) : changer les modèles utilisés (défaut :
  `gpt-4o-mini` et `gpt-image-1`).

Si `--ai-illustration` échoue (quota, modèle indisponible...), le script
continue : `out.png`/`out.md` sont déjà écrits avant cet appel, seul
l'avertissement est affiché.

## Vidéo continue (Street View + story-board + Sogni)

`python -m simple_mode.video` va plus loin que l'image annotée : il assemble
un **story-board continu** — une seule trajectoire chaînée (chaque figure
démarre exactement où la précédente finit, aucun saut), échantillonnée en
plusieurs **images de référence réelles** tout le long (Street View au
niveau rue, satellite en altitude), plus **un seul prompt** décrivant tout
le survol.

C'est délibéré : le modèle vidéo visé (Seedance 2.5, sur Sogni) accepte
jusqu'à 30 images de référence en un seul appel. Un appel par figure
produirait des clips séparés à recoller au montage — exactement les
coupures qu'une vidéo « continue non-stop » doit éviter. Ici, la continuité
se construit dans la trajectoire et dans l'appel, pas après coup.

```bash
python -m simple_mode.video "123 rue Principale, Boucherville, QC" --out video_out
```

Sorties dans `video_out/` :
- `satellite.png` — la photo satellite (référence pour les plans en altitude)
- `streetview_<pano_id>.jpg` — chaque panorama Street View retenu
- `storyboard.json` — `master_prompt_fr` (le prompt unique), `total_duration_s`,
  et la liste ordonnée des `keyframes` (position, altitude, référence)

Le collecteur Street View (`street_view.py`) s'inspire de
`src/hotel_pipeline/collectors/streetview.py` : interroger l'endpoint
`metadata` (gratuit) avant de télécharger, dédupliquer par identifiant de
panorama, diriger le cap vers le bâtiment. Simplifié pour une adresse seule :
échantillonnage sur des cercles concentriques plutôt que sur le réseau
routier OSM, puisque `simple_mode` n'a pas l'empreinte mesurée du bâtiment.

Le dernier point du trajet, la **« traversée du bâtiment »**, n'a par
construction aucune photo réelle possible (aucune image ne montre
l'intérieur d'un mur) — il réutilise la dernière référence réelle comme
simple ancrage de style, marqué `reference_kind: "generatif"` plutôt que
présenté comme une observation : seul le prompt porte l'effet.

### Un complexe plus grand qu'un bâtiment isolé

Le gabarit par défaut (`maneuvers.py`) est calibré pour un bâtiment isolé de
taille hôtel de centre-ville. Pour un site plus grand (hôtel avec jardins,
campus, centre commercial), `video.py` détecte l'étendue réelle via l'API
Google Places (`places.py`, viewport du lieu) et **met les figures à
l'échelle** en conséquence (`maneuvers.scale_maneuvers`) avant de les
chaîner — un complexe deux fois plus grand reçoit une orbite deux fois plus
ample, jusqu'à un plafond (x4 sur le rayon) au-delà duquel une détection
automatique n'est plus fiable.

```bash
python -m simple_mode.video "adresse" --extent-m 900   # force l'échelle plutôt que la détecter
python -m simple_mode.video "adresse" --max-keyframes 28  # plus d'images de référence le long du trajet
```

`GOOGLE_PLACES_API_KEY` doit être activée dans `.env` (déjà présente dans
`.env.example`) ; sans elle, ou si l'API ne trouve rien, le gabarit par
défaut s'applique sans mise à l'échelle — comportement dégradé, pas une
erreur bloquante.

### Vrai vol de drone : `--render-3d` (Cesium + tuiles 3D Google)

**C'est la voie qui applique réellement la trajectoire.** La voie Sogni
ci-dessous ne transmet au modèle que deux photos et un prompt : la
trajectoire calculée n'y est jamais utilisée, et le modèle invente tout
l'intermédiaire entre une vue satellite verticale et une photo de rue —
d'où des éléments qui apparaissent en cours de plan. Aucun prompt ne
corrige ça : on ne peut pas dériver une vue oblique de drone d'une image
satellite.

Ici, chaque waypoint devient une pose de caméra géoréférencée dans CesiumJS,
qui diffuse les tuiles 3D photoréalistes de Google (2500+ villes). Rendu
**déterministe**, géométrie **mesurée**, et aucun coût de génération IA.

```bash
pip install playwright && python -m playwright install chromium
winget install --id Gyan.FFmpeg

python -m simple_mode.video "adresse" --render-3d
python -m simple_mode.video "adresse" --render-3d \
  --render-3d-duration 20 --render-3d-fps 24 --render-3d-size 1920x1080
```

Sortie : `video_out/flight_3d.mp4`. La clé `GOOGLE_MAPS_API_KEY` doit avoir
accès à l'API Map Tiles — vérifier par un `HTTP 200` sur
`https://tile.googleapis.com/v1/3dtiles/root.json?key=...`.

#### Trois pièges rencontrés, et leur correctif

1. **Référentiel d'altitude.** Cesium attend une hauteur au-dessus de
   l'ellipsoïde WGS84 ; l'API Google Elevation renvoie une altitude
   au-dessus du géoïde. L'écart atteint ~33 m à Mountain View — assez pour
   transformer un vol rasant en vue d'horizon. Le sol est donc mesuré
   directement sur les tuiles rendues (`sampleHeightMostDetailed`), après
   quelques images de chauffe : échantillonner trop tôt donne une valeur
   instable (15 m d'écart observés d'une exécution à l'autre).
2. **Enveloppe de vol.** Ces tuiles sont reconstruites depuis des prises de
   vue aériennes : nettes de loin et d'en haut, elles s'effondrent de près
   (arbres informes, façades délavées). Mesuré : net à 45 m d'altitude /
   55 m de distance, inexploitable à 27 m / 15 m. D'où `MIN_ALTITUDE_M` et
   `MIN_RADIUS_M` dans `cesium_render.py`, qui bornent l'enveloppe sans
   déformer la figure. Ces bornes couvrent aussi les altitudes négatives que
   le chaînage peut produire.
3. **WebGL logiciel.** En headless sans GPU, une boucle de rendu continue
   sature le processus et fait expirer les captures. La scène est donc en
   rendu à la demande (`requestRenderMode`), avec un délai de capture relevé.

**Limite connue : pas d'intérieur.** Les tuiles 3D ne modélisent que
l'extérieur. Traverser une façade montrerait le maillage vu de l'envers,
c'est-à-dire du vide. L'effet de traversée du bâtiment doit rester une passe
séparée (IA ou transition stylisée), pas un waypoint qui entre dans le bâti.

### Génération réelle : `--generate-sogni` (chaîne de clips MiniMax H3)

Le schéma REST brut est confirmé (`sogni_client.py`), mais `referenceImageUrls`
exige des URL déjà publiques — nos images locales (satellite, Street View)
n'en sont pas. Ce module délègue donc l'upload à la CLI officielle
**`sogni-agent`** (`sogni_cli.py`), qui le gère en interne via le SDK
JavaScript de Sogni, à travers ses flags `--ref`/`--ref-end` (chemins
locaux acceptés, upload automatique).

**Le modèle n'est pas Seedance.** Seedance et HappyHorse ne sont pas
couverts par le forfait Sogni Unlimited de ce compte — chaque appel y
serait facturé séparément. On utilise à la place **MiniMax H3** (`--ref`
premier plan, `--ref-end` dernier plan), et on **découpe le trajet en
plusieurs clips qui se reprennent l'un l'autre** plutôt qu'un seul appel
multi-références : chaque clip démarre exactement sur la dernière image du
clip précédent (même patron que documenté par Sogni eux-mêmes dans
`references/loop-maker.md` de leur dépôt `sogni-creative-agent-skill` : un
clip par paire d'images adjacentes, puis recollage). `sogni-agent
--concat-videos` (nécessite `ffmpeg`) recolle ensuite tous les clips en une
seule vidéo continue.

Installation (une fois) :

```bash
npm install -g @sogni-ai/sogni-creative-agent-skill@latest
winget install --id Gyan.FFmpeg      # requis par --concat-videos
sogni-agent doctor                    # vérifie SOGNI_API_KEY, l'authentification, ffmpeg
```

#### Correctif obligatoire sur Windows : conflit `sharp` / `libvips`

**À réappliquer après chaque (ré)installation de `sogni-agent`.** Sans lui,
toute génération échoue sur
`ERR_DLOPEN_FAILED: The specified procedure could not be found`.

Cause : `sogni-agent` déclare `sharp: ^0.34.5` alors que son propre client
interne `@sogni-ai/sogni-intelligence-client` déclare `sharp: ^0.35.3`.
Ces plages semver étant incompatibles, npm installe légitimement **deux**
copies — donc deux `libvips-42.dll` de versions différentes (8.17.3 et
8.18.3). Sur Windows, deux DLL de même nom ne peuvent pas coexister dans un
processus : la seconde résout ses symboles contre la première, déjà
chargée, et échoue. (Chaque copie fonctionne isolément, ce qui rend le
symptôme trompeur ; c'est le chargement conjoint qui casse.) Bug de
packaging côté Sogni, pas de ce dépôt — à leur signaler.

Correctif : supprimer la copie imbriquée pour qu'il n'en reste qu'une.
Renommer plutôt que supprimer, pour rester réversible :

```bash
NESTED="$APPDATA/npm/node_modules/@sogni-ai/sogni-creative-agent-skill/node_modules/@sogni-ai/sogni-intelligence-client/node_modules"
mv "$NESTED/sharp" "$NESTED/sharp.bak"
mv "$NESTED/@img" "$NESTED/@img.bak"
```

Vérification (doit afficher deux fois la même version de libvips, sans erreur) :

```bash
cd "$APPDATA/npm/node_modules/@sogni-ai/sogni-creative-agent-skill"
node -e "const a=require('sharp'); const b=require(require.resolve('sharp',{paths:['./node_modules/@sogni-ai/sogni-intelligence-client']})); console.log(a.versions.vips, b.versions.vips)"
```

Puis, avec `SOGNI_API_KEY` dans `.env` :

```bash
python -m simple_mode.video "adresse" --generate-sogni
```

Par défaut : 6 images-clés ré-échantillonnées parmi celles du story-board
(donc 5 clips payants) à `minimax-h3-flf2v-turbo`, 5 s chacun. Réglable :

```bash
python -m simple_mode.video "adresse" --generate-sogni \
  --sogni-chain-keyframes 8 --sogni-clip-duration 6 --sogni-model minimax-h3-flf2v
```

Sorties dans `video_out/` : `video.mp4` (résultat final recollé) et
`video_clips/clip_00.mp4`, `clip_01.mp4`, ... (les clips individuels,
conservés pour inspection).

**Ceci déclenche une vraie génération à chaque clip, facturée sur le compte
Sogni — aucun mode bac-à-sable.** La chaîne s'arrête au premier clip en
échec plutôt que de continuer à en facturer d'autres sur un modèle ou une
référence déjà en défaut. `--probe-sogni` (ci-dessous) reste la façon sans
risque d'inspecter l'API brute, sans dépenser de crédit.

```bash
python -m simple_mode.video "adresse" --probe-sogni
```

## Structure

- `geocode.py` — adresse → (lat, lon) via Google Geocoding API
- `satellite.py` — (lat, lon) → image satellite + résolution m/pixel
- `street_view.py` — panoramas Street View autour de l'adresse (inspiré du
  pipeline principal, simplifié sans réseau routier ni empreinte mesurée)
- `geo_utils.py` — conversions locales mètres est/nord <-> lat/lon, distances
- `places.py` — étendue approximative du lieu via l'API Places (bâtiment
  isolé vs grand complexe), pour la mise à l'échelle des figures
- `maneuvers.py` — définit les figures de vol par défaut (géométrie en
  mètres est/nord/altitude autour du centre), plus `chain_maneuvers` (les
  recale bout à bout, sans saut) et `scale_maneuvers` (les met à l'échelle
  d'un site plus grand) — **c'est le fichier à modifier** pour changer les
  figures elles-mêmes
- `ai_plan.py` — variante `--ai-plan` : OpenAI choisit les paramètres des
  figures, converties par les mêmes fonctions géométriques que `maneuvers.py`
- `ai_image.py` — variante `--ai-illustration` : rendu artistique via
  `gpt-image-1`, séparé du tracé précis
- `render.py` — dessine les figures sur l'image (PIL, précis)
- `narrative.py` — génère le texte Markdown explicatif à partir des figures
  (gabarit ou IA — même fonction dans les deux cas)
- `storyboard.py` — échantillonne le trajet chaîné en plusieurs images de
  référence réelles + un prompt unique (`ContinuousStoryboard`), sérialisable
  en JSON — pensé pour un seul appel de génération vidéo, pas des clips à recoller
- `sogni_client.py` — appel REST brut à Sogni (schéma confirmé), `probe()`
  pour inspecter l'API sans risque — voir la limite connue ci-dessus
- `sogni_cli.py` — génération réelle via la CLI officielle `sogni-agent` :
  chaîne de clips MiniMax H3 (premier/dernier plan) puis recollage —
  **`--generate-sogni`, facturé**
- `cli.py` — CLI de l'image annotée (`python -m simple_mode`)
- `video.py` — CLI du story-board vidéo continu (`python -m simple_mode.video`)

## Limite connue (v1)

Les rayons/altitudes par défaut visent un bâtiment de type hôtel de
centre-ville. `python -m simple_mode` (l'image annotée) ne fait pas la mise
à l'échelle par étendue — seul `python -m simple_mode.video` la fait via
`places.py`. Sans détection du contour réel du bâtiment (ex: empreinte OSM
via Overpass), même mis à l'échelle par viewport, l'orbite peut déborder sur
la rue ou sembler trop serrée selon la forme réelle du site (un viewport
Places est une boîte, pas un contour) — ajuster `--zoom`, `--extent-m` et
les gabarits de `maneuvers.py` au cas par cas.
