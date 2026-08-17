"""Politique de pipeline — seuils et tolérances, versionnés (Lot 1B, généricité).

Ces valeurs ne décrivent pas un établissement : elles décrivent **notre
méthode**. Les placer dans le profil de chaque hôtel autoriserait un
recalibrage par site, et une calibration valable pour un seul corpus ne vaut
rien — c'est précisément ce qui a produit un seuil mesuré sur 36 images de
Boucherville puis appliqué comme s'il était universel.

Elles sont donc regroupées ici, versionnées, et destinées à être validées sur
plusieurs établissements avant d'être modifiées.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

#: Valeur de départ de toute calibration : aucune. Un nouveau projet ne peut
#: pas hériter du réglage d'un établissement qu'il ne connaît pas — c'est
#: exactement ce que faisait `welcominns-2026-08-36-images` posé en défaut.
UNCALIBRATED = "non-calibré — valeurs initiales, aucun site"

#: Formulations reconnues comme « aucune calibration ». Plusieurs, parce que
#: des politiques déjà écrites en portent d'autres : les convertir modifierait
#: des fichiers dont l'empreinte est citée par des rapports publiés. Ce n'est
#: pas une reconnaissance de texte libre — c'est une liste close, et tout ce
#: qui n'y figure pas est traité comme le nom d'une campagne réelle.
UNCALIBRATED_MARKERS: frozenset[str] = frozenset(
    {
        UNCALIBRATED,
        "non-calibré — valeurs initiales, un seul site",
    }
)


class Calibrated:
    """Règle commune aux sections portant une calibration.

    Le couple identifiant/nombre de sites ne peut pas mentir dans un sens :
    nommer une campagne sans déclarer aucun site en ferait une autorité sortie
    de nulle part. L'inverse est permis — des sites mesurés sans nom de
    campagne restent une information honnête.

    **Ce socle n'est pas un modèle**, et c'est délibéré. En hériter placerait
    `calibration_id` et `calibrated_on_sites` en tête du dump, y compris s'ils
    sont redéclarés : pydantic conserve la position du parent. Or
    `policy_digest` est l'empreinte du dump JSON, donc sensible à l'ordre — la
    politique du pilote, sans qu'aucune valeur ne bouge, serait passée de
    `a4564b71ddeec56e` à `1d6a92e87c91a80f`, et les rapports déjà publiés qui
    la citent seraient devenus incohérents. Les sections déclarent donc leurs
    deux champs à leur place ; ce mixin ne porte que la règle et sa lecture.

    Il n'annote pas non plus `calibration_id` ni `calibrated_on_sites` : une
    annotation nue sur une classe de base est collectée par pydantic comme un
    champ, ce qui reproduirait exactement le déplacement qu'on cherche à
    éviter. Les deux attributs sont donc supposés présents dans la section qui
    mélange ce mixin, et leur absence se voit à la première validation.
    """

    @property
    def names_a_campaign(self) -> bool:
        return self.calibration_id not in UNCALIBRATED_MARKERS

    @property
    def is_calibrated(self) -> bool:
        return self.calibrated_on_sites > 0 and self.names_a_campaign

    @model_validator(mode="after")
    def _a_named_calibration_names_its_sites(self) -> "Calibrated":
        if self.names_a_campaign and self.calibrated_on_sites == 0:
            raise ValueError(
                f"calibration {self.calibration_id!r} déclarée sur zéro site : "
                "un identifiant de campagne sans site mesuré n'a aucune "
                f"autorité. Laissez {UNCALIBRATED!r}, ou déclarez les sites."
            )
        return self


class ModelPolicy(BaseModel, Calibrated):
    """Classifieur et ses seuils."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    model_name: str = "ViT-B-32"
    pretrained: str = "laion2b_s34b_b79k"

    subject_accept: float = Field(default=0.50, ge=0.0, le=1.0)
    subject_reject: float = Field(default=0.20, ge=0.0, le=1.0)

    #: En deçà, une décision automatique n'est pas acceptée sans revue.
    review_confidence_floor: float = Field(default=0.60, ge=0.0, le=1.0)

    #: Sur quoi ces seuils ont été mesurés. Sans cette trace, un seuil est un
    #: nombre sans autorité — et le défaut ne nomme aucun établissement.
    calibration_id: str = UNCALIBRATED
    calibrated_on_sites: int = Field(default=0, ge=0)


class GeometryPolicy(BaseModel):
    """Visibilité et relations spatiales."""

    model_config = ConfigDict(extra="forbid")

    half_fov_deg: float = Field(default=45.0, gt=0, le=180)
    max_distance_m: float = Field(default=200.0, gt=0)

    #: Contiguïté franche puis association plausible entre bâtiment et parking.
    adjacency_strong_m: float = Field(default=8.0, gt=0)
    adjacency_max_m: float = Field(default=30.0, gt=0)

    #: Demi-ouverture admise autour de l'azimut d'un secteur, pour juger d'où
    #: un observateur regarde. **Valeur initiale provisoire**, et son usage est
    #: borné : elle sert à écarter un observateur clairement situé du mauvais
    #: côté, jamais à déclarer une façade couverte. Une vue oblique contribue
    #: légitimement à deux secteurs voisins — c'est pourquoi la demi-ouverture
    #: dépasse le huitième de tour, et pourquoi deux secteurs se recouvrent.
    sector_observer_half_width_deg: float = Field(default=67.5, gt=0, le=180)

    #: Écart de normale en deçà duquel deux segments d'empreinte appartiennent
    #: à la **même** façade. Un mur réel est découpé en plusieurs segments par
    #: les décrochements du relevé ; les traiter séparément ferait dépendre
    #: l'orientation du segment qu'un rayon touche en premier, non de la façade
    #: qu'il documente.
    facade_segment_merge_deg: float = Field(default=8.0, gt=0, le=45)

    #: Distance en deçà de laquelle deux caméras ne produisent pas deux
    #: observations indépendantes. Mesurée en mètres projetés, jamais en
    #: cellules de grille : deux positions distantes de six mètres tombant de
    #: part et d'autre d'une frontière comptaient pour deux points de vue.
    viewpoint_separation_m: float = Field(default=10.0, gt=0)


class VisibilityPolicy(BaseModel):
    """Réglages numériques du moteur de visibilité.

    Ce sont des paramètres de calcul, non des seuils d'acceptation : rien ici
    ne décide qu'une vue est bonne. Les fixer dans la politique les rend
    inspectables et reproductibles — un pas angulaire choisi dans le code
    changerait les fractions sans laisser de trace.
    """

    model_config = ConfigDict(extra="forbid")

    #: Pas angulaire maximal d'échantillonnage de la silhouette. Plus fin
    #: qu'utile coûte du temps ; plus grossier manque un obstacle étroit.
    max_angular_step_deg: float = Field(default=0.25, gt=0, le=10)

    #: Nombre minimal de cellules, quel que soit l'intervalle : une cible
    #: lointaine n'occupe qu'un demi-degré, et deux rayons n'en diraient rien.
    min_angular_cells: int = Field(default=48, ge=8)

    #: Pas d'échantillonnage des corridors, en mètres de projection.
    corridor_sample_step_m: float = Field(default=10.0, gt=0)

    #: Tolérance d'intersection, en mètres : en deçà, deux formes se touchent
    #: sans se couper.
    intersection_tolerance_m: float = Field(default=0.05, gt=0)

    #: Décimales des mesures publiées, fixées pour que deux exécutions
    #: identiques rendent des rapports identiques.
    output_precision: int = Field(default=4, ge=1, le=12)

    #: Méthode d'échantillonnage, inscrite au rapport.
    sampling_method: str = "uniform_angular_cells"

    #: Modèle de projection du cadrage. La règle des trois tangentes n'est
    #: valable que pour une caméra perspective : un panorama équirectangulaire
    #: demanderait un autre calcul, et le nommer permet de le refuser.
    projection_model: str = "pinhole_tangent"


class DedupPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    phash_hamming_threshold: int = Field(default=6, ge=0, le=64)
    position_tolerance_m: float = Field(default=10.0, gt=0)
    bearing_tolerance_deg: float = Field(default=25.0, gt=0, le=180)
    max_overlap_per_cluster: int = Field(default=2, ge=0)
    robust_crop_hash_enabled: bool = True
    #: Au moins cinq régions doivent concorder. Un ou deux segments fusionnent
    #: abusivement les routes, arbres et ciels répétés du corpus pilote.
    robust_region_cutoff: int = Field(default=5, ge=1, le=64)
    #: Le hash robuste ne compare que des republications plausibles : familles
    #: différentes, ou médias non positionnés. Les séquences routières restent
    #: des points de vue voisins, jamais une seule photographie.
    robust_plausible_pairs_only: bool = True


class CollectionPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    radius_m: int = Field(default=500, ge=25, le=2000)
    road_radius_m: int = Field(default=350, ge=25, le=2000)

    #: Résolution d'un **aperçu**. Une vue dont on ignore ce qu'elle montre se
    #: vérifie en miniature : la télécharger en pleine résolution dépenserait
    #: le volume avant de savoir s'il valait la peine.
    #: En deçà de cet écart de cap, deux cadrages d'un même panorama montrent
    #: la même chose. Deux requêtes pour une image : le pilote en produisait
    #: trois par panorama dont deux à 1,5° l'une de l'autre.
    framing_merge_bearing_deg: float = Field(default=15.0, gt=0, le=180)

    #: Version du contrat de **téléchargement** : ce que « télécharger »
    #: garantit — plafonds simultanés, refus avant lecture, format décodé,
    #: dimensions selon le fournisseur, publication atomique. La changer périme
    #: un plan mesuré sous l'ancien contrat : son volume avait été consenti
    #: sous d'autres garanties.
    download_contract_version: int = Field(default=1, ge=1)

    preview_resolution: str = Field(default="256", min_length=1)

    #: Résolution d'une acquisition complète, une fois le niveau établi.
    full_resolution: str = Field(default="2048", min_length=1)
    sample_spacing_m: float = Field(default=15.0, gt=0)
    snap_radius_m: int = Field(default=25, gt=0)
    max_panorama_distance_m: float = Field(default=220.0, gt=0)

    #: Champ de vision demandé aux vues Street View, et sa variante élargie
    #: pour les transitions route–entrée–stationnement.
    image_fov_deg: int = Field(default=80, gt=0, le=120)
    wide_fov_deg: int = Field(default=110, gt=0, le=120)

    #: Candidats enrichis d'un `sequence_id` **par besoin**. L'API ne le rend
    #: pas dans la recherche par zone : l'obtenir coûte un appel par image, et
    #: enrichir des centaines de vues avant toute sélection dépenserait le
    #: budget sur ce qu'on écartera.
    sequence_enrichment_per_demand: int = Field(default=12, ge=0)

    #: Membres d'une séquence explorés en expansion. Suivre une séquence
    #: entière la mènerait hors de la zone utile — un véhicule roule.
    sequence_expansion_max_members: int = Field(default=20, ge=0)

    #: Distance au-delà de laquelle un membre de séquence n'est plus exploré,
    #: quelle que soit la continuité qu'il promettrait.
    sequence_expansion_max_distance_m: float = Field(default=250.0, gt=0)


class CoveragePolicy(BaseModel):
    """Ce qu'une obligation exige, par intention.

    Ces seuils **ne sont pas** dans la commande qui construit les besoins :
    codés là, ils auraient été deux fois — une fois dans le générateur, une
    fois dans le manifeste produit — et le second aurait fini par mentir sur
    le premier. Ils vivent ici, versionnés, et une modification périme les
    besoins générés sans toucher aux artefacts LiDAR.

    Provisoires, comme tous les seuils de ce dépôt : mesurés sur aucun site,
    ils décrivent une intention de méthode, pas une validation.
    """

    model_config = ConfigDict(extra="forbid")

    #: Une façade ne se reconstruit pas depuis une seule position : deux vues
    #: indépendantes sont le minimum pour qu'un SfM y trouve de la parallaxe.
    building_viewpoints_required: int = Field(default=2, ge=1)

    #: Une vue de contexte documente un accès ou une transition ; une seule
    #: suffit à établir qu'on l'a regardé.
    context_viewpoints_required: int = Field(default=1, ge=1)

    #: Recouvrement attendu entre vues voisines d'un même besoin de bâtiment.
    #: Zéro signifierait « vues indépendantes acceptées », ce qui priverait le
    #: SfM de tout chaînage.
    building_continuity_required: float = Field(default=0.3, ge=0.0, le=1.0)
    context_continuity_required: float = Field(default=0.0, ge=0.0, le=1.0)

    #: Taille projetée minimale de la cible. Le critère décisif n'est pas la
    #: distance : un téléobjectif lointain vaut mieux qu'un grand-angle proche.
    building_min_projected_width: float = Field(default=0.15, ge=0.0, le=1.0)
    context_min_projected_width: float = Field(default=0.0, ge=0.0, le=1.0)

    #: Part de silhouette utile non masquée, en deçà de laquelle la vue ne
    #: répond pas au besoin.
    building_min_visible_fraction: float = Field(default=0.5, ge=0.0, le=1.0)
    context_min_visible_fraction: float = Field(default=0.0, ge=0.0, le=1.0)


class AdaptiveSearchPolicy(BaseModel, Calibrated):
    """Comment **choisir** des candidats. Distinct de ce qu'il faut obtenir.

    `CoveragePolicy` dit combien de vues un besoin exige ; celle-ci dit
    lesquelles préférer parmi les candidats possibles. Les mêler ferait
    dépendre un objectif de couverture d'une heuristique de recherche.

    La préférence de parallaxe n'est **pas monotone**, et c'est tout l'objet
    de cette section. Un écart angulaire plus grand donne plus de profondeur
    jusqu'à un point, puis les deux vues cessent de partager assez de surface
    pour qu'un appariement fonctionne. Préférer systématiquement le plus grand
    angle — ce que faisait un simple `delta / 90` — sélectionnait de la
    diversité en croyant sélectionner de la parallaxe.

    Aucune de ces valeurs n'est mesurée : elles décrivent une intention de
    méthode, sur zéro site.
    """

    model_config = ConfigDict(extra="forbid")

    #: Au-delà, un candidat n'est plus recommandé **automatiquement**. Ce
    #: n'est pas une preuve d'inutilité : sans les intrinsèques de la caméra,
    #: la distance seule ne dit pas qu'une cible serait trop petite. Un
    #: téléobjectif lointain peut valoir mieux qu'un grand-angle proche.
    #:
    #: Distinct du rayon d'interrogation de l'index : on interroge plus large
    #: qu'on ne recommande, et les rendre égaux masquerait cette différence
    #: sans rendre le seuil moins provisoire. Valeur **non calibrée**.
    automatic_candidate_max_distance_m: float = Field(default=250.0, gt=0)

    #: **Portée dure** : au-delà, un candidat n'est jamais recommandé, quel que
    #: soit le manque. Le repli faute de mieux pouvait proposer une vue à
    #: 1,7 km — bornée à l'aperçu, mais recommandée quand même. Une contrainte
    #: qu'un repli contourne n'est pas une contrainte.
    #:
    #: Distincte du seuil de recommandation automatique : entre les deux, un
    #: candidat reste examinable ; au-delà, il ne l'est plus.
    hard_max_distance_m: float = Field(default=600.0, gt=0)

    #: Écart toléré entre le cap mesuré et la direction de la cible. Répond à
    #: « l'objectif est-il tourné vers elle ? », question distincte de « de
    #: quel côté du bâtiment se tient la caméra ? ». Un cap **absent** ne vaut
    #: pas un cap qui vise ailleurs : il reste `None`.
    heading_tolerance_deg: float = Field(default=60.0, gt=0, le=180)

    #: En deçà, deux vues sont presque colinéaires : peu de profondeur.
    parallax_preferred_min_deg: float = Field(default=15.0, ge=0, le=180)

    #: Au-delà, le recouvrement se dégrade : les surfaces vues divergent.
    parallax_preferred_max_deg: float = Field(default=45.0, gt=0, le=180)

    #: Combien pénaliser un degré au-delà de la plage préférée, rapporté à
    #: l'utilité maximale. Zéro rendrait la préférence monotone à nouveau.
    parallax_excess_penalty: float = Field(default=0.015, ge=0.0)

    #: Base minimale entre deux positions : sous ce seuil, l'écart angulaire
    #: mesuré ne correspond à aucune parallaxe exploitable.
    baseline_min_m: float = Field(default=3.0, ge=0.0)

    status: str = "provisional"
    calibration_id: str = UNCALIBRATED
    calibrated_on_sites: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _the_preferred_range_is_a_range(self) -> "AdaptiveSearchPolicy":
        if self.parallax_preferred_max_deg <= self.parallax_preferred_min_deg:
            raise ValueError(
                "plage de parallaxe vide : le maximum préféré doit dépasser le "
                "minimum, sans quoi aucun angle ne serait jamais bon"
            )
        return self


class TerrainPolicy(BaseModel, Calibrated):
    """Seuils de la validation par pseudo-empreinte.

    Ils portent sur une **couverture spatiale**, non sur un nombre de points :
    trente cellules masquées représenteraient moins de un pour cent d'une
    empreinte de sept mille cellules, et l'essai n'aurait rien reproduit.
    """

    model_config = ConfigDict(extra="forbid")

    cell_m: float = Field(default=0.5, gt=0)
    ring_m: float = Field(default=20.0, gt=0)
    search_radius_m: float = Field(default=150.0, gt=0)

    #: Part de l'empreinte translatée devant être couverte par du sol connu,
    #: pour que l'essai reproduise réellement la situation.
    min_truth_coverage: float = Field(default=0.60, ge=0.0, le=1.0)

    #: Part de l'anneau devant porter des appuis.
    min_ring_coverage: float = Field(default=0.50, ge=0.0, le=1.0)

    #: Part de la vérité masquée que le TIN doit reconstruire.
    min_reconstructed: float = Field(default=0.90, ge=0.0, le=1.0)

    #: Densité de classe 6 au-delà de laquelle un emplacement est réputé
    #: contenir un autre bâtiment.
    max_building_points_per_m2: float = Field(default=0.5, ge=0.0)

    min_trials: int = Field(default=3, ge=1)

    #: Sur quoi ces seuils reposent. Distinct de la calibration du modèle
    #: photographique : une campagne mesurée sur des images n'a rien à dire
    #: d'une validation géospatiale.
    calibration_id: str = UNCALIBRATED
    calibrated_on_sites: int = Field(default=0, ge=0)


class TerrainQualificationThresholds(BaseModel):
    """Seuils de qualification d'un terrain **interpolé**.

    Aucune cellule de sol n'est mesurée sous l'emprise : ces seuils qualifient
    une inférence, pas une observation. Ils portent donc sur la fiabilité de
    l'interpolation, mesurée là où la vérité est connue.
    """

    model_config = ConfigDict(extra="forbid")

    min_dtm_defined: float = Field(default=0.98, ge=0.0, le=1.0)
    require_search_area_within_tile: bool = True
    min_accepted_trials: int = Field(default=3, ge=1)

    #: Sur le **pire** essai, non sur la moyenne : une moyenne dissimulerait
    #: un essai médiocre derrière deux bons.
    max_worst_trial_rmse_m: float = Field(default=0.50, gt=0)
    max_worst_trial_p95_m: float = Field(default=1.00, gt=0)
    max_abs_bias_m: float = Field(default=0.25, gt=0)

    max_support_distance_m: float = Field(default=15.0, gt=0)
    max_rejected_extrapolation: float = Field(default=0.02, ge=0.0, le=1.0)
    max_tin_idw_mae_m: float = Field(default=0.15, gt=0)


class RooflineQualificationThresholds(BaseModel):
    """Seuils d'une surface de toiture **observée**.

    Contrairement au terrain, la toiture est mesurée : ces seuils portent sur
    la couverture et la densité de l'observation, non sur une erreur d'inférence.
    """

    model_config = ConfigDict(extra="forbid")

    min_roof_observed: float = Field(default=0.95, ge=0.0, le=1.0)
    min_main_component: float = Field(default=0.95, ge=0.0, le=1.0)
    min_point_density_per_m2: float = Field(default=10.0, gt=0)
    min_ndsm_valid: float = Field(default=0.95, ge=0.0, le=1.0)
    max_negative_height_fraction: float = Field(default=0.001, ge=0.0, le=1.0)

    #: Une hauteur ne vaut pas mieux que le terrain qui la fonde.
    require_qualified_terrain: bool = True


class QualificationPolicy(BaseModel, Calibrated):
    """Seuils de passage en `inferred`, et ce qu'ils autorisent à en faire.

    `intended_use` n'est pas décoratif : ces seuils qualifient un proxy visuel,
    pas une donnée d'arpentage. Les citer hors de cet usage serait un abus.
    """

    model_config = ConfigDict(extra="forbid")

    status: str = "provisional"
    intended_use: str = "visual_proxy_not_survey"

    #: Distinct de `terrain.calibration_id` : ce dernier décrit la validation
    #: de la méthode, celui-ci le choix des seuils.
    calibration_id: str = UNCALIBRATED
    calibrated_on_sites: int = Field(default=0, ge=0)

    terrain: TerrainQualificationThresholds = Field(
        default_factory=TerrainQualificationThresholds
    )
    roofline: RooflineQualificationThresholds = Field(
        default_factory=RooflineQualificationThresholds
    )


class TemporalPolicy(BaseModel):
    """Ce qu'une datation inconnue autorise, selon l'usage.

    La géométrie d'un volume change peu : une vue non datée reste exploitable
    pour la structure. L'apparence d'une entrée rénovée, non — une image
    antérieure aux travaux y introduirait une erreur invisible.
    """

    model_config = ConfigDict(extra="forbid")

    allow_unknown_for_geometry: bool = True
    allow_unknown_for_appearance: bool = False
    require_current_for_sensitive_zones: bool = True

    #: Portées dont l'apparence engage la fidélité du rendu. Une datation
    #: inconnue y interdit l'usage d'apparence — jamais l'usage géométrique.
    sensitive_scopes: list[str] = Field(default_factory=lambda: ["entrance", "signage"])


class PipelinePolicy(BaseModel):
    """Politique complète, identifiée par sa version."""

    model_config = ConfigDict(extra="forbid")

    version: str = "1.3.0"
    model: ModelPolicy = Field(default_factory=ModelPolicy)
    geometry: GeometryPolicy = Field(default_factory=GeometryPolicy)
    visibility: VisibilityPolicy = Field(default_factory=VisibilityPolicy)
    dedup: DedupPolicy = Field(default_factory=DedupPolicy)
    collection: CollectionPolicy = Field(default_factory=CollectionPolicy)
    coverage: CoveragePolicy = Field(default_factory=CoveragePolicy)
    adaptive_search: AdaptiveSearchPolicy = Field(
        default_factory=AdaptiveSearchPolicy
    )
    terrain: TerrainPolicy = Field(default_factory=TerrainPolicy)
    qualification: QualificationPolicy = Field(default_factory=QualificationPolicy)
    temporal: TemporalPolicy = Field(default_factory=TemporalPolicy)


#: Politique par défaut. Toute fonction qui l'accepte doit la prendre en
#: paramètre, jamais lire une constante de module.
DEFAULT_POLICY = PipelinePolicy()
