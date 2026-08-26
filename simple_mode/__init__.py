"""Mode simple : adresse -> image satellite annotée d'un trajet caméra + texte.

Package autonome, sans dépendance à `hotel_pipeline` : il ne lit ni géométrie
mesurée, ni panoramas, ni LiDAR. Les trajectoires qu'il dessine sont des
gabarits de figures de vol standard, centrés sur l'adresse géocodée et mis à
l'échelle de l'image satellite — un aperçu pédagogique, pas un plan de vol
mesuré. Voir README.md pour l'usage et les limites.
"""
