# Trois chantiers : droits, LiDAR étendu, portabilité

## 1. Les droits, restaurés

Le dépistage parcourait le **système de fichiers** quand tout le reste du
pipeline raisonne sur des **assets** qualifiés. Mesuré : cent quarante-trois
des cent soixante-trois images dépistées n'existaient pas au manifeste, et les
références proposées ressortaient avec un statut de droits inconnu.

C'est une garantie que le dépôt tient partout ailleurs, et que la couche
d'identité contournait.

[`identity/candidates.py`](../src/hotel_pipeline/identity/candidates.py) fait
le pont : les candidats viennent du manifeste, avec leurs droits, leur azimut
et leur cohorte. Un recadrage hérite des droits de l'asset dont il dérive ; un
recadrage orphelin garde `unknown`, jamais un statut emprunté.

```
428 candidat(s) — 212 aux droits établis, 216 à vérifier
droits des retenues : {'open_data': 40}
```

Une image aux droits non établis reste classée mais **déclassée** au moment de
proposer des références : la proposer en tête reviendrait à cacher le problème.

### Deux régressions trouvées au passage

**Les noms de concurrents étaient perdus.** `identity screen` construisait sa
liste à la main et ne passait jamais `excluded_names`. L'OCR lisait bien
« TETRA 1205 TECH » sur l'immeuble voisin, mais le rendait `uncertain` faute de
savoir que ce nom disqualifie l'image — et le voisin remontait en tête des
références.

**Le numéro civique n'était pas exploité.** Une seconde vue du même immeuble
ne portait aucun logo, seulement « 1205 » en grand sur sa façade. Un numéro
civique voisin de celui du site — écart de quarante au plus — dément désormais
l'appartenance. Les numéros lointains, téléphones compris, sont ignorés.

---

## 2. Le LiDAR à l'échelle de la scène

`geo discover --scene` interroge l'index avec l'enveloppe des volumes rendus,
non la seule empreinte du bâtiment.

| | Emprise cible | Emprise scène |
|---|---|---|
| Étendue | 72 × 77 m | 1041 × 1077 m |
| Tuiles trouvées | 1 | **4** |

Les quatre tuiles couvrent les dix-huit volumes encore supposés, pour
**845 Mo** à télécharger. Le pipeline ne télécharge rien de lui-même : il
annonce le volume exact et attend un accord.

---

## 3. Portabilité, vérifiée sur un second site

Un site aux caractéristiques opposées — bâtiment de 18 m, autre CRS
(EPSG:32188), LiDAR distinct — traverse la chaîne **sans configuration** :

```
scène       CRS EPSG:32188, 4 volumes, rayon 29 m
hauteurs    1/4 mesurées | cible 18.0 m
végétation  3 couronnes, 1 mât
sol         classé, plages extraites
rendu       verdict condition_strongly
TOTAL       4 s
```

### Un défaut que seul le second site révélait

Sur un terrain de réflectance homogène, le seuil d'intensité — dérivé du relevé
— tombe au milieu d'une population unique et **rien ne le franchit**. Le sol
sortait alors avec zéro plage : il disparaissait entièrement du rendu.

Un terrain existe même quand on ignore son revêtement. Au-delà de soixante pour
cent d'indéterminé, le sol est désormais posé **sans nature établie**, avec sa
propre valeur de silhouette. Le pilote, lui, garde ses deux natures — le repli
ne s'active que là où la mesure ne distingue rien.

---

## Réserves

- Les 845 Mo de tuiles LiDAR **ne sont pas téléchargés** : décision en attente.
- Les dix-huit volumes hors couverture restent supposés jusque-là.
- Le second site est synthétique : il valide la portabilité du code, non celle
  des données réelles d'un autre territoire.
