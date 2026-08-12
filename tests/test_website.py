"""Collecteur du site officiel — extraction d'images, sans réseau."""

from __future__ import annotations

import pytest

bs4 = pytest.importorskip("bs4", reason="beautifulsoup4 requis")

from hotel_pipeline.collectors.website import _image_urls, _internal_links, _is_useful, _slug

BASE = "https://exemple.invalid/fr/"

HTML = """
<html><body>
  <img src="/img/facade.jpg">
  <img data-src="/img/entree.png">
  <img data-lazy-src="img/chambre.webp">
  <img src="/img/logo.png">
  <img src="/img/icon-menu.svg">
  <img srcset="/img/petit.jpg 400w, /img/grand.jpg 1600w">
  <div style="background-image:url('/img/hero.jpg')"></div>
  <a href="/fr/galerie">Galerie</a>
  <a href="https://autre.invalid/x">Externe</a>
  <a href="/fr/galerie#bas">Ancre</a>
</body></html>
"""


class TestImageExtraction:
    @pytest.fixture
    def urls(self):
        return _image_urls(HTML, BASE)

    def test_plain_src_found(self, urls):
        assert "https://exemple.invalid/img/facade.jpg" in urls

    def test_lazy_attributes_found(self, urls):
        """Les galeries modernes chargent en différé : ignorer data-src rate tout."""
        assert "https://exemple.invalid/img/entree.png" in urls
        assert "https://exemple.invalid/fr/img/chambre.webp" in urls

    def test_srcset_takes_the_largest(self, urls):
        assert "https://exemple.invalid/img/grand.jpg" in urls
        assert "https://exemple.invalid/img/petit.jpg" not in urls

    def test_css_background_found(self, urls):
        assert "https://exemple.invalid/img/hero.jpg" in urls


class TestFiltering:
    def test_photographs_are_kept(self):
        assert _is_useful("https://x.invalid/img/facade.jpg")

    def test_logos_and_icons_are_dropped(self):
        assert not _is_useful("https://x.invalid/img/logo.png")
        assert not _is_useful("https://x.invalid/img/icon-menu.png")

    def test_non_images_are_dropped(self):
        assert not _is_useful("https://x.invalid/style.css")
        assert not _is_useful("https://x.invalid/img/icon.svg")


class TestCrawlScope:
    def test_only_internal_links_followed(self):
        """Explorer un domaine tiers sortirait du périmètre du site officiel."""
        links = _internal_links(HTML, BASE)
        assert "https://exemple.invalid/fr/galerie" in links
        assert not any("autre.invalid" in link for link in links)

    def test_anchors_are_not_duplicated(self):
        links = _internal_links(HTML, BASE)
        assert links.count("https://exemple.invalid/fr/galerie") == 1


class TestSlug:
    def test_readable_identifier(self):
        assert _slug("https://x.invalid/img/facade-principale.jpg", 3) == "003-facade-principale"

    def test_unnamed_file_still_gets_an_id(self):
        assert _slug("https://x.invalid/img/.jpg", 7) == "007"
