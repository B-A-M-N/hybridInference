"""Tests for trusted_proxies and trusted_cloudflare_networks configuration."""

from __future__ import annotations

import pytest

from serving.config.settings import Settings


def test_trusted_proxies_valid_cidrs_from_string():
    """Comma-separated string is parsed into networks."""
    s = Settings(trusted_proxies="172.16.0.0/12,10.0.0.0/8,fd00::/8")
    assert len(s.trusted_proxies_parsed) == 3


def test_trusted_proxies_empty_by_default():
    s = Settings()
    assert s.trusted_proxies == ""
    assert s.trusted_proxies_parsed == ()
    assert s.trusted_cloudflare_networks == ""
    assert s.trusted_cloudflare_parsed == ()


def test_trusted_proxies_invalid_cidr_fails():
    with pytest.raises(ValueError, match="trusted_proxies entry"):
        Settings(trusted_proxies="not-a-cidr")


def test_trusted_cloudflare_invalid_cidr_fails():
    with pytest.raises(ValueError, match="trusted_cloudflare_networks entry"):
        Settings(trusted_cloudflare_networks="not-a-cidr")


def test_trusted_proxies_from_env_var(monkeypatch):
    """TRUSTED_PROXIES env var works with comma-separated values."""
    monkeypatch.setenv("TRUSTED_PROXIES", "172.16.0.0/12, 10.0.0.0/8")
    s = Settings()
    assert len(s.trusted_proxies_parsed) == 2


def test_trusted_proxies_empty_env_var(monkeypatch):
    """Empty TRUSTED_PROXIES env var produces no trusted networks."""
    monkeypatch.setenv("TRUSTED_PROXIES", "")
    s = Settings()
    assert s.trusted_proxies_parsed == ()


def test_trusted_proxies_env_var_with_whitespace(monkeypatch):
    """Whitespace is handled in env var parsing."""
    monkeypatch.setenv("TRUSTED_PROXIES", " 172.16.0.0/12 , 10.0.0.0/8 ")
    s = Settings()
    assert len(s.trusted_proxies_parsed) == 2


def test_trusted_cloudflare_from_env_var(monkeypatch):
    """TRUSTED_CLOUDFLARE_NETWORKS env var works."""
    monkeypatch.setenv("TRUSTED_CLOUDFLARE_NETWORKS", "172.16.0.0/12")
    s = Settings()
    assert len(s.trusted_cloudflare_parsed) == 1


def test_both_settings_from_env(monkeypatch):
    """Both trusted_proxies and trusted_cloudflare_networks from env."""
    monkeypatch.setenv("TRUSTED_PROXIES", "172.16.0.0/12")
    monkeypatch.setenv("TRUSTED_CLOUDFLARE_NETWORKS", "10.0.0.0/8")
    s = Settings()
    assert len(s.trusted_proxies_parsed) == 1
    assert len(s.trusted_cloudflare_parsed) == 1
