"""Tests for trusted_proxies CIDR validation at startup."""

from __future__ import annotations

import pytest

from serving.config.settings import Settings


def test_trusted_proxies_valid_cidrs():
    s = Settings(
        trusted_proxies=["172.16.0.0/12", "10.0.0.0/8", "fd00::/8"],
    )
    assert len(s.trusted_proxies_parsed) == 3


def test_trusted_proxies_empty_by_default():
    s = Settings()
    assert s.trusted_proxies == []
    assert s.trusted_proxies_parsed == ()


def test_trusted_proxies_invalid_cidr_fails():
    with pytest.raises(ValueError, match="trusted_proxies entry"):
        Settings(trusted_proxies=["not-a-cidr"])


def test_trusted_proxies_from_comma_string():
    """Accept a comma-separated env var (field_validator handles this)."""
    s = Settings(trusted_proxies="172.16.0.0/12, 10.0.0.0/8")
    assert len(s.trusted_proxies) == 2
    assert len(s.trusted_proxies_parsed) == 2
