"""Deterministic source-identity helpers for corroboration.

Two URLs are not automatically two independent sources.  At minimum they must
come from different publisher hosts; otherwise two paths, tracking variants, or
language editions on one site would be counted twice.  This helper deliberately
uses a conservative registrable-domain approximation and needs no network-backed
public-suffix lookup.
"""

from __future__ import annotations

import ipaddress
from urllib.parse import urlparse

_MULTI_LABEL_SUFFIXES = {
    "co.jp",
    "co.kr",
    "co.nz",
    "co.uk",
    "com.au",
    "com.br",
    "com.cn",
    "com.hk",
    "com.sg",
    "com.tw",
    "gov.au",
    "gov.uk",
    "org.au",
    "org.uk",
}


def publisher_identity(url: str | None, *, fallback: str = "") -> str:
    """Return a stable publisher identity for a URL.

    Different subdomains of the same registrable domain count as one publisher.
    IP addresses and localhost-style hosts are preserved.  When a source has no
    URL, its document id is used as a conservative fallback.
    """

    raw = (url or "").strip()
    host = (urlparse(raw).hostname or "").strip(".").casefold()
    if not host:
        return f"document:{fallback.strip().casefold()}" if fallback.strip() else ""

    try:
        ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        return host

    labels = [label for label in host.split(".") if label]
    if len(labels) <= 2:
        return host
    suffix = ".".join(labels[-2:])
    if suffix in _MULTI_LABEL_SUFFIXES and len(labels) >= 3:
        return ".".join(labels[-3:])
    return suffix
