"""
SSRF (Server-Side Request Forgery) Protection
==============================================

Validates URLs before making outbound HTTP requests to prevent
the server from being used as a proxy to access internal resources.

References:
- OWASP SSRF Prevention Cheat Sheet
- RFC 1918 (Private Internets), RFC 6890 (Special-Purpose IP Registries)
- IANA Special-Purpose Address Registry
"""

import ipaddress
import logging
import socket
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# IPv4 blocked ranges (RFC 1918, RFC 6890, IANA Special-Purpose Registry)
# ---------------------------------------------------------------------------
_BLOCKED_IPV4 = [
    ipaddress.ip_network("0.0.0.0/8"),          # "This" network (RFC 1122)
    ipaddress.ip_network("10.0.0.0/8"),          # Private Class A (RFC 1918)
    ipaddress.ip_network("100.64.0.0/10"),       # Carrier-grade NAT / Alibaba metadata 100.100.100.200 (RFC 6598)
    ipaddress.ip_network("127.0.0.0/8"),         # Loopback (RFC 1122)
    ipaddress.ip_network("169.254.0.0/16"),      # Link-local / cloud metadata (RFC 3927)
    ipaddress.ip_network("172.16.0.0/12"),       # Private Class B (RFC 1918)
    ipaddress.ip_network("192.0.0.0/24"),        # IETF Protocol Assignments / Oracle metadata 192.0.0.192 (RFC 6890)
    ipaddress.ip_network("192.0.2.0/24"),        # Documentation TEST-NET-1 (RFC 5737)
    ipaddress.ip_network("192.88.99.0/24"),      # 6to4 Relay deprecated (RFC 7526)
    ipaddress.ip_network("192.168.0.0/16"),      # Private Class C (RFC 1918)
    ipaddress.ip_network("198.18.0.0/15"),       # Benchmarking (RFC 2544)
    ipaddress.ip_network("198.51.100.0/24"),     # Documentation TEST-NET-2 (RFC 5737)
    ipaddress.ip_network("203.0.113.0/24"),      # Documentation TEST-NET-3 (RFC 5737)
    ipaddress.ip_network("224.0.0.0/4"),         # Multicast (RFC 5771)
    ipaddress.ip_network("240.0.0.0/4"),         # Reserved (RFC 1112)
    ipaddress.ip_network("255.255.255.255/32"),  # Broadcast (RFC 919)
]

# ---------------------------------------------------------------------------
# IPv6 blocked ranges
# ---------------------------------------------------------------------------
_BLOCKED_IPV6 = [
    ipaddress.ip_network("::1/128"),             # Loopback
    ipaddress.ip_network("::/128"),              # Unspecified
    ipaddress.ip_network("::ffff:0:0/96"),       # IPv4-mapped IPv6 (bypass vector)
    ipaddress.ip_network("64:ff9b:1::/48"),      # IPv4/IPv6 Translation (RFC 8215)
    ipaddress.ip_network("100::/64"),            # Discard (RFC 6666)
    ipaddress.ip_network("2001::/32"),           # Teredo tunneling (RFC 4380)
    ipaddress.ip_network("2001:10::/28"),        # Deprecated ORCHID (RFC 4843)
    ipaddress.ip_network("2001:db8::/32"),       # Documentation (RFC 3849)
    ipaddress.ip_network("2002::/16"),           # 6to4 deprecated (RFC 7526)
    ipaddress.ip_network("fc00::/7"),            # Unique Local Address (RFC 4193)
    ipaddress.ip_network("fe80::/10"),           # Link-local
    ipaddress.ip_network("ff00::/8"),            # Multicast
]

_BLOCKED_NETWORKS = _BLOCKED_IPV4 + _BLOCKED_IPV6

# ---------------------------------------------------------------------------
# Cloud metadata hostnames (resolve to internal IPs but bypass IP checks
# if the attacker controls DNS)
# ---------------------------------------------------------------------------
_BLOCKED_HOSTNAMES = {
    "localhost",
    "metadata.google.internal",
    "metadata",
    "metadata.packet.net",
    "kubernetes.default.svc",
    "kubernetes.default.svc.cluster.local",
    "kubernetes.default",
}

_ALLOWED_SCHEMES = {"http", "https"}


def _is_ip_blocked(ip_str: str) -> bool:
    """Check if an IP address falls within any blocked network."""
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        return True  # Unparseable IP → block

    for network in _BLOCKED_NETWORKS:
        if ip in network:
            return True
    return False


def validate_url(url: str) -> None:
    """
    Validate a URL against SSRF attacks.

    Checks:
    1. Scheme is http or https (blocks file://, gopher://, ftp://, etc.)
    2. Hostname is not a known cloud metadata DNS name
    3. All resolved IPs are public (not private/loopback/link-local/etc.)

    Raises ValueError with a safe message if the URL is blocked.
    """
    parsed = urlparse(url)

    # 1. Scheme check
    if parsed.scheme.lower() not in _ALLOWED_SCHEMES:
        raise ValueError(f"Blocked URL scheme: {parsed.scheme}")

    hostname = parsed.hostname
    if not hostname:
        raise ValueError("No hostname in URL")

    # 2. Block known cloud metadata hostnames
    if hostname.lower() in _BLOCKED_HOSTNAMES:
        raise ValueError("Blocked internal hostname")

    # 3. Resolve DNS → check every IP
    try:
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        addr_infos = socket.getaddrinfo(hostname, port)
    except socket.gaierror:
        raise ValueError(f"Cannot resolve hostname: {hostname}")

    for _family, _type, _proto, _canon, sockaddr in addr_infos:
        ip_str = sockaddr[0]
        if _is_ip_blocked(ip_str):
            logger.warning(
                "[SSRF] Blocked request to %s — resolved to internal IP %s",
                hostname,
                ip_str,
            )
            raise ValueError("URL resolves to a blocked internal address")
