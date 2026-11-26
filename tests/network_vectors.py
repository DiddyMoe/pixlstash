"""Synthetic network vectors for the locality gate.

These are *inputs*, not addresses of anything. They have to stay inside RFC 1918:
the assertions they feed prove the gate refuses private callers, so a vector from
RFC 5737's documentation ranges would exercise a different branch and the negative
assertion would stop meaning anything.

Which is why there are exactly three, and why they are these three. Push-time
secret scanning treats any other RFC 1918 quad as a private address worth stopping
a push over, and exempts only the first host of each block - every network has
something at ``.1``, so the address says nothing about whose. Do not write an
RFC 1918 literal that is not from here; ``tests/test_architecture_guardrails.py``
fails the build on one.

Loopback, CGNAT (``100.64.0.0/10``) and public addresses are not private and were
never a finding, so they stay written out where they are used.
"""

import ipaddress

LAN_IPV4 = "192.168.0.1"
PRIVATE_10_IPV4 = "10.0.0.1"
PRIVATE_172_IPV4 = "172.16.0.1"


def nat64(ipv4):
    """*ipv4* under the NAT64 well-known prefix, 64:ff9b::/96 (RFC 6052)."""
    prefix = int(ipaddress.IPv6Address("64:ff9b::"))
    return str(ipaddress.IPv6Address(prefix | int(ipaddress.IPv4Address(ipv4))))


def ipv4_compatible(ipv4):
    """*ipv4* as a deprecated IPv4-compatible address, in ::/96."""
    return str(ipaddress.IPv6Address(int(ipaddress.IPv4Address(ipv4))))


# Each carries a loopback, link-local or private IPv4, and ``ipaddress`` calls
# each global: a NAT64 translator, or a stack still honouring the compatible
# form, connects it to that IPv4.
WRAPPED_PRIVATE_IPV6 = [
    ipv4_compatible("127.0.0.1"),
    nat64("127.0.0.1"),
    nat64("169.254.169.254"),
    nat64(LAN_IPV4),
    nat64(PRIVATE_10_IPV4),
    ipv4_compatible("169.254.169.254"),
    ipv4_compatible(LAN_IPV4),
]

# IPv6 that Python 3.12's ``ipaddress`` calls global, yet no public host owns:
# an outbound check that trusts ``is_global`` alone lets each through.
GLOBAL_BUT_NOT_PUBLIC_IPV6 = [
    # Site-local, deprecated by RFC 3879 and still routed inside some sites.
    "fec0::1",
    # SIIT's IPv4-translated form, ::ffff:0:0/96 (RFC 6052 section 2.1),
    # here carrying 8.8.8.8: ``is_reserved``, so its IPv4 is never judged.
    "::ffff:0:808:808",
    # SRv6 segment identifiers, 5f00::/16 (RFC 9602).
    "5f00::1",
    # RFC 9637's documentation prefix, 3fff::/20, at both ends.
    "3fff::1",
    "3fff:fff:ffff:ffff:ffff:ffff:ffff:ffff",
]

# The public forms an outbound check must still let through: a public IPv4
# as itself, IPv4-mapped, and under NAT64 (DNS64 answers every IPv4 host so on
# an IPv6-only network), and a plain public IPv6 address.
PUBLIC_ADDRESS_FORMS = [
    "8.8.8.8",
    "::ffff:8.8.8.8",
    nat64("8.8.8.8"),
    "2606:4700:4700::1111",
]
