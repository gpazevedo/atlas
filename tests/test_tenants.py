"""Tenant isolation and registry tests."""

from __future__ import annotations

import pytest

from atlas_counsel.service.core import CounselService


def test_tenant_isolation():
    """Two tenants can't see each other's threads."""
    svc = CounselService()
    # Tenant A asks something answerable.
    r_a = svc.ask("Above what value does a single-source purchase need justification?",
                  tenant_id="acme")
    assert r_a.status.value == "answered"
    # Tenant B asks something that pauses (unanswerable).
    r_b = svc.ask("What is our policy on accepting gifts from suppliers?",
                   tenant_id="globex")
    assert r_b.status.value == "needs_input"
    # Resume B's thread as tenant A — should error because different DB.
    r = svc.resume(r_b.thread_id, "steer", tenant_id="acme")
    assert r.status.value == "error"
    assert "different tenant" in r.answer


def test_tenant_registry_caches():
    """Same tenant_id returns the same Tenant object."""
    from atlas_counsel.service.tenants import TenantRegistry
    registry = TenantRegistry()
    t1 = registry.get("acme")
    t2 = registry.get("acme")
    assert t1 is t2
    t3 = registry.get("globex")
    assert t1 is not t3


def test_tenant_id_validation():
    """Invalid tenant IDs are rejected."""
    from atlas_counsel.service.tenants import TenantRegistry
    registry = TenantRegistry()

    bad_values = ["", "UPPERCASE", "has space", "-starts-dash", "ends-dash-",
                  "a" * 65]
    for bad in bad_values:
        with pytest.raises(ValueError):
            registry.get(bad)

    for bad in [None, 123]:
        with pytest.raises((ValueError, TypeError)):
            registry.get(bad)

    # Valid IDs don't raise.
    registry.get("a")
    registry.get("acme")
    registry.get("buyer-team")
    registry.get("tenant-42")


def test_tenant_deep_health():
    """Registry deep_health reports tenant count."""
    from atlas_counsel.service.tenants import TenantRegistry
    registry = TenantRegistry()
    h = registry.deep_health()
    assert h["tenants"] == 0
    registry.get("acme")
    h = registry.deep_health()
    assert h["tenants"] == 1
    assert h["graph"] == "ok"


def test_tenant_mcp_validation():
    """MCP tools reject invalid tenant_id gracefully."""
    from atlas_counsel.service.mcp_server import build_mcp_server
    mcp = build_mcp_server()
    # counsel_ask with bad tenant_id
    for bad in ["UPPER", "has space", ""]:
        # We can't easily invoke MCP tools directly, but we can call the
        # underlying service directly.
        svc = CounselService()
        with pytest.raises(ValueError):
            svc.ask("test", tenant_id=bad)


def test_tenant_resume_needs_matching_tenant():
    """Resuming a thread from the wrong tenant returns an error."""
    svc = CounselService()
    r = svc.ask("What is our policy on accepting gifts from suppliers?",
                tenant_id="acme")
    assert r.status.value == "needs_input"
    r2 = svc.resume(r.thread_id, "steer", tenant_id="globex")
    assert r2.status.value == "error"
    assert "different tenant" in r2.answer
    # Resume with correct tenant should work.
    r3 = svc.resume(r.thread_id, "decline", tenant_id="acme")
    assert r3.status.value == "refused"
