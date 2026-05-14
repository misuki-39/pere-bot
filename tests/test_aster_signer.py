"""Golden-vector tests for the Aster V3 EIP-712 signer.

We don't have an official test vector from Aster's docs, but ECDSA over EIP-712
is deterministic (RFC 6979): a fixed (privkey, message) tuple always yields the
same signature. So we lock in *our own* baseline. If a future change to the
signer breaks compatibility with the venue, this test catches the drift.

The expected signature here was generated from the same code path on first run.
"""

from __future__ import annotations

import pytest

from perp_arb.exchanges.aster.signer import AsterSigner, NonceGenerator

# Demo creds from the Aster V3 docs (publicly published).
USER = "0x63DD5aCC6b1aa0f563956C0e534DD30B6dcF7C4e"
SIGNER = "0x21cF8Ae13Bb72632562c6Fff438652Ba1a151bb0"
PRIVKEY = "0x4fd0a42218f3eae43a6ce26d22544e986139a01e5b34a62db53757ffca81bae1"


def test_init_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError):
        AsterSigner(user="notahex", signer=SIGNER, signer_privkey=PRIVKEY)
    with pytest.raises(ValueError):
        AsterSigner(user=USER, signer="0xtoo_short", signer_privkey=PRIVKEY)
    with pytest.raises(ValueError):
        AsterSigner(user=USER, signer=SIGNER, signer_privkey="not_a_key")


def test_encode_msg_is_insertion_order() -> None:
    """The msg must be url-encoded in dict-insertion order, not alphabetical."""
    out = AsterSigner.encode_msg({"b": 2, "a": 1, "c": 3})
    assert out == "b=2&a=1&c=3"


def test_encode_msg_stringifies_values() -> None:
    out = AsterSigner.encode_msg({"qty": 0.05, "side": "BUY", "n": 100})
    assert "qty=0.05" in out
    assert "side=BUY" in out
    assert "n=100" in out


def test_sign_is_deterministic() -> None:
    """RFC 6979 ECDSA: same (privkey, message) -> same signature, every time."""
    s = AsterSigner(USER, SIGNER, PRIVKEY)
    sig1 = s.sign_msg("symbol=ETHUSDT&side=BUY&type=MARKET&quantity=0.05&nonce=1700000000000000&user=" + USER + "&signer=" + SIGNER)
    sig2 = s.sign_msg("symbol=ETHUSDT&side=BUY&type=MARKET&quantity=0.05&nonce=1700000000000000&user=" + USER + "&signer=" + SIGNER)
    assert sig1 == sig2
    # signatures are 65-byte ECDSA: 2 (0x) + 64 (r) + 64 (s) + 2 (v) = 132 chars
    assert sig1.startswith("0x")
    assert len(sig1) == 132


def test_sign_changes_with_message() -> None:
    s = AsterSigner(USER, SIGNER, PRIVKEY)
    a = s.sign_msg("symbol=ETHUSDT&side=BUY")
    b = s.sign_msg("symbol=ETHUSDT&side=SELL")
    assert a != b


def test_sign_appends_nonce_user_signer_in_order() -> None:
    s = AsterSigner(USER, SIGNER, PRIVKEY)
    req = s.sign(
        {"symbol": "ETHUSDT", "side": "BUY", "type": "MARKET", "quantity": "0.05"},
        nonce=1700000000000000,
    )
    # query_string order: original params, then nonce, then user, then signer, then signature
    assert req.query_string.startswith("symbol=ETHUSDT&side=BUY&type=MARKET&quantity=0.05&")
    assert "&nonce=1700000000000000&" in req.query_string
    assert f"&user={USER}&" in req.query_string
    assert f"&signer={SIGNER}&" in req.query_string
    assert "&signature=0x" in req.query_string

    assert req.form_body["symbol"] == "ETHUSDT"
    assert req.form_body["nonce"] == "1700000000000000"
    assert req.form_body["user"] == USER
    assert req.form_body["signer"] == SIGNER
    assert req.form_body["signature"].startswith("0x")


def test_sign_does_not_mutate_caller_dict() -> None:
    s = AsterSigner(USER, SIGNER, PRIVKEY)
    params = {"symbol": "ETHUSDT", "side": "BUY"}
    _ = s.sign(params, nonce=1700000000000001)
    assert params == {"symbol": "ETHUSDT", "side": "BUY"}


def test_nonce_strictly_monotonic_under_collision() -> None:
    n = NonceGenerator()
    seen = [n.next() for _ in range(100)]
    assert seen == sorted(seen)
    assert len(set(seen)) == 100   # all unique


def test_sign_golden_vector() -> None:
    """Lock the exact signature for a known params/nonce pair.

    ECDSA over EIP-712 with RFC 6979 is deterministic, so any drift here
    means we changed *how* we sign — which would break venue compatibility.
    """
    s = AsterSigner(USER, SIGNER, PRIVKEY)
    req = s.sign(
        {"symbol": "ETHUSDT", "side": "BUY", "type": "MARKET", "quantity": "0.05"},
        nonce=1700000000000000,
    )
    assert req.form_body["signature"] == (
        "0x4cd17e01d1cd6374c0f2fee45deec0c228a3f282efd87e536bf43b7de95948ea"
        "0da8a2b28f3f9b1714d873fd3068fc5249a75de1642fd385faf27a046c0f78b71b"
    )
    assert req.query_string == (
        "symbol=ETHUSDT&side=BUY&type=MARKET&quantity=0.05"
        "&nonce=1700000000000000"
        f"&user={USER}"
        f"&signer={SIGNER}"
        "&signature=" + req.form_body["signature"]
    )
