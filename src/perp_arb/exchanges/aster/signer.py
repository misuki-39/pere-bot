"""Aster V3 EIP-712 request signer.

Reference: ~/project/github-project/aster-api-docs/V3(Recommended)/EN/aster-finance-futures-api-v3.md

Signing flow:
  1. Caller passes a dict of request params (e.g. {"symbol": "ETHUSDT", "side": "BUY", ...}).
  2. We append `nonce`, `user`, `signer` to the dict (insertion-order preserved).
  3. URL-encode the dict -> `msg`.
  4. Build EIP-712 typed data:
        domain = AsterSignTransaction / v1 / chainId={1666 mainnet, 714 testnet} /
                 verifyingContract=0x00..00
        message = { "msg": <urlencoded params> }
  5. ECDSA-sign the typed-data digest with the API wallet's private key.
  6. Append `signature=0x...` to the params.

We deliberately do NOT mutate the caller's dict.
"""

from __future__ import annotations

import threading
import urllib.parse
from dataclasses import dataclass
from typing import Any

from eth_account import Account
from eth_account.messages import encode_typed_data

from ...utils.time import now_us

MAINNET_CHAIN_ID = 1666
TESTNET_CHAIN_ID = 714


def _require_hex(s: str, length: int, label: str) -> None:
    if not s.startswith("0x") or len(s) != length:
        raise ValueError(f"Invalid {label}: {s!r}")

_EIP712_TYPES: dict[str, Any] = {
    "EIP712Domain": [
        {"name": "name", "type": "string"},
        {"name": "version", "type": "string"},
        {"name": "chainId", "type": "uint256"},
        {"name": "verifyingContract", "type": "address"},
    ],
    "Message": [
        {"name": "msg", "type": "string"},
    ],
}


class NonceGenerator:
    """Microsecond-precision monotonic nonce.

    The Aster server requires nonce within ±10s of server time and refuses
    duplicates. When more than one call lands in the same microsecond we
    increment a counter so the nonce is still strictly increasing.
    """

    def __init__(self) -> None:
        self._last_us = 0
        self._lock = threading.Lock()

    def next(self) -> int:
        with self._lock:
            now = now_us()
            self._last_us = now if now > self._last_us else self._last_us + 1
            return self._last_us


@dataclass(frozen=True, slots=True)
class SignedRequest:
    query_string: str   # `key=value&...&signature=0x...` (use for GET/DELETE-by-url)
    form_body: dict[str, str]   # full body incl. `signature` (use for POST form-encoded)


class AsterSigner:
    """Stateful signer holding the API wallet credentials.

    `chain_id` selects mainnet (1666) or testnet (714). The EIP-712 domain
    encodes this, so a signature for the wrong chain will not verify.
    """

    def __init__(
        self,
        user: str,
        signer: str,
        signer_privkey: str,
        *,
        chain_id: int = MAINNET_CHAIN_ID,
    ) -> None:
        _require_hex(user, 42, "user address")
        _require_hex(signer, 42, "signer address")
        _require_hex(signer_privkey, 66, "signer_privkey (32-byte hex)")
        self.user = user
        self.signer = signer
        self.chain_id = chain_id
        self._privkey = signer_privkey
        self._nonces = NonceGenerator()

    # ----- pure helpers (used by golden tests) ------------------------------

    @staticmethod
    def encode_msg(params: dict[str, Any]) -> str:
        """Stable urlencode that mirrors `urllib.parse.urlencode` semantics."""
        return urllib.parse.urlencode({k: str(v) for k, v in params.items()})

    def build_typed_data(self, msg: str) -> dict[str, Any]:
        return {
            "types": _EIP712_TYPES,
            "primaryType": "Message",
            "domain": {
                "name": "AsterSignTransaction",
                "version": "1",
                "chainId": self.chain_id,
                "verifyingContract": "0x0000000000000000000000000000000000000000",
            },
            "message": {"msg": msg},
        }

    def sign_msg(self, msg: str) -> str:
        """EIP-712 sign a raw `msg` string. Returns `0x`-prefixed hex signature."""
        encoded = encode_typed_data(full_message=self.build_typed_data(msg))
        signed = Account.sign_message(encoded, private_key=self._privkey)
        return signed.signature.to_0x_hex()

    # ----- main API ---------------------------------------------------------

    def sign(
        self,
        params: dict[str, Any],
        *,
        nonce: int | None = None,
    ) -> SignedRequest:
        """Augment + sign + serialise. `nonce` override is for tests only."""
        if nonce is None:
            nonce = self._nonces.next()

        # Insertion order matters — the server reconstructs the exact same query string.
        body: dict[str, str] = {k: str(v) for k, v in params.items()}
        body["nonce"] = str(nonce)
        body["user"] = self.user
        body["signer"] = self.signer

        msg = urllib.parse.urlencode(body)
        signature = self.sign_msg(msg)

        body_with_sig = {**body, "signature": signature}
        return SignedRequest(
            query_string=msg + f"&signature={signature}",
            form_body=body_with_sig,
        )
