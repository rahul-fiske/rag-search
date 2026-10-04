"""Wire protocol shared by both daemons and all clients (stdlib only).

One request per connection, newline-delimited JSON:

    -> {"v": 1, "client": "claude", "action": "search", ...fields}
    <- {"ok": true, ...}                      (single reply)
    <- {"event": ..., ...}\n ... {"event": "end"}    (streaming actions, e.g. follow)

Errors: {"ok": false, "code": "...", "error": "human readable"}.  Bump PROTOCOL_VERSION
on any incompatible change; daemons accept MIN_PROTOCOL..PROTOCOL_VERSION.
"""

from __future__ import annotations

import json
import socket
from typing import Any

from .policy import normalize_client

PROTOCOL_VERSION = 1
MIN_PROTOCOL = 1
MAX_REQUEST_BYTES = 1 << 20

# error codes
BAD_REQUEST = "bad_request"
PROTOCOL_MISMATCH = "protocol_mismatch"
WARMING_UP = "warming_up"
UNAVAILABLE = "unavailable"
FORBIDDEN = "forbidden"
MODEL_ERROR = "model_error"
MODEL_MISMATCH = "model_mismatch"
INTERNAL = "internal"
BUSY = "busy"


def make_request(action: str, client: str = "cli", **fields: Any) -> dict[str, Any]:
    return {"v": PROTOCOL_VERSION, "client": client, "action": action, **fields}


def error(code: str, message: str, **extra: Any) -> dict[str, Any]:
    return {"ok": False, "code": code, "error": message, **extra}


def validate_request(req: Any) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Return (request, error_reply).  Fills in the normalised client id."""
    if not isinstance(req, dict):
        return None, error(BAD_REQUEST, "request must be a JSON object")
    v = req.get("v", PROTOCOL_VERSION)
    if not isinstance(v, int) or not (MIN_PROTOCOL <= v <= PROTOCOL_VERSION):
        return None, error(PROTOCOL_MISMATCH,
                           f"client speaks protocol {v!r}; daemon supports "
                           f"{MIN_PROTOCOL}..{PROTOCOL_VERSION}; upgrade rag-search",
                           supported=[MIN_PROTOCOL, PROTOCOL_VERSION])
    if not isinstance(req.get("action"), str):
        return None, error(BAD_REQUEST, "missing action")
    req = dict(req)
    req["client"] = normalize_client(req.get("client"))
    return req, None


def encode(obj: Any) -> bytes:
    return (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")


def read_line(conn: socket.socket, limit: int = MAX_REQUEST_BYTES) -> bytes:
    """Read up to the first newline (or EOF / limit)."""
    buf = b""
    while b"\n" not in buf and len(buf) < limit:
        chunk = conn.recv(65536)
        if not chunk:
            break
        buf += chunk
    return buf.split(b"\n", 1)[0]
