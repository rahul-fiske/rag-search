# rag-search containerisation plan (phase 1, agreed baseline)

Status: high-level approach agreed with the author (2026-10-01); phase 2 = detailed design.

Note (2026-10-02): the proposed VLM tier for scanned documents (`document-conversion-analysis.md`)
needs the Mac's GPU, which a Linux container on a Mac cannot use. Proposal: heavy indexing stays
native on the Mac; the container imports exported collections (§11 there). Not yet decided.

## 1. One core package, two ways to run it
- The `rag-search` Python package (the wheel from the **external** build) is the single core. Every feature lives in it.
- Two delivery modes:
  - **Native**: `install.sh` / `uv tool install` on a MacBook (or Linux). It uses Metal/MPS and ocrmac on Apple Silicon.
  - **Container**: the image installs the *same* wheel, at the same version, plus Linux-only extras and the default models, with container defaults.
- Only the external build exists. The internal variant, its optional host module and the internal-host blocks are removed. Claude and other hosts connect generically.
- Platform-specific dependencies become package extras:
  - `[mcp]`
  - `[mac]` (ocrmac)
  - `[linux-ocr]` (e.g. rapidocr)
  - optionally `[cuda]`
- New `rag-search serve`: a foreground supervisor that runs both daemons plus the HTTP server. It is the container entrypoint and can also be used natively.
- The container-only files (Dockerfile, entrypoint, compose, model-seeding script) live in the repo under `container/`, are built by the same release script and are tagged with the same version.

## 2. Mode defaults (all features in the package)

| Feature | Native default | Container default |
|---|---|---|
| Unix-socket daemons, CLI | yes | yes (`docker exec`) |
| stdio MCP adapter (`rag-search-mcp`) | yes | not used |
| HTTP server: UI `/` + MCP `/mcp` | on, all interfaces, port 8765 | on, all interfaces, port 8765 (published on all host interfaces) |
| Access keys (admin/reader) | required on HTTP, loopback included | required |
| Docs location | `<home>/docs` or `RAG_SEARCH_DOCS` | `/docs` bind mount |
| Data home | `~/Library/Application Support/rag-search` | private area `/var/lib/rag-search` |
| Model cache | HF cache (configurable) | inside the private area |
| Start at boot | launchd (macOS) | restart policy `unless-stopped` |
| Acceleration | MPS (Apple Silicon) | CPU; CUDA on Linux GPU hosts |

## 3. Storage (container mode)
- `/docs`: the host docs folder, bind-mounted (read-only by default). This is the only thing taken from the base machine.
- Private area `/var/lib/rag-search`: converted text, indexes, generations, config, keys, access rules, descriptions, logs, jobs, and models. The bundled models are seeded at first start, and later downloads go there too.
- **Lifetime:** the private area survives container restarts and host reboots; it is deleted when the container is deleted (accepted).
  - Default: it lives in the container's own writable layer, which matches the requested lifetime exactly.
  - Consequence: upgrading to a new image means a new container. That means a fresh private area and a re-index, unless `export` is used first.
  - Option: run with a named volume instead, which survives deletion and upgrades.
- Reboots: the restart policy is `unless-stopped`. On macOS the container runtime (Docker Desktop / OrbStack) must be set to start at login.
- Stored paths are relative to the docs root in both modes.

## 4. Sharing
- `rag-search export` builds a new image = base image + a snapshot of the private area, ready to search. With the writable-layer default this maps to `docker commit` plus scrubbing.
- Keys are excluded from the export, and the recipient gets a new admin key.
- The converted text is the full document content, so sharing exposes it.
- Imported collections are marked frozen so the recipient's indexing doesn't drop them.
- Distribution is via a registry or a `docker save` tarball, plus a compose file.

## 5. Network
- One plain-HTTP port (default 8765). HTTPS is not planned.
- `/` serves the UI; `/mcp` serves MCP over streamable HTTP.
- Listens on **all local addresses** (IPv4 `0.0.0.0` and IPv6 `::`). Local clients use `http://localhost:8765`; remote clients use any of the machine's IPs.
  - Native: the server binds all interfaces directly.
  - Container: it binds `0.0.0.0` inside, and the port is published as `-p 8765:8765`, which the runtime exposes on all host interfaces (Docker Desktop/OrbStack forward this on macOS).
  - The bind address stays configurable (e.g. `127.0.0.1` for local-only).
- Every HTTP request needs a key, loopback included.
- Firewall notes:
  - macOS may ask to allow incoming connections (native).
  - On Linux, Docker-published ports bypass ufw/firewalld rules.
- The Connections page shows a local URL (`http://localhost:8765`) and a remote URL. The remote URL uses the machine's IPs natively, and in the container is pre-filled from the browser's address and editable (the "Public URL" setting).

## 6. Authentication
- Named access keys. Each key carries a client name, a role and allowed collections.
- Roles:
  - **admin**: exactly one. Created at first start; `rag-search token show admin` (natively, or via `docker exec`). Sees all collections and all keys, and runs setup, indexing, users, models and config.
  - **reader**: any number. Can list, search and grep within granted collections only. New collections are hidden from readers until granted.
- Keys are stored readable (mode 0600) in the data home. Commands: `token create|list|show|revoke|rotate`.
- MCP:
  - A reader key gets list, search and grep.
  - The admin key also gets the indexing tools and describe.
- The native stdio adapter and the CLI are local-user, as today.

## 7. Initial setup (UI Setup tab + CLI equivalents)
1. Read the admin key with the CLI, open the UI and log in.
2. Choose the docs root (container: within `/docs`) and review the detected collections.
3. Confirm the remote URL.
4. Create reader keys and assign collections.
5. Set the pipeline settings.
6. Run the first index.

## 8. Connections page (admin)
Pick a key, a client type, and local or remote, to get a copy-ready snippet:

- **Claude Code**: `claude mcp add --transport http rag-search <URL>/mcp --header "Authorization: Bearer <KEY>"`
- **Claude Desktop**: `npx -y mcp-remote <URL>/mcp --allow-http --header Authorization:${RAG_AUTH}`, with env `RAG_AUTH="Bearer <KEY>"`. This needs Node.js on the host; remote custom connectors come from Anthropic's cloud and can't reach a LAN or localhost server. On a native install the existing stdio adapter remains the simplest option.
- **Other MCP hosts**: generic URL + bearer header.
- **Web UI**: open the URL and paste the key.

## 9. Operations
- Container sizing: about 6–8 GB RAM, and about 10 GB of disk for the image plus the private area.
- Benchmark CPU indexing on a sample.
- Backups: `export` (container) or a copy of the data home (native).
- One version number for the wheel and the image (`0.8.x`). An index-format change triggers an automatic re-index.

## 10. Code changes in rag-search
- HTTP server: binds all interfaces (dual-stack) by default with a configurable bind address, no loopback check, MCP streamable-HTTP transport.
- Token store, with role and collection enforcement at the HTTP edge.
- `rag-search serve` supervisor.
- Setup and Connections UI, with the local and remote URLs.
- Relative paths, and a configurable model cache location.
- `configured_model` reads the actual home.
- No `ps` dependency.
- `export` and frozen collections.
- Package extras per platform.
- Remove the internal-host build; `register` writes host config only in native mode.
- The release script builds the wheel and the image from the same version.
