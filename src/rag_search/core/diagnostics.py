"""`rag-search setup` (download models) and `rag-search doctor` (health checks)."""

from __future__ import annotations

import importlib
import importlib.metadata as md
import importlib.util
import json
import os
import platform
import shutil
import sys
import tempfile
import time
from pathlib import Path

from .. import api, client, models
from ..config import ConfigStore
from ..paths import Paths, default_home, ensure_dirs, get_paths, read_json

OK, WARN, FAIL = "ok", "warn", "FAIL"

HF_IGNORE = models.HF_IGNORE      # repo files we never need (saves ~2-4 GB of downloads)


def _line(status: str, label: str, detail: str = "") -> tuple[str, str, str]:
    return status, label, detail


def _access_lines(paths: Paths, store: ConfigStore) -> list[tuple[str, str, str]]:
    """Who may use which collections (access.json), plus notes for people upgrading from <= 0.2.4."""
    from .. import access, policy

    out: list[tuple[str, str, str]] = []
    rules, err = policy.load_rules(paths)
    if err:
        out.append(_line(WARN, "access rules", f"{err} - restricted collections stay closed "
                         "until it is fixed"))
    elif rules.by_name:
        text = "; ".join(f"{n}: {', '.join(c) if c else 'nobody'}"
                         for n, c in sorted(rules.by_name.items()))
        out.append(_line(OK, "access rules", f"{len(rules.by_name)} restricted collection(s) - {text}"))
    else:
        out.append(_line(OK, "access rules", "none: every collection is open to every client "
                         "(restrict with: rag-search access restrict COLLECTION CLIENT)"))
    try:
        legacy = "policy" in json.loads(paths.config_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        legacy = False
    if legacy:
        out.append(_line(WARN, "config.json policy", "the 'policy' section (private collections) "
                         "was replaced in 0.2.5 and is ignored; use `rag-search access`"))
    open_legacy = [n for n in ("personal", "private")
                   if n in access.known_collections(paths) and not rules.restricted(n)]
    if open_legacy:
        out.append(_line(WARN, "access: " + ", ".join(open_legacy),
                         "these were private by default before 0.2.5 and are now open to every "
                         "client; restrict them with: rag-search access restrict NAME claude"))
    return out


def download_models(skip_docling: bool = False) -> list[str]:
    from huggingface_hub import snapshot_download

    from .embedding import prepare_environment

    prepare_environment()
    msgs = []
    for repo in (models.selection(kind)[0] for kind in models.KINDS):
        print(f"Downloading {repo} (cached after the first time) ...", flush=True)
        path = snapshot_download(repo_id=repo, ignore_patterns=HF_IGNORE)
        msgs.append(f"{repo}: {path}")
    if not skip_docling:
        print("Downloading docling layout/table models ...", flush=True)
        try:
            from docling.utils.model_downloader import download_models as dl

            dl()
            msgs.append("docling models: downloaded")
        except Exception as exc:  # noqa: BLE001
            msgs.append(f"docling models: not pre-downloaded ({exc}); they will download on first use")
    return msgs


def _valid_id(kind: str, model_id: str) -> bool:
    try:
        models.spec_for(kind, model_id)
        return True
    except models.ModelError:
        return False


def run_checks(paths: Paths) -> list[tuple[str, str, str]]:
    out: list[tuple[str, str, str]] = []
    v = sys.version_info
    out.append(_line(OK if (3, 11) <= v[:2] < (3, 13) else WARN, "python",
                     f"{platform.python_version()} ({sys.executable})"))
    out.append(_line(OK, "platform", f"{platform.system()} {platform.machine()}"))
    from .docling_convert import convert_profile
    out.append(_line(OK, "conversion settings", convert_profile()))
    from .. import __version__
    out.append(_line(OK, "rag-search", f"{__version__} ({Path(__file__).resolve().parent.parent})"))

    for mod, dist in [("numpy", "numpy"), ("torch", "torch"),
                      ("sentence_transformers", "sentence-transformers"),
                      ("huggingface_hub", "huggingface_hub"), ("docling", "docling")]:
        try:
            importlib.import_module(mod)
            out.append(_line(OK, f"import {mod}", md.version(dist)))
        except Exception as exc:  # noqa: BLE001
            out.append(_line(FAIL, f"import {mod}", f"{type(exc).__name__}: {exc}"))
            if mod == "sentence_transformers":
                out.append(_line(WARN, "hint", "a new transformers needs PyTorch >= 2.4 (Intel Macs "
                                 "stop at 2.2.2): install with `transformers<5` and `huggingface_hub<1`, "
                                 "e.g. re-run install.sh from release 0.2.1 or later"))

    try:
        from .embedding import pick_device

        out.append(_line(OK, "compute device", pick_device()))
    except Exception as exc:  # noqa: BLE001
        out.append(_line(WARN, "compute device", str(exc)))

    try:
        from .docling_convert import convert_settings, pdf_backend_class

        backend = convert_settings()["pdf_backend"]
        cls = pdf_backend_class(backend)
        out.append(_line(OK, "PDF backend", f"{backend}" + (f" ({cls.__name__})" if cls else " (docling's own)")))
    except Exception as exc:  # noqa: BLE001 - docling missing is reported above; a bad setting here
        out.append(_line(WARN, "PDF backend", f"{type(exc).__name__}: {exc}"))

    try:
        from .docling_convert import convert_settings

        cs = convert_settings()
        engines = []
        for mod, label in (("ocrmac", "ocrmac (Apple Vision)"), ("rapidocr", "rapidocr"),
                           ("rapidocr_onnxruntime", "rapidocr"), ("easyocr", "easyocr"),
                           ("tesserocr", "tesserocr")):
            if importlib.util.find_spec(mod) and label not in engines:
                engines.append(label)
        if shutil.which("tesseract"):
            engines.append("tesseract")
        detail = (f"mode {cs['ocr']}, engine {cs['engine']}; installed: "
                  f"{', '.join(engines) or 'none found'}; threads {cs['threads'] or 'docling default'}, "
                  f"time limit {int(cs['timeout']) or 'none'}{'s' if cs['timeout'] else ''} per document")
        slow = False
        if platform.system() == "Darwin" and "ocrmac (Apple Vision)" not in engines and cs["ocr"] != "off":
            detail += ("; Apple's on-device OCR is not installed, so scanned PDFs and photographed "
                       "documents will fail to index (not just run slower): re-run install.sh (it "
                       "now installs ocrmac on macOS automatically), or add it by hand with "
                       "`uv tool install --force --with ocrmac <path to the wheel/zip you installed "
                       "from>`")
            slow = True
        out.append(_line(WARN if slow else OK, "OCR",
                         detail + ("; force reads every page as an image, the slowest setting "
                                   "(RAG_SEARCH_OCR=smart skips it for clean PDFs)" if cs["ocr"] == "force" else "")))
    except Exception as exc:  # noqa: BLE001
        out.append(_line(WARN, "OCR", f"{type(exc).__name__}: {exc}"))

    try:
        from .conversion import tesseract

        why = tesseract.why_not()
        out.append(_line(OK if not why else WARN, "Tesseract (last-resort page reader)",
                         f"languages {tesseract.languages()}" if not why else
                         why + "; pages the document reader cannot read then stay flagged instead of being read as plain text"))
    except Exception as exc:  # noqa: BLE001
        out.append(_line(WARN, "Tesseract (last-resort page reader)", f"{type(exc).__name__}: {exc}"))
    try:
        if platform.system() == "Darwin" and platform.machine() == "arm64" and importlib.util.find_spec("mlx_vlm"):

            for kind, what in ((models.READER, "document reader"), (models.REPAIR, "repair model (re-reads flagged pages)")):
                mid = models.vlm_selection(kind)[0]
                st = models.cache_state(mid)
                out.append(_line(OK if st["cached"] else WARN, what,
                                 mid + ("" if st["cached"] else
                                        f": not downloaded ({st.get('why') or 'rag-search models download --reader --repair'})")))
    except Exception as exc:  # noqa: BLE001
        out.append(_line(WARN, "document reader models", f"{type(exc).__name__}: {exc}"))

    dpy = os.environ.get("RAG_SEARCH_DOCLING_PYTHON")
    if dpy:
        out.append(_line(OK if Path(dpy).is_file() else FAIL, "RAG_SEARCH_DOCLING_PYTHON", dpy))

    try:
        ensure_dirs(paths)
        probe = paths.home / ".write_test"
        probe.write_text("x")
        probe.unlink()
        free_gb = shutil.disk_usage(paths.home).free / 1e9
        out.append(_line(OK if free_gb > 10 else WARN, "data folder",
                         f"{paths.home}  ({free_gb:.0f} GB free)"))
    except OSError as exc:
        out.append(_line(FAIL, "data folder", f"{paths.home}: {exc}"))
    from .. import locations

    _, loc_err = locations.load(paths)
    if loc_err:
        out.append(_line(WARN, "source locations", loc_err))
    for st in locations.status(paths):
        out.append(_line(OK if st["reachable"] else WARN, f"location {st['collection']}",
                         st["folder"] + ("" if st["reachable"] else
                                         "  (unreachable: skipped by indexing, keeps its index)")))
    imported = locations.imported_names(paths)
    if imported:
        out.append(_line(OK, "imported collections", ", ".join(
            f"{c} ({locations.origin(paths, c).get('model', '?')})" for c in imported)))

    for kind in models.KINDS:
        repo, source = models.selection(kind)
        cs = models.cache_state(repo)
        spec = models.spec_for(kind, repo) if _valid_id(kind, repo) else None
        lacking = models.missing_requirements(spec) if spec else []
        if lacking:
            out.append(_line(FAIL, f"model {repo}", "; ".join(lacking)))
        else:
            out.append(_line(OK if cs["cached"] else WARN, f"model {repo}",
                             f"{kind} model ({source}), " + ("cached" if cs["cached"] else
                             "download interrupted: run `rag-search models download`" if cs["partial"]
                             else "not downloaded yet (run: rag-search setup)")))
    serving = models.serving_model(paths)
    wanted = models.selection(models.EMBEDDING)[0]
    if serving and serving != wanted:
        out.append(_line(WARN, "embedding model switch",
                         f"the published index was built with {serving} but {wanted} is configured: "
                         "search keeps using the old one until `rag-search index new` has "
                         "re-embedded every document"))

    try:
        importlib.import_module("mcp")
        out.append(_line(OK, "import mcp (adapter)", md.version("mcp")))
    except Exception as exc:  # noqa: BLE001
        out.append(_line(WARN, "import mcp (adapter)",
                         f"{type(exc).__name__}: {exc} - CLI works; MCP hosts need rag-search[mcp]"))

    store = ConfigStore(paths)
    store.get()
    out.append(_line(WARN if store.error else OK, "config.json", store.error or str(paths.config_file)))
    out.extend(_access_lines(paths, store))
    from ..ui import info as ui_info

    missing = [n for n in ("index.html", "core.js", "style.css", "docs/README.md", "docs/ARCHITECTURE.md")
               if not (ui_info.STATIC / n).is_file()]
    out.append(_line(WARN if missing else OK, "web dashboard",
                     f"files missing: {', '.join(missing)} - reinstall" if missing
                     else "rag-search ui  (http://127.0.0.1:8765 by default)"))

    try:
        from .. import register

        for label, status, detail in register.registration_status():
            out.append(_line({"ok": OK, "warn": WARN, "fail": FAIL}[status], f"MCP: {label}", detail))
    except Exception as exc:  # noqa: BLE001 - a status row must never break doctor
        out.append(_line(WARN, "MCP registration", f"{type(exc).__name__}: {exc}"))

    info = client.ping(paths, "search", 1.0)
    if info:
        bad = info.get("state") == "error"
        w = info.get("warmup") or {}
        warm = (f", warmed up in {w.get('total_s')}s" if w.get("status") == "warm"
                else f", warming up ({w.get('phase', '')}, {w.get('elapsed_s')}s so far)"
                if w.get("status") == "warming_up" else "")
        rss = (info.get("memory") or {}).get("rss_bytes")
        mem = f", {rss / 2**20:.0f} MB resident" if rss else ""
        out.append(_line(FAIL if bad else OK, "search daemon",
                         f"{info['state']} (pid {info['pid']}, generation {info.get('generation')}, "
                         f"{info.get('chunks', 0)} chunks{warm}{mem})"))
        if bad:
            out.append(_line(FAIL, "search daemon error", info.get("error") or "see search.log"))
        if info.get("index_error"):
            out.append(_line(WARN, "search index", info["index_error"]))
    else:
        out.append(_line(WARN, "search daemon", "not running (starts on demand; "
                         "`rag-search service install` keeps it up)"))
    info = client.ping(paths, "indexer", 1.0)
    out.append(_line(OK if info else WARN, "indexer daemon",
                     (f"pid {info['pid']}, {'run active' if info.get('running') else 'idle'}")
                     if info else "not running (starts on demand)"))
    return out


def print_checks(rows: list[tuple[str, str, str]]) -> int:
    worst = 0
    for status, label, detail in rows:
        print(f"[{status:>4}] {label:<38} {detail}")
        worst = max(worst, {OK: 0, WARN: 0, FAIL: 1}[status])
    return worst


SAMPLE_DOCS = {
    "aurora.md": "# Aurora\n\nAn aurora is a natural light display in the sky, caused by "
                 "charged solar particles colliding with atmospheric gases near the poles.",
    "sourdough.md": "# Sourdough\n\nSourdough bread rises using a fermented starter of wild "
                    "yeast and lactic acid bacteria instead of commercial yeast.",
    "tcp.md": "# TCP\n\n<!-- page 1 -->\n\nTCP uses a three-way handshake (SYN, SYN-ACK, ACK) "
              "to establish a reliable connection between two hosts.",
}


def roundtrip() -> int:
    """End-to-end smoke test of the full stack on a throw-away data folder.

    Starts real daemons against a temporary home: indexer daemon -> worker -> publish ->
    search daemon reload -> search.  Uses the real models, so the first run downloads them.
    """
    chosen = read_json(default_home() / "config.json").get("models") or {}   # the models in use
    saved = {k: os.environ.pop(k, None) for k in ("RAG_SEARCH_HOME",)}
    tmp = Path(tempfile.mkdtemp(prefix="rag-doctor-"))
    p = get_paths(tmp)
    os.environ["RAG_SEARCH_HOME"] = str(tmp)
    try:
        ensure_dirs(p)
        picked = {k: chosen[k] for k in models.KINDS if isinstance(chosen.get(k), str) and chosen[k]}
        if picked:
            from ..config import update_config

            update_config(p, "models", picked)
        samples = tmp / "sources" / "samples"
        samples.mkdir(parents=True, exist_ok=True)
        for name, text in SAMPLE_DOCS.items():
            (samples / name).write_text(text)
        from .. import locations as _loc

        _loc._save(p, {"samples": str(samples)})
        t0 = time.time()
        print("indexing 3 sample documents (starts the daemons, loads the models) ...", flush=True)
        r = api.index_start(p, client="cli")
        if not r.get("ok"):
            print("FAIL: could not start indexing:", r)
            return 1
        job: dict = {}
        end = time.time() + 1800
        while time.time() < end:
            st = api.index_status(p)
            job = st.get("job") or {}
            if not st.get("running") and job.get("status") not in (None, "queued", "running"):
                break
            time.sleep(1.0)
        print(f"  indexing: {job.get('status')} {job.get('summary', {}).get('indexed')} doc(s)")
        if job.get("status") != "succeeded":
            print("FAIL: indexing did not succeed:", job.get("error") or job.get("summary"))
            return 1
        checks = [("how do sourdough loaves rise", "sourdough"),
                  ("what causes northern lights", "aurora"),
                  ("SYN-ACK handshake", "tcp")]
        bad = 0
        for q, want in checks:
            resp = api.search(p, q, top_k=1, wait_s=600, client="cli")
            res = resp.get("result", {}).get("results", []) if resp.get("ok") else []
            got = res[0]["file"] if res else None
            print(f"  {'ok ' if got == want else 'BAD'} {q!r} -> {got}"
                  + ("" if resp.get("ok") else f"  ({resp.get('error')})"))
            bad += got != want
        g = api.grep(p, "three-way handshake", client="cli")
        grep_ok = bool(g.get("ok") and g["result"].get("matches"))
        print(f"  {'ok ' if grep_ok else 'BAD'} grep 'three-way handshake'")
        bad += not grep_ok
        print(f"roundtrip {'PASSED' if not bad else 'FAILED'} in {time.time() - t0:.0f}s")
        return 1 if bad else 0
    finally:
        api.daemon_stop(p, "all")
        shutil.rmtree(tmp, ignore_errors=True)
        for k, v in saved.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v
