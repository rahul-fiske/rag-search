"""The menu-bar item (macOS): a status icon that opens the dashboard on a click and has the common operations on a
right-click.  ``rag-search menubar`` runs it; ``rag-search service install`` (and ``setup``) start it at every login.

Everything it does is a call the CLI and the dashboard make too (``rag_search.api``, ``ui.server``); this module adds the
*decisions* -- what the icon says, which menu items exist and when, how a stale dashboard is replaced -- as plain functions
with the standard library only, so tests drive them without a menu bar.  AppKit (PyObjC, installed with rag-search on a Mac)
is imported only inside :func:`run`, which is the thin layer that draws the icon and the menu.

Staleness, which this exists to end: the dashboard and this process are long-running, so after an upgrade they keep serving
old code.  A click on the icon compares the version the running dashboard reports with the installed one and replaces it
when they differ; this process compares its own loaded version with the one on disk every few seconds and re-launches itself.
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from . import __version__, api, client, machine
from .paths import Paths, get_paths

REFRESH_S = 5.0
ICONS = {"ready": "doc.text.magnifyingglass", "warming": "hourglass", "indexing": "arrow.triangle.2.circlepath",
         "down": "exclamationmark.triangle"}


def pid_file(paths: Paths) -> Path:
    return paths.run / "menubar.pid"


# ── what is going on ──────────────────────────────────────────────────────────────────────────────────────────────

def installed_version() -> str:
    """The version on disk now (an upgrade replaces the files under a running process)."""
    try:
        text = (Path(__file__).with_name("__init__.py")).read_text(encoding="utf-8")
        m = re.search(r'^__version__\s*=\s*"([^"]+)"', text, re.M)
        return m.group(1) if m else __version__
    except OSError:
        return __version__


def needs_relaunch(loaded: str = __version__, on_disk: str | None = None) -> bool:
    return (on_disk or installed_version()) != loaded


def snapshot(paths: Paths) -> dict[str, Any]:
    """The state the icon and the menu are made from.  Asks the daemons and the dashboard; never starts any."""
    from .ui import server

    search = client.ping(paths, "search")
    indexer = client.ping(paths, "indexer")
    snap: dict[str, Any] = {"search": search, "indexer": indexer, "job": None, "dashboard_port": 0,
                            "dashboard_version": "", "installed": installed_version()}
    if indexer:
        r = api.index_status(paths, docs=0, client="cli")
        job = (r.get("job") or {}) if r.get("ok") else {}
        if r.get("running") and job:
            snap["job"] = {"phase": (job.get("progress") or {}).get("phase", ""),
                           "done": (job.get("progress") or {}).get("done"),
                           "total": (job.get("progress") or {}).get("total"), "id": job.get("id", "")}
    port = server.running_port(paths)
    snap["dashboard_port"] = port
    if port:
        snap["dashboard_version"] = (server._ping(port) or {}).get("version", "")
    return snap


def icon_state(snap: dict[str, Any]) -> str:
    search, indexer = snap.get("search"), snap.get("indexer")
    if not search or not indexer:
        return "down"
    if snap.get("job"):
        return "indexing"
    return "ready" if search.get("state") == "ready" else "warming"


def status_lines(snap: dict[str, Any]) -> list[str]:
    """Short lines for the top of the menu."""
    search, indexer, job = snap.get("search"), snap.get("indexer"), snap.get("job")
    if not search:
        lines = ["Search daemon: not running"]
    elif search.get("state") == "ready":
        lines = [f"Search ready · {search.get('chunks', 0):,} chunks"]
    else:
        lines = [f"Search {str(search.get('state', 'starting')).replace('_', ' ')}…"]
    if not indexer:
        lines.append("Indexer: not running")
    elif job:
        n = f" {job['done']} of {job['total']}" if job.get("total") else ""
        lines.append(f"Indexing: {job.get('phase') or 'starting'}{n}")
    else:
        lines.append("Indexer idle")
    old = sorted({d.get("version", "") for d in (search, indexer) if d and d.get("version") not in ("", snap["installed"])})
    if old:
        lines.append(f"Running {', '.join(old)}; installed {snap['installed']}: use Restart everything")
    return lines


def menu_model(snap: dict[str, Any]) -> list[dict[str, Any]]:
    """The right-click menu as data: ``{"id", "title", "enabled", "confirm"}`` items and ``"-"`` separators.  ``confirm`` is
    a question to ask first (a restart or a stop would cancel the indexing run in progress)."""
    indexing = bool(snap.get("job"))
    up = bool(snap.get("search") or snap.get("indexer"))
    cancel_note = "An indexing run is in progress; this cancels it (documents already finished are kept). Continue?" \
        if indexing else ""
    items: list[dict[str, Any]] = [{"id": "", "title": line, "enabled": False} for line in status_lines(snap)]
    items += ["-",
              {"id": "open_dashboard", "title": "Open dashboard", "enabled": True},
              {"id": "index_new", "title": "Index new documents now", "enabled": bool(snap.get("indexer")) and not indexing},
              {"id": "cancel_index", "title": "Cancel indexing", "enabled": indexing}, "-",
              {"id": "restart_all", "title": "Restart everything (daemons and dashboard)", "enabled": True,
               "confirm": cancel_note},
              {"id": "restart_dashboard", "title": "Restart dashboard", "enabled": True},
              {"id": "start_daemons", "title": "Start daemons", "enabled": not (snap.get("search") and snap.get("indexer"))},
              {"id": "stop_daemons", "title": "Stop daemons", "enabled": up, "confirm": cancel_note}, "-",
              {"id": "open_data", "title": "Open data folder", "enabled": True},
              {"id": "open_logs", "title": "Open logs folder", "enabled": True}, "-",
              {"id": "quit", "title": "Quit menu-bar item", "enabled": True}]
    return items


# ── what the items do ─────────────────────────────────────────────────────────────────────────────────────────────

def _dashboard_url(paths: Paths) -> str:
    from .ui import server

    port = server.running_port(paths)
    return f"http://127.0.0.1:{port}/?token={server.load_token(paths)}" if port else ""


def open_dashboard(paths: Paths, *, opener=None) -> str:
    """Open the dashboard, starting it when it is not running and replacing it when it runs another version."""
    from .ui import server

    opener = opener or server._open
    port = server.running_port(paths)
    if port and (server._ping(port) or {}).get("version") == installed_version() == __version__:
        opener(_dashboard_url(paths))
        return "opened"
    if port:
        server.stop_running(paths)
    rc = server.run(paths, port=server.DEFAULT_PORT, open_browser=False, detach=True)
    if rc != 0:
        return "the dashboard did not start (see the ui log)"
    opener(_dashboard_url(paths))
    return "started and opened" if not port else "restarted and opened"


def restart_dashboard(paths: Paths, *, open_browser: bool = False, opener=None) -> str:
    from .ui import server

    was = server.running_port(paths)
    if was:
        server.stop_running(paths)
    rc = server.run(paths, port=server.DEFAULT_PORT, open_browser=False, detach=True)
    if rc != 0:
        return "the dashboard did not start (see the ui log)"
    if open_browser and was:
        (opener or server._open)(_dashboard_url(paths))
    return "dashboard restarted"


def restart_everything(paths: Paths) -> str:
    api.daemon_restart(paths, "all")
    msg = restart_dashboard(paths)
    return "daemons restarted; " + msg


def act(action: str, paths: Paths) -> str:
    """Do menu item *action*; a short message about the outcome."""
    if action == "open_dashboard":
        return open_dashboard(paths)
    if action == "restart_dashboard":
        return restart_dashboard(paths)
    if action == "restart_all":
        return restart_everything(paths)
    if action == "start_daemons":
        api.daemon_start(paths, "all")
        return "daemons started"
    if action == "stop_daemons":
        api.daemon_stop(paths, "all")
        return "daemons stopped"
    if action == "index_new":
        r = api.index_start(paths, mode="new", client="cli")
        return "indexing started" if r.get("ok") else str(r.get("error", "could not start indexing"))
    if action == "cancel_index":
        r = api.index_cancel(paths, client="cli")
        return "indexing cancelled" if r.get("ok") else str(r.get("error", "could not cancel"))
    if action in ("open_data", "open_logs"):
        folder = paths.home if action == "open_data" else paths.run
        subprocess.run(["open", str(folder)], check=False)
        return "opened"
    return ""


# ── one instance, and stopping it ─────────────────────────────────────────────────────────────────────────────────

def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def running_pid(paths: Paths) -> int:
    try:
        pid = int(pid_file(paths).read_text().strip())
    except (OSError, ValueError):
        return 0
    return pid if pid != os.getpid() and _alive(pid) else 0


def stop_running(paths: Paths) -> str:
    pid = running_pid(paths)
    if not pid:
        pid_file(paths).unlink(missing_ok=True)
        return "the menu-bar item is not running"
    os.kill(pid, signal.SIGTERM)
    for _ in range(50):
        if not _alive(pid):
            break
        time.sleep(0.1)
    pid_file(paths).unlink(missing_ok=True)
    return f"stopped the menu-bar item (pid {pid})"


# ── the icon and the menu (AppKit) ────────────────────────────────────────────────────────────────────────────────

def run(paths: Paths | None = None) -> int:
    paths = paths or get_paths()
    if not machine.menubar_possible():
        print("the menu-bar item needs a Mac", file=sys.stderr)
        return 1
    if running_pid(paths):
        print("the menu-bar item is already running", file=sys.stderr)
        return 0
    try:
        import objc
        from AppKit import (NSAlert, NSApplication, NSApplicationActivationPolicyAccessory, NSEvent,
                            NSEventMaskLeftMouseUp, NSEventMaskRightMouseUp, NSEventModifierFlagControl,
                            NSEventTypeRightMouseUp, NSImage, NSMenu, NSMenuItem, NSStatusBar,
                            NSVariableStatusItemLength)
        from Foundation import NSObject, NSTimer
    except ImportError as exc:
        print(f"the menu-bar item needs PyObjC (installed with rag-search on a Mac): {exc}", file=sys.stderr)
        return 1
    paths.run.mkdir(parents=True, exist_ok=True)
    pid_file(paths).write_text(str(os.getpid()))
    state: dict[str, Any] = {"snap": {"search": None, "indexer": None, "job": None, "installed": __version__},
                             "busy": False, "note": ""}

    def refresh() -> None:
        try:
            state["snap"] = snapshot(paths)
        except Exception:  # noqa: BLE001 - a failed look keeps the last picture
            pass
        state["busy"] = False

    def background(fn, *args) -> None:
        threading.Thread(target=fn, args=args, daemon=True).start()

    class App(NSObject):
        def applicationDidFinishLaunching_(self, _note):
            self.item = NSStatusBar.systemStatusBar().statusItemWithLength_(NSVariableStatusItemLength)
            button = self.item.button()
            button.setTarget_(self)
            button.setAction_(b"clicked:")
            button.sendActionOn_(NSEventMaskLeftMouseUp | NSEventMaskRightMouseUp)
            self.shown = ""
            self.set_icon("down")
            background(refresh)
            self.timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
                REFRESH_S, self, b"tick:", None, True)

        def set_icon(self, name):
            if name == self.shown:
                return
            self.shown = name
            button = self.item.button()
            img = NSImage.imageWithSystemSymbolName_accessibilityDescription_(ICONS[name], "rag-search")
            if img is not None:
                img.setTemplate_(True)
                button.setImage_(img)
                button.setTitle_("")
            else:
                button.setTitle_("RAG" if name == "ready" else "RAG…")

        def tick_(self, _timer):
            if needs_relaunch():
                pid_file(paths).unlink(missing_ok=True)
                os.execv(sys.executable, [sys.executable, "-m", "rag_search.menubar"])
            self.set_icon(icon_state(state["snap"]))
            self.item.button().setToolTip_("\n".join(status_lines(state["snap"])))
            if not state["busy"]:
                state["busy"] = True
                background(refresh)

        def clicked_(self, sender):
            ev = NSEvent.currentEvent() if hasattr(NSEvent, "currentEvent") else NSApplication.sharedApplication().currentEvent()
            right = ev is not None and (ev.type() == NSEventTypeRightMouseUp or bool(ev.modifierFlags() & NSEventModifierFlagControl))
            if not right:
                background(act, "open_dashboard", paths)
                return
            menu = NSMenu.alloc().init()
            menu.setAutoenablesItems_(False)
            for entry in menu_model(state["snap"]):
                if entry == "-":
                    menu.addItem_(NSMenuItem.separatorItem())
                    continue
                mi = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(entry["title"], b"chosen:" if entry["id"] else None, "")
                mi.setTarget_(self)
                mi.setEnabled_(bool(entry["enabled"]))
                mi.setRepresentedObject_(entry)
                menu.addItem_(mi)
            self.item.setMenu_(menu)
            self.item.button().performClick_(None)
            self.item.setMenu_(None)

        def chosen_(self, sender):
            entry = sender.representedObject()
            action = entry["id"]
            if action == "quit":
                pid_file(paths).unlink(missing_ok=True)
                NSApplication.sharedApplication().terminate_(None)
                return
            if entry.get("confirm"):
                alert = NSAlert.alloc().init()
                alert.setMessageText_(entry["title"])
                alert.setInformativeText_(entry["confirm"])
                alert.addButtonWithTitle_("Continue")
                alert.addButtonWithTitle_("Cancel")
                if alert.runModal() != 1000:        # NSAlertFirstButtonReturn
                    return
            background(lambda: (act(action, paths), refresh()))

    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)      # no Dock icon, no app menu
    delegate = App.alloc().init()
    app.setDelegate_(delegate)
    signal.signal(signal.SIGTERM, lambda *_: (pid_file(paths).unlink(missing_ok=True), os._exit(0)))
    objc.setVerbose(False)
    app.run()
    return 0


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="the rag-search menu-bar item")
    ap.add_argument("--stop", action="store_true", help="stop a running menu-bar item")
    a = ap.parse_args(argv)
    if a.stop:
        print(stop_running(get_paths()))
        return 0
    return run()


if __name__ == "__main__":
    sys.exit(main())
