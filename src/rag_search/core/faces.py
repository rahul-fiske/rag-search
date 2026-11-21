"""Faces in photographs: where each face is, its fingerprint, and the names you give them -- kept apart from the index.

Three layers, from facts to decisions to guesses::

    faces      each face found in a photograph: the photograph's checksum, where the face is, a small picture of it
    vectors    the face's fingerprint (an embedding), one per face and face model: two faces of one person are close
    people     names (with aliases such as "me") and groups (such as "family") -- what you decide, in one place
    labels     your answers: "this face is Avni" or "this face is not Avni"; the only thing you make, never recomputed

Who every other face is -- the guesses -- is worked out from these when it is needed (``people.py``), never stored: a
face takes the person whose labelled faces are closest, when that is clearly the best match.  So renaming a person is one
row, regrouping touches no face, and a better face model computes the vectors again while your labels stay.  Nothing is
written into the photographs or their Markdown except the face count; the names are added when a photograph is found or
read.

Faces are found by OpenCV's YuNet and fingerprinted by SFace (128 numbers per face, compared by cosine similarity), two
small models from the OpenCV model zoo that run on the CPU with the OpenCV already installed.  They are downloaded on
request (``rag-search people setup``, or the People tab) into ``<home>/models/faces`` and checked against the checksums
below.  The store is ``<home>/people.sqlite``: outside the indexer's workspace, so a rebuild of the index keeps it.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import os
import sqlite3
import threading
import time
import urllib.request
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import numpy as np

from ..paths import Paths

DETECTOR = "yunet-2023mar"
RECOGNIZER = "sface-2021dec"
_ZOO = "https://github.com/opencv/opencv_zoo/raw/main/models/"
MODEL_FILES = {   # name: (file, url, sha256, bytes)
    DETECTOR: ("face_detection_yunet_2023mar.onnx", _ZOO + "face_detection_yunet/face_detection_yunet_2023mar.onnx",
               "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4", 232589),
    RECOGNIZER: ("face_recognition_sface_2021dec.onnx", _ZOO + "face_recognition_sface/face_recognition_sface_2021dec.onnx",
                 "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79", 38696353),
}
LONG_SIDE = 1600            # pixels: faces are looked for in the photograph at this size
FIND_SCORE = 0.8            # YuNet's confidence for a face to be kept
THUMB = 112                 # pixels: the small picture of each face the People tab shows
DIM = 128


def model_dir(paths: Paths) -> Path:
    return paths.home / "models" / "faces"


def store_file(paths: Paths) -> Path:
    return paths.home / "people.sqlite"


def missing_models(paths: Paths) -> list[str]:
    return [name for name, (f, _u, _s, size) in MODEL_FILES.items()
            if not (model_dir(paths) / f).is_file() or (model_dir(paths) / f).stat().st_size != size]


def available(paths: Paths) -> str:
    """"" when faces can be found on this machine, else why not."""
    try:
        import cv2
    except ImportError:
        return "finding faces needs OpenCV (cv2), which is not installed"
    if not (hasattr(cv2, "FaceDetectorYN") and hasattr(cv2, "FaceRecognizerSF")):
        return "this OpenCV has no face detector (FaceDetectorYN); it needs OpenCV 4.5.4 or later"
    if missing_models(paths):
        return "the face models are not downloaded (rag-search people setup, or the People tab)"
    return ""


def download(paths: Paths, log: Callable[[str], None] = lambda m: None) -> list[str]:
    """Fetch the face models that are missing (about 38 MB), each checked against its checksum.  Returns what was fetched."""
    d = model_dir(paths)
    d.mkdir(parents=True, exist_ok=True)
    got = []
    for name in missing_models(paths):
        f, url, sha, size = MODEL_FILES[name]
        log(f"downloading {f} ({size / 2**20:.1f} MB) from the OpenCV model zoo")
        tmp = d / (f + ".partial")
        h = hashlib.sha256()
        with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as out:   # noqa: S310 - a fixed https URL
            while chunk := r.read(1 << 20):
                h.update(chunk)
                out.write(chunk)
        if h.hexdigest() != sha:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"{f}: the download does not match its checksum; nothing was installed")
        os.replace(tmp, d / f)
        got.append(f)
    return got


# ── finding faces ────────────────────────────────────────────────────────────

def load_upright(src: Path, long_side: int = LONG_SIDE) -> Any:
    """The photograph upright (EXIF orientation applied), in RGB, no larger than *long_side*, as a PIL image."""
    from PIL import Image, ImageOps

    with contextlib.suppress(Exception):
        import pillow_heif

        pillow_heif.register_heif_opener()
    with Image.open(src) as im:
        im.draft("RGB", (long_side, long_side))               # a JPEG is decoded at a fraction of its size: quicker
        im = ImageOps.exif_transpose(im).convert("RGB")
    im.thumbnail((long_side, long_side))
    return im


class Finder:
    """YuNet (where the faces are, with five points on each) and SFace (the fingerprint of each face, from a crop lined up
    on those points) -- the pair is made to work together.  One per process; not for threads."""

    def __init__(self, paths: Paths):
        import cv2

        with contextlib.suppress(Exception):           # OpenCV's notes on its graph engine are not for the log
            cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
        d = model_dir(paths)
        self.cv2 = cv2
        self.detector = cv2.FaceDetectorYN.create(str(d / MODEL_FILES[DETECTOR][0]), "", (320, 320), FIND_SCORE, 0.3, 5000)
        self.recognizer = cv2.FaceRecognizerSF.create(str(d / MODEL_FILES[RECOGNIZER][0]), "")

    def faces(self, src: Path) -> list[dict[str, Any]]:
        """Each face in *src*: its box as a fraction of the upright picture (x, y, w, h), YuNet's score, its size in
        pixels (the shorter side, in the picture as searched), its fingerprint (unit length) and a small JPEG of it."""
        im = load_upright(src)
        rgb = np.asarray(im)
        bgr = np.ascontiguousarray(rgb[:, :, ::-1])
        h, w = bgr.shape[:2]
        self.detector.setInputSize((w, h))
        _ok, dets = self.detector.detect(bgr)
        out = []
        for det in (dets if dets is not None else []):
            aligned = self.recognizer.alignCrop(bgr, det)
            vec = np.asarray(self.recognizer.feature(aligned), dtype=np.float32).reshape(-1)
            vec /= float(np.linalg.norm(vec)) or 1.0
            x, y, fw, fh = (float(v) for v in det[:4])
            out.append({"x": max(0.0, x / w), "y": max(0.0, y / h), "w": fw / w, "h": fh / h, "score": float(det[-1]),
                        "px": int(min(fw, fh)), "vec": vec, "thumb": _thumb(im, x, y, fw, fh)})
        return out


def _thumb(im: Any, x: float, y: float, w: float, h: float) -> bytes:
    """A square JPEG of the face with some room around it."""
    side = max(w, h) * 1.5
    cx, cy = x + w / 2, y + h / 2
    box = (int(max(0, cx - side / 2)), int(max(0, cy - side / 2)),
           int(min(im.width, cx + side / 2)), int(min(im.height, cy + side / 2)))
    crop = im.crop(box)
    crop.thumbnail((THUMB, THUMB))
    buf = io.BytesIO()
    crop.save(buf, "JPEG", quality=82)
    return buf.getvalue()


_FINDER: Finder | None = None
_FINDER_LOCK = threading.Lock()


def finder(paths: Paths) -> Finder:
    global _FINDER
    with _FINDER_LOCK:
        if _FINDER is None:
            _FINDER = Finder(paths)
        return _FINDER


def for_photo(paths: Paths, src: Path, sha: str, *, store: "Store | None" = None) -> int | None:
    """Find and remember the faces of one photograph (a photograph already looked at is not looked at again); returns how
    many faces it has, or None when faces cannot be found here.  Never fails a photograph."""
    if available(paths):
        return None
    own = store is None
    st = store or Store(paths)
    try:
        n = st.photo_faces(sha)
        if n is not None:
            st.seen_at(sha, src)
            return n
        found = finder(paths).faces(src)
        st.add_photo(sha, src, found)
        return len(found)
    except Exception:  # noqa: BLE001 - no faces is better than a failed photograph
        return None
    finally:
        if own:
            st.close()


# ── the store ────────────────────────────────────────────────────────────────

SCHEMA = """
CREATE TABLE IF NOT EXISTS photos (sha TEXT PRIMARY KEY, src TEXT NOT NULL, detector TEXT NOT NULL, faces INTEGER NOT NULL,
                                   ts REAL NOT NULL);
CREATE INDEX IF NOT EXISTS photos_src ON photos (src);
CREATE TABLE IF NOT EXISTS faces (id INTEGER PRIMARY KEY, sha TEXT NOT NULL, x REAL, y REAL, w REAL, h REAL, score REAL,
                                  px INTEGER, thumb BLOB);
CREATE INDEX IF NOT EXISTS faces_sha ON faces (sha);
CREATE TABLE IF NOT EXISTS vectors (face INTEGER NOT NULL, model TEXT NOT NULL, vec BLOB NOT NULL, PRIMARY KEY (face, model));
CREATE TABLE IF NOT EXISTS people (id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE COLLATE NOCASE,
                                   aliases TEXT NOT NULL DEFAULT '', ts REAL NOT NULL);
CREATE TABLE IF NOT EXISTS groups (id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE COLLATE NOCASE);
CREATE TABLE IF NOT EXISTS members (grp INTEGER NOT NULL, person INTEGER NOT NULL, PRIMARY KEY (grp, person));
CREATE TABLE IF NOT EXISTS labels (face INTEGER NOT NULL, person INTEGER NOT NULL, yes INTEGER NOT NULL, ts REAL NOT NULL,
                                   PRIMARY KEY (face, person));
"""


class StoreError(ValueError):
    """A change that cannot be made; the message says why."""


class Store:
    """``<home>/people.sqlite``: the indexer adds faces, the dashboard adds names and labels, search reads."""

    def __init__(self, paths: Paths):
        self.path = store_file(paths)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.con = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
        self.con.execute("PRAGMA journal_mode=WAL")
        self.con.executescript(SCHEMA)

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self.con.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # -- faces (facts) -------------------------------------------------------
    def photo_faces(self, sha: str) -> int | None:
        row = self.con.execute("SELECT faces FROM photos WHERE sha = ? AND detector = ?", (sha, DETECTOR)).fetchone()
        return None if row is None else int(row[0])

    def seen_at(self, sha: str, src: Path) -> None:
        with self.con:
            self.con.execute("UPDATE photos SET src = ? WHERE sha = ? AND src != ?", (str(src), sha, str(src)))

    def add_photo(self, sha: str, src: Path, found: Iterable[dict[str, Any]], model: str = RECOGNIZER) -> None:
        with self.con:                  # once per photograph: another process may have just looked at it too
            if self.con.execute("SELECT 1 FROM photos WHERE sha = ?", (sha,)).fetchone():
                return
            found = list(found)
            for f in found:
                cur = self.con.execute("INSERT INTO faces (sha, x, y, w, h, score, px, thumb) VALUES (?,?,?,?,?,?,?,?)",
                                       (sha, f["x"], f["y"], f["w"], f["h"], f["score"], f["px"], f.get("thumb")))
                self.con.execute("INSERT INTO vectors (face, model, vec) VALUES (?,?,?)",
                                 (cur.lastrowid, model, np.asarray(f["vec"], dtype=np.float32).tobytes()))
            self.con.execute("INSERT INTO photos (sha, src, detector, faces, ts) VALUES (?,?,?,?,?)",
                             (sha, str(src), DETECTOR, len(found), time.time()))

    def thumb(self, face: int) -> bytes | None:
        row = self.con.execute("SELECT thumb FROM faces WHERE id = ?", (face,)).fetchone()
        return row[0] if row and row[0] else None

    def counts(self) -> dict[str, int]:
        one = lambda q: int(self.con.execute(q).fetchone()[0])    # noqa: E731
        return {"photos": one("SELECT COUNT(*) FROM photos"), "with_faces": one("SELECT COUNT(*) FROM photos WHERE faces > 0"),
                "faces": one("SELECT COUNT(*) FROM faces"), "labelled": one("SELECT COUNT(DISTINCT face) FROM labels WHERE yes = 1"),
                "people": one("SELECT COUNT(*) FROM people"), "groups": one("SELECT COUNT(*) FROM groups")}

    def vectors(self, model: str = RECOGNIZER) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray, np.ndarray]:
        """Every face with a fingerprint by *model*: (ids, vectors n x 128, photograph checksums, size in px, score)."""
        rows = self.con.execute("SELECT f.id, v.vec, f.sha, f.px, f.score FROM faces f JOIN vectors v ON v.face = f.id "
                                "WHERE v.model = ? ORDER BY f.id", (model,)).fetchall()
        if not rows:
            return (np.zeros(0, dtype=np.int64), np.zeros((0, DIM), dtype=np.float32), [], np.zeros(0), np.zeros(0))
        ids = np.array([r[0] for r in rows], dtype=np.int64)
        vecs = np.frombuffer(b"".join(r[1] for r in rows), dtype=np.float32).reshape(len(rows), -1)
        return ids, vecs, [r[2] for r in rows], np.array([r[3] or 0 for r in rows]), np.array([r[4] or 0.0 for r in rows])

    def sources(self, shas: Iterable[str]) -> dict[str, str]:
        out: dict[str, str] = {}
        shas = list(shas)
        for i in range(0, len(shas), 500):
            part = shas[i:i + 500]
            q = f"SELECT sha, src FROM photos WHERE sha IN ({','.join('?' * len(part))})"
            out.update(dict(self.con.execute(q, part).fetchall()))
        return out

    # -- people and groups (names) ---------------------------------------------
    def people(self) -> list[dict[str, Any]]:
        return [{"id": r[0], "name": r[1], "aliases": [a for a in r[2].split(",") if a]}
                for r in self.con.execute("SELECT id, name, aliases FROM people ORDER BY name COLLATE NOCASE")]

    def groups(self) -> list[dict[str, Any]]:
        out = []
        for gid, name in self.con.execute("SELECT id, name FROM groups ORDER BY name COLLATE NOCASE").fetchall():
            ids = [r[0] for r in self.con.execute("SELECT person FROM members WHERE grp = ?", (gid,))]
            out.append({"id": gid, "name": name, "members": ids})
        return out

    def _taken(self, word: str, *, person: int | None = None, group: int | None = None) -> str:
        """What else is already called *word* (a person's name or alias, or a group), or ""."""
        w = word.casefold()
        for p in self.people():
            if p["id"] != person and (p["name"].casefold() == w or w in (a.casefold() for a in p["aliases"])):
                return f"person {p['name']!r}"
        for g in self.groups():
            if g["id"] != group and g["name"].casefold() == w:
                return f"group {g['name']!r}"
        return ""

    @staticmethod
    def _clean(name: str) -> str:
        name = " ".join(str(name or "").split())
        if not name or len(name) > 80 or any(ch in name for ch in ",\n\t"):
            raise StoreError("a name is 1 to 80 characters, without commas")
        return name

    def person_id(self, name: str, *, create: bool = False) -> int | None:
        row = self.con.execute("SELECT id FROM people WHERE name = ? COLLATE NOCASE", (name,)).fetchone()
        if row:
            return int(row[0])
        if not create:
            return None
        name = self._clean(name)
        other = self._taken(name)
        if other:
            raise StoreError(f"{name!r} is already the name of {other}")
        with self.con:
            return int(self.con.execute("INSERT INTO people (name, ts) VALUES (?, ?)", (name, time.time())).lastrowid)

    def rename(self, person: int, name: str, *, merge: bool = False) -> dict[str, Any]:
        """A new name for *person*.  When another person already has it, the two are one person: with *merge* the labels
        of *person* move to the other and *person* goes; without it nothing changes and the answer says so."""
        name = self._clean(name)
        other = self.person_id(name)
        if other is not None and other != person:
            if not merge:
                return {"merge_needed": True, "into": other}
            with self.con:
                self.con.execute("DELETE FROM labels WHERE person = ? AND face IN (SELECT face FROM labels WHERE person = ?)",
                                 (person, other))
                self.con.execute("UPDATE labels SET person = ? WHERE person = ?", (other, person))
                self.con.execute("INSERT OR IGNORE INTO members (grp, person) SELECT grp, ? FROM members WHERE person = ?",
                                 (other, person))
                self.con.execute("DELETE FROM members WHERE person = ?", (person,))
                old = self.con.execute("SELECT aliases FROM people WHERE id = ?", (person,)).fetchone()
                self.con.execute("DELETE FROM people WHERE id = ?", (person,))
            if old and old[0]:
                mine = next(p for p in self.people() if p["id"] == other)["aliases"]
                self.set_aliases(other, mine + [a for a in old[0].split(",") if a])
            return {"merged_into": other}
        taken = self._taken(name, person=person)
        if taken:
            raise StoreError(f"{name!r} is already the name of {taken}")
        with self.con:
            self.con.execute("UPDATE people SET name = ? WHERE id = ?", (name, person))
        return {"renamed": person}

    def set_aliases(self, person: int, aliases: Iterable[str]) -> list[str]:
        clean: list[str] = []
        for a in aliases:
            a = " ".join(str(a).split())
            if not a or a.casefold() in (c.casefold() for c in clean):
                continue
            a = self._clean(a)
            taken = self._taken(a, person=person)
            if taken:
                raise StoreError(f"{a!r} is already the name of {taken}")
            clean.append(a)
        with self.con:
            self.con.execute("UPDATE people SET aliases = ? WHERE id = ?", (",".join(clean), person))
        return clean

    def delete_person(self, person: int) -> None:
        with self.con:
            self.con.execute("DELETE FROM labels WHERE person = ?", (person,))
            self.con.execute("DELETE FROM members WHERE person = ?", (person,))
            self.con.execute("DELETE FROM people WHERE id = ?", (person,))

    def group_id(self, name: str, *, create: bool = False) -> int | None:
        row = self.con.execute("SELECT id FROM groups WHERE name = ? COLLATE NOCASE", (name,)).fetchone()
        if row:
            return int(row[0])
        if not create:
            return None
        name = self._clean(name)
        taken = self._taken(name)
        if taken:
            raise StoreError(f"{name!r} is already the name of {taken}")
        with self.con:
            return int(self.con.execute("INSERT INTO groups (name) VALUES (?)", (name,)).lastrowid)

    def rename_group(self, group: int, name: str) -> None:
        name = self._clean(name)
        taken = self._taken(name, group=group)
        if taken:
            raise StoreError(f"{name!r} is already the name of {taken}")
        with self.con:
            self.con.execute("UPDATE groups SET name = ? WHERE id = ?", (name, group))

    def set_members(self, group: int, people: Iterable[int]) -> None:
        with self.con:
            self.con.execute("DELETE FROM members WHERE grp = ?", (group,))
            self.con.executemany("INSERT OR IGNORE INTO members (grp, person) VALUES (?, ?)", [(group, int(p)) for p in people])

    def delete_group(self, group: int) -> None:
        with self.con:
            self.con.execute("DELETE FROM members WHERE grp = ?", (group,))
            self.con.execute("DELETE FROM groups WHERE id = ?", (group,))

    # -- labels (your answers) --------------------------------------------------
    def label(self, faces: Iterable[int], person: int) -> int:
        """These faces are *person* (a face is one person: another "yes" on it goes)."""
        faces = [int(f) for f in faces]
        now = time.time()
        with self.con:
            for f in faces:
                self.con.execute("DELETE FROM labels WHERE face = ? AND (yes = 1 OR person = ?)", (f, person))
                self.con.execute("INSERT INTO labels (face, person, yes, ts) VALUES (?, ?, 1, ?)", (f, person, now))
        return len(faces)

    def reject(self, face: int, person: int) -> None:
        """This face is not *person*."""
        with self.con:
            self.con.execute("INSERT OR REPLACE INTO labels (face, person, yes, ts) VALUES (?, ?, 0, ?)", (face, person, time.time()))

    def unlabel(self, face: int) -> None:
        with self.con:
            self.con.execute("DELETE FROM labels WHERE face = ?", (face,))

    def labels(self) -> tuple[dict[int, int], set[tuple[int, int]]]:
        """({face: person} for "yes", {(face, person)} for "no")."""
        yes: dict[int, int] = {}
        no: set[tuple[int, int]] = set()
        for f, p, y in self.con.execute("SELECT face, person, yes FROM labels"):
            if y:
                yes[f] = p
            else:
                no.add((f, p))
        return yes, no

    def version(self) -> tuple[int, ...]:
        """Changes when anything in the store changes (what a cached view of it is keyed by)."""
        out = []
        for suffix in ("", "-wal"):
            with contextlib.suppress(OSError):
                st = os.stat(str(self.path) + suffix)
                out += [st.st_mtime_ns, st.st_size]
        return tuple(out)
