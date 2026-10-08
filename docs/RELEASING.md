# Releasing rag-search

A release is built and published by GitHub Actions from a clean checkout; nothing is ever uploaded from a laptop
(a local build contains files that are gitignored on purpose). The package version is written once, in
`src/rag_search/__init__.py`; the tag `vX.Y.Z` only labels the commit that has it, and the workflow refuses a tag that
does not equal the version.

## One-time setup

1. **PyPI and TestPyPI accounts** with two-factor authentication.
2. **Trusted publishers** (no token is stored anywhere). On pypi.org and on test.pypi.org: *Your projects > Publishing >
   Add a new pending publisher* with project `rag-search-local`, owner `rahul-fiske`, repository `rag-search`, workflow
   `release.yml` and environment `pypi` (on pypi.org) / `testpypi` (on test.pypi.org).
3. **GitHub environments** (*Settings > Environments*): `testpypi`, and `pypi` with *Required reviewers* set to you, so
   that nothing reaches PyPI without a click.
4. **Secret `RELEASE_DENY_WORDS`** (optional but recommended; *Settings > Secrets and variables > Actions*): comma-separated
   words that must not appear in any released file (names of employers, internal tools). Without it the build warns and
   skips that check; the files are built from a clean checkout either way.
5. **Tag protection** (optional; *Settings > Rules > Rulesets*): only you may create tags `v*`; protect `main` and `rag_1.0`
   against force pushes and deletion.

## A release

1. On the branch to release, set `__version__` to the release number (`1.2.0`, or `1.2.0rc1` for a candidate): no `.dev`.
2. Run the tests (`PYTHONPATH=src python -m unittest discover -s tests/portable -t .`, and tier B and C on the Mac when
   conversion changed), update `README.md`/`ARCHITECTURE.md`, merge to `main` through a pull request (CI must be green).
3. Tag that commit and push the tag:
   ```bash
   git tag -a v1.2.0 -m "rag-search 1.2.0" && git push origin v1.2.0
   ```
4. The **Release** workflow runs: tag equals version, build, `twine check`, the wheel and source archive checked
   (`scripts/check_release.py`), the leak guard (`scripts/release_guard.py`), an install of the wheel on macOS and Linux,
   then **TestPyPI**. Approve the `pypi` environment when it waits, and it publishes to **PyPI**, installs the release
   from PyPI on a Mac, and creates the **GitHub Release** with the files attached and generated notes.
5. On `main`, set the next development version (`1.2.1.dev0`).

A PyPI version can never be replaced: a mistake costs a new version number (yank the bad one on pypi.org).
To rehearse, run the workflow by hand (*Actions > Release > Run workflow*): it builds and publishes a development
version to TestPyPI only.

## Hotfix of a released line

Fix on `main` first, cherry-pick the commit onto the release branch (`rag_1.0`), set `1.0.1` there, tag `v1.0.1`.
