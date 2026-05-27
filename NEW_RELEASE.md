# tsai — Release Runbook

Maintainer runbook covering every step from "branch is ready" to "PyPI is live". Follow top-to-bottom for a new release.

---

## 0. Pre-flight

Your feature branch should already be:
- Up to date with `main`
- Have all intended code changes committed
- (Ideally) have green CI on the most recent push

Pick the env you'll use throughout (must match the new dep matrix if you bumped deps):
```bash
conda activate py311t27
```

If you bumped any deps in `pyproject.toml`, reinstall first:
```bash
pip install -U -e ".[dev]"
```

---

## 1. Local validation

### 1a. Strip and re-export notebooks
```bash
nbdev-clean
nbdev-export
```

Then check nothing else moved unexpectedly:
```bash
git status -s | head
```

If `nbdev-export` produced changes you didn't expect, investigate before continuing — usually means a notebook was edited but never re-exported.

### 1b. Run tests

**macOS (Apple Silicon)** — use the in-process runner to avoid `MTLCompilerService` crashes in the parallel runner:
```bash
nbdev-test --n_workers 0 --do_print --timing
```

**Linux / x86**:
```bash
nbdev-prepare        # bundles export + test + clean
```

All tests must pass before continuing. If you only want to re-run a subset:
```bash
nbdev-test --n_workers 0 --file_re '^(004_|005_|009_)'
```

### 1c. Build the docs (recommended for major releases)
```bash
nbdev-docs
```

Common failure modes and fixes:
- **Class with malformed signature** (closing `):` at column 0 inside a method) → reindent so `):` matches the `def`.
- **Function/method with numpy array as default value** (e.g. `alphas=np.logspace(...)`) → fastcore's `docments` chokes on `np_array != empty`. Switch to a tuple literal (e.g. `alphas=(0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)`).
- **Quarto missing** → `nbdev-install-quarto`.

If the build's successful you'll see `Output created: _docs/...`. You can preview with `nbdev-preview`.

---

## 2. Verify version + update CHANGELOG

### 2a. Verify version
The version is **not bumped here**. The post-release `nbdev-bump-version` step from the *previous* release (see §7) already advanced `tsai/__init__.py` to the upcoming patch version. The only version bump in the release flow is §7 — setting up the *next* dev cycle.

Confirm `tsai/__init__.py` matches the target:
```bash
grep "__version__" tsai/__init__.py
```

If it already says `X.Y.Z`: proceed to §2b.

If it doesn't match (e.g., you're cutting a minor/major like 1.0.2 → 1.1.0 but §7 of the prior release only patch-bumped to 1.0.2): manually edit `tsai/__init__.py` to `X.Y.Z` now, before continuing. `pyproject.toml` reads the version dynamically — no other file to touch.

### 2b. Update CHANGELOG.md
Add a new section at the top (after `<!-- do not remove -->`):

```markdown
## X.Y.Z

### Upgrade guide        ← only for major releases / breaking changes
- **PyPI users:** ...
- **Conda users:** ...
- **Contributors with an existing checkout:** ...

### Breaking changes     ← only when applicable
- ...

### New Features
- ...

### Bugs Squashed
- ...
```

Keep entries actionable ("if you do X, switch to Y") rather than just descriptive ("X changed").

### 2c. For a major release, also update README install section if anything user-visible changed
The README is generated from `nbs/index.ipynb`. Edit the notebook, then:
```bash
nbdev-readme
```

---

## 3. Commit and push to branch

```bash
git add -A
git commit -m "release X.Y.Z"          # use a multi-line message for majors
git push origin <branch>
```

Wait for CI to go green on the branch.

---

## 4. Open PR and merge to main

### 4a. Create the PR
The most reliable way is the GitHub web UI (PAT scope issues can block `gh pr create`):
```
https://github.com/timeseriesAI/tsai/compare/main...<branch>
```

Title: `release X.Y.Z`
Body: short summary + link to the `## X.Y.Z` section of `CHANGELOG.md`. For majors include the upgrade guide highlights.

### 4b. Squash merge
On the PR page, click the dropdown next to "Merge" → choose **"Squash and merge"**.

This collapses all branch commits into a single `release X.Y.Z` commit on `main`. Clean history, easy to revert if needed.

---

## 5. Sync, tag, and create GitHub release

### 5a. Sync local main
```bash
git checkout main
git pull origin main
git log --oneline -3       # confirm the squashed release commit is at the top
```

### 5b. Confirm `quarto-ghp3` deploy succeeded
The deploy workflow fires on push to `main`. Check the Actions tab; if it failed, the docs site won't be up to date.

### 5c. Tag the release
```bash
git tag -a vX.Y.Z -m "tsai X.Y.Z"
git push origin vX.Y.Z
```

### 5d. Create the GitHub release
Open: https://github.com/timeseriesAI/tsai/releases/new

- Tag: select `vX.Y.Z` from the dropdown
- Title: `tsai X.Y.Z`
- Body: paste the `## X.Y.Z` section from `CHANGELOG.md`
- Release label: **Latest**
- **Don't attach binaries** — GitHub auto-attaches source archives, and PyPI is the canonical place for wheel/sdist
- (Optional) tick "Create a discussion for this release" for major versions
- Click **Publish release**

---

## 6. Build and upload to PyPI

### 6a. Clean and build
From the repo root:
```bash
rm -rf dist/ build/ tsai.egg-info/
pip install -U build twine
python -m build
```

This produces:
- `dist/tsai-X.Y.Z-py3-none-any.whl`
- `dist/tsai-X.Y.Z.tar.gz`

### 6b. Verify the build
```bash
twine check dist/*
```

Both should report PASSED. Optionally inspect metadata:
```bash
python -c "
import zipfile
with zipfile.ZipFile('dist/tsai-X.Y.Z-py3-none-any.whl') as z:
    with z.open('tsai-X.Y.Z.dist-info/METADATA') as f:
        for line in f.read().decode().splitlines():
            if line.startswith(('Version', 'Requires-Python', 'Requires-Dist')):
                print(line)
"
```
Check that `Version: X.Y.Z`, `Requires-Python: >=3.10`, and all dep pins look right.

### 6c. Smoke-test the wheel locally (before upload)

PyPI uploads are immutable, so artifact-structure bugs (e.g. #962, where the 1.0.0 wheel was missing every subpackage) must be caught **before** `twine upload`. Install the local wheel into a throwaway venv with `--no-deps` so it's fast — no torch download — and confirm every subpackage is actually bundled:

```bash
python -m venv /tmp/tsai_artifact
/tmp/tsai_artifact/bin/pip install --no-deps dist/*.whl
(cd /tmp && /tmp/tsai_artifact/bin/python -c "
import tsai, importlib.util
import tsai.data, tsai.models, tsai.callback
missing = [m for m in ('tsai.all','tsai.basics','tsai.inference','tsai.optimizer','tsai.tslearner')
             if importlib.util.find_spec(m) is None]
if missing: raise SystemExit(f'missing from wheel: {missing}')
print(f'tsai {tsai.__version__}: wheel structure OK')
")
rm -rf /tmp/tsai_artifact
```

The `cd /tmp` matters: without it, the source-tree `tsai/` directory shadows the installed wheel via cwd-on-sys.path and the check passes even on a broken build.

If this fails, **do not upload** — investigate `[tool.setuptools.packages.find]` in `pyproject.toml`. CI's `verify-build` job catches the same class of bug on every PR; this local step is the last line of defense before the upload becomes immutable.

### 6d. Upload to PyPI
```bash
twine upload dist/*
```

If `~/.pypirc` doesn't have credentials stored:
- Username: `__token__` (literal, with underscores)
- Password: PyPI API token (starts with `pypi-AgEI...`). Project-scoped to `tsai` is best.

Token generation: https://pypi.org/manage/account/token/

### 6e. Verify it's live on PyPI

Confirm via the PyPI JSON API (more reliable than `pip index`, which is cached):
```bash
curl -s https://pypi.org/pypi/tsai/json | python -c "
import sys, json
d = json.load(sys.stdin)
print('Latest:    ', d['info']['version'])
print('Released:  ', d['releases'].get('X.Y.Z', [{}])[0].get('upload_time'))
print('Python req:', d['info']['requires_python'])
print('Project:   ', d['info']['package_url'])
"
```

Expected output: `Latest: X.Y.Z`, a fresh upload timestamp, and the correct Python requirement.

Also check visually:
- https://pypi.org/project/tsai/ — the landing page should show "tsai X.Y.Z" at the top.
- The "Project description" panel should render the README correctly (no broken markdown).
- The "Release history" tab should list X.Y.Z as the newest.

`pip index versions tsai` may keep showing the previous version for a few minutes due to pip's local index cache — that's normal and not a real problem.

### 6f. Smoke-test the published package

Install fresh from PyPI in a throwaway venv to confirm the wheel resolves, dependencies install, and `tsai` imports cleanly:
```bash
python -m venv /tmp/tsai_smoke
source /tmp/tsai_smoke/bin/activate
pip install tsai==X.Y.Z
python -c "
import tsai; print('version:', tsai.__version__)
from tsai.basics import *
print('basics import OK')
from tsai.models.InceptionTime import InceptionTime
print('models import OK')
"
deactivate && rm -rf /tmp/tsai_smoke
```

Things to verify in this output:
- Version matches X.Y.Z
- All imports succeed
- No dependency resolution warnings about conflicting versions
- `pip install` should resolve `torch`, `fastai`, `sklearn`, etc. to versions matching the new constraints in `pyproject.toml`

If this venv smoke test fails, **do not announce the release yet** — investigate. Most likely causes:
- Missing dependency in `pyproject.toml` that worked locally because it was already installed.
- Version pin too restrictive and not satisfiable on a fresh install.
- A bug in a hand-written `tsai/*.py` (e.g. `tsai/imports.py`) that doesn't surface from the editable install in your dev env.

### 6g. Post-release sanity checklist

- [ ] PyPI page shows X.Y.Z as latest
- [ ] GitHub release page shows the `vX.Y.Z` tag with notes
- [ ] `gh-pages` was updated by `quarto-ghp3` (docs site reflects the new version)
- [ ] Throwaway venv smoke test passes
- [ ] CHANGELOG, README, and version (`tsai/__init__.py`) all agree on X.Y.Z

---

## 7. Bump version to start the next development cycle

After the release is confirmed live and tested, immediately bump `tsai/__init__.py` so any subsequent commits to `main` are clearly post-release.

```bash
git checkout main
git pull origin main           # make sure local is at the merged release commit

nbdev-bump-version             # defaults to bumping the patch part (e.g. 1.0.0 → 1.0.1)
                               # use --part 1 for a minor bump, --part 0 for a major bump
```

Verify and commit directly to `main` (small version-bump commits are conventional for this project — see git history for prior `bumped version to ...` commits):

```bash
grep "__version__" tsai/__init__.py     # confirm new version
git add tsai/__init__.py
git commit -m "bumped version to X.Y.Z"
git push origin main
```

That's the final step — the release is done and the repo is ready for the next iteration.

---

## Known gotchas

- **PyPI uploads are immutable.** If something's wrong with X.Y.Z after upload, you can't replace it — you have to release X.Y.Z+1.
- **Mac Metal flake.** The default parallel test runner can fail on Apple Silicon under load. Always use `--n_workers 0` locally. Linux CI is fine without it.
- **GitHub PAT scope.** Fine-grained PATs sometimes lack `pull_requests: write`. If `gh pr create` fails, use the web UI.
- **Doc URL bug in nbdev 3.0.17.** Module docstrings get malformed URLs (`tsaianalysis.html.md` missing a `/`). Cosmetic only; fixed in upstream nbdev once they ship it.

---

## Useful nbdev 3 shortcuts (alternatives to the explicit steps above)

| Command | Equivalent of |
|---|---|
| `nbdev-prepare` | `nbdev-export` + `nbdev-test` + `nbdev-clean` |
| `nbdev-pypi` | rebuild dist + `twine upload` |
| `nbdev-bump-version` | edit `__version__` in `tsai/__init__.py` |
| `nbdev-changelog` | generate CHANGELOG entries from merged GitHub PRs |
| `nbdev-release-both` | tag + GitHub release + PyPI upload |

For routine patch releases the shortcuts save time. For majors, the explicit steps in this runbook give more control and inspection points.
