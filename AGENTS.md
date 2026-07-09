# airo-mono

Python monorepo for robot manipulation tooling from the [AIRO lab](https://airo.ugent.be/) at Ghent University.

## New users

If a user is new to airo-mono and needs help getting started, you can point them towards `docs/getting_started.md`.

## Packages

| Package | Purpose |
|---------|---------|
| `airo-typing` | Shared type aliases and unit/convention contracts (scalar-last quaternions, metric units, row-major homogeneous matrices) |
| `airo-spatial-algebra` | SE3 poses and transforms via a `spatialmath-python` wrapper |
| `airo-robots` | UR robot (RTDE) and Robotiq gripper interfaces; async `AwaitableAction` pattern |
| `airo-camera-toolkit` | RGB(D) camera interfaces (ZED, RealSense, USB), image processing, point clouds, hand-eye calibration |
| `airo-dataset-tools` | COCO dataset utilities, pose/intrinsics formats, CVAT workflow, optional FiftyOne visualization |

## Setup

Use `uv` to create a local `.venv` inside the repo — this keeps the environment in the project folder, which works well with sandbox path restrictions.

```bash
uv venv --python 3.12            # creates .venv/ in the repo root
uv pip install -e airo-typing/ airo-spatial-algebra/ airo-robots/ airo-camera-toolkit/ airo-dataset-tools/
uv pip install -r dev-requirements.txt
pre-commit install
```

To install a single package:

```bash
uv pip install -e airo-robots/
```

The human-oriented full setup (conda + all packages) is documented in `README.md`.

## Development Workflow

### Commits

Pre-commit hooks must pass on every commit. If not already installed:

```bash
pre-commit install
```

Never bypass hooks with `--no-verify`. If a hook fails, fix the issue and commit again.

### Features & PRs

Before opening or marking a PR ready:

1. Run tests for every package touched by the change:
   ```bash
   make pytest airo-robots/          # repeat for each affected package
   ```
2. Update `CHANGELOG.md` under the `## Unreleased` section. Use the appropriate sub-section (`Added`, `Changed`, `Fixed`, `Removed`, `Breaking changes`) and prefix the entry with the package name, e.g.:
   ```
   - `airo-robots`: Added support for UR20.
   ```

---

## Running Tests

```bash
make pytest airo-robots/          # runs pytest with coverage; works for any package dir
pytest airo-camera-toolkit/ -m "not expensive"   # skip hardware/slow tests
```

Each package has a `test/` directory. At least one test per package is required (enforced by CI).

Jupyter notebooks are tested via `pytest-nbmake` in CI but not by `make pytest` locally.

## Type Checking

```bash
mypy airo-robots/     # repeat for any package
```

mypy is configured in `mypy.ini` at repo root. All public functions must be typed (`disallow_untyped_defs = True`).

## Code Style

Pre-commit hooks run automatically on commit. To run manually:

```bash
pre-commit run --all-files
```

- **Black** — formatter, 119-character line length
- **isort** — `--profile black --line-length 119`
- **autoflake** — removes unused imports/variables
- **Flake8** — max-line-length 120, max-complexity 10; ignores E203, E501, E266, E402
- **Docstrings** — Google format

## Coding Conventions

- Use **loguru** for logging, not `print`.
- Return **native datatypes / NumPy arrays** from methods; use dataclasses for structured data.
- Use **Python properties** for getters/setters, not `get_x()` methods.
- Use the **`AwaitableAction`** pattern for async hardware commands (see `airo-robots`).
- Keep packages **framework-agnostic** — avoid pulling in ROS, PyTorch, etc. at the package level.
- Prefer using existing libraries over writing new algorithms.

## CLI Entry Points

`airo-dataset-tools` and `airo-camera-toolkit` expose CLIs via Click:

```bash
airo-dataset-tools --help
airo-camera-toolkit --help
```

Entry points are declared in each package's `setup.py` under `console_scripts`.

## CI

Three GitHub Actions workflows:

| Workflow | Trigger | What it does |
|----------|---------|--------------|
| `pre-commit.yaml` | Every push | Format checks |
| `pytest.yaml` | PRs / main | Tests (Python 3.10, 3.11, 3.12 × each package) |
| `mypy.yaml` | PRs / main | Static type checking |

## Versioning & Releases

- All packages share one version (CalVer: `YYYY.M.PATCH`, e.g. `2026.5.0`).
- Use `bump-my-version` to update; see `docs/releasing.md` for the full procedure.
- Packages are published to PyPI with `scripts/build-airo-mono.sh` + `twine`.

## Key Dependencies

- **Python** ≥ 3.10
- **numpy** ≥ 2.0
- **opencv-contrib-python** == 4.10.0.84 (pinned — contrib modules + ArUco API)
- **pydantic** > 2.0
- **spatialmath-python**, **scipy** (spatial algebra)
- **rerun-sdk** ≥ 0.23.4 (visualization)
- **ur-rtde** ≥ 1.6.0 (UR robot driver — hardware only)

Optional / hardware-specific deps (`fiftyone`, `airo-ipc`, `airo-tulip`) are declared as optional extras or installed separately.
