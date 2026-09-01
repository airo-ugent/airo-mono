# Releasing

Releases are automated by two GitHub Actions workflows: `prepare-release.yaml` and `publish-release.yaml`. You should not need to run anything locally for a normal release — see [Manual release](#manual-release-fallback) for the fallback procedure if the automation is unavailable.

## Automated release flow

1. **Prepare** (`.github/workflows/prepare-release.yaml`) runs on every push to `main` (and can also be triggered manually via "Run workflow" in the Actions tab). It:
   - Computes the next CalVer version from the version currently in `airo-typing/setup.py`, using `scripts/next_version.py` (see [versioning](./versioning.md) for the scheme).
   - If the version has changed, it updates the `version` field in every package's `setup.py` and in `CITATION.cff`, closes the `## Unreleased` section of `CHANGELOG.md` under the new version heading, reopens a fresh empty `## Unreleased` section above it for subsequent work, commits all of this to a new `release/<version>` branch, and opens a pull request titled `Release <version>` against `main`.
   - If nothing has changed since the last release (no version bump needed), it does nothing.
2. **Review and merge** the version-bump PR like any other PR. This is the point where a human decides to actually cut the release — merging it is what triggers publishing.
3. **Publish** (`.github/workflows/publish-release.yaml`) runs when a push to `main` contains an `Update version numbers for` commit message, i.e. when the version-bump PR from step 1 is merged. It:
   - Builds sdists/wheels for all 5 packages.
   - Publishes them to PyPI using [Trusted Publishing](https://docs.pypi.org/trusted-publishers/) (OIDC) — no PyPI API token is stored in the repo.
   - Creates and pushes a git tag `v<version>`.
   - Creates a GitHub Release for that tag, with release notes pulled directly from the matching section of `CHANGELOG.md`.

## Keeping the changelog releasable

`CHANGELOG.md` must always have a `## Unreleased` section (even if empty) directly above the latest release heading. Every PR should add its own entry there, per the contribution guidelines in `AGENTS.md`. If this section is missing when `prepare-release.yaml` tries to close it, the workflow fails loudly rather than silently skipping the changelog update.

## One-time setup

This is already configured for the airo-mono repo; documented here for reference in case it ever needs to be reproduced (e.g. after transferring the repo, or if PyPI project ownership changes):

- Each of the 5 PyPI projects (`airo-typing`, `airo-spatial-algebra`, `airo-robots`, `airo-camera-toolkit`, `airo-dataset-tools`) has a [Trusted Publisher](https://docs.pypi.org/trusted-publishers/) registered on its "Publishing" settings page, pointing at:
  - Owner: `airo-ugent`
  - Repository: `airo-mono`
  - Workflow name: `publish-release.yaml`
  - Environment name: `pypi`
- The repo has a `pypi` GitHub Actions environment (Settings → Environments), which `publish-release.yaml` runs under. This is what the Trusted Publisher entries above authorize.
- "Allow GitHub Actions to create and approve pull requests" is enabled under Settings → Actions → General → Workflow permissions, which `prepare-release.yaml` needs in order to open the version-bump PR.

## Manual release (fallback)

If the automated workflows are unavailable, you can still release by hand:

1. Bump the version manually — see [versioning](./versioning.md). Update the `version` field in every package's `setup.py`, `CITATION.cff`, and close out the `## Unreleased` section in `CHANGELOG.md`.
2. Use `scripts/build-airo-mono.sh` to build and publish the distribution.

You can find instructions on how to use this script at the top of the file, also repeated here verbatim:

```
This script is used to build and publish the AIRO mono packages.

Usage:
1. Make sure to update version numbers in ALL setup.py files before running this script.
2. Install the dev-requirements.txt file using pip: `pip install -r dev-requirements.txt`.
3. Make sure you have access to the PyPI projects and have your PyPI API tokens ready.
4. Run this script from airo-mono's root directory as `./scripts/build-airo-mono.sh`.
5. Follow the prompts to build and publish the packages.

For step 3, you can create a ~/.pypirc file with the following content:
[pypi]
username = __token__
password = <your PyPI API Token>

Is it the first time that you're using this script? You should use TestPyPI first.
```

You'll also need to create the git tag and GitHub Release by hand in this case (`git tag v<version> && git push origin v<version>`, then create the release from that tag on GitHub with the matching `CHANGELOG.md` section as notes).
