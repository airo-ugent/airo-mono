# Versioning


The airo-mono repo uses a single global version using a `YYYY.MM.N version scheme`.
Compatability between the packages is only guaranteed between identical versions, so you should use the same version of all packages.

The single point of truth for the version is in the `.bumpversion.toml` file.

## Scheme
We use a [Calender Version](https://calver.org/) scheme as follows:

```
YYYY.MM.N
```

where YYYY.MM is the year and month in which the release is made and N is the MICRO part used to distinguish between multiple releases in the same month. No semantic meaning can be attached to the parts and breaking changes might occur in each release for now.

## Versioning strategy
The main branch ('our trunk') will always live at last released version. All development builds will hence have the **previous** release as base.
This is slightly counter-intuitive but actually common practice (e.g. [Twisted](https://github.com/twisted/twisted/tree/trunk)).

## version bumping

To create a new version, you simply bump the version number as follows:

- take the year and month of the current day, e.g. 2024/01
- check the previous version (`.bumpversion.toml` file), if it was released in the same month, you bump the micro part. E.g. if the current version was 2024.1.2, the new version becomes 2024.1.3. If the latest version was from a previous month, you set the micro part to zero. e.g. if the current version was 2023.10.4, the new version is 2024.1.0.

use the `bump-my-version` command to set the new version:

run following command:
`bump-my-version bump --new-version <YYYY.MM.N>`

Where YYYY.MM are the year and month of the date and N is the micro part used to discriminate between releases in the same month. N starts at 0.

so for the initial release under this scheme on 8, Jan 2024 the following command was used: `bump-my-version bump --new-version 2024.1.0`

You can use the `--verbose` flag for more output and `-n` to dry-run.


Next to updating the version strings in all relevant files, this will also create a git tag to be associated with the release.

## Development distributions/builds
Development builds can be used to build the development trunk (main branch) and their versioning scheme is
```
YYYY.MM.N-build.date.sha
```

the regular version is the current version. The build date is added after the tag, as well as the SHA of the corresponding commit for ease of use.
To create a development build, run `bump-my-version bump dev`. Note that this command is meant for CI builds and it is not possible to bump build versions.


## Rationale
### Why not semantic versioning?

We don't want to create friction for making breaking changes for now.

### Why a global version instead of independent versions for each package?

Creates additional bookkeeping for adding compatibilities and breaks with the idea of 'version has unique commit/tag'.
Gives limited advantages in return.

Global version simply seems to be more in line with the 'mono-repo' approach.

