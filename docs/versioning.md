# Versioning


The airo-mono repo uses a single global version using a `YYYY.MM.N version scheme`.
Compatability between the packages is only guaranteed between identical versions, so you should use the same version of all packages.


## Scheme
We use a [Calender Version](https://calver.org/) scheme as follows:

```
YYYY.MM.N
```

where N is the MICRO part, used to distinguish between multiple releases in the same month. No semantic meaning can be attached to the versions and breaking changes might occur in each release for now.

## Releasing strategy
The main branch ('our trunk') will always live at an unreleased version.
When releasing a new version, it will be bumped immediately after the release.

For info on how to create releases, see [releasing.md](releasing.md).


## Development distributions/builds
Development builds might be available and their versioning scheme is
```
YYYY.MM.N-build.date.sha
```

the regular version is the to-be-released version. The build date is added after the tag, as well as the SHA of the corresponding commit for ease of use.
## Rationale
### Why not semantic versioning?

We don't want to create friction for making breaking changes for now.

### Why a global version instead of independent versions for each package?

Creates additional bookkeeping for adding compatibilities and breaks with the idea of 'version has unique commit/tag'.
Gives limited advantages in return.

Global version simply seems to be more in line with the 'mono-repo' approach.

