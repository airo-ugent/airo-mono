# Versioning


The airo-mono repo uses a single global version using a `YYYY.MM.N version scheme`.
Compatability between the packages is only guaranteed between identical versions, so you should use the same version of all packages.

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
- check the previous version. If it was released in the same month, you bump the micro part. E.g. if the current version was 2024.1.2, the new version becomes 2024.1.3. If the latest version was from a previous month, you set the micro part to zero. e.g. if the current version was 2023.10.4, the new version is 2024.1.0.

Update this version in all `setup.py` files and `CITATION.cff`.

## Rationale
### Why not semantic versioning?

We don't want to create friction for making breaking changes for now.

### Why a global version instead of independent versions for each package?

Even though packages may not change between versions when using a global version (e.g., `airo-spatial-algebra` is one package that may not need frequent changes), using a single, global, version makes it easier to track version compatibilities between packages inside of the monorepo.
