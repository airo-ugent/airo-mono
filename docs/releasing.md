# Releasing


To create a release manually there are two steps:


1. version bumping
2. create & publish distribution



## Version bumping

run following command:
`bump-my-version bump --new-version <YYYY.MM.N>`

Where YYYY.MM are the year and month of the date and N is the micro part used to discriminate between releases in the same month. N starts at 0.

so for the initial release under this scheme on 8, Jan 2024 the following command was used: `bump-my-version bump --new-version 2024.1.0`

You can use the `--verbose` flag for more output and `-n` to dry-run.


Next to updating the version strings in all relevant files, this will also create a git tag to be associated with the release.


## Creating distribution