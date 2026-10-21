# Contributing

## Documentation must always be updated

Any change to the extension's behaviour or API MUST update the docs in the same change. That
includes:

- a new class, method or parameter
- an exception message
- a serialization key
- a build step
- a supported PHP version
- a benchmark result

Update both files:

- [docs/DOCUMENTATION.md](docs/DOCUMENTATION.md): the full documentation.
- [README.md](README.md): the summary on the repository front page.

Follow the checklist in [docs/DOCUMENTATION.md section 10.3](docs/DOCUMENTATION.md#103-adding-a-new-model-checklist).
At minimum, touch these parts of DOCUMENTATION.md:

- the overview table
- the API reference, with runnable `<?php` examples
- the serialization keys table
- the test table
- the changelog

`tests/docs_examples.phpt` executes every ` ```php ` block that starts with `<?php` in
`README.md` and `docs/DOCUMENTATION.md`. Examples must therefore be self-contained and must
actually run. Keep the output comments in examples accurate.

## Build and test with Docker only

Don't install toolchains or PHP packages on your machine. Build, test and run everything in
Docker:

```bash
docker build --target test .
```

To test another PHP version, add `--build-arg PHP_VERSION=8.x`. Every supported version must
pass before a change is merged. GitHub Actions checks this on every pull request.

## Releases

Releases are published automatically by
[.github/workflows/ci.yml](.github/workflows/ci.yml). Don't create tags or releases by hand.

1. Bump `PHP_MLNATIVE_VERSION` in `php_mlnative.h`, following [semver](https://semver.org).
2. Add a `### <version>` entry at the top of the changelog (section 13 of
   `docs/DOCUMENTATION.md`). CI fails if the entry is missing.
3. Merge into `main`. CI tests all PHP versions and architectures, then publishes the release
   `v<version>` with prebuilt binaries.

A merge without a version bump is tested, but nothing is released. See
[docs/DOCUMENTATION.md section 9.1](docs/DOCUMENTATION.md#91-continuous-integration-and-releases)
for details.
