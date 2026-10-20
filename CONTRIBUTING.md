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
pass before a change is merged.
