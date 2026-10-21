# PHP-ML-Native Documentation

The `mlnative` extension brings machine learning to PHP. Every algorithm is written in C++
and runs as native code inside the PHP engine, so training and prediction avoid PHP's
interpreter loops entirely.

> **Executable docs:** every code block that starts with `<?php`, in this file and in
> [README.md](../README.md), is run by the test suite
> ([tests/docs_examples.phpt](../tests/docs_examples.phpt)). If an example breaks, the build
> breaks. When you change the API or behaviour, update this file and the README in the same
> change (see [CONTRIBUTING.md](../CONTRIBUTING.md)).

---

## Table of contents

1. [Overview](#1-overview)
2. [Requirements and compatibility](#2-requirements-and-compatibility)
3. [Installation](#3-installation)
   - [3.1 Docker (recommended)](#31-docker-recommended)
   - [3.2 Adding the extension to your own Docker image](#32-adding-the-extension-to-your-own-docker-image)
   - [3.3 Building from source with phpize](#33-building-from-source-with-phpize)
   - [3.4 Enabling and verifying](#34-enabling-and-verifying)
   - [3.5 Build options](#35-build-options)
   - [3.6 Prebuilt binaries from GitHub Releases](#36-prebuilt-binaries-from-github-releases)
4. [Core concepts](#4-core-concepts)
5. [API reference](#5-api-reference)
   - [LinearRegression](#51-mlregressionlinearregression)
   - [LogisticRegression](#52-mlclassificationlogisticregression)
   - [KNearestNeighbors](#53-mlclassificationknearestneighbors)
   - [GaussianNB](#54-mlclassificationgaussiannb)
   - [KMeans](#55-mlclusteringkmeans)
   - [StandardScaler](#56-mlpreprocessingstandardscaler)
   - [Metrics](#57-mlmetrics)
   - [Exceptions](#58-exceptions)
6. [Saving and loading models](#6-saving-and-loading-models)
7. [Recipes](#7-recipes)
8. [Benchmarks](#8-benchmarks)
9. [Testing](#9-testing)
   - [9.1 Continuous integration and releases](#91-continuous-integration-and-releases)
10. [Architecture and extending the extension](#10-architecture-and-extending-the-extension)
11. [Troubleshooting](#11-troubleshooting)
12. [Limitations and roadmap](#12-limitations-and-roadmap)
13. [Changelog](#13-changelog)

---

## 1. Overview

| Task | Class | Algorithm |
|---|---|---|
| Regression | `ML\Regression\LinearRegression` | Least squares, optional L2 (ridge), closed-form solution via Cholesky |
| Classification | `ML\Classification\LogisticRegression` | Softmax regression (binary + multiclass), full-batch gradient descent |
| Classification | `ML\Classification\KNearestNeighbors` | k-NN with majority vote, euclidean / manhattan distance |
| Classification | `ML\Classification\GaussianNB` | Gaussian naive Bayes |
| Clustering | `ML\Clustering\KMeans` | Lloyd's algorithm with k-means++ initialization |
| Preprocessing | `ML\Preprocessing\StandardScaler` | Zero mean / unit variance scaling |
| Evaluation | `ML\Metrics` | accuracy, MSE, MAE, R² |

A minimal example:

```php
<?php
use ML\Regression\LinearRegression;

// y = 2a + 3b + 1
$model = (new LinearRegression())->fit(
    [[1, 2], [2, 1], [3, 4], [4, 3], [5, 5]],
    [9, 8, 19, 18, 26],
);

echo $model->predict([6, 6]), "\n";      // 31
```

---

## 2. Requirements and compatibility

| Requirement | Version |
|---|---|
| PHP | 8.1, 8.2, 8.3, 8.4 (NTS builds, every version tested on x86_64 and aarch64 by [GitHub Actions](#91-continuous-integration-and-releases)) |
| Compiler | Any C++17 compiler (GCC ≥ 7, Clang ≥ 5) |
| Build tools | `phpize` / `php-config` (php-dev package), `autoconf`, `make` |
| OS | Linux (tested: Debian bookworm, x86_64 and aarch64). macOS should work with Homebrew PHP but is untested. Windows is not supported yet. |
| PHP extensions | `spl` (always built in) |

PHP 8.0 and older are **not** supported or tested. PHP 8.0 is end-of-life and has no Debian
bookworm image to test against.

Thread-safe (ZTS) PHP builds are expected to work, because the extension has no global
state, but they are not covered by tests yet.

### Testing a specific PHP version

The Dockerfile takes the PHP version as a build argument:

```bash
docker build --build-arg PHP_VERSION=8.4 --target test .
```

---

## 3. Installation

### 3.1 Docker (recommended)

The repository's [Dockerfile](../Dockerfile) has four stages:

| Stage | What it does |
|---|---|
| `build` | Installs the compiler toolchain, runs `phpize`, `configure`, `make` |
| `test` | Runs the phpt test suite (fails the build on any failing test) |
| `artifact` | Holds only `mlnative.so`. Use it to export the compiled module: `docker build --target artifact --output type=local,dest=out .` writes `out/mlnative.so`. |
| `runtime` (default) | Clean `php:<version>-cli` image with only `mlnative.so` installed and enabled |

```bash
# Compile and run the tests
docker build --target test -t mlnative:test .

# Build the runtime image
docker build -t mlnative .

# Use it
docker run --rm mlnative php --ri mlnative
docker run --rm mlnative php examples/quickstart.php
docker run --rm -v "$PWD:/app" -w /app mlnative php your-script.php
```

### 3.2 Adding the extension to your own Docker image

Use a multi-stage build so that your final image doesn't contain the compiler:

```dockerfile
FROM php:8.3-fpm-bookworm AS mlnative-build
RUN apt-get update && apt-get install -y --no-install-recommends $PHPIZE_DEPS
COPY --from=ml-src config.m4 php_mlnative.h /usr/src/mlnative/
COPY --from=ml-src src /usr/src/mlnative/src
WORKDIR /usr/src/mlnative
RUN phpize && ./configure --enable-mlnative && make -j"$(nproc)"

FROM php:8.3-fpm-bookworm
COPY --from=mlnative-build /usr/src/mlnative/modules/mlnative.so /tmp/
RUN mv /tmp/mlnative.so "$(php-config --extension-dir)/" && docker-php-ext-enable mlnative
```

For `ml-src`, either use a named build context
(`docker build --build-context ml-src=path/to/PHP-ML-Extension .`) or replace those two
`COPY` lines with a `git clone` of this repository.

The build stage and the final image **must use the same PHP version and variant**
(e.g. both `8.3-fpm`). A `.so` built for one PHP API version won't load in another.

### 3.3 Building from source with phpize

On Debian/Ubuntu (replace `8.3` with your PHP version):

```bash
sudo apt-get install php8.3-dev g++ make autoconf
```

```bash
git clone https://github.com/nimabm/PHP-ML-Extension.git && cd PHP-ML-Extension
phpize
./configure --enable-mlnative
make -j"$(nproc)"
make test          # optional: run the test suite
sudo make install  # copies mlnative.so into `php-config --extension-dir`
```

If several PHP versions are installed, call the matching tools explicitly:
`phpize8.3` and `./configure --with-php-config=/usr/bin/php-config8.3`.

### 3.4 Enabling and verifying

Add the extension to your `php.ini` (or a file in the `conf.d` directory):

```ini
extension=mlnative
```

Check that it's loaded:

```bash
php -m | grep mlnative
php --ri mlnative        # version and list of models
php --re mlnative        # full reflection: classes, methods, signatures
```

From PHP code:

```php
<?php
if (!extension_loaded('mlnative')) {
    throw new RuntimeException('Install the mlnative extension');
}
echo phpversion('mlnative'), "\n";   // 0.1.0
```

Remember that the CLI and PHP-FPM usually read **different** ini files.
Run `php --ini` to see which files the CLI loads, and restart FPM after enabling the extension.

### 3.5 Build options

| Option | Effect |
|---|---|
| `--enable-mlnative` | Build the extension (required) |
| `CXXFLAGS="-O3 -march=native" ./configure ...` | Faster math, tuned to the build machine. Only use this when you build and run on the same CPU type. |
| `CXXFLAGS="-O0 -g" ./configure ...` | Debug build for gdb/valgrind |

The default flags are `-g -O2 -std=c++17`.

### 3.6 Prebuilt binaries from GitHub Releases

Every [release](https://github.com/nimabm/PHP-ML-Extension/releases) includes a compiled
`mlnative.so` for each supported PHP version and architecture, so you don't need a compiler.

| | |
|---|---|
| Archive name | `mlnative-<version>-php<8.x>-nts-linux-<x86_64\|aarch64>.tar.gz` |
| Contents | `mlnative.so`, `LICENSE`, `README.md`, `stubs/` |
| Built on | The official `php:<8.x>-cli-bookworm` Docker images (Debian 12, GCC 12) |
| Works with | NTS (non-thread-safe), non-debug PHP of the **same minor version** on Linux with glibc 2.36+, e.g. Debian 12, Ubuntu 24.04 or any official `php:*-bookworm` image |
| Doesn't work with | Alpine/musl, ZTS builds, debug builds, macOS, Windows. Build from source for those (3.2 / 3.3). |
| Checksums | `SHA256SUMS` in each release |

Check your PHP version, build type and architecture:

```bash
php -v
php -i | grep -E "^Thread Safety|^Debug Build"
uname -m
```

Then download the matching archive, verify it and install it (example: 0.1.0, PHP 8.3, x86_64):

```bash
VERSION=0.1.0 PHP=8.3 ARCH=x86_64
NAME="mlnative-$VERSION-php$PHP-nts-linux-$ARCH"
curl -fLO "https://github.com/nimabm/PHP-ML-Extension/releases/download/v$VERSION/$NAME.tar.gz"
curl -fLO "https://github.com/nimabm/PHP-ML-Extension/releases/download/v$VERSION/SHA256SUMS"
sha256sum --check --ignore-missing SHA256SUMS
tar -xzf "$NAME.tar.gz"
sudo cp "$NAME/mlnative.so" "$(php-config --extension-dir)/"
# then add `extension=mlnative` to your php.ini (see 3.4) and check:
php --ri mlnative
```

In a Dockerfile:

```dockerfile
FROM php:8.3-fpm-bookworm
ARG MLNATIVE_VERSION=0.1.0
RUN arch="$(uname -m)" \
    && name="mlnative-${MLNATIVE_VERSION}-php8.3-nts-linux-${arch}" \
    && curl -fsSL "https://github.com/nimabm/PHP-ML-Extension/releases/download/v${MLNATIVE_VERSION}/${name}.tar.gz" \
       | tar -xz -C /tmp \
    && mv "/tmp/${name}/mlnative.so" "$(php-config --extension-dir)/" \
    && rm -rf "/tmp/${name}" \
    && docker-php-ext-enable mlnative
```

---

## 4. Core concepts

### 4.1 Samples and features

A **sample** is an array of numbers (`int` or `float`). A **dataset** is a list of samples,
and every sample in it must have the same number of features.

```php
<?php
$samples = [
    [5.1, 3.5, 1.4],   // sample 0: 3 features
    [4.9, 3.0, 1.3],   // sample 1
];
```

- Only `int` and `float` are accepted as features. Numeric strings, `bool` and `null` are
  rejected with `ML\Exception\InvalidArgumentException`. This catches bugs such as passing
  raw CSV strings.
- **Array keys are ignored.** Values are read in iteration order, so `['area' => 50, 'rooms' => 1]`
  works the same as `[50, 1]`. Key order must match between training and prediction.
- PHP references inside arrays are fine.

### 4.2 Single sample or list of samples

Every `predict()`, `predictProbability()` and `transform()` method accepts either form:

- **A list of samples** (the first element is an array) returns a list of results.
- **A single sample** (the first element is a number) returns a single result.

```php
<?php
use ML\Regression\LinearRegression;

$m = (new LinearRegression())->fit([[1], [2], [3]], [2, 4, 6]);

var_dump($m->predict([4]));          // float(8)
var_dump($m->predict([[4], [5]]));   // array(2) { [0]=> float(8) [1]=> float(10) } (± float rounding)
```

**Performance tip:** prefer one batch call over a PHP loop of single-sample calls. Each call
has a fixed overhead for crossing the PHP → C++ boundary.

### 4.3 Labels

Classifiers accept `int` or `string` labels, and you can mix both in one dataset.

- `getClasses()` returns the distinct labels in sorted order: ints first (numerically), then
  strings (byte order).
- `predict()` returns labels with their original type.
- `predictProbability()` returns `label => probability` arrays. Because these are PHP array
  keys, a numeric-string label such as `"7"` shows up as the integer key `7`.
- `float`, `bool` and `null` labels are rejected. Round or cast them first.

### 4.4 Model lifecycle

```
new Model(hyperparameters)  →  fit(data)  →  predict(...) / get*()
                                  ↑    ↓
                             fit again (retrain)
```

- **Hyperparameters** are set only in the constructor and validated there.
- **`fit()` returns `$this`**, so you can write `$y = (new Model())->fit($X, $t)->predict($Z);`.
- **`fit()` is transactional.** If it throws (bad data, singular matrix, divergence), the
  model keeps its previous trained state.
- Calling `predict()` or a getter before `fit()` throws `ML\Exception\NotFittedException`.
- **`clone $model`** creates a fully independent deep copy.
- Models are **final** classes and **don't allow dynamic properties**.

```php
<?php
use ML\Regression\LinearRegression;

$m = (new LinearRegression())->fit([[1], [2]], [1, 2]);
try {
    $m->fit([[1, 1], [2, 2]], [1, 2]);   // collinear features: fails
} catch (ML\Exception\RuntimeException $e) {
    echo "fit failed: model still predicts ", $m->predict([3]), "\n";   // 3
}
```

### 4.5 Error handling

All exceptions thrown by the extension implement `ML\Exception\MLException`. You can catch
them all at once, or by kind (see [5.8 Exceptions](#58-exceptions)):

```php
<?php
use ML\Exception\MLException;
use ML\Regression\LinearRegression;

try {
    (new LinearRegression())->fit([[1, 2], [3]], [1, 2]);
} catch (MLException $e) {
    echo get_class($e), ': ', $e->getMessage(), "\n";
    // ML\Exception\InvalidArgumentException: Sample #1 has 1 features, expected 2
}
```

Argument **type** errors, such as passing a string where an array is expected, are still
reported by PHP itself as `TypeError`.

---

## 5. API reference

The signatures are also available as an IDE stub in [stubs/mlnative.stub.php](../stubs/mlnative.stub.php).

### 5.1 `ML\Regression\LinearRegression`

Fits `y = intercept + coefficients · x` by minimizing squared error, with an optional L2
penalty (ridge regression). The model is solved exactly in one step: no learning rate, no
iterations.

```php
final class LinearRegression {
    public function __construct(float $l2 = 0.0);
    public function fit(array $samples, array $targets): static;
    public function predict(array $samples): array|float;
    public function getCoefficients(): array;   // list<float>, one per feature
    public function getIntercept(): float;
}
```

| Parameter | Default | Description |
|---|---|---|
| `l2` | `0.0` | Ridge penalty (≥ 0). `0` is plain least squares. The penalty is applied to the sum of squared errors (like scikit-learn's `Ridge(alpha)`), and the intercept is never penalized. |

`$targets` is a list of `int|float`, one per sample.

**Example: house prices**

```php
<?php
use ML\Regression\LinearRegression;
use ML\Metrics;

// [area m², rooms] → price
$X = [[50, 1], [60, 2], [80, 2], [100, 3], [120, 3], [150, 4]];
$y = [150, 185, 235, 295, 345, 430];

$model = (new LinearRegression())->fit($X, $y);

printf("price(90 m², 3 rooms) = %.1f\n", $model->predict([90, 3]));
printf("coefficients = %s, intercept = %.2f\n",
    json_encode(array_map(fn ($c) => round($c, 3), $model->getCoefficients())),
    $model->getIntercept());
printf("R² = %.4f\n", Metrics::r2Score($y, $model->predict($X)));
```

**Example: ridge for correlated features**

When features are perfectly collinear, plain least squares has no unique solution and `fit()`
throws `RuntimeException`. A small `l2` fixes this:

```php
<?php
use ML\Regression\LinearRegression;

$X = [[1, 2], [2, 4], [3, 6]];   // second column = 2 × first column
$y = [1, 2, 3];

try {
    (new LinearRegression())->fit($X, $y);
} catch (ML\Exception\RuntimeException $e) {
    echo $e->getMessage(), "\n";
}

$ridge = (new LinearRegression(l2: 0.001))->fit($X, $y);
printf("%.3f\n", $ridge->predict([4, 8]));   // ≈ 4.000
```

**Notes**

- Training time is O(n·d² + d³) for n samples and d features. It scales well with many
  samples and up to a few hundred features.
- Fitting fails (`RuntimeException`) when a feature is constant, features are collinear, or
  there are fewer samples than features. Use `l2 > 0` or remove those features.

---

### 5.2 `ML\Classification\LogisticRegression`

Multinomial (softmax) logistic regression. Binary classification is simply the case with two
classes. It's trained with full-batch gradient descent and outputs calibrated probabilities.

```php
final class LogisticRegression {
    public function __construct(float $learningRate = 0.1, int $maxIterations = 1000,
                                float $l2 = 0.0, float $tolerance = 1.0E-6);
    public function fit(array $samples, array $targets): static;
    public function predict(array $samples): array|int|string;
    public function predictProbability(array $samples): array;
    public function getClasses(): array;   // list<int|string>
    public function getWeights(): array;   // label => list<float> (one weight per feature)
}
```

| Parameter | Default | Description |
|---|---|---|
| `learningRate` | `0.1` | Gradient descent step size (> 0) |
| `maxIterations` | `1000` | Maximum number of full passes over the data (≥ 1) |
| `l2` | `0.0` | L2 penalty on the weights (≥ 0). The biases are not penalized. |
| `tolerance` | `1e-6` | Stops early when the largest absolute gradient falls below this value (≥ 0). Use `0` to always run `maxIterations`. |

You need at least two distinct labels.

**Example: binary classification with probabilities**

```php
<?php
use ML\Classification\LogisticRegression;

$X = [[0, 0], [0, 1], [1, 0], [1, 1], [4, 4], [4, 5], [5, 4], [5, 5]];
$y = ['no', 'no', 'no', 'no', 'yes', 'yes', 'yes', 'yes'];

$clf = (new LogisticRegression(learningRate: 0.5, maxIterations: 2000))->fit($X, $y);

var_dump($clf->predict([4.8, 5.1]));                        // string(3) "yes"
print_r($clf->predict([[0.5, 0.5], [4.5, 4.5]]));           // [no, yes]

$p = $clf->predictProbability([2.5, 2.5]);                  // ['no' => …, 'yes' => …]
printf("P(no)=%.3f P(yes)=%.3f\n", $p['no'], $p['yes']);
```

**Example: multiclass**

```php
<?php
use ML\Classification\LogisticRegression;

$X = [[0, 0], [0, 1], [1, 0], [5, 5], [5, 6], [6, 5], [0, 9], [1, 9], [0, 10]];
$y = ['low', 'low', 'low', 'mid', 'mid', 'mid', 'high', 'high', 'high'];

$clf = (new LogisticRegression(0.1, 3000))->fit($X, $y);
print_r($clf->getClasses());                                  // [high, low, mid]
print_r($clf->predict([[0.5, 0.5], [5.5, 5.5], [0.5, 9.5]])); // [low, mid, high]

foreach ($clf->getWeights() as $label => $weights) {
    echo $label, ': ', json_encode(array_map(fn ($w) => round($w, 2), $weights)), "\n";
}
```

**Notes**

- **Scale your features** (see [StandardScaler](#56-mlpreprocessingstandardscaler)). Gradient
  descent converges slowly, or diverges, when features have very different ranges. Divergence
  throws `RuntimeException` ("LogisticRegression diverged; …").
- Each iteration costs O(n·d·K) for K classes.
- Training is deterministic: weights start at zero and there is no randomness.

---

### 5.3 `ML\Classification\KNearestNeighbors`

k-nearest-neighbours classifier. `fit()` stores the training data, and `predict()` returns
the majority label among the `k` closest training samples. **Ties** are broken in favour of
the class whose neighbours are closer to the query.

```php
final class KNearestNeighbors {
    public function __construct(int $k = 3, string $metric = 'euclidean');
    public function fit(array $samples, array $targets): static;
    public function predict(array $samples): array|int|string;
    public function getClasses(): array;
}
```

| Parameter | Default | Description |
|---|---|---|
| `k` | `3` | Number of neighbours (≥ 1, and ≤ number of training samples) |
| `metric` | `'euclidean'` | `'euclidean'` or `'manhattan'` |

```php
<?php
use ML\Classification\KNearestNeighbors;

$X = [[1, 1], [1, 2], [2, 1], [8, 8], [8, 9], [9, 8]];
$y = ['red', 'red', 'red', 'blue', 'blue', 'blue'];

$knn = (new KNearestNeighbors(k: 3))->fit($X, $y);
echo $knn->predict([1.5, 1.5]), "\n";                    // red
print_r($knn->predict([[8.5, 8.5], [2, 2]]));            // [blue, red]

$manhattan = (new KNearestNeighbors(1, 'manhattan'))->fit($X, $y);
echo $manhattan->predict([7, 7]), "\n";                  // blue
```

**Notes**

- Each prediction costs O(n·d) because it compares against every training sample. That's
  fast up to tens of thousands of training samples.
- The model keeps the whole training set, and so does its serialized form.
- Distances are sensitive to feature scale, so standardize features with different units.

---

### 5.4 `ML\Classification\GaussianNB`

Gaussian naive Bayes. Each feature is modelled per class as an independent normal
distribution. Training is a single pass and needs very little data, and the model outputs
probabilities.

```php
final class GaussianNB {
    public function __construct(float $varSmoothing = 1.0E-9);
    public function fit(array $samples, array $targets): static;
    public function predict(array $samples): array|int|string;
    public function predictProbability(array $samples): array;
    public function getClasses(): array;
}
```

| Parameter | Default | Description |
|---|---|---|
| `varSmoothing` | `1e-9` | Fraction of the largest feature variance that is added to every variance for numerical stability (≥ 0). This has the same meaning as in scikit-learn. |

```php
<?php
use ML\Classification\GaussianNB;

$X = [[1.0, 2.0], [1.2, 1.8], [0.8, 2.2], [5.0, 6.0], [5.2, 5.8], [4.8, 6.2]];
$y = [0, 0, 0, 1, 1, 1];

$nb = (new GaussianNB())->fit($X, $y);
print_r($nb->predict([[1, 2], [5, 6]]));          // [0, 1]

$p = $nb->predictProbability([3.0, 4.0]);         // exactly between the classes
printf("%.2f / %.2f\n", $p[0], $p[1]);            // 0.50 / 0.50
```

**Notes**

- Probabilities from naive Bayes are often overconfident (close to 0 or 1). Use them to rank
  predictions rather than as exact probabilities.
- Works with one class too, though `predict()` will then always return that class.

---

### 5.5 `ML\Clustering\KMeans`

Groups unlabeled samples into `clusters` groups. It uses Lloyd's algorithm with
k-means++ initialization.

```php
final class KMeans {
    public function __construct(int $clusters = 8, int $maxIterations = 300,
                                float $tolerance = 1.0E-4, ?int $seed = null);
    public function fit(array $samples): static;
    public function predict(array $samples): array|int;   // cluster index (0-based)
    public function getCentroids(): array;   // list<list<float>>, one row per cluster
    public function getLabels(): array;      // cluster of each training sample
    public function getInertia(): float;     // sum of squared distances to nearest centroid
    public function getIterations(): int;    // iterations actually run
}
```

| Parameter | Default | Description |
|---|---|---|
| `clusters` | `8` | Number of clusters (≥ 1, and ≤ number of samples) |
| `maxIterations` | `300` | Iteration cap (≥ 1) |
| `tolerance` | `1e-4` | Stops when the summed squared movement of all centroids in one iteration is ≤ this value |
| `seed` | `null` | Seed for the k-means++ initialization. `null` uses a random seed on every `fit()`. |

```php
<?php
use ML\Clustering\KMeans;

$points = [
    [1, 1], [1.5, 2], [1, 1.5],          // group A
    [8, 8], [8.5, 9], [9, 8],            // group B
    [1, 9], [1.5, 8.5], [2, 9],          // group C
];

$km = (new KMeans(clusters: 3, seed: 42))->fit($points);

echo json_encode($km->getLabels()), "\n";         // e.g. [2,2,2,1,1,1,0,0,0]
echo $km->predict([8.2, 8.4]), "\n";              // same cluster as group B
printf("inertia %.3f after %d iterations\n", $km->getInertia(), $km->getIterations());
```

**Choosing the number of clusters (elbow method)**

```php
<?php
use ML\Clustering\KMeans;

$points = [];
mt_srand(3);
foreach ([[0, 0], [10, 0], [5, 8]] as [$cx, $cy]) {
    for ($i = 0; $i < 30; $i++) {
        $points[] = [$cx + mt_rand(-100, 100) / 100, $cy + mt_rand(-100, 100) / 100];
    }
}
foreach (range(1, 6) as $k) {
    $inertia = (new KMeans($k, seed: 1))->fit($points)->getInertia();
    printf("k=%d inertia=%8.2f\n", $k, $inertia);   // the drop flattens after k=3
}
```

**Notes**

- Cluster numbers are arbitrary: cluster `0` has no special meaning, and the numbers can
  change between seeds.
- With a fixed `seed`, results are reproducible on the same platform and compiler. The C++
  standard doesn't guarantee identical random streams across standard libraries.
- If a cluster ends up empty, it is re-seeded with the sample that is farthest from its
  centroid.
- Scale features first if they have different units.

---

### 5.6 `ML\Preprocessing\StandardScaler`

Transforms each feature to zero mean and unit variance: `(x - mean) / std`, using the
population standard deviation. Constant features get a scale of `1`, so they are only
centered.

```php
final class StandardScaler {
    public function fit(array $samples): static;
    public function transform(array $samples): array;      // sample or list of samples
    public function fitTransform(array $samples): array;   // list of samples
    public function getMeans(): array;
    public function getScales(): array;
}
```

```php
<?php
use ML\Preprocessing\StandardScaler;

$train = [[1, 10, 5], [3, 20, 5], [5, 30, 5]];

$scaler = new StandardScaler();
$scaled = $scaler->fitTransform($train);
echo json_encode(array_map(fn ($r) => array_map(fn ($v) => round($v, 3), $r), $scaled)), "\n";
// [[-1.225,-1.225,0],[0,0,0],[1.225,1.225,0]]

// Apply the same transformation to new data:
echo json_encode(array_map(fn ($v) => round($v, 3), $scaler->transform([4, 25, 5]))), "\n";
// [0.612,0.612,0]
```

**Always fit the scaler on the training data only**, then use `transform()` on the test or
production data. See [Recipe 7.2](#72-pipeline-scaling--classifier).

---

### 5.7 `ML\Metrics`

Static evaluation helpers. The class cannot be instantiated.

| Method | Formula / meaning |
|---|---|
| `Metrics::accuracy(array $expected, array $predicted): float` | Fraction of positions where `expected === predicted`. The comparison is strict, so `1 !== '1'`. |
| `Metrics::meanSquaredError(array $expected, array $predicted): float` | mean((e − p)²) |
| `Metrics::meanAbsoluteError(array $expected, array $predicted): float` | mean(\|e − p\|) |
| `Metrics::r2Score(array $expected, array $predicted): float` | 1 − SS_res / SS_tot. If the expected values are constant, it returns 1.0 for perfect predictions and 0.0 otherwise. |

Both arrays must be non-empty and the same length. They are compared by position (keys are
ignored). The regression metrics require `int|float` values.

```php
<?php
use ML\Metrics;

echo Metrics::accuracy(['a', 'b', 'c', 'a'], ['a', 'b', 'a', 'a']), "\n";   // 0.75

$expected  = [3, -0.5, 2, 7];
$predicted = [2.5, 0.0, 2, 8];
echo Metrics::meanSquaredError($expected, $predicted), "\n";    // 0.375
echo Metrics::meanAbsoluteError($expected, $predicted), "\n";   // 0.5
printf("%.4f\n", Metrics::r2Score($expected, $predicted));      // 0.9486
```

---

### 5.8 Exceptions

```
ML\Exception\MLException  (interface)
├── ML\Exception\InvalidArgumentException  extends \InvalidArgumentException
├── ML\Exception\NotFittedException        extends \LogicException
└── ML\Exception\RuntimeException          extends \RuntimeException
```

| Exception | When | Example messages |
|---|---|---|
| `InvalidArgumentException` | Bad hyperparameter or bad data | `Sample #1 has 1 features, expected 2` · `Sample #0, feature #1 must be int or float, string given` · `Sample #0 must be an array of numbers, int given` · `Samples must not be empty` · `Number of samples (2) does not match number of targets (1)` · `Target #1 must be int or float, string given` · `Label #0 must be int or string, float given` · `k (10) cannot exceed the number of training samples (6)` · `LogisticRegression needs at least two distinct classes` · `Invalid serialized data: …` |
| `NotFittedException` | Model is used before `fit()` | `ML\Classification\GaussianNB is not fitted yet; call fit() first` |
| `RuntimeException` | Numerical failure, or the system ran out of memory in native code | `Cannot fit LinearRegression: the feature matrix is singular …` · `LogisticRegression diverged; lower the learning rate or scale the features …` |

Because the SPL exceptions are the parents, existing `catch (\InvalidArgumentException $e)`
blocks keep working.

---

## 6. Saving and loading models

Every model supports PHP's `serialize()` / `unserialize()` through
`__serialize()` / `__unserialize()`. Train once (in a CLI script or a cron job), save the
model, then load it on each web request:

```php
<?php
use ML\Classification\GaussianNB;

$file = sys_get_temp_dir() . '/spam.model';

// --- train.php (offline) ---
$model = (new GaussianNB())->fit([[0.1, 3], [0.2, 2], [0.9, 30], [0.8, 25]], ['ham', 'ham', 'spam', 'spam']);
file_put_contents($file, serialize($model), LOCK_EX);

// --- request time ---
$model = unserialize(file_get_contents($file), ['allowed_classes' => [GaussianNB::class]]);
echo $model->predict([0.85, 28]), "\n";   // spam
unlink($file);
```

- **Use `allowed_classes`** whenever the serialized data could come from somewhere you don't
  fully control. The extension validates the state it restores (array sizes, label indices,
  positive variances, …) and throws `InvalidArgumentException` for inconsistent data, but
  limiting the allowed classes is still good practice.
- You can cache the serialized string in APCu or Redis to skip the disk read.
- The format is a plain PHP array, so you can inspect it with `$model->__serialize()`:

| Class | Keys |
|---|---|
| LinearRegression | `l2`, `coefficients`, `intercept` |
| LogisticRegression | `learningRate`, `maxIterations`, `l2`, `tolerance`, `classes`, `features`, `weights` (flat, classes × features), `bias` |
| KNearestNeighbors | `k`, `metric`, `classes`, `features`, `samples` (flat, rows × features), `labels` (class indices) |
| GaussianNB | `varSmoothing`, `classes`, `features`, `means`, `variances` (flat, classes × features), `logPriors` |
| KMeans | `clusters`, `maxIterations`, `tolerance`, `seed`, `features`, `centroids` (flat), `labels`, `inertia`, `iterations` |
| StandardScaler | `means`, `scales` |

`var_export()` and `json_encode()` of a model don't include the trained state.
Use `serialize()`, or `__serialize()` for an array you can export.

---

## 7. Recipes

### 7.1 Train/test split and evaluation

The extension doesn't include a splitting helper, because it's a few lines of PHP:

```php
<?php
use ML\Classification\KNearestNeighbors;
use ML\Metrics;

function trainTestSplit(array $X, array $y, float $testSize = 0.25, int $seed = 42): array
{
    $idx = array_keys($X);
    mt_srand($seed);
    shuffle($idx);
    $nTest = (int) round(count($idx) * $testSize);
    $pick = fn (array $keys, array $from) => array_map(fn ($k) => $from[$k], $keys);
    $test = array_slice($idx, 0, $nTest);
    $train = array_slice($idx, $nTest);
    return [$pick($train, $X), $pick($test, $X), $pick($train, $y), $pick($test, $y)];
}

// Synthetic two-class data
mt_srand(7);
$X = $y = [];
for ($i = 0; $i < 200; $i++) {
    $cls = $i % 2;
    $X[] = [mt_rand(0, 100) / 25 + 3 * $cls, mt_rand(0, 100) / 25 + 3 * $cls];
    $y[] = $cls ? 'B' : 'A';
}

[$Xtrain, $Xtest, $ytrain, $ytest] = trainTestSplit($X, $y, 0.3);
$knn = (new KNearestNeighbors(5))->fit($Xtrain, $ytrain);
printf("test accuracy: %.3f\n", Metrics::accuracy($ytest, $knn->predict($Xtest)));
```

### 7.2 Pipeline: scaling + classifier

```php
<?php
use ML\Classification\LogisticRegression;
use ML\Preprocessing\StandardScaler;

// Features on very different scales: [age, yearly income]
$X = [[22, 18000], [25, 21000], [30, 25000], [45, 90000], [50, 120000], [38, 85000]];
$y = [0, 0, 0, 1, 1, 1];

$scaler = new StandardScaler();
$clf = (new LogisticRegression(learningRate: 0.5))->fit($scaler->fitTransform($X), $y);

$new = [[28, 23000], [47, 100000]];
print_r($clf->predict($scaler->transform($new)));   // [0, 1]

// Persist both together so production applies the same scaling:
$bundle = serialize(['scaler' => $scaler, 'model' => $clf]);
['scaler' => $s, 'model' => $m] = unserialize($bundle);
print_r($m->predict($s->transform($new)));          // [0, 1]
```

### 7.3 Picking `k` with a validation set

```php
<?php
use ML\Classification\KNearestNeighbors;
use ML\Metrics;

mt_srand(11);
$make = function (int $n) {
    $X = $y = [];
    for ($i = 0; $i < $n; $i++) {
        $x = [mt_rand() / mt_getrandmax() * 10, mt_rand() / mt_getrandmax() * 10];
        $X[] = $x;
        $y[] = ($x[0] + $x[1] + mt_rand(-150, 150) / 100) > 10 ? 'up' : 'down';
    }
    return [$X, $y];
};
[$Xtrain, $ytrain] = $make(300);
[$Xval, $yval] = $make(100);

$best = null;
foreach ([1, 3, 5, 7, 9, 15] as $k) {
    $acc = Metrics::accuracy($yval, (new KNearestNeighbors($k))->fit($Xtrain, $ytrain)->predict($Xval));
    printf("k=%2d accuracy=%.2f\n", $k, $acc);
    if ($best === null || $acc > $best[1]) {
        $best = [$k, $acc];
    }
}
echo "best k = {$best[0]}\n";
```

### 7.4 Confusion matrix

```php
<?php
use ML\Classification\GaussianNB;

$X = [[1, 1], [1.2, 0.8], [5, 5], [5.5, 4.8], [9, 1], [8.7, 1.3], [5.1, 5.2]];
$y = ['a', 'a', 'b', 'b', 'c', 'c', 'b'];
$pred = (new GaussianNB())->fit($X, $y)->predict($X);

$matrix = [];
foreach ($y as $i => $actual) {
    $matrix[$actual][$pred[$i]] = ($matrix[$actual][$pred[$i]] ?? 0) + 1;
}
foreach ($matrix as $actual => $row) {
    echo "$actual: ", json_encode($row), "\n";
}
```

### 7.5 Loading data from CSV

CSV values arrive as strings, but the extension only accepts `int|float`. Cast them while
reading:

```php
<?php
use ML\Regression\LinearRegression;

$csv = "area,rooms,price\n50,1,150\n80,2,235\n100,3,295\n150,4,430\n";
$file = tempnam(sys_get_temp_dir(), 'csv');
file_put_contents($file, $csv);

$X = $y = [];
$h = fopen($file, 'r');
fgetcsv($h, escape: '');                                 // skip header
while (($row = fgetcsv($h, escape: '')) !== false) {
    $X[] = [(float) $row[0], (float) $row[1]];
    $y[] = (float) $row[2];
}
fclose($h);
unlink($file);

$model = (new LinearRegression(l2: 0.01))->fit($X, $y);
printf("%.1f\n", $model->predict([120, 3]));
```

---

## 8. Benchmarks

[benchmarks/benchmark.php](../benchmarks/benchmark.php) compares the extension with
straightforward pure-PHP implementations of the same algorithms, on synthetic data with a
fixed seed.

```bash
docker build -t mlnative .
docker run --rm mlnative php benchmarks/benchmark.php
```

Results on PHP 8.3.35 (Docker on Apple Silicon, aarch64, default `-O2` build, opcache/JIT off):

| Benchmark | Pure PHP | mlnative | Speedup |
|---|---:|---:|---:|
| KNN: fit 3000×16 + predict 300 | 271.6 ms | 6.2 ms | 44× |
| LogisticRegression: 200 epochs on 3000×16 | 220.8 ms | 29.3 ms | 7.5× |
| LinearRegression: fit 20000×32 | 373.0 ms | 6.2 ms | 60× |

How to read these numbers:

- Every call first copies the PHP arrays into contiguous native memory. That cost grows
  linearly with the data size. The more computation a call does per sample, the bigger the
  speedup.
- The native LogisticRegression always computes a full softmax (K score vectors), while the
  PHP reference uses a single sigmoid for the binary case. The native side does about twice
  the arithmetic in that benchmark.
- Absolute times depend on your hardware, so run the benchmark on your target machine.
  Building with `CXXFLAGS="-O3 -march=native"` can speed up the native side further.

When you add a model, add a pure-PHP reference and a row to the benchmark
(see [10.3](#103-adding-a-new-model-checklist)).

---

## 9. Testing

The tests are phpt files in [tests/](../tests/), PHP's standard extension test format:

| File | Covers |
|---|---|
| `001_loaded.phpt` | Extension loads, all classes and the exception hierarchy exist |
| `linear_regression.phpt` | Exact recovery, ridge, singular matrix, transactional fit, serialize, clone |
| `logistic_regression.phpt` | Binary + multiclass, probabilities, weights, errors, tampered serialized data |
| `knn.phpt` | Both metrics, tie-breaking, k validation, serialization validation |
| `gaussian_nb.phpt` | Predictions, probabilities, numeric-string labels, serialization validation |
| `kmeans.phpt` | Clustering, inertia, seeds, errors, serialization |
| `standard_scaler.phpt` | Scaling, constant features, errors |
| `metrics.phpt` | All metrics and their edge cases |
| `input_validation.phpt` | Every input error message, `TypeError`, references, dynamic properties |
| `docs_examples.phpt` | Runs every `<?php` example in this document and in `README.md` |

```bash
# Run all tests (in Docker)
docker build --target test .

# Run against another PHP version
docker build --build-arg PHP_VERSION=8.1 --target test .

# From a phpize build directory
make test TESTS="--show-diff tests"
```

**Memory checking with valgrind.** The suite runs clean under valgrind: 0 errors, 0 leaks.

```bash
docker build --target build -t mlnative:build .
docker run --rm -v "$PWD/tests:/t:ro" -v "$PWD/docs:/usr/src/mlnative/docs:ro" \
  -v "$PWD/README.md:/usr/src/mlnative/README.md:ro" mlnative:build sh -c \
  'apt-get update -qq && apt-get install -y -qq valgrind >/dev/null && cp -r /t tests && \
   NO_INTERACTION=1 php run-tests.php -m -d extension=$PWD/modules/mlnative.so tests'
```

### 9.1 Continuous integration and releases

[.github/workflows/ci.yml](../.github/workflows/ci.yml) runs on every pull request, on every
push to `main` (that includes merges) and on demand from the Actions tab
(`workflow_dispatch`).

| Job | What it does |
|---|---|
| **Read version** | Reads `PHP_MLNATIVE_VERSION` from `php_mlnative.h`. It **fails** if the [changelog](#13-changelog) has no `### <version>` entry. On `main`, it also decides whether to release: only when the tag `v<version>` doesn't exist yet. |
| **Build** (8 jobs) | PHP 8.1, 8.2, 8.3 and 8.4 × x86_64 (`ubuntu-24.04`) and aarch64 (`ubuntu-24.04-arm`). Each job runs the full test suite (`--target test`), exports `mlnative.so` (`--target artifact`), packages it as `mlnative-<version>-php<8.x>-nts-linux-<arch>.tar.gz`, and uploads it as a workflow artifact. Pull requests get these artifacts too, for manual testing. |
| **Publish release** | Only on `main`, and only when the version is new. It creates the tag `v<version>` on the merged commit and publishes a GitHub release. The release has the 8 archives, a `SHA256SUMS` file, and the changelog entry plus install instructions as release notes. |

**A merge into `main` does not always create a release.** A release happens only when
`PHP_MLNATIVE_VERSION` has changed to a version that hasn't been released yet. Merges without
a version bump are still tested and built, but they don't publish anything. This prevents
duplicate or accidental releases.

**How to publish a release**

1. Bump `PHP_MLNATIVE_VERSION` in `php_mlnative.h` (e.g. `0.1.0` → `0.2.0`), following
   [semantic versioning](https://semver.org):
   - patch for fixes
   - minor for new features
   - major for breaking API changes
2. Add a `### 0.2.0` entry at the top of the [changelog](#13-changelog) describing the changes.
   The text of this entry becomes the release notes.
3. Open a pull request. CI checks that the changelog entry exists and that all 8 builds pass.
4. Merge into `main`. The release `v0.2.0` is published automatically when CI finishes.

To check the release notes locally before merging:

```bash
docker run --rm -v "$PWD:/repo:ro" -w /repo debian:bookworm-slim sh scripts/release-notes.sh 0.2.0
```

**Repository settings needed:** GitHub must be allowed to run Actions on the repository. The
release job uses the built-in `GITHUB_TOKEN` with `contents: write`, so it needs no secret.
If your organization restricts it, enable *Settings → Actions → General → Workflow
permissions → Read and write permissions*. The aarch64 jobs use GitHub's hosted
`ubuntu-24.04-arm` runners. These are free for public repositories; private repositories
need a plan that includes arm64 runners.

---

## 10. Architecture and extending the extension

### 10.1 Source layout

```
config.m4                 Build definition (phpize/autoconf)
php_mlnative.h            Module header, version constant (single source of the release version)
Dockerfile                build / test / artifact / runtime stages
.github/workflows/ci.yml  CI matrix and automatic releases (see 9.1)
scripts/release-notes.sh  Extracts a version's changelog entry (used by CI)
src/
  mlnative.cpp            Module entry; MINIT registers all classes
  php_bridge.hpp/.cpp     Zend glue (see below)
  arginfo.hpp             Shared argument/return type info
  classes/*.cpp           One file per PHP class: arginfo, methods, (un)serialize
  core/*.hpp/.cpp         Pure C++17 algorithms, no PHP dependency
stubs/mlnative.stub.php   IDE stub / public API description
tests/*.phpt              Test suite
benchmarks/benchmark.php  Native vs pure-PHP benchmark
examples/quickstart.php   Tour of all features
docs/DOCUMENTATION.md     This document
```

### 10.2 Design rules

1. **`src/core` has no PHP dependency.** Algorithms work on `mlcore::Matrix` (dense,
   row-major `double`) and `std::vector`, so they can be unit-tested or reused outside PHP.
2. **No C++ exception may reach the Zend engine.** Every method body that calls native code
   runs inside `mlphp::guard()`, which converts exceptions as follows:
   - `std::invalid_argument` → `InvalidArgumentException`
   - `mlcore::NotFittedError` → `NotFittedException`
   - `std::bad_alloc` and other `std::exception` → `RuntimeException`
3. **Convert first, compute, then write results.** PHP arrays are converted into native
   memory up front (`read_samples`, `read_sample`, `read_numbers`). Results are written to
   `return_value` only after the computation has succeeded.
4. **`fit()` is transactional.** Core models train into local variables and only commit
   them at the end.
5. **One object = one native model.** `mlphp::register_class<T>()` installs the
   `create_object`, `free_obj` and `clone_obj` handlers. `T` must be default-constructible
   and copy-assignable.
6. **`__unserialize` validates everything** (sizes, index ranges) before it commits, so
   crafted serialized data can never cause out-of-bounds access.

### 10.3 Adding a new model (checklist)

1. **Core algorithm:** add `src/core/<model>.hpp/.cpp`, with a strong exception guarantee in `fit()`.
2. **PHP class:** add `src/classes/<model>.cpp` with:
   - arginfo
   - `PHP_METHOD`s wrapped in `guard()`
   - `__serialize`/`__unserialize` with validation
   - a `mlphp::register_<model>()` function
   For classifiers, wrap the core model in a struct with a `LabelEncoder` and reuse
   `predict_labels` / `predict_probabilities`.
3. **Registration:** declare the register function in `src/php_bridge.hpp`, call it from
   `PHP_MINIT_FUNCTION` in `src/mlnative.cpp`, and add both `.cpp` files to `config.m4`.
4. **Stub:** add the class to `stubs/mlnative.stub.php`.
5. **Tests:** add `tests/<model>.phpt` covering behaviour, errors and a serialize round trip.
6. **Benchmark:** add a pure-PHP reference and a row to `benchmarks/benchmark.php`.
7. **Docs:** update this document:
   - [Overview](#1-overview) table
   - an [API reference](#5-api-reference) section with runnable examples
   - serialization keys in [section 6](#6-saving-and-loading-models)
   - the [test table](#9-testing)
   - the [Changelog](#13-changelog)
   Also update the summary in `README.md`.
8. **Verify:** run `docker build --target test .`, and run valgrind for non-trivial changes.
9. **Release (optional):** bump `PHP_MLNATIVE_VERSION` and add the changelog entry. CI
   publishes the release when the change is merged (see [9.1](#91-continuous-integration-and-releases)).

---

## 11. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `Class "ML\..." not found` | The extension isn't loaded in this SAPI. Check `php -m` and `php --ini`. FPM and CLI use different ini files, and FPM needs a restart after enabling it. |
| `PHP Startup: Unable to load dynamic library 'mlnative'` … `API=...` mismatch | The `.so` was built for a different PHP version or variant (ZTS/NTS). Rebuild it with the `phpize`/`php-config` of the PHP that loads it. |
| `phpize: command not found` | Install the PHP development package (`php8.x-dev`), or use the Docker build. |
| `Cannot fit LinearRegression: the feature matrix is singular …` | Constant or duplicated features, or fewer samples than features. Use `new LinearRegression(l2: 0.01)` or drop those features. |
| `LogisticRegression diverged …`, or poor accuracy | Scale features with `StandardScaler`, or lower `learningRate`. If training stops early, raise `maxIterations`. |
| `Sample #0, feature #1 must be int or float, string given` | Data from CSV, a form or a database arrives as strings. Cast it with `(float)`. |
| `Label #0 must be int or string, float given` | Labels must be discrete. Cast them with `(int)` or use strings. |
| KMeans gives different results on each run | Pass a `seed`. |
| Prebuilt `mlnative.so` fails with `GLIBC_2.36 not found` or `GLIBCXX_… not found` | Your system is older than Debian 12 / Ubuntu 24.04. Build from source (3.2 / 3.3). |
| A merge into `main` didn't create a release | Releases happen only when `PHP_MLNATIVE_VERSION` changes to a version without an existing `v<version>` tag (see 9.1). |
| CI fails with `No changelog entry "### x.y.z"` | Add that version's entry to the [changelog](#13-changelog). |
| Memory grows beyond `memory_limit` | Native buffers are not counted by `memory_limit` (see [Limitations](#12-limitations-and-roadmap)). Train on smaller batches or in a separate CLI process. |

---

## 12. Limitations and roadmap

**Current limitations**

- **Dense data only:** there is no sparse matrix support, so very high-dimensional sparse
  data (e.g. bag of words) uses a lot of memory.
- **Single-threaded:** all algorithms run on one thread.
- **Memory limit:** native memory comes from the C++ allocator, so PHP's `memory_limit`
  does not count it.
- **No incremental training:** there is no `partialFit()`, so every `fit()` retrains from
  scratch.
- **Untested platforms:** Windows builds are not supported yet, and ZTS builds are untested.

**Possible next steps:** decision trees and random forests, SVM, PCA, MinMaxScaler and
one-hot encoding, a cross-validation helper, multithreaded prediction, and sparse input.

---

## 13. Changelog

### 0.1.0

- Initial release: LinearRegression (with ridge), LogisticRegression (softmax),
  KNearestNeighbors, GaussianNB, KMeans (k-means++), StandardScaler, Metrics.
- Typed exception hierarchy, transactional `fit()`, `clone` support, validated
  serialization.
- Docker build/test pipeline, phpt test suite, benchmark, executable documentation
  (this document and the README examples are run as tests).
- GitHub Actions: tests PHP 8.1–8.4 on x86_64 and aarch64, and publishes a release with
  prebuilt `mlnative.so` archives whenever a new version is merged into `main`.
