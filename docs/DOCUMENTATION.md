# PHP-ML-Native Documentation

The `mlnative` extension brings machine learning to PHP. Every algorithm is written in C++
and runs as native code inside the PHP engine, so training and prediction avoid PHP's
interpreter loops entirely.

Version 0.1.2 provides more than 100 classes, covering:

- classification, regression, clustering and anomaly detection
- ensembles, pipelines and hyper-parameter search
- preprocessing (scaling, encoding, imputation, dimensionality reduction)
- text processing
- neural networks
- evaluation

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
4. [Core concepts](#4-core-concepts)
5. [Classification](#5-classification)
6. [Regression](#6-regression)
7. [Clustering](#7-clustering)
8. [Anomaly detection](#8-anomaly-detection)
9. [Ensembles, pipelines and model selection](#9-ensembles-pipelines-and-model-selection)
10. [Preprocessing](#10-preprocessing)
11. [Dimensionality reduction](#11-dimensionality-reduction)
12. [Text](#12-text)
13. [Neural networks](#13-neural-networks)
14. [Evaluation: metrics, reports and validators](#14-evaluation-metrics-reports-and-validators)
15. [Datasets and distances](#15-datasets-and-distances)
16. [Interfaces and exceptions](#16-interfaces-and-exceptions)
17. [Saving and loading models](#17-saving-and-loading-models)
18. [Recipes](#18-recipes)
19. [Benchmarks](#19-benchmarks)
20. [Testing](#20-testing)
21. [Architecture and extending the extension](#21-architecture-and-extending-the-extension)
22. [Troubleshooting](#22-troubleshooting)
23. [Limitations and roadmap](#23-limitations-and-roadmap)
24. [Changelog](#24-changelog)

---

## 1. Overview

| Area | Classes |
|---|---|
| [Classification](#5-classification) | `LogisticRegression`, `KNearestNeighbors`, `KDNeighbors`, `RadiusNeighbors`, `GaussianNB`, `NaiveBayes`, `MultinomialNB`, `ClassificationTree`, `ExtraTreeClassifier`, `RandomForest`, `AdaBoost`, `LogitBoost`, `SVC`, `MLPClassifier` (namespace `ML\Classification`) |
| [Regression](#6-regression) | `LinearRegression`, `ElasticNet`, `RegressionTree`, `ExtraTreeRegressor`, `RandomForestRegressor`, `GradientBoost`, `KNNRegressor`, `KDNeighborsRegressor`, `RadiusNeighborsRegressor`, `SVR`, `MLPRegressor`, `Adaline` (`ML\Regression`) |
| [Clustering](#7-clustering) | `KMeans`, `KMedoids`, `FuzzyCMeans`, `GaussianMixture`, `MeanShift`, `DBSCAN`, `OPTICS`, `Ward` (`ML\Clustering`) |
| [Anomaly detection](#8-anomaly-detection) | `RobustZScore`, `GaussianMLE`, `IsolationForest`, `LocalOutlierFactor`, `Loda`, `OneClassSVM` (`ML\AnomalyDetection`) |
| [Meta estimators](#9-ensembles-pipelines-and-model-selection) | `OneVsRest`, `BootstrapAggregator`, `CommitteeMachine` (`ML\Ensemble`), `ML\Pipeline`, `GridSearch`, `BayesianSearch` (`ML\ModelSelection`) |
| [Preprocessing](#10-preprocessing) | `StandardScaler`, `MinMaxNormalizer`, `MaxAbsoluteScaler`, `RobustStandardizer`, `PowerTransformer`, `L1Normalizer`, `L2Normalizer`, `PolynomialExpander`, `OneHotEncoder`, `IntervalDiscretizer`, `NumericStringConverter`, `BooleanConverter`, `MissingDataImputer`, `KNNImputer`, `HotDeckImputer`, `LambdaFunction` (`ML\Preprocessing`) |
| [Dimensionality reduction](#11-dimensionality-reduction) | `PCA`, `TruncatedSVD`, `LinearDiscriminantAnalysis`, `GaussianRandomProjector`, `SparseRandomProjector`, `TSNE` (`ML\Preprocessing`) |
| [Text](#12-text) | `WordTokenizer`, `WhitespaceTokenizer`, `SentenceTokenizer`, `NGramTokenizer`, `KSkipNGramTokenizer`, `WordStemmer`, `PorterStemmer`, `TextNormalizer`, `MultibyteTextNormalizer`, `RegexFilter`, `StopWordFilter`, `WordCountVectorizer`, `TokenHashingVectorizer`, `TfIdfTransformer`, `BM25Transformer` (`ML\Text`) |
| [Neural networks](#13-neural-networks) | Layers `Dense`, `Activation`, `Dropout`, `BatchNorm`, `PReLU`, `Swish`, `Noise`; optimizers `Stochastic`, `Momentum`, `AdaGrad`, `RMSProp`, `Adam`, `AdaMax`; schedules `Constant`, `StepDecay`, `Cosine`, `Cyclical`, `Ramp` (`ML\NeuralNet\…`) |
| [Evaluation](#14-evaluation-metrics-reports-and-validators) | `ML\Metrics` (24 metrics), `ML\CrossValidation\Reports`, validators `HoldOut`, `KFold`, `LeavePOut`, `MonteCarlo` |
| [Datasets and distances](#15-datasets-and-distances) | `ML\Datasets` (generators and `trainTestSplit`), `ML\Distance` (11 distance metrics plus Gower) |

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
| PHP | 8.1, 8.2, 8.3, 8.4 (NTS builds, every version tested on x86_64 and aarch64 by [GitHub Actions](#201-continuous-integration-and-releases)) |
| Compiler | Any C++17 compiler (GCC ≥ 7, Clang ≥ 5) with `std::thread` support |
| Build tools | `phpize` / `php-config` (php-dev package), `autoconf`, `make` |
| OS | Linux (tested: Debian bookworm, x86_64 and aarch64). macOS should work with Homebrew PHP but is untested. Windows is not supported yet. |
| PHP extensions | `spl` (always built in). `mbstring` is only needed for `MultibyteTextNormalizer`, and `pcre` (built in) for `RegexFilter`. |

PHP 8.0 and older are **not** supported or tested. PHP 8.0 is end-of-life and has no Debian
bookworm image to test against.

Thread-safe (ZTS) PHP builds are expected to work, because the extension has no global
state, but they are not covered by tests yet.

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

### 3.4 Enabling, verifying and configuring

Add the extension to your `php.ini` (or a file in the `conf.d` directory):

```ini
extension=mlnative
; optional: worker threads for native code (default 1, 0 = all CPU cores)
mlnative.threads=4
```

Check that it's loaded:

```bash
php -m | grep mlnative
php --ri mlnative        # version, list of models and the mlnative.threads setting
php --re mlnative        # full reflection: classes, methods, signatures
```

From PHP code:

```php
<?php
if (!extension_loaded('mlnative')) {
    throw new RuntimeException('Install the mlnative extension');
}
echo phpversion('mlnative'), "\n";   // 0.1.2
```

Remember that the CLI and PHP-FPM usually read **different** ini files.
Run `php --ini` to see which files the CLI loads, and restart FPM after enabling the extension.

### 3.5 Build options

| Option | Effect |
|---|---|
| `--enable-mlnative` | Build the extension (required) |
| `CXXFLAGS="-O3 -march=native" ./configure ...` | Faster math, tuned to the build machine. Only use this when you build and run on the same CPU type. |
| `CXXFLAGS="-O0 -g" ./configure ...` | Debug build for gdb/valgrind |

The default flags are `-g -O2 -std=c++17 -pthread`.

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

Then download the matching archive, verify it and install it (example: 0.1.2, PHP 8.3, x86_64):

```bash
VERSION=0.1.2 PHP=8.3 ARCH=x86_64
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
ARG MLNATIVE_VERSION=0.1.2
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

- **Numeric only (most models):** most models accept only `int` and `float` features.
  Numeric strings, `bool` and `null` are rejected with `ML\Exception\InvalidArgumentException`.
  This catches bugs such as passing raw CSV strings.
- **Mixed data:** these classes also accept strings, booleans and missing values (`null` or `NAN`):
  - [`NaiveBayes`](#56-naivebayes) works on categorical features.
  - The mixed-data transformers in [10.3](#103-categorical-and-mixed-data) and
    [10.4](#104-missing-values) handle the rest. Use them to turn mixed data into numbers.
- **Array keys are ignored:** values are read in iteration order, so `['area' => 50, 'rooms' => 1]`
  works the same as `[50, 1]`. Key order must match between training and prediction.

### 4.2 Single sample or list of samples

Every `predict()`, `predictProbability()`, `score()` and `transform()` method accepts either
form:

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
has a fixed overhead for crossing the PHP → C++ boundary. Batch calls are also split across
threads when `mlnative.threads` > 1.

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

- **Hyperparameters** are set only in the constructor and validated there. All constructor
  parameters can be passed by name, e.g. `new RandomForest(estimators: 200, seed: 1)`.
- **`fit()` returns `$this`**, so you can write `$y = (new Model())->fit($X, $t)->predict($Z);`.
- **`fit()` is transactional.** If it throws (bad data, singular matrix, divergence), the
  model keeps its previous trained state.
- **Before `fit()`:** calling `predict()` or a getter throws `ML\Exception\NotFittedException`.
- **`clone $model`** creates a fully independent deep copy. That includes meta estimators
  such as `Pipeline` and the objects they wrap.
- **`seed`:** every model with randomness has a `seed` parameter. With a seed, results are
  reproducible on the same platform. Without one, every `fit()` uses fresh randomness.
- **Class restrictions:** models are **final** classes and **don't allow dynamic properties**.

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

### 4.5 Multithreading

Native code can use several CPU cores. The setting is the ini entry `mlnative.threads`:

| Value | Meaning |
|---|---|
| `1` (default) | Single-threaded, which is safe for PHP-FPM workers. |
| `n` | Up to `n` threads. |
| `0` | All CPU cores. |

The setting is `PHP_INI_ALL`, so you can change it at runtime with `ini_set()`.

What runs in parallel:

- training of `RandomForest`, `RandomForestRegressor` and `IsolationForest` trees
- the one-vs-one problems of `SVC`
- `MeanShift` seeds and `t-SNE` gradients
- batch `predict()`, `predictProbability()`, `score()` and `transform()` of native models
- `Distance::pairwise()`

**Results are identical for any number of threads.** Random seeds are drawn before the work
is split, so a seeded forest gives the same trees with 1 or 16 threads.

```php
<?php
use ML\Classification\RandomForest;
use ML\Datasets;

$data = Datasets::agglomerate(['a' => Datasets::blob([0, 0], 1.0, 200, 1), 'b' => Datasets::blob([3, 3], 1.0, 200, 2)]);

ini_set('mlnative.threads', '4');
$forest = (new RandomForest(100, seed: 1))->fit($data['samples'], $data['labels']);
ini_set('mlnative.threads', '1');

echo count($forest->predict($data['samples'])), "\n";   // 400
```

### 4.6 Distance metrics

Models with a `metric` parameter accept these names. You can also call them directly with
[`ML\Distance`](#152-distances).

| Name | Formula |
|---|---|
| `euclidean` (default) | √Σ(aᵢ−bᵢ)² |
| `squared_euclidean` | Σ(aᵢ−bᵢ)² |
| `manhattan` | Σ\|aᵢ−bᵢ\| |
| `chebyshev` | max\|aᵢ−bᵢ\| |
| `minkowski` / `minkowski(p)` | (Σ\|aᵢ−bᵢ\|ᵖ)^(1/p), p ≥ 1, default 3 |
| `cosine` | 1 − a·b / (\|a\|\|b\|) |
| `canberra` | Σ\|aᵢ−bᵢ\| / (\|aᵢ\|+\|bᵢ\|) |
| `hamming` | share of coordinates that differ |
| `jaccard` | 1 − Σmin(aᵢ,bᵢ) / Σmax(aᵢ,bᵢ) (for non-negative features) |
| `safe_euclidean` | Euclidean that skips `NAN` coordinates and rescales |

The K-d tree models (`KDNeighbors`, `KDNeighborsRegressor`) only support the Minkowski
family: `euclidean`, `squared_euclidean`, `manhattan`, `chebyshev` and `minkowski`.

### 4.7 Error handling

All exceptions thrown by the extension implement `ML\Exception\MLException`. You can catch
them all at once, or by kind (see [16.2](#162-exceptions)):

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

## 5. Classification

All classifiers live in `ML\Classification` and implement `ML\Learner` and `ML\Classifier`.
The probabilistic ones also implement `ML\Probabilistic`. Every classifier has:

| Method | Description |
|---|---|
| `fit(array $samples, array $labels): static` | Train. Labels are `int` or `string`. |
| `predict(array $samples): array\|int\|string` | Predict labels. |
| `predictProbability(array $samples): array` | `label => probability` (probabilistic classifiers). |
| `getClasses(): array` | Distinct labels in sorted order. |

The examples below share this small dataset of three groups:

```php
<?php
// Used by the examples in this section (each example repeats the lines it needs).
$X = [[1.0, 1.1], [1.2, 0.9], [0.8, 1.0], [5.0, 5.2], [5.1, 4.9], [4.9, 5.0], [1.0, 5.0], [1.2, 5.1], [0.9, 4.8]];
$y = ['cat', 'cat', 'cat', 'dog', 'dog', 'dog', 'fox', 'fox', 'fox'];
echo count($X), " samples\n";
```

### 5.1 `LogisticRegression`

Multinomial (softmax) logistic regression trained with full-batch gradient descent. Binary
classification is the case with two classes.

```php
new LogisticRegression(float $learningRate = 0.1, int $maxIterations = 1000, float $l2 = 0.0, float $tolerance = 1e-6)
```

| Parameter | Description |
|---|---|
| `learningRate` | Gradient descent step size (> 0) |
| `maxIterations` | Maximum passes over the data (≥ 1) |
| `l2` | L2 penalty on the weights (≥ 0). The biases are not penalized. |
| `tolerance` | Stops early when the largest absolute gradient falls below this value |

The class also has `getWeights(): array`, which returns `label => feature weights`.

```php
<?php
use ML\Classification\LogisticRegression;

$X = [[0, 0], [0, 1], [1, 0], [1, 1], [4, 4], [4, 5], [5, 4], [5, 5]];
$y = ['no', 'no', 'no', 'no', 'yes', 'yes', 'yes', 'yes'];

$clf = (new LogisticRegression(learningRate: 0.5, maxIterations: 2000))->fit($X, $y);
var_dump($clf->predict([4.8, 5.1]));                        // string(3) "yes"
$p = $clf->predictProbability([2.5, 2.5]);
printf("P(no)=%.3f P(yes)=%.3f\n", $p['no'], $p['yes']);
```

Scale your features (see [`StandardScaler`](#101-scaling-and-normalization)). Gradient descent
converges slowly, or diverges, when features have very different ranges.

### 5.2 `KNearestNeighbors`

k-nearest-neighbours by brute force. `fit()` stores the data, and `predict()` returns the
majority label of the `k` closest samples. Ties go to the class whose neighbours are closer.

```php
new KNearestNeighbors(int $k = 3, string $metric = 'euclidean', bool $weighted = false)
```

| Parameter | Description |
|---|---|
| `k` | Number of neighbours (≥ 1, ≤ training samples) |
| `metric` | Any [distance metric](#46-distance-metrics) |
| `weighted` | Weight votes by inverse distance |

```php
<?php
use ML\Classification\KNearestNeighbors;

$X = [[1, 1], [1, 2], [2, 1], [8, 8], [8, 9], [9, 8]];
$y = ['red', 'red', 'red', 'blue', 'blue', 'blue'];

$knn = (new KNearestNeighbors(3))->fit($X, $y);
echo $knn->predict([1.5, 1.5]), "\n";                       // red
print_r($knn->predictProbability([7, 7]));                   // [blue => 1, red => 0]
echo (new KNearestNeighbors(1, 'manhattan', true))->fit($X, $y)->predict([9, 9]), "\n";   // blue
```

### 5.3 `KDNeighbors`

Same results as `KNearestNeighbors`, but the neighbour search uses a **K-d tree**. That's much
faster on low-dimensional data (roughly up to 20 features) with many training samples.

```php
new KDNeighbors(int $k = 5, bool $weighted = false, int $maxLeafSize = 30, string $metric = 'euclidean')
```

The `metric` must belong to the Minkowski family (euclidean, squared_euclidean, manhattan,
chebyshev, minkowski).

```php
<?php
use ML\Classification\KDNeighbors;
use ML\Datasets;

$data = Datasets::agglomerate(['a' => Datasets::blob([0, 0], 1.0, 500, 1), 'b' => Datasets::blob([4, 4], 1.0, 500, 2)], true, 1);
$kd = (new KDNeighbors(k: 7))->fit($data['samples'], $data['labels']);
echo $kd->predict([3.5, 4.2]), "\n";   // b
```

### 5.4 `RadiusNeighbors`

Votes among **all** training samples within a fixed `radius`. Samples with no neighbour in
range get the `outlierClass` label.

```php
new RadiusNeighbors(float $radius = 1.0, bool $weighted = false, int|string $outlierClass = '?', string $metric = 'euclidean')
```

```php
<?php
use ML\Classification\RadiusNeighbors;

$X = [[1, 1], [1, 2], [2, 1], [8, 8], [8, 9], [9, 8]];
$y = ['red', 'red', 'red', 'blue', 'blue', 'blue'];

$model = (new RadiusNeighbors(radius: 2.0, outlierClass: 'unknown'))->fit($X, $y);
print_r($model->predict([[1.5, 1.5], [50, 50]]));   // [red, unknown]
```

For outliers, `predictProbability()` returns all zeros.

### 5.5 `GaussianNB`

Gaussian naive Bayes. Each continuous feature is modelled per class as a normal distribution.
It trains in a single pass and needs little data.

```php
new GaussianNB(float $varSmoothing = 1e-9)
```

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

### 5.6 `NaiveBayes`

Naive Bayes for **categorical** features. Each feature value is an `int`, `string` or `bool`
category. Categories that weren't seen in training get the smoothed probability of an empty
count.

```php
new NaiveBayes(float $smoothing = 1.0, ?array $priors = null)
```

| Parameter | Description |
|---|---|
| `smoothing` | Additive (Laplace) smoothing (> 0) |
| `priors` | Optional `label => probability` class priors. By default they're learned from the data. |

```php
<?php
use ML\Classification\NaiveBayes;

$X = [['sunny', 'hot'], ['sunny', 'mild'], ['rain', 'mild'], ['rain', 'cool'], ['overcast', 'hot']];
$y = ['no', 'no', 'yes', 'yes', 'yes'];

$nb = (new NaiveBayes())->fit($X, $y);
print_r($nb->predict([['sunny', 'cool'], ['rain', 'hot']]));   // [no, yes]
printf("P(no | sunny, hot) = %.4f\n", $nb->predictProbability(['sunny', 'hot'])['no']);   // 0.7313
```

To use continuous features as well, convert them into bins with
[`IntervalDiscretizer`](#103-categorical-and-mixed-data).

### 5.7 `MultinomialNB`

Multinomial naive Bayes for **count** features, such as word counts or TF-IDF weights. It's
the classic text classifier (see [12](#12-text)).

```php
new MultinomialNB(float $smoothing = 1.0, ?array $priors = null)
```

```php
<?php
use ML\Classification\MultinomialNB;

// columns: counts of "win", "meeting", "free"
$X = [[3, 0, 1], [2, 0, 2], [0, 4, 0], [0, 3, 1]];
$y = ['spam', 'spam', 'ham', 'ham'];
$nb = (new MultinomialNB())->fit($X, $y);
print_r($nb->predict([[1, 0, 1], [0, 2, 0]]));   // [spam, ham]
```

### 5.8 `ClassificationTree`

CART decision tree with Gini impurity. It's easy to interpret: `rules()` prints the learned
if/else rules.

```php
new ClassificationTree(int $maxHeight = PHP_INT_MAX, int $maxLeafSize = 3, float $minPurityIncrease = 1e-7, ?int $maxFeatures = null, ?int $seed = null)
```

| Parameter | Description |
|---|---|
| `maxHeight` | Maximum depth |
| `maxLeafSize` | Nodes with this many samples or fewer become leaves |
| `minPurityIncrease` | Minimum impurity decrease for a split |
| `maxFeatures` | Features considered per split (`null` = all). Randomness uses `seed`. |

Additional methods:

- `getFeatureImportances(): array` (sums to 1)
- `getHeight(): int`
- `getLeafCount(): int`
- `rules(?array $featureNames = null): string`

```php
<?php
use ML\Classification\ClassificationTree;

$X = [[25, 30000], [35, 60000], [45, 80000], [22, 20000], [52, 110000], [30, 40000]];
$y = ['no', 'yes', 'yes', 'no', 'yes', 'no'];

$tree = (new ClassificationTree(maxHeight: 2))->fit($X, $y);
echo $tree->rules(['age', 'income']);
echo $tree->predict([40, 70000]), "\n";                       // yes
print_r($tree->getFeatureImportances());
```

### 5.9 `ExtraTreeClassifier`

Extremely randomized tree: each split uses a random threshold instead of the best one. A
single extra tree is weak on its own. It works best as part of an ensemble
(`RandomForest(extraTrees: true)`), but it's also fast to train.
It has the same parameters and methods as `ClassificationTree`.

```php
<?php
use ML\Classification\ExtraTreeClassifier;

$X = [[1, 1], [1, 2], [2, 1], [8, 8], [8, 9], [9, 8]];
$y = ['a', 'a', 'a', 'b', 'b', 'b'];
echo (new ExtraTreeClassifier(seed: 3))->fit($X, $y)->predict([8.5, 8.5]), "\n";   // b
```

### 5.10 `RandomForest`

A bagged ensemble of decision trees. Each tree sees a bootstrap sample and a random subset of
features; by default that's √d features. Trees are trained in parallel when
`mlnative.threads` > 1.

```php
new RandomForest(int $estimators = 100, float $ratio = 1.0, bool $balanced = false, int $maxHeight = PHP_INT_MAX, int $maxLeafSize = 3, float $minPurityIncrease = 1e-7, ?int $maxFeatures = null, bool $extraTrees = false, ?int $seed = null)
```

| Parameter | Description |
|---|---|
| `estimators` | Number of trees |
| `ratio` | Bootstrap sample size as a fraction of the training set (0–1] |
| `balanced` | Sample every class with equal probability (for imbalanced data) |
| `extraTrees` | Use extremely randomized trees, i.e. an Extra-Trees forest |
| others | Same as `ClassificationTree` |

```php
<?php
use ML\Classification\RandomForest;
use ML\Datasets;
use ML\Metrics;

$data = Datasets::agglomerate([
    'a' => Datasets::blob([0, 0, 0], 1.0, 100, 1),
    'b' => Datasets::blob([2, 2, 0], 1.0, 100, 2),
], true, 1);
[$Xtrain, $Xtest, $ytrain, $ytest] = Datasets::trainTestSplit($data['samples'], $data['labels'], 0.3, true, 1);

$forest = (new RandomForest(estimators: 50, seed: 1))->fit($Xtrain, $ytrain);
printf("accuracy %.2f\n", Metrics::accuracy($ytest, $forest->predict($Xtest)));
print_r(array_map(fn ($v) => round($v, 2), $forest->getFeatureImportances()));   // the third feature is noise
```

### 5.11 `AdaBoost`

AdaBoost (SAMME, multiclass). It's a sequence of weighted decision trees, where each tree
focuses on the samples its predecessors got wrong. By default the trees are stumps (depth 1).

```php
new AdaBoost(int $estimators = 100, float $rate = 1.0, float $ratio = 0.8, int $maxHeight = 1, float $minChange = 1e-4, ?int $seed = null)
```

| Parameter | Description |
|---|---|
| `estimators` | Maximum boosting rounds |
| `rate` | Learning rate (shrinks each tree's vote) |
| `ratio` | Fraction of samples drawn (without replacement) per round |
| `maxHeight` | Depth of each weak learner |
| `minChange` | Stops when the weighted error changes less than this |

Additional methods:

- `getLosses(): array`, the weighted error per round
- `getEstimatorCount(): int`
- `getFeatureImportances(): array`

```php
<?php
use ML\Classification\AdaBoost;

$X = [[1, 1], [2, 1], [1, 2], [6, 5], [5, 6], [6, 6], [1, 6], [2, 5], [1, 5]];
$y = ['a', 'a', 'a', 'b', 'b', 'b', 'c', 'c', 'c'];
$ada = (new AdaBoost(estimators: 20, seed: 1))->fit($X, $y);
print_r($ada->predict([[1.5, 1.5], [5.5, 5.5], [1.5, 5.5]]));   // [a, b, c]
echo $ada->getEstimatorCount(), " rounds\n";
```

### 5.12 `LogitBoost`

LogitBoost (Friedman, Hastie and Tibshirani) for **binary** classification. It's an additive
logistic model built from regression trees and gives well-calibrated probabilities. For more
than two classes, wrap it in [`OneVsRest`](#91-onevsrest).

```php
new LogitBoost(int $estimators = 100, float $rate = 0.1, float $ratio = 0.5, int $maxHeight = 3, float $minChange = 1e-4, ?int $seed = null)
```

```php
<?php
use ML\Classification\LogitBoost;

$X = [[1, 1], [2, 1], [1, 2], [2, 2], [6, 5], [5, 6], [6, 6], [5, 5]];
$y = ['low', 'low', 'low', 'low', 'high', 'high', 'high', 'high'];
$model = (new LogitBoost(seed: 1))->fit($X, $y);
echo $model->predict([5.5, 5.5]), "\n";                       // high
printf("P(high) = %.2f\n", $model->predictProbability([5.5, 5.5])['high']);
```

### 5.13 `SVC`

Support vector classifier. The C-SVM is solved with an SMO solver (the WSS2 algorithm, as in
libsvm) with an LRU kernel cache. More than two classes are handled one-vs-one, and the pairs
are trained in parallel.

```php
new SVC(float $c = 1.0, string $kernel = 'rbf', ?float $gamma = null, int $degree = 3, float $coef0 = 0.0, float $tolerance = 0.001, int $cacheSize = 128)
```

| Parameter | Description |
|---|---|
| `c` | Penalty for margin violations (> 0) |
| `kernel` | `linear`, `polynomial` (`(γ·a·b + coef0)^degree`), `rbf` (`exp(−γ‖a−b‖²)`) or `sigmoid` (`tanh(γ·a·b + coef0)`) |
| `gamma` | Kernel coefficient. `null` = 1 / number of features |
| `tolerance` | Stopping tolerance of the solver |
| `cacheSize` | Kernel cache in MB |

`getSupportVectorCount(): int` returns the number of support vectors kept. Scale your features
before using an SVM.

```php
<?php
use ML\Classification\SVC;
use ML\Datasets;

// Two concentric circles: not linearly separable.
$data = Datasets::agglomerate(['inner' => Datasets::circle(0, 0, 1.0, 0.05, 100, 1), 'outer' => Datasets::circle(0, 0, 3.0, 0.05, 100, 2)], true, 1);
$svc = (new SVC(c: 1.0, kernel: 'rbf'))->fit($data['samples'], $data['labels']);
print_r($svc->predict([[0.2, 0.3], [2.9, 0.4]]));   // [inner, outer]
echo $svc->getSupportVectorCount() > 0 ? "has support vectors\n" : '';
```

### 5.14 `MLPClassifier`

A multilayer perceptron (feed-forward neural network) with a softmax output and cross-entropy
loss. It's trained with mini-batches, and early stopping uses a hold-out set.
[Section 13](#13-neural-networks) describes the layers, optimizers and schedules.

```php
new MLPClassifier(array $hiddenLayers = [], int $batchSize = 128, ?Optimizer $optimizer = null, int $epochs = 1000, float $minChange = 1e-4, int $window = 5, float $holdOut = 0.1, float $l2 = 0.0, ?Schedule $schedule = null, ?int $seed = null)
```

| Parameter | Description |
|---|---|
| `hiddenLayers` | List of `ML\NeuralNet\Layers\*` objects. An empty list gives multinomial logistic regression. |
| `batchSize` | Mini-batch size |
| `optimizer` | `ML\NeuralNet\Optimizers\*`. The default is `Adam(0.001)`. |
| `epochs` | Maximum passes over the training data |
| `minChange` / `window` | Stops when the validation loss hasn't improved by `minChange` for `window` epochs. The best weights are kept. |
| `holdOut` | Fraction of the data held out for early stopping (0 = use the training loss) |
| `l2` | L2 penalty on the output layer |
| `schedule` | Learning-rate schedule (`ML\NeuralNet\Schedules\*`) |

Additional methods:

- `getLosses(): array`, the training loss per epoch
- `getScores(): array`, the validation loss per epoch

```php
<?php
use ML\Classification\MLPClassifier;
use ML\NeuralNet\Layers\{Dense, Activation};
use ML\NeuralNet\Optimizers\Adam;
use ML\Datasets;

$data = Datasets::agglomerate(['inner' => Datasets::circle(0, 0, 1.0, 0.05, 150, 1), 'outer' => Datasets::circle(0, 0, 3.0, 0.05, 150, 2)], true, 1);
$net = new MLPClassifier(
    hiddenLayers: [new Dense(16), new Activation('relu'), new Dense(16), new Activation('relu')],
    batchSize: 32,
    optimizer: new Adam(0.01),
    epochs: 300,
    seed: 1,
);
$net->fit($data['samples'], $data['labels']);
print_r($net->predict([[0.2, 0.3], [2.9, 0.4]]));   // [inner, outer]
echo count($net->getLosses()), " epochs trained\n";
```

---

## 6. Regression

All regressors live in `ML\Regression` and implement `ML\Learner` and `ML\Regressor`:

| Method | Description |
|---|---|
| `fit(array $samples, array $targets): static` | Train. Targets are `int\|float`. |
| `predict(array $samples): array\|float` | Predict values. |

### 6.1 `LinearRegression`

Least squares with an optional L2 (ridge) penalty, solved exactly in one step (Cholesky on the
normal equations of centered data).

```php
new LinearRegression(float $l2 = 0.0)
```

The class also has `getCoefficients(): array` and `getIntercept(): float`.

```php
<?php
use ML\Regression\LinearRegression;
use ML\Metrics;

// [area m², rooms] → price
$X = [[50, 1], [60, 2], [80, 2], [100, 3], [120, 3], [150, 4]];
$y = [150, 185, 235, 295, 345, 430];

$model = (new LinearRegression())->fit($X, $y);
printf("price(90 m², 3 rooms) = %.1f\n", $model->predict([90, 3]));
printf("R² = %.4f\n", Metrics::r2Score($y, $model->predict($X)));

// Perfectly collinear features need a penalty:
$ridge = (new LinearRegression(l2: 0.001))->fit([[1, 2], [2, 4], [3, 6]], [1, 2, 3]);
printf("%.3f\n", $ridge->predict([4, 8]));   // ≈ 4.000
```

`fit()` throws `RuntimeException` when the feature matrix is singular, i.e. when features are
constant or collinear, or when there are fewer samples than features. Use `l2 > 0` in that
case.

### 6.2 `ElasticNet`

Linear regression with combined L1 and L2 penalties, solved by coordinate descent. The L1 part
sets the coefficients of irrelevant features to exactly zero.

```php
new ElasticNet(float $alpha = 1.0, float $l1Ratio = 0.5, int $maxIterations = 1000, float $tolerance = 1e-4)
```

| Parameter | Description |
|---|---|
| `alpha` | Overall penalty strength (≥ 0) |
| `l1Ratio` | Mix between the penalties: 1 = Lasso (L1 only), 0 = Ridge (L2 only) |

The class also has `getCoefficients()` and `getIntercept()`.

```php
<?php
use ML\Regression\ElasticNet;
use ML\Datasets;

// Only the first two of five features matter.
$data = Datasets::hyperplane([3, -2, 0, 0, 0], 1.0, 0.05, 200, 1);
$lasso = (new ElasticNet(alpha: 0.05, l1Ratio: 1.0))->fit($data['samples'], $data['targets']);
print_r(array_map(fn ($c) => round($c, 2), $lasso->getCoefficients()));   // irrelevant features → 0
```

### 6.3 `RegressionTree`

CART regression tree (variance reduction). It has the same parameters and methods as
[`ClassificationTree`](#58-classificationtree), including `rules()` and
`getFeatureImportances()`.

```php
<?php
use ML\Regression\RegressionTree;

$X = [[1], [2], [3], [10], [11], [12]];
$y = [1.0, 1.2, 0.9, 5.0, 5.3, 4.8];
$tree = (new RegressionTree(maxHeight: 1))->fit($X, $y);
echo $tree->rules(['size']);
printf("%.2f\n", $tree->predict([11.5]));   // 5.03 (mean of the right leaf)
```

### 6.4 `ExtraTreeRegressor`

Regression tree with random split thresholds. Like `ExtraTreeClassifier`, it's mostly used
inside ensembles, e.g. `RandomForestRegressor(extraTrees: true)`.

```php
<?php
use ML\Regression\ExtraTreeRegressor;

$X = [[1], [2], [3], [10], [11], [12]];
$y = [1.0, 1.2, 0.9, 5.0, 5.3, 4.8];
printf("%.1f\n", (new ExtraTreeRegressor(seed: 1))->fit($X, $y)->predict([2.5]));   // ≈ 1
```

### 6.5 `RandomForestRegressor`

Bagged regression trees; the prediction is the average of the trees. By default each split
considers d/3 features.

```php
new RandomForestRegressor(int $estimators = 100, float $ratio = 1.0, int $maxHeight = PHP_INT_MAX, int $maxLeafSize = 3, float $minPurityIncrease = 1e-7, ?int $maxFeatures = null, bool $extraTrees = false, ?int $seed = null)
```

```php
<?php
use ML\Regression\RandomForestRegressor;
use ML\Datasets;
use ML\Metrics;

$data = Datasets::hyperplane([2, -1, 0.5], 0.0, 0.1, 300, 1);
$forest = (new RandomForestRegressor(estimators: 50, seed: 1))->fit($data['samples'], $data['targets']);
printf("R² = %.2f\n", Metrics::r2Score($data['targets'], $forest->predict($data['samples'])));
```

### 6.6 `GradientBoost`

Gradient-boosted regression trees with least-squares loss. A hold-out set is used for early
stopping, and only the rounds up to the best validation score are kept.

```php
new GradientBoost(int $estimators = 100, float $rate = 0.1, float $ratio = 0.5, int $maxHeight = 3, int $maxLeafSize = 3, float $minChange = 1e-4, int $window = 5, float $holdOut = 0.1, ?int $seed = null)
```

| Parameter | Description |
|---|---|
| `estimators` | Maximum boosting rounds |
| `rate` | Learning rate (shrinkage) |
| `ratio` | Fraction of the training samples used per round (stochastic gradient boosting) |
| `maxHeight`, `maxLeafSize` | Size of each tree |
| `minChange`, `window` | Stops when the validation MSE hasn't improved by `minChange` for `window` rounds |
| `holdOut` | Fraction of the data held out for early stopping (0 = no early stopping) |

Additional methods:

- `getLosses()`, the training MSE per round
- `getValidationScores()`, the validation MSE per round
- `getEstimatorCount()`
- `getFeatureImportances()`

```php
<?php
use ML\Regression\GradientBoost;
use ML\Datasets;
use ML\Metrics;

$data = Datasets::hyperplane([3, -2, 1], 1.0, 0.1, 400, 1);
[$Xtrain, $Xtest, $ytrain, $ytest] = Datasets::trainTestSplit($data['samples'], $data['targets'], 0.25, false, 1);
$gb = (new GradientBoost(estimators: 300, seed: 1))->fit($Xtrain, $ytrain);
printf("R² = %.2f using %d trees\n", Metrics::r2Score($ytest, $gb->predict($Xtest)), $gb->getEstimatorCount());
```

### 6.7 `KNNRegressor`, `KDNeighborsRegressor` and `RadiusNeighborsRegressor`

These predict the average target of the nearest neighbours, or the inverse-distance weighted
average when `weighted` is true.

```php
new KNNRegressor(int $k = 5, bool $weighted = false, string $metric = 'euclidean')
new KDNeighborsRegressor(int $k = 5, bool $weighted = false, int $maxLeafSize = 30, string $metric = 'euclidean')
new RadiusNeighborsRegressor(float $radius = 1.0, bool $weighted = false, string $metric = 'euclidean')
```

`RadiusNeighborsRegressor` returns `NAN` for samples with no neighbour inside the radius.

```php
<?php
use ML\Regression\{KNNRegressor, KDNeighborsRegressor, RadiusNeighborsRegressor};

$X = [[0], [1], [2], [10], [11]];
$y = [0.0, 1.0, 2.0, 10.0, 11.0];
var_dump((new KNNRegressor(2))->fit($X, $y)->predict([0.4]));                 // float(0.5)
var_dump((new KDNeighborsRegressor(2, weighted: true))->fit($X, $y)->predict([10.9]) > 10.5);   // bool(true)
var_dump(is_nan((new RadiusNeighborsRegressor(0.5))->fit($X, $y)->predict([5])));   // bool(true)
```

### 6.8 `SVR`

Epsilon-insensitive support vector regression, using the same solver and kernels as
[`SVC`](#513-svc). Errors smaller than `epsilon` aren't penalized.

```php
new SVR(float $c = 1.0, float $epsilon = 0.1, string $kernel = 'rbf', ?float $gamma = null, int $degree = 3, float $coef0 = 0.0, float $tolerance = 0.001, int $cacheSize = 128)
```

```php
<?php
use ML\Regression\SVR;
use ML\Metrics;

$X = $y = [];
for ($i = 0; $i < 100; $i++) {
    $X[] = [$i / 16];
    $y[] = sin($i / 16);
}
$svr = (new SVR(c: 10.0, epsilon: 0.05, kernel: 'rbf', gamma: 1.0))->fit($X, $y);
printf("R² on a sine wave = %.3f\n", Metrics::r2Score($y, $svr->predict($X)));
```

### 6.9 `MLPRegressor`

A multilayer perceptron with a single linear output neuron. It works like
[`MLPClassifier`](#514-mlpclassifier) and adds one parameter, `costFn`. That is either
`'least_squares'` (the default) or `'huber(<delta>)'`, a pseudo-Huber loss that is robust to
outliers.

```php
new MLPRegressor(array $hiddenLayers = [], int $batchSize = 128, ?Optimizer $optimizer = null, int $epochs = 1000, float $minChange = 1e-4, int $window = 5, float $holdOut = 0.1, float $l2 = 0.0, string $costFn = 'least_squares', ?Schedule $schedule = null, ?int $seed = null)
```

```php
<?php
use ML\Regression\MLPRegressor;
use ML\NeuralNet\Layers\{Dense, Activation};
use ML\NeuralNet\Optimizers\Adam;
use ML\Metrics;

$X = $y = [];
for ($i = 0; $i < 200; $i++) {
    $x = $i / 100 - 1;           // inputs in [-1, 1]
    $X[] = [$x];
    $y[] = $x * $x;
}
$net = (new MLPRegressor([new Dense(16), new Activation('tanh')], 32, new Adam(0.02), 500, seed: 1))->fit($X, $y);
printf("R² = %.2f\n", Metrics::r2Score($y, $net->predict($X)));
```

### 6.10 `Adaline`

The adaptive linear neuron: a linear model trained by mini-batch gradient descent. It's an
`MLPRegressor` without hidden layers, and it scales to large datasets where the closed form of
`LinearRegression` would be expensive.

```php
new Adaline(int $batchSize = 128, ?Optimizer $optimizer = null, float $l2 = 1e-4, int $epochs = 1000, float $minChange = 1e-4, int $window = 5, float $holdOut = 0.1, string $costFn = 'least_squares', ?int $seed = null)
```

```php
<?php
use ML\Regression\Adaline;
use ML\NeuralNet\Optimizers\Adam;
use ML\Datasets;
use ML\Metrics;

$data = Datasets::hyperplane([2, -1], 0.5, 0.05, 300, 1);
$adaline = (new Adaline(optimizer: new Adam(0.05), seed: 1))->fit($data['samples'], $data['targets']);
printf("R² = %.3f\n", Metrics::r2Score($data['targets'], $adaline->predict($data['samples'])));
```

---

## 7. Clustering

The clusterers live in `ML\Clustering` and implement `ML\Clusterer`. Cluster numbers start at
0, and they're arbitrary: the same grouping can come out with different numbers for a
different seed. Use the clustering metrics in [14.1](#141-metrics) to compare a clustering
with known labels.

There are two kinds of clusterer:

- **Learners** (`KMeans`, `KMedoids`, `FuzzyCMeans`, `GaussianMixture`, `MeanShift`) have
  `fit()`. They can then assign new samples with `predict()`.
- **Transductive clusterers** (`DBSCAN`, `OPTICS`, `Ward`) cluster exactly the samples you
  pass to `predict()`.

The examples use three groups of points:

```php
<?php
use ML\Datasets;

$points = Datasets::agglomerate([
    'a' => Datasets::blob([0, 0], 0.5, 50, 1),
    'b' => Datasets::blob([5, 5], 0.5, 50, 2),
    'c' => Datasets::blob([0, 6], 0.5, 50, 3),
], true, 7)['samples'];
echo count($points), " points\n";   // 150 points
```

### 7.1 `KMeans`

Lloyd's algorithm. `seeder` chooses how the initial centroids are picked:

| Seeder | Description |
|---|---|
| `plusplus` | k-means++ (the default) |
| `random` | Random samples |
| `kmc2` | A fast MCMC approximation of k-means++ |
| `preset` | Your own initial centroids. Passing `centroids` selects this automatically. |

```php
new KMeans(int $clusters = 8, int $maxIterations = 300, float $tolerance = 1e-4, ?int $seed = null, string $seeder = 'plusplus', ?array $centroids = null)
```

Methods:

- `predict()`
- `predictProbability()`, a membership that's proportional to inverse distance
- `getCentroids()`
- `getLabels()`, the cluster of each training sample
- `getInertia()`, the sum of squared distances to the nearest centroid
- `getIterations()`

```php
<?php
use ML\Clustering\KMeans;
use ML\Datasets;

$points = Datasets::agglomerate(['a' => Datasets::blob([0, 0], 0.5, 50, 1), 'b' => Datasets::blob([5, 5], 0.5, 50, 2)], true, 7)['samples'];

$km = (new KMeans(clusters: 2, seed: 1))->fit($points);
echo count(array_unique($km->getLabels())), " clusters\n";   // 2 clusters
echo $km->predict([5.1, 4.8]) === $km->predict([5, 5]) ? "same cluster\n" : "different\n";
printf("inertia %.1f\n", $km->getInertia());

$preset = (new KMeans(2, centroids: [[0, 0], [5, 5]]))->fit($points);
echo $preset->predict([0.2, -0.1]), "\n";   // 0: the cluster seeded at [0, 0]
```

**Choosing the number of clusters (the elbow method):**

```php
<?php
use ML\Clustering\KMeans;
use ML\Datasets;

$points = Datasets::agglomerate(['a' => Datasets::blob([0, 0], 0.5, 40, 1), 'b' => Datasets::blob([5, 5], 0.5, 40, 2), 'c' => Datasets::blob([0, 6], 0.5, 40, 3)], true, 7)['samples'];
foreach (range(1, 5) as $k) {
    printf("k=%d inertia=%8.2f\n", $k, (new KMeans($k, seed: 1))->fit($points)->getInertia());   // drops sharply until k=3
}
```

### 7.2 `KMedoids`

Partitioning around medoids. The cluster centers are **actual samples** (medoids), and any
[distance metric](#46-distance-metrics) works. It's more robust to outliers than k-means.

Training has two phases:

1. Alternating assignment/update steps.
2. The PAM **SWAP** phase, which replaces medoids while that lowers the total distance. It
   escapes local optima, such as an outlier picked as a seed. It runs for up to 5000 samples,
   because it's O(n²) per pass.

```php
new KMedoids(int $clusters = 8, int $maxIterations = 300, string $metric = 'euclidean', string $seeder = 'plusplus', ?int $seed = null)
```

Methods: `predict()`, `getMedoids()`, `getLabels()`, `getInertia()` (the sum of distances) and
`getIterations()`.

```php
<?php
use ML\Clustering\KMedoids;

$points = [[1, 1], [1.2, 0.8], [0.9, 1.1], [8, 8], [8.2, 7.9], [7.8, 8.1], [20, 20]];   // [20, 20] is an outlier
$km = (new KMedoids(2, metric: 'manhattan', seed: 1))->fit($points);
print_r($km->getMedoids());   // [[1, 1], [8, 8]]: real samples, not pulled towards the outlier
```

### 7.3 `FuzzyCMeans`

Fuzzy c-means: every sample belongs to every cluster to some degree. `predictProbability()`
returns those membership degrees.

```php
new FuzzyCMeans(int $clusters = 8, float $fuzz = 2.0, int $maxIterations = 300, float $minChange = 1e-4, ?int $seed = null)
```

`fuzz` must be greater than 1. Larger values give softer memberships. The class also has
`getCentroids()`, `getInertia()` and `getIterations()`.

```php
<?php
use ML\Clustering\FuzzyCMeans;

$points = [[0, 0], [0.2, 0.1], [0.1, 0.3], [4, 4], [4.2, 3.9], [3.9, 4.1]];
$fcm = (new FuzzyCMeans(2, seed: 1))->fit($points);
print_r(array_map(fn ($u) => round($u, 2), $fcm->predictProbability([2, 2])));   // about 0.5 / 0.5
```

### 7.4 `GaussianMixture`

A mixture of Gaussians with diagonal covariances, fitted by expectation-maximization. It gives
soft assignments via `predictProbability()` and models clusters of different sizes and shapes.

```php
new GaussianMixture(int $components = 8, float $smoothing = 1e-9, int $maxIterations = 100, float $minChange = 0.001, ?int $seed = null)
```

Methods: `getMeans()`, `getVariances()`, `getPriors()`, `getLogLikelihood()` and
`getIterations()`.

```php
<?php
use ML\Clustering\GaussianMixture;
use ML\Datasets;

$points = Datasets::agglomerate(['a' => Datasets::blob([0, 0], 0.5, 100, 1), 'b' => Datasets::blob([4, 4], 1.5, 100, 2)], true, 1)['samples'];
$gmm = (new GaussianMixture(components: 2, seed: 1))->fit($points);
print_r(array_map(fn ($p) => round($p, 2), $gmm->getPriors()));   // about [0.5, 0.5]
print_r(array_map(fn ($p) => round($p, 3), $gmm->predictProbability([0, 0])));
```

### 7.5 `MeanShift`

Mean shift with a flat kernel. It **finds the number of clusters itself** by moving each point
towards the densest area within `radius`. If you don't give a radius, it's estimated from a
percentile of the pairwise distances.

```php
new MeanShift(?float $radius = null, float $percentile = 30.0, int $maxIterations = 100, float $minShift = 1e-4, int $maxSeeds = 1000, ?int $seed = null)
```

Methods:

- `predict()`, which assigns the nearest centroid
- `getCentroids()`
- `getRadius()`, the radius that was used
- `MeanShift::estimateRadius(array $samples, float $percentile = 30.0, ?int $seed = null)` (static)

```php
<?php
use ML\Clustering\MeanShift;
use ML\Datasets;

$points = Datasets::agglomerate(['a' => Datasets::blob([0, 0], 0.4, 60, 1), 'b' => Datasets::blob([5, 5], 0.4, 60, 2), 'c' => Datasets::blob([0, 6], 0.4, 60, 3)], true, 7)['samples'];
$ms = (new MeanShift(seed: 1))->fit($points);
echo count($ms->getCentroids()), " clusters found\n";   // 3 clusters found
```

### 7.6 `DBSCAN`

Density-based clustering. It finds clusters of any shape and marks low-density samples as
**noise (-1)**.

- A **core point** has at least `minDensity` samples, itself included, within `radius`.
- **Clusters** grow from core points.

```php
new DBSCAN(float $radius = 0.5, int $minDensity = 5, string $metric = 'euclidean')
```

`predict(array $samples): array` clusters the given samples.

```php
<?php
use ML\Clustering\DBSCAN;

$points = [[0], [0.1], [0.2], [5], [5.1], [5.2], [20]];
print_r((new DBSCAN(radius: 0.5, minDensity: 2))->predict($points));   // [0, 0, 0, 1, 1, 1, -1]
```

### 7.7 `OPTICS`

OPTICS orders the points by density reachability, which reveals the cluster structure at
every radius at once. `predict()` extracts the clusters at `radius`, like DBSCAN.
`reachability()` returns the full ordering for analysis.

```php
new OPTICS(int $minDensity = 5, float $radius = 0.5, float $maxRadius = INF, string $metric = 'euclidean')
```

`reachability(array $samples)` returns these keys:

- `ordering`, the sample indices in visit order
- `reachability`
- `coreDistances`
- `labels`

```php
<?php
use ML\Clustering\OPTICS;

$points = [[0], [0.1], [0.2], [5], [5.1], [5.2]];
$optics = new OPTICS(minDensity: 2, radius: 1.0);
print_r($optics->predict($points));                       // [0, 0, 0, 1, 1, 1]
print_r($optics->reachability($points)['ordering']);
```

### 7.8 `Ward`

Agglomerative (hierarchical) clustering with Ward linkage: it repeatedly merges the two
clusters whose merge increases the total variance the least. It uses the nearest-neighbour
chain algorithm, which needs O(n²) time and O(n) memory.

```php
new Ward(int $clusters = 2)
```

```php
<?php
use ML\Clustering\Ward;

print_r((new Ward(2))->predict([[0], [0.1], [0.2], [10], [10.1]]));   // [0, 0, 0, 1, 1]
```

---

## 8. Anomaly detection

The anomaly detectors live in `ML\AnomalyDetection`. All of them implement `ML\Learner` and
`ML\AnomalyDetector`, and share this interface:

| Method | Description |
|---|---|
| `fit(array $samples): static` | Learn what "normal" looks like (no labels). |
| `predict(array $samples): array\|int` | **1 = anomaly**, 0 = inlier. |
| `score(array $samples): array\|float` | Anomaly score; higher means more anomalous. |
| `getThreshold(): float` | The score above which a sample is an anomaly. |

Detectors with a `contamination` parameter set the threshold so that this fraction of the
**training** data is flagged.

The examples use 200 normal points plus three outliers:

```php
<?php
use ML\Datasets;

$normal = Datasets::blob([0, 0], 1.0, 200, 1);
$data = array_merge($normal, [[8, 8], [-7, 6], [9, -9]]);
echo count($data), " samples\n";
```

### 8.1 `RobustZScore`

The robust z-score is `0.6745·|x − median| / MAD` per feature. The scores of all features are
combined as `beta·max + (1−beta)·mean`. Because it uses the median and MAD, the outliers
themselves don't distort it.

```php
new RobustZScore(float $threshold = 3.5, float $beta = 0.5, float $smoothing = 1e-9)
```

```php
<?php
use ML\AnomalyDetection\RobustZScore;
use ML\Datasets;

$data = array_merge(Datasets::blob([0, 0], 1.0, 200, 1), [[8, 8], [-7, 6], [9, -9]]);
$detector = (new RobustZScore())->fit($data);
print_r($detector->predict([[0.5, -0.3], [8, 8]]));   // [0, 1]
```

### 8.2 `GaussianMLE`

Fits one Gaussian per feature. The score is the negative log-likelihood.

```php
new GaussianMLE(float $contamination = 0.1, float $smoothing = 1e-9)
```

```php
<?php
use ML\AnomalyDetection\GaussianMLE;
use ML\Datasets;

$data = array_merge(Datasets::blob([0, 0], 1.0, 200, 1), [[8, 8], [-7, 6], [9, -9]]);
$detector = (new GaussianMLE(contamination: 0.02))->fit($data);
print_r(array_slice($detector->predict($data), -3));   // [1, 1, 1]
```

### 8.3 `IsolationForest`

Isolation Forest (Liu et al.) isolates samples with random splits; anomalies need fewer splits.
Scores lie in (0, 1). Without `contamination`, samples scoring above 0.5 are anomalies. The
trees are built in parallel.

```php
new IsolationForest(int $estimators = 100, int $maxSamples = 256, ?float $contamination = null, ?int $seed = null)
```

```php
<?php
use ML\AnomalyDetection\IsolationForest;
use ML\Datasets;

$forest = (new IsolationForest(seed: 1))->fit(Datasets::blob([0, 0], 1.0, 300, 1));
print_r(array_map(fn ($s) => round($s, 2), $forest->score([[0, 0], [8, 8]])));   // low vs high
print_r($forest->predict([[0, 0], [8, 8]]));                                    // [0, 1]
```

### 8.4 `LocalOutlierFactor`

LOF compares a sample's local density with the density of its `k` neighbours. Points that lie
in a sparser area than their neighbours score above 1. It finds local anomalies that global
methods miss.

```php
new LocalOutlierFactor(int $k = 20, float $contamination = 0.1, string $metric = 'euclidean')
```

```php
<?php
use ML\AnomalyDetection\LocalOutlierFactor;
use ML\Datasets;

$data = array_merge(Datasets::blob([0, 0], 1.0, 200, 1), [[8, 8], [-7, 6], [9, -9]]);
$lof = (new LocalOutlierFactor(k: 10, contamination: 0.02))->fit($data);
print_r(array_slice($lof->predict($data), -3));   // [1, 1, 1]
```

### 8.5 `Loda`

Loda (lightweight on-line detector of anomalies) builds histograms of the data along many
random sparse projections. The score is the average negative log-density. It's very fast and
suits high-dimensional data.

```php
new Loda(int $estimators = 100, ?int $bins = null, float $contamination = 0.1, ?int $seed = null)
```

```php
<?php
use ML\AnomalyDetection\Loda;
use ML\Datasets;

$data = array_merge(Datasets::blob([0, 0, 0, 0], 1.0, 300, 1), [[9, 9, 9, 9]]);
$loda = (new Loda(contamination: 0.01, seed: 1))->fit($data);
echo $loda->predict([9, 9, 9, 9]), "\n";   // 1
```

### 8.6 `OneClassSVM`

The one-class SVM (Schölkopf) learns a boundary around the normal data with a kernel. `nu` is
an upper bound on the fraction of training samples outside the boundary. The score is −f(x),
and the threshold is 0.

```php
new OneClassSVM(float $nu = 0.1, string $kernel = 'rbf', ?float $gamma = null, int $degree = 3, float $coef0 = 0.0, float $tolerance = 0.001, int $cacheSize = 128)
```

```php
<?php
use ML\AnomalyDetection\OneClassSVM;
use ML\Datasets;

$svm = (new OneClassSVM(nu: 0.05, gamma: 0.1))->fit(Datasets::blob([0, 0], 1.0, 200, 1));
print_r($svm->predict([[0.3, 0.2], [7, 7]]));   // [0, 1]
```

Training on data that already contains a few extreme outliers works. With a very large
`gamma`, though, isolated outliers can end up exactly on the boundary (score ≈ 0). If that
happens, use a smaller `gamma` or a larger `nu`.

---

## 9. Ensembles, pipelines and model selection

These classes wrap other estimators. They work with **any** object that has the right
methods: native models, and your own PHP classes that implement `ML\Learner` plus
`ML\Classifier` or `ML\Regressor`.

Their state is stored in PHP properties, which has two consequences:

- `serialize()` stores the wrapped models too.
- `clone` makes a deep copy, so cloning a pipeline also clones its transformers and estimator.

### 9.1 `OneVsRest`

`ML\Ensemble\OneVsRest` turns a binary classifier into a multiclass one. It trains one
"class k vs. the rest" model per class and predicts the class with the highest score. If the
base model is probabilistic, its probabilities are used as the scores.

```php
new OneVsRest(ML\Classifier $base)
```

```php
<?php
use ML\Ensemble\OneVsRest;
use ML\Classification\LogisticRegression;

$X = [[1, 1], [1.2, 0.8], [5, 5], [5.2, 4.9], [1, 5], [1.1, 5.2]];
$y = ['a', 'a', 'b', 'b', 'c', 'c'];
$ovr = (new OneVsRest(new LogisticRegression(learningRate: 0.5)))->fit($X, $y);
echo $ovr->predict([5.1, 5.0]), "\n";                          // b
print_r(array_map(fn ($p) => round($p, 2), $ovr->predictProbability([1, 5])));   // c is the most likely
```

### 9.2 `BootstrapAggregator`

`ML\Ensemble\BootstrapAggregator` (bagging) trains `estimators` copies of any learner on
bootstrap samples and combines them:

- **regressors:** the average
- **classifiers and anomaly detectors:** a majority vote

```php
new BootstrapAggregator(ML\Learner $base, int $estimators = 10, float $ratio = 0.5, ?int $seed = null)
```

`fit(array $samples, ?array $targets = null)`: targets are optional for unsupervised base
models.

```php
<?php
use ML\Ensemble\BootstrapAggregator;
use ML\Regression\RegressionTree;
use ML\Datasets;
use ML\Metrics;

$data = Datasets::hyperplane([1, -1], 0.0, 0.1, 200, 1);
$bag = (new BootstrapAggregator(new RegressionTree(), estimators: 20, ratio: 1.0, seed: 1))->fit($data['samples'], $data['targets']);
printf("R² = %.2f\n", Metrics::r2Score($data['targets'], $bag->predict($data['samples'])));
```

### 9.3 `CommitteeMachine`

`ML\Ensemble\CommitteeMachine` combines different expert models of the same kind with optional
`influences` (weights):

- **classifiers:** a weighted vote
- **regressors:** a weighted average

```php
new CommitteeMachine(array $experts, ?array $influences = null)
```

```php
<?php
use ML\Ensemble\CommitteeMachine;
use ML\Classification\{KNearestNeighbors, GaussianNB, ClassificationTree};

// A user-written expert: any class implementing ML\Learner + ML\Classifier works.
final class AlwaysA implements ML\Learner, ML\Classifier
{
    public function fit(array $samples, array $labels): static { return $this; }
    public function predict(array $samples): array { return array_fill(0, count($samples), 'a'); }
}

$X = [[1, 1], [1.2, 0.8], [0.9, 1.1], [5, 5], [5.2, 4.9], [4.8, 5.1]];
$y = ['a', 'a', 'a', 'b', 'b', 'b'];
$committee = (new CommitteeMachine([new KNearestNeighbors(3), new GaussianNB(), new ClassificationTree(), new AlwaysA()], [2, 2, 2, 1]))->fit($X, $y);
echo $committee->predict([5, 5]), "\n";   // b (the three real experts outvote AlwaysA)
print_r($committee->getInfluences());      // normalized weights
```

### 9.4 `Pipeline`

`ML\Pipeline` chains transformers and an estimator into one model:

- **`fit()`** fits each stateful transformer in turn, transforms the data, and then fits the
  estimator.
- **`predict()`**, **`predictProbability()`** and **`score()`** apply the same
  transformations first.

Supervised transformers, such as `LinearDiscriminantAnalysis`, get the targets automatically.
Save the whole pipeline to keep the preprocessing and the model together.

```php
new Pipeline(array $transformers, ML\Estimator $estimator)
```

Methods:

- `fit(array $samples, ?array $targets = null)`
- `predict()`, `predictProbability()`, `score()`
- `transform()`, which applies only the transformers
- `getEstimator()`, `getTransformers()`

```php
<?php
use ML\Pipeline;
use ML\Preprocessing\{MissingDataImputer, StandardScaler};
use ML\Classification\LogisticRegression;

// [age, income] with missing values
$X = [[22, 18000], [25, null], [30, 25000], [45, 90000], [null, 120000], [38, 85000]];
$y = ['basic', 'basic', 'basic', 'premium', 'premium', 'premium'];

$model = (new Pipeline([new MissingDataImputer(), new StandardScaler()], new LogisticRegression(learningRate: 0.5)))->fit($X, $y);
print_r($model->predict([[28, 23000], [47, null]]));   // [basic, premium]

$copy = unserialize(serialize($model));   // scaler, imputer and model saved together
echo $copy->predict([50, 100000]), "\n";   // premium
```

### 9.5 `GridSearch`

`ML\ModelSelection\GridSearch` tries every combination of the given constructor parameters and
cross-validates each one. It then refits the best combination on all the data, and the search
object itself predicts with that best model.

```php
new GridSearch(string $class, array $grid, ?ML\CrossValidation\Validator $validator = null, string|callable|null $metric = null)
```

| Parameter | Description |
|---|---|
| `class` | Class name of a learner, e.g. `RandomForest::class` |
| `grid` | Constructor parameter name ⇒ list of values. A single non-array value is used as is. |
| `validator` | The default is `new KFold(3)`. |
| `metric` | A metric name (see [14.4](#144-metric-names-for-validators-and-searches)), a callable `fn(array $expected, array $predicted): float`, or `null` (accuracy for classifiers, R² for regressors). Error metrics such as `mse` are minimized. |

Methods:

- `getBest()`, which returns `['params' => [...], 'score' => float]`
- `getResults()`, with the scores of every combination
- `getEstimator()`, the refit best model
- `predict()`, `predictProbability()`, `score()`

```php
<?php
use ML\ModelSelection\GridSearch;
use ML\CrossValidation\KFold;
use ML\Classification\KNearestNeighbors;
use ML\Datasets;

$data = Datasets::agglomerate(['a' => Datasets::blob([0, 0], 1.0, 40, 1), 'b' => Datasets::blob([2.5, 2.5], 1.0, 40, 2)], true, 1);
$search = new GridSearch(KNearestNeighbors::class, ['k' => [1, 5, 15], 'weighted' => [false, true]], new KFold(4, stratify: true, seed: 1));
$search->fit($data['samples'], $data['labels']);
print_r($search->getBest()['params']);
echo count($search->getResults()), " combinations tried\n";   // 6 combinations tried
echo $search->predict([2.4, 2.6]), "\n";                       // b
```

### 9.6 `BayesianSearch`

`ML\ModelSelection\BayesianSearch` tunes hyper-parameters with Bayesian optimization:

1. It evaluates a few random configurations.
2. It fits a Gaussian-process model of the score.
3. It picks each next configuration by **expected improvement**.

It usually finds good settings in far fewer trials than a grid.

```php
new BayesianSearch(string $class, array $space, int $trials = 10, ?Validator $validator = null, string|callable|null $metric = null, ?int $seed = null)
```

Each entry of `space` is one of:

- `['min' => a, 'max' => b]`: a numeric range. It's an integer range when both bounds are
  ints. Add `'log' => true` to search on a log scale.
- a list of choices, e.g. `['euclidean', 'manhattan']`
- a scalar, which is used as a fixed value

```php
<?php
use ML\ModelSelection\BayesianSearch;
use ML\CrossValidation\HoldOut;
use ML\Regression\GradientBoost;
use ML\Datasets;

$data = Datasets::hyperplane([2, -1, 0.5], 0.0, 0.1, 200, 1);
$search = new BayesianSearch(GradientBoost::class, [
    'rate' => ['min' => 0.01, 'max' => 0.5, 'log' => true],
    'maxHeight' => ['min' => 1, 'max' => 5],
    'ratio' => [0.5, 0.8, 1.0],
    'seed' => 1,
], trials: 8, validator: new HoldOut(0.3, seed: 1), metric: 'r2', seed: 1);
$search->fit($data['samples'], $data['targets']);
printf("best R² %.3f with %s\n", $search->getBest()['score'], json_encode($search->getBest()['params']));
```

---

## 10. Preprocessing

Transformers live in `ML\Preprocessing` and implement `ML\Transformer`. **Stateful**
transformers also implement `ML\Stateful` and learn from data with `fit()`:

| Method | Description |
|---|---|
| `fit(array $samples): static` | Learn parameters (stateful transformers only). |
| `transform(array $samples): array` | Transform a list of samples, or a single sample. |
| `fitTransform(array $samples): array` | `fit()` then `transform()` on the same data. |

**Always fit transformers on the training data only**, then call `transform()` on the test or
production data. Use [`Pipeline`](#94-pipeline) to do this automatically.

### 10.1 Scaling and normalization

| Class | Transformation |
|---|---|
| `StandardScaler` | `(x − mean) / std` per feature. Also has `getMeans()` and `getScales()`. |
| `MinMaxNormalizer(float $min = 0.0, float $max = 1.0)` | Rescales each feature to `[min, max]` |
| `MaxAbsoluteScaler` | `x / max\|x\|` per feature (keeps sign and sparsity) |
| `RobustStandardizer(bool $center = true)` | `(x − median) / MAD`, so outliers have little effect |
| `PowerTransformer(?array $lambdas = null, bool $normalize = true)` | Yeo-Johnson transform to make features more Gaussian. The lambdas are estimated by maximum likelihood unless you give them, and `getLambdas()` returns them. |
| `L1Normalizer` | Divides each **sample** by its L1 norm (stateless) |
| `L2Normalizer` | Divides each **sample** by its L2 norm (stateless) |

Constant features are left unscaled, i.e. their scale is 1.

```php
<?php
use ML\Preprocessing\{StandardScaler, MinMaxNormalizer, MaxAbsoluteScaler, RobustStandardizer, PowerTransformer, L2Normalizer};

$X = [[1, 10, -5], [2, 20, 0], [3, 30, 5], [100, 40, 10]];
$round = fn ($rows) => json_encode(array_map(fn ($r) => array_map(fn ($v) => round($v, 2), $r), $rows));

echo $round((new StandardScaler())->fitTransform($X)), "\n";
echo $round((new MinMaxNormalizer())->fitTransform($X)), "\n";
echo $round((new MaxAbsoluteScaler())->fitTransform($X)), "\n";
echo $round((new RobustStandardizer())->fitTransform($X)), "\n";   // the outlier 100 doesn't squash the rest
echo $round((new PowerTransformer())->fitTransform($X)), "\n";
echo json_encode((new L2Normalizer())->transform([3, 4])), "\n";     // [0.6,0.8]
```

### 10.2 Feature expansion

`PolynomialExpander(int $degree = 2)` adds the powers x, x², …, x^degree of every feature, so
a linear model can fit curves. It's stateless.

```php
<?php
use ML\Preprocessing\PolynomialExpander;
use ML\Regression\LinearRegression;

echo json_encode((new PolynomialExpander(3))->transform([2, 3])), "\n";   // [2,4,8,3,9,27]

// Fit y = x² with a linear model on expanded features.
$X = [[-2], [-1], [0], [1], [2], [3]];
$expanded = (new PolynomialExpander(2))->transform($X);
$model = (new LinearRegression())->fit($expanded, [4, 1, 0, 1, 4, 9]);
printf("%.2f\n", $model->predict((new PolynomialExpander(2))->transform([4])));   // 16.00
```

### 10.3 Categorical and mixed data

These transformers accept samples that mix numbers, strings, booleans and missing values.

| Class | Description |
|---|---|
| `OneHotEncoder(?array $columns = null)` | Replaces each categorical column with one 0/1 column per category. By default, every column that holds strings or booleans is categorical; `columns` lists the column indices explicitly. Numeric columns pass through. Unseen categories encode as all zeros. `getCategories()` returns them. |
| `IntervalDiscretizer(int $bins = 5, bool $equiWidth = true)` | Turns numeric columns into categories named `a`, `b`, `c`, …: equal-width bins, or equal-frequency bins when `equiWidth` is false. Non-numeric columns pass through. |
| `NumericStringConverter` | Converts numeric strings (`"12"`, `"3.5"`) to int/float and `"NAN"` to `NAN`. It's stateless. |
| `BooleanConverter(int\|string $trueValue = 'true', int\|string $falseValue = 'false')` | Replaces booleans with the given values. It's stateless. |

```php
<?php
use ML\Preprocessing\{OneHotEncoder, IntervalDiscretizer, NumericStringConverter, BooleanConverter};

$encoder = new OneHotEncoder();
echo json_encode($encoder->fitTransform([['red', 1.5, true], ['blue', 2, false], ['red', 3, true]])), "\n";
// [[0,1,1.5,0,1],[1,0,2,1,0],[0,1,3,0,1]]
echo json_encode($encoder->getCategories()), "\n";            // {"0":["blue","red"],"2":[0,1]}

echo json_encode((new IntervalDiscretizer(3))->fitTransform([[1], [5], [9]])), "\n";   // [["a"],["b"],["c"]]
echo json_encode((new NumericStringConverter())->transform(['12', '3.5', 'abc'])), "\n";   // [12,3.5,"abc"]
echo json_encode((new BooleanConverter(1, 0))->transform([true, false])), "\n";          // [1,0]
```

### 10.4 Missing values

A value counts as **missing** when it's `null`, `NAN`, or a string equal to `placeholder`
(default `'?'`).

**`MissingDataImputer(string $continuous = 'mean', string $categorical = 'most_frequent', string $placeholder = '?', ?int $seed = null)`**
fills missing values per column.

Continuous strategies:

| Strategy | Fills missing values with |
|---|---|
| `mean` | The column mean |
| `median` | The column median |
| `percentile(<p>)` | The p-th percentile, p in [0, 100] |
| `constant(<v>)` | The number v |
| `wild_guess` | A random value between the column min and max |

Categorical strategies:

| Strategy | Fills missing values with |
|---|---|
| `most_frequent` | The most common category |
| `k_most_frequent(<k>)` | A random pick among the k most common categories |
| `prior` | A random category, sampled by frequency |
| `constant(<value>)` | The given value |

**`KNNImputer(int $k = 5, bool $weighted = false, string $placeholder = '?', ?int $seed = null)`**
fills each gap from the k most similar complete training rows (Gower-style distance):

- **numeric columns:** the mean of those rows
- **categorical columns:** a vote among those rows

**`HotDeckImputer(int $k = 5, bool $weighted = false, string $placeholder = '?', ?int $seed = null)`**
copies the value from one random donor among the k most similar complete rows.

```php
<?php
use ML\Preprocessing\{MissingDataImputer, KNNImputer, HotDeckImputer};

$data = [[1, 'a'], [null, 'b'], [3, '?'], [10, 'a'], [NAN, 'a']];
echo json_encode((new MissingDataImputer('median', 'most_frequent'))->fitTransform($data)), "\n";
// [[1,"a"],[3,"b"],[3,"a"],[10,"a"],[3,"a"]]

$rows = [[1, 1, 'small'], [1.1, null, '?'], [5, 5, 'large'], [5.2, 5.1, 'large'], [0.9, 1.2, 'small']];
echo json_encode((new KNNImputer(k: 2))->fitTransform($rows)[1]), "\n";          // [1.1,1.1,"small"]
echo json_encode((new HotDeckImputer(k: 1, seed: 1))->fitTransform($rows)[1]), "\n";
```

### 10.5 `LambdaFunction`

`LambdaFunction(callable $callback, mixed $context = null)` applies your own function to each
sample, which is useful inside a `Pipeline`. The callback is called as
`$callback(array $sample, int $index, mixed $context)` and must return the new sample.

A pipeline that contains a closure can't be serialized, because PHP can't serialize closures.

```php
<?php
use ML\Preprocessing\LambdaFunction;

$addRatio = new LambdaFunction(fn (array $s) => [...$s, $s[0] / max($s[1], 1e-9)]);
echo json_encode($addRatio->transform([[6, 3], [1, 4]])), "\n";   // [[6,3,2],[1,4,0.25]]
```

---

## 11. Dimensionality reduction

These classes live in `ML\Preprocessing` and work like the other transformers.

### 11.1 `PCA`

Principal component analysis: it projects the data onto the directions of greatest variance.

```php
new PCA(int $dimensions = 2)
```

Methods:

- `getExplainedVariance()`
- `getExplainedVarianceRatio()`
- `getComponents()`, one row per component

```php
<?php
use ML\Preprocessing\PCA;

$line = [];
for ($i = -5; $i <= 5; $i++) {
    $line[] = [$i, 2 * $i + 0.01 * ($i % 2)];   // almost on the line y = 2x
}
$pca = (new PCA(1))->fit($line);
printf("explained: %.4f\n", $pca->getExplainedVarianceRatio()[0]);   // ≈ 1
echo json_encode(array_map(fn ($v) => round($v, 2), $pca->getComponents()[0])), "\n";   // [0.45,0.89]
```

### 11.2 `TruncatedSVD`

A truncated singular value decomposition, **without centering**. That suits sparse count data
such as TF-IDF, where it's known as latent semantic analysis.

```php
new TruncatedSVD(int $dimensions = 2)
```

Methods: `getSingularValues()`, `getExplainedVarianceRatio()` and `getComponents()`.

```php
<?php
use ML\Preprocessing\TruncatedSVD;

$counts = [[2, 1, 0, 0], [3, 1, 0, 0], [0, 0, 4, 2], [0, 0, 3, 3]];
$svd = (new TruncatedSVD(2))->fit($counts);
echo json_encode(array_map(fn ($r) => array_map(fn ($v) => round($v, 2), $r), $svd->transform($counts))), "\n";
```

### 11.3 `LinearDiscriminantAnalysis`

Fisher LDA is **supervised**: it finds the projection that best separates the classes. It has
at most `classes − 1` dimensions. `fit()` and `fitTransform()` take the labels as a second
argument, and `Pipeline` passes them automatically.

```php
new LinearDiscriminantAnalysis(int $dimensions = 1)
```

```php
<?php
use ML\Preprocessing\LinearDiscriminantAnalysis;

$X = [[1, 2, 0.5], [1.5, 1.8, 0.4], [1.2, 2.2, 0.6], [4, 4.5, 0.5], [4.2, 4.1, 0.4], [3.8, 4.4, 0.6]];
$y = ['a', 'a', 'a', 'b', 'b', 'b'];
$lda = new LinearDiscriminantAnalysis(1);
echo json_encode(array_map(fn ($r) => round($r[0], 2), $lda->fitTransform($X, $y))), "\n";   // two well-separated groups
```

### 11.4 `GaussianRandomProjector` and `SparseRandomProjector`

Random projections approximately preserve distances (Johnson–Lindenstrauss) at a tiny cost,
which makes them useful for very wide data.

- `GaussianRandomProjector(int $dimensions, ?int $seed = null)` uses N(0, 1/k) entries.
- `SparseRandomProjector(int $dimensions, float $sparsity = 3.0, ?int $seed = null)` uses
  Achlioptas' sparse ±√(s/k) entries.

`::minDimensions(int $samples, float $maxDistortion = 0.5)` returns the number of dimensions
needed to keep pairwise distances within the given distortion.

```php
<?php
use ML\Preprocessing\{GaussianRandomProjector, SparseRandomProjector};

echo GaussianRandomProjector::minDimensions(1000, 0.3), "\n";   // 768
$wide = [array_fill(0, 50, 1.0), array_fill(0, 50, 2.0)];
echo count((new SparseRandomProjector(5, seed: 1))->fitTransform($wide)[0]), "\n";   // 5
```

### 11.5 `TSNE`

t-SNE embeds high-dimensional data in 2 or 3 dimensions for visualization. It keeps nearby
points close together.

```php
new TSNE(int $dimensions = 2, float $rate = 100.0, float $perplexity = 30.0, float $exaggeration = 12.0, int $epochs = 1000, float $minGradient = 1e-7, ?int $seed = null)
```

A few things to know:

- t-SNE has no separate `fit()`: `transform()` and `fitTransform()` both embed the samples you
  pass.
- `getLosses()` returns the KL divergence, recorded every 10 epochs.
- It's exact, i.e. O(n²) per epoch, so use it for up to a few thousand samples.
- `perplexity` must be smaller than the number of samples − 1.

```php
<?php
use ML\Preprocessing\TSNE;
use ML\Datasets;
use ML\Metrics;

$data = Datasets::agglomerate(['a' => Datasets::blob([0, 0, 0, 0], 0.3, 30, 1), 'b' => Datasets::blob([3, 3, 3, 3], 0.3, 30, 2)], true, 1);
$embedding = (new TSNE(dimensions: 2, perplexity: 10, epochs: 300, seed: 1))->fitTransform($data['samples']);
printf("trustworthiness %.2f\n", Metrics::trustworthiness($data['samples'], $embedding, 5));
```

---

## 12. Text

Text classes live in `ML\Text`. Text transformers take a **list of documents** (strings).

### 12.1 Tokenizers

Tokenizers implement `ML\Text\Tokenizer` and have `tokenize(string $text): array`.

| Class | Output |
|---|---|
| `WordTokenizer` | Words: letters, digits, `_` and inner apostrophes. Non-Latin scripts (Persian, Arabic, CJK, …) are supported, and Unicode punctuation such as `،` `؟` `«»` `。` separates words. |
| `WhitespaceTokenizer(string $delimiter = ' ')` | Splits on the delimiter |
| `SentenceTokenizer` | Sentences ending in `.`, `!` or `?` |
| `NGramTokenizer(int $min = 2, int $max = 2)` | Word n-grams from `min` to `max` words |
| `KSkipNGramTokenizer(int $min = 2, int $max = 2, int $skip = 2)` | n-grams that may skip up to `skip` words |
| `WordStemmer` | Lowercased, Porter-stemmed words |

`PorterStemmer::stem(string $word): string` stems a single English word.

```php
<?php
use ML\Text\{WordTokenizer, SentenceTokenizer, NGramTokenizer, KSkipNGramTokenizer, WordStemmer, PorterStemmer};

echo json_encode((new WordTokenizer())->tokenize("Don't stop, I'm fine!")), "\n";      // ["Don't","stop","I'm","fine"]
echo json_encode((new WordTokenizer())->tokenize('سلام دنیا، خوبی؟'), JSON_UNESCAPED_UNICODE), "\n";   // ["سلام","دنیا","خوبی"]
echo json_encode((new SentenceTokenizer())->tokenize('Hi there. How are you?')), "\n"; // ["Hi there.","How are you?"]
echo json_encode((new NGramTokenizer(1, 2))->tokenize('the quick fox')), "\n";        // ["the","quick","fox","the quick","quick fox"]
echo json_encode((new KSkipNGramTokenizer(2, 2, 1))->tokenize('a b c')), "\n";        // ["a b","a c","b c"]
echo json_encode((new WordStemmer())->tokenize('Running runners connected')), "\n";   // ["run","runner","connect"]
echo PorterStemmer::stem('generalizations'), "\n";                                    // gener
```

Your own tokenizer only has to implement the interface and the `tokenize()` method:

```php
<?php
final class CommaTokenizer implements ML\Text\Tokenizer
{
    public function tokenize(string $text): array
    {
        return array_map('trim', explode(',', $text));
    }
}

$vectorizer = (new ML\Text\WordCountVectorizer(tokenizer: new CommaTokenizer()))->fit(['red, green', 'green, blue']);
print_r($vectorizer->getVocabulary());   // [blue, green, red]
```

### 12.2 Filters and normalizers

| Class | Description |
|---|---|
| `TextNormalizer(bool $uppercase = false, bool $trim = true)` | ASCII lower/upper case. With `trim`, it also collapses runs of whitespace. |
| `MultibyteTextNormalizer(bool $uppercase = false, bool $trim = true)` | Unicode-aware case conversion through `mbstring` |
| `RegexFilter(array $patterns)` | Removes every match of the PCRE patterns. Patterns are validated in the constructor. Built-in constants: `URL`, `EMAIL`, `MENTION`, `HASHTAG`, `NUMBER`, `PUNCTUATION`, `EMOJI`. |
| `StopWordFilter(array $stopWords)` | Removes whole words in the list (case-insensitive, ignoring surrounding punctuation) |

```php
<?php
use ML\Text\{TextNormalizer, RegexFilter, StopWordFilter};

$docs = ["Check https://example.com @bob #news NOW", "The cat is on a mat."];
$docs = (new RegexFilter([RegexFilter::URL, RegexFilter::MENTION, RegexFilter::HASHTAG]))->transform($docs);
$docs = (new TextNormalizer())->transform($docs);
$docs = (new StopWordFilter(['the', 'is', 'a', 'on']))->transform($docs);
print_r($docs);   // ["check now", "cat mat."]
```

### 12.3 Vectorizers and term weighting

| Class | Description |
|---|---|
| `WordCountVectorizer(int $maxVocabularySize = PHP_INT_MAX, int $minDocumentCount = 1, float $maxDocumentRatio = 1.0, ?Tokenizer $tokenizer = null)` | Learns a vocabulary and turns documents into word-count vectors. `getVocabulary()` returns it in alphabetical order. Words in more than `maxDocumentRatio` of the documents are dropped. |
| `TokenHashingVectorizer(int $dimensions = 1024, ?Tokenizer $tokenizer = null, string $hash = 'murmur3')` | Stateless counts in a fixed number of hashed buckets (`murmur3`, `crc32` or `fnv1a`). Use it for unbounded vocabularies or streaming. |
| `TfIdfTransformer(float $smoothing = 1.0, bool $dampening = false)` | Multiplies counts by `idf = log((n + s) / (df + s)) + 1`. `dampening` uses `1 + log(tf)`. |
| `BM25Transformer(float $dampening = 1.2, float $normalization = 0.75)` | Okapi BM25 weighting (k1 = dampening, b = normalization) with document-length normalization |

```php
<?php
use ML\Pipeline;
use ML\Text\{WordCountVectorizer, TfIdfTransformer};
use ML\Classification\MultinomialNB;

$texts = ['win money now', 'cheap pills win', 'meeting at noon', 'lunch with team', 'win a free prize', 'project meeting notes'];
$labels = ['spam', 'spam', 'ham', 'ham', 'spam', 'ham'];

$spamFilter = (new Pipeline([new WordCountVectorizer(), new TfIdfTransformer()], new MultinomialNB()))->fit($texts, $labels);
print_r($spamFilter->predict(['free money', 'team lunch']));   // [spam, ham]
```

---

## 13. Neural networks

[`MLPClassifier`](#514-mlpclassifier), [`MLPRegressor`](#69-mlpregressor) and
[`Adaline`](#610-adaline) are configured with objects from `ML\NeuralNet`. The output layer is
added automatically: softmax for classifiers, and one linear neuron for regressors. Neural
networks need **scaled features** to train well, so use `StandardScaler`.

### 13.1 Layers (`ML\NeuralNet\Layers`)

| Layer | Description |
|---|---|
| `Dense(int $neurons, float $l2 = 0.0, bool $bias = true, string $initializer = 'he')` | Fully connected layer |
| `Activation(string $function = 'relu')` | Element-wise activation (see below) |
| `Dropout(float $ratio = 0.5)` | Randomly zeroes inputs during training |
| `BatchNorm(float $decay = 0.1)` | Batch normalization with a learned scale and shift. `decay` is the momentum of the running statistics. |
| `PReLU(float $initial = 0.25)` | Leaky ReLU with a learned leak per neuron |
| `Swish(float $initial = 1.0)` | `x·σ(βx)` with a learned β per neuron |
| `Noise(float $stddev = 0.1)` | Adds Gaussian noise during training (regularization) |

**Activation functions:**

| Function | Notes |
|---|---|
| `relu` | |
| `leaky_relu(<a>)` | Default `a` = 0.01 |
| `elu(<a>)` | Default `a` = 1 |
| `selu` | |
| `gelu` | |
| `sigmoid` | |
| `tanh` | |
| `softplus` | |
| `softsign` | |
| `silu` | |
| `thresholded_relu(<θ>)` | Default θ = 1 |
| `linear` | |

**Initializers** (the `initializer` parameter of `Dense`):

| Initializer | Notes |
|---|---|
| `he` | |
| `lecun` | |
| `normal(<stddev>)` | |
| `uniform(<beta>)` | |
| `xavier1` | Uniform |
| `xavier2` | Normal |
| `constant(<value>)` | |

### 13.2 Optimizers (`ML\NeuralNet\Optimizers`)

| Optimizer | Update |
|---|---|
| `Stochastic(float $rate = 0.01)` | Plain gradient descent |
| `Momentum(float $rate = 0.001, float $decay = 0.1, bool $lookahead = false)` | Momentum (1 − decay). `lookahead` selects Nesterov momentum. |
| `AdaGrad(float $rate = 0.01)` | Per-parameter rates from accumulated squared gradients |
| `RMSProp(float $rate = 0.001, float $decay = 0.1)` | Moving average of squared gradients |
| `Adam(float $rate = 0.001, float $momentumDecay = 0.1, float $normDecay = 0.001)` | Adam with β1 = 1 − momentumDecay and β2 = 1 − normDecay. This is the default optimizer. |
| `AdaMax(float $rate = 0.001, float $momentumDecay = 0.1, float $normDecay = 0.001)` | Adam with the infinity norm |

### 13.3 Learning-rate schedules (`ML\NeuralNet\Schedules`)

A schedule multiplies the optimizer's rate at each epoch:

| Schedule | Multiplier |
|---|---|
| `Constant` | 1 |
| `StepDecay(int $steps = 10, float $factor = 0.5)` | `factor^⌊epoch/steps⌋` |
| `Cosine(int $period = 50, float $minimum = 0.0)` | Cosine annealing from 1 down to `minimum`, restarting every `period` epochs |
| `Cyclical(int $steps = 10, float $minimum = 0.1)` | Triangular cycle between `minimum` and 1 |
| `Ramp(int $warmup = 10)` | Linear warm-up from 1/warmup to 1 |

```php
<?php
use ML\Classification\MLPClassifier;
use ML\NeuralNet\Layers\{Dense, BatchNorm, PReLU, Dropout};
use ML\NeuralNet\Optimizers\RMSProp;
use ML\NeuralNet\Schedules\StepDecay;
use ML\Preprocessing\StandardScaler;
use ML\Datasets;
use ML\Metrics;

$data = Datasets::agglomerate([
    'a' => Datasets::halfMoon(0, 0, 1.0, 0, 0.1, 150, 1),
    'b' => Datasets::halfMoon(1, 0.5, 1.0, 180, 0.1, 150, 2),
], true, 1);
$X = (new StandardScaler())->fitTransform($data['samples']);

$net = new MLPClassifier(
    [new Dense(32, initializer: 'he'), new BatchNorm(), new PReLU(), new Dropout(0.1), new Dense(16), new PReLU()],
    batchSize: 32,
    optimizer: new RMSProp(0.01),
    epochs: 200,
    schedule: new StepDecay(50, 0.5),
    seed: 1,
);
$net->fit($X, $data['labels']);
printf("training accuracy %.2f after %d epochs\n", Metrics::accuracy($data['labels'], $net->predict($X)), count($net->getLosses()));
```

---

## 14. Evaluation: metrics, reports and validators

### 14.1 Metrics

`ML\Metrics` provides static functions. Arrays are compared by position (keys are ignored),
and both arrays must be non-empty and the same length.

| Method | Meaning |
|---|---|
| `accuracy($expected, $predicted)` | Share of positions where `expected === predicted` (strict) |
| `precision($expected, $predicted, $positiveClass = null)` | Macro-averaged precision, or the precision of `positiveClass` |
| `recall($expected, $predicted, $positiveClass = null)` | Macro-averaged recall, or the recall of `positiveClass` |
| `fBeta($expected, $predicted, $beta = 1.0, $positiveClass = null)` | F-beta score (macro, or for one class) |
| `f1Score($expected, $predicted, $positiveClass = null)` | F1 score |
| `informedness($expected, $predicted, $positiveClass = null)` | Recall + specificity − 1 (Youden's J) |
| `mcc($expected, $predicted)` | Multiclass Matthews correlation coefficient (−1 … 1) |
| `rocAuc($expected, $scores, $positiveClass)` | Area under the ROC curve, from scores for `positiveClass` |
| `averagePrecision($expected, $scores, $positiveClass)` | Area under the precision-recall curve |
| `brierScore($expected, $probabilities)` | Mean squared error of probability rows (`label => p`) |
| `probabilisticAccuracy($expected, $probabilities)` | Mean probability given to the true label |
| `topKAccuracy($expected, $probabilities, $k = 3)` | True label among the k most probable |
| `meanSquaredError`, `rootMeanSquaredError`, `meanAbsoluteError`, `medianAbsoluteError` | Regression errors |
| `r2Score($expected, $predicted)` | Coefficient of determination |
| `smape($expected, $predicted)` | Symmetric mean absolute percentage error (0–100) |
| `homogeneity($labels, $clusters)`, `completeness(...)`, `vMeasure(..., $beta = 1.0)` | Entropy-based clustering quality (0–1) |
| `randIndex($labels, $clusters)` | Adjusted Rand index (1 = identical clusterings) |
| `purity($labels, $clusters)` | Share of samples in their cluster's majority class |
| `trustworthiness($original, $embedded, $k = 5)` | How well an embedding preserves neighbourhoods (0–1) |

```php
<?php
use ML\Metrics;

$expected = ['a', 'b', 'a', 'c', 'b', 'a'];
$predicted = ['a', 'b', 'b', 'c', 'a', 'a'];
printf("accuracy %.3f, macro F1 %.3f, MCC %.3f\n", Metrics::accuracy($expected, $predicted), Metrics::f1Score($expected, $predicted), Metrics::mcc($expected, $predicted));
printf("precision of 'a' %.3f\n", Metrics::precision($expected, $predicted, 'a'));   // 0.667

printf("ROC AUC %.3f\n", Metrics::rocAuc([1, 0, 1, 1, 0], [0.9, 0.1, 0.8, 0.4, 0.5], 1));   // 0.833
printf("RMSE %.3f, R² %.3f\n", Metrics::rootMeanSquaredError([3, 5, 2], [2.5, 5, 3]), Metrics::r2Score([3, 5, 2], [2.5, 5, 3]));
printf("V-measure %.3f\n", Metrics::vMeasure(['x', 'x', 'y', 'y'], [0, 0, 1, 1]));     // 1.000
```

### 14.2 Reports

The static functions of `ML\CrossValidation\Reports` return detailed arrays:

| Method | Returns |
|---|---|
| `confusionMatrix($expected, $predicted)` | `actual => [predicted => count]` for every class |
| `multiclassBreakdown($expected, $predicted)` | `['overall' => [...], 'classes' => [label => [...]]]`. Overall: accuracy, macro precision/recall/F1, MCC, informedness, cardinality, classes. Per class: precision, recall, specificity, F1, informedness, TP/FP/FN/TN, support, proportion. |
| `errorAnalysis($expected, $predicted)` | MAE, median AE, MSE, RMSE, R², SMAPE, error mean, variance, std dev, min and max, cardinality |
| `contingencyTable($labels, $clusters)` | `cluster => [label => count]` |

```php
<?php
use ML\CrossValidation\Reports;

$expected = ['cat', 'dog', 'cat', 'dog', 'cat'];
$predicted = ['cat', 'dog', 'dog', 'dog', 'cat'];
echo json_encode(Reports::confusionMatrix($expected, $predicted)), "\n";   // {"cat":{"cat":2,"dog":1},"dog":{"cat":0,"dog":2}}
$report = Reports::multiclassBreakdown($expected, $predicted);
printf("recall(cat) %.3f, support %d\n", $report['classes']['cat']['recall'], $report['classes']['cat']['support']);
printf("RMSE %.3f\n", Reports::errorAnalysis([1, 2, 3], [1.5, 2, 2])['rootMeanSquaredError']);
```

### 14.3 Validators

Validators (`ML\CrossValidation\*`, interface `ML\CrossValidation\Validator`) estimate how well
a model generalizes:

1. They clone the estimator for each split, so your instance stays untouched.
2. They train the clone on one part of the data.
3. They score it on the other part.

| Validator | Splits |
|---|---|
| `HoldOut(float $ratio = 0.2, bool $stratify = false, ?int $seed = null)` | One train/test split |
| `KFold(int $k = 5, bool $stratify = false, ?int $seed = null)` | k folds; every sample is tested once |
| `LeavePOut(int $p = 10)` | Consecutive folds of p samples (no shuffling) |
| `MonteCarlo(int $simulations = 10, float $ratio = 0.2, bool $stratify = false, ?int $seed = null)` | Repeated random hold-out splits |

Each validator has two methods:

- `test(object $estimator, array $samples, array $targets, string|callable|null $metric = null): float`
  returns the mean score.
- `scores(...)` takes the same arguments and returns one score per split.

`stratify` keeps the class proportions in every split.

```php
<?php
use ML\CrossValidation\{KFold, HoldOut, MonteCarlo};
use ML\Classification\{RandomForest, GaussianNB};
use ML\Datasets;

$data = Datasets::agglomerate(['a' => Datasets::blob([0, 0], 1.0, 60, 1), 'b' => Datasets::blob([2, 2], 1.0, 60, 2)], true, 1);
$X = $data['samples'];
$y = $data['labels'];

printf("5-fold accuracy %.3f\n", (new KFold(5, stratify: true, seed: 1))->test(new GaussianNB(), $X, $y));
printf("hold-out F1 %.3f\n", (new HoldOut(0.3, true, 1))->test(new RandomForest(30, seed: 1), $X, $y, 'f1'));
print_r(array_map(fn ($s) => round($s, 2), (new MonteCarlo(3, seed: 1))->scores(new GaussianNB(), $X, $y)));
```

### 14.4 Metric names for validators and searches

| Kind | Names | Direction |
|---|---|---|
| Classification | `accuracy`, `precision`, `recall`, `f1`, `mcc`, `informedness` (macro averages) | Higher is better |
| Regression | `r2` | Higher is better |
| Regression | `mse`, `mae`, `rmse`, `median_absolute_error`, `smape` | Lower is better |
| Custom | Any callable `fn (array $expected, array $predicted): float` | Higher is better |

Without a metric, classifiers default to `accuracy` and regressors to `r2`. Pipelines and
ensembles are looked through to find the kind of their estimator.

---

## 15. Datasets and distances

### 15.1 Synthetic data and splitting

`ML\Datasets` has static generators, which are useful for tests, demos and benchmarks. With
a `seed`, the output is reproducible.

| Method | Returns |
|---|---|
| `blob(array $center, float\|array $stddev = 1.0, int $n = 100, ?int $seed = null)` | n Gaussian samples around `center`. `stddev` can also be a list with one value per feature. |
| `circle(float $x = 0, float $y = 0, float $scale = 1, float $noise = 0.1, int $n = 100, ?int $seed = null)` | Points on a circle |
| `halfMoon(float $x = 0, float $y = 0, float $scale = 1, float $rotation = 90, float $noise = 0.1, int $n = 100, ?int $seed = null)` | Points on a half circle |
| `swissRoll(..., int $n = 100, ?int $seed = null)` | `['samples' => 3-D points, 'targets' => position along the roll]` |
| `hyperplane(array $coefficients, float $intercept = 0, float $noise = 0.1, int $n = 100, ?int $seed = null)` | `['samples', 'targets']` with `y = intercept + w·x + noise` |
| `agglomerate(array $groups, bool $shuffle = true, ?int $seed = null)` | Combines `label => samples` groups into `['samples', 'labels']` |
| `trainTestSplit(array $samples, array $targets, float $testRatio = 0.25, bool $stratify = false, ?int $seed = null)` | `[$trainSamples, $testSamples, $trainTargets, $testTargets]` |

```php
<?php
use ML\Datasets;

$data = Datasets::agglomerate([
    'apple' => Datasets::blob([1, 1], 0.2, 30, 1),
    'pear' => Datasets::blob([2, 1.5], 0.2, 30, 2),
], true, 3);
[$Xtrain, $Xtest, $ytrain, $ytest] = Datasets::trainTestSplit($data['samples'], $data['labels'], 0.2, true, 3);
echo count($Xtrain), ' train / ', count($Xtest), " test\n";   // 48 train / 12 test

$regression = Datasets::hyperplane([2, -1], intercept: 5.0, noise: 0.0, n: 3, seed: 1);
printf("%.3f\n", $regression['targets'][0] - (5 + 2 * $regression['samples'][0][0] - $regression['samples'][0][1]));   // 0.000
```

### 15.2 Distances

`ML\Distance` computes the [distance metrics](#46-distance-metrics) directly:

| Method | Returns |
|---|---|
| `compute(string $metric, array $a, array $b): float` | The distance between two samples |
| `pairwise(string $metric, array $samples): array` | The full distance matrix (computed in parallel) |
| `gower(array $a, array $b, array $ranges): float` | Gower distance for mixed data (see below) |

The Gower distance handles each feature by its type:

- **numbers:** `|a − b| / range`
- **other values:** 0 if equal, 1 otherwise
- **missing values:** skipped

```php
<?php
use ML\Distance;

printf("%.2f %.2f %.2f\n", Distance::compute('euclidean', [0, 0], [3, 4]), Distance::compute('manhattan', [0, 0], [3, 4]), Distance::compute('cosine', [1, 0], [0, 1]));   // 5.00 7.00 1.00
echo json_encode(Distance::pairwise('chebyshev', [[0, 0], [1, 3], [4, 1]])), "\n";
printf("%.2f\n", Distance::gower([30, 'red', null], [40, 'blue', 5], [50, 0, 1]));   // (0.2 + 1) / 2 = 0.60
```

---

## 16. Interfaces and exceptions

### 16.1 Interfaces

The interfaces are markers without methods. Meta estimators, validators and type hints in your
own code use them to tell what an object can do.

| Interface | Meaning |
|---|---|
| `ML\Estimator` | Has `predict()` |
| `ML\Learner` | Has `fit()` |
| `ML\Classifier`, `ML\Regressor`, `ML\Clusterer`, `ML\AnomalyDetector` | The kind of estimator (each extends `ML\Estimator`) |
| `ML\Probabilistic` | Has `predictProbability()` |
| `ML\RanksFeatures` | Has `getFeatureImportances()` |
| `ML\Transformer` | Has `transform()` |
| `ML\Stateful` | A transformer that must be fitted |
| `ML\Text\Tokenizer` | Has `tokenize(string): array` |
| `ML\CrossValidation\Validator`, `ML\NeuralNet\Layer`, `ML\NeuralNet\Optimizer`, `ML\NeuralNet\Schedule` | Configuration objects |

```php
<?php
use ML\Classification\RandomForest;

$model = new RandomForest();
var_dump($model instanceof ML\Classifier, $model instanceof ML\Probabilistic, $model instanceof ML\RanksFeatures);   // true, true, true
```

### 16.2 Exceptions

```
ML\Exception\MLException  (interface)
├── ML\Exception\InvalidArgumentException  extends \InvalidArgumentException
├── ML\Exception\NotFittedException        extends \LogicException
└── ML\Exception\RuntimeException          extends \RuntimeException
```

| Exception | When | Example messages |
|---|---|---|
| `InvalidArgumentException` | Bad hyperparameter, bad data, or invalid serialized data | `Sample #1 has 1 features, expected 2` · `Label #0 must be int or string, float given` · `k (10) cannot exceed the number of training samples (6)` · `Unknown distance metric "bogus"; …` · `Invalid serialized data: truncated` |
| `NotFittedException` | The model is used before `fit()` | `ML\Classification\RandomForest is not fitted yet; call fit() first` |
| `RuntimeException` | Numerical failure, or the system ran out of memory in native code | `Cannot fit LinearRegression: the feature matrix is singular …` · `Training diverged; lower the learning rate or scale the features` |

Because the SPL exceptions are the parents, existing `catch (\InvalidArgumentException $e)`
blocks keep working. Exceptions thrown by your own PHP code inside a meta estimator propagate
unchanged; this includes custom estimators, tokenizers, `LambdaFunction` callbacks and metric
callables.

---

## 17. Saving and loading models

Every model supports PHP's `serialize()` / `unserialize()`. Train once (in a CLI script or a
cron job), save the model, then load it on each web request:

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

**Format.** Native models serialize to `['mlnative' => 1, 'state' => <binary payload>]`. The
binary format is compact (raw little-endian doubles), so a 100-tree forest stays small and
loads quickly. These rules apply:

- **Header:** every payload starts with a magic number, a class tag and a format version.
- **Class check:** a payload of one class can't be loaded into another.
- **Validation:** every read is bounds-checked. Model invariants are verified too, such as tree
  node indices, matrix shapes, label indices and positive variances.
- **Rejection:** truncated, corrupted or forged data throws
  `ML\Exception\InvalidArgumentException` instead of crashing. The test suite fuzzes this.

**Compatibility.** The six 0.1.0 classes keep their 0.1.0 array format, so models saved with
0.1.0 or 0.1.1 still load. Those classes are `LinearRegression`, `LogisticRegression`,
`KNearestNeighbors`, `GaussianNB`, `KMeans` and `StandardScaler`. New keys that 0.1.2 added are
optional:

- `KNearestNeighbors`: `weighted`
- `KMeans`: `seeder`, `presetCentroids`

**Meta estimators** (`Pipeline`, `OneVsRest`, `BootstrapAggregator`, `CommitteeMachine`,
`GridSearch`, `BayesianSearch`) serialize their wrapped objects as normal PHP properties.

**Security.** Use `allowed_classes` whenever the serialized data could come from somewhere you
don't fully control. For meta estimators, include the wrapped classes in the list.

---

## 18. Recipes

### 18.1 Train, evaluate and report

```php
<?php
use ML\Classification\RandomForest;
use ML\CrossValidation\Reports;
use ML\Datasets;
use ML\Metrics;

$data = Datasets::agglomerate([
    'low' => Datasets::blob([0, 0], 1.0, 80, 1),
    'mid' => Datasets::blob([3, 0], 1.0, 80, 2),
    'high' => Datasets::blob([1.5, 3], 1.0, 80, 3),
], true, 1);
[$Xtrain, $Xtest, $ytrain, $ytest] = Datasets::trainTestSplit($data['samples'], $data['labels'], 0.25, true, 1);

$model = (new RandomForest(100, seed: 1))->fit($Xtrain, $ytrain);
$predicted = $model->predict($Xtest);
printf("accuracy %.3f\n", Metrics::accuracy($ytest, $predicted));
foreach (Reports::multiclassBreakdown($ytest, $predicted)['classes'] as $label => $stats) {
    printf("%-5s precision %.2f recall %.2f\n", $label, $stats['precision'], $stats['recall']);
}
```

### 18.2 Compare several models with cross-validation

```php
<?php
use ML\CrossValidation\KFold;
use ML\Classification\{LogisticRegression, KNearestNeighbors, GaussianNB, RandomForest, SVC};
use ML\Pipeline;
use ML\Preprocessing\StandardScaler;
use ML\Datasets;

$data = Datasets::agglomerate(['a' => Datasets::halfMoon(0, 0, 1, 0, 0.15, 100, 1), 'b' => Datasets::halfMoon(1, 0.5, 1, 180, 0.15, 100, 2)], true, 1);
$candidates = [
    'logistic' => new Pipeline([new StandardScaler()], new LogisticRegression()),
    'knn' => new KNearestNeighbors(7),
    'naive bayes' => new GaussianNB(),
    'forest' => new RandomForest(50, seed: 1),
    'svm' => new Pipeline([new StandardScaler()], new SVC()),
];
$cv = new KFold(5, stratify: true, seed: 1);
foreach ($candidates as $name => $model) {
    printf("%-12s %.3f\n", $name, $cv->test($model, $data['samples'], $data['labels'], 'accuracy'));
}
```

### 18.3 Text classification with persistence

```php
<?php
use ML\Pipeline;
use ML\Text\{TextNormalizer, WordCountVectorizer, TfIdfTransformer, WordStemmer};
use ML\Classification\MultinomialNB;

$reviews = ['Great product, works perfectly', 'Terrible, broke after one day', 'Love it, excellent quality',
    'Awful quality, waste of money', 'Works great, very happy', 'Bad product, very disappointed'];
$sentiment = ['pos', 'neg', 'pos', 'neg', 'pos', 'neg'];

$model = (new Pipeline(
    [new TextNormalizer(), new WordCountVectorizer(tokenizer: new WordStemmer()), new TfIdfTransformer()],
    new MultinomialNB(),
))->fit($reviews, $sentiment);

$saved = serialize($model);   // store in a file, Redis, APCu, ...
print_r(unserialize($saved)->predict(['excellent, works great', 'broke, waste of money']));   // [pos, neg]
```

### 18.4 Fraud detection with an anomaly detector

```php
<?php
use ML\AnomalyDetection\IsolationForest;
use ML\Pipeline;
use ML\Preprocessing\RobustStandardizer;
use ML\Datasets;

// [amount, hour of day] of normal card payments
$payments = array_map(fn ($p) => [abs($p[0]) * 40 + 10, fmod(abs($p[1]) * 4 + 12, 24)], Datasets::blob([0, 0], 1.0, 500, 1));
$detector = (new Pipeline([new RobustStandardizer()], new IsolationForest(200, contamination: 0.01, seed: 1)))->fit($payments);

print_r($detector->predict([[35, 13], [2400, 3]]));   // [0, 1]
print_r(array_map(fn ($s) => round($s, 2), $detector->score([[35, 13], [2400, 3]])));
```

### 18.5 Tune hyper-parameters, then save the best model

```php
<?php
use ML\ModelSelection\GridSearch;
use ML\CrossValidation\KFold;
use ML\Classification\RandomForest;
use ML\Datasets;

$data = Datasets::agglomerate(['a' => Datasets::blob([0, 0], 1.2, 60, 1), 'b' => Datasets::blob([2, 2], 1.2, 60, 2)], true, 1);
$search = (new GridSearch(RandomForest::class, ['estimators' => [20, 60], 'maxHeight' => [2, 6], 'seed' => 1], new KFold(3, true, 1)))
    ->fit($data['samples'], $data['labels']);
echo json_encode($search->getBest()), "\n";
$best = $search->getEstimator();   // already refit on all data
echo get_class(unserialize(serialize($best))), "\n";   // ML\Classification\RandomForest
```

### 18.6 Loading data from CSV

CSV values arrive as strings. Cast them while reading, or let `NumericStringConverter` do it:

```php
<?php
use ML\Preprocessing\NumericStringConverter;
use ML\Regression\LinearRegression;

$csv = "area,rooms,price\n50,1,150\n80,2,235\n100,3,295\n150,4,430\n";
$rows = array_map(fn ($line) => str_getcsv($line, escape: ''), array_slice(explode("\n", trim($csv)), 1));
$rows = (new NumericStringConverter())->transform($rows);

$X = array_map(fn ($r) => [$r[0], $r[1]], $rows);
$y = array_column($rows, 2);
printf("%.1f\n", (new LinearRegression(l2: 0.01))->fit($X, $y)->predict([120, 3]));
```

---

## 19. Benchmarks

[benchmarks/benchmark.php](../benchmarks/benchmark.php) runs two comparisons on synthetic data
with a fixed seed:

1. The extension against straightforward pure-PHP implementations of the same algorithms.
2. The effect of `mlnative.threads`.

> **Other languages and libraries:** the [cross-language benchmark report](BENCHMARK_REPORT.md)
> compares mlnative with the same six algorithms written in PHP, Python, JavaScript (Node.js),
> Ruby and Lua, and with NumPy, scikit-learn and PyTorch.

```bash
docker build -t mlnative .
docker run --rm mlnative php benchmarks/benchmark.php
docker run --rm -e BENCH_THREADS=8 mlnative php benchmarks/benchmark.php
```

Results on PHP 8.3.36 (Docker on Apple Silicon, aarch64, 4 CPUs, default `-O2` build,
opcache/JIT off):

| Benchmark | Pure PHP | mlnative | Speedup |
|---|---:|---:|---:|
| KNN: fit 3000×16 + predict 300 | 280.8 ms | 6.6 ms | 43× |
| LogisticRegression: 200 epochs on 3000×16 | 228.6 ms | 16.9 ms | 14× |
| LinearRegression: fit 20000×32 | 383.3 ms | 5.6 ms | 68× |
| KMeans: 5000×8, 4 clusters, 20 iterations | 63.3 ms | 4.0 ms | 16× |

| Multithreading | 1 thread | 4 threads | Speedup |
|---|---:|---:|---:|
| RandomForest: 200 trees on 3000×16 | 833.2 ms | 222.4 ms | 3.7× |
| KNN: predict 3000 samples | 63.1 ms | 20.1 ms | 3.1× |
| IsolationForest: 300 trees on 3000×16 | 70.9 ms | 20.0 ms | 3.5× |

How to read these numbers:

- **Conversion cost:** every call first copies the PHP arrays into contiguous native memory.
  That cost grows linearly with the data size, so the more computation a call does per
  sample, the bigger the speedup.
- **Logistic regression:** with two classes the native version trains a single sigmoid, the
  same arithmetic as the PHP reference. With more classes it computes a full softmax.
- **Hardware:** absolute times depend on the machine, so run the benchmark on your target
  hardware. Building with `CXXFLAGS="-O3 -march=native"` can speed up the native side further.

---

## 20. Testing

The tests are phpt files in [tests/](../tests/), PHP's standard extension test format.
They share [tests/helpers.inc](../tests/helpers.inc).

| File | Covers |
|---|---|
| `001_loaded.phpt`, `inventory.phpt` | The extension loads. The inventory test lists **every** class and interface with its marker interfaces, so any API change shows up there. |
| `linear_regression.phpt`, `logistic_regression.phpt`, `knn.phpt`, `gaussian_nb.phpt`, `kmeans.phpt`, `standard_scaler.phpt`, `metrics.phpt`, `input_validation.phpt` | The 0.1.0 API and its behaviour |
| `trees.phpt` | Trees, Extra Trees and forests; rules; importances; thread-independent results; clone; serialization |
| `neighbors.phpt` | K-d tree equals brute force for every supported metric; radius models; regressors; 0.1.0 payload compatibility |
| `naive_bayes.phpt` | Categorical and multinomial naive Bayes, checked against hand-computed probabilities |
| `metrics_extended.phpt`, `cross_validation.phpt` | All metrics and reports with exact values, and all validators |
| `preprocessing_extended.phpt`, `datasets_distance.phpt` | Every transformer, imputer, generator and distance |
| `clustering_extended.phpt`, `anomaly.phpt` | All clusterers, seeders and anomaly detectors |
| `boosting.phpt`, `ensemble_pipeline.phpt`, `model_selection.phpt` | Boosting, ensembles (including a user-written PHP estimator), Pipeline, GridSearch, BayesianSearch |
| `text.phpt` | Tokenizers (including Persian and CJK), 70 reference Porter stems, filters, vectorizers, custom tokenizers |
| `decomposition.phpt`, `svm_neuralnet.phpt` | PCA/SVD/LDA/projections/t-SNE; SVM kernels; every layer, activation, initializer, optimizer and schedule |
| `serialization_safety.phpt` | Truncated, garbage and randomly bit-flipped payloads for many classes are rejected cleanly |
| `docs_examples.phpt` | Runs every `<?php` example in this document and in `README.md` |

```bash
# Run all tests (in Docker)
docker build --target test .

# Run against another PHP version
docker build --build-arg PHP_VERSION=8.1 --target test .

# From a phpize build directory
make test TESTS="--show-diff tests"
```

**Memory checking with valgrind.** The suite runs clean under valgrind: 0 errors and 0 leaks. Pass
`-d pcre.jit=0`. Otherwise machine code generated by the PCRE JIT (used by `RegexFilter` and every
`preg_*` call) makes valgrind report false "uninitialised value" errors that have nothing to do with
the extension.

```bash
docker build --target build -t mlnative:build .
docker run --rm -v "$PWD/tests:/t:ro" -v "$PWD/docs:/usr/src/mlnative/docs:ro" \
  -v "$PWD/README.md:/usr/src/mlnative/README.md:ro" mlnative:build sh -c \
  'apt-get update -qq && apt-get install -y -qq valgrind >/dev/null && cp -r /t tests && \
   NO_INTERACTION=1 php run-tests.php -m -d pcre.jit=0 -d extension=$PWD/modules/mlnative.so tests'
```

### 20.1 Continuous integration and releases

[.github/workflows/ci.yml](../.github/workflows/ci.yml) runs on every pull request, on every
push to `main` (that includes merges) and on demand from the Actions tab
(`workflow_dispatch`).

| Job | What it does |
|---|---|
| **Read version** | Reads `PHP_MLNATIVE_VERSION` from `php_mlnative.h`. It **fails** if the [changelog](#24-changelog) has no `### <version>` entry. On `main`, it also decides whether to release: only when the tag `v<version>` doesn't exist yet. |
| **Build** (8 jobs) | PHP 8.1, 8.2, 8.3 and 8.4 × x86_64 (`ubuntu-24.04`) and aarch64 (`ubuntu-24.04-arm`). Each job runs the full test suite (`--target test`), exports `mlnative.so` (`--target artifact`), packages it as `mlnative-<version>-php<8.x>-nts-linux-<arch>.tar.gz`, and uploads it as a workflow artifact. Pull requests get these artifacts too, for manual testing. |
| **Publish release** | Only on `main`, and only when the version is new. It creates the tag `v<version>` on the merged commit and publishes a GitHub release. The release has the 8 archives, a `SHA256SUMS` file, and the changelog entry plus install instructions as release notes. |

**A merge into `main` does not always create a release.** A release happens only when
`PHP_MLNATIVE_VERSION` has changed to a version that hasn't been released yet. Merges without
a version bump are still tested and built, but they don't publish anything.

**How to publish a release**

1. Bump `PHP_MLNATIVE_VERSION` in `php_mlnative.h`, following
   [semantic versioning](https://semver.org).
2. Add a `### <version>` entry at the top of the [changelog](#24-changelog). It becomes the
   release notes.
3. Open a pull request. CI checks the changelog entry and all 8 builds.
4. Merge into `main`. The release is published automatically when CI finishes.

To check the release notes locally before merging:

```bash
docker run --rm -v "$PWD:/repo:ro" -w /repo debian:bookworm-slim sh scripts/release-notes.sh 0.1.2
```

**Repository settings needed:** GitHub must be allowed to run Actions on the repository. The
release job uses the built-in `GITHUB_TOKEN` with `contents: write`, so it needs no secret.
If your organization restricts it, enable *Settings → Actions → General → Workflow
permissions → Read and write permissions*. The aarch64 jobs use GitHub's hosted
`ubuntu-24.04-arm` runners. These are free for public repositories; private repositories
need a plan that includes arm64 runners.

---

## 21. Architecture and extending the extension

### 21.1 Source layout

```
config.m4                   Build definition (phpize/autoconf): list of sources
php_mlnative.h              Module header, version constant (single source of the release version)
Dockerfile                  build / test / artifact / runtime stages
.github/workflows/ci.yml    CI matrix and automatic releases (see 20.1)
scripts/release-notes.sh    Extracts a version's changelog entry (used by CI)
src/
  mlnative.cpp              Module entry: ini settings, MINIT registers all classes
  php_bridge.hpp/.cpp       Zend glue: zval <-> Matrix/Table, labels, interfaces, object wrapper,
                            calling back into PHP, the C++ -> PHP exception boundary
  binding.hpp               Generic method handlers (fit/predict/transform/serialize templates)
  arginfo.hpp               Shared argument/return type info
  metric_eval.hpp           Metric names/callables shared by validators and searches
  classes/*.cpp             PHP classes, grouped by area (trees, neighbors, boosting, text, ...)
  core/*.hpp/.cpp           Pure C++17 algorithms, no PHP dependency:
    common.hpp              Matrix, RNG, parallel_for, helpers
    archive.hpp             Bounds-checked binary (de)serialization
    distance, linalg, tree, neighbors (K-d tree), naive_bayes, metrics, transformers,
    clustering, boosting, anomaly, decomposition, text, svm, neuralnet, ...
stubs/mlnative.stub.php     IDE stub / public API description
tests/*.phpt                Test suite
benchmarks/benchmark.php    Native vs pure-PHP and threading benchmarks
examples/quickstart.php     Tour of the main features
docs/DOCUMENTATION.md       This document
```

### 21.2 Design rules

1. **`src/core` has no PHP dependency.** Algorithms work on `mlcore::Matrix` (dense,
   row-major `double`) and `std::vector`, so they can be unit-tested or reused outside PHP.
2. **No C++ exception may reach the Zend engine.** Every method body that calls native code
   runs inside `mlphp::guard()`, which converts exceptions as follows:
   - `std::invalid_argument` → `InvalidArgumentException`
   - `mlcore::NotFittedError` → `NotFittedException`
   - `std::bad_alloc` and other `std::exception` → `RuntimeException`
   - `PhpError` means a PHP exception is already pending, and it's left unchanged.
3. **Convert first, compute, then write results.** PHP arrays are converted into native
   memory up front. Results are written to `return_value` only after the computation has
   succeeded.
4. **`fit()` is transactional.** The generic handlers train a copy and commit it only on
   success.
5. **Threads never touch the Zend engine.** `parallel_for` runs pure C++ work only, and it
   rethrows the first exception in the calling thread.
6. **One object = one native model.** `register_class<T>()` installs the `create_object`,
   `free_obj` and `clone_obj` handlers. `T` must be default-constructible and
   copy-assignable. Meta estimators instead keep PHP objects in declared properties and
   implement `__clone` as a deep copy.
7. **Untrusted data can't hurt.** Every model's `load()` validates its invariants, and every
   archive read is bounds-checked.

### 21.3 Adding a new model (checklist)

1. **Core algorithm:** add `src/core/<area>.hpp/.cpp` with `fit()`, prediction, and
   `save()/load()` (validate every invariant in `load()`).
2. **PHP class:** add a wrapper struct with a unique `kTag` in `src/classes/<area>.cpp`, then
   reuse the templates in `binding.hpp`:
   - `fit_classifier`, `predict_classifier`, `predict_proba`
   - `fit_regressor`, `predict_regressor`
   - `fit_unsupervised`, `transform`, `serialize_native`, …

   Write the constructor with parameter validation, and register the class with the right
   marker interfaces.
3. **Registration:** declare the register function in `src/php_bridge.hpp`, call it from
   `PHP_MINIT_FUNCTION` in `src/mlnative.cpp`, and add the `.cpp` files to `config.m4`.
4. **Stub:** add the class to `stubs/mlnative.stub.php`.
5. **Tests:** add a phpt test for behaviour, errors and a serialize round trip. Also update
   `tests/inventory.phpt`.
6. **Benchmark:** add a pure-PHP reference and a row to `benchmarks/benchmark.php` where it
   makes sense.
7. **Docs:** update this document:
   - the [overview](#1-overview) table
   - a section with a runnable example
   - the [test table](#20-testing)
   - the [changelog](#24-changelog)

   Also update the summary in `README.md`.
8. **Verify:** run `docker build --target test .` for every supported PHP version, and run
   valgrind.
9. **Release (optional):** bump `PHP_MLNATIVE_VERSION` and add the changelog entry. CI
   publishes the release when the change is merged (see [20.1](#201-continuous-integration-and-releases)).

---

## 22. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `Class "ML\..." not found` | The extension isn't loaded in this SAPI. Check `php -m` and `php --ini`. FPM and CLI use different ini files, and FPM needs a restart after enabling it. |
| `PHP Startup: Unable to load dynamic library 'mlnative'` … `API=...` mismatch | The `.so` was built for a different PHP version or variant (ZTS/NTS). Rebuild it with the `phpize`/`php-config` of the PHP that loads it. |
| Prebuilt `mlnative.so` fails with `GLIBC_2.36 not found` or `GLIBCXX_… not found` | Your system is older than Debian 12 / Ubuntu 24.04. Build from source (3.2 / 3.3). |
| `phpize: command not found` | Install the PHP development package (`php8.x-dev`), or use the Docker build. |
| `Cannot fit LinearRegression: the feature matrix is singular …` | Constant or duplicated features, or fewer samples than features. Use `new LinearRegression(l2: 0.01)` or drop those features. |
| `… diverged`, or poor accuracy (LogisticRegression, MLP, SVM) | Scale the features with `StandardScaler`, or lower the learning rate. |
| MLP stops after a few epochs with poor accuracy | Early stopping triggered while the loss was on a plateau (e.g. saturated sigmoid units). Raise `window`, use a higher learning rate, or switch to `relu`/`tanh`. |
| `Sample #0, feature #1 must be int or float, string given` | Data from CSV, a form or a database arrives as strings. Cast it, or use `NumericStringConverter`. |
| `Label #0 must be int or string, float given` | Labels must be discrete. Cast them with `(int)` or use strings. |
| Results differ between runs | Pass a `seed` to the model (and to `Datasets` generators / splits). |
| Training is slow on a multi-core server | Set `mlnative.threads` (e.g. `0` for all cores) in php.ini or with `ini_set()`. |
| Memory grows beyond `memory_limit` | Native buffers are not counted by `memory_limit` (see [23](#23-limitations-and-roadmap)). Train on smaller batches or in a separate CLI process. |
| A merge into `main` didn't create a release | Releases happen only when `PHP_MLNATIVE_VERSION` changes to a version without an existing `v<version>` tag (see 20.1). |
| CI fails with `No changelog entry "### x.y.z"` | Add that version's entry to the [changelog](#24-changelog). |

---

## 23. Limitations and roadmap

**Current limitations**

- **Dense data only:** there is no sparse matrix support, so very high-dimensional sparse
  data (e.g. bag of words with a huge vocabulary) uses a lot of memory.
  `TokenHashingVectorizer` with a moderate `dimensions` value keeps the width bounded.
- **Memory limit:** native memory comes from the C++ allocator, so PHP's `memory_limit`
  does not count it.
- **No incremental training:** there is no `partialFit()`, so every `fit()` retrains from
  scratch.
- **Quadratic algorithms:** t-SNE, Ward, OPTICS, LOF, DBSCAN with non-Minkowski metrics, and
  SVM training are O(n²) in time or kernel evaluations. They suit data sets of up to roughly
  tens of thousands of samples.
- **No GPU:** neural networks run on the CPU only, so they're meant for small and medium
  networks.
- **Untested platforms:** Windows builds are not supported yet, and ZTS builds are untested.

**Possible next steps:** sparse inputs, `partialFit()` for online learning, gradient boosting
for classification, probability calibration for SVC, Barnes–Hut t-SNE, and recurrent or
convolutional layers.

---

## 24. Changelog

### 0.1.2

All five phases inspired by the Rubix ML feature set.

**New infrastructure**

- **Interfaces:** marker interfaces `ML\Estimator`, `Learner`, `Classifier`, `Regressor`,
  `Clusterer`, `AnomalyDetector`, `Probabilistic`, `Transformer`, `Stateful` and
  `RanksFeatures`. The 0.1.0 classes now implement them too.
- **Multithreading:** native multithreading through the `mlnative.threads` ini setting.
  Results don't depend on the thread count.
- **Distance metrics:** 11 metrics, usable by all neighbor, clustering and anomaly models and
  by `ML\Distance`.
- **Serialization:** a compact, bounds-checked binary format for every new model, with
  invariant validation and fuzz tests.

**Classification**

- New models: `ClassificationTree`, `ExtraTreeClassifier`, `RandomForest` (bootstrap,
  balanced, extra trees, feature importances), `KDNeighbors` (K-d tree), `RadiusNeighbors`,
  `NaiveBayes` (categorical), `MultinomialNB`, `AdaBoost` (SAMME), `LogitBoost`, `SVC`
  (SMO, linear/polynomial/RBF/sigmoid kernels, one-vs-one) and `MLPClassifier`.
- `KNearestNeighbors` gains every distance metric, `weighted` voting and
  `predictProbability()`. The old serialized format still loads.

**Regression**

`ElasticNet`, `RegressionTree`, `ExtraTreeRegressor`, `RandomForestRegressor`, `GradientBoost`
(with early stopping), `KNNRegressor`, `KDNeighborsRegressor`, `RadiusNeighborsRegressor`,
`SVR`, `MLPRegressor` and `Adaline`.

**Clustering**

- `KMeans` gains the `random`, `kmc2` and `preset` seeders, and `predictProbability()`.
- New clusterers: `KMedoids`, `FuzzyCMeans`, `GaussianMixture`, `MeanShift`, `DBSCAN`,
  `OPTICS` and `Ward`.

**Anomaly detection**

`RobustZScore`, `GaussianMLE`, `IsolationForest`, `LocalOutlierFactor`, `Loda` and
`OneClassSVM`.

**Meta estimators**

`OneVsRest`, `BootstrapAggregator`, `CommitteeMachine`, `Pipeline`, `GridSearch` and
`BayesianSearch` (Gaussian process with expected improvement). They work with user-written PHP
estimators too.

**Preprocessing**

- Scaling and transforms: `MinMaxNormalizer`, `MaxAbsoluteScaler`, `RobustStandardizer`,
  `PowerTransformer` (Yeo-Johnson), `L1Normalizer`, `L2Normalizer` and `PolynomialExpander`.
- Mixed data: `OneHotEncoder`, `IntervalDiscretizer`, `NumericStringConverter` and
  `BooleanConverter`.
- Missing values: `MissingDataImputer`, `KNNImputer` and `HotDeckImputer`.
- `LambdaFunction` for your own per-sample function.

**Dimensionality reduction**

`PCA`, `TruncatedSVD`, `LinearDiscriminantAnalysis`, `GaussianRandomProjector`,
`SparseRandomProjector` and `TSNE`.

**Text**

- Tokenizers: Word (Unicode-aware), Whitespace, Sentence, N-gram, K-skip-n-gram and Porter
  `WordStemmer`, plus `PorterStemmer`.
- Normalizers and filters: `TextNormalizer`, `MultibyteTextNormalizer`, `RegexFilter` (with
  ready-made patterns) and `StopWordFilter`.
- Vectorizers and weighting: `WordCountVectorizer`, `TokenHashingVectorizer`,
  `TfIdfTransformer` and `BM25Transformer`. Custom PHP tokenizers are supported.

**Neural networks**

- Layers `Dense`, `Activation` (12 functions), `Dropout`, `BatchNorm`, `PReLU`, `Swish` and
  `Noise`, with 7 initializers.
- Optimizers `Stochastic`, `Momentum` (with Nesterov), `AdaGrad`, `RMSProp`, `Adam` and
  `AdaMax`.
- Learning-rate schedules `Constant`, `StepDecay`, `Cosine`, `Cyclical` and `Ramp`.
- Early stopping on a hold-out set that keeps the best weights.

**Evaluation**

- `ML\Metrics` gains 20 metrics:
  - precision, recall, F-beta, F1, MCC and informedness
  - ROC AUC and average precision
  - Brier score, probabilistic and top-k accuracy
  - RMSE, median absolute error and SMAPE
  - homogeneity, completeness, V-measure, adjusted Rand index and purity
  - trustworthiness
- `ML\CrossValidation\Reports`: confusion matrix, multiclass breakdown, error analysis and
  contingency table.
- Validators `HoldOut`, `KFold`, `LeavePOut` and `MonteCarlo`, with stratification and custom
  metrics.

**Datasets**

`ML\Datasets` generators (blob, circle, half moon, swiss roll, hyperplane, agglomerate) and
`trainTestSplit`.

**Tests**

The suite grew from 10 to 27 test files, including an API inventory and serialization fuzzing.
Every example in this document and in the README is executed.

**Changed**

- `KNearestNeighbors::__construct()` gains a third parameter, `weighted`.
- The error message for an unknown metric now lists all metrics.

### 0.1.0

- Initial release: LinearRegression (with ridge), LogisticRegression (softmax),
  KNearestNeighbors, GaussianNB, KMeans (k-means++), StandardScaler, Metrics.
- Typed exception hierarchy, transactional `fit()`, `clone` support, validated
  serialization.
- Docker build/test pipeline, phpt test suite, benchmark, executable documentation
  (this document and the README examples are run as tests).
- GitHub Actions: tests PHP 8.1–8.4 on x86_64 and aarch64, and publishes a release with
  prebuilt `mlnative.so` archives whenever a new version is merged into `main`.
