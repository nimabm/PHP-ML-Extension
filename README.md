# PHP-ML-Native

[![CI](https://github.com/nimabm/PHP-ML-Extension/actions/workflows/ci.yml/badge.svg)](https://github.com/nimabm/PHP-ML-Extension/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/nimabm/PHP-ML-Extension)](https://github.com/nimabm/PHP-ML-Extension/releases)
![PHP](https://img.shields.io/badge/PHP-8.1%20%7C%208.2%20%7C%208.3%20%7C%208.4-777BB4?logo=php&logoColor=white)
![C++](https://img.shields.io/badge/C%2B%2B-17-00599C?logo=cplusplus&logoColor=white)
![License](https://img.shields.io/badge/license-GPLv3-blue)

**Fast machine learning for PHP, written in C++ as a native Zend extension.**

PHP-ML-Native (`mlnative`) gives PHP a set of machine learning models: regression,
classification, clustering and preprocessing. You train and use them from ordinary PHP code,
but the math runs as compiled C++ inside the PHP engine instead of in interpreted PHP loops.
In the benchmark below that makes it **7× to 60× faster** than the same algorithms written
in pure PHP.

```php
$model = (new ML\Regression\LinearRegression())->fit($samples, $targets);
$model->predict([90, 3]);
```

## Features

| Task | Class |
|---|---|
| Regression | `ML\Regression\LinearRegression`: least squares with optional ridge (L2) |
| Classification | `ML\Classification\LogisticRegression`: binary and multiclass, with probabilities |
| Classification | `ML\Classification\KNearestNeighbors`: euclidean / manhattan |
| Classification | `ML\Classification\GaussianNB`: Gaussian naive Bayes, with probabilities |
| Clustering | `ML\Clustering\KMeans`: k-means++ initialization, reproducible seeds |
| Preprocessing | `ML\Preprocessing\StandardScaler`: zero mean / unit variance |
| Evaluation | `ML\Metrics`: accuracy, MSE, MAE, R² |

All models share the same behaviour:

- **Familiar API:** `fit()` and `predict()`. `predict()` takes one sample or a whole batch.
- **Labels:** classifiers accept `int` or `string` labels.
- **Persistence:** you can save a model with `serialize()` and load it later.
- **Errors:** all errors are typed exceptions (`ML\Exception\*`) with clear messages.
- **Safety:** `fit()` is transactional, `clone` gives an independent copy, and the extension is valgrind-clean.

## Benchmark

This compares mlnative with straightforward pure-PHP implementations of the same algorithms.
It ran on PHP 8.3, in Docker on aarch64, with the default `-O2` build:

| Benchmark | Pure PHP | mlnative | Speedup |
|---|---:|---:|---:|
| KNN: fit 3000×16 + predict 300 | 271.6 ms | 6.2 ms | **44×** |
| LogisticRegression: 200 epochs on 3000×16 | 220.8 ms | 29.3 ms | **7.5×** |
| LinearRegression: fit 20000×32 | 373.0 ms | 6.2 ms | **60×** |

To reproduce it, run:

```bash
docker run --rm mlnative php benchmarks/benchmark.php
```

The methodology and notes are in [the documentation](docs/DOCUMENTATION.md#8-benchmarks).

## Installation

The quickest way to try the extension is Docker. Nothing gets installed on your machine:

```bash
git clone https://github.com/nimabm/PHP-ML-Extension.git && cd PHP-ML-Extension
docker build -t mlnative .                       # compile and build the runtime image
docker run --rm mlnative php examples/quickstart.php
```

**Prebuilt binaries** are attached to every
[release](https://github.com/nimabm/PHP-ML-Extension/releases). Each release has a
`mlnative.so` for PHP 8.1–8.4 on Linux x86_64 and aarch64 (NTS, glibc 2.36+). Download the
archive matching your PHP version and architecture, copy `mlnative.so` into
`php-config --extension-dir`, and add `extension=mlnative` to your php.ini.
[Step-by-step instructions are in the documentation.](docs/DOCUMENTATION.md#36-prebuilt-binaries-from-github-releases)

To build from source with `phpize` instead (requires a C++17 compiler):

```bash
phpize && ./configure --enable-mlnative && make && make test && sudo make install
```

Then add `extension=mlnative` to your php.ini. The documentation covers
[installation in detail](docs/DOCUMENTATION.md#3-installation), including how to add the
extension to your own Docker image.

## Examples

### Regression

```php
<?php
use ML\Regression\LinearRegression;
use ML\Metrics;

// [area m², rooms] → price
$X = [[50, 1], [60, 2], [80, 2], [100, 3], [120, 3], [150, 4]];
$y = [150, 185, 235, 295, 345, 430];

$model = (new LinearRegression())->fit($X, $y);

echo $model->predict([90, 3]), "\n";                          // 270
print_r($model->predict([[70, 2], [130, 4]]));                // batch prediction
printf("R² = %.3f\n", Metrics::r2Score($y, $model->predict($X)));
```

### Classification with probabilities

```php
<?php
use ML\Classification\GaussianNB;
use ML\Classification\KNearestNeighbors;

$X = [[1.0, 1.1], [1.2, 0.9], [0.8, 1.0], [3.0, 3.2], [3.1, 2.9], [2.9, 3.0]];
$y = ['cat', 'cat', 'cat', 'dog', 'dog', 'dog'];

$knn = (new KNearestNeighbors(k: 3))->fit($X, $y);
echo $knn->predict([1.1, 1.0]), "\n";                         // cat

$nb = (new GaussianNB())->fit($X, $y);
print_r($nb->predict([[1.1, 1.0], [3.0, 3.1]]));              // [cat, dog]
print_r($nb->predictProbability([1.1, 1.0]));                 // ['cat' => ~1.0, 'dog' => ~0.0]
```

### Scaling + logistic regression

```php
<?php
use ML\Classification\LogisticRegression;
use ML\Preprocessing\StandardScaler;

// [age, yearly income]: very different scales, so standardize first
$X = [[22, 18000], [25, 21000], [30, 25000], [45, 90000], [50, 120000], [38, 85000]];
$y = ['basic', 'basic', 'basic', 'premium', 'premium', 'premium'];

$scaler = new StandardScaler();
$clf = (new LogisticRegression(learningRate: 0.5))->fit($scaler->fitTransform($X), $y);

print_r($clf->predict($scaler->transform([[28, 23000], [47, 100000]])));   // [basic, premium]
```

### Clustering

```php
<?php
use ML\Clustering\KMeans;

$points = [[1, 1], [1.5, 2], [1, 1.5], [8, 8], [8.5, 9], [9, 8], [1, 9], [1.5, 8.5], [2, 9]];

$kmeans = (new KMeans(clusters: 3, seed: 42))->fit($points);
echo json_encode($kmeans->getLabels()), "\n";                 // cluster index of each point
echo $kmeans->predict([8.2, 8.4]), "\n";                      // cluster of a new point
```

### Saving and loading a trained model

Train once, for example in a CLI script or cron job. Then load the trained model on every
request, without retraining:

```php
<?php
use ML\Classification\LogisticRegression;
use ML\Preprocessing\StandardScaler;

$file = sys_get_temp_dir() . '/churn.model';

// --- train.php: run once, offline ---
$X = [[1, 200], [2, 180], [3, 150], [20, 20], [25, 10], [30, 5]];   // [months inactive, logins]
$y = ['stays', 'stays', 'stays', 'churns', 'churns', 'churns'];

$scaler = new StandardScaler();
$model = (new LogisticRegression())->fit($scaler->fitTransform($X), $y);

// Save the scaler together with the model so production scales data the same way.
file_put_contents($file, serialize(['scaler' => $scaler, 'model' => $model]), LOCK_EX);

// --- on each request ---
['scaler' => $scaler, 'model' => $model] = unserialize(
    file_get_contents($file),
    ['allowed_classes' => [StandardScaler::class, LogisticRegression::class]],
);
echo $model->predict($scaler->transform([22, 15])), "\n";      // churns
unlink($file);
```

Restored models are validated, so corrupted or tampered data throws an exception instead of
loading.

### Error handling

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

## Documentation

📖 **[docs/DOCUMENTATION.md](docs/DOCUMENTATION.md)** is the complete reference. It covers:

- installation
- PHP version compatibility
- every class and parameter, with examples
- model persistence
- recipes (train/test split, choosing `k`, confusion matrix, CSV loading)
- benchmarks
- testing
- the extension's architecture and how to add a new model

For IDE autocompletion, point your IDE at [stubs/mlnative.stub.php](stubs/mlnative.stub.php).

## Project layout

```
src/core/      C++17 algorithms (no PHP dependency)
src/classes/   PHP classes (Zend API bindings)
src/           Module entry and the PHP <-> C++ bridge
tests/         phpt test suite (including every example in the docs)
benchmarks/    Native vs pure-PHP benchmark
examples/      Runnable examples
docs/          Full documentation
```

## Contributing

Run the test suite in Docker:

```bash
docker build --target test .
```

To test another PHP version, add `--build-arg PHP_VERSION=8.4`. Every change to the API or
its behaviour must update the documentation; see [CONTRIBUTING.md](CONTRIBUTING.md).

GitHub Actions tests every pull request on PHP 8.1–8.4 (x86_64 + aarch64). To publish a new
release:

1. Bump `PHP_MLNATIVE_VERSION` in `php_mlnative.h`.
2. Add a changelog entry for that version.
3. Merge into `main`.

The [release process](docs/DOCUMENTATION.md#91-continuous-integration-and-releases) is
described in the documentation.

## License

[GNU GPL v3](LICENSE)
