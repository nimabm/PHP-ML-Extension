<?php

declare(strict_types=1);

// Compares the native extension with straightforward pure-PHP versions of the
// same algorithms on synthetic data. Run it inside the Docker runtime image:
//   docker run --rm mlnative php benchmarks/benchmark.php

use ML\Classification\KNearestNeighbors;
use ML\Classification\LogisticRegression;
use ML\Regression\LinearRegression;

mt_srand(1);

function dataset(int $n, int $d): array
{
    $X = [];
    $y = [];
    for ($i = 0; $i < $n; $i++) {
        $row = [];
        $sum = 0.0;
        for ($j = 0; $j < $d; $j++) {
            $v = mt_rand() / mt_getrandmax() * 2 - 1;
            $row[] = $v;
            $sum += $v * ($j + 1);
        }
        $X[] = $row;
        $y[] = $sum;
    }
    return [$X, $y];
}

function timed(callable $fn): array
{
    $start = hrtime(true);
    $result = $fn();
    return [$result, (hrtime(true) - $start) / 1e6];
}

// ---- pure PHP reference implementations ------------------------------------

function php_knn_predict(array $X, array $labels, array $queries, int $k): array
{
    $out = [];
    foreach ($queries as $q) {
        $dist = [];
        foreach ($X as $i => $row) {
            $s = 0.0;
            foreach ($row as $j => $v) {
                $diff = $v - $q[$j];
                $s += $diff * $diff;
            }
            $dist[$i] = $s;
        }
        asort($dist);
        $votes = [];
        foreach (array_slice($dist, 0, $k, true) as $i => $_) {
            $votes[$labels[$i]] = ($votes[$labels[$i]] ?? 0) + 1;
        }
        arsort($votes);
        $out[] = array_key_first($votes);
    }
    return $out;
}

function php_logistic_fit(array $X, array $y, float $lr, int $iterations): array
{
    $d = count($X[0]);
    $n = count($X);
    $w = array_fill(0, $d, 0.0);
    $b = 0.0;
    for ($it = 0; $it < $iterations; $it++) {
        $gw = array_fill(0, $d, 0.0);
        $gb = 0.0;
        foreach ($X as $i => $row) {
            $z = $b;
            foreach ($row as $j => $v) {
                $z += $w[$j] * $v;
            }
            $err = 1.0 / (1.0 + exp(-$z)) - $y[$i];
            $gb += $err;
            foreach ($row as $j => $v) {
                $gw[$j] += $err * $v;
            }
        }
        for ($j = 0; $j < $d; $j++) {
            $w[$j] -= $lr * $gw[$j] / $n;
        }
        $b -= $lr * $gb / $n;
    }
    return [$w, $b];
}

function php_linear_fit(array $X, array $y): array
{
    // Normal equations with Gaussian elimination, intercept as extra column.
    $d = count($X[0]) + 1;
    $A = array_fill(0, $d, array_fill(0, $d, 0.0));
    $rhs = array_fill(0, $d, 0.0);
    foreach ($X as $i => $row) {
        $row[] = 1.0;
        for ($a = 0; $a < $d; $a++) {
            $rhs[$a] += $row[$a] * $y[$i];
            for ($c = 0; $c < $d; $c++) {
                $A[$a][$c] += $row[$a] * $row[$c];
            }
        }
    }
    for ($p = 0; $p < $d; $p++) {
        for ($r = $p + 1; $r < $d; $r++) {
            $f = $A[$r][$p] / $A[$p][$p];
            for ($c = $p; $c < $d; $c++) {
                $A[$r][$c] -= $f * $A[$p][$c];
            }
            $rhs[$r] -= $f * $rhs[$p];
        }
    }
    $w = array_fill(0, $d, 0.0);
    for ($r = $d - 1; $r >= 0; $r--) {
        $s = $rhs[$r];
        for ($c = $r + 1; $c < $d; $c++) {
            $s -= $A[$r][$c] * $w[$c];
        }
        $w[$r] = $s / $A[$r][$r];
    }
    return $w;
}

// ---- runs --------------------------------------------------------------------

$results = [];

[$X, $y] = dataset(3000, 16);
$labels = array_map(fn ($v) => $v > 0 ? 1 : 0, $y);
[$queries] = dataset(300, 16);

[, $php] = timed(fn () => php_knn_predict($X, $labels, $queries, 5));
[, $native] = timed(fn () => (new KNearestNeighbors(5))->fit($X, $labels)->predict($queries));
$results[] = ['KNN: fit 3000x16 + predict 300', $php, $native];

[, $php] = timed(fn () => php_logistic_fit($X, $labels, 0.5, 200));
[, $native] = timed(fn () => (new LogisticRegression(0.5, 200, 0.0, 0.0))->fit($X, $labels));
$results[] = ['LogisticRegression: 200 epochs on 3000x16', $php, $native];

[$Xl, $yl] = dataset(20000, 32);
[, $php] = timed(fn () => php_linear_fit($Xl, $yl));
[, $native] = timed(fn () => (new LinearRegression())->fit($Xl, $yl));
$results[] = ['LinearRegression: fit 20000x32', $php, $native];

printf("PHP %s, mlnative %s\n\n", PHP_VERSION, phpversion('mlnative'));
printf("%-44s %12s %12s %9s\n", 'benchmark', 'pure PHP', 'mlnative', 'speedup');
foreach ($results as [$name, $php, $native]) {
    printf("%-44s %9.1f ms %9.1f ms %8.1fx\n", $name, $php, $native, $php / $native);
}
