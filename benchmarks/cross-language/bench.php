<?php
// Usage: php bench.php <data dir> <pure|native> <label>

use ML\Classification\KNearestNeighbors;
use ML\Classification\LogisticRegression;
use ML\Regression\LinearRegression;
use ML\Clustering\KMeans;
use ML\Classification\ClassificationTree;
use ML\Classification\MLPClassifier;
use ML\NeuralNet\Layers\Activation;
use ML\NeuralNet\Layers\Dense;
use ML\NeuralNet\Optimizers\Adam;

function load(string $file): array
{
    $lines = file($file, FILE_IGNORE_NEW_LINES | FILE_SKIP_EMPTY_LINES);
    array_shift($lines);
    return array_map(fn ($l) => array_map('floatval', explode(' ', $l)), $lines);
}

function bench(string $label, string $name, callable $fn, callable $check): void
{
    $repeats = (int) (getenv('REPEATS') ?: 5);
    $result = $fn(); // warm-up
    $times = [];
    for ($r = 0; $r < $repeats; $r++) {
        $t = hrtime(true);
        $result = $fn();
        $times[] = (hrtime(true) - $t) / 1e6;
    }
    sort($times);
    printf("%s|%s|%.3f|%.3f|%s\n", $label, $name, $times[intdiv($repeats, 2)], $times[0], $check($result));
}

// ---- pure PHP implementations ------------------------------------------------

function knn_predict(array $X, array $labels, array $queries, int $k): array
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

function logistic_fit(array $X, array $y, float $lr, int $iterations): array
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

function linear_fit(array $X, array $y): array
{
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

function kmeans(array $X, array $centroids, int $iterations): array
{
    $k = count($centroids);
    $d = count($X[0]);
    for ($it = 0; $it < $iterations; $it++) {
        $sums = array_fill(0, $k, array_fill(0, $d, 0.0));
        $counts = array_fill(0, $k, 0);
        foreach ($X as $x) {
            $best = 0;
            $bestDist = INF;
            foreach ($centroids as $c => $center) {
                $dist = 0.0;
                foreach ($x as $j => $v) {
                    $diff = $v - $center[$j];
                    $dist += $diff * $diff;
                }
                if ($dist < $bestDist) {
                    $bestDist = $dist;
                    $best = $c;
                }
            }
            $counts[$best]++;
            foreach ($x as $j => $v) {
                $sums[$best][$j] += $v;
            }
        }
        for ($c = 0; $c < $k; $c++) {
            if ($counts[$c] > 0) {
                for ($j = 0; $j < $d; $j++) {
                    $centroids[$c][$j] = $sums[$c][$j] / $counts[$c];
                }
            }
        }
    }
    return $centroids;
}

// CART with Gini impurity on two classes; nodes are [feature, threshold, left, right, class].
function tree_fit(array $X, array $y, int $maxHeight, int $maxLeaf, float $minPurity): array
{
    $d = count($X[0]);
    $nodes = [null];
    $stack = [[0, range(0, count($X) - 1), 0]];
    while ($stack) {
        [$id, $rows, $depth] = array_pop($stack);
        $n = count($rows);
        $c1 = 0;
        foreach ($rows as $r) {
            $c1 += $y[$r];
        }
        $c0 = $n - $c1;
        $impurity = $n - ($c0 * $c0 + $c1 * $c1) / $n;
        $leaf = [-1, 0.0, 0, 0, $c1 > $c0 ? 1 : 0];
        if ($depth >= $maxHeight || $n <= $maxLeaf || $impurity <= 1e-12) {
            $nodes[$id] = $leaf;
            continue;
        }
        $bestGain = 0.0;
        $bestF = -1;
        $bestT = 0.0;
        for ($f = 0; $f < $d; $f++) {
            $col = [];
            foreach ($rows as $r) {
                $col[$r] = $X[$r][$f];
            }
            asort($col);
            $keys = array_keys($col);
            $v = array_values($col);
            if (!($v[$n - 1] > $v[0])) {
                continue;
            }
            $l0 = $l1 = 0;
            for ($i = 0; $i < $n - 1; $i++) {
                if ($y[$keys[$i]]) {
                    $l1++;
                } else {
                    $l0++;
                }
                if ($v[$i] == $v[$i + 1]) {
                    continue;
                }
                $ln = $i + 1;
                $rn = $n - $ln;
                $r0 = $c0 - $l0;
                $r1 = $c1 - $l1;
                $gain = $impurity - ($ln - ($l0 * $l0 + $l1 * $l1) / $ln) - ($rn - ($r0 * $r0 + $r1 * $r1) / $rn);
                if ($gain > $bestGain) {
                    $bestGain = $gain;
                    $bestF = $f;
                    $bestT = $v[$i] + ($v[$i + 1] - $v[$i]) / 2;
                }
            }
        }
        if ($bestF < 0 || $bestGain / $n < $minPurity) {
            $nodes[$id] = $leaf;
            continue;
        }
        $left = $right = [];
        foreach ($rows as $r) {
            if ($X[$r][$bestF] < $bestT) {
                $left[] = $r;
            } else {
                $right[] = $r;
            }
        }
        if (!$left || !$right) {
            $nodes[$id] = $leaf;
            continue;
        }
        $l = count($nodes);
        $nodes[] = null;
        $nodes[] = null;
        $nodes[$id] = [$bestF, $bestT, $l, $l + 1, 0];
        $stack[] = [$l + 1, $right, $depth + 1];
        $stack[] = [$l, $left, $depth + 1];
    }
    return $nodes;
}

function tree_predict(array $nodes, array $queries): array
{
    $out = [];
    foreach ($queries as $q) {
        $i = 0;
        while ($nodes[$i][0] >= 0) {
            $i = $q[$nodes[$i][0]] < $nodes[$i][1] ? $nodes[$i][2] : $nodes[$i][3];
        }
        $out[] = $nodes[$i][4];
    }
    return $out;
}

// Dense ReLU network with a softmax output, mini-batch Adam. $W[l] is out x in.
function mlp_forward(array $W, array $b, array $x): array
{
    $acts = [$x];
    $L = count($W);
    for ($l = 0; $l < $L; $l++) {
        $a = [];
        foreach ($W[$l] as $o => $row) {
            $z = $b[$l][$o];
            foreach ($row as $i => $w) {
                $z += $w * $x[$i];
            }
            $a[] = $l < $L - 1 && $z < 0 ? 0.0 : $z;
        }
        $acts[] = $x = $a;
    }
    return $acts;
}

function mlp_fit(array $X, array $y, array $W, array $perms, int $batch, float $lr): array
{
    $L = count($W);
    $b = $mW = $vW = $mb = $vb = [];
    for ($l = 0; $l < $L; $l++) {
        $b[$l] = $mb[$l] = $vb[$l] = array_fill(0, count($W[$l]), 0.0);
        $mW[$l] = $vW[$l] = array_map(fn ($r) => array_fill(0, count($r), 0.0), $W[$l]);
    }
    $t = 0;
    foreach ($perms as $perm) {
        foreach (array_chunk($perm, $batch) as $rows) {
            $inv = 1.0 / count($rows);
            $gW = array_map(fn ($m) => array_map(fn ($r) => array_fill(0, count($r), 0.0), $m), $W);
            $gb = array_map(fn ($v) => array_fill(0, count($v), 0.0), $b);
            foreach ($rows as $r) {
                $acts = mlp_forward($W, $b, $X[$r]);
                $out = $acts[$L];
                $max = max($out);
                $e = array_map(fn ($v) => exp($v - $max), $out);
                $s = array_sum($e);
                $delta = [];
                foreach ($e as $k => $v) {
                    $delta[$k] = ($v / $s - ($k === $y[$r] ? 1.0 : 0.0)) * $inv;
                }
                for ($l = $L - 1; $l >= 0; $l--) {
                    $prev = $acts[$l];
                    $back = $l > 0 ? array_fill(0, count($prev), 0.0) : null;
                    foreach ($delta as $o => $dv) {
                        $gb[$l][$o] += $dv;
                        $row = $W[$l][$o];
                        foreach ($prev as $i => $p) {
                            $gW[$l][$o][$i] += $dv * $p;
                            if ($back !== null) {
                                $back[$i] += $row[$i] * $dv;
                            }
                        }
                    }
                    if ($back !== null) {
                        foreach ($back as $i => $v) {
                            $back[$i] = $prev[$i] > 0 ? $v : 0.0;
                        }
                        $delta = $back;
                    }
                }
            }
            $t++;
            $c1 = 1 - 0.9 ** $t;
            $c2 = 1 - 0.999 ** $t;
            for ($l = 0; $l < $L; $l++) {
                foreach ($W[$l] as $o => $row) {
                    foreach ($row as $i => $w) {
                        $g = $gW[$l][$o][$i];
                        $mW[$l][$o][$i] = $m = 0.9 * $mW[$l][$o][$i] + 0.1 * $g;
                        $vW[$l][$o][$i] = $v = 0.999 * $vW[$l][$o][$i] + 0.001 * $g * $g;
                        $W[$l][$o][$i] = $w - $lr * ($m / $c1) / (sqrt($v / $c2) + 1e-8);
                    }
                    $g = $gb[$l][$o];
                    $mb[$l][$o] = $m = 0.9 * $mb[$l][$o] + 0.1 * $g;
                    $vb[$l][$o] = $v = 0.999 * $vb[$l][$o] + 0.001 * $g * $g;
                    $b[$l][$o] -= $lr * ($m / $c1) / (sqrt($v / $c2) + 1e-8);
                }
            }
        }
    }
    return [$W, $b];
}

function mlp_correct(array $X, array $y, array $W, array $b): int
{
    $n = 0;
    foreach ($X as $i => $x) {
        $out = mlp_forward($W, $b, $x)[count($W)];
        $n += (int) (($out[1] > $out[0] ? 1 : 0) === $y[$i]);
    }
    return $n;
}

function logistic_positives(array $X, array $w, float $b): int
{
    $n = 0;
    foreach ($X as $row) {
        $z = $b;
        foreach ($row as $j => $v) {
            $z += $w[$j] * $v;
        }
        $n += $z >= 0 ? 1 : 0;
    }
    return $n;
}

// ---- runs ----------------------------------------------------------------------

[$dir, $mode, $label] = array_slice($argv, 1);
$X = load("$dir/cls_X.txt");
$labels = array_map(fn ($r) => (int) $r[0], load("$dir/cls_y.txt"));
$Q = load("$dir/knn_Q.txt");
$Xl = load("$dir/lin_X.txt");
$yl = array_map(fn ($r) => $r[0], load("$dir/lin_y.txt"));
$Xk = load("$dir/km_X.txt");
$seeds = array_slice($Xk, 0, 4);

$Xt = load("$dir/tree_X.txt");
$yt = array_map(fn ($r) => (int) $r[0], load("$dir/tree_y.txt"));
$Qt = load("$dir/tree_Q.txt");
$W = [load("$dir/mlp_W1.txt"), load("$dir/mlp_W2.txt"), load("$dir/mlp_W3.txt")];
$perms = array_map(fn ($r) => array_map('intval', $r), load("$dir/mlp_perm.txt"));

$sumCentroids = fn ($cs) => sprintf('%.6f', array_sum(array_map('array_sum', $cs)));

if ($mode === 'pure') {
    bench($label, 'knn', fn () => knn_predict($X, $labels, $Q, 5), fn ($p) => array_sum($p));
    bench($label, 'logreg', fn () => logistic_fit($X, $labels, 0.5, 200), fn ($m) => logistic_positives($X, $m[0], $m[1]));
    bench($label, 'linreg', fn () => linear_fit($Xl, $yl), fn ($w) => sprintf('%.6f', array_sum($w)));
    bench($label, 'kmeans', fn () => kmeans($Xk, $seeds, 20), $sumCentroids);
    bench($label, 'tree', fn () => tree_fit($Xt, $yt, 10, 3, 1e-7), fn ($t) => array_sum(tree_predict($t, $Qt)));
    bench($label, 'mlp', fn () => mlp_fit($X, $labels, $W, $perms, 64, 0.005), fn ($m) => mlp_correct($X, $labels, $m[0], $m[1]));
} else {
    bench($label, 'knn', fn () => (new KNearestNeighbors(5))->fit($X, $labels)->predict($Q), fn ($p) => array_sum($p));
    bench($label, 'logreg', fn () => (new LogisticRegression(0.5, 200, 0.0, 0.0))->fit($X, $labels),
        fn ($m) => array_sum($m->predict($X)));
    bench($label, 'linreg', fn () => (new LinearRegression())->fit($Xl, $yl),
        fn ($m) => sprintf('%.6f', array_sum($m->getCoefficients()) + $m->getIntercept()));
    bench($label, 'kmeans', fn () => (new KMeans(4, 20, 0.0, centroids: $seeds))->fit($Xk), fn ($m) => $sumCentroids($m->getCentroids()));
    bench($label, 'tree', fn () => (new ClassificationTree(10, 3, 1e-7))->fit($Xt, $yt), fn ($m) => array_sum($m->predict($Qt)));
    // Fixed 10 epochs: no hold-out, no early stopping.
    bench($label, 'mlp', fn () => (new MLPClassifier(
        [new Dense(32), new Activation('relu'), new Dense(16), new Activation('relu')],
        64, new Adam(0.005), 10, 0.0, 1000, 0.0
    ))->fit($X, $labels), function ($m) use ($X, $labels) {
        $n = 0;
        foreach ($m->predict($X) as $i => $p) {
            $n += (int) ($p === $labels[$i]);
        }
        return $n;
    });
}
