--TEST--
KernelPCA, Isomap, UMAP, FastICA, NMF and the eigensolver for large matrices
--FILE--
<?php
require __DIR__ . '/helpers.inc';

use ML\Preprocessing\{KernelPCA, Isomap, UMAP, FastICA, NMF, PCA};
use ML\Clustering\KMeans;
use ML\Datasets;
use ML\Metrics;

echo "--- eigensolver\n";
// 40 features (above the Jacobi limit): the components must stay orthonormal and
// the explained variances must add up to the total variance.
$stddev = [];
for ($j = 0; $j < 40; $j++) {
    $stddev[] = $j === 0 ? 20.0 : sqrt(40 - $j);
}
$wide = Datasets::blob(array_fill(0, 40, 0.0), $stddev, 300, 5);
$pca = (new PCA(40))->fit($wide);
$W = $pca->getComponents();
$ortho = 0.0;
foreach ($W as $a => $u) {
    foreach ($W as $b => $v) {
        $dot = 0.0;
        foreach ($u as $j => $x) {
            $dot += $x * $v[$j];
        }
        $ortho = max($ortho, abs($dot - ($a === $b ? 1.0 : 0.0)));
    }
}
check('components are orthonormal', $ortho < 1e-9);
check('explained variance ratio sums to 1', abs(array_sum($pca->getExplainedVarianceRatio()) - 1.0) < 1e-9);
$variances = $sorted = $pca->getExplainedVariance();
rsort($sorted);
check('variances are sorted', $sorted === $variances);
check('first component follows the widest feature', abs($W[0][0]) > 0.9);

echo "--- KernelPCA\n";
$circles = Datasets::agglomerate([
    'in' => Datasets::circle(0, 0, 1.0, 0.05, 100, 1),
    'out' => Datasets::circle(0, 0, 3.0, 0.05, 100, 2),
], true, 1);
$kpca = new KernelPCA(1, 'rbf', gamma: 0.3);
$Z = $kpca->fitTransform($circles['samples']);
check('one RBF component separates the circles', Metrics::vMeasure($circles['labels'], (new KMeans(2, seed: 1))->fit($Z)->getLabels()) > 0.99);
check('transform() reproduces fitTransform()', abs($kpca->transform($circles['samples'][3])[0] - $Z[3][0]) < 1e-9);
check('eigenvalues', count($kpca->getEigenvalues()) === 1 && $kpca->getEigenvalues()[0] > 0);
$blob = Datasets::blob([0, 0, 0], [3, 2, 1], 50, 4);
$linear = (new KernelPCA(2, 'linear'))->fitTransform($blob);
$plain = (new PCA(2))->fitTransform($blob);
$diff = 0.0;
foreach ($linear as $i => $row) {
    foreach ($row as $j => $v) {
        $diff = max($diff, abs(abs($v) - abs($plain[$i][$j])));
    }
}
check('a linear kernel is PCA', $diff < 1e-9);

echo "--- Isomap\n";
$corr = function (array $x, array $y): float {
    $mx = array_sum($x) / count($x);
    $my = array_sum($y) / count($y);
    $sxy = $sxx = $syy = 0.0;
    foreach ($x as $i => $v) {
        $sxy += ($v - $mx) * ($y[$i] - $my);
        $sxx += ($v - $mx) ** 2;
        $syy += ($y[$i] - $my) ** 2;
    }
    return abs($sxy / sqrt($sxx * $syy));
};
$roll = Datasets::swissRoll(depth: 6.0, n: 300, seed: 3);
$iso = new Isomap(2, 10);
$E = $iso->fitTransform($roll['samples']);
// The first coordinate follows the position along the roll; PCA cannot unroll it.
check('Isomap unrolls the swiss roll', $corr(array_column($E, 0), $roll['targets']) > 0.95);
check('PCA does not', $corr(array_column((new PCA(2))->fitTransform($roll['samples']), 0), $roll['targets']) < 0.5);
$t = $iso->transform($roll['samples'][7]);
check('training points map onto their embedding', abs($t[0] - $E[7][0]) < 1e-6 && abs($t[1] - $E[7][1]) < 1e-6);
check('eigenvalues are descending', $iso->getEigenvalues()[0] >= $iso->getEigenvalues()[1]);
expect_error(fn () => (new Isomap(2, 5))->fit(array_merge(Datasets::blob([0, 0], 0.1, 10, 1), Datasets::blob([50, 50], 0.1, 10, 2))));

echo "--- UMAP\n";
$blobs = Datasets::agglomerate([
    'a' => Datasets::blob([0, 0, 0, 0], 1.0, 60, 1),
    'b' => Datasets::blob([8, 8, 0, 0], 1.0, 60, 2),
    'c' => Datasets::blob([0, 8, 8, 0], 1.0, 60, 3),
], true, 1);
$umap = new UMAP(2, neighbors: 10, epochs: 200, seed: 1);
$U = $umap->fitTransform($blobs['samples']);
check('UMAP keeps neighborhoods', Metrics::trustworthiness($blobs['samples'], $U, 10) > 0.9);
$km = (new KMeans(3, seed: 1))->fit($U);
check('UMAP keeps the clusters apart', Metrics::vMeasure($blobs['labels'], $km->getLabels()) > 0.99);
check('getEmbedding() is the training embedding', $umap->getEmbedding() === $U);
check('seeded UMAP repeats', (new UMAP(2, neighbors: 10, epochs: 200, seed: 1))->fitTransform($blobs['samples']) === $U);
$fresh = Datasets::agglomerate(['a' => Datasets::blob([0, 0, 0, 0], 1.0, 5, 7), 'b' => Datasets::blob([8, 8, 0, 0], 1.0, 5, 8)], false);
$placed = $umap->transform($fresh['samples']);
check('new samples land in their cluster', Metrics::vMeasure($fresh['labels'], $km->predict($placed)) > 0.99);
check('single sample transform', count($umap->transform($fresh['samples'][0])) === 2);
ini_set('mlnative.threads', 4);
check('results do not depend on threads', (new UMAP(2, neighbors: 10, epochs: 200, seed: 1))->fitTransform($blobs['samples']) === $U
    && $umap->transform($fresh['samples']) === $placed);
ini_set('mlnative.threads', 1);
expect_error(fn () => (new UMAP(5))->fit([[1, 2], [3, 4], [5, 6]]));
expect_error(fn () => new UMAP(minDistance: 2.0));
expect_error(fn () => new UMAP(init: 'spectral'));

echo "--- FastICA\n";
$sources = $mixed = [];
for ($i = 0; $i < 500; $i++) {
    $s = [sin($i / 10), ($i % 40) < 20 ? 1.0 : -1.0];
    $sources[] = $s;
    $mixed[] = [$s[0] + 0.5 * $s[1], 0.3 * $s[0] + $s[1]];
}
$ica = new FastICA(seed: 1);
$R = $ica->fitTransform($mixed);
$best = max($corr(array_column($R, 0), array_column($sources, 0)), $corr(array_column($R, 1), array_column($sources, 0)));
$best2 = max($corr(array_column($R, 0), array_column($sources, 1)), $corr(array_column($R, 1), array_column($sources, 1)));
check('FastICA recovers both sources', $best > 0.99 && $best2 > 0.99);
check('mixing matrix shape', count($ica->getMixingMatrix()) === 2 && count($ica->getMixingMatrix()[0]) === 2);
check('unmixing components', count($ica->getComponents()) === 2);
check('converged', $ica->getIterations() < 200);
foreach (['exp', 'cube'] as $contrast) {
    $R2 = (new FastICA(contrast: $contrast, seed: 1))->fitTransform($mixed);
    check("contrast $contrast", max($corr(array_column($R2, 0), array_column($sources, 1)), $corr(array_column($R2, 1), array_column($sources, 1))) > 0.98);
}
expect_error(fn () => (new FastICA(2))->fit([[1, 2], [2, 4], [3, 6]]));
expect_error(fn () => new FastICA(contrast: 'tanh'));

echo "--- NMF\n";
$V = [];
for ($i = 0; $i < 60; $i++) {
    $w = [$i % 3 === 0 ? 1 : 0, $i % 3 === 1 ? 1 : 0, $i % 3 === 2 ? 1 : 0];
    $V[] = [5 * $w[0] + $w[2], 5 * $w[0], 3 * $w[1], 3 * $w[1] + $w[2], 2 * $w[2]];
}
$nmf = new NMF(3, seed: 1);
$Wfit = $nmf->fitTransform($V);
check('NMF reconstructs a rank-3 matrix', $nmf->getReconstructionError() < 1e-3);
check('components are non-negative', min(array_merge(...$nmf->getComponents())) >= 0.0);
check('transform() agrees with the fit', max(array_map(fn ($a, $b) => abs($a - $b), $nmf->transform($V[1]), $Wfit[1])) < 1e-3);
check('NMF iterations', $nmf->getIterations() >= 1);
expect_error(fn () => (new NMF(2))->fit([[1, 2], [3, -4]]));
expect_error(fn () => (new NMF(9))->fit([[1, 2], [3, 4]]));
expect_error(fn () => (new NMF())->transform([1, 2]));

echo "--- serialization\n";
foreach ([$kpca, $iso, $umap, $ica, $nmf] as $model) {
    $X = $model === $nmf ? $V : ($model === $ica ? $mixed : ($model === $kpca ? $circles['samples'] : ($model === $iso ? $roll['samples'] : $fresh['samples'])));
    $copy = unserialize(serialize($model));
    check(get_class($model), $copy->transform(array_slice($X, 0, 5)) === $model->transform(array_slice($X, 0, 5)));
}
$clone = clone $umap;
check('clone is independent', $clone->transform($fresh['samples']) === $placed);
?>
--EXPECT--
--- eigensolver
components are orthonormal: ok
explained variance ratio sums to 1: ok
variances are sorted: ok
first component follows the widest feature: ok
--- KernelPCA
one RBF component separates the circles: ok
transform() reproduces fitTransform(): ok
eigenvalues: ok
a linear kernel is PCA: ok
--- Isomap
Isomap unrolls the swiss roll: ok
PCA does not: ok
training points map onto their embedding: ok
eigenvalues are descending: ok
The neighborhood graph has 2 disconnected components; increase neighbors (currently 5)
--- UMAP
UMAP keeps neighborhoods: ok
UMAP keeps the clusters apart: ok
getEmbedding() is the training embedding: ok
seeded UMAP repeats: ok
new samples land in their cluster: ok
single sample transform: ok
results do not depend on threads: ok
dimensions (5) must be less than the number of samples (3)
minDistance must be between 0 and 1
init must be "pca" or "random"
--- FastICA
FastICA recovers both sources: ok
mixing matrix shape: ok
unmixing components: ok
converged: ok
contrast exp: ok
contrast cube: ok
dimensions (2) cannot exceed the rank of the data (1)
Unknown contrast function "tanh" (expected logcosh, exp or cube)
--- NMF
NMF reconstructs a rank-3 matrix: ok
components are non-negative: ok
transform() agrees with the fit: ok
NMF iterations: ok
NMF needs non-negative features; sample #1, feature #1 is negative
dimensions (9) cannot exceed the number of features (2)
ML\Preprocessing\NMF is not fitted yet; call fit() first
--- serialization
ML\Preprocessing\KernelPCA: ok
ML\Preprocessing\Isomap: ok
ML\Preprocessing\UMAP: ok
ML\Preprocessing\FastICA: ok
ML\Preprocessing\NMF: ok
clone is independent: ok
