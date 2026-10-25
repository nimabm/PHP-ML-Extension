--TEST--
PCA, TruncatedSVD, LinearDiscriminantAnalysis, random projectors and t-SNE
--FILE--
<?php
require __DIR__ . '/helpers.inc';

use ML\Preprocessing\{PCA, TruncatedSVD, LinearDiscriminantAnalysis, GaussianRandomProjector, SparseRandomProjector, TSNE};
use ML\Metrics;

// Points on the line y = 2x: one component explains everything.
$line = [];
for ($i = -5; $i <= 5; $i++) {
    $line[] = [$i, 2 * $i];
}
$pca = (new PCA(1))->fit($line);
echo json_encode(rounded($pca->getExplainedVarianceRatio())), ' ', json_encode(rounded($pca->getComponents())), "\n";
echo json_encode(rounded($pca->transform([1, 2]))), "\n";

$d = blobs(30, 19, 0.5);
$X = array_map(fn ($r) => [$r[0], $r[1], $r[0] + $r[1], 0.5 * $r[0]], $d['samples']);
$pca2 = new PCA(2);
$Z = $pca2->fitTransform($X);
check('PCA keeps nearly all variance', array_sum($pca2->getExplainedVarianceRatio()) > 0.999);
check('PCA output width', count($Z[0]) === 2);
$svd = (new TruncatedSVD(2))->fit($X);
check('TruncatedSVD', count($svd->getSingularValues()) === 2 && $svd->getSingularValues()[0] >= $svd->getSingularValues()[1]);
check('TruncatedSVD ratio', array_sum($svd->getExplainedVarianceRatio()) > 0.999);
$lda = new LinearDiscriminantAnalysis(2);
$L = $lda->fitTransform($X, $d['labels']);
check('LDA separates the classes', Metrics::vMeasure($d['labels'], (new ML\Clustering\KMeans(3, seed: 1))->fit($L)->getLabels()) > 0.99);
check('LDA explained variance', count($lda->getExplainedVariance()) === 2);
var_dump(GaussianRandomProjector::minDimensions(1000, 0.3), SparseRandomProjector::minDimensions(100, 0.5));
$g = (new GaussianRandomProjector(3, 1))->fitTransform($X);
$s = (new SparseRandomProjector(3, 3.0, 1))->fitTransform($X);
check('projector widths', count($g[0]) === 3 && count($s[0]) === 3);
check('seeded projections repeat', (new GaussianRandomProjector(3, 1))->fitTransform($X) === $g);

$tsne = new TSNE(2, perplexity: 10, epochs: 300, seed: 1);
$E = $tsne->fitTransform($d['samples']);
check('t-SNE preserves neighborhoods', Metrics::trustworthiness($d['samples'], $E, 5) > 0.9);
check('t-SNE losses recorded', count($tsne->getLosses()) > 5);

echo "--- serialization\n";
foreach ([$pca2, $svd, $lda, (new GaussianRandomProjector(2, 3))->fit($X)] as $t) {
    check(get_class($t), unserialize(serialize($t))->transform($X) === $t->transform($X));
}

echo "--- errors\n";
expect_error(fn () => (new PCA(9))->fit($X));
expect_error(fn () => (new LinearDiscriminantAnalysis(3))->fit($X, $d['labels']));
expect_error(fn () => (new TSNE(perplexity: 50))->fitTransform(array_slice($d['samples'], 0, 20)));
expect_error(fn () => new SparseRandomProjector(2, 0.5));
expect_error(fn () => (new PCA())->transform([1, 2]));
?>
--EXPECT--
[1] [[0.447,0.894]]
[2.236]
PCA keeps nearly all variance: ok
PCA output width: ok
TruncatedSVD: ok
TruncatedSVD ratio: ok
LDA separates the classes: ok
LDA explained variance: ok
int(768)
int(222)
projector widths: ok
seeded projections repeat: ok
t-SNE preserves neighborhoods: ok
t-SNE losses recorded: ok
--- serialization
ML\Preprocessing\PCA: ok
ML\Preprocessing\TruncatedSVD: ok
ML\Preprocessing\LinearDiscriminantAnalysis: ok
ML\Preprocessing\GaussianRandomProjector: ok
--- errors
dimensions (9) cannot exceed the number of features (4)
dimensions (3) cannot exceed min(features, classes - 1) (2)
perplexity must be less than the number of samples - 1
sparsity must be at least 1
ML\Preprocessing\PCA is not fitted yet; call fit() first
