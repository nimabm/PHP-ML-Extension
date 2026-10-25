--TEST--
MiniBatchKMeans (batch and incremental training) and SpectralClustering
--FILE--
<?php
require __DIR__ . '/helpers.inc';

use ML\Clustering\{MiniBatchKMeans, SpectralClustering, KMeans};
use ML\Datasets;
use ML\Metrics;

echo "--- MiniBatchKMeans\n";
$big = Datasets::agglomerate([
    'a' => Datasets::blob([0, 0], 0.5, 400, 1),
    'b' => Datasets::blob([5, 5], 0.5, 400, 2),
    'c' => Datasets::blob([0, 5], 0.5, 400, 3),
], true, 1);
$mb = (new MiniBatchKMeans(3, 100, seed: 1))->fit($big['samples']);
check('finds the three blobs', Metrics::vMeasure($big['labels'], $mb->getLabels()) > 0.99);
$km = (new KMeans(3, seed: 1))->fit($big['samples']);
check('inertia close to full k-means', $mb->getInertia() <= 1.01 * $km->getInertia());
check('centroids', count($mb->getCentroids()) === 3);
check('stops early', $mb->getIterations() < 100);
check('predict', $mb->predict($big['samples'][0]) === $mb->getLabels()[0]);
$p = $mb->predictProbability([0.0, 0.0]);
check('memberships sum to 1', abs(array_sum($p) - 1.0) < 1e-12 && max($p) > 0.9);
check('seeded runs repeat', (new MiniBatchKMeans(3, 100, seed: 1))->fit($big['samples'])->getCentroids() === $mb->getCentroids());
ini_set('mlnative.threads', 4);
check('threads do not change the result', (new MiniBatchKMeans(3, 100, seed: 1))->fit($big['samples'])->getCentroids() === $mb->getCentroids());
ini_set('mlnative.threads', 1);
check('implements ML\Online', $mb instanceof ML\Online);

$online = new MiniBatchKMeans(3, seed: 2);
foreach (array_chunk($big['samples'], 200) as $chunk) {
    $online->partialFit($chunk);
}
check('partialFit learns incrementally', Metrics::vMeasure($big['labels'], $online->predict($big['samples'])) > 0.99);
var_dump($online->getIterations(), count($online->getLabels()));
$preset = (new MiniBatchKMeans(2, centroids: [[0, 0], [5, 5]]))->fit([[0, 1], [1, 0], [5, 6], [6, 5]]);
echo json_encode($preset->getLabels()), "\n";
check('serialization', same_after_serialize($mb, $big['samples']));
$copy = clone $online;
$copy->partialFit([[100, 100], [100, 101], [101, 100]]);
check('clone is independent', $online->getCentroids() !== $copy->getCentroids());
expect_error(fn () => (new MiniBatchKMeans(5))->partialFit([[1, 2], [3, 4]]));
expect_error(fn () => $online->partialFit([[1, 2, 3]]));
expect_error(fn () => new MiniBatchKMeans(batchSize: 0));
expect_error(fn () => new MiniBatchKMeans(seeder: 'preset'));
expect_error(fn () => (new MiniBatchKMeans())->predict([1, 2]));

echo "--- SpectralClustering\n";
$c = circles(100, 1);
check('k-means cannot split concentric circles', Metrics::vMeasure($c['labels'], (new KMeans(2, seed: 1))->fit($c['samples'])->getLabels()) < 0.1);
check('spectral (nearest neighbors) can', Metrics::vMeasure($c['labels'], (new SpectralClustering(2, 'nearest', neighbors: 10, seed: 1))->predict($c['samples'])) > 0.99);
check('spectral (rbf) can', Metrics::vMeasure($c['labels'], (new SpectralClustering(2, 'rbf', gamma: 2.0, seed: 1))->predict($c['samples'])) > 0.99);
$moons = Datasets::agglomerate([
    'a' => Datasets::halfMoon(0, 0, 1, 90, 0.05, 80, 1),
    'b' => Datasets::halfMoon(1, -0.5, 1, 270, 0.05, 80, 2),
], true, 1);
check('half moons', Metrics::vMeasure($moons['labels'], (new SpectralClustering(2, 'nearest', seed: 1))->predict($moons['samples'])) > 0.99);
$s = new SpectralClustering(3, seed: 4);
$labels = $s->predict(blobs(20, 3)['samples']);
check('labels are cluster numbers', min($labels) === 0 && max($labels) === 2);
check('serialization keeps parameters', unserialize(serialize($s))->predict(blobs(20, 3)['samples']) === $labels);
expect_error(fn () => (new SpectralClustering(5))->predict([[1, 2], [3, 4]]));
expect_error(fn () => new SpectralClustering(affinity: 'cosine'));
expect_error(fn () => new SpectralClustering(gamma: 0.0));
?>
--EXPECT--
--- MiniBatchKMeans
finds the three blobs: ok
inertia close to full k-means: ok
centroids: ok
stops early: ok
predict: ok
memberships sum to 1: ok
seeded runs repeat: ok
threads do not change the result: ok
implements ML\Online: ok
partialFit learns incrementally: ok
int(6)
int(200)
[0,0,1,1]
serialization: ok
clone is independent: ok
The first partialFit() batch needs at least as many samples as clusters (5), 2 given
Sample #0 has 3 features, expected 2
batchSize must be at least 1
The preset seeder needs the centroids parameter
ML\Clustering\MiniBatchKMeans is not fitted yet; call fit() first
--- SpectralClustering
k-means cannot split concentric circles: ok
spectral (nearest neighbors) can: ok
spectral (rbf) can: ok
half moons: ok
labels are cluster numbers: ok
serialization keeps parameters: ok
Number of clusters (5) cannot exceed the number of samples (2)
affinity must be "rbf" or "nearest"
gamma must be greater than 0
