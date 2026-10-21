--TEST--
Clustering: KMeans seeders, KMedoids, FuzzyCMeans, GaussianMixture, MeanShift, DBSCAN, OPTICS, Ward
--FILE--
<?php
require __DIR__ . '/helpers.inc';

use ML\Clustering\{KMeans, KMedoids, FuzzyCMeans, GaussianMixture, MeanShift, DBSCAN, OPTICS, Ward};
use ML\Metrics;

$d = blobs(50, 7, 0.4);
$X = $d['samples'];
$y = $d['labels'];
$perfect = fn (array $clusters) => Metrics::vMeasure($y, $clusters) > 0.999;

foreach (['plusplus', 'random', 'kmc2'] as $seeder) {
    check("KMeans $seeder", $perfect((new KMeans(3, seed: 1, seeder: $seeder))->fit($X)->getLabels()));
}
$preset = (new KMeans(3, centroids: [[0, 0], [4, 4], [0, 5]]))->fit($X);
check('KMeans preset', $perfect($preset->getLabels()) && $preset->predict([0.1, 0.1]) === 0);
$p = $preset->predictProbability([0, 0]);
check('KMeans probabilities', abs(array_sum($p) - 1) < 1e-12 && $p[0] > 0.8);

$m = (new KMedoids(3, seed: 1))->fit($X);
check('KMedoids', $perfect($m->getLabels()) && count($m->getMedoids()) === 3);
check('medoids are samples', in_array($m->getMedoids()[0], $X, true));

$f = (new FuzzyCMeans(3, seed: 1))->fit($X);
check('FuzzyCMeans', $perfect($f->predict($X)));
$u = $f->predictProbability([2, 2.5]);
check('fuzzy memberships', abs(array_sum($u) - 1) < 1e-9 && max($u) < 0.9);

$g = (new GaussianMixture(3, seed: 1))->fit($X);
check('GaussianMixture', $perfect($g->predict($X)) && abs(array_sum($g->getPriors()) - 1) < 1e-9);
check('mixture shapes', count($g->getMeans()) === 3 && count($g->getVariances()[0]) === 2 && $g->getLogLikelihood() < 0);

$ms = (new MeanShift(seed: 1))->fit($X);
check('MeanShift', $perfect($ms->predict($X)) && count($ms->getCentroids()) === 3);
check('MeanShift radius estimate', MeanShift::estimateRadius($X, 30, 1) > 0);

check('DBSCAN', $perfect((new DBSCAN(1.0, 4))->predict($X)));
echo json_encode((new DBSCAN(0.5, 2))->predict([[0], [0.1], [5], [5.1], [20]])), "\n";
check('DBSCAN cosine', count((new DBSCAN(0.1, 2, 'cosine'))->predict($X)) === count($X));

$op = new OPTICS(4, 1.0);
check('OPTICS', $perfect($op->predict($X)));
$r = (new OPTICS(2, 1.0))->reachability([[0], [0.1], [0.2], [5], [5.1], [5.2]]);
echo json_encode(array_keys($r)), ' ', json_encode($r['labels']), "\n";

check('Ward', $perfect((new Ward(3))->predict($X)));
echo json_encode((new Ward(2))->predict([[0], [0.1], [0.2], [10], [10.1]])), "\n";
echo json_encode((new Ward(5))->predict([[0], [1], [2], [3], [4]])), "\n";

echo "--- serialization\n";
foreach ([$m, $f, $g, $ms, $preset] as $model) {
    check(get_class($model), same_after_serialize($model, $X));
}
check('DBSCAN', unserialize(serialize(new DBSCAN(1.0, 4)))->predict($X) === (new DBSCAN(1.0, 4))->predict($X));

echo "--- errors\n";
expect_error(fn () => new KMeans(3, seeder: 'preset'));
expect_error(fn () => new KMeans(3, seeder: 'bogus'));
expect_error(fn () => new KMeans(2, centroids: [[0, 0]]));
expect_error(fn () => (new KMedoids(500))->fit($X));
expect_error(fn () => new FuzzyCMeans(3, 1.0));
expect_error(fn () => new OPTICS(3, 1.0, 0.5));
expect_error(fn () => (new Ward(10))->predict([[1], [2]]));
?>
--EXPECT--
KMeans plusplus: ok
KMeans random: ok
KMeans kmc2: ok
KMeans preset: ok
KMeans probabilities: ok
KMedoids: ok
medoids are samples: ok
FuzzyCMeans: ok
fuzzy memberships: ok
GaussianMixture: ok
mixture shapes: ok
MeanShift: ok
MeanShift radius estimate: ok
DBSCAN: ok
[0,0,1,1,-1]
DBSCAN cosine: ok
OPTICS: ok
["ordering","reachability","coreDistances","labels"] [0,0,0,1,1,1]
Ward: ok
[0,0,0,1,1]
[0,1,2,3,4]
--- serialization
ML\Clustering\KMedoids: ok
ML\Clustering\FuzzyCMeans: ok
ML\Clustering\GaussianMixture: ok
ML\Clustering\MeanShift: ok
ML\Clustering\KMeans: ok
DBSCAN: ok
--- errors
The preset seeder needs the centroids parameter
Unknown seeder "bogus"; expected plusplus, random, kmc2 or preset
centroids must contain exactly one sample per cluster
Number of clusters (500) cannot exceed the number of samples (150)
fuzz must be greater than 1
maxRadius must be at least radius
clusters must be between 1 and the number of samples
