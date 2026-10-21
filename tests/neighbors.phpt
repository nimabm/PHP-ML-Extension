--TEST--
Nearest-neighbor classifiers and regressors (brute force, K-d tree, radius)
--FILE--
<?php
require __DIR__ . '/helpers.inc';

use ML\Classification\{KNearestNeighbors, KDNeighbors, RadiusNeighbors};
use ML\Regression\{KNNRegressor, KDNeighborsRegressor, RadiusNeighborsRegressor};
use ML\Datasets;
use ML\Metrics;

$d = blobs(60, 3, 1.0);
[$Xt, $Xs, $yt, $ys] = Datasets::trainTestSplit($d['samples'], $d['labels'], 0.3, true, 3);

foreach (['euclidean', 'manhattan', 'chebyshev', 'minkowski(1.5)'] as $metric) {
    $brute = (new KNearestNeighbors(5, $metric))->fit($Xt, $yt);
    $tree = (new KDNeighbors(5, false, 4, $metric))->fit($Xt, $yt);
    check("K-d tree equals brute force ($metric)", $brute->predict($Xs) === $tree->predict($Xs)
        && $brute->predictProbability($Xs) === $tree->predictProbability($Xs));
}
foreach (['cosine', 'canberra', 'squared_euclidean', 'safe_euclidean'] as $metric) {
    check("KNN with $metric", accuracy((new KNearestNeighbors(5, $metric))->fit($Xt, $yt), $Xs, $ys) > 0.8);
}
check('weighted KNN', accuracy((new KNearestNeighbors(7, weighted: true))->fit($Xt, $yt), $Xs, $ys) > 0.9);

echo "--- radius\n";
$r = (new RadiusNeighbors(1.5, outlierClass: 'none'))->fit($Xt, $yt);
check('radius accuracy', accuracy($r, $Xs, $ys) > 0.9);
var_dump($r->predict([100, 100]));
var_dump($r->predictProbability([100, 100]));
$r2 = (new RadiusNeighbors(0.5, outlierClass: -1, metric: 'cosine'))->fit($Xt, $yt);
var_dump($r2->predict([0, 0]));

echo "--- regressors\n";
$h = Datasets::hyperplane([2, -1], 0.0, 0.05, 300, 2);
[$Ht, $Hs, $ht, $hs] = Datasets::trainTestSplit($h['samples'], $h['targets'], 0.3, false, 2);
$knn = (new KNNRegressor(5))->fit($Ht, $ht);
$kd = (new KDNeighborsRegressor(5))->fit($Ht, $ht);
check('regressor r2', Metrics::r2Score($hs, $knn->predict($Hs)) > 0.9);
check('K-d regressor equals brute force', $knn->predict($Hs) === $kd->predict($Hs));
check('weighted regressor', Metrics::r2Score($hs, (new KNNRegressor(5, true))->fit($Ht, $ht)->predict($Hs)) > 0.9);
$rr = (new RadiusNeighborsRegressor(0.5))->fit($Ht, $ht);
check('radius regressor', Metrics::r2Score($hs, $rr->predict($Hs)) > 0.8);
var_dump(is_nan($rr->predict([50, 50])));
var_dump((new KNNRegressor(2))->fit([[0], [1], [10]], [0.0, 2.0, 100.0])->predict([0.4]));

echo "--- serialization\n";
foreach ([$knn, $kd, $rr, (new KDNeighbors())->fit($Xt, $yt), $r] as $m) {
    check(get_class($m), same_after_serialize($m, $Xs));
}
$legacy = 'O:35:"ML\Classification\KNearestNeighbors":6:{s:1:"k";i:1;s:6:"metric";s:9:"manhattan";s:7:"classes";a:2:{i:0;s:1:"a";i:1;s:1:"b";}s:8:"features";i:1;s:7:"samples";a:2:{i:0;d:0;i:1;d:5;}s:6:"labels";a:2:{i:0;i:0;i:1;i:1;}}';
echo unserialize($legacy)->predict([4]), "\n";

echo "--- errors\n";
expect_error(fn () => new KDNeighbors(3, metric: 'cosine'));
expect_error(fn () => new RadiusNeighbors(0));
expect_error(fn () => (new KNNRegressor(10))->fit([[1], [2]], [1, 2]));
expect_error(fn () => new KNearestNeighbors(3, 'minkowski(0.5)'));
?>
--EXPECT--
K-d tree equals brute force (euclidean): ok
K-d tree equals brute force (manhattan): ok
K-d tree equals brute force (chebyshev): ok
K-d tree equals brute force (minkowski(1.5)): ok
KNN with cosine: ok
KNN with canberra: ok
KNN with squared_euclidean: ok
KNN with safe_euclidean: ok
weighted KNN: ok
--- radius
radius accuracy: ok
string(4) "none"
array(3) {
  ["a"]=>
  float(0)
  ["b"]=>
  float(0)
  ["c"]=>
  float(0)
}
int(-1)
--- regressors
regressor r2: ok
K-d regressor equals brute force: ok
weighted regressor: ok
radius regressor: ok
bool(true)
float(1)
--- serialization
ML\Regression\KNNRegressor: ok
ML\Regression\KDNeighborsRegressor: ok
ML\Regression\RadiusNeighborsRegressor: ok
ML\Classification\KDNeighbors: ok
ML\Classification\RadiusNeighbors: ok
b
--- errors
K-d tree models only support euclidean, squared_euclidean, manhattan, chebyshev and minkowski distances
radius must be greater than 0
k (10) cannot exceed the number of training samples (2)
Minkowski p must be a number >= 1, "0.5" given
