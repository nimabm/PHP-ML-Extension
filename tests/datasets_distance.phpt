--TEST--
ML\Datasets generators and splitting, ML\Distance
--FILE--
<?php
require __DIR__ . '/helpers.inc';

use ML\Datasets;
use ML\Distance;

$blob = Datasets::blob([10, -10], 0.5, 500, 1);
$mean = [array_sum(array_column($blob, 0)) / 500, array_sum(array_column($blob, 1)) / 500];
check('blob center', abs($mean[0] - 10) < 0.1 && abs($mean[1] + 10) < 0.1);
check('blob per-feature stddev', count(Datasets::blob([0, 0], [1, 0], 5, 1)) === 5 && Datasets::blob([0, 0], [1, 0], 5, 1)[0][1] == 0.0);
check('seeded generators repeat', Datasets::blob([0], 1.0, 3, 9) === Datasets::blob([0], 1.0, 3, 9));
$circle = Datasets::circle(0, 0, 2.0, 0.0, 50, 1);
check('circle radius', abs(sqrt($circle[0][0] ** 2 + $circle[0][1] ** 2) - 2.0) < 1e-9);
$moon = Datasets::halfMoon(0, 0, 1.0, 0.0, 0.0, 50, 1);
check('half moon is the upper half', min(array_column($moon, 1)) >= -1e-9);
$roll = Datasets::swissRoll(n: 30, seed: 1);
check('swiss roll shape', count($roll['samples']) === 30 && count($roll['samples'][0]) === 3 && count($roll['targets']) === 30);
$plane = Datasets::hyperplane([2, -3], 1.0, 0.0, 20, 1);
check('hyperplane is exact without noise', abs($plane['targets'][0] - (1 + 2 * $plane['samples'][0][0] - 3 * $plane['samples'][0][1])) < 1e-12);
$agg = Datasets::agglomerate(['x' => [[1], [2]], 7 => [[3]]], false);
echo json_encode($agg), "\n";

[$a, $b, $c, $d] = Datasets::trainTestSplit(range(1, 20), array_merge(array_fill(0, 10, 'p'), array_fill(0, 10, 'q')), 0.3, true, 1);
echo count($a), ' ', count($b), ' ', count(array_filter($d, fn ($l) => $l === 'p')), "\n";

echo "--- distances\n";
foreach (['euclidean', 'squared_euclidean', 'manhattan', 'chebyshev', 'minkowski', 'minkowski(1)', 'cosine', 'canberra', 'hamming', 'jaccard', 'safe_euclidean'] as $m) {
    printf("%s %.4f\n", $m, Distance::compute($m, [0, 3, 1], [4, 0, 1]));
}
printf("safe_euclidean with NAN %.4f\n", Distance::compute('safe_euclidean', [0, NAN], [3, 7]));
echo json_encode(Distance::pairwise('manhattan', [[0], [2], [5]])), "\n";
printf("gower %.4f\n", Distance::gower([1, 'a', null], [3, 'b', 5], [4, 0, 1]));

echo "--- errors\n";
expect_error(fn () => Distance::compute('bogus', [1], [2]));
expect_error(fn () => Distance::compute('euclidean', [1, 2], [2]));
expect_error(fn () => Datasets::trainTestSplit([1, 2], ['a', 'b'], 1.5));
expect_error(fn () => Datasets::blob([0], -1.0));
?>
--EXPECT--
blob center: ok
blob per-feature stddev: ok
seeded generators repeat: ok
circle radius: ok
half moon is the upper half: ok
swiss roll shape: ok
hyperplane is exact without noise: ok
{"samples":[[1],[2],[3]],"labels":["x","x",7]}
14 6 3
--- distances
euclidean 5.0000
squared_euclidean 25.0000
manhattan 7.0000
chebyshev 4.0000
minkowski 4.4979
minkowski(1) 7.0000
cosine 0.9233
canberra 2.0000
hamming 0.6667
jaccard 0.8750
safe_euclidean 5.0000
safe_euclidean with NAN 4.2426
[[0,2,5],[2,0,3],[5,3,0]]
gower 0.7500
--- errors
Unknown distance metric "bogus"; expected euclidean, squared_euclidean, manhattan, chebyshev, minkowski, minkowski(<p>), cosine, canberra, hamming, jaccard or safe_euclidean
Sample has 1 features, expected 2
testRatio must be between 0 and 1 (exclusive)
stddev must be greater than or equal to 0
