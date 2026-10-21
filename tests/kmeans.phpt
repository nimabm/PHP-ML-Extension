--TEST--
ML\Clustering\KMeans
--FILE--
<?php
use ML\Clustering\KMeans;

$offsets = [[0, 0], [0.5, 0], [0, 0.5], [-0.5, 0], [0, -0.5]];
$X = [];
foreach ([[0, 0], [10, 10], [0, 10]] as [$cx, $cy]) {
    foreach ($offsets as [$dx, $dy]) {
        $X[] = [$cx + $dx, $cy + $dy];
    }
}

$km = (new KMeans(3, seed: 42))->fit($X);
$labels = $km->getLabels();
$groups = array_chunk($labels, 5);
foreach ($groups as $group) {
    var_dump(count(array_unique($group)) === 1);
}
var_dump(count(array_unique(array_column($groups, 0))) === 3);
printf("inertia %.4f\n", $km->getInertia());

$centroids = array_map(fn ($c) => array_map(fn ($v) => round($v, 4) + 0.0, $c), $km->getCentroids());
sort($centroids);
echo json_encode($centroids), "\n";

var_dump($km->predict([9, 9]) === $labels[5]);
var_dump($km->predict([[0.1, 0.1], [0.2, 9.8]]) === [$labels[0], $labels[10]]);
var_dump($km->getIterations() >= 1);

echo "--- seeded runs are reproducible\n";
var_dump((new KMeans(3, seed: 7))->fit($X)->getCentroids() === (new KMeans(3, seed: 7))->fit($X)->getCentroids());

echo "--- errors\n";
try {
    (new KMeans(100))->fit($X);
} catch (ML\Exception\InvalidArgumentException $e) {
    echo $e->getMessage(), "\n";
}
try {
    (new KMeans())->getCentroids();
} catch (ML\Exception\NotFittedException $e) {
    echo $e->getMessage(), "\n";
}

echo "--- serialize\n";
$copy = unserialize(serialize($km));
var_dump($copy->predict($X) === $km->predict($X), $copy->getLabels() === $labels);
?>
--EXPECT--
bool(true)
bool(true)
bool(true)
bool(true)
inertia 3.0000
[[0,0],[0,10],[10,10]]
bool(true)
bool(true)
bool(true)
--- seeded runs are reproducible
bool(true)
--- errors
Number of clusters (100) cannot exceed the number of samples (15)
ML\Clustering\KMeans is not fitted yet; call fit() first
--- serialize
bool(true)
bool(true)
