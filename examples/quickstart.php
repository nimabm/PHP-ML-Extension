<?php

declare(strict_types=1);

use ML\Classification\GaussianNB;
use ML\Classification\KNearestNeighbors;
use ML\Classification\LogisticRegression;
use ML\Clustering\KMeans;
use ML\Exception\MLException;
use ML\Metrics;
use ML\Preprocessing\StandardScaler;
use ML\Regression\LinearRegression;

if (!extension_loaded('mlnative')) {
    fwrite(STDERR, "The mlnative extension is not loaded.\n");
    exit(1);
}

// --- Regression: house price from [area m², rooms] ------------------------
$houses = [[50, 1], [60, 2], [80, 2], [100, 3], [120, 3], [150, 4]];
$prices = [150, 185, 235, 295, 345, 430];

$regression = (new LinearRegression())->fit($houses, $prices);
printf("Price of a 90 m², 3-room house: %.1f\n", $regression->predict([90, 3]));
printf("R² on the training set: %.3f\n\n", Metrics::r2Score($prices, $regression->predict($houses)));

// --- Classification: two features, string labels ---------------------------
$samples = [[1.0, 1.1], [1.2, 0.9], [0.8, 1.0], [3.0, 3.2], [3.1, 2.9], [2.9, 3.0]];
$labels = ['cat', 'cat', 'cat', 'dog', 'dog', 'dog'];
$queries = [[1.1, 1.0], [3.0, 3.1]];

$classifiers = [
    'KNearestNeighbors' => new KNearestNeighbors(k: 3),
    'GaussianNB' => new GaussianNB(),
    'LogisticRegression' => new LogisticRegression(learningRate: 0.5, maxIterations: 2000),
];
foreach ($classifiers as $name => $classifier) {
    $classifier->fit($samples, $labels);
    printf("%-18s → %s\n", $name, implode(', ', $classifier->predict($queries)));
}
echo 'Probabilities for [2, 2]: ', json_encode($classifiers['GaussianNB']->predictProbability([2, 2])), "\n\n";

// --- Preprocessing + clustering -------------------------------------------
$points = [[1, 200], [2, 210], [1.5, 190], [8, 900], [9, 950], [8.5, 920]];
$scaled = (new StandardScaler())->fitTransform($points);
$kmeans = (new KMeans(clusters: 2, seed: 1))->fit($scaled);
echo 'Cluster of each point: ', json_encode($kmeans->getLabels()), "\n\n";

// --- Persisting a trained model --------------------------------------------
$file = sys_get_temp_dir() . '/regression.model';
file_put_contents($file, serialize($regression));
$restored = unserialize(file_get_contents($file));
printf("Restored model predicts %.1f\n\n", $restored->predict([90, 3]));
unlink($file);

// --- Errors are typed exceptions -------------------------------------------
try {
    $regression->predict([90]);
} catch (MLException $e) {
    echo get_class($e), ': ', $e->getMessage(), "\n";
}
