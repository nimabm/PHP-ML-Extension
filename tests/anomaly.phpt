--TEST--
Anomaly detection: RobustZScore, GaussianMLE, IsolationForest, LocalOutlierFactor, Loda, OneClassSVM
--FILE--
<?php
require __DIR__ . '/helpers.inc';

use ML\AnomalyDetection\{RobustZScore, GaussianMLE, IsolationForest, LocalOutlierFactor, Loda, OneClassSVM};
use ML\Datasets;

$normal = Datasets::blob([0, 0], 1.0, 200, 1);
$outliers = [[8, 8], [-7, 6], [9, -9]];
$all = array_merge($normal, $outliers);

$detectors = [
    new RobustZScore(),
    new GaussianMLE(0.02),
    new IsolationForest(seed: 1, contamination: 0.02),
    new LocalOutlierFactor(10, 0.02),
    new Loda(seed: 1, contamination: 0.02),
    new OneClassSVM(0.05, 'rbf', 0.1),
];
foreach ($detectors as $d) {
    $d->fit($all);
    $flags = $d->predict($all);
    $name = (new ReflectionClass($d))->getShortName();
    check("$name catches the outliers", array_sum(array_slice($flags, -3)) === 3);
    check("$name few false alarms", array_sum(array_slice($flags, 0, 200)) <= 12);
    $scores = $d->score([[0, 0], [8, 8]]);
    check("$name scores outliers higher", $scores[1] > $scores[0]);
    check("$name single sample", in_array($d->predict([8, 8]), [0, 1], true) && is_float($d->score([8, 8])));
    check("$name serialize", same_after_serialize($d, $all) && unserialize(serialize($d))->getThreshold() === $d->getThreshold());
}

$forest = (new IsolationForest(seed: 1))->fit($normal);
check('isolation scores are in (0, 1)', min($forest->score($all)) > 0 && max($forest->score($all)) < 1);
var_dump($forest->getThreshold());
var_dump((new RobustZScore(2.0))->fit($normal)->getThreshold());
check('OneClassSVM support vectors', (new OneClassSVM(0.1))->fit($normal)->getSupportVectorCount() >= 20);

echo "--- errors\n";
expect_error(fn () => new GaussianMLE(0.5));
expect_error(fn () => new IsolationForest(maxSamples: 1));
expect_error(fn () => (new LocalOutlierFactor(20))->fit([[1], [2], [3]]));
expect_error(fn () => new RobustZScore(beta: 2));
expect_error(fn () => new OneClassSVM(0));
expect_error(fn () => (new Loda())->score([1, 2]));
?>
--EXPECT--
RobustZScore catches the outliers: ok
RobustZScore few false alarms: ok
RobustZScore scores outliers higher: ok
RobustZScore single sample: ok
RobustZScore serialize: ok
GaussianMLE catches the outliers: ok
GaussianMLE few false alarms: ok
GaussianMLE scores outliers higher: ok
GaussianMLE single sample: ok
GaussianMLE serialize: ok
IsolationForest catches the outliers: ok
IsolationForest few false alarms: ok
IsolationForest scores outliers higher: ok
IsolationForest single sample: ok
IsolationForest serialize: ok
LocalOutlierFactor catches the outliers: ok
LocalOutlierFactor few false alarms: ok
LocalOutlierFactor scores outliers higher: ok
LocalOutlierFactor single sample: ok
LocalOutlierFactor serialize: ok
Loda catches the outliers: ok
Loda few false alarms: ok
Loda scores outliers higher: ok
Loda single sample: ok
Loda serialize: ok
OneClassSVM catches the outliers: ok
OneClassSVM few false alarms: ok
OneClassSVM scores outliers higher: ok
OneClassSVM single sample: ok
OneClassSVM serialize: ok
isolation scores are in (0, 1): ok
float(0.5)
float(2)
OneClassSVM support vectors: ok
--- errors
contamination must be at least 0 and less than 0.5
maxSamples must be at least 2
k (20) must be less than the number of training samples (3)
beta must be between 0 and 1
nu must be greater than 0 and at most 1
ML\AnomalyDetection\Loda is not fitted yet; call fit() first
