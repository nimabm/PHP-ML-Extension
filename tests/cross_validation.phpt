--TEST--
ML\CrossValidation: reports and validators
--FILE--
<?php
require __DIR__ . '/helpers.inc';

use ML\CrossValidation\{Reports, HoldOut, KFold, LeavePOut, MonteCarlo};
use ML\Classification\{KNearestNeighbors, GaussianNB};
use ML\Regression\LinearRegression;
use ML\Datasets;
use ML\Metrics;

$e = ['a', 'b', 'a', 'c', 'b', 'a'];
$p = ['a', 'b', 'b', 'c', 'a', 'a'];
echo json_encode(Reports::confusionMatrix($e, $p)), "\n";
$b = Reports::multiclassBreakdown($e, $p);
echo json_encode(rounded($b['overall'], 4)), "\n";
echo json_encode(rounded($b['classes']['a'], 4)), "\n";
echo json_encode(rounded(Reports::errorAnalysis([3, -0.5, 2, 7], [2.5, 0.0, 2, 8]), 4)), "\n";
echo json_encode(Reports::contingencyTable([0, 0, 0, 1, 1, 1], [1, 1, 0, 0, 2, 2])), "\n";

echo "--- validators\n";
$d = blobs(20, 5, 0.6);
$X = $d['samples'];
$y = $d['labels'];
$knn = new KNearestNeighbors(3);
check('KFold has k scores', count((new KFold(4, false, 1))->scores($knn, $X, $y)) === 4);
check('stratified KFold', (new KFold(3, true, 1))->test($knn, $X, $y) > 0.95);
check('HoldOut', (new HoldOut(0.25, true, 1))->test(new GaussianNB(), $X, $y) > 0.95);
check('LeavePOut folds', count((new LeavePOut(20))->scores($knn, $X, $y)) === 3);
check('MonteCarlo simulations', count((new MonteCarlo(5, 0.2, false, 1))->scores($knn, $X, $y)) === 5);
check('estimator untouched', (function () use ($knn) {
    try {
        $knn->predict([0, 0]);
        return false;
    } catch (ML\Exception\NotFittedException $e) {
        return true;
    }
})());
check('f1 metric', (new KFold(3, true, 1))->test($knn, $X, $y, 'f1') > 0.95);
check('callable metric', (new KFold(3, true, 1))->test($knn, $X, $y, fn ($a, $b) => Metrics::accuracy($a, $b)) > 0.95);
$h = Datasets::hyperplane([1, 2], 0.0, 0.01, 60, 1);
check('regressor defaults to r2', (new KFold(3, false, 1))->test(new LinearRegression(), $h['samples'], $h['targets']) > 0.99);
check('mse is an error', (new KFold(3, false, 1))->test(new LinearRegression(), $h['samples'], $h['targets'], 'mse') < 0.01);
$kf = unserialize(serialize(new KFold(5, true, 9)));
check('validator serializes', $kf instanceof ML\CrossValidation\Validator);

echo "--- errors\n";
expect_error(fn () => (new KFold())->test($knn, $X, $y, 'bogus'));
expect_error(fn () => new KFold(1));
expect_error(fn () => new HoldOut(1.0));
expect_error(fn () => (new KFold(3))->test($knn, $X, array_slice($y, 1)));
expect_error(fn () => (new KFold(3))->test(new ML\Clustering\DBSCAN(), $X, $y, 'accuracy'));
?>
--EXPECT--
{"a":{"a":2,"b":1,"c":0},"b":{"a":1,"b":1,"c":0},"c":{"a":0,"b":0,"c":1}}
{"accuracy":0.6667,"precision":0.7222,"recall":0.7222,"f1Score":0.7222,"mcc":0.4545,"informedness":0.5278,"cardinality":6,"classes":3}
{"precision":0.6667,"recall":0.6667,"specificity":0.6667,"f1Score":0.6667,"informedness":0.3333,"truePositives":2,"falsePositives":1,"falseNegatives":1,"trueNegatives":2,"support":3,"proportion":0.5}
{"meanAbsoluteError":0.5,"medianAbsoluteError":0.5,"meanSquaredError":0.375,"rootMeanSquaredError":0.6124,"r2":0.9486,"smape":28.9394,"errorMean":-0.25,"errorVariance":0.3125,"errorStdDev":0.559,"errorMin":-1,"errorMax":0.5,"cardinality":4}
[[1,1],[2,0],[0,2]]
--- validators
KFold has k scores: ok
stratified KFold: ok
HoldOut: ok
LeavePOut folds: ok
MonteCarlo simulations: ok
estimator untouched: ok
f1 metric: ok
callable metric: ok
regressor defaults to r2: ok
mse is an error: ok
validator serializes: ok
--- errors
Unknown metric "bogus"; expected accuracy, precision, recall, f1, mcc, informedness, r2, mse, mae, rmse, median_absolute_error, smape or a callable
k must be at least 2
ratio must be between 0 and 1 (exclusive)
Number of samples (60) does not match number of targets (59)
estimator must implement ML\Learner, ML\Clustering\DBSCAN given
