--TEST--
AdaBoost, LogitBoost, GradientBoost and ElasticNet
--FILE--
<?php
require __DIR__ . '/helpers.inc';

use ML\Classification\{AdaBoost, LogitBoost};
use ML\Regression\{GradientBoost, ElasticNet};
use ML\Datasets;
use ML\Metrics;

$d = blobs(60, 11, 1.2);
[$Xt, $Xs, $yt, $ys] = Datasets::trainTestSplit($d['samples'], $d['labels'], 0.3, true, 1);

$ada = (new AdaBoost(50, seed: 1))->fit($Xt, $yt);
check('AdaBoost accuracy', accuracy($ada, $Xs, $ys) > 0.85);
check('AdaBoost rounds', $ada->getEstimatorCount() >= 2 && count($ada->getLosses()) === $ada->getEstimatorCount());
check('AdaBoost probabilities', abs(array_sum($ada->predictProbability($Xs[0])) - 1) < 1e-9);
check('AdaBoost importances', abs(array_sum($ada->getFeatureImportances()) - 1) < 1e-9);
check('AdaBoost serialize', same_after_serialize($ada, $Xs));
check('deeper weak learners', accuracy((new AdaBoost(30, maxHeight: 3, seed: 1))->fit($Xt, $yt), $Xs, $ys) > 0.85);

$binary = array_map(fn ($l) => $l === 'a' ? 'yes' : 'no', $yt);
$logit = (new LogitBoost(seed: 1))->fit($Xt, $binary);
check('LogitBoost accuracy', Metrics::accuracy(array_map(fn ($l) => $l === 'a' ? 'yes' : 'no', $ys), $logit->predict($Xs)) > 0.9);
echo json_encode($logit->getClasses()), "\n";
$losses = $logit->getLosses();
check('LogitBoost loss decreases', $losses[0] > end($losses));
check('LogitBoost serialize', same_after_serialize($logit, $Xs));

$h = Datasets::hyperplane([3, -2, 0, 0, 1], 1.0, 0.1, 400, 1);
[$Ht, $Hs, $ht, $hs] = Datasets::trainTestSplit($h['samples'], $h['targets'], 0.3, false, 1);
$gb = (new GradientBoost(300, seed: 1))->fit($Ht, $ht);
check('GradientBoost r2', Metrics::r2Score($hs, $gb->predict($Hs)) > 0.9);
check('early stopping kept the best round', $gb->getEstimatorCount() <= count($gb->getValidationScores()));
check('GradientBoost without hold-out', (new GradientBoost(50, holdOut: 0.0, seed: 1))->fit($Ht, $ht)->getValidationScores() === []);
check('GradientBoost serialize', same_after_serialize($gb, $Hs));

$en = (new ElasticNet(0.05, 0.9))->fit($Ht, $ht);
$coef = $en->getCoefficients();
check('ElasticNet r2', Metrics::r2Score($hs, $en->predict($Hs)) > 0.95);
check('ElasticNet zeroes irrelevant features', $coef[2] == 0.0 && $coef[3] == 0.0 && abs($coef[0]) > 2);
check('ElasticNet without penalty matches least squares', abs((new ElasticNet(0.0, 0.5))->fit([[1], [2], [3]], [2, 4, 6])->getCoefficients()[0] - 2) < 1e-6);
check('ElasticNet serialize', same_after_serialize($en, $Hs));

echo "--- errors\n";
expect_error(fn () => (new LogitBoost())->fit($Xt, $yt));
expect_error(fn () => (new AdaBoost())->fit([[1], [2]], ['a', 'a']));
expect_error(fn () => new GradientBoost(holdOut: 0.6));
expect_error(fn () => new ElasticNet(l1Ratio: 2));
?>
--EXPECT--
AdaBoost accuracy: ok
AdaBoost rounds: ok
AdaBoost probabilities: ok
AdaBoost importances: ok
AdaBoost serialize: ok
deeper weak learners: ok
LogitBoost accuracy: ok
["no","yes"]
LogitBoost loss decreases: ok
LogitBoost serialize: ok
GradientBoost r2: ok
early stopping kept the best round: ok
GradientBoost without hold-out: ok
GradientBoost serialize: ok
ElasticNet r2: ok
ElasticNet zeroes irrelevant features: ok
ElasticNet without penalty matches least squares: ok
ElasticNet serialize: ok
--- errors
LogitBoost is a binary classifier and needs exactly two distinct classes, 3 given; wrap it in OneVsRest for more
AdaBoost needs at least two distinct classes
holdOut must be at least 0 and less than 0.5
l1Ratio must be between 0 and 1
