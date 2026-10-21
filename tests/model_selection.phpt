--TEST--
GridSearch and BayesianSearch
--FILE--
<?php
require __DIR__ . '/helpers.inc';

use ML\ModelSelection\{GridSearch, BayesianSearch};
use ML\CrossValidation\{KFold, HoldOut};
use ML\Classification\KNearestNeighbors;
use ML\Regression\{ElasticNet, GradientBoost};
use ML\Datasets;
use ML\Metrics;

$d = blobs(30, 17, 1.0);
$grid = (new GridSearch(KNearestNeighbors::class, ['k' => [1, 3, 9], 'weighted' => [false, true]], new KFold(3, true, 1)))
    ->fit($d['samples'], $d['labels']);
check('six combinations', count($grid->getResults()) === 6);
$best = $grid->getBest();
echo json_encode(array_keys($best)), ' ', json_encode(array_keys($best['params'])), "\n";
check('best is the max', $best['score'] === max(array_column($grid->getResults(), 'score')));
check('refit on all data', $grid->getEstimator() instanceof KNearestNeighbors && count($grid->predict($d['samples'])) === 90);
check('probabilities delegate', abs(array_sum($grid->predictProbability($d['samples'][0])) - 1) < 1e-9);
check('scalar grid values', count((new GridSearch(KNearestNeighbors::class, ['k' => 3]))->fit($d['samples'], $d['labels'])->getResults()) === 1);

$h = Datasets::hyperplane([2, 0, -1], 0.5, 0.05, 120, 1);
$en = (new GridSearch(ElasticNet::class, ['alpha' => [0.001, 0.1, 10.0]], metric: 'mse'))->fit($h['samples'], $h['targets']);
echo json_encode($en->getBest()['params']), "\n";
check('lower-is-better metric picks the min', $en->getBest()['score'] === min(array_column($en->getResults(), 'score')));

$bayes = (new BayesianSearch(GradientBoost::class, [
    'rate' => ['min' => 0.01, 'max' => 0.5, 'log' => true],
    'maxHeight' => ['min' => 1, 'max' => 5],
    'ratio' => [0.5, 0.8],
    'seed' => 1,
], 6, new HoldOut(0.3, false, 1), 'r2', 1))->fit($h['samples'], $h['targets']);
$params = $bayes->getBest()['params'];
check('bayes trials', count($bayes->getResults()) === 6);
check('bayes parameter types', is_float($params['rate']) && is_int($params['maxHeight']) && in_array($params['ratio'], [0.5, 0.8], true) && $params['seed'] === 1);
check('bayes ranges respected', $params['rate'] >= 0.01 && $params['rate'] <= 0.5 && $params['maxHeight'] >= 1 && $params['maxHeight'] <= 5);
check('bayes found a good model', $bayes->getBest()['score'] > 0.8);
check('search serializes', unserialize(serialize($grid))->predict($d['samples']) === $grid->predict($d['samples']));

echo "--- errors\n";
expect_error(fn () => new GridSearch('NoSuchClass', []));
expect_error(fn () => new GridSearch(ML\Metrics::class, []));
expect_error(fn () => new GridSearch(KNearestNeighbors::class, ['k' => []]));
expect_error(fn () => (new GridSearch(KNearestNeighbors::class, ['bogus' => [1]]))->fit($d['samples'], $d['labels']));
expect_error(fn () => (new GridSearch(KNearestNeighbors::class, ['k' => [0]]))->fit($d['samples'], $d['labels']));
expect_error(fn () => new BayesianSearch(ElasticNet::class, ['alpha' => ['min' => 1, 'max' => 0]]));
expect_error(fn () => (new GridSearch(KNearestNeighbors::class, ['k' => [1]]))->predict([1, 2]));
?>
--EXPECT--
six combinations: ok
["params","score"] ["k","weighted"]
best is the max: ok
refit on all data: ok
probabilities delegate: ok
scalar grid values: ok
{"alpha":0.001}
lower-is-better metric picks the min: ok
bayes trials: ok
bayes parameter types: ok
bayes ranges respected: ok
bayes found a good model: ok
search serializes: ok
--- errors
Class "NoSuchClass" does not exist
Class "ML\Metrics" must implement ML\Learner
Every grid entry needs at least one value
Error: Unknown named parameter $bogus
k must be at least 1
Range min must be less than max
ML\ModelSelection\GridSearch is not fitted yet; call fit() first
