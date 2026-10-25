--TEST--
OneVsRest, BootstrapAggregator, CommitteeMachine and Pipeline (with native and user-defined estimators)
--FILE--
<?php
require __DIR__ . '/helpers.inc';

use ML\Ensemble\{OneVsRest, BootstrapAggregator, CommitteeMachine};
use ML\Pipeline;
use ML\Classification\{LogisticRegression, LogitBoost, ClassificationTree, KNearestNeighbors, GaussianNB};
use ML\Regression\{LinearRegression, RegressionTree};
use ML\Preprocessing\{StandardScaler, LinearDiscriminantAnalysis};
use ML\AnomalyDetection\IsolationForest;
use ML\Datasets;
use ML\Metrics;

// A user-written PHP classifier: always predicts the most frequent label.
final class Majority implements ML\Learner, ML\Classifier
{
    private $label = null;
    public function fit(array $samples, array $labels): static
    {
        $counts = array_count_values($labels);
        arsort($counts);
        $this->label = array_key_first($counts);
        return $this;
    }
    public function predict(array $samples): array
    {
        return array_fill(0, count($samples), $this->label);
    }
}

$d = blobs(40, 13, 0.8);
[$Xt, $Xs, $yt, $ys] = Datasets::trainTestSplit($d['samples'], $d['labels'], 0.3, true, 1);

$ovr = (new OneVsRest(new LogitBoost(seed: 1)))->fit($Xt, $yt);
check('OneVsRest(LogitBoost)', accuracy($ovr, $Xs, $ys) > 0.9);
echo json_encode($ovr->getClasses()), "\n";
$p = $ovr->predictProbability($Xs[0]);
check('OneVsRest probabilities', abs(array_sum($p) - 1) < 1e-9 && array_keys($p) === ['a', 'b', 'c']);
check('OneVsRest single sample', in_array($ovr->predict($Xs[0]), ['a', 'b', 'c'], true));

$bag = (new BootstrapAggregator(new ClassificationTree(), 15, 0.8, 1))->fit($Xt, $yt);
check('Bagging classifier', accuracy($bag, $Xs, $ys) > 0.9);
$h = Datasets::hyperplane([1, -1], 0.0, 0.05, 200, 1);
$bagr = (new BootstrapAggregator(new RegressionTree(), 10, 1.0, 1))->fit($h['samples'], $h['targets']);
check('Bagging regressor', Metrics::r2Score($h['targets'], $bagr->predict($h['samples'])) > 0.9);
check('Bagging anomaly detector', count((new BootstrapAggregator(new IsolationForest(20, seed: 1), 3, 0.5, 1))->fit($Xt)->predict($Xs)) === count($Xs));

$committee = (new CommitteeMachine([new KNearestNeighbors(5), new GaussianNB(), new Majority()], [3, 3, 1]))->fit($Xt, $yt);
check('Committee with a PHP expert', accuracy($committee, $Xs, $ys) > 0.9);
echo json_encode(rounded($committee->getInfluences())), "\n";
$cr = (new CommitteeMachine([new LinearRegression(), new RegressionTree()]))->fit($h['samples'], $h['targets']);
check('Committee regressor', Metrics::r2Score($h['targets'], $cr->predict($h['samples'])) > 0.95);

$pipe = (new Pipeline([new StandardScaler(), new LinearDiscriminantAnalysis(2)], new LogisticRegression()))->fit($Xt, $yt);
check('Pipeline with a supervised transformer', accuracy($pipe, $Xs, $ys) > 0.9);
check('Pipeline transform', count($pipe->transform($Xs)[0]) === 2);
check('Pipeline probabilities', abs(array_sum($pipe->predictProbability($Xs[0])) - 1) < 1e-9);
check('Pipeline accessors', $pipe->getEstimator() instanceof LogisticRegression && count($pipe->getTransformers()) === 2);

echo "--- clone and serialize\n";
$copy = clone $pipe;
$copy->fit([[0, 0], [1, 1], [5, 5], [6, 6], [0, 9], [1, 9]], ['x', 'x', 'y', 'y', 'z', 'z']);
check('cloned pipeline is independent', $pipe->predict($Xs) !== $copy->predict($Xs) && $pipe->getEstimator()->getClasses() === ['a', 'b', 'c']);
foreach ([$ovr, $bag, $pipe] as $model) {
    check(get_class($model), same_after_serialize($model, $Xs));
}
check('validators see through the pipeline', (new ML\CrossValidation\KFold(3, true, 1))->test(new Pipeline([new StandardScaler()], new GaussianNB()), $Xt, $yt) > 0.9);

echo "--- errors\n";
expect_error(fn () => new CommitteeMachine([new LinearRegression(), new GaussianNB()]));
expect_error(fn () => new CommitteeMachine([new GaussianNB()], [1, 2]));
expect_error(fn () => (new CommitteeMachine([new GaussianNB()]))->predict([1, 2]));
expect_error(fn () => new Pipeline([new GaussianNB()], new GaussianNB()));
expect_error(fn () => (new Pipeline([], new ML\Clustering\DBSCAN()))->predictProbability([[1]]));
expect_error(fn () => (new OneVsRest(new GaussianNB()))->fit([[1], [2]], ['a', 'a']));
?>
--EXPECT--
OneVsRest(LogitBoost): ok
["a","b","c"]
OneVsRest probabilities: ok
OneVsRest single sample: ok
Bagging classifier: ok
Bagging regressor: ok
Bagging anomaly detector: ok
Committee with a PHP expert: ok
[0.429,0.429,0.143]
Committee regressor: ok
Pipeline with a supervised transformer: ok
Pipeline transform: ok
Pipeline probabilities: ok
Pipeline accessors: ok
--- clone and serialize
cloned pipeline is independent: ok
ML\Ensemble\OneVsRest: ok
ML\Ensemble\BootstrapAggregator: ok
ML\Pipeline: ok
validators see through the pipeline: ok
--- errors
All experts must be of the same kind (all classifiers, all regressors or all anomaly detectors)
influences must contain one weight per expert
ML\Ensemble\CommitteeMachine is not fitted yet; call fit() first
Each transformer must implement ML\Transformer, ML\Classification\GaussianNB given
The pipeline's estimator must implement ML\Probabilistic, ML\Clustering\DBSCAN given
OneVsRest needs at least two distinct classes
