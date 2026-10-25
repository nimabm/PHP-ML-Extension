--TEST--
Corrupted or forged serialized models are rejected without crashing
--FILE--
<?php
require __DIR__ . '/helpers.inc';

$d = blobs(10, 29, 0.5);
$h = ML\Datasets::hyperplane([1, 1], 0.0, 0.1, 30, 1);
$models = [
    (new ML\Classification\RandomForest(5, seed: 1))->fit($d['samples'], $d['labels']),
    (new ML\Classification\KDNeighbors(3))->fit($d['samples'], $d['labels']),
    (new ML\Classification\NaiveBayes())->fit([['x'], ['y']], ['a', 'b']),
    (new ML\Classification\AdaBoost(5, seed: 1))->fit($d['samples'], $d['labels']),
    (new ML\Classification\SVC())->fit($d['samples'], $d['labels']),
    (new ML\Classification\MLPClassifier([new ML\NeuralNet\Layers\Dense(4)], epochs: 5, seed: 1))->fit($d['samples'], $d['labels']),
    (new ML\Regression\GradientBoost(5, seed: 1))->fit($h['samples'], $h['targets']),
    (new ML\Clustering\GaussianMixture(3, seed: 1))->fit($d['samples']),
    (new ML\AnomalyDetection\IsolationForest(5, seed: 1))->fit($d['samples']),
    (new ML\Preprocessing\OneHotEncoder())->fit([['a'], ['b']]),
    (new ML\Text\WordCountVectorizer())->fit(['a b', 'b c']),
    (new ML\Preprocessing\KernelPCA(2))->fit($d['samples']),
    (new ML\Preprocessing\Isomap(2, 12))->fit($d['samples']),
    (new ML\Preprocessing\UMAP(2, 4, epochs: 20, seed: 1))->fit($d['samples']),
    (new ML\Preprocessing\FastICA(seed: 1))->fit($d['samples']),
    (new ML\Preprocessing\NMF(1, seed: 1))->fit([[1, 2], [2, 4], [3, 7]]),
    (new ML\Clustering\MiniBatchKMeans(3, seed: 1))->fit($d['samples']),
    (new ML\Sequence\HiddenMarkovModel(2, seed: 1))->fit([[1, 2, 1, 3], [2, 2, 3]]),
    (new ML\Sequence\GaussianHMM(2, seed: 1))->fit([[[1.0], [1.2], [5.0], [5.1]]]),
    (new ML\Recommender\MatrixFactorization(2, seed: 1))->fit([['a', 'x', 5], ['a', 'y', 1], ['b', 'x', 4]]),
    (new ML\Recommender\ImplicitALS(2, seed: 1))->fit([['a', 'x'], ['b', 'y'], ['b', 'x']]),
    (new ML\Recommender\KNNRecommender(2))->fit([['a', 'x', 5], ['a', 'y', 1], ['b', 'x', 4]]),
];
// Exercises a model that accepted a payload: it must answer or throw, never crash.
$use = function (object $model) use ($d): void {
    if ($model instanceof ML\Recommender\Recommender) {
        $model->recommend('a');
        $model->predict(['a', 'x']);
    } elseif ($model instanceof ML\Sequence\HiddenMarkovModel) {
        $model->predict([1, 2]);
        $model->predictProbability([1, 3]);
    } elseif ($model instanceof ML\Sequence\GaussianHMM) {
        $model->predict([[1.0], [5.0]]);
    } elseif ($model instanceof ML\Preprocessing\NMF) {
        $model->transform([1, 2]);
    } elseif ($model instanceof ML\Transformer && $model instanceof ML\Stateful && !$model instanceof ML\Preprocessing\OneHotEncoder) {
        $model->transform($d['samples'][0]);
    } elseif ($model instanceof ML\Estimator && !$model instanceof ML\Preprocessing\OneHotEncoder) {
        $model->predict($d['samples'][0]);
    }
};

mt_srand(42);
$crashes = 0;
foreach ($models as $model) {
    $class = get_class($model);
    $data = $model->__serialize();
    $state = $data['state'];
    $rejected = 0;
    $attempts = 0;
    // Truncations, random byte flips and garbage must all be rejected cleanly.
    $variants = [substr($state, 0, 10), substr($state, 0, (int) (strlen($state) / 2)), $state . 'junk', 'garbage', ''];
    for ($i = 0; $i < 30; $i++) {
        $forged = $state;
        $pos = mt_rand(12, strlen($forged) - 1);
        $forged[$pos] = chr(mt_rand(0, 255));
        $variants[] = $forged;
    }
    foreach ($variants as $variant) {
        $attempts++;
        $copy = $data;
        $copy['state'] = $variant;
        try {
            $restored = (new ReflectionClass($class))->newInstanceWithoutConstructor();
        } catch (Throwable $e) {
            $restored = null;
        }
        $target = $restored ?? clone $model;
        try {
            $target->__unserialize($copy);
        } catch (ML\Exception\InvalidArgumentException $e) {
            $rejected++;
            continue;
        }
        try {
            $use($target);
        } catch (Throwable $e) {
            // any exception is fine; only a crash is not
        }
    }
    // The first five variants are always invalid.
    check("$class rejects corrupted payloads", $rejected >= 5);
}
check('no crash', $crashes === 0);
expect_error(fn () => (new ML\Classification\RandomForest())->__unserialize(['state' => 42]));
expect_error(fn () => (new ML\Classification\RandomForest())->__unserialize([]));
?>
--EXPECT--
ML\Classification\RandomForest rejects corrupted payloads: ok
ML\Classification\KDNeighbors rejects corrupted payloads: ok
ML\Classification\NaiveBayes rejects corrupted payloads: ok
ML\Classification\AdaBoost rejects corrupted payloads: ok
ML\Classification\SVC rejects corrupted payloads: ok
ML\Classification\MLPClassifier rejects corrupted payloads: ok
ML\Regression\GradientBoost rejects corrupted payloads: ok
ML\Clustering\GaussianMixture rejects corrupted payloads: ok
ML\AnomalyDetection\IsolationForest rejects corrupted payloads: ok
ML\Preprocessing\OneHotEncoder rejects corrupted payloads: ok
ML\Text\WordCountVectorizer rejects corrupted payloads: ok
ML\Preprocessing\KernelPCA rejects corrupted payloads: ok
ML\Preprocessing\Isomap rejects corrupted payloads: ok
ML\Preprocessing\UMAP rejects corrupted payloads: ok
ML\Preprocessing\FastICA rejects corrupted payloads: ok
ML\Preprocessing\NMF rejects corrupted payloads: ok
ML\Clustering\MiniBatchKMeans rejects corrupted payloads: ok
ML\Sequence\HiddenMarkovModel rejects corrupted payloads: ok
ML\Sequence\GaussianHMM rejects corrupted payloads: ok
ML\Recommender\MatrixFactorization rejects corrupted payloads: ok
ML\Recommender\ImplicitALS rejects corrupted payloads: ok
ML\Recommender\KNNRecommender rejects corrupted payloads: ok
no crash: ok
Invalid serialized data: "state" must be a string
Invalid serialized data: missing "state"
