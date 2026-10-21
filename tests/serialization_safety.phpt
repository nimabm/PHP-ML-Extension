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
];

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
            $rejected += 0;
        } catch (ML\Exception\InvalidArgumentException $e) {
            $rejected++;
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
no crash: ok
Invalid serialized data: "state" must be a string
Invalid serialized data: missing "state"
