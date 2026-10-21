--TEST--
Decision trees, Extra Trees and Random Forests
--FILE--
<?php
require __DIR__ . '/helpers.inc';

use ML\Classification\{ClassificationTree, ExtraTreeClassifier, RandomForest};
use ML\Regression\{RegressionTree, ExtraTreeRegressor, RandomForestRegressor};
use ML\Datasets;
use ML\Metrics;

$d = blobs(40, 1, 0.8);
[$Xt, $Xs, $yt, $ys] = Datasets::trainTestSplit($d['samples'], $d['labels'], 0.3, true, 1);

foreach ([new ClassificationTree(seed: 1), new ExtraTreeClassifier(seed: 1), new RandomForest(30, seed: 1),
          new RandomForest(30, extraTrees: true, balanced: true, seed: 1)] as $m) {
    $m->fit($Xt, $yt);
    $name = (new ReflectionClass($m))->getShortName();
    check("$name accuracy", accuracy($m, $Xs, $ys) >= 0.9);
    check("$name importances sum to 1", abs(array_sum($m->getFeatureImportances()) - 1.0) < 1e-9);
    $p = $m->predictProbability($Xs[0]);
    check("$name probabilities", abs(array_sum($p) - 1.0) < 1e-9 && array_keys($p) === ['a', 'b', 'c']);
    check("$name serialize", same_after_serialize($m, $Xs));
}

echo "--- tree structure\n";
$stump = (new ClassificationTree(maxHeight: 1))->fit([[1], [2], [3], [10], [11], [12]], ['x', 'x', 'x', 'y', 'y', 'y']);
var_dump($stump->getHeight(), $stump->getLeafCount());
echo $stump->rules(['size']);
check('maxLeafSize stops splitting', (new ClassificationTree(maxLeafSize: 100))->fit($Xt, $yt)->getLeafCount() === 1);

echo "--- regression\n";
$h = Datasets::hyperplane([3, -2, 0.5], 1.0, 0.05, 300, 1);
[$Ht, $Hs, $ht, $hs] = Datasets::trainTestSplit($h['samples'], $h['targets'], 0.3, false, 1);
foreach ([new RegressionTree(seed: 1), new ExtraTreeRegressor(seed: 1), new RandomForestRegressor(30, seed: 1)] as $m) {
    $m->fit($Ht, $ht);
    $name = (new ReflectionClass($m))->getShortName();
    check("$name r2", Metrics::r2Score($hs, $m->predict($Hs)) > 0.8);
    check("$name single sample", is_float($m->predict($Hs[0])));
    check("$name serialize", same_after_serialize($m, $Hs));
}
echo (new RegressionTree(maxHeight: 1))->fit([[1], [2], [10], [11]], [1.0, 1.0, 5.0, 7.0])->rules();

echo "--- seeds, clone and threads\n";
$a = (new RandomForest(20, seed: 7))->fit($Xt, $yt)->predictProbability($Xs);
$b = (new RandomForest(20, seed: 7))->fit($Xt, $yt)->predictProbability($Xs);
check('same seed, same forest', $a === $b);
ini_set('mlnative.threads', '4');
$c = (new RandomForest(20, seed: 7))->fit($Xt, $yt)->predictProbability($Xs);
ini_set('mlnative.threads', '1');
check('threads do not change results', $a === $c);
$f = (new RandomForest(10, seed: 1))->fit($Xt, $yt);
$g = clone $f;
$g->fit([[0], [1]], ['p', 'q']);
check('clone is independent', $f->getClasses() === ['a', 'b', 'c'] && $g->getClasses() === ['p', 'q']);

echo "--- errors\n";
expect_error(fn () => new ClassificationTree(maxHeight: 0));
expect_error(fn () => new RandomForest(ratio: 1.5));
expect_error(fn () => (new RandomForest())->predict([1, 2]));
expect_error(fn () => $f->predict([1, 2, 3]));
$data = $f->__serialize();
$data['state'] = substr($data['state'], 0, -10);
expect_error(fn () => (new RandomForest())->__unserialize($data));
$tree = (new ClassificationTree())->__serialize();
expect_error(fn () => (new RandomForest())->__unserialize($tree));
?>
--EXPECT--
ClassificationTree accuracy: ok
ClassificationTree importances sum to 1: ok
ClassificationTree probabilities: ok
ClassificationTree serialize: ok
ExtraTreeClassifier accuracy: ok
ExtraTreeClassifier importances sum to 1: ok
ExtraTreeClassifier probabilities: ok
ExtraTreeClassifier serialize: ok
RandomForest accuracy: ok
RandomForest importances sum to 1: ok
RandomForest probabilities: ok
RandomForest serialize: ok
RandomForest accuracy: ok
RandomForest importances sum to 1: ok
RandomForest probabilities: ok
RandomForest serialize: ok
--- tree structure
int(1)
int(2)
if size < 6.5:
    => x (p=1)
else:  # size >= 6.5
    => y (p=1)
maxLeafSize stops splitting: ok
--- regression
RegressionTree r2: ok
RegressionTree single sample: ok
RegressionTree serialize: ok
ExtraTreeRegressor r2: ok
ExtraTreeRegressor single sample: ok
ExtraTreeRegressor serialize: ok
RandomForestRegressor r2: ok
RandomForestRegressor single sample: ok
RandomForestRegressor serialize: ok
if feature 0 < 6:
    => 1
else:  # feature 0 >= 6
    => 6
--- seeds, clone and threads
same seed, same forest: ok
threads do not change results: ok
clone is independent: ok
--- errors
maxHeight must be at least 1
ratio must be greater than 0 and at most 1
ML\Classification\RandomForest is not fitted yet; call fit() first
Sample has 3 features, expected 2
Invalid serialized data: truncated
Invalid serialized data: payload belongs to a different class
