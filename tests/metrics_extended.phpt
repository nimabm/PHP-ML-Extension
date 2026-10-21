--TEST--
ML\Metrics: classification, ranking, probabilistic, regression and clustering metrics
--FILE--
<?php
require __DIR__ . '/helpers.inc';

use ML\Metrics;

$e = ['a', 'b', 'a', 'c', 'b', 'a'];
$p = ['a', 'b', 'b', 'c', 'a', 'a'];
printf("precision %.4f recall %.4f f1 %.4f\n", Metrics::precision($e, $p), Metrics::recall($e, $p), Metrics::f1Score($e, $p));
printf("precision(a) %.4f recall(b) %.4f f2(a) %.4f\n", Metrics::precision($e, $p, 'a'), Metrics::recall($e, $p, 'b'), Metrics::fBeta($e, $p, 2.0, 'a'));
printf("mcc %.4f informedness %.4f\n", Metrics::mcc($e, $p), Metrics::informedness($e, $p));
printf("perfect mcc %.1f\n", Metrics::mcc([1, 2, 1], [1, 2, 1]));

$labels = [1, 0, 1, 1, 0];
$scores = [0.9, 0.1, 0.8, 0.4, 0.5];
printf("roc auc %.4f average precision %.4f\n", Metrics::rocAuc($labels, $scores, 1), Metrics::averagePrecision($labels, $scores, 1));

$probs = [['a' => 0.7, 'b' => 0.3], ['a' => 0.2, 'b' => 0.8]];
printf("brier %.4f probabilistic accuracy %.4f top-1 %.2f top-2 %.2f\n", Metrics::brierScore(['a', 'b'], $probs),
    Metrics::probabilisticAccuracy(['a', 'b'], $probs), Metrics::topKAccuracy(['a', 'a'], $probs, 1), Metrics::topKAccuracy(['a', 'a'], $probs, 2));

$y = [3, -0.5, 2, 7];
$q = [2.5, 0.0, 2, 8];
printf("rmse %.4f median ae %.4f smape %.4f\n", Metrics::rootMeanSquaredError($y, $q), Metrics::medianAbsoluteError($y, $q), Metrics::smape($y, $q));

$truth = [0, 0, 0, 1, 1, 1];
$clusters = [1, 1, 0, 0, 2, 2];
printf("homogeneity %.4f completeness %.4f v %.4f ari %.4f purity %.4f\n", Metrics::homogeneity($truth, $clusters),
    Metrics::completeness($truth, $clusters), Metrics::vMeasure($truth, $clusters), Metrics::randIndex($truth, $clusters),
    Metrics::purity($truth, $clusters));
printf("perfect clustering %.1f %.1f\n", Metrics::vMeasure(['x', 'x', 'y'], [5, 5, 9]), Metrics::randIndex(['x', 'x', 'y'], [5, 5, 9]));

$grid = [];
for ($i = 0; $i < 20; $i++) {
    $grid[] = [$i, $i * 2];
}
printf("trustworthiness of a scaled copy %.1f\n", Metrics::trustworthiness($grid, array_map(fn ($r) => [$r[0] * 3, $r[1] * 3], $grid), 3));

echo "--- errors\n";
expect_error(fn () => Metrics::rocAuc([1, 1], [0.1, 0.2], 1));
expect_error(fn () => Metrics::precision(['a'], ['a'], 'z'));
expect_error(fn () => Metrics::precision([1.5], [1.5]));
expect_error(fn () => Metrics::brierScore(['a'], [0.5]));
?>
--EXPECT--
precision 0.7222 recall 0.7222 f1 0.7222
precision(a) 0.6667 recall(b) 0.5000 f2(a) 0.6667
mcc 0.4545 informedness 0.5278
perfect mcc 1.0
roc auc 0.8333 average precision 0.9167
brier 0.1300 probabilistic accuracy 0.7500 top-1 0.50 top-2 1.00
rmse 0.6124 median ae 0.5000 smape 28.9394
homogeneity 0.6667 completeness 0.4206 v 0.5158 ari 0.2424 purity 0.8333
perfect clustering 1.0 1.0
trustworthiness of a scaled copy 1.0
--- errors
ROC AUC needs both positive and negative samples
positiveClass "z" does not occur in the labels
Expected label #0 must be int or string, float given
probabilities must be a list of [label => probability] arrays
