--TEST--
MatrixFactorization, ImplicitALS and KNNRecommender, ranking metrics and synthetic ratings
--FILE--
<?php
require __DIR__ . '/helpers.inc';

use ML\Recommender\{MatrixFactorization, ImplicitALS, KNNRecommender, Recommender};
use ML\Datasets;
use ML\Metrics;

// Two kinds of users and items: even users love even items, odd users odd ones.
mt_srand(11);
$ratings = [];
for ($u = 0; $u < 120; $u++) {
    for ($i = 0; $i < 40; $i++) {
        if (mt_rand(0, 99) < 30) {
            $ratings[] = ["u$u", "i$i", $u % 2 === $i % 2 ? mt_rand(4, 5) : mt_rand(1, 2)];
        }
    }
}
shuffle($ratings);
$test = array_slice($ratings, 0, 200);
$train = array_slice($ratings, 200);
$pairs = array_map(fn ($r) => [$r[0], $r[1]], $test);
$truth = array_column($test, 2);
$matches = function (array $items, int $user): int {
    return count(array_filter(array_keys($items), fn ($item) => (int) substr($item, 1) % 2 === $user % 2));
};

echo "--- MatrixFactorization\n";
$mf = (new MatrixFactorization(4, seed: 1))->fit($train);
check('implements Recommender', $mf instanceof Recommender);
check('accurate ratings', Metrics::rootMeanSquaredError($truth, $mf->predict($pairs)) < 0.8);
check('single prediction', is_float($mf->predict(['u0', 'i0'])));
$top = $mf->recommend('u0', 5);
check('recommends items of the right kind', $matches($top, 0) === 5);
check('scores are sorted', array_values($top) === array_values((function ($s) { arsort($s); return $s; })($top)));
$seen = array_column(array_filter($train, fn ($r) => $r[0] === 'u0'), 1);
check('excludes rated items', array_intersect(array_keys($top), $seen) === []);
check('can include rated items', count(array_intersect(array_keys($mf->recommend('u0', 40, false)), $seen)) === count($seen));
check('similar items are of the same kind', $matches($mf->similarItems('i0', 5), 0) === 5);
$losses = $mf->getLosses();
check('losses decrease', $losses[0] > end($losses));
check('factors keyed by label', count($mf->getUserFactors()['u0']) === 4 && count($mf->getItemFactors()) === 40);
check('users and items', count($mf->getUsers()) === 120 && $mf->getItems()[0] === 'i0');
check('cold start user gets popular items', count($mf->recommend('stranger', 3)) === 3);
check('cold start prediction stays in range', ($v = $mf->predict(['stranger', 'nothing'])) >= 1 && $v <= 5);
check('seeded runs repeat', (new MatrixFactorization(4, seed: 1))->fit($train)->predict($pairs) === $mf->predict($pairs));
ini_set('mlnative.threads', 4);
check('threads do not change the result', (new MatrixFactorization(4, seed: 1))->fit($train)->predict($pairs) === $mf->predict($pairs));
ini_set('mlnative.threads', 1);

echo "--- KNNRecommender\n";
$variants = [
    'cosine/item' => new KNNRecommender(20),
    'pearson/item' => new KNNRecommender(20, 'pearson'),
    'cosine/user' => new KNNRecommender(20, 'cosine', 'user', 5.0),
];
foreach ($variants as $name => $knn) {
    $knn->fit($train);
    check("$name RMSE", Metrics::rootMeanSquaredError($truth, $knn->predict($pairs)) < 1.2);
    check("$name recommendations", $matches($knn->recommend('u1', 5), 1) >= 4);
    check("$name similar items", count($knn->similarItems('i3', 3)) === 3);
}

echo "--- ImplicitALS\n";
// Only the fact that a user liked an item (no rating value).
$liked = array_values(array_map(fn ($r) => [$r[0], $r[1]], array_filter($train, fn ($r) => $r[2] >= 4)));
$als = (new ImplicitALS(4, seed: 1))->fit($liked);
check('implicit recommendations of the right kind', $matches($als->recommend('u0', 10), 0) === 10);
check('similar items', $matches($als->similarItems('i1', 5), 1) === 5);
check('preference scores', $als->predict(['u0', 'i0']) > $als->predict(['u0', 'i1']));
$knnImplicit = (new KNNRecommender(10))->fit($liked);
check('neighborhood model on implicit feedback', $matches($knnImplicit->recommend('u0', 10), 0) >= 9);
check('jaccard user neighbors', $matches((new KNNRecommender(10, 'jaccard', 'user'))->fit($liked)->recommend('u1', 10), 1) >= 9);
$counts = (new ImplicitALS(2, seed: 1))->fit([['a', 'x', 3], ['a', 'x', 2], ['b', 'y']]);
check('repeated interactions add up', count($counts->getItems()) === 2);

echo "--- ranking metrics\n";
printf("%.4f %.4f %.4f\n", Metrics::precisionAtK(['a', 'b', 'c'], ['c', 'a'], 2), Metrics::recallAtK(['a', 'b', 'c'], ['c', 'a'], 2), Metrics::ndcgAtK(['a', 'b', 'c'], ['c', 'a'], 3));
printf("%.4f\n", Metrics::precisionAtK([['a', 'b'], ['x', 'y'], ['p']], [['a'], ['y'], []], 2));
$relevant = [];
foreach ($test as [$u, $i, $r]) {
    if ($r >= 4) {
        $relevant[$u][] = $i;
    }
}
$lists = $truths = [];
foreach ($relevant as $u => $items) {
    $lists[] = array_keys($mf->recommend($u, 10));
    $truths[] = $items;
}
check('MF beats chance at recall@10', Metrics::recallAtK($lists, $truths, 10) > 0.3);

echo "--- Datasets::ratings\n";
$synthetic = Datasets::ratings(50, 20, 0.3, seed: 2);
check('triples', count($synthetic[0]) === 3 && min(array_column($synthetic, 2)) >= 1 && max(array_column($synthetic, 2)) <= 5);
check('density', abs(count($synthetic) / 1000 - 0.3) < 0.05);
check('seeded', Datasets::ratings(50, 20, 0.3, seed: 2) === $synthetic);

echo "--- serialization\n";
foreach ([$mf, $als, $knnImplicit, (new KNNRecommender(5, basis: 'user'))->fit($train)] as $model) {
    $copy = unserialize(serialize($model));
    check(get_class($model), $copy->predict($pairs) === $model->predict($pairs) && $copy->recommend('u3') === $model->recommend('u3'));
}

echo "--- errors\n";
expect_error(fn () => (new MatrixFactorization())->fit([]));
expect_error(fn () => (new MatrixFactorization())->fit([['u', 'i', 'five']]));
expect_error(fn () => (new MatrixFactorization())->fit([['u']]));
expect_error(fn () => (new ImplicitALS())->fit([['u', 'i', -1]]));
expect_error(fn () => $mf->similarItems('missing'));
expect_error(fn () => $mf->predict([['u0']]));
expect_error(fn () => (new KNNRecommender())->recommend('u0'));
expect_error(fn () => new KNNRecommender(basis: 'both'));
expect_error(fn () => new KNNRecommender(similarity: 'euclidean'));
?>
--EXPECT--
--- MatrixFactorization
implements Recommender: ok
accurate ratings: ok
single prediction: ok
recommends items of the right kind: ok
scores are sorted: ok
excludes rated items: ok
can include rated items: ok
similar items are of the same kind: ok
losses decrease: ok
factors keyed by label: ok
users and items: ok
cold start user gets popular items: ok
cold start prediction stays in range: ok
seeded runs repeat: ok
threads do not change the result: ok
--- KNNRecommender
cosine/item RMSE: ok
cosine/item recommendations: ok
cosine/item similar items: ok
pearson/item RMSE: ok
pearson/item recommendations: ok
pearson/item similar items: ok
cosine/user RMSE: ok
cosine/user recommendations: ok
cosine/user similar items: ok
--- ImplicitALS
implicit recommendations of the right kind: ok
similar items: ok
preference scores: ok
neighborhood model on implicit feedback: ok
jaccard user neighbors: ok
repeated interactions add up: ok
--- ranking metrics
0.5000 0.5000 0.9197
0.5000
MF beats chance at recall@10: ok
--- Datasets::ratings
triples: ok
density: ok
seeded: ok
--- serialization
ML\Recommender\MatrixFactorization: ok
ML\Recommender\ImplicitALS: ok
ML\Recommender\KNNRecommender: ok
ML\Recommender\KNNRecommender: ok
--- errors
Interactions must not be empty
Interaction #0: the value must be a finite int or float
Interaction #0 must have 2 or 3 elements, 1 given
ImplicitALS needs non-negative interaction values
Unknown item "missing"
Pair #0 must be [user, item]
ML\Recommender\KNNRecommender is not fitted yet; call fit() first
basis must be "item" or "user"
Unknown similarity "euclidean" (expected cosine, pearson or jaccard)
