--TEST--
Numeric and mixed-data transformers
--FILE--
<?php
require __DIR__ . '/helpers.inc';

use ML\Preprocessing\{MinMaxNormalizer, MaxAbsoluteScaler, RobustStandardizer, PowerTransformer, L1Normalizer, L2Normalizer,
    PolynomialExpander, OneHotEncoder, IntervalDiscretizer, NumericStringConverter, BooleanConverter, MissingDataImputer,
    KNNImputer, HotDeckImputer, LambdaFunction};

$X = [[1, 10, -5], [2, 20, 0], [3, 30, 5], [100, 40, 10]];
echo json_encode(rounded((new MinMaxNormalizer())->fitTransform($X))), "\n";
echo json_encode(rounded((new MinMaxNormalizer(-1, 1))->fit($X)->transform([2, 20, 0]))), "\n";
echo json_encode(rounded((new MaxAbsoluteScaler())->fitTransform($X))), "\n";
echo json_encode(rounded((new RobustStandardizer())->fitTransform($X))), "\n";
echo json_encode(rounded((new RobustStandardizer(false))->fit($X)->transform([2.5, 25, 2.5]))), "\n";

$pt = new PowerTransformer();
$T = $pt->fitTransform($X);
$column = array_column($T, 0);
check('power transform is standardized', abs(array_sum($column)) < 1e-9 && abs(array_sum(array_map(fn ($v) => $v * $v, $column)) / 4 - 1) < 1e-9);
check('power transform learned lambdas', count($pt->getLambdas()) === 3);
echo json_encode(rounded((new PowerTransformer([1, 1, 1], false))->fitTransform([[1, 2, 3]]))), "\n";

echo json_encode((new L1Normalizer())->transform([[1, 3], [0, 0]])), json_encode((new L2Normalizer())->transform([3, 4])), "\n";
echo json_encode((new PolynomialExpander(3))->transform([[2, 3]])), "\n";

echo "--- categorical and missing data\n";
$oh = new OneHotEncoder();
echo json_encode($oh->fitTransform([['red', 1.5, true], ['blue', 2, false], ['red', 3, true]])), "\n";
echo json_encode($oh->getCategories()), json_encode($oh->transform(['green', 1, true])), "\n";
echo json_encode((new OneHotEncoder([1]))->fitTransform([[1.5, 10], [2.5, 20]])), "\n";
echo json_encode((new IntervalDiscretizer(3))->fitTransform([[1, 'x'], [5, 'y'], [9, 'x'], [3, 'y']])), "\n";
echo json_encode((new IntervalDiscretizer(2, false))->fitTransform([[1], [2], [3], [100]])), "\n";
$converted = (new NumericStringConverter())->transform(['1', '2.5', 'abc', 'NAN', 7]);
var_dump($converted[0], $converted[1], $converted[2], is_nan($converted[3]), $converted[4]);
echo json_encode((new BooleanConverter())->transform([[true, false, 1]])), json_encode((new BooleanConverter(1, 0))->transform([true, false])), "\n";

echo json_encode((new MissingDataImputer('median', 'most_frequent'))->fitTransform([[1, 'a'], [null, 'b'], [3, '?'], [10, 'a'], [NAN, 'a']])), "\n";
echo json_encode((new MissingDataImputer('constant(-1)', 'constant(none)'))->fitTransform([[null, null], [2, 'x']])), "\n";
echo json_encode((new MissingDataImputer('percentile(100)'))->fitTransform([[null], [1], [9]])), "\n";
$wild = (new MissingDataImputer('wild_guess', 'prior', seed: 1))->fitTransform([[null, null], [2, 'x'], [4, 'x']]);
check('wild guess within range', $wild[0][0] >= 2 && $wild[0][0] <= 4 && $wild[0][1] === 'x');
echo json_encode((new KNNImputer(2))->fitTransform([[1, 1, 'a'], [1.1, null, '?'], [5, 5, 'b'], [5.2, 5.1, 'b'], [0.9, 1.2, 'a']])), "\n";
echo json_encode((new HotDeckImputer(1, seed: 1))->fitTransform([[1, 1], [1.1, null], [5, 5]])), "\n";
echo json_encode((new LambdaFunction(fn ($s, $i, $ctx) => [$s[0] * $ctx, $i], 10))->transform([[1], [2]])), "\n";

echo "--- serialization\n";
foreach ([(new MinMaxNormalizer())->fit($X), (new PowerTransformer())->fit($X), (new RobustStandardizer())->fit($X)] as $t) {
    check(get_class($t), unserialize(serialize($t))->transform($X) === $t->transform($X));
}
$oh2 = unserialize(serialize($oh));
check('OneHotEncoder', $oh2->transform(['blue', 9, false]) === $oh->transform(['blue', 9, false]));
$im = (new MissingDataImputer())->fit([[1, 'a'], [3, 'b']]);
check('MissingDataImputer', unserialize(serialize($im))->transform([[null, null]]) === $im->transform([[null, null]]));
$knn = (new KNNImputer())->fit([[1, 'a'], [3, 'b']]);
check('KNNImputer', unserialize(serialize($knn))->transform([[null, 'a']]) === $knn->transform([[null, 'a']]));

echo "--- errors\n";
expect_error(fn () => new MinMaxNormalizer(1, 1));
expect_error(fn () => new PolynomialExpander(0));
expect_error(fn () => (new OneHotEncoder())->fit([['a', 1]])->transform([['a', 'b']]));
expect_error(fn () => new MissingDataImputer('mode'));
expect_error(fn () => (new MissingDataImputer())->fit([[null], [null]]));
expect_error(fn () => (new KNNImputer())->fit([[null, 1], [1, null]]));
expect_error(fn () => new PowerTransformer([]));
?>
--EXPECT--
[[0,0,0],[0.01,0.333,0.333],[0.02,0.667,0.667],[1,1,1]]
[-0.98,-0.333,-0.333]
[[0.01,0.25,-0.5],[0.02,0.5,0],[0.03,0.75,0.5],[1,1,1]]
[[-1.5,-1.5,-1.5],[-0.5,-0.5,-0.5],[0.5,0.5,0.5],[97.5,1.5,1.5]]
[2.5,2.5,0.5]
power transform is standardized: ok
power transform learned lambdas: ok
[[1,2,3]]
[[0.25,0.75],[0,0]][0.6,0.8]
[[2,4,8,3,9,27]]
--- categorical and missing data
[[0,1,1.5,0,1],[1,0,2,1,0],[0,1,3,0,1]]
{"0":["blue","red"],"2":[0,1]}[0,0,1,0,1]
[[1.5,1,0],[2.5,0,1]]
[["a","x"],["b","y"],["c","x"],["a","y"]]
[["a"],["a"],["b"],["b"]]
int(1)
float(2.5)
string(3) "abc"
bool(true)
int(7)
[["true","false",1]][1,0]
[[1,"a"],[3,"b"],[3,"a"],[10,"a"],[3,"a"]]
[[-1,"none"],[2,"x"]]
[[9],[1],[9]]
wild guess within range: ok
[[1,1,"a"],[1.1,1.1,"a"],[5,5,"b"],[5.2,5.1,"b"],[0.9,1.2,"a"]]
[[1,1],[1.1,1],[5,5]]
[[10,0],[20,1]]
--- serialization
ML\Preprocessing\MinMaxNormalizer: ok
ML\Preprocessing\PowerTransformer: ok
ML\Preprocessing\RobustStandardizer: ok
OneHotEncoder: ok
MissingDataImputer: ok
KNNImputer: ok
--- errors
min must be less than max
degree must be between 1 and 64
Sample #0, feature #1 must be numeric (it was numeric during fit)
Unknown continuous strategy "mode"; expected mean, median, percentile(<p>), constant(<value>) or wild_guess
Feature #0 has no non-missing values to learn from
Need at least one sample without missing values to use as a donor
lambdas must not be empty
