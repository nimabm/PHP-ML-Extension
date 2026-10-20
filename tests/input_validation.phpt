--TEST--
Input validation and the exception hierarchy
--FILE--
<?php
use ML\Regression\LinearRegression;
use ML\Classification\KNearestNeighbors;

$cases = [
    'ragged samples' => fn () => (new LinearRegression())->fit([[1, 2], [3]], [1, 2]),
    'non-numeric feature' => fn () => (new LinearRegression())->fit([[1, 'x']], [1]),
    'flat samples' => fn () => (new LinearRegression())->fit([1, 2], [1, 2]),
    'empty samples' => fn () => (new LinearRegression())->fit([], []),
    'too few targets' => fn () => (new LinearRegression())->fit([[1], [2]], [1]),
    'non-numeric target' => fn () => (new LinearRegression())->fit([[1], [2]], [1, 'a']),
    'empty feature list' => fn () => (new LinearRegression())->fit([[]], [1]),
    'bad label type' => fn () => (new KNearestNeighbors(1))->fit([[1]], [true]),
    'empty predict' => fn () => (new KNearestNeighbors(1))->fit([[1]], [1])->predict([]),
];
foreach ($cases as $name => $case) {
    try {
        $case();
        echo "$name: no exception\n";
    } catch (ML\Exception\MLException $e) {
        printf("%s: %s (%s)\n", $name, $e->getMessage(), $e instanceof InvalidArgumentException ? 'SPL' : 'other');
    }
}

echo "--- engine type checks still apply\n";
try {
    (new LinearRegression())->fit('x', []);
} catch (TypeError $e) {
    echo $e->getMessage(), "\n";
}

echo "--- references and string keys are fine\n";
$row = [1.0, 2.0];
$ref = &$row[0];
$m = (new LinearRegression(0.01))->fit(['a' => $row, 'b' => [2, 3], 'c' => [3, 5]], ['x' => 1, 'y' => 2, 'z' => 3]);
var_dump(is_float($m->predict(['p' => 4, 'q' => 7])));

echo "--- no dynamic properties\n";
try {
    $m->foo = 1;
} catch (Error $e) {
    echo $e->getMessage(), "\n";
}
?>
--EXPECT--
ragged samples: Sample #1 has 1 features, expected 2 (SPL)
non-numeric feature: Sample #0, feature #1 must be int or float, string given (SPL)
flat samples: Sample #0 must be an array of numbers, int given (SPL)
empty samples: Samples must not be empty (SPL)
too few targets: Number of samples (2) does not match number of targets (1) (SPL)
non-numeric target: Target #1 must be int or float, string given (SPL)
empty feature list: Sample #0 has no features (SPL)
bad label type: Label #0 must be int or string, bool given (SPL)
empty predict: Samples must not be empty (SPL)
--- engine type checks still apply
ML\Regression\LinearRegression::fit(): Argument #1 ($samples) must be of type array, string given
--- references and string keys are fine
bool(true)
--- no dynamic properties
Cannot create dynamic property ML\Regression\LinearRegression::$foo
