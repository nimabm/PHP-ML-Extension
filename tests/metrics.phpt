--TEST--
ML\Metrics
--FILE--
<?php
use ML\Metrics;

var_dump(Metrics::accuracy(['a', 'b', 'c', 'a'], ['a', 'b', 'a', 'a']));
var_dump(Metrics::accuracy([1, 2], ['1', 2]));  // strict comparison

$expected = [3, -0.5, 2, 7];
$predicted = [2.5, 0.0, 2, 8];
var_dump(Metrics::meanSquaredError($expected, $predicted));
var_dump(Metrics::meanAbsoluteError($expected, $predicted));
printf("%.6f\n", Metrics::r2Score($expected, $predicted));
var_dump(Metrics::r2Score([1, 1], [1, 1]), Metrics::r2Score([1, 1], [1, 2]));

echo "--- errors\n";
try {
    Metrics::accuracy([1, 2], [1]);
} catch (ML\Exception\InvalidArgumentException $e) {
    echo $e->getMessage(), "\n";
}
try {
    Metrics::meanSquaredError([], []);
} catch (ML\Exception\InvalidArgumentException $e) {
    echo $e->getMessage(), "\n";
}
try {
    new Metrics();
} catch (Error $e) {
    echo $e->getMessage(), "\n";
}
?>
--EXPECT--
float(0.75)
float(0.5)
float(0.375)
float(0.5)
0.948608
float(1)
float(0)
--- errors
Expected (2) and predicted (1) arrays must have the same size
Arrays must not be empty
Call to private ML\Metrics::__construct() from global scope
