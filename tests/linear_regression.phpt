--TEST--
ML\Regression\LinearRegression
--FILE--
<?php
use ML\Regression\LinearRegression;

// y = 2a + 3b + 1
$X = [[1, 2], [2, 1], [3, 4], [4, 3], [5, 5]];
$y = [9, 8, 19, 18, 26];

$m = new LinearRegression();
var_dump($m->fit($X, $y) === $m);
vprintf("coef %.6f %.6f intercept %.6f\n", [...$m->getCoefficients(), $m->getIntercept()]);
printf("single %.4f\n", $m->predict([6, 6]));
echo 'batch ', json_encode(array_map(fn ($v) => round($v, 6), $m->predict([[0, 0], [1, 1]]))), "\n";

echo "--- ridge shrinks coefficients\n";
$ridge = (new LinearRegression(l2: 10.0))->fit($X, $y);
$norm = fn (array $c) => sqrt(array_sum(array_map(fn ($v) => $v * $v, $c)));
var_dump($norm($ridge->getCoefficients()) < $norm($m->getCoefficients()));

echo "--- collinear features\n";
try {
    (new LinearRegression())->fit([[1, 2], [2, 4], [3, 6]], [1, 2, 3]);
} catch (ML\Exception\RuntimeException $e) {
    echo get_class($e), "\n";
}
$fixed = (new LinearRegression(0.001))->fit([[1, 2], [2, 4], [3, 6]], [1, 2, 3]);
printf("%.3f\n", $fixed->predict([4, 8]));

echo "--- errors\n";
try {
    (new LinearRegression())->predict([1]);
} catch (ML\Exception\NotFittedException $e) {
    echo $e->getMessage(), "\n";
}
try {
    $m->predict([1, 2, 3]);
} catch (ML\Exception\InvalidArgumentException $e) {
    echo $e->getMessage(), "\n";
}
try {
    new LinearRegression(-1);
} catch (ML\Exception\InvalidArgumentException $e) {
    echo $e->getMessage(), "\n";
}

echo "--- failed fit keeps the previous model\n";
try {
    $m->fit([[1, 1], [2, 2]], [1, 2]);
} catch (ML\Exception\RuntimeException $e) {
}
printf("%.4f\n", $m->predict([6, 6]));

echo "--- serialize / clone\n";
$copy = unserialize(serialize($m));
var_dump($copy->predict([6, 6]) === $m->predict([6, 6]));
$clone = clone $m;
$clone->fit([[0], [1]], [0, 1]);
var_dump(count($m->getCoefficients()), count($clone->getCoefficients()));
?>
--EXPECT--
bool(true)
coef 2.000000 3.000000 intercept 1.000000
single 31.0000
batch [1,6]
--- ridge shrinks coefficients
bool(true)
--- collinear features
ML\Exception\RuntimeException
4.000
--- errors
ML\Regression\LinearRegression is not fitted yet; call fit() first
Sample has 3 features, expected 2
l2 must be greater than or equal to 0
--- failed fit keeps the previous model
31.0000
--- serialize / clone
bool(true)
int(2)
int(1)
