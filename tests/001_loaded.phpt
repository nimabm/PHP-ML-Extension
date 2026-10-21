--TEST--
mlnative is loaded and registers its classes
--FILE--
<?php
var_dump(extension_loaded('mlnative'));
$names = [
    'ML\Exception\MLException',
    'ML\Exception\InvalidArgumentException',
    'ML\Exception\RuntimeException',
    'ML\Exception\NotFittedException',
    'ML\Regression\LinearRegression',
    'ML\Classification\LogisticRegression',
    'ML\Classification\KNearestNeighbors',
    'ML\Classification\GaussianNB',
    'ML\Clustering\KMeans',
    'ML\Preprocessing\StandardScaler',
    'ML\Metrics',
];
foreach ($names as $name) {
    echo $name, ': ', class_exists($name) || interface_exists($name) ? 'ok' : 'missing', "\n";
}
var_dump((new ReflectionClass(ML\Regression\LinearRegression::class))->isFinal());
var_dump(is_subclass_of(ML\Exception\NotFittedException::class, LogicException::class));
var_dump(is_subclass_of(ML\Exception\RuntimeException::class, ML\Exception\MLException::class));
?>
--EXPECT--
bool(true)
ML\Exception\MLException: ok
ML\Exception\InvalidArgumentException: ok
ML\Exception\RuntimeException: ok
ML\Exception\NotFittedException: ok
ML\Regression\LinearRegression: ok
ML\Classification\LogisticRegression: ok
ML\Classification\KNearestNeighbors: ok
ML\Classification\GaussianNB: ok
ML\Clustering\KMeans: ok
ML\Preprocessing\StandardScaler: ok
ML\Metrics: ok
bool(true)
bool(true)
bool(true)
