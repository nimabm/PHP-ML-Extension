--TEST--
ML\Classification\KNearestNeighbors
--FILE--
<?php
use ML\Classification\KNearestNeighbors;

$X = [[1, 1], [1, 2], [2, 1], [8, 8], [8, 9], [9, 8]];
$y = ['red', 'red', 'red', 'blue', 'blue', 'blue'];

$knn = (new KNearestNeighbors(3))->fit($X, $y);
var_dump($knn->predict([1.5, 1.5]));
echo json_encode($knn->predict([[8.5, 8.5], [2, 2]])), "\n";
echo json_encode($knn->getClasses()), "\n";

$manhattan = (new KNearestNeighbors(1, 'manhattan'))->fit($X, $y);
var_dump($manhattan->predict([7, 7]));

echo "--- ties go to the nearer class\n";
$tie = (new KNearestNeighbors(2))->fit([[0], [3]], ['a', 'b']);
echo $tie->predict([1]), $tie->predict([2]), "\n";

echo "--- errors\n";
try {
    (new KNearestNeighbors(10))->fit($X, $y);
} catch (ML\Exception\InvalidArgumentException $e) {
    echo $e->getMessage(), "\n";
}
try {
    new KNearestNeighbors(3, 'bogus');
} catch (ML\Exception\InvalidArgumentException $e) {
    echo $e->getMessage(), "\n";
}
try {
    new KNearestNeighbors(0);
} catch (ML\Exception\InvalidArgumentException $e) {
    echo $e->getMessage(), "\n";
}

echo "--- serialize\n";
$copy = unserialize(serialize($knn));
var_dump($copy->predict([[8.5, 8.5], [2, 2]]) === $knn->predict([[8.5, 8.5], [2, 2]]));
$data = $knn->__serialize();
$data['labels'][0] = 99;
try {
    (new KNearestNeighbors())->__unserialize($data);
} catch (ML\Exception\InvalidArgumentException $e) {
    echo $e->getMessage(), "\n";
}
?>
--EXPECT--
string(3) "red"
["blue","red"]
["blue","red"]
string(4) "blue"
--- ties go to the nearer class
ab
--- errors
k (10) cannot exceed the number of training samples (6)
Unknown distance metric "bogus"; expected euclidean, squared_euclidean, manhattan, chebyshev, minkowski, minkowski(<p>), cosine, canberra, hamming, jaccard or safe_euclidean
k must be at least 1
--- serialize
bool(true)
Invalid serialized data: label index out of range
