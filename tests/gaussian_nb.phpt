--TEST--
ML\Classification\GaussianNB
--FILE--
<?php
use ML\Classification\GaussianNB;

$X = [[1.0, 2.0], [1.2, 1.8], [0.8, 2.2], [5.0, 6.0], [5.2, 5.8], [4.8, 6.2]];
$y = [0, 0, 0, 1, 1, 1];

$nb = (new GaussianNB())->fit($X, $y);
echo json_encode($nb->predict([[1, 2], [5, 6]])), "\n";

// The midpoint of two mirrored classes is a coin flip.
$p = $nb->predictProbability([3.0, 4.0]);
printf("%.3f %.3f\n", $p[0], $p[1]);
$batch = $nb->predictProbability([[1, 2]]);
var_dump($batch[0][0] > 0.99);

echo "--- numeric-string labels\n";
$s = (new GaussianNB())->fit([[0], [0.1], [5], [5.1]], ['7', '7', 'x', 'x']);
var_dump($s->getClasses());
var_dump($s->predict([0.05]));
echo json_encode(array_keys($s->predictProbability([0.05]))), "\n";

echo "--- serialize\n";
$copy = unserialize(serialize($nb));
var_dump($copy->predictProbability([2, 3]) === $nb->predictProbability([2, 3]));
$data = $nb->__serialize();
$data['variances'][0] = 0.0;
try {
    (new GaussianNB())->__unserialize($data);
} catch (ML\Exception\InvalidArgumentException $e) {
    echo $e->getMessage(), "\n";
}
?>
--EXPECT--
[0,1]
0.500 0.500
bool(true)
--- numeric-string labels
array(2) {
  [0]=>
  string(1) "7"
  [1]=>
  string(1) "x"
}
string(1) "7"
[7,"x"]
--- serialize
bool(true)
Invalid serialized data: variances must be positive
