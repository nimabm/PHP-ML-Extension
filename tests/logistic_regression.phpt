--TEST--
ML\Classification\LogisticRegression
--FILE--
<?php
use ML\Classification\LogisticRegression;

echo "--- binary, string labels\n";
$X = [[0, 0], [0, 1], [1, 0], [1, 1], [4, 4], [4, 5], [5, 4], [5, 5]];
$y = ['no', 'no', 'no', 'no', 'yes', 'yes', 'yes', 'yes'];
$m = (new LogisticRegression(learningRate: 0.5, maxIterations: 2000))->fit($X, $y);
echo json_encode($m->getClasses()), "\n";
echo json_encode($m->predict([[0.5, 0.5], [4.5, 4.5]])), "\n";
var_dump($m->predict([4.8, 5.1]));
$p = $m->predictProbability([0.2, 0.3]);
echo json_encode(array_keys($p)), "\n";
printf("%.6f\n", array_sum($p));
var_dump($p['no'] > 0.9);

echo "--- multiclass, int labels\n";
$X3 = [[0, 0], [0, 1], [1, 0], [5, 5], [5, 6], [6, 5], [0, 9], [1, 9], [0, 10]];
$y3 = [2, 2, 2, 0, 0, 0, 1, 1, 1];
$m3 = (new LogisticRegression(0.1, 3000))->fit($X3, $y3);
echo json_encode($m3->getClasses()), "\n";
echo json_encode($m3->predict([[0.5, 0.5], [5.5, 5.5], [0.5, 9.5]])), "\n";
$w = $m3->getWeights();
echo json_encode(array_keys($w)), ' ', count($w[0]), "\n";
$probs = $m3->predictProbability([[0.5, 0.5], [5.5, 5.5]]);
foreach ($probs as $row) {
    printf("%.6f\n", array_sum($row));
}

echo "--- errors\n";
try {
    (new LogisticRegression())->fit([[1], [2]], ['a', 'a']);
} catch (ML\Exception\InvalidArgumentException $e) {
    echo $e->getMessage(), "\n";
}
try {
    (new LogisticRegression())->fit([[1], [2]], [1.5, 2.5]);
} catch (ML\Exception\InvalidArgumentException $e) {
    echo $e->getMessage(), "\n";
}
try {
    new LogisticRegression(learningRate: 0);
} catch (ML\Exception\InvalidArgumentException $e) {
    echo $e->getMessage(), "\n";
}

echo "--- serialize\n";
$copy = unserialize(serialize($m3));
var_dump($copy->predictProbability([3, 3]) === $m3->predictProbability([3, 3]));
$data = $m3->__serialize();
array_pop($data['bias']);
try {
    (new LogisticRegression())->__unserialize($data);
} catch (ML\Exception\InvalidArgumentException $e) {
    echo $e->getMessage(), "\n";
}
?>
--EXPECT--
--- binary, string labels
["no","yes"]
["no","yes"]
string(3) "yes"
["no","yes"]
1.000000
bool(true)
--- multiclass, int labels
[0,1,2]
[2,0,1]
[0,1,2] 2
1.000000
1.000000
--- errors
LogisticRegression needs at least two distinct classes
Label #0 must be int or string, float given
learningRate must be greater than 0
--- serialize
bool(true)
Invalid serialized data: inconsistent LogisticRegression state
