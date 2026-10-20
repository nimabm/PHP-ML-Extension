--TEST--
ML\Preprocessing\StandardScaler
--FILE--
<?php
use ML\Preprocessing\StandardScaler;

$round = fn (array $row) => array_map(fn ($v) => round($v, 4) + 0.0, $row);

$s = new StandardScaler();
$out = $s->fitTransform([[1, 10, 5], [3, 20, 5], [5, 30, 5]]);
echo json_encode(array_map($round, $out)), "\n";
echo json_encode($round($s->getMeans())), ' ', json_encode($round($s->getScales())), "\n";

// A constant feature is only centered.
echo json_encode($round($s->transform([3, 20, 6]))), "\n";

$t = (new StandardScaler())->fit([[2], [4]]);
echo json_encode($t->transform([[2], [4], [6]])), "\n";

echo "--- errors\n";
try {
    (new StandardScaler())->transform([[1]]);
} catch (ML\Exception\NotFittedException $e) {
    echo $e->getMessage(), "\n";
}
try {
    $s->transform([[1, 2]]);
} catch (ML\Exception\InvalidArgumentException $e) {
    echo $e->getMessage(), "\n";
}

echo "--- serialize\n";
$copy = unserialize(serialize($s));
var_dump($copy->transform([[4, 25, 5]]) === $s->transform([[4, 25, 5]]));
?>
--EXPECT--
[[-1.2247,-1.2247,0],[0,0,0],[1.2247,1.2247,0]]
[3,20,5] [1.633,8.165,1]
[0,0,1]
[[-1],[1],[3]]
--- errors
ML\Preprocessing\StandardScaler is not fitted yet; call fit() first
Sample #0 has 2 features, expected 3
--- serialize
bool(true)
