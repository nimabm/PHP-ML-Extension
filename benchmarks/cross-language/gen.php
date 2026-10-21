<?php
// Writes the shared datasets once so every language works on identical numbers.
// Format: first line "rows cols", then one row per line, space separated.

mt_srand(1);

function dataset(int $n, int $d): array
{
    $X = [];
    $y = [];
    for ($i = 0; $i < $n; $i++) {
        $row = [];
        $sum = 0.0;
        for ($j = 0; $j < $d; $j++) {
            $v = mt_rand() / mt_getrandmax() * 2 - 1;
            $row[] = $v;
            $sum += $v * ($j + 1);
        }
        $X[] = $row;
        $y[] = $sum;
    }
    return [$X, $y];
}

function save(string $file, array $rows): void
{
    $rows = array_map(fn ($r) => is_array($r) ? $r : [$r], $rows);
    $out = count($rows) . ' ' . count($rows[0]) . "\n";
    foreach ($rows as $r) {
        $out .= implode(' ', array_map(fn ($v) => is_int($v) ? (string) $v : sprintf('%.17g', $v), $r)) . "\n";
    }
    file_put_contents($file, $out);
}

$dir = $argv[1];
[$X, $y] = dataset(3000, 16);
save("$dir/cls_X.txt", $X);
save("$dir/cls_y.txt", array_map(fn ($v) => $v > 0 ? 1 : 0, $y));
[$Q] = dataset(300, 16);
save("$dir/knn_Q.txt", $Q);
[$Xl, $yl] = dataset(20000, 32);
save("$dir/lin_X.txt", $Xl);
save("$dir/lin_y.txt", $yl);
[$Xk] = dataset(5000, 8);
save("$dir/km_X.txt", $Xk);

// Decision tree: a non-linear boundary so the tree has to grow deep.
$treeLabel = fn ($x) => $x[0] * $x[1] + $x[2] * $x[2] - $x[3] + 0.5 * sin(3 * $x[4]) > 0.1 ? 1 : 0;
[$Xt] = dataset(5000, 16);
save("$dir/tree_X.txt", $Xt);
save("$dir/tree_y.txt", array_map($treeLabel, $Xt));
[$Qt] = dataset(1000, 16);
save("$dir/tree_Q.txt", $Qt);

// MLP 16-32-16-2: shared initial weights (He normal for hidden layers, Xavier
// uniform for the output, zero biases) and the shuffle order of every epoch.
function gauss(): float
{
    $u = (mt_rand() + 1) / (mt_getrandmax() + 2);
    $v = mt_rand() / mt_getrandmax();
    return sqrt(-2 * log($u)) * cos(2 * M_PI * $v);
}
$layer = function (int $in, int $out, bool $he): array {
    $rows = [];
    $limit = sqrt(6 / ($in + $out));
    for ($o = 0; $o < $out; $o++) {
        for ($i = 0; $i < $in; $i++) {
            $rows[$o][] = $he ? gauss() * sqrt(2 / $in) : (mt_rand() / mt_getrandmax() * 2 - 1) * $limit;
        }
    }
    return $rows;
};
save("$dir/mlp_W1.txt", $layer(16, 32, true));
save("$dir/mlp_W2.txt", $layer(32, 16, true));
save("$dir/mlp_W3.txt", $layer(16, 2, false));
$perms = [];
for ($e = 0; $e < 10; $e++) {
    $p = range(0, 2999);
    for ($i = 2999; $i > 0; $i--) {
        $j = mt_rand(0, $i);
        [$p[$i], $p[$j]] = [$p[$j], $p[$i]];
    }
    $perms[] = $p;
}
save("$dir/mlp_perm.txt", $perms);
echo "datasets written\n";
