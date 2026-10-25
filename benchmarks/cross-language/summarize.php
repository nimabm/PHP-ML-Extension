<?php
// Turns the "runtime|benchmark|median ms|min ms|checksum" lines of run.sh into a
// Markdown table of medians, each with its slowdown relative to mlnative.

$columns = ['knn' => 'KNN', 'logreg' => 'LogReg', 'linreg' => 'LinReg', 'kmeans' => 'KMeans', 'tree' => 'Tree', 'mlp' => 'MLP'];
$results = [];
foreach (file($argv[1], FILE_IGNORE_NEW_LINES | FILE_SKIP_EMPTY_LINES) as $line) {
    [$runtime, $bench, $median] = explode('|', $line);
    $results[$runtime][$bench] = (float) $median;
}
$base = $results['PHP+mlnative'] ?? [];

echo '| Runtime | ', implode(' | ', $columns), " |\n";
echo '|---|', str_repeat('---:|', count($columns)), "\n";
foreach ($results as $runtime => $times) {
    $cells = [];
    foreach ($columns as $key => $_) {
        if (!isset($times[$key])) {
            $cells[] = '—';
        } elseif ($runtime === 'PHP+mlnative' || !isset($base[$key])) {
            $cells[] = sprintf('%.1f', $times[$key]);
        } else {
            $cells[] = sprintf('%.1f (%.1f×)', $times[$key], $times[$key] / $base[$key]);
        }
    }
    echo "| $runtime | ", implode(' | ', $cells), " |\n";
}
echo "\nMedian wall time in ms; (n×) = times slower than mlnative, below 1 = faster.\n";
