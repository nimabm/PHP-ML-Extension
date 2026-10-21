<?php
// Shared helpers for the phpt tests.

function check(string $label, bool $ok): void
{
    echo $label, ': ', $ok ? 'ok' : 'FAIL', "\n";
}

function rounded(array $values, int $precision = 3): array
{
    return array_map(fn ($v) => is_array($v) ? rounded($v, $precision) : round($v, $precision) + 0.0, $values);
}

// Prints the message of the exception thrown by $fn (prefixed with its class
// when it is not an ML\Exception\MLException).
function expect_error(callable $fn): void
{
    try {
        $fn();
        echo "no exception\n";
    } catch (Throwable $e) {
        echo $e instanceof ML\Exception\MLException ? '' : get_class($e) . ': ', $e->getMessage(), "\n";
    }
}

// Three well-separated 2-D blobs labeled a, b and c.
function blobs(int $n = 40, int $seed = 1, float $spread = 0.5): array
{
    return ML\Datasets::agglomerate([
        'a' => ML\Datasets::blob([0, 0], $spread, $n, $seed),
        'b' => ML\Datasets::blob([4, 4], $spread, $n, $seed + 1),
        'c' => ML\Datasets::blob([0, 5], $spread, $n, $seed + 2),
    ], true, $seed);
}

// Two concentric circles (not linearly separable).
function circles(int $n = 100, int $seed = 1): array
{
    return ML\Datasets::agglomerate([
        'in' => ML\Datasets::circle(0, 0, 1.0, 0.05, $n, $seed),
        'out' => ML\Datasets::circle(0, 0, 3.0, 0.05, $n, $seed + 1),
    ], true, $seed);
}

function accuracy(object $model, array $samples, array $labels): float
{
    return ML\Metrics::accuracy($labels, $model->predict($samples));
}

function same_after_serialize(object $model, array $samples): bool
{
    // Compared through serialize() so that NAN predictions compare equal.
    return serialize(unserialize(serialize($model))->predict($samples)) === serialize($model->predict($samples));
}
