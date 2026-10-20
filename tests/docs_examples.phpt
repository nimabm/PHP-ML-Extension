--TEST--
Every <?php example in README.md and docs/DOCUMENTATION.md runs without errors
--SKIPIF--
<?php
if (!is_file(__DIR__ . '/../docs/DOCUMENTATION.md')) die('skip docs not available');
if (!is_file(__DIR__ . '/../README.md')) die('skip README not available');
if (!function_exists('proc_open')) die('skip proc_open() is disabled');
?>
--FILE--
<?php
// Each example runs in its own process so that functions, `use` statements and
// globals cannot leak between examples.
$extension = __DIR__ . '/../modules/mlnative.so';
if (!is_file($extension)) {
    $extension = 'mlnative';  // installed into extension_dir
}

$failures = 0;
foreach (['README.md' => 5, 'docs/DOCUMENTATION.md' => 20] as $path => $minExamples) {
    $doc = file_get_contents(__DIR__ . '/../' . $path);
    preg_match_all('/^```php\n(<\?php\n.*?)^```$/ms', $doc, $blocks, PREG_OFFSET_CAPTURE);

    foreach ($blocks[1] as [$code, $offset]) {
        $line = substr_count($doc, "\n", 0, $offset) + 1;
        $file = tempnam(sys_get_temp_dir(), 'mldoc');
        file_put_contents($file, $code);
        $cmd = [PHP_BINARY, '-n', '-d', 'extension_dir=' . ini_get('extension_dir'),
                '-d', 'extension=' . $extension, '-d', 'error_reporting=-1', '-d', 'display_errors=1', $file];
        $proc = proc_open($cmd, [1 => ['pipe', 'w'], 2 => ['redirect', 1]], $pipes);
        $output = stream_get_contents($pipes[1]);
        fclose($pipes[1]);
        $status = proc_close($proc);
        unlink($file);

        if ($status !== 0 || preg_match('/(Fatal error|Warning|Notice|Deprecated|Uncaught)/', $output)) {
            $failures++;
            echo "Example at $path:$line failed (exit $status):\n$output\n";
        }
    }

    echo $path, ': ', count($blocks[1]) >= $minExamples ? "examples found\n" : "too few examples found\n";
}

echo $failures === 0 ? "all examples ran\n" : "$failures example(s) failed\n";
?>
--EXPECT--
README.md: examples found
docs/DOCUMENTATION.md: examples found
all examples ran
