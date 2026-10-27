--TEST--
stubs/mlnative.stub.php matches the extension (classes, interfaces, signatures)
--SKIPIF--
<?php
if (!is_file(__DIR__ . '/../stubs/mlnative.stub.php')) die('skip stub not available');
?>
--FILE--
<?php
// The stub is loaded under a Stub\ prefix (interfaces first, since a stub may
// declare them after the classes that use them) and compared through reflection.
$src = file_get_contents(__DIR__ . '/../stubs/mlnative.stub.php');
$iface = '/^\s*interface \w+[^{]*\{\}\s*$/m';
preg_match_all('/^namespace ([\w\\\\]+) \{(.*?)^\}/ms', $src, $blocks, PREG_SET_ORDER);
$prelude = '';
foreach ($blocks as [, $ns, $body]) {
    if (preg_match_all($iface, $body, $found)) {
        $prelude .= "namespace $ns {\n" . implode("\n", $found[0]) . "\n}\n";
    }
}
$src = preg_replace('/^<\?php\n/', "<?php\n" . $prelude, preg_replace($iface, '', $src), 1);
$src = preg_replace('/^namespace ML/m', 'namespace Stub\ML', str_replace('\\ML\\', '\\Stub\\ML\\', $src));
$file = tempnam(sys_get_temp_dir(), 'stub');
file_put_contents($file, $src);
require $file;
unlink($file);

$strip = fn (string $name): string => str_replace('Stub\\', '', $name);
$signature = function (ReflectionMethod $m) use ($strip): string {
    $params = array_map(function (ReflectionParameter $p) use ($strip) {
        $default = $p->isDefaultValueAvailable() ? ' = ' . var_export($p->getDefaultValue(), true) : '';
        return trim(($p->getType() ? $strip((string) $p->getType()) : '') . ' $' . $p->getName() . $default);
    }, $m->getParameters());
    return ($m->isStatic() ? 'static ' : '') . '(' . implode(', ', $params) . ')' .
        ($m->hasReturnType() ? ': ' . $strip((string) $m->getReturnType()) : '');
};

$problems = [];
$names = array_filter(array_merge(get_declared_classes(), get_declared_interfaces()), fn ($n) => str_starts_with($n, 'ML\\'));
foreach ($names as $name) {
    if (!class_exists("Stub\\$name") && !interface_exists("Stub\\$name")) {
        $problems[] = "$name is missing from the stub";
        continue;
    }
    $real = new ReflectionClass($name);
    $stub = new ReflectionClass("Stub\\$name");
    $ri = $real->getInterfaceNames();
    $si = array_map($strip, $stub->getInterfaceNames());
    sort($ri);
    sort($si);
    if ($ri !== $si) {
        $problems[] = "$name implements " . implode(', ', $ri) . ' but the stub says ' . implode(', ', $si);
    }
    foreach ($real->getMethods() as $m) {
        $method = $m->getName();
        if (str_starts_with($method, '__') && ($method !== '__construct' || $m->getNumberOfParameters() === 0)) {
            continue;
        }
        if (!$stub->hasMethod($method)) {
            $problems[] = "$name::$method is missing from the stub";
        } elseif ($signature($m) !== $signature($stub->getMethod($method))) {
            $problems[] = "$name::$method is {$signature($m)} but the stub says {$signature($stub->getMethod($method))}";
        }
    }
    foreach ($stub->getMethods() as $m) {
        if (!$real->hasMethod($m->getName())) {
            $problems[] = "$name::{$m->getName()} exists only in the stub";
        }
    }
}
foreach (array_merge(get_declared_classes(), get_declared_interfaces()) as $name) {
    if (str_starts_with($name, 'Stub\\ML\\') && !in_array($strip($name), $names, true)) {
        $problems[] = $strip($name) . ' exists only in the stub';
    }
}
echo $problems ? implode("\n", $problems) . "\n" : "stub matches " . count($names) . " names\n";
?>
--EXPECT--
stub matches 142 names
