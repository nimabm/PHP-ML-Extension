--TEST--
SVC, SVR, MLPClassifier, MLPRegressor, Adaline and the neural network building blocks
--FILE--
<?php
require __DIR__ . '/helpers.inc';

use ML\Classification\{SVC, MLPClassifier};
use ML\Regression\{SVR, MLPRegressor, Adaline};
use ML\NeuralNet\Layers\{Dense, Activation, Dropout, BatchNorm, PReLU, Swish, Noise};
use ML\NeuralNet\Optimizers\{Stochastic, Momentum, AdaGrad, RMSProp, Adam, AdaMax};
use ML\NeuralNet\Schedules\{Constant, StepDecay, Cosine, Cyclical, Ramp};
use ML\Datasets;
use ML\Metrics;

$c = circles(100, 1);
[$Xt, $Xs, $yt, $ys] = Datasets::trainTestSplit($c['samples'], $c['labels'], 0.3, true, 1);

echo "--- SVM\n";
$svc = (new SVC(1.0, 'rbf'))->fit($Xt, $yt);
check('SVC rbf separates circles', accuracy($svc, $Xs, $ys) > 0.95);
check('SVC linear cannot', accuracy((new SVC(1.0, 'linear'))->fit($Xt, $yt), $Xs, $ys) < 0.9);
check('SVC polynomial', accuracy((new SVC(1.0, 'polynomial', degree: 2, coef0: 1.0))->fit($Xt, $yt), $Xs, $ys) > 0.95);
check('SVC sigmoid runs', count((new SVC(1.0, 'sigmoid'))->fit($Xt, $yt)->predict($Xs)) === count($Xs));
$b = blobs(30, 23, 0.5);
check('SVC multiclass (one-vs-one)', accuracy((new SVC())->fit($b['samples'], $b['labels']), $b['samples'], $b['labels']) > 0.99);
check('support vectors', $svc->getSupportVectorCount() > 0 && $svc->getSupportVectorCount() < count($Xt));
check('SVC serialize', same_after_serialize($svc, $Xs));
$h = Datasets::hyperplane([2, -1], 0.5, 0.1, 150, 1);
check('SVR linear', Metrics::r2Score($h['targets'], (new SVR(10.0, 0.1, 'linear'))->fit($h['samples'], $h['targets'])->predict($h['samples'])) > 0.98);
$sx = $sy = [];
for ($i = 0; $i < 120; $i++) {
    $x = $i / 20;
    $sx[] = [$x];
    $sy[] = sin($x);
}
$svr = (new SVR(10.0, 0.05, 'rbf', 1.0))->fit($sx, $sy);
check('SVR rbf fits a sine', Metrics::r2Score($sy, $svr->predict($sx)) > 0.98);
check('SVR serialize', same_after_serialize($svr, $sx));

echo "--- neural networks\n";
$mlp = new MLPClassifier([new Dense(16), new Activation('relu'), new Dense(16), new Activation('tanh')], 32, new Adam(0.01), 300, seed: 1);
$mlp->fit($Xt, $yt);
check('MLP separates circles', accuracy($mlp, $Xs, $ys) > 0.95);
check('MLP probabilities', abs(array_sum($mlp->predictProbability($Xs[0])) - 1) < 1e-9 && array_keys($mlp->predictProbability($Xs[0])) === ['in', 'out']);
check('training curves', count($mlp->getLosses()) === count($mlp->getScores()) && count($mlp->getLosses()) <= 300);
check('MLP serialize', same_after_serialize($mlp, $Xs));
check('seeded training repeats', (new MLPClassifier([new Dense(8)], 32, new Adam(0.01), 50, seed: 5))->fit($Xt, $yt)->predictProbability($Xs)
    === (new MLPClassifier([new Dense(8)], 32, new Adam(0.01), 50, seed: 5))->fit($Xt, $yt)->predictProbability($Xs));

$architectures = [
    'batch norm + PReLU' => [new Dense(16), new BatchNorm(), new PReLU()],
    'swish + dropout' => [new Dense(16), new Swish(), new Dropout(0.2)],
    'noise + leaky relu' => [new Dense(16), new Noise(0.05), new Activation('leaky_relu(0.1)')],
    'gelu' => [new Dense(16), new Activation('gelu')],
    'selu' => [new Dense(16, initializer: 'lecun'), new Activation('selu')],
    'silu' => [new Dense(16), new Activation('silu')],
    'elu, xavier2' => [new Dense(16, initializer: 'xavier2'), new Activation('elu')],
    'softplus, l2' => [new Dense(16, 0.001), new Activation('softplus')],
    'softsign, uniform' => [new Dense(16, initializer: 'uniform(0.3)'), new Activation('softsign')],

    'thresholded relu' => [new Dense(16), new Activation('thresholded_relu(0.1)')],
];
foreach ($architectures as $name => $layers) {
    check("layers: $name", accuracy((new MLPClassifier($layers, 32, new Adam(0.01), 300, seed: 1))->fit($Xt, $yt), $Xs, $ys) > 0.9);
}
// Sigmoid saturates early on: a higher rate and a longer early-stopping window let it train.
check('layers: sigmoid', accuracy((new MLPClassifier([new Dense(16, initializer: 'normal(0.3)'), new Activation('sigmoid')], 32,
    new Adam(0.05), 300, window: 20, seed: 1))->fit($Xt, $yt), $Xs, $ys) > 0.9);
foreach ([new Stochastic(0.1), new Momentum(0.05), new Momentum(0.05, 0.1, true), new AdaGrad(0.1), new RMSProp(0.01), new AdaMax(0.02)] as $o) {
    check('optimizer ' . (new ReflectionClass($o))->getShortName(),
        accuracy((new MLPClassifier([new Dense(16), new Activation('relu')], 32, $o, 300, seed: 1))->fit($Xt, $yt), $Xs, $ys) > 0.9);
}
foreach ([new Constant(), new StepDecay(20, 0.5), new Cosine(50, 0.1), new Cyclical(10, 0.2), new Ramp(10)] as $s) {
    check('schedule ' . (new ReflectionClass($s))->getShortName(),
        accuracy((new MLPClassifier([new Dense(16), new Activation('relu')], 32, new Adam(0.01), 200, schedule: $s, seed: 1))->fit($Xt, $yt), $Xs, $ys) > 0.9);
}

$scaled = (new ML\Preprocessing\StandardScaler())->fitTransform($sx);
$reg = (new MLPRegressor([new Dense(32), new Activation('tanh'), new Dense(32), new Activation('tanh')], 32, new Adam(0.01), 500, seed: 1))->fit($scaled, $sy);
check('MLPRegressor fits a sine', Metrics::r2Score($sy, $reg->predict($scaled)) > 0.97);
$huber = (new MLPRegressor([new Dense(8), new Activation('relu')], 32, new Adam(0.01), 300, costFn: 'huber(1.0)', seed: 1))->fit($h['samples'], $h['targets']);
check('MLPRegressor with Huber loss', Metrics::r2Score($h['targets'], $huber->predict($h['samples'])) > 0.95);
$ada = (new Adaline(32, new Adam(0.05), 1e-4, 500, seed: 1))->fit($h['samples'], $h['targets']);
check('Adaline', Metrics::r2Score($h['targets'], $ada->predict($h['samples'])) > 0.98);
check('Adaline serialize', same_after_serialize($ada, $h['samples']));
check('building blocks serialize', unserialize(serialize(new Dense(4))) instanceof ML\NeuralNet\Layer
    && unserialize(serialize(new Adam())) instanceof ML\NeuralNet\Optimizer && unserialize(serialize(new Cosine())) instanceof ML\NeuralNet\Schedule);

echo "--- errors\n";
expect_error(fn () => new SVC(0));
expect_error(fn () => new SVC(kernel: 'bogus'));
expect_error(fn () => new Activation('bogus'));
expect_error(fn () => new Dense(0));
expect_error(fn () => new Dense(4, initializer: 'normal(-1)'));
expect_error(fn () => new Dropout(1.0));
expect_error(fn () => new Adam(-1));
expect_error(fn () => new MLPRegressor(costFn: 'cross_entropy'));
expect_error(fn () => new MLPClassifier([new Adam()]));
expect_error(fn () => (new MLPClassifier())->fit([[1], [2]], ['a', 'a']));
?>
--EXPECT--
--- SVM
SVC rbf separates circles: ok
SVC linear cannot: ok
SVC polynomial: ok
SVC sigmoid runs: ok
SVC multiclass (one-vs-one): ok
support vectors: ok
SVC serialize: ok
SVR linear: ok
SVR rbf fits a sine: ok
SVR serialize: ok
--- neural networks
MLP separates circles: ok
MLP probabilities: ok
training curves: ok
MLP serialize: ok
seeded training repeats: ok
layers: batch norm + PReLU: ok
layers: swish + dropout: ok
layers: noise + leaky relu: ok
layers: gelu: ok
layers: selu: ok
layers: silu: ok
layers: elu, xavier2: ok
layers: softplus, l2: ok
layers: softsign, uniform: ok
layers: thresholded relu: ok
layers: sigmoid: ok
optimizer Stochastic: ok
optimizer Momentum: ok
optimizer Momentum: ok
optimizer AdaGrad: ok
optimizer RMSProp: ok
optimizer AdaMax: ok
schedule Constant: ok
schedule StepDecay: ok
schedule Cosine: ok
schedule Cyclical: ok
schedule Ramp: ok
MLPRegressor fits a sine: ok
MLPRegressor with Huber loss: ok
Adaline: ok
Adaline serialize: ok
building blocks serialize: ok
--- errors
c must be greater than 0
Unknown kernel "bogus"; expected linear, polynomial, rbf or sigmoid
Unknown activation function "bogus"; expected relu, leaky_relu, elu, selu, gelu, sigmoid, tanh, softplus, softsign, silu, thresholded_relu or linear
neurons must be at least 1
Initializer scale must be greater than 0
ratio must be at least 0 and less than 1
rate must be greater than 0
Unknown cost function "cross_entropy"; expected least_squares or huber(<delta>)
Each hidden layer must implement ML\NeuralNet\Layer, ML\NeuralNet\Optimizers\Adam given
A classifier needs at least two distinct classes
