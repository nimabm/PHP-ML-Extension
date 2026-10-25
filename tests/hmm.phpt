--TEST--
HiddenMarkovModel and GaussianHMM: Baum-Welch, labeled training, Viterbi, posteriors, scoring and sampling
--FILE--
<?php
require __DIR__ . '/helpers.inc';

use ML\Sequence\{HiddenMarkovModel, GaussianHMM};

// The occasionally dishonest casino: a fair die and a loaded one (half sixes).
mt_srand(5);
$rolls = $truth = [];
$state = 0;
for ($t = 0; $t < 1500; $t++) {
    if (mt_rand() / mt_getrandmax() < 0.05) {
        $state = 1 - $state;
    }
    $p = $state ? [0.1, 0.1, 0.1, 0.1, 0.1, 0.5] : array_fill(0, 6, 1 / 6);
    $u = mt_rand() / mt_getrandmax();
    $face = 6;
    $acc = 0.0;
    foreach ($p as $k => $pk) {
        $acc += $pk;
        if ($u < $acc) {
            $face = $k + 1;
            break;
        }
    }
    $rolls[] = $face;
    $truth[] = $state ? 'loaded' : 'fair';
}

echo "--- labeled training\n";
$hmm = (new HiddenMarkovModel())->fitLabeled([$rolls], [$truth]);
echo json_encode($hmm->getStates()), ' ', json_encode($hmm->getSymbols()), "\n";
$T = $hmm->getTransitionMatrix();
check('transition rows sum to 1', abs(array_sum($T['fair']) - 1) < 1e-12 && abs(array_sum($T['loaded']) - 1) < 1e-12);
check('states are sticky', $T['fair']['fair'] > 0.9 && $T['loaded']['loaded'] > 0.9);
$E = $hmm->getEmissionProbabilities();
check('the loaded die favors six', $E['loaded'][6] > 0.4 && abs($E['fair'][6] - 1 / 6) < 0.05);
check('start probabilities', abs(array_sum($hmm->getStartProbabilities()) - 1) < 1e-12);
$path = $hmm->predict($rolls);
$hits = 0;
foreach ($path as $i => $s) {
    $hits += $s === $truth[$i];
}
check('Viterbi recovers most states', $hits / count($truth) > 0.7);
$post = $hmm->predictProbability([6, 6, 6, 6, 6, 6, 1]);
check('posteriors sum to 1 per step', abs(array_sum($post[3]) - 1) < 1e-12);
check('a run of sixes looks loaded', $post[3]['loaded'] > 0.8);
check('score of one sequence', is_float($hmm->score([1, 2, 3])));
$scores = $hmm->score([[1, 2, 3], [6, 6, 6, 6, 6]]);
check('scores of several sequences', count($scores) === 2 && $scores[1] < $scores[0]);
check('several paths at once', count($hmm->predict([[1, 2], [6, 6, 6]])) === 2);
echo json_encode($hmm->predict([6, 6, 'joker', 6])), "\n";
$sample = $hmm->sample(8, 3);
echo json_encode(array_keys($sample)), ' ', count($sample['observations']), ' ', count($sample['states']), "\n";
check('seeded sampling repeats', $hmm->sample(8, 3) === $sample);
check('log-likelihood after fit', $hmm->getLogLikelihood() < 0 && $hmm->getIterations() === 0);

echo "--- Baum-Welch\n";
$em = (new HiddenMarkovModel(2, seed: 1))->fit([$rolls]);
$six = array_map(fn ($row) => $row[6], $em->getEmissionProbabilities());
sort($six);
check('Baum-Welch finds the loaded die', $six[1] > 0.4 && $six[0] < 0.25);
check('states are numbered', $em->getStates() === [0, 1]);
check('log-likelihood improves on the start', $em->getLogLikelihood() > (new HiddenMarkovModel(2, maxIterations: 1, seed: 1))->fit([$rolls])->getLogLikelihood());
ini_set('mlnative.threads', 4);
$chunks = array_chunk($rolls, 100);
$a = (new HiddenMarkovModel(2, seed: 1))->fit($chunks)->getTransitionMatrix();
ini_set('mlnative.threads', 1);
check('threads do not change the result', (new HiddenMarkovModel(2, seed: 1))->fit($chunks)->getTransitionMatrix() === $a);
check('serialization', same_after_serialize($em, $rolls));

echo "--- GaussianHMM\n";
mt_srand(7);
$obs = $states = [];
$state = 0;
for ($t = 0; $t < 500; $t++) {
    if (mt_rand() / mt_getrandmax() < 0.03) {
        $state = 1 - $state;
    }
    $obs[] = [($state ? 5.0 : 0.0) + (mt_rand() / mt_getrandmax() - 0.5) * 2, ($state ? -2.0 : 2.0) + (mt_rand() / mt_getrandmax() - 0.5)];
    $states[] = $state ? 'high' : 'low';
}
$g = (new GaussianHMM(2, seed: 1))->fit([$obs]);
$means = $g->getMeans();
usort($means, fn ($x, $y) => $x[0] <=> $y[0]);
echo json_encode(rounded($means, 1)), "\n";
$p = $g->predict($obs);
$agree = 0;
foreach ($p as $i => $s) {
    $agree += ($s === 1) === ($states[$i] === 'high') ? 1 : 0;
}
check('state paths match up to relabeling', max($agree, 500 - $agree) / 500 > 0.98);
$labeled = (new GaussianHMM())->fitLabeled([$obs], [$states]);
echo json_encode($labeled->getStates()), ' ', json_encode(rounded($labeled->getMeans()['high'], 1)), "\n";
check('variances are positive', min(array_merge(...array_values($labeled->getVariances()))) > 0);
check('Viterbi with labeled states', $labeled->predict(array_slice($obs, 0, 5)) === array_slice($states, 0, 5));
check('several sequences', count($labeled->predict([array_slice($obs, 0, 3), array_slice($obs, 3, 4)])) === 2);
$s = $labeled->sample(4, 1);
check('sampled observations are vectors', count($s['observations']) === 4 && count($s['observations'][0]) === 2);
check('serialization', same_after_serialize($labeled, $obs));

echo "--- errors\n";
expect_error(fn () => (new HiddenMarkovModel())->fit([]));
expect_error(fn () => (new HiddenMarkovModel())->fit([[]]));
expect_error(fn () => (new HiddenMarkovModel())->fitLabeled([[1, 2, 3]], [['a', 'b']]));
expect_error(fn () => (new HiddenMarkovModel())->fitLabeled([[1, 2]], [['a', 'b'], ['a']]));
expect_error(fn () => (new HiddenMarkovModel())->predict([1, 2]));
expect_error(fn () => new HiddenMarkovModel(smoothing: 0.0));
expect_error(fn () => (new GaussianHMM(3))->fit([[[1.0], [2.0]]]));
expect_error(fn () => $labeled->predict([[1.0, 2.0, 3.0]]));
expect_error(fn () => $hmm->sample(0));
?>
--EXPECT--
--- labeled training
["fair","loaded"] [1,2,3,4,5,6]
transition rows sum to 1: ok
states are sticky: ok
the loaded die favors six: ok
start probabilities: ok
Viterbi recovers most states: ok
posteriors sum to 1 per step: ok
a run of sixes looks loaded: ok
score of one sequence: ok
scores of several sequences: ok
several paths at once: ok
["fair","fair","fair","fair"]
["observations","states"] 8 8
seeded sampling repeats: ok
log-likelihood after fit: ok
--- Baum-Welch
Baum-Welch finds the loaded die: ok
states are numbered: ok
log-likelihood improves on the start: ok
threads do not change the result: ok
serialization: ok
--- GaussianHMM
[[0,2],[5,-2]]
state paths match up to relabeling: ok
["high","low"] [5,-2]
variances are positive: ok
Viterbi with labeled states: ok
several sequences: ok
sampled observations are vectors: ok
serialization: ok
--- errors
Sequences must not be empty
Sequence #0 must not be empty
Sequence #0 has 3 observations but 2 states
Number of sequences (1) does not match number of state sequences (2)
ML\Sequence\HiddenMarkovModel is not fitted yet; call fit() first
smoothing must be greater than 0
Number of states (3) cannot exceed the number of observations (2)
Sample #0 has 3 features, expected 2
length must be between 1 and 10000000
