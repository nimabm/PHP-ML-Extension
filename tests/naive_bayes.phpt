--TEST--
NaiveBayes (categorical) and MultinomialNB
--FILE--
<?php
require __DIR__ . '/helpers.inc';

use ML\Classification\{NaiveBayes, MultinomialNB};

// no:  [sunny, hot], [sunny, mild]       yes: [rain, mild], [rain, cool], [overcast, hot]
$X = [['sunny', 'hot'], ['sunny', 'mild'], ['rain', 'mild'], ['rain', 'cool'], ['overcast', 'hot']];
$y = ['no', 'no', 'yes', 'yes', 'yes'];

$nb = (new NaiveBayes())->fit($X, $y);
echo json_encode($nb->predict([['sunny', 'cool'], ['rain', 'hot'], ['fog', 'hot']])), "\n";
// P(no) ∝ 2/5 * (2+1)/(2+4) * (1+1)/(2+4);  P(yes) ∝ 3/5 * 1/7 * 2/7
echo json_encode(rounded($nb->predictProbability(['sunny', 'hot']), 4)), "\n";
echo json_encode($nb->getClasses()), "\n";

$even = (new NaiveBayes(1.0, ['no' => 0.5, 'yes' => 0.5]))->fit($X, $y);
echo json_encode(rounded($even->predictProbability(['sunny', 'hot']), 3)), "\n";

$ints = (new NaiveBayes())->fit([[1, true], [1, false], [2, true]], [0, 0, 1]);
var_dump($ints->predict([2, true]));
check('serialize', same_after_serialize($nb, $X));

echo "--- MultinomialNB\n";
$counts = [[3, 0, 1], [2, 0, 0], [0, 4, 1], [0, 3, 2]];
$mn = (new MultinomialNB())->fit($counts, ['spam', 'spam', 'ham', 'ham']);
echo json_encode($mn->predict([[1, 0, 0], [0, 2, 0]])), "\n";
check('probabilities sum to 1', abs(array_sum($mn->predictProbability([1, 1, 1])) - 1) < 1e-12);
check('serialize', same_after_serialize($mn, $counts));

echo "--- errors\n";
expect_error(fn () => (new NaiveBayes())->fit([[1.5]], ['a']));
expect_error(fn () => (new NaiveBayes(1.0, ['no' => 1]))->fit($X, $y));
expect_error(fn () => new NaiveBayes(0));
expect_error(fn () => (new MultinomialNB())->fit([[-1]], ['a']));
?>
--EXPECT--
["no","yes","yes"]
{"no":0.7313,"yes":0.2687}
["no","yes"]
{"no":0.803,"yes":0.197}
int(1)
serialize: ok
--- MultinomialNB
["spam","ham"]
probabilities sum to 1: ok
serialize: ok
--- errors
Sample #0, feature #0 must be a categorical value (int, string or bool); use GaussianNB or IntervalDiscretizer for continuous features
priors must contain a probability for every class in the training labels
smoothing must be greater than 0
MultinomialNB requires non-negative features (counts or frequencies)
