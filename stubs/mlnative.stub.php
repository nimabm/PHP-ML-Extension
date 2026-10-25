<?php

/**
 * IDE stub for the mlnative extension (version 0.1.3). Not loaded at runtime:
 * the classes are implemented in C++ (src/). Keep in sync with the arginfo in
 * src/ and with docs/DOCUMENTATION.md.
 *
 * Samples are arrays of int|float features. Methods that take `$samples`
 * accept either a list of samples (returns one result per sample) or a single
 * sample (returns a single result).
 */

namespace ML {
    /** Marker interfaces describing what an object can do. */
    interface Estimator {}
    interface Learner {}
    interface Classifier extends Estimator {}
    interface Regressor extends Estimator {}
    interface Clusterer extends Estimator {}
    interface AnomalyDetector extends Estimator {}
    interface Probabilistic {}
    interface Transformer {}
    interface Stateful {}
    interface RanksFeatures {}
    /** Learners that can also train incrementally with partialFit(). */
    interface Online {}

    final class Metrics
    {
        private function __construct() {}
        public static function accuracy(array $expected, array $predicted): float {}
        public static function precision(array $expected, array $predicted, int|string|null $positiveClass = null): float {}
        public static function recall(array $expected, array $predicted, int|string|null $positiveClass = null): float {}
        public static function informedness(array $expected, array $predicted, int|string|null $positiveClass = null): float {}
        public static function fBeta(array $expected, array $predicted, float $beta = 1.0, int|string|null $positiveClass = null): float {}
        public static function f1Score(array $expected, array $predicted, int|string|null $positiveClass = null): float {}
        public static function mcc(array $expected, array $predicted): float {}
        public static function rocAuc(array $expected, array $scores, int|string $positiveClass): float {}
        public static function averagePrecision(array $expected, array $scores, int|string $positiveClass): float {}
        /** @param list<array<int|string, float>> $probabilities */
        public static function brierScore(array $expected, array $probabilities): float {}
        public static function probabilisticAccuracy(array $expected, array $probabilities): float {}
        public static function topKAccuracy(array $expected, array $probabilities, int $k = 3): float {}
        /**
         * Ranking metrics for recommendations. $recommended is a ranked list of items and
         * $relevant the items that are relevant, or a list of such lists (one per user,
         * averaged; users without relevant items are skipped).
         */
        public static function precisionAtK(array $recommended, array $relevant, int $k = 10): float {}
        public static function recallAtK(array $recommended, array $relevant, int $k = 10): float {}
        public static function ndcgAtK(array $recommended, array $relevant, int $k = 10): float {}
        public static function meanSquaredError(array $expected, array $predicted): float {}
        public static function meanAbsoluteError(array $expected, array $predicted): float {}
        public static function rootMeanSquaredError(array $expected, array $predicted): float {}
        public static function medianAbsoluteError(array $expected, array $predicted): float {}
        public static function r2Score(array $expected, array $predicted): float {}
        /** Symmetric mean absolute percentage error, 0..100. */
        public static function smape(array $expected, array $predicted): float {}
        public static function homogeneity(array $labels, array $clusters): float {}
        public static function completeness(array $labels, array $clusters): float {}
        public static function vMeasure(array $labels, array $clusters, float $beta = 1.0): float {}
        /** Adjusted Rand index. */
        public static function randIndex(array $labels, array $clusters): float {}
        public static function purity(array $labels, array $clusters): float {}
        public static function trustworthiness(array $original, array $embedded, int $k = 5): float {}
    }

    final class Datasets
    {
        /** @param float|list<float> $stddev */
        public static function blob(array $center, float|array $stddev = 1.0, int $n = 100, ?int $seed = null): array {}
        public static function circle(float $x = 0.0, float $y = 0.0, float $scale = 1.0, float $noise = 0.1, int $n = 100, ?int $seed = null): array {}
        public static function halfMoon(float $x = 0.0, float $y = 0.0, float $scale = 1.0, float $rotation = 90.0, float $noise = 0.1, int $n = 100, ?int $seed = null): array {}
        /** @return array{samples: list<list<float>>, targets: list<float>} */
        public static function swissRoll(float $x = 0.0, float $y = 0.0, float $z = 0.0, float $scale = 1.0, float $depth = 21.0, float $noise = 0.1, int $n = 100, ?int $seed = null): array {}
        /** @return array{samples: list<list<float>>, targets: list<float>} */
        public static function hyperplane(array $coefficients, float $intercept = 0.0, float $noise = 0.1, int $n = 100, ?int $seed = null): array {}
        /** @return list<array{0: int, 1: int, 2: int}> [user, item, rating 1-5] triples */
        public static function ratings(int $users = 100, int $items = 50, float $density = 0.2, int $factors = 2, float $noise = 0.5, ?int $seed = null): array {}
        /** @param array<int|string, list<array>> $groups label => samples @return array{samples: list<array>, labels: list<int|string>} */
        public static function agglomerate(array $groups, bool $shuffle = true, ?int $seed = null): array {}
        /** @return array{0: list<array>, 1: list<array>, 2: list, 3: list} [trainSamples, testSamples, trainTargets, testTargets] */
        public static function trainTestSplit(array $samples, array $targets, float $testRatio = 0.25, bool $stratify = false, ?int $seed = null): array {}
    }

    final class Distance
    {
        /** @param string $metric euclidean, squared_euclidean, manhattan, chebyshev, minkowski, minkowski(<p>), cosine, canberra, hamming, jaccard or safe_euclidean */
        public static function compute(string $metric, array $a, array $b): float {}
        public static function pairwise(string $metric, array $samples): array {}
        public static function gower(array $a, array $b, array $ranges): float {}
    }

    final class Pipeline implements Learner, Estimator
    {
        /** @param list<Transformer> $transformers */
        public function __construct(array $transformers, Estimator $estimator) {}
        public function fit(array $samples, ?array $targets = null): static {}
        public function predict(array $samples) {}
        public function predictProbability(array $samples): array {}
        public function score(array $samples) {}
        public function transform(array $samples): array {}
        public function getEstimator(): Estimator {}
        public function getTransformers(): array {}
    }
}

namespace ML\Exception {
    interface MLException {}
    class InvalidArgumentException extends \InvalidArgumentException implements MLException {}
    class RuntimeException extends \RuntimeException implements MLException {}
    class NotFittedException extends \LogicException implements MLException {}
}

namespace ML\Regression {
    use ML\NeuralNet\Optimizer;
    use ML\NeuralNet\Schedule;

    final class LinearRegression implements \ML\Learner, \ML\Regressor
    {
        public function __construct(float $l2 = 0.0) {}
        public function fit(array $samples, array $targets): static {}
        public function predict(array $samples): array|float {}
        public function getCoefficients(): array {}
        public function getIntercept(): float {}
    }

    final class ElasticNet implements \ML\Learner, \ML\Regressor
    {
        public function __construct(float $alpha = 1.0, float $l1Ratio = 0.5, int $maxIterations = 1000, float $tolerance = 1.0E-4) {}
        public function fit(array $samples, array $targets): static {}
        public function predict(array $samples): array|float {}
        public function getCoefficients(): array {}
        public function getIntercept(): float {}
    }

    final class RegressionTree implements \ML\Learner, \ML\Regressor, \ML\RanksFeatures
    {
        public function __construct(int $maxHeight = PHP_INT_MAX, int $maxLeafSize = 3, float $minPurityIncrease = 1.0E-7, ?int $maxFeatures = null, ?int $seed = null) {}
        public function fit(array $samples, array $targets): static {}
        public function predict(array $samples): array|float {}
        public function getFeatureImportances(): array {}
        public function getHeight(): int {}
        public function getLeafCount(): int {}
        public function rules(?array $featureNames = null): string {}
    }

    final class ExtraTreeRegressor implements \ML\Learner, \ML\Regressor, \ML\RanksFeatures
    {
        public function __construct(int $maxHeight = PHP_INT_MAX, int $maxLeafSize = 3, float $minPurityIncrease = 1.0E-7, ?int $maxFeatures = null, ?int $seed = null) {}
        public function fit(array $samples, array $targets): static {}
        public function predict(array $samples): array|float {}
        public function getFeatureImportances(): array {}
        public function getHeight(): int {}
        public function getLeafCount(): int {}
        public function rules(?array $featureNames = null): string {}
    }

    final class RandomForestRegressor implements \ML\Learner, \ML\Regressor, \ML\RanksFeatures
    {
        public function __construct(int $estimators = 100, float $ratio = 1.0, int $maxHeight = PHP_INT_MAX, int $maxLeafSize = 3, float $minPurityIncrease = 1.0E-7, ?int $maxFeatures = null, bool $extraTrees = false, ?int $seed = null) {}
        public function fit(array $samples, array $targets): static {}
        public function predict(array $samples): array|float {}
        public function getFeatureImportances(): array {}
    }

    final class GradientBoost implements \ML\Learner, \ML\Regressor, \ML\RanksFeatures
    {
        public function __construct(int $estimators = 100, float $rate = 0.1, float $ratio = 0.5, int $maxHeight = 3, int $maxLeafSize = 3, float $minChange = 1.0E-4, int $window = 5, float $holdOut = 0.1, ?int $seed = null) {}
        public function fit(array $samples, array $targets): static {}
        public function predict(array $samples): array|float {}
        public function getFeatureImportances(): array {}
        public function getLosses(): array {}
        public function getValidationScores(): array {}
        public function getEstimatorCount(): int {}
    }

    final class KNNRegressor implements \ML\Learner, \ML\Regressor
    {
        public function __construct(int $k = 5, bool $weighted = false, string $metric = 'euclidean') {}
        public function fit(array $samples, array $targets): static {}
        public function predict(array $samples): array|float {}
    }

    final class KDNeighborsRegressor implements \ML\Learner, \ML\Regressor
    {
        public function __construct(int $k = 5, bool $weighted = false, int $maxLeafSize = 30, string $metric = 'euclidean') {}
        public function fit(array $samples, array $targets): static {}
        public function predict(array $samples): array|float {}
    }

    final class RadiusNeighborsRegressor implements \ML\Learner, \ML\Regressor
    {
        public function __construct(float $radius = 1.0, bool $weighted = false, string $metric = 'euclidean') {}
        public function fit(array $samples, array $targets): static {}
        /** NAN for samples without neighbors in the radius. */
        public function predict(array $samples): array|float {}
    }

    final class SVR implements \ML\Learner, \ML\Regressor
    {
        public function __construct(float $c = 1.0, float $epsilon = 0.1, string $kernel = 'rbf', ?float $gamma = null, int $degree = 3, float $coef0 = 0.0, float $tolerance = 0.001, int $cacheSize = 128) {}
        public function fit(array $samples, array $targets): static {}
        public function predict(array $samples): array|float {}
        public function getSupportVectorCount(): int {}
    }

    final class MLPRegressor implements \ML\Learner, \ML\Regressor
    {
        /** @param list<\ML\NeuralNet\Layer> $hiddenLayers */
        public function __construct(array $hiddenLayers = [], int $batchSize = 128, ?Optimizer $optimizer = null, int $epochs = 1000, float $minChange = 1.0E-4, int $window = 5, float $holdOut = 0.1, float $l2 = 0.0, string $costFn = 'least_squares', ?Schedule $schedule = null, ?int $seed = null) {}
        public function fit(array $samples, array $targets): static {}
        public function predict(array $samples): array|float {}
        public function getLosses(): array {}
        public function getScores(): array {}
    }

    final class Adaline implements \ML\Learner, \ML\Regressor
    {
        public function __construct(int $batchSize = 128, ?Optimizer $optimizer = null, float $l2 = 1.0E-4, int $epochs = 1000, float $minChange = 1.0E-4, int $window = 5, float $holdOut = 0.1, string $costFn = 'least_squares', ?int $seed = null) {}
        public function fit(array $samples, array $targets): static {}
        public function predict(array $samples): array|float {}
        public function getLosses(): array {}
        public function getScores(): array {}
    }
}

namespace ML\Classification {
    use ML\NeuralNet\Optimizer;
    use ML\NeuralNet\Schedule;

    final class LogisticRegression implements \ML\Learner, \ML\Classifier, \ML\Probabilistic
    {
        public function __construct(float $learningRate = 0.1, int $maxIterations = 1000, float $l2 = 0.0, float $tolerance = 1.0E-6) {}
        public function fit(array $samples, array $targets): static {}
        public function predict(array $samples): array|int|string {}
        public function predictProbability(array $samples): array {}
        public function getClasses(): array {}
        public function getWeights(): array {}
    }

    final class KNearestNeighbors implements \ML\Learner, \ML\Classifier, \ML\Probabilistic
    {
        public function __construct(int $k = 3, string $metric = 'euclidean', bool $weighted = false) {}
        public function fit(array $samples, array $targets): static {}
        public function predict(array $samples): array|int|string {}
        public function predictProbability(array $samples): array {}
        public function getClasses(): array {}
    }

    final class KDNeighbors implements \ML\Learner, \ML\Classifier, \ML\Probabilistic
    {
        public function __construct(int $k = 5, bool $weighted = false, int $maxLeafSize = 30, string $metric = 'euclidean') {}
        public function fit(array $samples, array $targets): static {}
        public function predict(array $samples): array|int|string {}
        public function predictProbability(array $samples): array {}
        public function getClasses(): array {}
    }

    final class RadiusNeighbors implements \ML\Learner, \ML\Classifier, \ML\Probabilistic
    {
        public function __construct(float $radius = 1.0, bool $weighted = false, int|string $outlierClass = '?', string $metric = 'euclidean') {}
        public function fit(array $samples, array $targets): static {}
        public function predict(array $samples): array|int|string {}
        public function predictProbability(array $samples): array {}
        public function getClasses(): array {}
    }

    final class GaussianNB implements \ML\Learner, \ML\Classifier, \ML\Probabilistic
    {
        public function __construct(float $varSmoothing = 1.0E-9) {}
        public function fit(array $samples, array $targets): static {}
        public function predict(array $samples): array|int|string {}
        public function predictProbability(array $samples): array {}
        public function getClasses(): array {}
    }

    final class NaiveBayes implements \ML\Learner, \ML\Classifier, \ML\Probabilistic
    {
        /** @param array<int|string, float>|null $priors label => prior probability */
        public function __construct(float $smoothing = 1.0, ?array $priors = null) {}
        /** @param list<list<int|string|bool>> $samples categorical features */
        public function fit(array $samples, array $targets): static {}
        public function predict(array $samples): array|int|string {}
        public function predictProbability(array $samples): array {}
        public function getClasses(): array {}
    }

    final class MultinomialNB implements \ML\Learner, \ML\Classifier, \ML\Probabilistic
    {
        public function __construct(float $smoothing = 1.0, ?array $priors = null) {}
        public function fit(array $samples, array $targets): static {}
        public function predict(array $samples): array|int|string {}
        public function predictProbability(array $samples): array {}
        public function getClasses(): array {}
    }

    final class ClassificationTree implements \ML\Learner, \ML\Classifier, \ML\Probabilistic, \ML\RanksFeatures
    {
        public function __construct(int $maxHeight = PHP_INT_MAX, int $maxLeafSize = 3, float $minPurityIncrease = 1.0E-7, ?int $maxFeatures = null, ?int $seed = null) {}
        public function fit(array $samples, array $targets): static {}
        public function predict(array $samples): array|int|string {}
        public function predictProbability(array $samples): array {}
        public function getClasses(): array {}
        public function getFeatureImportances(): array {}
        public function getHeight(): int {}
        public function getLeafCount(): int {}
        public function rules(?array $featureNames = null): string {}
    }

    final class ExtraTreeClassifier implements \ML\Learner, \ML\Classifier, \ML\Probabilistic, \ML\RanksFeatures
    {
        public function __construct(int $maxHeight = PHP_INT_MAX, int $maxLeafSize = 3, float $minPurityIncrease = 1.0E-7, ?int $maxFeatures = null, ?int $seed = null) {}
        public function fit(array $samples, array $targets): static {}
        public function predict(array $samples): array|int|string {}
        public function predictProbability(array $samples): array {}
        public function getClasses(): array {}
        public function getFeatureImportances(): array {}
        public function getHeight(): int {}
        public function getLeafCount(): int {}
        public function rules(?array $featureNames = null): string {}
    }

    final class RandomForest implements \ML\Learner, \ML\Classifier, \ML\Probabilistic, \ML\RanksFeatures
    {
        public function __construct(int $estimators = 100, float $ratio = 1.0, bool $balanced = false, int $maxHeight = PHP_INT_MAX, int $maxLeafSize = 3, float $minPurityIncrease = 1.0E-7, ?int $maxFeatures = null, bool $extraTrees = false, ?int $seed = null) {}
        public function fit(array $samples, array $targets): static {}
        public function predict(array $samples): array|int|string {}
        public function predictProbability(array $samples): array {}
        public function getClasses(): array {}
        public function getFeatureImportances(): array {}
    }

    final class AdaBoost implements \ML\Learner, \ML\Classifier, \ML\Probabilistic, \ML\RanksFeatures
    {
        public function __construct(int $estimators = 100, float $rate = 1.0, float $ratio = 0.8, int $maxHeight = 1, float $minChange = 1.0E-4, ?int $seed = null) {}
        public function fit(array $samples, array $targets): static {}
        public function predict(array $samples): array|int|string {}
        public function predictProbability(array $samples): array {}
        public function getClasses(): array {}
        public function getFeatureImportances(): array {}
        public function getLosses(): array {}
        public function getEstimatorCount(): int {}
    }

    final class LogitBoost implements \ML\Learner, \ML\Classifier, \ML\Probabilistic, \ML\RanksFeatures
    {
        public function __construct(int $estimators = 100, float $rate = 0.1, float $ratio = 0.5, int $maxHeight = 3, float $minChange = 1.0E-4, ?int $seed = null) {}
        public function fit(array $samples, array $targets): static {}
        public function predict(array $samples): array|int|string {}
        public function predictProbability(array $samples): array {}
        public function getClasses(): array {}
        public function getFeatureImportances(): array {}
        public function getLosses(): array {}
        public function getEstimatorCount(): int {}
    }

    final class SVC implements \ML\Learner, \ML\Classifier
    {
        public function __construct(float $c = 1.0, string $kernel = 'rbf', ?float $gamma = null, int $degree = 3, float $coef0 = 0.0, float $tolerance = 0.001, int $cacheSize = 128) {}
        public function fit(array $samples, array $targets): static {}
        public function predict(array $samples): array|int|string {}
        public function getClasses(): array {}
        public function getSupportVectorCount(): int {}
    }

    final class MLPClassifier implements \ML\Learner, \ML\Classifier, \ML\Probabilistic
    {
        /** @param list<\ML\NeuralNet\Layer> $hiddenLayers */
        public function __construct(array $hiddenLayers = [], int $batchSize = 128, ?Optimizer $optimizer = null, int $epochs = 1000, float $minChange = 1.0E-4, int $window = 5, float $holdOut = 0.1, float $l2 = 0.0, ?Schedule $schedule = null, ?int $seed = null) {}
        public function fit(array $samples, array $targets): static {}
        public function predict(array $samples): array|int|string {}
        public function predictProbability(array $samples): array {}
        public function getClasses(): array {}
        public function getLosses(): array {}
        public function getScores(): array {}
    }
}

namespace ML\Clustering {
    final class KMeans implements \ML\Learner, \ML\Clusterer, \ML\Probabilistic
    {
        /** @param string $seeder plusplus, random, kmc2 or preset @param list<array>|null $centroids preset initial centroids */
        public function __construct(int $clusters = 8, int $maxIterations = 300, float $tolerance = 1.0E-4, ?int $seed = null, string $seeder = 'plusplus', ?array $centroids = null) {}
        public function fit(array $samples): static {}
        public function predict(array $samples): array|int {}
        public function predictProbability(array $samples): array {}
        public function getCentroids(): array {}
        public function getLabels(): array {}
        public function getInertia(): float {}
        public function getIterations(): int {}
    }

    final class KMedoids implements \ML\Learner, \ML\Clusterer
    {
        public function __construct(int $clusters = 8, int $maxIterations = 300, string $metric = 'euclidean', string $seeder = 'plusplus', ?int $seed = null) {}
        public function fit(array $samples): static {}
        public function predict(array $samples): array|int {}
        public function getMedoids(): array {}
        public function getLabels(): array {}
        public function getInertia(): float {}
        public function getIterations(): int {}
    }

    final class FuzzyCMeans implements \ML\Learner, \ML\Clusterer, \ML\Probabilistic
    {
        public function __construct(int $clusters = 8, float $fuzz = 2.0, int $maxIterations = 300, float $minChange = 1.0E-4, ?int $seed = null) {}
        public function fit(array $samples): static {}
        public function predict(array $samples): array|int {}
        public function predictProbability(array $samples): array {}
        public function getCentroids(): array {}
        public function getInertia(): float {}
        public function getIterations(): int {}
    }

    final class GaussianMixture implements \ML\Learner, \ML\Clusterer, \ML\Probabilistic
    {
        public function __construct(int $components = 8, float $smoothing = 1.0E-9, int $maxIterations = 100, float $minChange = 0.001, ?int $seed = null) {}
        public function fit(array $samples): static {}
        public function predict(array $samples): array|int {}
        public function predictProbability(array $samples): array {}
        public function getMeans(): array {}
        public function getVariances(): array {}
        public function getPriors(): array {}
        public function getLogLikelihood(): float {}
        public function getIterations(): int {}
    }

    final class MeanShift implements \ML\Learner, \ML\Clusterer
    {
        public function __construct(?float $radius = null, float $percentile = 30.0, int $maxIterations = 100, float $minShift = 1.0E-4, int $maxSeeds = 1000, ?int $seed = null) {}
        public function fit(array $samples): static {}
        public function predict(array $samples): array|int {}
        public function getCentroids(): array {}
        public function getRadius(): float {}
        public static function estimateRadius(array $samples, float $percentile = 30.0, ?int $seed = null): float {}
    }

    final class DBSCAN implements \ML\Clusterer
    {
        public function __construct(float $radius = 0.5, int $minDensity = 5, string $metric = 'euclidean') {}
        /** Clusters the given samples; -1 marks noise. */
        public function predict(array $samples): array {}
    }

    final class OPTICS implements \ML\Clusterer
    {
        public function __construct(int $minDensity = 5, float $radius = 0.5, float $maxRadius = INF, string $metric = 'euclidean') {}
        public function predict(array $samples): array {}
        /** @return array{ordering: list<int>, reachability: list<float>, coreDistances: list<float>, labels: list<int>} */
        public function reachability(array $samples): array {}
    }

    final class Ward implements \ML\Clusterer
    {
        public function __construct(int $clusters = 2) {}
        public function predict(array $samples): array {}
    }

    final class MiniBatchKMeans implements \ML\Learner, \ML\Online, \ML\Clusterer, \ML\Probabilistic
    {
        /** @param string $seeder plusplus, random, kmc2 or preset @param list<array>|null $centroids preset initial centroids */
        public function __construct(int $clusters = 8, int $batchSize = 256, int $maxIterations = 100, float $tolerance = 1.0E-4, ?int $seed = null, string $seeder = 'plusplus', ?array $centroids = null) {}
        public function fit(array $samples): static {}
        /** Updates the centroids with one more batch (the first call also initializes them). */
        public function partialFit(array $samples): static {}
        public function predict(array $samples): array|int {}
        public function predictProbability(array $samples): array {}
        public function getCentroids(): array {}
        /** Clusters of the samples of the last fit() or partialFit() call. */
        public function getLabels(): array {}
        public function getInertia(): float {}
        public function getIterations(): int {}
    }

    final class SpectralClustering implements \ML\Clusterer
    {
        /** @param string $affinity rbf or nearest */
        public function __construct(int $clusters = 8, string $affinity = 'rbf', float $gamma = 1.0, int $neighbors = 10, int $runs = 10, ?int $seed = null) {}
        public function predict(array $samples): array {}
    }
}

namespace ML\AnomalyDetection {
    /** predict() returns 1 for anomalies and 0 for inliers; score() returns raw scores (higher = more anomalous). */
    final class RobustZScore implements \ML\Learner, \ML\AnomalyDetector
    {
        public function __construct(float $threshold = 3.5, float $beta = 0.5, float $smoothing = 1.0E-9) {}
        public function fit(array $samples): static {}
        public function predict(array $samples): array|int {}
        public function score(array $samples): array|float {}
        public function getThreshold(): float {}
    }

    final class GaussianMLE implements \ML\Learner, \ML\AnomalyDetector
    {
        public function __construct(float $contamination = 0.1, float $smoothing = 1.0E-9) {}
        public function fit(array $samples): static {}
        public function predict(array $samples): array|int {}
        public function score(array $samples): array|float {}
        public function getThreshold(): float {}
    }

    final class IsolationForest implements \ML\Learner, \ML\AnomalyDetector
    {
        public function __construct(int $estimators = 100, int $maxSamples = 256, ?float $contamination = null, ?int $seed = null) {}
        public function fit(array $samples): static {}
        public function predict(array $samples): array|int {}
        public function score(array $samples): array|float {}
        public function getThreshold(): float {}
    }

    final class LocalOutlierFactor implements \ML\Learner, \ML\AnomalyDetector
    {
        public function __construct(int $k = 20, float $contamination = 0.1, string $metric = 'euclidean') {}
        public function fit(array $samples): static {}
        public function predict(array $samples): array|int {}
        public function score(array $samples): array|float {}
        public function getThreshold(): float {}
    }

    final class Loda implements \ML\Learner, \ML\AnomalyDetector
    {
        public function __construct(int $estimators = 100, ?int $bins = null, float $contamination = 0.1, ?int $seed = null) {}
        public function fit(array $samples): static {}
        public function predict(array $samples): array|int {}
        public function score(array $samples): array|float {}
        public function getThreshold(): float {}
    }

    final class OneClassSVM implements \ML\Learner, \ML\AnomalyDetector
    {
        public function __construct(float $nu = 0.1, string $kernel = 'rbf', ?float $gamma = null, int $degree = 3, float $coef0 = 0.0, float $tolerance = 0.001, int $cacheSize = 128) {}
        public function fit(array $samples): static {}
        public function predict(array $samples): array|int {}
        public function score(array $samples): array|float {}
        public function getThreshold(): float {}
        public function getSupportVectorCount(): int {}
    }
}

namespace ML\Ensemble {
    final class OneVsRest implements \ML\Learner, \ML\Classifier, \ML\Probabilistic
    {
        public function __construct(\ML\Classifier $base) {}
        public function fit(array $samples, array $targets): static {}
        public function predict(array $samples): array|int|string {}
        public function predictProbability(array $samples): array {}
        public function getClasses(): array {}
    }

    final class BootstrapAggregator implements \ML\Learner, \ML\Estimator
    {
        public function __construct(\ML\Learner $base, int $estimators = 10, float $ratio = 0.5, ?int $seed = null) {}
        public function fit(array $samples, ?array $targets = null): static {}
        public function predict(array $samples) {}
    }

    final class CommitteeMachine implements \ML\Learner, \ML\Estimator
    {
        /** @param list<\ML\Learner> $experts @param list<float>|null $influences */
        public function __construct(array $experts, ?array $influences = null) {}
        public function fit(array $samples, ?array $targets = null): static {}
        public function predict(array $samples) {}
        public function getInfluences(): array {}
    }
}

namespace ML\ModelSelection {
    use ML\CrossValidation\Validator;

    final class GridSearch implements \ML\Learner, \ML\Estimator
    {
        /** @param class-string $class @param array<string, mixed> $grid parameter => list of values @param string|callable|null $metric */
        public function __construct(string $class, array $grid, ?Validator $validator = null, mixed $metric = null) {}
        public function fit(array $samples, array $targets): static {}
        public function predict(array $samples) {}
        public function predictProbability(array $samples): array {}
        public function score(array $samples) {}
        /** @return array{params: array, score: float} */
        public function getBest(): array {}
        public function getResults(): array {}
        public function getEstimator(): \ML\Estimator {}
    }

    final class BayesianSearch implements \ML\Learner, \ML\Estimator
    {
        /** @param array<string, array{min: int|float, max: int|float, log?: bool}|list<mixed>|mixed> $space */
        public function __construct(string $class, array $space, int $trials = 10, ?Validator $validator = null, mixed $metric = null, ?int $seed = null) {}
        public function fit(array $samples, array $targets): static {}
        public function predict(array $samples) {}
        public function predictProbability(array $samples): array {}
        public function score(array $samples) {}
        public function getBest(): array {}
        public function getResults(): array {}
        public function getEstimator(): \ML\Estimator {}
    }
}

namespace ML\CrossValidation {
    interface Validator {}

    final class HoldOut implements Validator
    {
        public function __construct(float $ratio = 0.2, bool $stratify = false, ?int $seed = null) {}
        /** @param string|callable|null $metric */
        public function test(object $estimator, array $samples, array $targets, mixed $metric = null): float {}
        public function scores(object $estimator, array $samples, array $targets, mixed $metric = null): array {}
    }

    final class KFold implements Validator
    {
        public function __construct(int $k = 5, bool $stratify = false, ?int $seed = null) {}
        public function test(object $estimator, array $samples, array $targets, mixed $metric = null): float {}
        public function scores(object $estimator, array $samples, array $targets, mixed $metric = null): array {}
    }

    final class LeavePOut implements Validator
    {
        public function __construct(int $p = 10) {}
        public function test(object $estimator, array $samples, array $targets, mixed $metric = null): float {}
        public function scores(object $estimator, array $samples, array $targets, mixed $metric = null): array {}
    }

    final class MonteCarlo implements Validator
    {
        public function __construct(int $simulations = 10, float $ratio = 0.2, bool $stratify = false, ?int $seed = null) {}
        public function test(object $estimator, array $samples, array $targets, mixed $metric = null): float {}
        public function scores(object $estimator, array $samples, array $targets, mixed $metric = null): array {}
    }

    final class Reports
    {
        public static function confusionMatrix(array $expected, array $predicted): array {}
        public static function multiclassBreakdown(array $expected, array $predicted): array {}
        public static function errorAnalysis(array $expected, array $predicted): array {}
        public static function contingencyTable(array $labels, array $clusters): array {}
    }
}

namespace ML\Preprocessing {
    final class StandardScaler implements \ML\Transformer, \ML\Stateful
    {
        public function fit(array $samples): static {}
        public function transform(array $samples): array {}
        public function fitTransform(array $samples): array {}
        public function getMeans(): array {}
        public function getScales(): array {}
    }

    final class MinMaxNormalizer implements \ML\Transformer, \ML\Stateful
    {
        public function __construct(float $min = 0.0, float $max = 1.0) {}
        public function fit(array $samples): static {}
        public function transform(array $samples): array {}
        public function fitTransform(array $samples): array {}
    }

    final class MaxAbsoluteScaler implements \ML\Transformer, \ML\Stateful
    {
        public function fit(array $samples): static {}
        public function transform(array $samples): array {}
        public function fitTransform(array $samples): array {}
    }

    final class RobustStandardizer implements \ML\Transformer, \ML\Stateful
    {
        public function __construct(bool $center = true) {}
        public function fit(array $samples): static {}
        public function transform(array $samples): array {}
        public function fitTransform(array $samples): array {}
    }

    final class PowerTransformer implements \ML\Transformer, \ML\Stateful
    {
        public function __construct(?array $lambdas = null, bool $normalize = true) {}
        public function fit(array $samples): static {}
        public function transform(array $samples): array {}
        public function fitTransform(array $samples): array {}
        public function getLambdas(): array {}
    }

    final class L1Normalizer implements \ML\Transformer
    {
        public function transform(array $samples): array {}
    }

    final class L2Normalizer implements \ML\Transformer
    {
        public function transform(array $samples): array {}
    }

    final class PolynomialExpander implements \ML\Transformer
    {
        public function __construct(int $degree = 2) {}
        public function transform(array $samples): array {}
    }

    final class OneHotEncoder implements \ML\Transformer, \ML\Stateful
    {
        /** @param list<int>|null $columns columns to encode; null = every column holding strings or booleans */
        public function __construct(?array $columns = null) {}
        public function fit(array $samples): static {}
        public function transform(array $samples): array {}
        public function fitTransform(array $samples): array {}
        public function getCategories(): array {}
    }

    final class IntervalDiscretizer implements \ML\Transformer, \ML\Stateful
    {
        public function __construct(int $bins = 5, bool $equiWidth = true) {}
        public function fit(array $samples): static {}
        public function transform(array $samples): array {}
        public function fitTransform(array $samples): array {}
    }

    final class NumericStringConverter implements \ML\Transformer
    {
        public function transform(array $samples): array {}
    }

    final class BooleanConverter implements \ML\Transformer
    {
        public function __construct(int|string $trueValue = 'true', int|string $falseValue = 'false') {}
        public function transform(array $samples): array {}
    }

    final class MissingDataImputer implements \ML\Transformer, \ML\Stateful
    {
        /** @param string $continuous mean, median, percentile(<p>), constant(<v>) or wild_guess @param string $categorical most_frequent, k_most_frequent(<k>), prior or constant(<value>) */
        public function __construct(string $continuous = 'mean', string $categorical = 'most_frequent', string $placeholder = '?', ?int $seed = null) {}
        public function fit(array $samples): static {}
        public function transform(array $samples): array {}
        public function fitTransform(array $samples): array {}
    }

    final class KNNImputer implements \ML\Transformer, \ML\Stateful
    {
        public function __construct(int $k = 5, bool $weighted = false, string $placeholder = '?', ?int $seed = null) {}
        public function fit(array $samples): static {}
        public function transform(array $samples): array {}
        public function fitTransform(array $samples): array {}
    }

    final class HotDeckImputer implements \ML\Transformer, \ML\Stateful
    {
        public function __construct(int $k = 5, bool $weighted = false, string $placeholder = '?', ?int $seed = null) {}
        public function fit(array $samples): static {}
        public function transform(array $samples): array {}
        public function fitTransform(array $samples): array {}
    }

    final class LambdaFunction implements \ML\Transformer
    {
        /** @param callable(array $sample, int $index, mixed $context): array $callback */
        public function __construct(callable $callback, mixed $context = null) {}
        public function transform(array $samples): array {}
    }

    final class PCA implements \ML\Transformer, \ML\Stateful
    {
        public function __construct(int $dimensions = 2) {}
        public function fit(array $samples): static {}
        public function transform(array $samples): array {}
        public function fitTransform(array $samples): array {}
        public function getExplainedVariance(): array {}
        public function getExplainedVarianceRatio(): array {}
        public function getComponents(): array {}
    }

    final class TruncatedSVD implements \ML\Transformer, \ML\Stateful
    {
        public function __construct(int $dimensions = 2) {}
        public function fit(array $samples): static {}
        public function transform(array $samples): array {}
        public function fitTransform(array $samples): array {}
        public function getSingularValues(): array {}
        public function getExplainedVarianceRatio(): array {}
        public function getComponents(): array {}
    }

    final class LinearDiscriminantAnalysis implements \ML\Transformer, \ML\Stateful
    {
        public function __construct(int $dimensions = 1) {}
        public function fit(array $samples, array $labels): static {}
        public function transform(array $samples): array {}
        public function fitTransform(array $samples, array $labels): array {}
        public function getExplainedVariance(): array {}
    }

    final class GaussianRandomProjector implements \ML\Transformer, \ML\Stateful
    {
        public function __construct(int $dimensions, ?int $seed = null) {}
        public function fit(array $samples): static {}
        public function transform(array $samples): array {}
        public function fitTransform(array $samples): array {}
        public static function minDimensions(int $samples, float $maxDistortion = 0.5): int {}
    }

    final class SparseRandomProjector implements \ML\Transformer, \ML\Stateful
    {
        public function __construct(int $dimensions, float $sparsity = 3.0, ?int $seed = null) {}
        public function fit(array $samples): static {}
        public function transform(array $samples): array {}
        public function fitTransform(array $samples): array {}
        public static function minDimensions(int $samples, float $maxDistortion = 0.5): int {}
    }

    final class TSNE implements \ML\Transformer
    {
        public function __construct(int $dimensions = 2, float $rate = 100.0, float $perplexity = 30.0, float $exaggeration = 12.0, int $epochs = 1000, float $minGradient = 1.0E-7, ?int $seed = null) {}
        /** Embeds the given samples (t-SNE has no separate fit). */
        public function transform(array $samples): array {}
        public function fitTransform(array $samples): array {}
        public function getLosses(): array {}
    }

    final class KernelPCA implements \ML\Transformer, \ML\Stateful
    {
        /** @param string $kernel linear, polynomial, rbf or sigmoid @param float|null $gamma null = 1 / features */
        public function __construct(int $dimensions = 2, string $kernel = 'rbf', ?float $gamma = null, int $degree = 3, float $coef0 = 1.0) {}
        public function fit(array $samples): static {}
        public function transform(array $samples): array {}
        public function fitTransform(array $samples): array {}
        public function getEigenvalues(): array {}
    }

    final class FastICA implements \ML\Transformer, \ML\Stateful
    {
        /** @param int|null $dimensions null = one component per feature @param string $contrast logcosh, exp or cube */
        public function __construct(?int $dimensions = null, string $contrast = 'logcosh', int $maxIterations = 200, float $tolerance = 1.0E-4, ?int $seed = null) {}
        public function fit(array $samples): static {}
        public function transform(array $samples): array {}
        public function fitTransform(array $samples): array {}
        /** Unmixing matrix, one row per component. */
        public function getComponents(): array {}
        /** Mixing matrix (features x components): x ~ mean + mixing * s. */
        public function getMixingMatrix(): array {}
        public function getIterations(): int {}
    }

    final class NMF implements \ML\Transformer, \ML\Stateful
    {
        public function __construct(int $dimensions = 2, int $maxIterations = 200, float $tolerance = 1.0E-4, ?int $seed = null) {}
        public function fit(array $samples): static {}
        public function transform(array $samples): array {}
        /** Returns the coefficients W found while fitting. */
        public function fitTransform(array $samples): array {}
        /** H: one non-negative row per component. */
        public function getComponents(): array {}
        public function getReconstructionError(): float {}
        public function getIterations(): int {}
    }

    final class Isomap implements \ML\Transformer, \ML\Stateful
    {
        public function __construct(int $dimensions = 2, int $neighbors = 5, string $metric = 'euclidean') {}
        public function fit(array $samples): static {}
        public function transform(array $samples): array {}
        public function fitTransform(array $samples): array {}
        public function getEigenvalues(): array {}
    }

    final class UMAP implements \ML\Transformer, \ML\Stateful
    {
        /** @param int|null $epochs null = 500 up to 10,000 samples, else 200 @param string $init pca or random */
        public function __construct(int $dimensions = 2, int $neighbors = 15, float $minDistance = 0.1, float $spread = 1.0, ?int $epochs = null, float $rate = 1.0, int $negativeSamples = 5, float $repulsion = 1.0, string $init = 'pca', string $metric = 'euclidean', ?int $seed = null) {}
        public function fit(array $samples): static {}
        /** Places new samples against the fixed training embedding. */
        public function transform(array $samples): array {}
        /** Returns the training embedding itself. */
        public function fitTransform(array $samples): array {}
        public function getEmbedding(): array {}
    }
}

namespace ML\Sequence {
    /**
     * Hidden Markov model over discrete symbols (int|string). A sequence is a list of
     * symbols; methods that take sequences also accept a list of them. Symbols never
     * seen in training carry no information.
     */
    final class HiddenMarkovModel
    {
        public function __construct(int $states = 2, int $maxIterations = 100, float $tolerance = 0.001, float $smoothing = 0.01, ?int $seed = null) {}
        /** Baum-Welch on unlabeled sequences; states are numbered 0 .. states - 1. */
        public function fit(array $sequences): static {}
        /** Counts start, transition and emission frequencies; the state labels come from $states. */
        public function fitLabeled(array $sequences, array $states): static {}
        /** Most likely state path (Viterbi). */
        public function predict(array $sequences): array {}
        /** Posterior state probabilities per step, keyed by state. */
        public function predictProbability(array $sequences): array {}
        /** Log-likelihood. */
        public function score(array $sequences): array|float {}
        /** @return array{observations: list<int|string>, states: list<int|string>} */
        public function sample(int $length, ?int $seed = null): array {}
        public function getStates(): array {}
        public function getSymbols(): array {}
        public function getStartProbabilities(): array {}
        public function getTransitionMatrix(): array {}
        public function getEmissionProbabilities(): array {}
        public function getLogLikelihood(): float {}
        public function getIterations(): int {}
    }

    /** Hidden Markov model with Gaussian (diagonal) emissions; a sequence is a list of observation vectors. */
    final class GaussianHMM
    {
        public function __construct(int $states = 2, int $maxIterations = 100, float $tolerance = 0.001, float $smoothing = 0.01, float $minVariance = 1.0E-6, ?int $seed = null) {}
        public function fit(array $sequences): static {}
        public function fitLabeled(array $sequences, array $states): static {}
        public function predict(array $sequences): array {}
        public function predictProbability(array $sequences): array {}
        public function score(array $sequences): array|float {}
        /** @return array{observations: list<list<float>>, states: list<int|string>} */
        public function sample(int $length, ?int $seed = null): array {}
        public function getStates(): array {}
        public function getStartProbabilities(): array {}
        public function getTransitionMatrix(): array {}
        public function getMeans(): array {}
        public function getVariances(): array {}
        public function getLogLikelihood(): float {}
        public function getIterations(): int {}
    }
}

namespace ML\Recommender {
    /**
     * Collaborative filtering. fit() takes [user, item] or [user, item, value] rows;
     * users and items are int|string labels. predict() takes one [user, item] pair or a
     * list of pairs; unknown users or items fall back to baselines.
     */
    interface Recommender {}

    final class MatrixFactorization implements Recommender
    {
        public function __construct(int $factors = 10, float $l2 = 0.1, int $epochs = 15, ?int $seed = null) {}
        public function fit(array $interactions): static {}
        public function predict(array $pairs): array|float {}
        /** @return array<int|string, float> item => score, best first */
        public function recommend(int|string $user, int $n = 10, bool $excludeSeen = true): array {}
        public function similarItems(int|string $item, int $n = 10): array {}
        public function getUsers(): array {}
        public function getItems(): array {}
        public function getUserFactors(): array {}
        public function getItemFactors(): array {}
        /** Training RMSE after each epoch. */
        public function getLosses(): array {}
    }

    final class ImplicitALS implements Recommender
    {
        public function __construct(int $factors = 10, float $l2 = 0.01, float $alpha = 40.0, int $epochs = 15, ?int $seed = null) {}
        public function fit(array $interactions): static {}
        public function predict(array $pairs): array|float {}
        public function recommend(int|string $user, int $n = 10, bool $excludeSeen = true): array {}
        public function similarItems(int|string $item, int $n = 10): array {}
        public function getUsers(): array {}
        public function getItems(): array {}
        public function getUserFactors(): array {}
        public function getItemFactors(): array {}
    }

    final class KNNRecommender implements Recommender
    {
        /** @param string $similarity cosine, pearson or jaccard @param string $basis item or user */
        public function __construct(int $neighbors = 20, string $similarity = 'cosine', string $basis = 'item', float $shrinkage = 0.0) {}
        public function fit(array $interactions): static {}
        public function predict(array $pairs): array|float {}
        public function recommend(int|string $user, int $n = 10, bool $excludeSeen = true): array {}
        public function similarItems(int|string $item, int $n = 10): array {}
        public function getUsers(): array {}
        public function getItems(): array {}
    }
}

namespace ML\Text {
    /**
     * Marker interface. The vectorizers call tokenize(string $text): array on
     * any object that implements it, so user-defined tokenizers must provide it.
     *
     * @method array tokenize(string $text)
     */
    interface Tokenizer {}

    final class WordTokenizer implements Tokenizer
    {
        public function tokenize(string $text): array {}
    }

    final class WhitespaceTokenizer implements Tokenizer
    {
        public function __construct(string $delimiter = ' ') {}
        public function tokenize(string $text): array {}
    }

    final class SentenceTokenizer implements Tokenizer
    {
        public function tokenize(string $text): array {}
    }

    final class NGramTokenizer implements Tokenizer
    {
        public function __construct(int $min = 2, int $max = 2) {}
        public function tokenize(string $text): array {}
    }

    final class KSkipNGramTokenizer implements Tokenizer
    {
        public function __construct(int $min = 2, int $max = 2, int $skip = 2) {}
        public function tokenize(string $text): array {}
    }

    final class WordStemmer implements Tokenizer
    {
        public function tokenize(string $text): array {}
    }

    final class PorterStemmer
    {
        public static function stem(string $word): string {}
    }

    final class TextNormalizer implements \ML\Transformer
    {
        public function __construct(bool $uppercase = false, bool $trim = true) {}
        /** @param list<string> $documents */
        public function transform(array $documents): array {}
    }

    final class MultibyteTextNormalizer implements \ML\Transformer
    {
        public function __construct(bool $uppercase = false, bool $trim = true) {}
        public function transform(array $documents): array {}
    }

    final class RegexFilter implements \ML\Transformer
    {
        public const URL = '/\b(?:https?:\/\/|www\.)\S+/iu';
        public const EMAIL = '/[\w.+-]+@[\w-]+\.[\w.-]+/u';
        public const MENTION = '/(?<!\w)@\w+/u';
        public const HASHTAG = '/(?<!\w)#\w+/u';
        public const NUMBER = '/\d+(?:[.,]\d+)*/u';
        public const PUNCTUATION = '/[\p{P}\p{S}]+/u';
        public const EMOJI = '/[\x{1F000}-\x{1FAFF}\x{2600}-\x{27BF}\x{FE0F}]/u';
        public function __construct(array $patterns) {}
        public function transform(array $documents): array {}
    }

    final class StopWordFilter implements \ML\Transformer
    {
        public function __construct(array $stopWords) {}
        public function transform(array $documents): array {}
    }

    final class WordCountVectorizer implements \ML\Transformer, \ML\Stateful
    {
        public function __construct(int $maxVocabularySize = PHP_INT_MAX, int $minDocumentCount = 1, float $maxDocumentRatio = 1.0, ?Tokenizer $tokenizer = null) {}
        public function fit(array $documents): static {}
        public function transform(array $documents): array {}
        public function fitTransform(array $documents): array {}
        public function getVocabulary(): array {}
    }

    final class TokenHashingVectorizer implements \ML\Transformer
    {
        /** @param string $hash murmur3, crc32 or fnv1a */
        public function __construct(int $dimensions = 1024, ?Tokenizer $tokenizer = null, string $hash = 'murmur3') {}
        public function transform(array $documents): array {}
    }

    final class TfIdfTransformer implements \ML\Transformer, \ML\Stateful
    {
        public function __construct(float $smoothing = 1.0, bool $dampening = false) {}
        public function fit(array $samples): static {}
        public function transform(array $samples): array {}
        public function fitTransform(array $samples): array {}
    }

    final class BM25Transformer implements \ML\Transformer, \ML\Stateful
    {
        public function __construct(float $dampening = 1.2, float $normalization = 0.75) {}
        public function fit(array $samples): static {}
        public function transform(array $samples): array {}
        public function fitTransform(array $samples): array {}
    }
}

namespace ML\NeuralNet {
    interface Layer {}
    interface Optimizer {}
    interface Schedule {}
}

namespace ML\NeuralNet\Layers {
    final class Dense implements \ML\NeuralNet\Layer
    {
        /** @param string $initializer he, lecun, normal(<stddev>), uniform(<beta>), xavier1, xavier2 or constant(<value>) */
        public function __construct(int $neurons, float $l2 = 0.0, bool $bias = true, string $initializer = 'he') {}
    }

    final class Activation implements \ML\NeuralNet\Layer
    {
        /** @param string $function relu, leaky_relu(<a>), elu(<a>), selu, gelu, sigmoid, tanh, softplus, softsign, silu, thresholded_relu(<t>) or linear */
        public function __construct(string $function = 'relu') {}
    }

    final class Dropout implements \ML\NeuralNet\Layer
    {
        public function __construct(float $ratio = 0.5) {}
    }

    final class BatchNorm implements \ML\NeuralNet\Layer
    {
        public function __construct(float $decay = 0.1) {}
    }

    final class PReLU implements \ML\NeuralNet\Layer
    {
        public function __construct(float $initial = 0.25) {}
    }

    final class Swish implements \ML\NeuralNet\Layer
    {
        public function __construct(float $initial = 1.0) {}
    }

    final class Noise implements \ML\NeuralNet\Layer
    {
        public function __construct(float $stddev = 0.1) {}
    }
}

namespace ML\NeuralNet\Optimizers {
    final class Stochastic implements \ML\NeuralNet\Optimizer
    {
        public function __construct(float $rate = 0.01) {}
    }

    final class Momentum implements \ML\NeuralNet\Optimizer
    {
        public function __construct(float $rate = 0.001, float $decay = 0.1, bool $lookahead = false) {}
    }

    final class AdaGrad implements \ML\NeuralNet\Optimizer
    {
        public function __construct(float $rate = 0.01) {}
    }

    final class RMSProp implements \ML\NeuralNet\Optimizer
    {
        public function __construct(float $rate = 0.001, float $decay = 0.1) {}
    }

    final class Adam implements \ML\NeuralNet\Optimizer
    {
        public function __construct(float $rate = 0.001, float $momentumDecay = 0.1, float $normDecay = 0.001) {}
    }

    final class AdaMax implements \ML\NeuralNet\Optimizer
    {
        public function __construct(float $rate = 0.001, float $momentumDecay = 0.1, float $normDecay = 0.001) {}
    }
}

namespace ML\NeuralNet\Schedules {
    final class Constant implements \ML\NeuralNet\Schedule {}

    final class StepDecay implements \ML\NeuralNet\Schedule
    {
        public function __construct(int $steps = 10, float $factor = 0.5) {}
    }

    final class Cosine implements \ML\NeuralNet\Schedule
    {
        public function __construct(int $period = 50, float $minimum = 0.0) {}
    }

    final class Cyclical implements \ML\NeuralNet\Schedule
    {
        public function __construct(int $steps = 10, float $minimum = 0.1) {}
    }

    final class Ramp implements \ML\NeuralNet\Schedule
    {
        public function __construct(int $warmup = 10) {}
    }
}
