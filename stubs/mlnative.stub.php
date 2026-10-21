<?php

/**
 * IDE stub for the mlnative extension. Not loaded at runtime: the classes are
 * implemented in C++ (src/). Keep in sync with the arginfo in src/.
 *
 * Samples are arrays of int|float features. Methods that take `$samples`
 * accept either a list of samples (returns one result per sample) or a single
 * sample (returns a single result).
 */

namespace ML\Exception {
    interface MLException {}

    class InvalidArgumentException extends \InvalidArgumentException implements MLException {}

    class RuntimeException extends \RuntimeException implements MLException {}

    class NotFittedException extends \LogicException implements MLException {}
}

namespace ML\Regression {
    final class LinearRegression
    {
        /** @param float $l2 Ridge penalty; 0 is ordinary least squares. */
        public function __construct(float $l2 = 0.0) {}

        /**
         * @param list<array<int|float>> $samples
         * @param list<int|float> $targets
         */
        public function fit(array $samples, array $targets): static {}

        /** @return float|list<float> */
        public function predict(array $samples): array|float {}

        /** @return list<float> */
        public function getCoefficients(): array {}

        public function getIntercept(): float {}

        public function __serialize(): array {}

        public function __unserialize(array $data): void {}
    }
}

namespace ML\Classification {
    final class LogisticRegression
    {
        public function __construct(float $learningRate = 0.1, int $maxIterations = 1000, float $l2 = 0.0, float $tolerance = 1.0E-6) {}

        /**
         * @param list<array<int|float>> $samples
         * @param list<int|string> $targets
         */
        public function fit(array $samples, array $targets): static {}

        /** @return int|string|list<int|string> */
        public function predict(array $samples): array|int|string {}

        /** @return array<int|string, float>|list<array<int|string, float>> label => probability */
        public function predictProbability(array $samples): array {}

        /** @return list<int|string> */
        public function getClasses(): array {}

        /** @return array<int|string, list<float>> label => feature weights */
        public function getWeights(): array {}

        public function __serialize(): array {}

        public function __unserialize(array $data): void {}
    }

    final class KNearestNeighbors
    {
        /** @param 'euclidean'|'manhattan' $metric */
        public function __construct(int $k = 3, string $metric = 'euclidean') {}

        /**
         * @param list<array<int|float>> $samples
         * @param list<int|string> $targets
         */
        public function fit(array $samples, array $targets): static {}

        /** @return int|string|list<int|string> */
        public function predict(array $samples): array|int|string {}

        /** @return list<int|string> */
        public function getClasses(): array {}

        public function __serialize(): array {}

        public function __unserialize(array $data): void {}
    }

    final class GaussianNB
    {
        public function __construct(float $varSmoothing = 1.0E-9) {}

        /**
         * @param list<array<int|float>> $samples
         * @param list<int|string> $targets
         */
        public function fit(array $samples, array $targets): static {}

        /** @return int|string|list<int|string> */
        public function predict(array $samples): array|int|string {}

        /** @return array<int|string, float>|list<array<int|string, float>> label => probability */
        public function predictProbability(array $samples): array {}

        /** @return list<int|string> */
        public function getClasses(): array {}

        public function __serialize(): array {}

        public function __unserialize(array $data): void {}
    }
}

namespace ML\Clustering {
    final class KMeans
    {
        /** @param int|null $seed Fixed seed for reproducible k-means++ initialization. */
        public function __construct(int $clusters = 8, int $maxIterations = 300, float $tolerance = 1.0E-4, ?int $seed = null) {}

        /** @param list<array<int|float>> $samples */
        public function fit(array $samples): static {}

        /** @return int|list<int> cluster index */
        public function predict(array $samples): array|int {}

        /** @return list<list<float>> */
        public function getCentroids(): array {}

        /** @return list<int> cluster of each training sample */
        public function getLabels(): array {}

        public function getInertia(): float {}

        public function getIterations(): int {}

        public function __serialize(): array {}

        public function __unserialize(array $data): void {}
    }
}

namespace ML\Preprocessing {
    final class StandardScaler
    {
        /** @param list<array<int|float>> $samples */
        public function fit(array $samples): static {}

        /** @return list<float>|list<list<float>> */
        public function transform(array $samples): array {}

        /** @return list<list<float>> */
        public function fitTransform(array $samples): array {}

        /** @return list<float> */
        public function getMeans(): array {}

        /** @return list<float> */
        public function getScales(): array {}

        public function __serialize(): array {}

        public function __unserialize(array $data): void {}
    }
}

namespace ML {
    final class Metrics
    {
        private function __construct() {}

        /** Share of positions where expected === predicted. */
        public static function accuracy(array $expected, array $predicted): float {}

        public static function meanSquaredError(array $expected, array $predicted): float {}

        public static function meanAbsoluteError(array $expected, array $predicted): float {}

        public static function r2Score(array $expected, array $predicted): float {}
    }
}
