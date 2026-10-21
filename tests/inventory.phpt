--TEST--
Inventory of every ML\ class and interface, and the interfaces each implements
--FILE--
<?php
// When this test changes, docs/DOCUMENTATION.md must change too.
$names = array_filter(array_merge(get_declared_classes(), get_declared_interfaces()), fn ($n) => str_starts_with($n, 'ML\\'));
sort($names);
$markers = ['ML\Learner', 'ML\Estimator', 'ML\Classifier', 'ML\Regressor', 'ML\Clusterer', 'ML\AnomalyDetector',
    'ML\Probabilistic', 'ML\Transformer', 'ML\Stateful', 'ML\RanksFeatures'];
foreach ($names as $name) {
    $r = new ReflectionClass($name);
    if ($r->isInterface()) {
        echo "interface $name\n";
        continue;
    }
    $tags = array_map(fn ($i) => substr($i, 3), array_values(array_intersect($markers, $r->getInterfaceNames())));
    sort($tags);
    echo $name, $tags ? ' [' . implode(', ', $tags) . ']' : '', "\n";
}
echo count($names), " names\n";
?>
--EXPECT--
ML\AnomalyDetection\GaussianMLE [AnomalyDetector, Estimator, Learner]
ML\AnomalyDetection\IsolationForest [AnomalyDetector, Estimator, Learner]
ML\AnomalyDetection\LocalOutlierFactor [AnomalyDetector, Estimator, Learner]
ML\AnomalyDetection\Loda [AnomalyDetector, Estimator, Learner]
ML\AnomalyDetection\OneClassSVM [AnomalyDetector, Estimator, Learner]
ML\AnomalyDetection\RobustZScore [AnomalyDetector, Estimator, Learner]
interface ML\AnomalyDetector
ML\Classification\AdaBoost [Classifier, Estimator, Learner, Probabilistic, RanksFeatures]
ML\Classification\ClassificationTree [Classifier, Estimator, Learner, Probabilistic, RanksFeatures]
ML\Classification\ExtraTreeClassifier [Classifier, Estimator, Learner, Probabilistic, RanksFeatures]
ML\Classification\GaussianNB [Classifier, Estimator, Learner, Probabilistic]
ML\Classification\KDNeighbors [Classifier, Estimator, Learner, Probabilistic]
ML\Classification\KNearestNeighbors [Classifier, Estimator, Learner, Probabilistic]
ML\Classification\LogisticRegression [Classifier, Estimator, Learner, Probabilistic]
ML\Classification\LogitBoost [Classifier, Estimator, Learner, Probabilistic, RanksFeatures]
ML\Classification\MLPClassifier [Classifier, Estimator, Learner, Probabilistic]
ML\Classification\MultinomialNB [Classifier, Estimator, Learner, Probabilistic]
ML\Classification\NaiveBayes [Classifier, Estimator, Learner, Probabilistic]
ML\Classification\RadiusNeighbors [Classifier, Estimator, Learner, Probabilistic]
ML\Classification\RandomForest [Classifier, Estimator, Learner, Probabilistic, RanksFeatures]
ML\Classification\SVC [Classifier, Estimator, Learner]
interface ML\Classifier
interface ML\Clusterer
ML\Clustering\DBSCAN [Clusterer, Estimator]
ML\Clustering\FuzzyCMeans [Clusterer, Estimator, Learner, Probabilistic]
ML\Clustering\GaussianMixture [Clusterer, Estimator, Learner, Probabilistic]
ML\Clustering\KMeans [Clusterer, Estimator, Learner, Probabilistic]
ML\Clustering\KMedoids [Clusterer, Estimator, Learner]
ML\Clustering\MeanShift [Clusterer, Estimator, Learner]
ML\Clustering\OPTICS [Clusterer, Estimator]
ML\Clustering\Ward [Clusterer, Estimator]
ML\CrossValidation\HoldOut
ML\CrossValidation\KFold
ML\CrossValidation\LeavePOut
ML\CrossValidation\MonteCarlo
ML\CrossValidation\Reports
interface ML\CrossValidation\Validator
ML\Datasets
ML\Distance
ML\Ensemble\BootstrapAggregator [Estimator, Learner]
ML\Ensemble\CommitteeMachine [Estimator, Learner]
ML\Ensemble\OneVsRest [Classifier, Estimator, Learner, Probabilistic]
interface ML\Estimator
ML\Exception\InvalidArgumentException
interface ML\Exception\MLException
ML\Exception\NotFittedException
ML\Exception\RuntimeException
interface ML\Learner
ML\Metrics
ML\ModelSelection\BayesianSearch [Estimator, Learner]
ML\ModelSelection\GridSearch [Estimator, Learner]
interface ML\NeuralNet\Layer
ML\NeuralNet\Layers\Activation
ML\NeuralNet\Layers\BatchNorm
ML\NeuralNet\Layers\Dense
ML\NeuralNet\Layers\Dropout
ML\NeuralNet\Layers\Noise
ML\NeuralNet\Layers\PReLU
ML\NeuralNet\Layers\Swish
interface ML\NeuralNet\Optimizer
ML\NeuralNet\Optimizers\AdaGrad
ML\NeuralNet\Optimizers\AdaMax
ML\NeuralNet\Optimizers\Adam
ML\NeuralNet\Optimizers\Momentum
ML\NeuralNet\Optimizers\RMSProp
ML\NeuralNet\Optimizers\Stochastic
interface ML\NeuralNet\Schedule
ML\NeuralNet\Schedules\Constant
ML\NeuralNet\Schedules\Cosine
ML\NeuralNet\Schedules\Cyclical
ML\NeuralNet\Schedules\Ramp
ML\NeuralNet\Schedules\StepDecay
ML\Pipeline [Estimator, Learner]
ML\Preprocessing\BooleanConverter [Transformer]
ML\Preprocessing\GaussianRandomProjector [Stateful, Transformer]
ML\Preprocessing\HotDeckImputer [Stateful, Transformer]
ML\Preprocessing\IntervalDiscretizer [Stateful, Transformer]
ML\Preprocessing\KNNImputer [Stateful, Transformer]
ML\Preprocessing\L1Normalizer [Transformer]
ML\Preprocessing\L2Normalizer [Transformer]
ML\Preprocessing\LambdaFunction [Transformer]
ML\Preprocessing\LinearDiscriminantAnalysis [Stateful, Transformer]
ML\Preprocessing\MaxAbsoluteScaler [Stateful, Transformer]
ML\Preprocessing\MinMaxNormalizer [Stateful, Transformer]
ML\Preprocessing\MissingDataImputer [Stateful, Transformer]
ML\Preprocessing\NumericStringConverter [Transformer]
ML\Preprocessing\OneHotEncoder [Stateful, Transformer]
ML\Preprocessing\PCA [Stateful, Transformer]
ML\Preprocessing\PolynomialExpander [Transformer]
ML\Preprocessing\PowerTransformer [Stateful, Transformer]
ML\Preprocessing\RobustStandardizer [Stateful, Transformer]
ML\Preprocessing\SparseRandomProjector [Stateful, Transformer]
ML\Preprocessing\StandardScaler [Stateful, Transformer]
ML\Preprocessing\TSNE [Transformer]
ML\Preprocessing\TruncatedSVD [Stateful, Transformer]
interface ML\Probabilistic
interface ML\RanksFeatures
ML\Regression\Adaline [Estimator, Learner, Regressor]
ML\Regression\ElasticNet [Estimator, Learner, Regressor]
ML\Regression\ExtraTreeRegressor [Estimator, Learner, RanksFeatures, Regressor]
ML\Regression\GradientBoost [Estimator, Learner, RanksFeatures, Regressor]
ML\Regression\KDNeighborsRegressor [Estimator, Learner, Regressor]
ML\Regression\KNNRegressor [Estimator, Learner, Regressor]
ML\Regression\LinearRegression [Estimator, Learner, Regressor]
ML\Regression\MLPRegressor [Estimator, Learner, Regressor]
ML\Regression\RadiusNeighborsRegressor [Estimator, Learner, Regressor]
ML\Regression\RandomForestRegressor [Estimator, Learner, RanksFeatures, Regressor]
ML\Regression\RegressionTree [Estimator, Learner, RanksFeatures, Regressor]
ML\Regression\SVR [Estimator, Learner, Regressor]
interface ML\Regressor
interface ML\Stateful
ML\Text\BM25Transformer [Stateful, Transformer]
ML\Text\KSkipNGramTokenizer
ML\Text\MultibyteTextNormalizer [Transformer]
ML\Text\NGramTokenizer
ML\Text\PorterStemmer
ML\Text\RegexFilter [Transformer]
ML\Text\SentenceTokenizer
ML\Text\StopWordFilter [Transformer]
ML\Text\TextNormalizer [Transformer]
ML\Text\TfIdfTransformer [Stateful, Transformer]
ML\Text\TokenHashingVectorizer [Transformer]
interface ML\Text\Tokenizer
ML\Text\WhitespaceTokenizer
ML\Text\WordCountVectorizer [Stateful, Transformer]
ML\Text\WordStemmer
ML\Text\WordTokenizer
interface ML\Transformer
128 names
