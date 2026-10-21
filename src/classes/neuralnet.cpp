// Neural networks: MLPClassifier, MLPRegressor and Adaline, configured with
// layer (ML\NeuralNet\Layers\*), optimizer (ML\NeuralNet\Optimizers\*) and
// learning-rate schedule (ML\NeuralNet\Schedules\*) objects.

#include <map>

#include "binding.hpp"
#include "core/neuralnet.hpp"

namespace {

using namespace mlphp;
using mlcore::ArchiveReader;
using mlcore::ArchiveWriter;
using mlcore::LayerSpec;
using mlcore::OptimizerSpec;
using mlcore::ScheduleSpec;

zend_class_entry *ce_layer, *ce_optimizer, *ce_schedule;

// Configuration objects share one native type each; the class decides the kind.
struct LayerModel {
    static constexpr std::uint32_t kTag = mlcore::tag("NLAY");
    LayerSpec spec;
    void save(ArchiveWriter &w) const { spec.save(w); }
    void load(ArchiveReader &r) { spec.load(r); }
};
struct OptimizerModel {
    static constexpr std::uint32_t kTag = mlcore::tag("NOPT");
    OptimizerSpec spec;
    void save(ArchiveWriter &w) const { spec.save(w); }
    void load(ArchiveReader &r) { spec.load(r); }
};
struct ScheduleModel {
    static constexpr std::uint32_t kTag = mlcore::tag("NSCH");
    ScheduleSpec spec;
    void save(ArchiveWriter &w) const { spec.save(w); }
    void load(ArchiveReader &r) { spec.load(r); }
};

std::map<zend_class_entry *, int> config_kinds;  // class -> kind enum value

template <typename M>
void ZEND_FASTCALL config_unserialize(INTERNAL_FUNCTION_PARAMETERS) {
    HashTable *data;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ARRAY_HT(data)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        zval *state = require_key(data, "state");
        check(Z_TYPE_P(state) == IS_STRING, "Invalid serialized data: \"state\" must be a string");
        M restored;
        ArchiveReader r(Z_STRVAL_P(state), Z_STRLEN_P(state), M::kTag);
        restored.load(r);
        r.finish();
        check(static_cast<int>(restored.spec.kind) == config_kinds.at(Z_OBJCE_P(ZEND_THIS)), "Invalid serialized data: kind mismatch");
        reset(ZEND_THIS, std::move(restored));
    });
}

// ---- layers ---------------------------------------------------------------------------------

ZEND_BEGIN_ARG_INFO_EX(arginfo_dense_construct, 0, 0, 1)
    ZEND_ARG_TYPE_INFO(0, neurons, IS_LONG, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, l2, IS_DOUBLE, 0, "0.0")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, bias, _IS_BOOL, 0, "true")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, initializer, IS_STRING, 0, "\"he\"")
ZEND_END_ARG_INFO()

void ZEND_FASTCALL dense_construct(INTERNAL_FUNCTION_PARAMETERS) {
    zend_long neurons;
    double l2 = 0.0;
    bool bias = true;
    zend_string *init = nullptr;
    ZEND_PARSE_PARAMETERS_START(1, 4)
        Z_PARAM_LONG(neurons)
        Z_PARAM_OPTIONAL
        Z_PARAM_DOUBLE(l2)
        Z_PARAM_BOOL(bias)
        Z_PARAM_STR(init)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        LayerModel m;
        m.spec.kind = LayerSpec::Kind::Dense;
        m.spec.neurons = positive(neurons, "neurons");
        check(m.spec.neurons <= (1u << 20), "neurons must be at most 1048576");
        m.spec.l2 = non_negative(l2, "l2");
        m.spec.bias = bias;
        if (init) m.spec.init = mlcore::Initializer::parse(zstr(init));
        reset(ZEND_THIS, m);
    });
}

ZEND_BEGIN_ARG_INFO_EX(arginfo_activation_construct, 0, 0, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, function, IS_STRING, 0, "\"relu\"")
ZEND_END_ARG_INFO()

void ZEND_FASTCALL activation_construct(INTERNAL_FUNCTION_PARAMETERS) {
    zend_string *fn = nullptr;
    ZEND_PARSE_PARAMETERS_START(0, 1)
        Z_PARAM_OPTIONAL
        Z_PARAM_STR(fn)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        LayerModel m;
        m.spec.kind = LayerSpec::Kind::Activation;
        m.spec.activation = mlcore::ActivationFn::parse(fn ? zstr(fn) : "relu");
        reset(ZEND_THIS, m);
    });
}

template <LayerSpec::Kind K>
void ZEND_FASTCALL scalar_layer_construct(INTERNAL_FUNCTION_PARAMETERS) {
    constexpr double fallback = K == LayerSpec::Kind::Dropout ? 0.5 : K == LayerSpec::Kind::BatchNorm ? 0.1
                              : K == LayerSpec::Kind::PReLU ? 0.25 : K == LayerSpec::Kind::Swish ? 1.0 : 0.1;
    double value = fallback;
    ZEND_PARSE_PARAMETERS_START(0, 1)
        Z_PARAM_OPTIONAL
        Z_PARAM_DOUBLE(value)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        LayerModel m;
        m.spec.kind = K;
        switch (K) {
            case LayerSpec::Kind::Dropout:
                check(value >= 0.0 && value < 1.0, "ratio must be at least 0 and less than 1");
                m.spec.ratio = value;
                break;
            case LayerSpec::Kind::BatchNorm:
                check(value > 0.0 && value <= 1.0, "decay must be greater than 0 and at most 1");
                m.spec.decay = value;
                break;
            case LayerSpec::Kind::Noise:
                m.spec.stddev = non_negative(value, "stddev");
                break;
            default:
                check(std::isfinite(value), "initial must be finite");
                m.spec.initial = value;
        }
        reset(ZEND_THIS, m);
    });
}

#define ML_CONFIG(M, ctor, arginfo)                                            \
    ML_METHOD(__construct, ctor, arginfo)                                       \
    ML_METHOD(__serialize, serialize_native<M>, arginfo_ml_get_array)           \
    ML_METHOD(__unserialize, config_unserialize<M>, arginfo_ml_unserialize)

ZEND_BEGIN_ARG_INFO_EX(arginfo_ratio, 0, 0, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, ratio, IS_DOUBLE, 0, "0.5")
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_INFO_EX(arginfo_decay_bn, 0, 0, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, decay, IS_DOUBLE, 0, "0.1")
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_INFO_EX(arginfo_prelu, 0, 0, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, initial, IS_DOUBLE, 0, "0.25")
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_INFO_EX(arginfo_swish, 0, 0, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, initial, IS_DOUBLE, 0, "1.0")
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_INFO_EX(arginfo_noise, 0, 0, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, stddev, IS_DOUBLE, 0, "0.1")
ZEND_END_ARG_INFO()

const zend_function_entry dense_methods[] = {ML_CONFIG(LayerModel, dense_construct, arginfo_dense_construct) PHP_FE_END};
const zend_function_entry activation_methods[] = {ML_CONFIG(LayerModel, activation_construct, arginfo_activation_construct) PHP_FE_END};
const zend_function_entry dropout_methods[] = {ML_CONFIG(LayerModel, scalar_layer_construct<LayerSpec::Kind::Dropout>, arginfo_ratio) PHP_FE_END};
const zend_function_entry batch_norm_methods[] = {ML_CONFIG(LayerModel, scalar_layer_construct<LayerSpec::Kind::BatchNorm>, arginfo_decay_bn) PHP_FE_END};
const zend_function_entry prelu_methods[] = {ML_CONFIG(LayerModel, scalar_layer_construct<LayerSpec::Kind::PReLU>, arginfo_prelu) PHP_FE_END};
const zend_function_entry swish_methods[] = {ML_CONFIG(LayerModel, scalar_layer_construct<LayerSpec::Kind::Swish>, arginfo_swish) PHP_FE_END};
const zend_function_entry noise_methods[] = {ML_CONFIG(LayerModel, scalar_layer_construct<LayerSpec::Kind::Noise>, arginfo_noise) PHP_FE_END};

// ---- optimizers -------------------------------------------------------------------------------

ZEND_BEGIN_ARG_INFO_EX(arginfo_rate_only, 0, 0, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, rate, IS_DOUBLE, 0, "0.01")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_momentum, 0, 0, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, rate, IS_DOUBLE, 0, "0.001")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, decay, IS_DOUBLE, 0, "0.1")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, lookahead, _IS_BOOL, 0, "false")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_rmsprop, 0, 0, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, rate, IS_DOUBLE, 0, "0.001")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, decay, IS_DOUBLE, 0, "0.1")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_adam, 0, 0, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, rate, IS_DOUBLE, 0, "0.001")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, momentumDecay, IS_DOUBLE, 0, "0.1")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, normDecay, IS_DOUBLE, 0, "0.001")
ZEND_END_ARG_INFO()

template <OptimizerSpec::Kind K>
void ZEND_FASTCALL optimizer_construct(INTERNAL_FUNCTION_PARAMETERS) {
    const bool sgd_like = K == OptimizerSpec::Kind::Stochastic || K == OptimizerSpec::Kind::AdaGrad;
    double rate = sgd_like ? 0.01 : 0.001, a = 0.1, b = 0.001;
    bool lookahead = false;
    const uint32_t max_args = sgd_like ? 1 : (K == OptimizerSpec::Kind::RMSProp ? 2 : 3);
    ZEND_PARSE_PARAMETERS_START(0, max_args)
        Z_PARAM_OPTIONAL
        Z_PARAM_DOUBLE(rate)
        Z_PARAM_DOUBLE(a)
        if (K == OptimizerSpec::Kind::Momentum) {
            Z_PARAM_BOOL(lookahead)
        } else {
            Z_PARAM_DOUBLE(b)
        }
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        OptimizerModel m;
        m.spec.kind = K;
        m.spec.rate = strictly_positive(rate, "rate");
        if (K == OptimizerSpec::Kind::Momentum || K == OptimizerSpec::Kind::RMSProp) {
            check(a > 0.0 && a < 1.0, "decay must be between 0 and 1 (exclusive)");
            m.spec.decay = a;
            m.spec.lookahead = lookahead;
        } else if (K == OptimizerSpec::Kind::Adam || K == OptimizerSpec::Kind::AdaMax) {
            check(a > 0.0 && a < 1.0, "momentumDecay must be between 0 and 1 (exclusive)");
            check(b > 0.0 && b < 1.0, "normDecay must be between 0 and 1 (exclusive)");
            m.spec.momentum_decay = a;
            m.spec.norm_decay = b;
        }
        reset(ZEND_THIS, m);
    });
}

ZEND_BEGIN_ARG_INFO_EX(arginfo_adagrad, 0, 0, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, rate, IS_DOUBLE, 0, "0.01")
ZEND_END_ARG_INFO()

const zend_function_entry stochastic_methods[] = {ML_CONFIG(OptimizerModel, optimizer_construct<OptimizerSpec::Kind::Stochastic>, arginfo_rate_only) PHP_FE_END};
const zend_function_entry momentum_methods[] = {ML_CONFIG(OptimizerModel, optimizer_construct<OptimizerSpec::Kind::Momentum>, arginfo_momentum) PHP_FE_END};
const zend_function_entry adagrad_methods[] = {ML_CONFIG(OptimizerModel, optimizer_construct<OptimizerSpec::Kind::AdaGrad>, arginfo_adagrad) PHP_FE_END};
const zend_function_entry rmsprop_methods[] = {ML_CONFIG(OptimizerModel, optimizer_construct<OptimizerSpec::Kind::RMSProp>, arginfo_rmsprop) PHP_FE_END};
const zend_function_entry adam_methods[] = {ML_CONFIG(OptimizerModel, optimizer_construct<OptimizerSpec::Kind::Adam>, arginfo_adam) PHP_FE_END};
const zend_function_entry adamax_methods[] = {ML_CONFIG(OptimizerModel, optimizer_construct<OptimizerSpec::Kind::AdaMax>, arginfo_adam) PHP_FE_END};

// ---- schedules ---------------------------------------------------------------------------------

ZEND_BEGIN_ARG_INFO_EX(arginfo_step_decay, 0, 0, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, steps, IS_LONG, 0, "10")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, factor, IS_DOUBLE, 0, "0.5")
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_INFO_EX(arginfo_cosine, 0, 0, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, period, IS_LONG, 0, "50")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, minimum, IS_DOUBLE, 0, "0.0")
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_INFO_EX(arginfo_cyclical, 0, 0, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, steps, IS_LONG, 0, "10")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, minimum, IS_DOUBLE, 0, "0.1")
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_INFO_EX(arginfo_ramp, 0, 0, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, warmup, IS_LONG, 0, "10")
ZEND_END_ARG_INFO()

template <ScheduleSpec::Kind K>
void ZEND_FASTCALL schedule_construct(INTERNAL_FUNCTION_PARAMETERS) {
    zend_long steps = K == ScheduleSpec::Kind::Cosine ? 50 : 10;
    double factor = K == ScheduleSpec::Kind::Cosine ? 0.0 : K == ScheduleSpec::Kind::Cyclical ? 0.1 : 0.5;
    const uint32_t max_args = K == ScheduleSpec::Kind::Constant ? 0 : K == ScheduleSpec::Kind::Ramp ? 1 : 2;
    ZEND_PARSE_PARAMETERS_START(0, max_args)
        Z_PARAM_OPTIONAL
        Z_PARAM_LONG(steps)
        Z_PARAM_DOUBLE(factor)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        ScheduleModel m;
        m.spec.kind = K;
        m.spec.steps = positive(steps, "steps");
        check(factor >= 0.0 && factor <= 1.0, "factor/minimum must be between 0 and 1");
        m.spec.factor = factor;
        reset(ZEND_THIS, m);
    });
}

const zend_function_entry constant_methods[] = {ML_CONFIG(ScheduleModel, schedule_construct<ScheduleSpec::Kind::Constant>, arginfo_ml_none) PHP_FE_END};
const zend_function_entry step_decay_methods[] = {ML_CONFIG(ScheduleModel, schedule_construct<ScheduleSpec::Kind::StepDecay>, arginfo_step_decay) PHP_FE_END};
const zend_function_entry cosine_methods[] = {ML_CONFIG(ScheduleModel, schedule_construct<ScheduleSpec::Kind::Cosine>, arginfo_cosine) PHP_FE_END};
const zend_function_entry cyclical_methods[] = {ML_CONFIG(ScheduleModel, schedule_construct<ScheduleSpec::Kind::Cyclical>, arginfo_cyclical) PHP_FE_END};
const zend_function_entry ramp_methods[] = {ML_CONFIG(ScheduleModel, schedule_construct<ScheduleSpec::Kind::Ramp>, arginfo_ramp) PHP_FE_END};

// ---- networks --------------------------------------------------------------------------------------

template <bool Classification, std::uint32_t Tag>
struct NetworkModel {
    static constexpr std::uint32_t kTag = Tag;
    mlcore::MLP core;
    LabelEncoder labels;

    NetworkModel() {
        core.classification = Classification;
        core.cost = mlcore::CostSpec::parse(Classification ? "cross_entropy" : "least_squares", Classification);
    }
    bool fitted() const { return core.fitted(); }
    std::size_t features() const { return core.features; }
    std::size_t classes() const { return core.outputs; }
    void fit(const mlcore::Matrix &X, const std::vector<int> &y, std::size_t K, unsigned) {
        core.fit(X, std::vector<double>(y.begin(), y.end()), K);
    }
    void fit(const mlcore::Matrix &X, const std::vector<double> &y, unsigned) { core.fit(X, y, 1); }
    void probabilities(const double *x, double *out) const { core.predict_raw(x, out); }
    auto predict_one(const double *x) const {
        if constexpr (Classification) {
            std::vector<double> p(core.outputs);
            core.predict_raw(x, p.data());
            return static_cast<int>(mlcore::argmax(p.data(), p.size()));
        } else {
            double v;
            core.predict_raw(x, &v);
            return v;
        }
    }
    void save(ArchiveWriter &w) const {
        core.save(w);
        labels.save(w);
    }
    void load(ArchiveReader &r) {
        core.load(r);
        labels.load(r);
        ArchiveReader::require(core.classification == Classification, "network kind mismatch");
        ArchiveReader::require(!Classification || !core.fitted() || labels.size() == core.outputs, "label count mismatch");
    }
};

using MLPClassifierModel = NetworkModel<true, mlcore::tag("MLPC")>;
using MLPRegressorModel = NetworkModel<false, mlcore::tag("MLPR")>;
using AdalineModel = NetworkModel<false, mlcore::tag("ADAL")>;

std::vector<LayerSpec> read_layers(HashTable *layers) {
    std::vector<LayerSpec> out;
    for (zval *layer : values_of(layers)) {
        require_instance(layer, ce_layer, "Each hidden layer");
        out.push_back(native<LayerModel>(layer).spec);
    }
    return out;
}

void common_network(mlcore::MLP &core, zend_long batch_size, zval *optimizer, zend_long epochs, double min_change,
                    zend_long window, double hold_out, double l2, zval *schedule, zend_long seed, bool seed_null) {
    core.batch_size = positive(batch_size, "batchSize");
    if (optimizer && Z_TYPE_P(optimizer) == IS_OBJECT) {
        core.optimizer = native<OptimizerModel>(optimizer).spec;
    }
    core.epochs = positive(epochs, "epochs");
    core.min_change = non_negative(min_change, "minChange");
    core.window = positive(window, "window");
    check(hold_out >= 0.0 && hold_out < 0.5, "holdOut must be at least 0 and less than 0.5");
    core.hold_out = hold_out;
    core.l2 = non_negative(l2, "l2");
    if (schedule && Z_TYPE_P(schedule) == IS_OBJECT) {
        core.schedule = native<ScheduleModel>(schedule).spec;
    }
    core.has_seed = !seed_null;
    core.seed = static_cast<std::uint64_t>(seed);
}

ZEND_BEGIN_ARG_INFO_EX(arginfo_mlp_classifier_construct, 0, 0, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, hiddenLayers, IS_ARRAY, 0, "[]")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, batchSize, IS_LONG, 0, "128")
    ZEND_ARG_OBJ_INFO_WITH_DEFAULT_VALUE(0, optimizer, ML\\NeuralNet\\Optimizer, 1, "null")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, epochs, IS_LONG, 0, "1000")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, minChange, IS_DOUBLE, 0, "1.0E-4")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, window, IS_LONG, 0, "5")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, holdOut, IS_DOUBLE, 0, "0.1")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, l2, IS_DOUBLE, 0, "0.0")
    ZEND_ARG_OBJ_INFO_WITH_DEFAULT_VALUE(0, schedule, ML\\NeuralNet\\Schedule, 1, "null")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, seed, IS_LONG, 1, "null")
ZEND_END_ARG_INFO()

void ZEND_FASTCALL mlp_classifier_construct(INTERNAL_FUNCTION_PARAMETERS) {
    HashTable *hidden = nullptr;
    zend_long batch = 128, epochs = 1000, window = 5, seed = 0;
    zval *optimizer = nullptr, *schedule = nullptr;
    double min_change = 1e-4, hold_out = 0.1, l2 = 0.0;
    bool seed_null = true;
    ZEND_PARSE_PARAMETERS_START(0, 10)
        Z_PARAM_OPTIONAL
        Z_PARAM_ARRAY_HT(hidden)
        Z_PARAM_LONG(batch)
        Z_PARAM_OBJECT_OF_CLASS_OR_NULL(optimizer, ce_optimizer)
        Z_PARAM_LONG(epochs)
        Z_PARAM_DOUBLE(min_change)
        Z_PARAM_LONG(window)
        Z_PARAM_DOUBLE(hold_out)
        Z_PARAM_DOUBLE(l2)
        Z_PARAM_OBJECT_OF_CLASS_OR_NULL(schedule, ce_schedule)
        Z_PARAM_LONG_OR_NULL(seed, seed_null)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        MLPClassifierModel fresh;
        if (hidden) fresh.core.hidden = read_layers(hidden);
        common_network(fresh.core, batch, optimizer, epochs, min_change, window, hold_out, l2, schedule, seed, seed_null);
        reset(ZEND_THIS, std::move(fresh));
    });
}

ZEND_BEGIN_ARG_INFO_EX(arginfo_mlp_regressor_construct, 0, 0, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, hiddenLayers, IS_ARRAY, 0, "[]")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, batchSize, IS_LONG, 0, "128")
    ZEND_ARG_OBJ_INFO_WITH_DEFAULT_VALUE(0, optimizer, ML\\NeuralNet\\Optimizer, 1, "null")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, epochs, IS_LONG, 0, "1000")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, minChange, IS_DOUBLE, 0, "1.0E-4")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, window, IS_LONG, 0, "5")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, holdOut, IS_DOUBLE, 0, "0.1")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, l2, IS_DOUBLE, 0, "0.0")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, costFn, IS_STRING, 0, "\"least_squares\"")
    ZEND_ARG_OBJ_INFO_WITH_DEFAULT_VALUE(0, schedule, ML\\NeuralNet\\Schedule, 1, "null")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, seed, IS_LONG, 1, "null")
ZEND_END_ARG_INFO()

void ZEND_FASTCALL mlp_regressor_construct(INTERNAL_FUNCTION_PARAMETERS) {
    HashTable *hidden = nullptr;
    zend_long batch = 128, epochs = 1000, window = 5, seed = 0;
    zval *optimizer = nullptr, *schedule = nullptr;
    double min_change = 1e-4, hold_out = 0.1, l2 = 0.0;
    zend_string *cost = nullptr;
    bool seed_null = true;
    ZEND_PARSE_PARAMETERS_START(0, 11)
        Z_PARAM_OPTIONAL
        Z_PARAM_ARRAY_HT(hidden)
        Z_PARAM_LONG(batch)
        Z_PARAM_OBJECT_OF_CLASS_OR_NULL(optimizer, ce_optimizer)
        Z_PARAM_LONG(epochs)
        Z_PARAM_DOUBLE(min_change)
        Z_PARAM_LONG(window)
        Z_PARAM_DOUBLE(hold_out)
        Z_PARAM_DOUBLE(l2)
        Z_PARAM_STR(cost)
        Z_PARAM_OBJECT_OF_CLASS_OR_NULL(schedule, ce_schedule)
        Z_PARAM_LONG_OR_NULL(seed, seed_null)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        MLPRegressorModel fresh;
        if (hidden) fresh.core.hidden = read_layers(hidden);
        common_network(fresh.core, batch, optimizer, epochs, min_change, window, hold_out, l2, schedule, seed, seed_null);
        if (cost) fresh.core.cost = mlcore::CostSpec::parse(zstr(cost), false);
        reset(ZEND_THIS, std::move(fresh));
    });
}

ZEND_BEGIN_ARG_INFO_EX(arginfo_adaline_construct, 0, 0, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, batchSize, IS_LONG, 0, "128")
    ZEND_ARG_OBJ_INFO_WITH_DEFAULT_VALUE(0, optimizer, ML\\NeuralNet\\Optimizer, 1, "null")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, l2, IS_DOUBLE, 0, "1.0E-4")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, epochs, IS_LONG, 0, "1000")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, minChange, IS_DOUBLE, 0, "1.0E-4")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, window, IS_LONG, 0, "5")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, holdOut, IS_DOUBLE, 0, "0.1")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, costFn, IS_STRING, 0, "\"least_squares\"")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, seed, IS_LONG, 1, "null")
ZEND_END_ARG_INFO()

void ZEND_FASTCALL adaline_construct(INTERNAL_FUNCTION_PARAMETERS) {
    zend_long batch = 128, epochs = 1000, window = 5, seed = 0;
    zval *optimizer = nullptr;
    double l2 = 1e-4, min_change = 1e-4, hold_out = 0.1;
    zend_string *cost = nullptr;
    bool seed_null = true;
    ZEND_PARSE_PARAMETERS_START(0, 9)
        Z_PARAM_OPTIONAL
        Z_PARAM_LONG(batch)
        Z_PARAM_OBJECT_OF_CLASS_OR_NULL(optimizer, ce_optimizer)
        Z_PARAM_DOUBLE(l2)
        Z_PARAM_LONG(epochs)
        Z_PARAM_DOUBLE(min_change)
        Z_PARAM_LONG(window)
        Z_PARAM_DOUBLE(hold_out)
        Z_PARAM_STR(cost)
        Z_PARAM_LONG_OR_NULL(seed, seed_null)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        AdalineModel fresh;
        common_network(fresh.core, batch, optimizer, epochs, min_change, window, hold_out, l2, nullptr, seed, seed_null);
        if (cost) fresh.core.cost = mlcore::CostSpec::parse(zstr(cost), false);
        reset(ZEND_THIS, std::move(fresh));
    });
}

template <typename M, bool Scores>
void ZEND_FASTCALL network_curve(INTERNAL_FUNCTION_PARAMETERS) {
    ZEND_PARSE_PARAMETERS_NONE();
    guard([&] {
        const M &m = native<M>(ZEND_THIS);
        require_fitted(m.fitted(), ZEND_THIS);
        write_doubles(return_value, Scores ? m.core.scores : m.core.losses);
    });
}

#define ML_NETWORK_COMMON(M)                                                         \
    ML_METHOD(getLosses, (network_curve<M, false>), arginfo_ml_get_array)             \
    ML_METHOD(getScores, (network_curve<M, true>), arginfo_ml_get_array)              \
    ML_METHOD(__serialize, serialize_native<M>, arginfo_ml_get_array)                 \
    ML_METHOD(__unserialize, unserialize_native<M>, arginfo_ml_unserialize)

const zend_function_entry mlp_classifier_methods[] = {
    ML_METHOD(__construct, mlp_classifier_construct, arginfo_mlp_classifier_construct)
    ML_METHOD(fit, fit_classifier<MLPClassifierModel>, arginfo_ml_fit_supervised)
    ML_METHOD(predict, predict_classifier<MLPClassifierModel>, arginfo_ml_predict_label)
    ML_METHOD(predictProbability, predict_proba<MLPClassifierModel>, arginfo_ml_samples_to_array)
    ML_METHOD(getClasses, get_classes<MLPClassifierModel>, arginfo_ml_get_array)
    ML_NETWORK_COMMON(MLPClassifierModel)
    PHP_FE_END};

const zend_function_entry mlp_regressor_methods[] = {
    ML_METHOD(__construct, mlp_regressor_construct, arginfo_mlp_regressor_construct)
    ML_METHOD(fit, fit_regressor<MLPRegressorModel>, arginfo_ml_fit_supervised)
    ML_METHOD(predict, predict_regressor<MLPRegressorModel>, arginfo_ml_predict_float)
    ML_NETWORK_COMMON(MLPRegressorModel)
    PHP_FE_END};

const zend_function_entry adaline_methods[] = {
    ML_METHOD(__construct, adaline_construct, arginfo_adaline_construct)
    ML_METHOD(fit, fit_regressor<AdalineModel>, arginfo_ml_fit_supervised)
    ML_METHOD(predict, predict_regressor<AdalineModel>, arginfo_ml_predict_float)
    ML_NETWORK_COMMON(AdalineModel)
    PHP_FE_END};

template <typename M>
void config_class(const char *name, const zend_function_entry *methods, zend_class_entry *iface, int kind) {
    zend_class_entry *ce = register_class<M>(name, methods, {iface});
    config_kinds[ce] = kind;
}

zend_class_entry *interface(const char *name) {
    zend_class_entry tmp;
    INIT_CLASS_ENTRY_EX(tmp, name, std::strlen(name), nullptr);
    return zend_register_internal_interface(&tmp);
}

}  // namespace

void mlphp::register_neuralnet() {
    ce_layer = interface("ML\\NeuralNet\\Layer");
    ce_optimizer = interface("ML\\NeuralNet\\Optimizer");
    ce_schedule = interface("ML\\NeuralNet\\Schedule");

    using LK = LayerSpec::Kind;
    config_class<LayerModel>("ML\\NeuralNet\\Layers\\Dense", dense_methods, ce_layer, static_cast<int>(LK::Dense));
    config_class<LayerModel>("ML\\NeuralNet\\Layers\\Activation", activation_methods, ce_layer, static_cast<int>(LK::Activation));
    config_class<LayerModel>("ML\\NeuralNet\\Layers\\Dropout", dropout_methods, ce_layer, static_cast<int>(LK::Dropout));
    config_class<LayerModel>("ML\\NeuralNet\\Layers\\BatchNorm", batch_norm_methods, ce_layer, static_cast<int>(LK::BatchNorm));
    config_class<LayerModel>("ML\\NeuralNet\\Layers\\PReLU", prelu_methods, ce_layer, static_cast<int>(LK::PReLU));
    config_class<LayerModel>("ML\\NeuralNet\\Layers\\Swish", swish_methods, ce_layer, static_cast<int>(LK::Swish));
    config_class<LayerModel>("ML\\NeuralNet\\Layers\\Noise", noise_methods, ce_layer, static_cast<int>(LK::Noise));

    using OK = OptimizerSpec::Kind;
    config_class<OptimizerModel>("ML\\NeuralNet\\Optimizers\\Stochastic", stochastic_methods, ce_optimizer, static_cast<int>(OK::Stochastic));
    config_class<OptimizerModel>("ML\\NeuralNet\\Optimizers\\Momentum", momentum_methods, ce_optimizer, static_cast<int>(OK::Momentum));
    config_class<OptimizerModel>("ML\\NeuralNet\\Optimizers\\AdaGrad", adagrad_methods, ce_optimizer, static_cast<int>(OK::AdaGrad));
    config_class<OptimizerModel>("ML\\NeuralNet\\Optimizers\\RMSProp", rmsprop_methods, ce_optimizer, static_cast<int>(OK::RMSProp));
    config_class<OptimizerModel>("ML\\NeuralNet\\Optimizers\\Adam", adam_methods, ce_optimizer, static_cast<int>(OK::Adam));
    config_class<OptimizerModel>("ML\\NeuralNet\\Optimizers\\AdaMax", adamax_methods, ce_optimizer, static_cast<int>(OK::AdaMax));

    using SK = ScheduleSpec::Kind;
    config_class<ScheduleModel>("ML\\NeuralNet\\Schedules\\Constant", constant_methods, ce_schedule, static_cast<int>(SK::Constant));
    config_class<ScheduleModel>("ML\\NeuralNet\\Schedules\\StepDecay", step_decay_methods, ce_schedule, static_cast<int>(SK::StepDecay));
    config_class<ScheduleModel>("ML\\NeuralNet\\Schedules\\Cosine", cosine_methods, ce_schedule, static_cast<int>(SK::Cosine));
    config_class<ScheduleModel>("ML\\NeuralNet\\Schedules\\Cyclical", cyclical_methods, ce_schedule, static_cast<int>(SK::Cyclical));
    config_class<ScheduleModel>("ML\\NeuralNet\\Schedules\\Ramp", ramp_methods, ce_schedule, static_cast<int>(SK::Ramp));

    register_class<MLPClassifierModel>("ML\\Classification\\MLPClassifier", mlp_classifier_methods,
                                       {ce_learner, ce_classifier, ce_probabilistic});
    register_class<MLPRegressorModel>("ML\\Regression\\MLPRegressor", mlp_regressor_methods, {ce_learner, ce_regressor});
    register_class<AdalineModel>("ML\\Regression\\Adaline", adaline_methods, {ce_learner, ce_regressor});
}
