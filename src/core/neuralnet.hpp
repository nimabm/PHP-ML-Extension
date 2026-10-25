#pragma once

#include "archive.hpp"
#include "common.hpp"

namespace mlcore {

// Element-wise activation, from a spec such as "relu", "leaky_relu(0.1)", "elu",
// "selu", "gelu", "sigmoid", "tanh", "softplus", "softsign", "silu",
// "thresholded_relu(1.0)" or "linear".
struct ActivationFn {
    enum class Type : std::uint8_t { Linear, ReLU, LeakyReLU, ELU, SELU, GELU, Sigmoid, Tanh, Softplus, Softsign, SiLU, ThresholdedReLU };
    Type type = Type::ReLU;
    double param = 0.0;

    static ActivationFn parse(const std::string &spec);
    std::string spec() const;
    double apply(double x) const;
    double derivative(double x, double y) const;  // y = apply(x)
};

// Weight initializers: he, lecun, normal(<stddev>), uniform(<beta>), xavier1, xavier2, constant(<value>).
struct Initializer {
    enum class Type : std::uint8_t { He, LeCun, Normal, Uniform, Xavier1, Xavier2, Constant };
    Type type = Type::He;
    double param = 0.0;

    static Initializer parse(const std::string &spec);
    std::string spec() const;
    void fill(std::vector<double> &w, std::size_t fan_in, std::size_t fan_out, Rng &rng) const;
};

struct LayerSpec {
    enum class Kind : std::uint8_t { Dense, Activation, Dropout, BatchNorm, PReLU, Swish, Noise };
    Kind kind = Kind::Dense;
    std::size_t neurons = 0;
    double l2 = 0.0;
    bool bias = true;
    Initializer init;
    ActivationFn activation;
    double ratio = 0.5;     // dropout
    double decay = 0.1;     // batch norm running statistics
    double initial = 0.25;  // PReLU leak / Swish beta
    double stddev = 0.1;    // noise

    void save(ArchiveWriter &w) const;
    void load(ArchiveReader &r);
};

struct OptimizerSpec {
    enum class Kind : std::uint8_t { Stochastic, Momentum, AdaGrad, RMSProp, Adam, AdaMax };
    Kind kind = Kind::Adam;
    double rate = 0.001;
    double decay = 0.1;            // Momentum (1 - momentum), RMSProp
    double momentum_decay = 0.1;   // Adam/AdaMax: 1 - beta1
    double norm_decay = 0.001;     // Adam/AdaMax: 1 - beta2
    bool lookahead = false;        // Nesterov momentum

    void save(ArchiveWriter &w) const;
    void load(ArchiveReader &r);
};

// Learning-rate multiplier per epoch.
struct ScheduleSpec {
    enum class Kind : std::uint8_t { Constant, StepDecay, Cosine, Cyclical, Ramp };
    Kind kind = Kind::Constant;
    std::size_t steps = 10;
    double factor = 0.5;   // StepDecay decay factor / Cosine & Cyclical minimum

    double multiplier(std::size_t epoch) const;
    void save(ArchiveWriter &w) const;
    void load(ArchiveReader &r);
};

// A trainable parameter tensor with its gradient and optimizer state.
struct Param {
    std::vector<double> value, grad, m1, m2;
    void resize(std::size_t n) {
        value.assign(n, 0.0);
        grad.assign(n, 0.0);
        m1.assign(n, 0.0);
        m2.assign(n, 0.0);
    }
};

struct Layer {
    LayerSpec spec;
    std::size_t in = 0, out = 0;
    Param W, b, gamma, beta, alpha;  // which are used depends on the kind
    std::vector<double> running_mean, running_var;

    // Training caches (not serialized).
    Matrix input, output, mask, xhat;
    std::vector<double> inv_std;

    void build(std::size_t fan_in, Rng &rng);
    void forward(const Matrix &x, Matrix &y, bool training, Rng *rng, bool cache);
    void infer(const Matrix &x, Matrix &y) const;
    // input_grad = false lets a Dense layer skip grad_in (nothing below it to train).
    void backward(const Matrix &grad_out, Matrix &grad_in, bool input_grad = true);
    void params(std::vector<Param *> &out);
    void save(ArchiveWriter &w) const;
    void load(ArchiveReader &r, std::size_t fan_in);
};

struct CostSpec {
    enum class Kind : std::uint8_t { CrossEntropy, LeastSquares, Huber };
    Kind kind = Kind::LeastSquares;
    double delta = 1.0;
    static CostSpec parse(const std::string &spec, bool classification);
    std::string spec() const;
};

// Feed-forward network trained with mini-batch gradient descent. The output
// layer is added automatically: softmax over `outputs` classes for
// classification, a single linear neuron for regression.
class MLP {
public:
    bool classification = true;
    std::vector<LayerSpec> hidden;
    std::size_t batch_size = 128;
    OptimizerSpec optimizer;
    ScheduleSpec schedule;
    std::size_t epochs = 1000;
    double min_change = 1e-4;
    std::size_t window = 5;
    double hold_out = 0.1;
    double l2 = 0.0;  // on the output layer
    CostSpec cost;
    bool has_seed = false;
    std::uint64_t seed = 0;

    std::size_t features = 0, outputs = 0;
    std::vector<Layer> layers;
    std::vector<double> losses, scores;

    // targets: class index (classification) or value (regression).
    void fit(const Matrix &X, const std::vector<double> &targets, std::size_t n_classes);
    void predict_raw(const double *x, double *out) const;  // probabilities or the regression value
    bool fitted() const { return !layers.empty(); }
    void save(ArchiveWriter &w) const;
    void load(ArchiveReader &r);

private:
    void forward(const Matrix &X, Matrix &out, bool training, Rng *rng);
    double batch_loss(const Matrix &out, const std::vector<double> &targets, const std::vector<std::size_t> &rows,
                      std::size_t begin, std::size_t end, Matrix *grad) const;
    double evaluate(const Matrix &X, const std::vector<double> &targets, const std::vector<std::size_t> &rows) const;
};

}  // namespace mlcore
