#include "neuralnet.hpp"

#include <cstdio>
#include <cstdlib>
#include <numeric>

namespace mlcore {

namespace {

constexpr double kSeluLambda = 1.0507009873554805;
constexpr double kSeluAlpha = 1.6732632423543772;
constexpr double kSqrt2OverPi = 0.7978845608028654;

double sigmoid(double x) { return x >= 0 ? 1.0 / (1.0 + std::exp(-x)) : std::exp(x) / (1.0 + std::exp(x)); }

// Splits "name(arg)" into name and an optional numeric argument.
std::string split_spec(const std::string &spec, bool &has_arg, double &arg) {
    const auto open = spec.find('(');
    has_arg = false;
    if (open == std::string::npos) {
        return spec;
    }
    if (spec.back() != ')') {
        throw std::invalid_argument("Malformed specification \"" + spec + "\"");
    }
    const std::string number = spec.substr(open + 1, spec.size() - open - 2);
    char *end = nullptr;
    arg = std::strtod(number.c_str(), &end);
    if (number.empty() || end != number.c_str() + number.size() || !std::isfinite(arg)) {
        throw std::invalid_argument("Malformed numeric argument in \"" + spec + "\"");
    }
    has_arg = true;
    return spec.substr(0, open);
}

std::string with_arg(const std::string &name, double arg) {
    char buffer[64];
    std::snprintf(buffer, sizeof buffer, "%s(%.17g)", name.c_str(), arg);
    return buffer;
}

// out (n x k) = a (n x m) * b (m x k)
void matmul(const Matrix &a, const std::vector<double> &b, std::size_t k, Matrix &out) {
    out = Matrix(a.rows, k);
    for (std::size_t i = 0; i < a.rows; ++i) {
        double *o = out.row(i);
        const double *x = a.row(i);
        for (std::size_t p = 0; p < a.cols; ++p) {
            const double v = x[p];
            if (v == 0.0) continue;
            const double *w = &b[p * k];
            for (std::size_t j = 0; j < k; ++j) o[j] += v * w[j];
        }
    }
}

}  // namespace

// ---- ActivationFn ---------------------------------------------------------------------

ActivationFn ActivationFn::parse(const std::string &spec) {
    bool has_arg;
    double arg = 0.0;
    const std::string name = split_spec(spec, has_arg, arg);
    ActivationFn f;
    if (name == "linear") f.type = Type::Linear;
    else if (name == "relu") f.type = Type::ReLU;
    else if (name == "leaky_relu") { f.type = Type::LeakyReLU; f.param = has_arg ? arg : 0.01; }
    else if (name == "elu") { f.type = Type::ELU; f.param = has_arg ? arg : 1.0; }
    else if (name == "selu") f.type = Type::SELU;
    else if (name == "gelu") f.type = Type::GELU;
    else if (name == "sigmoid") f.type = Type::Sigmoid;
    else if (name == "tanh") f.type = Type::Tanh;
    else if (name == "softplus") f.type = Type::Softplus;
    else if (name == "softsign") f.type = Type::Softsign;
    else if (name == "silu") f.type = Type::SiLU;
    else if (name == "thresholded_relu") { f.type = Type::ThresholdedReLU; f.param = has_arg ? arg : 1.0; }
    else {
        throw std::invalid_argument("Unknown activation function \"" + spec +
                                    "\"; expected relu, leaky_relu, elu, selu, gelu, sigmoid, tanh, softplus, softsign, "
                                    "silu, thresholded_relu or linear");
    }
    return f;
}

std::string ActivationFn::spec() const {
    switch (type) {
        case Type::Linear: return "linear";
        case Type::ReLU: return "relu";
        case Type::LeakyReLU: return with_arg("leaky_relu", param);
        case Type::ELU: return with_arg("elu", param);
        case Type::SELU: return "selu";
        case Type::GELU: return "gelu";
        case Type::Sigmoid: return "sigmoid";
        case Type::Tanh: return "tanh";
        case Type::Softplus: return "softplus";
        case Type::Softsign: return "softsign";
        case Type::SiLU: return "silu";
        case Type::ThresholdedReLU: return with_arg("thresholded_relu", param);
    }
    return "relu";
}

double ActivationFn::apply(double x) const {
    switch (type) {
        case Type::Linear: return x;
        case Type::ReLU: return x > 0 ? x : 0.0;
        case Type::LeakyReLU: return x > 0 ? x : param * x;
        case Type::ELU: return x > 0 ? x : param * (std::exp(x) - 1.0);
        case Type::SELU: return kSeluLambda * (x > 0 ? x : kSeluAlpha * (std::exp(x) - 1.0));
        case Type::GELU: return 0.5 * x * (1.0 + std::tanh(kSqrt2OverPi * (x + 0.044715 * x * x * x)));
        case Type::Sigmoid: return sigmoid(x);
        case Type::Tanh: return std::tanh(x);
        case Type::Softplus: return x > 30 ? x : std::log1p(std::exp(x));
        case Type::Softsign: return x / (1.0 + std::fabs(x));
        case Type::SiLU: return x * sigmoid(x);
        case Type::ThresholdedReLU: return x > param ? x : 0.0;
    }
    return x;
}

double ActivationFn::derivative(double x, double y) const {
    switch (type) {
        case Type::Linear: return 1.0;
        case Type::ReLU: return x > 0 ? 1.0 : 0.0;
        case Type::LeakyReLU: return x > 0 ? 1.0 : param;
        case Type::ELU: return x > 0 ? 1.0 : y + param;
        case Type::SELU: return kSeluLambda * (x > 0 ? 1.0 : kSeluAlpha * std::exp(x));
        case Type::GELU: {
            const double t = std::tanh(kSqrt2OverPi * (x + 0.044715 * x * x * x));
            return 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * kSqrt2OverPi * (1.0 + 3.0 * 0.044715 * x * x);
        }
        case Type::Sigmoid: return y * (1.0 - y);
        case Type::Tanh: return 1.0 - y * y;
        case Type::Softplus: return sigmoid(x);
        case Type::Softsign: {
            const double d = 1.0 + std::fabs(x);
            return 1.0 / (d * d);
        }
        case Type::SiLU: {
            const double s = sigmoid(x);
            return s * (1.0 + x * (1.0 - s));
        }
        case Type::ThresholdedReLU: return x > param ? 1.0 : 0.0;
    }
    return 1.0;
}

// ---- Initializer -----------------------------------------------------------------------

Initializer Initializer::parse(const std::string &spec) {
    bool has_arg;
    double arg = 0.0;
    const std::string name = split_spec(spec, has_arg, arg);
    Initializer init;
    if (name == "he") init.type = Type::He;
    else if (name == "lecun") init.type = Type::LeCun;
    else if (name == "normal") { init.type = Type::Normal; init.param = has_arg ? arg : 0.05; }
    else if (name == "uniform") { init.type = Type::Uniform; init.param = has_arg ? arg : 0.05; }
    else if (name == "xavier1") init.type = Type::Xavier1;
    else if (name == "xavier2") init.type = Type::Xavier2;
    else if (name == "constant") { init.type = Type::Constant; init.param = has_arg ? arg : 0.0; }
    else {
        throw std::invalid_argument("Unknown initializer \"" + spec +
                                    "\"; expected he, lecun, normal(<stddev>), uniform(<beta>), xavier1, xavier2 or constant(<value>)");
    }
    if ((init.type == Type::Normal || init.type == Type::Uniform) && !(init.param > 0.0)) {
        throw std::invalid_argument("Initializer scale must be greater than 0");
    }
    return init;
}

std::string Initializer::spec() const {
    switch (type) {
        case Type::He: return "he";
        case Type::LeCun: return "lecun";
        case Type::Normal: return with_arg("normal", param);
        case Type::Uniform: return with_arg("uniform", param);
        case Type::Xavier1: return "xavier1";
        case Type::Xavier2: return "xavier2";
        case Type::Constant: return with_arg("constant", param);
    }
    return "he";
}

void Initializer::fill(std::vector<double> &w, std::size_t fan_in, std::size_t fan_out, Rng &rng) const {
    const double in = static_cast<double>(fan_in), out = static_cast<double>(fan_out);
    auto normal = [&](double sd) {
        std::normal_distribution<double> d(0.0, sd);
        for (double &v : w) v = d(rng);
    };
    auto uniform = [&](double beta) {
        std::uniform_real_distribution<double> d(-beta, beta);
        for (double &v : w) v = d(rng);
    };
    switch (type) {
        case Type::He: normal(std::sqrt(2.0 / in)); break;
        case Type::LeCun: normal(std::sqrt(1.0 / in)); break;
        case Type::Normal: normal(param); break;
        case Type::Uniform: uniform(param); break;
        case Type::Xavier1: uniform(std::sqrt(6.0 / (in + out))); break;
        case Type::Xavier2: normal(std::sqrt(2.0 / (in + out))); break;
        case Type::Constant: std::fill(w.begin(), w.end(), param); break;
    }
}

// ---- specs ---------------------------------------------------------------------------------

void LayerSpec::save(ArchiveWriter &w) const {
    w.u8(static_cast<std::uint8_t>(kind));
    w.size(neurons);
    w.f64(l2);
    w.boolean(bias);
    w.str(init.spec());
    w.str(activation.spec());
    w.f64(ratio);
    w.f64(decay);
    w.f64(initial);
    w.f64(stddev);
}

void LayerSpec::load(ArchiveReader &r) {
    const std::uint8_t k = r.u8();
    ArchiveReader::require(k <= 6, "bad layer kind");
    kind = static_cast<Kind>(k);
    neurons = r.size(1u << 24);
    l2 = r.f64();
    bias = r.boolean();
    try {
        init = Initializer::parse(r.str());
        activation = ActivationFn::parse(r.str());
    } catch (const std::invalid_argument &) {
        ArchiveReader::fail("bad layer specification");
    }
    ratio = r.f64();
    decay = r.f64();
    initial = r.f64();
    stddev = r.f64();
    ArchiveReader::require(kind != Kind::Dense || neurons >= 1, "dense layer without neurons");
    ArchiveReader::require(ratio >= 0.0 && ratio < 1.0, "bad dropout ratio");
}

void OptimizerSpec::save(ArchiveWriter &w) const {
    w.u8(static_cast<std::uint8_t>(kind));
    w.f64(rate);
    w.f64(decay);
    w.f64(momentum_decay);
    w.f64(norm_decay);
    w.boolean(lookahead);
}

void OptimizerSpec::load(ArchiveReader &r) {
    const std::uint8_t k = r.u8();
    ArchiveReader::require(k <= 5, "bad optimizer");
    kind = static_cast<Kind>(k);
    rate = r.f64();
    decay = r.f64();
    momentum_decay = r.f64();
    norm_decay = r.f64();
    lookahead = r.boolean();
}

double ScheduleSpec::multiplier(std::size_t epoch) const {
    constexpr double kPi = 3.14159265358979323846;
    const double s = static_cast<double>(std::max<std::size_t>(steps, 1));
    switch (kind) {
        case Kind::Constant: return 1.0;
        case Kind::StepDecay: return std::pow(factor, std::floor(static_cast<double>(epoch) / s));
        case Kind::Cosine: {
            const double pos = static_cast<double>(epoch % std::max<std::size_t>(steps, 1));
            return factor + (1.0 - factor) * 0.5 * (1.0 + std::cos(kPi * pos / s));
        }
        case Kind::Cyclical: {
            const double pos = static_cast<double>(epoch % (2 * std::max<std::size_t>(steps, 1)));
            const double tri = pos < s ? pos / s : (2.0 * s - pos) / s;
            return factor + (1.0 - factor) * tri;
        }
        case Kind::Ramp: return std::min(1.0, static_cast<double>(epoch + 1) / s);
    }
    return 1.0;
}

void ScheduleSpec::save(ArchiveWriter &w) const {
    w.u8(static_cast<std::uint8_t>(kind));
    w.size(steps);
    w.f64(factor);
}

void ScheduleSpec::load(ArchiveReader &r) {
    const std::uint8_t k = r.u8();
    ArchiveReader::require(k <= 4, "bad schedule");
    kind = static_cast<Kind>(k);
    steps = r.size();
    factor = r.f64();
}

CostSpec CostSpec::parse(const std::string &spec, bool classification) {
    bool has_arg;
    double arg = 1.0;
    const std::string name = split_spec(spec, has_arg, arg);
    CostSpec c;
    if (classification) {
        if (name != "cross_entropy") {
            throw std::invalid_argument("Classifiers use the cross_entropy cost function, \"" + spec + "\" given");
        }
        c.kind = Kind::CrossEntropy;
        return c;
    }
    if (name == "least_squares") {
        c.kind = Kind::LeastSquares;
    } else if (name == "huber") {
        c.kind = Kind::Huber;
        c.delta = has_arg ? arg : 1.0;
        if (!(c.delta > 0.0)) throw std::invalid_argument("Huber delta must be greater than 0");
    } else {
        throw std::invalid_argument("Unknown cost function \"" + spec + "\"; expected least_squares or huber(<delta>)");
    }
    return c;
}

std::string CostSpec::spec() const {
    switch (kind) {
        case Kind::CrossEntropy: return "cross_entropy";
        case Kind::LeastSquares: return "least_squares";
        case Kind::Huber: return with_arg("huber", delta);
    }
    return "least_squares";
}

// ---- Layer ----------------------------------------------------------------------------------

void Layer::build(std::size_t fan_in, Rng &rng) {
    in = fan_in;
    switch (spec.kind) {
        case LayerSpec::Kind::Dense:
            out = spec.neurons;
            W.resize(in * out);
            spec.init.fill(W.value, in, out, rng);
            b.resize(spec.bias ? out : 0);
            break;
        case LayerSpec::Kind::BatchNorm:
            out = in;
            gamma.resize(in);
            std::fill(gamma.value.begin(), gamma.value.end(), 1.0);
            beta.resize(in);
            running_mean.assign(in, 0.0);
            running_var.assign(in, 1.0);
            break;
        case LayerSpec::Kind::PReLU:
        case LayerSpec::Kind::Swish:
            out = in;
            alpha.resize(in);
            std::fill(alpha.value.begin(), alpha.value.end(), spec.initial);
            break;
        default:
            out = in;
    }
}

namespace {

void apply_elementwise(const Layer &L, const Matrix &x, Matrix &y) {
    y = Matrix(x.rows, x.cols);
    for (std::size_t i = 0; i < x.rows; ++i) {
        for (std::size_t j = 0; j < x.cols; ++j) {
            const double v = x(i, j);
            switch (L.spec.kind) {
                case LayerSpec::Kind::Activation: y(i, j) = L.spec.activation.apply(v); break;
                case LayerSpec::Kind::PReLU: y(i, j) = v > 0 ? v : L.alpha.value[j] * v; break;
                case LayerSpec::Kind::Swish: y(i, j) = v * sigmoid(L.alpha.value[j] * v); break;
                default: y(i, j) = v;
            }
        }
    }
}

void dense(const Layer &L, const Matrix &x, Matrix &y) {
    matmul(x, L.W.value, L.out, y);
    if (!L.b.value.empty()) {
        for (std::size_t i = 0; i < y.rows; ++i) {
            for (std::size_t j = 0; j < L.out; ++j) y(i, j) += L.b.value[j];
        }
    }
}

}  // namespace

void Layer::infer(const Matrix &x, Matrix &y) const {
    switch (spec.kind) {
        case LayerSpec::Kind::Dense:
            dense(*this, x, y);
            return;
        case LayerSpec::Kind::BatchNorm:
            y = Matrix(x.rows, x.cols);
            for (std::size_t i = 0; i < x.rows; ++i) {
                for (std::size_t j = 0; j < x.cols; ++j) {
                    y(i, j) = gamma.value[j] * (x(i, j) - running_mean[j]) / std::sqrt(running_var[j] + 1e-8) + beta.value[j];
                }
            }
            return;
        case LayerSpec::Kind::Dropout:
        case LayerSpec::Kind::Noise:
            y = x;
            return;
        default:
            apply_elementwise(*this, x, y);
    }
}

void Layer::forward(const Matrix &x, Matrix &y, bool training, Rng *rng, bool cache) {
    if (cache) input = x;
    if (!training) {
        infer(x, y);
    } else {
        switch (spec.kind) {
            case LayerSpec::Kind::Dropout: {
                mask = Matrix(x.rows, x.cols);
                std::bernoulli_distribution keep(1.0 - spec.ratio);
                const double scale = 1.0 / (1.0 - spec.ratio);
                y = x;
                for (std::size_t i = 0; i < x.data.size(); ++i) {
                    mask.data[i] = keep(*rng) ? scale : 0.0;
                    y.data[i] *= mask.data[i];
                }
                break;
            }
            case LayerSpec::Kind::Noise: {
                std::normal_distribution<double> noise(0.0, spec.stddev);
                y = x;
                for (double &v : y.data) v += noise(*rng);
                break;
            }
            case LayerSpec::Kind::BatchNorm: {
                const std::size_t m = x.rows, d = x.cols;
                std::vector<double> mean(d, 0.0), var(d, 0.0);
                for (std::size_t i = 0; i < m; ++i)
                    for (std::size_t j = 0; j < d; ++j) mean[j] += x(i, j) / static_cast<double>(m);
                for (std::size_t i = 0; i < m; ++i)
                    for (std::size_t j = 0; j < d; ++j) var[j] += (x(i, j) - mean[j]) * (x(i, j) - mean[j]) / static_cast<double>(m);
                inv_std.resize(d);
                for (std::size_t j = 0; j < d; ++j) {
                    inv_std[j] = 1.0 / std::sqrt(var[j] + 1e-8);
                    running_mean[j] = (1.0 - spec.decay) * running_mean[j] + spec.decay * mean[j];
                    running_var[j] = (1.0 - spec.decay) * running_var[j] + spec.decay * var[j];
                }
                xhat = Matrix(m, d);
                y = Matrix(m, d);
                for (std::size_t i = 0; i < m; ++i) {
                    for (std::size_t j = 0; j < d; ++j) {
                        xhat(i, j) = (x(i, j) - mean[j]) * inv_std[j];
                        y(i, j) = gamma.value[j] * xhat(i, j) + beta.value[j];
                    }
                }
                break;
            }
            default:
                infer(x, y);
        }
    }
    if (cache) output = y;
}

void Layer::backward(const Matrix &g, Matrix &grad_in, bool input_grad) {
    const std::size_t m = g.rows;
    switch (spec.kind) {
        case LayerSpec::Kind::Dense: {
            std::fill(W.grad.begin(), W.grad.end(), 0.0);
            for (std::size_t i = 0; i < m; ++i) {
                const double *x = input.row(i);
                const double *gi = g.row(i);
                for (std::size_t p = 0; p < in; ++p) {
                    const double v = x[p];
                    if (v == 0.0) continue;
                    double *gw = &W.grad[p * out];
                    for (std::size_t j = 0; j < out; ++j) gw[j] += v * gi[j];
                }
            }
            if (spec.l2 > 0.0) {
                for (std::size_t k = 0; k < W.value.size(); ++k) W.grad[k] += spec.l2 * W.value[k];
            }
            if (!b.value.empty()) {
                std::fill(b.grad.begin(), b.grad.end(), 0.0);
                for (std::size_t i = 0; i < m; ++i)
                    for (std::size_t j = 0; j < out; ++j) b.grad[j] += g(i, j);
            }
            if (!input_grad) {
                return;
            }
            grad_in = Matrix(m, in);
            for (std::size_t i = 0; i < m; ++i) {
                const double *gi = g.row(i);
                double *gx = grad_in.row(i);
                for (std::size_t p = 0; p < in; ++p) gx[p] = dot(&W.value[p * out], gi, out);
            }
            return;
        }
        case LayerSpec::Kind::Activation:
            grad_in = Matrix(m, in);
            for (std::size_t k = 0; k < g.data.size(); ++k) {
                grad_in.data[k] = g.data[k] * spec.activation.derivative(input.data[k], output.data[k]);
            }
            return;
        case LayerSpec::Kind::Dropout:
            grad_in = g;
            for (std::size_t k = 0; k < g.data.size(); ++k) grad_in.data[k] *= mask.data[k];
            return;
        case LayerSpec::Kind::Noise:
            grad_in = g;
            return;
        case LayerSpec::Kind::BatchNorm: {
            const double mm = static_cast<double>(m);
            std::fill(gamma.grad.begin(), gamma.grad.end(), 0.0);
            std::fill(beta.grad.begin(), beta.grad.end(), 0.0);
            std::vector<double> sum_dxhat(in, 0.0), sum_dxhat_xhat(in, 0.0);
            for (std::size_t i = 0; i < m; ++i) {
                for (std::size_t j = 0; j < in; ++j) {
                    gamma.grad[j] += g(i, j) * xhat(i, j);
                    beta.grad[j] += g(i, j);
                    const double dxhat = g(i, j) * gamma.value[j];
                    sum_dxhat[j] += dxhat;
                    sum_dxhat_xhat[j] += dxhat * xhat(i, j);
                }
            }
            grad_in = Matrix(m, in);
            for (std::size_t i = 0; i < m; ++i) {
                for (std::size_t j = 0; j < in; ++j) {
                    const double dxhat = g(i, j) * gamma.value[j];
                    grad_in(i, j) = inv_std[j] / mm * (mm * dxhat - sum_dxhat[j] - xhat(i, j) * sum_dxhat_xhat[j]);
                }
            }
            return;
        }
        case LayerSpec::Kind::PReLU: {
            std::fill(alpha.grad.begin(), alpha.grad.end(), 0.0);
            grad_in = Matrix(m, in);
            for (std::size_t i = 0; i < m; ++i) {
                for (std::size_t j = 0; j < in; ++j) {
                    const double x = input(i, j);
                    if (x > 0) {
                        grad_in(i, j) = g(i, j);
                    } else {
                        grad_in(i, j) = g(i, j) * alpha.value[j];
                        alpha.grad[j] += g(i, j) * x;
                    }
                }
            }
            return;
        }
        case LayerSpec::Kind::Swish: {
            std::fill(alpha.grad.begin(), alpha.grad.end(), 0.0);
            grad_in = Matrix(m, in);
            for (std::size_t i = 0; i < m; ++i) {
                for (std::size_t j = 0; j < in; ++j) {
                    const double x = input(i, j), beta_j = alpha.value[j];
                    const double s = sigmoid(beta_j * x);
                    grad_in(i, j) = g(i, j) * (s + beta_j * x * s * (1.0 - s));
                    alpha.grad[j] += g(i, j) * x * x * s * (1.0 - s);
                }
            }
            return;
        }
    }
}

void Layer::params(std::vector<Param *> &list) {
    for (Param *p : {&W, &b, &gamma, &beta, &alpha}) {
        if (!p->value.empty()) list.push_back(p);
    }
}

void Layer::save(ArchiveWriter &w) const {
    spec.save(w);
    w.size(in);
    w.size(out);
    w.doubles(W.value);
    w.doubles(b.value);
    w.doubles(gamma.value);
    w.doubles(beta.value);
    w.doubles(alpha.value);
    w.doubles(running_mean);
    w.doubles(running_var);
}

void Layer::load(ArchiveReader &r, std::size_t fan_in) {
    spec.load(r);
    in = r.size();
    out = r.size();
    ArchiveReader::require(in == fan_in, "layer input size mismatch");
    auto restore = [&](Param &p, std::size_t expected) {
        const std::vector<double> v = r.doubles();
        ArchiveReader::require(v.size() == expected, "layer parameter size mismatch");
        p.resize(expected);
        p.value = v;
    };
    const bool dense = spec.kind == LayerSpec::Kind::Dense;
    const bool bn = spec.kind == LayerSpec::Kind::BatchNorm;
    const bool per_neuron = spec.kind == LayerSpec::Kind::PReLU || spec.kind == LayerSpec::Kind::Swish;
    ArchiveReader::require(dense ? out == spec.neurons : out == in, "layer output size mismatch");
    if (dense) ArchiveReader::require(in == 0 || out <= (1u << 30) / in, "layer too large");
    restore(W, dense ? in * out : 0);
    restore(b, dense && spec.bias ? out : 0);
    restore(gamma, bn ? in : 0);
    restore(beta, bn ? in : 0);
    restore(alpha, per_neuron ? in : 0);
    running_mean = r.doubles(bn ? in : 0);
    running_var = r.doubles(bn ? in : 0);
}

// ---- MLP --------------------------------------------------------------------------------------

namespace {

void step(Param &p, const OptimizerSpec &o, double rate, std::size_t t) {
    const double b1 = 1.0 - o.momentum_decay, b2 = 1.0 - o.norm_decay;
    const double tt = static_cast<double>(t);
    const std::size_t n = p.value.size();
    double *value = p.value.data();
    const double *grad = p.grad.data();
    double *m1 = p.m1.data();
    double *m2 = p.m2.data();
    // One loop per optimizer, with the bias corrections computed once per step.
    switch (o.kind) {
        case OptimizerSpec::Kind::Stochastic:
            for (std::size_t i = 0; i < n; ++i) value[i] -= rate * grad[i];
            break;
        case OptimizerSpec::Kind::Momentum: {
            const double mu = 1.0 - o.decay;
            for (std::size_t i = 0; i < n; ++i) {
                m1[i] = mu * m1[i] + rate * grad[i];
                value[i] -= o.lookahead ? mu * m1[i] + rate * grad[i] : m1[i];
            }
            break;
        }
        case OptimizerSpec::Kind::AdaGrad:
            for (std::size_t i = 0; i < n; ++i) {
                m2[i] += grad[i] * grad[i];
                value[i] -= rate * grad[i] / (std::sqrt(m2[i]) + 1e-8);
            }
            break;
        case OptimizerSpec::Kind::RMSProp:
            for (std::size_t i = 0; i < n; ++i) {
                m2[i] = (1.0 - o.decay) * m2[i] + o.decay * grad[i] * grad[i];
                value[i] -= rate * grad[i] / (std::sqrt(m2[i]) + 1e-8);
            }
            break;
        case OptimizerSpec::Kind::Adam: {
            const double c1 = 1.0 - std::pow(b1, tt), c2 = 1.0 - std::pow(b2, tt);
            for (std::size_t i = 0; i < n; ++i) {
                const double g = grad[i];
                m1[i] = b1 * m1[i] + (1.0 - b1) * g;
                m2[i] = b2 * m2[i] + (1.0 - b2) * g * g;
                const double mhat = m1[i] / c1;
                const double vhat = m2[i] / c2;
                value[i] -= rate * mhat / (std::sqrt(vhat) + 1e-8);
            }
            break;
        }
        case OptimizerSpec::Kind::AdaMax: {
            const double scale = rate / (1.0 - std::pow(b1, tt));
            for (std::size_t i = 0; i < n; ++i) {
                m1[i] = b1 * m1[i] + (1.0 - b1) * grad[i];
                m2[i] = std::max(b2 * m2[i], std::fabs(grad[i]));
                value[i] -= scale * m1[i] / (m2[i] + 1e-8);
            }
            break;
        }
    }
}

void clear_caches(std::vector<Layer> &layers) {
    for (Layer &L : layers) {
        L.input = Matrix();
        L.output = Matrix();
        L.mask = Matrix();
        L.xhat = Matrix();
    }
}

}  // namespace

void MLP::forward(const Matrix &X, Matrix &out, bool training, Rng *rng) {
    Matrix current = X, next;
    for (Layer &L : layers) {
        L.forward(current, next, training, rng, training);
        std::swap(current, next);
    }
    out = std::move(current);
}

double MLP::batch_loss(const Matrix &out, const std::vector<double> &targets, const std::vector<std::size_t> &rows,
                       std::size_t begin, std::size_t end, Matrix *grad) const {
    const std::size_t m = end - begin;
    const double inv = 1.0 / static_cast<double>(m);
    double loss = 0.0;
    if (grad) *grad = Matrix(m, out.cols);
    for (std::size_t i = 0; i < m; ++i) {
        const double target = targets[rows[begin + i]];
        if (classification) {
            std::vector<double> p(out.row(i), out.row(i) + out.cols);
            softmax(p.data(), p.size());
            const std::size_t k = static_cast<std::size_t>(target);
            loss -= std::log(std::max(p[k], 1e-15));
            if (grad) {
                for (std::size_t c = 0; c < out.cols; ++c) (*grad)(i, c) = (p[c] - (c == k ? 1.0 : 0.0)) * inv;
            }
        } else {
            const double r = out(i, 0) - target;
            if (cost.kind == CostSpec::Kind::Huber) {
                const double q = std::sqrt(1.0 + (r / cost.delta) * (r / cost.delta));
                loss += cost.delta * cost.delta * (q - 1.0);
                if (grad) (*grad)(i, 0) = r / q * inv;
            } else {
                loss += 0.5 * r * r;
                if (grad) (*grad)(i, 0) = r * inv;
            }
        }
    }
    return loss * inv;
}

double MLP::evaluate(const Matrix &X, const std::vector<double> &targets, const std::vector<std::size_t> &rows) const {
    double total = 0.0;
    for (std::size_t begin = 0; begin < rows.size(); begin += batch_size) {
        const std::size_t end = std::min(rows.size(), begin + batch_size);
        Matrix batch(end - begin, X.cols), current, next;
        for (std::size_t i = begin; i < end; ++i) std::copy(X.row(rows[i]), X.row(rows[i]) + X.cols, batch.row(i - begin));
        current = std::move(batch);
        for (const Layer &L : layers) {
            L.infer(current, next);
            std::swap(current, next);
        }
        total += batch_loss(current, targets, rows, begin, end, nullptr) * static_cast<double>(end - begin);
    }
    return total / static_cast<double>(rows.size());
}

void MLP::fit(const Matrix &X, const std::vector<double> &targets, std::size_t n_classes) {
    require_training_set(X, targets.size());
    if (classification && n_classes < 2) {
        throw std::invalid_argument("A classifier needs at least two distinct classes");
    }
    const std::size_t n = X.rows;
    Rng rng = make_rng(has_seed, seed);

    MLP next = *this;
    next.features = X.cols;
    next.outputs = classification ? n_classes : 1;
    next.layers.clear();
    std::size_t width = X.cols;
    for (const LayerSpec &spec : hidden) {
        Layer L;
        L.spec = spec;
        L.build(width, rng);
        width = L.out;
        next.layers.push_back(std::move(L));
    }
    Layer output;
    output.spec.kind = LayerSpec::Kind::Dense;
    output.spec.neurons = next.outputs;
    output.spec.l2 = l2;
    output.spec.init = Initializer::parse("xavier1");
    output.build(width, rng);
    next.layers.push_back(std::move(output));

    std::vector<std::size_t> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::shuffle(order.begin(), order.end(), rng);
    std::size_t n_val = hold_out > 0.0 ? static_cast<std::size_t>(std::llround(hold_out * static_cast<double>(n))) : 0;
    if (n_val >= n) n_val = 0;
    std::vector<std::size_t> val(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(n_val));
    std::vector<std::size_t> train(order.begin() + static_cast<std::ptrdiff_t>(n_val), order.end());

    std::vector<Param *> params;
    for (Layer &L : next.layers) L.params(params);

    next.losses.clear();
    next.scores.clear();
    std::vector<Layer> best = next.layers;
    double best_score = std::numeric_limits<double>::infinity();
    std::size_t stall = 0, t = 0;
    Matrix batch, out, grad, grad_in;
    for (std::size_t epoch = 0; epoch < epochs; ++epoch) {
        std::shuffle(train.begin(), train.end(), rng);
        const double rate = optimizer.rate * schedule.multiplier(epoch);
        double total = 0.0;
        for (std::size_t begin = 0; begin < train.size(); begin += batch_size) {
            const std::size_t end = std::min(train.size(), begin + batch_size);
            batch = Matrix(end - begin, X.cols);
            for (std::size_t i = begin; i < end; ++i) std::copy(X.row(train[i]), X.row(train[i]) + X.cols, batch.row(i - begin));
            next.forward(batch, out, true, &rng);
            total += next.batch_loss(out, targets, train, begin, end, &grad) * static_cast<double>(end - begin);
            for (std::size_t li = next.layers.size(); li-- > 0;) {
                next.layers[li].backward(grad, grad_in, li > 0);
                std::swap(grad, grad_in);
            }
            ++t;
            for (Param *p : params) step(*p, optimizer, rate, t);
        }
        const double loss = total / static_cast<double>(train.size());
        if (!std::isfinite(loss)) {
            throw std::runtime_error("Training diverged; lower the learning rate or scale the features");
        }
        next.losses.push_back(loss);
        const double score = val.empty() ? loss : next.evaluate(X, targets, val);
        if (!val.empty()) next.scores.push_back(score);
        if (score < best_score - min_change) {
            best_score = score;
            clear_caches(next.layers);
            best = next.layers;
            stall = 0;
        } else if (++stall >= window) {
            break;
        }
    }
    next.layers = std::move(best);
    clear_caches(next.layers);
    *this = std::move(next);
}

void MLP::predict_raw(const double *x, double *out) const {
    Matrix current(1, features), next;
    std::copy(x, x + features, current.row(0));
    for (const Layer &L : layers) {
        L.infer(current, next);
        std::swap(current, next);
    }
    std::copy(current.row(0), current.row(0) + current.cols, out);
    if (classification) softmax(out, current.cols);
}

void MLP::save(ArchiveWriter &w) const {
    w.boolean(classification);
    w.size(hidden.size());
    for (const LayerSpec &s : hidden) s.save(w);
    w.size(batch_size);
    optimizer.save(w);
    schedule.save(w);
    w.size(epochs);
    w.f64(min_change);
    w.size(window);
    w.f64(hold_out);
    w.f64(l2);
    w.str(cost.spec());
    w.boolean(has_seed);
    w.u64(seed);
    w.size(features);
    w.size(outputs);
    w.size(layers.size());
    for (const Layer &L : layers) L.save(w);
    w.doubles(losses);
    w.doubles(scores);
}

void MLP::load(ArchiveReader &r) {
    classification = r.boolean();
    hidden.assign(r.count(16), {});
    for (LayerSpec &s : hidden) s.load(r);
    batch_size = r.size();
    optimizer.load(r);
    schedule.load(r);
    epochs = r.size();
    min_change = r.f64();
    window = r.size();
    hold_out = r.f64();
    l2 = r.f64();
    try {
        cost = CostSpec::parse(r.str(), classification);
    } catch (const std::invalid_argument &) {
        ArchiveReader::fail("bad cost function");
    }
    has_seed = r.boolean();
    seed = r.u64();
    features = r.size(1u << 26);
    outputs = r.size(1u << 20);
    const std::size_t count = r.count(16);
    layers.assign(count, Layer{});
    std::size_t width = features;
    for (Layer &L : layers) {
        L.load(r, width);
        width = L.out;
    }
    ArchiveReader::require(layers.empty() || width == outputs, "network output size mismatch");
    ArchiveReader::require(batch_size >= 1, "bad batch size");
    losses = r.doubles();
    scores = r.doubles();
}

}  // namespace mlcore
