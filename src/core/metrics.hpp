#pragma once

#include "common.hpp"

namespace mlcore {

// ---- classification (labels encoded as 0..K-1) ------------------------------

// K x K counts: [actual][predicted].
std::vector<double> confusion(const std::vector<int> &expected, const std::vector<int> &predicted, std::size_t K);
double accuracy(const std::vector<int> &expected, const std::vector<int> &predicted);

struct ClassStats {
    double tp = 0, fp = 0, fn = 0, tn = 0;
    double precision() const { return tp + fp > 0 ? tp / (tp + fp) : 0.0; }
    double recall() const { return tp + fn > 0 ? tp / (tp + fn) : 0.0; }
    double specificity() const { return tn + fp > 0 ? tn / (tn + fp) : 0.0; }
    double f_beta(double beta) const {
        const double p = precision(), r = recall(), b2 = beta * beta;
        return p + r > 0 ? (1 + b2) * p * r / (b2 * p + r) : 0.0;
    }
    double support() const { return tp + fn; }
};
std::vector<ClassStats> class_stats(const std::vector<double> &confusion, std::size_t K);

// Macro averages over classes that appear in expected or predicted labels.
double macro_precision(const std::vector<ClassStats> &stats);
double macro_recall(const std::vector<ClassStats> &stats);
double macro_f_beta(const std::vector<ClassStats> &stats, double beta);
double macro_informedness(const std::vector<ClassStats> &stats);
// Multiclass Matthews correlation coefficient (Gorodkin's R_K).
double mcc(const std::vector<double> &confusion, std::size_t K);

// Area under the ROC curve for binary relevance (`positive`) and scores; ties averaged.
double roc_auc(const std::vector<bool> &positive, const std::vector<double> &scores);
double average_precision(const std::vector<bool> &positive, const std::vector<double> &scores);

// ---- regression ----------------------------------------------------------------

double mean_squared_error(const std::vector<double> &e, const std::vector<double> &p);
double mean_absolute_error(const std::vector<double> &e, const std::vector<double> &p);
double median_absolute_error(const std::vector<double> &e, const std::vector<double> &p);
double r2_score(const std::vector<double> &e, const std::vector<double> &p);
// Symmetric mean absolute percentage error, in percent (0..100).
double smape(const std::vector<double> &e, const std::vector<double> &p);

// ---- clustering (labels and clusters encoded) ---------------------------------

// Contingency counts: [label][cluster].
std::vector<double> contingency(const std::vector<int> &labels, std::size_t L, const std::vector<int> &clusters, std::size_t C);
double homogeneity(const std::vector<double> &table, std::size_t L, std::size_t C);
double completeness(const std::vector<double> &table, std::size_t L, std::size_t C);
double v_measure(const std::vector<double> &table, std::size_t L, std::size_t C, double beta);
double adjusted_rand_index(const std::vector<double> &table, std::size_t L, std::size_t C);
double purity(const std::vector<double> &table, std::size_t L, std::size_t C);

// How well the k-neighborhoods of `original` are preserved in `embedded` (0..1).
double trustworthiness(const Matrix &original, const Matrix &embedded, std::size_t k);

}  // namespace mlcore
