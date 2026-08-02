#include "potential.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>
#include <utility>

namespace bn {

namespace {

int product(const std::vector<int>& dims) {
    int p = 1;
    for (int d : dims) p *= d;
    return p;
}

}  // namespace

Potential::Potential() : log_(1, 0.0f) { computeCached(); }  // identity: P() = 1

Potential::Potential(std::vector<Variable> variables, std::vector<float> probabilities)
    : variables_(std::move(variables)), log_(probabilities.size()) {
    computeCached();
    if (static_cast<int>(probabilities.size()) != num_entries_) {
        throw std::invalid_argument(
            "Potential: number of probabilities does not match the product of "
            "variable state counts");
    }
    for (std::size_t i = 0; i < probabilities.size(); ++i) {
        if (probabilities[i] < 0.0f) {
            throw std::invalid_argument("Potential: negative probability");
        }
        log_[i] = std::log(probabilities[i]);
    }
}

Potential Potential::fromLog(std::vector<Variable> variables,
                             std::vector<float> log_table) {
    Potential p;
    p.variables_ = std::move(variables);
    p.log_ = std::move(log_table);
    p.computeCached();
    if (static_cast<int>(p.log_.size()) != p.num_entries_) {
        throw std::invalid_argument(
            "Potential::fromLog: log-table size does not match the product of "
            "variable state counts");
    }
    return p;
}

void Potential::computeCached() {
    dims_.clear();
    dims_.reserve(variables_.size());
    for (const Variable& v : variables_) dims_.push_back(v.num_states());
    strides_.assign(dims_.size(), 1);
    int acc = 1;
    for (int i = static_cast<int>(dims_.size()) - 1; i >= 0; --i) {
        strides_[i] = acc;
        acc *= dims_[i];
    }
    num_entries_ = acc;
}

int Potential::numEntries() const { return num_entries_; }

bool Potential::contains(const Variable& v) const {
    return std::any_of(variables_.begin(), variables_.end(),
                       [&](const Variable& x) { return x == v; });
}

bool Potential::contains(int variable_id) const {
    return std::any_of(variables_.begin(), variables_.end(),
                       [&](const Variable& x) { return x.id() == variable_id; });
}

int Potential::positionOf(const Variable& v) const {
    for (std::size_t i = 0; i < variables_.size(); ++i) {
        if (variables_[i] == v) return static_cast<int>(i);
    }
    return -1;
}

int Potential::positionOfId(int variable_id) const {
    for (std::size_t i = 0; i < variables_.size(); ++i) {
        if (variables_[i].id() == variable_id) return static_cast<int>(i);
    }
    return -1;
}

float Potential::probability(const std::map<Variable, int>& assignment) const {
    int idx = 0;
    for (std::size_t i = 0; i < variables_.size(); ++i) {
        auto it = assignment.find(variables_[i]);
        if (it == assignment.end()) {
            throw std::invalid_argument(
                "Potential::probability: missing assignment for variable " +
                variables_[i].name());
        }
        if (it->second < 0 || it->second >= dims_[i]) {
            throw std::invalid_argument(
                "Potential::probability: state out of range for variable " +
                variables_[i].name());
        }
        idx += it->second * strides_[i];
    }
    return std::exp(log_[idx]);
}

std::vector<float> Potential::probabilities() const {
    std::vector<float> out(log_.size());
    for (std::size_t i = 0; i < log_.size(); ++i) out[i] = std::exp(log_[i]);
    return out;
}

Potential Potential::operator*(const Potential& other) const {
    // Scope of the result: this scope followed by other's variables not present.
    std::vector<Variable> result_vars = variables_;
    // For every this-variable, its position inside `other` (-1 if absent).
    std::vector<int> shared_other_pos(variables_.size(), -1);
    std::vector<Variable> other_new;  // other's variables not in this (trailing dims)
    std::vector<int> other_new_pos;   // their positions inside `other`
    for (std::size_t i = 0; i < variables_.size(); ++i) {
        const int pb = other.positionOf(variables_[i]);
        shared_other_pos[i] = pb;
        if (pb >= 0 && variables_[i].num_states() != other.variables_[pb].num_states()) {
            throw std::invalid_argument(
                "Potential::operator*: shared variable has mismatched state "
                "counts");
        }
    }
    for (std::size_t i = 0; i < other.variables_.size(); ++i) {
        if (!contains(other.variables_[i])) {
            other_new.push_back(other.variables_[i]);
            other_new_pos.push_back(static_cast<int>(i));
        }
    }
    result_vars.insert(result_vars.end(), other_new.begin(), other_new.end());

    // The trailing (other-only) dimensions vary fastest, so the result index
    // is ia * nb + c, where ia enumerates this's cells and c the new dims.
    int nb = 1;
    for (const Variable& v : other_new) nb *= v.num_states();

    // Precompute, for every combination c of the new dims, the contribution
    // to the index inside `other`.
    std::vector<int> bnew(nb);
    {
        std::vector<int> dims;
        for (const Variable& v : other_new) dims.push_back(v.num_states());
        std::vector<int> cur(other_new.size(), 0);
        for (int c = 0; c < nb; ++c) {
            int idx = 0;
            for (std::size_t i = 0; i < other_new_pos.size(); ++i) {
                idx += cur[i] * other.strides_[other_new_pos[i]];
            }
            bnew[c] = idx;
            for (int r = static_cast<int>(dims.size()) - 1; r >= 0; --r) {
                if (++cur[r] < dims[r]) break;
                cur[r] = 0;
            }
        }
    }

    const int n_this = numEntries();
    std::vector<float> out(n_this * nb);
    const float* log_this = log_.data();
    const float* log_other = other.log_.data();
    const std::vector<int>& bstride = other.strides_;
    std::vector<int> cur_this(variables_.size(), 0);
    for (int ia = 0; ia < n_this; ++ia) {
        int ib_shared = 0;
        for (std::size_t i = 0; i < variables_.size(); ++i) {
            const int pb = shared_other_pos[i];
            if (pb >= 0) ib_shared += cur_this[i] * bstride[pb];
        }
        const float la = log_this[ia];
        float* out_base = out.data() + ia * nb;
        for (int c = 0; c < nb; ++c) {
            out_base[c] = la + log_other[ib_shared + bnew[c]];
        }
        for (int r = static_cast<int>(variables_.size()) - 1; r >= 0; --r) {
            if (++cur_this[r] < dims_[r]) break;
            cur_this[r] = 0;
        }
    }
    return fromLog(std::move(result_vars), std::move(out));
}

Potential Potential::operator/(const Potential& divisor) const {
    for (const Variable& v : divisor.variables_) {
        if (!contains(v)) {
            throw std::invalid_argument(
                "Potential::operator/: divisor variable is not in the dividend "
                "scope");
        }
    }
    // Per dividend position: how much the divisor index changes when the
    // odometer increments that dimension (0 for dimensions not in divisor).
    std::vector<int> div_inc(variables_.size(), 0);
    for (std::size_t i = 0; i < divisor.variables_.size(); ++i) {
        div_inc[positionOf(divisor.variables_[i])] = divisor.strides_[i];
    }

    const int total = numEntries();
    std::vector<float> out(total);
    const float* log_this = log_.data();
    const float* log_div = divisor.log_.data();
    std::vector<int> cur(variables_.size(), 0);
    int div_idx = 0;
    for (int k = 0; k < total; ++k) {
        out[k] = log_this[k] - log_div[div_idx];
        for (int r = static_cast<int>(variables_.size()) - 1; r >= 0; --r) {
            ++cur[r];
            div_idx += div_inc[r];
            if (cur[r] < dims_[r]) break;
            cur[r] = 0;
            div_idx -= dims_[r] * div_inc[r];
        }
    }
    return fromLog(variables_, std::move(out));
}

Potential Potential::marginalize(const std::vector<Variable>& to_sum_out) const {
    for (const Variable& v : to_sum_out) {
        if (!contains(v)) {
            throw std::invalid_argument(
                "Potential::marginalize: variable to sum out is not in scope");
        }
    }
    std::vector<Variable> keep_vars;
    std::vector<int> keep_pos, sum_pos;
    for (std::size_t i = 0; i < variables_.size(); ++i) {
        const bool in_sum = std::any_of(
            to_sum_out.begin(), to_sum_out.end(),
            [&](const Variable& v) { return v == variables_[i]; });
        if (in_sum) {
            sum_pos.push_back(static_cast<int>(i));
        } else {
            keep_pos.push_back(static_cast<int>(i));
            keep_vars.push_back(variables_[i]);
        }
    }

    const std::vector<int>& dims = dims_;
    const std::vector<int>& stride = strides_;
    std::vector<int> keep_dim, sum_dim;
    for (int p : keep_pos) keep_dim.push_back(dims[p]);
    for (int p : sum_pos) sum_dim.push_back(dims[p]);

    const int total_keep = product(keep_dim);
    const int total_sum = product(sum_dim);
    std::vector<float> out(total_keep);

    // Index contribution of each combination of the summed-out variables is
    // independent of the kept variables, so it is precomputed once.
    std::vector<int> sum_contrib(total_sum);
    {
        std::vector<int> cur(sum_pos.size(), 0);
        for (int s = 0; s < total_sum; ++s) {
            int idx = 0;
            for (std::size_t i = 0; i < sum_pos.size(); ++i) {
                idx += cur[i] * stride[sum_pos[i]];
            }
            sum_contrib[s] = idx;
            for (int r = static_cast<int>(sum_pos.size()) - 1; r >= 0; --r) {
                if (++cur[r] < sum_dim[r]) break;
                cur[r] = 0;
            }
        }
    }

    std::vector<int> cur_keep(keep_pos.size(), 0);
    const float* data = log_.data();
    for (int k = 0; k < total_keep; ++k) {
        int keep_base = 0;
        for (std::size_t i = 0; i < keep_pos.size(); ++i) {
            keep_base += cur_keep[i] * stride[keep_pos[i]];
        }
        // log-sum-exp over the summed-out dimensions (two passes, no buffer).
        float m = -std::numeric_limits<float>::infinity();
        for (int s = 0; s < total_sum; ++s) {
            const float v = data[keep_base + sum_contrib[s]];
            if (v > m) m = v;
        }
        if (std::isfinite(m)) {
            double acc = 0.0;
            for (int s = 0; s < total_sum; ++s) {
                acc += std::exp(static_cast<double>(data[keep_base + sum_contrib[s]]) - m);
            }
            out[k] = m + static_cast<float>(std::log(acc));
        } else {
            out[k] = m;  // every entry was log(0)
        }
        for (int r = static_cast<int>(keep_pos.size()) - 1; r >= 0; --r) {
            if (++cur_keep[r] < keep_dim[r]) break;
            cur_keep[r] = 0;
        }
    }
    return fromLog(std::move(keep_vars), std::move(out));
}

Potential Potential::reorder(const std::vector<Variable>& order) const {
    if (order.size() != variables_.size()) {
        throw std::invalid_argument(
            "Potential::reorder: order must contain every variable exactly once");
    }
    std::set<int> expected, actual;
    for (const Variable& v : order) expected.insert(v.id());
    for (const Variable& v : variables_) actual.insert(v.id());
    if (expected != actual) {
        throw std::invalid_argument(
            "Potential::reorder: order is not a permutation of the scope");
    }

    const std::vector<int>& stride = strides_;

    std::vector<int> map(order.size());
    for (std::size_t i = 0; i < order.size(); ++i) map[i] = positionOf(order[i]);
    std::vector<int> rdims;
    rdims.reserve(order.size());
    for (const Variable& v : order) rdims.push_back(v.num_states());

    const int total = numEntries();
    std::vector<float> out(total);
    const float* data = log_.data();
    std::vector<int> cur(order.size(), 0);
    int src = 0;
    for (int k = 0; k < total; ++k) {
        out[k] = data[src];
        // Odometer that also tracks the source index incrementally.
        for (int r = static_cast<int>(order.size()) - 1; r >= 0; --r) {
            ++cur[r];
            src += stride[map[r]];
            if (cur[r] < rdims[r]) break;
            cur[r] = 0;
            src -= rdims[r] * stride[map[r]];
        }
    }
    return fromLog(order, std::move(out));
}

Potential Potential::restrict(const std::map<Variable, int>& assignment) const {
    std::vector<Variable> keep_vars;
    std::vector<int> keep_pos, fix_pos;
    for (std::size_t i = 0; i < variables_.size(); ++i) {
        if (assignment.count(variables_[i]) > 0) {
            fix_pos.push_back(static_cast<int>(i));
        } else {
            keep_pos.push_back(static_cast<int>(i));
            keep_vars.push_back(variables_[i]);
        }
    }

    const std::vector<int>& dims = dims_;
    const std::vector<int>& stride = strides_;
    std::vector<int> keep_dim;
    for (int p : keep_pos) keep_dim.push_back(dims[p]);

    // Index contribution of the fixed variables is constant across all cells.
    int fixed_offset = 0;
    for (int p : fix_pos) {
        const auto it = assignment.find(variables_[p]);
        if (it->second < 0 || it->second >= dims[p]) {
            throw std::invalid_argument(
                "Potential::restrict: state out of range");
        }
        fixed_offset += it->second * stride[p];
    }

    const int total_keep = product(keep_dim);
    std::vector<float> out(total_keep);
    std::vector<int> cur_keep(keep_pos.size(), 0);
    for (int k = 0; k < total_keep; ++k) {
        int idx = fixed_offset;
        for (std::size_t i = 0; i < keep_pos.size(); ++i) {
            idx += cur_keep[i] * stride[keep_pos[i]];
        }
        out[k] = log_[idx];
        for (int r = static_cast<int>(keep_pos.size()) - 1; r >= 0; --r) {
            if (++cur_keep[r] < keep_dim[r]) break;
            cur_keep[r] = 0;
        }
    }
    return fromLog(std::move(keep_vars), std::move(out));
}

Potential& Potential::normalize() {
    // Two passes over the flat buffer, no intermediate copy.
    float m = -std::numeric_limits<float>::infinity();
    for (float v : log_) {
        if (v > m) m = v;
    }
    if (std::isfinite(m)) {
        double acc = 0.0;
        for (float v : log_) {
            acc += std::exp(static_cast<double>(v) - m);
        }
        const float z = m + static_cast<float>(std::log(acc));
        for (float& v : log_) v -= z;
    }
    return *this;
}

Potential& Potential::normalizeOver(const std::vector<Variable>& to_normalize) {
    std::vector<int> norm_pos, other_pos;
    for (std::size_t i = 0; i < variables_.size(); ++i) {
        const bool in_norm = std::any_of(
            to_normalize.begin(), to_normalize.end(),
            [&](const Variable& v) { return v == variables_[i]; });
        if (in_norm) {
            norm_pos.push_back(static_cast<int>(i));
        } else {
            other_pos.push_back(static_cast<int>(i));
        }
    }

    const std::vector<int>& dims = dims_;
    const std::vector<int>& stride = strides_;
    std::vector<int> norm_dim, other_dim;
    for (int p : norm_pos) norm_dim.push_back(dims[p]);
    for (int p : other_pos) other_dim.push_back(dims[p]);

    const int total_other = product(other_dim);
    const int total_norm = product(norm_dim);

    // Index contribution of each combination of the normalized variables.
    std::vector<int> norm_contrib(total_norm);
    {
        std::vector<int> cur(norm_pos.size(), 0);
        for (int s = 0; s < total_norm; ++s) {
            int idx = 0;
            for (std::size_t i = 0; i < norm_pos.size(); ++i) {
                idx += cur[i] * stride[norm_pos[i]];
            }
            norm_contrib[s] = idx;
            for (int r = static_cast<int>(norm_pos.size()) - 1; r >= 0; --r) {
                if (++cur[r] < norm_dim[r]) break;
                cur[r] = 0;
            }
        }
    }

    std::vector<int> cur_other(other_pos.size(), 0);
    float* data = log_.data();
    for (int k = 0; k < total_other; ++k) {
        int other_base = 0;
        for (std::size_t i = 0; i < other_pos.size(); ++i) {
            other_base += cur_other[i] * stride[other_pos[i]];
        }
        // log-sum-exp over the normalized dimensions, then write back.
        float m = -std::numeric_limits<float>::infinity();
        for (int s = 0; s < total_norm; ++s) {
            const float v = data[other_base + norm_contrib[s]];
            if (v > m) m = v;
        }
        float z;
        if (std::isfinite(m)) {
            double acc = 0.0;
            for (int s = 0; s < total_norm; ++s) {
                acc += std::exp(static_cast<double>(data[other_base + norm_contrib[s]]) - m);
            }
            z = m + static_cast<float>(std::log(acc));
        } else {
            z = m;
        }
        for (int s = 0; s < total_norm; ++s) {
            data[other_base + norm_contrib[s]] -= z;
        }
        for (int r = static_cast<int>(other_pos.size()) - 1; r >= 0; --r) {
            if (++cur_other[r] < other_dim[r]) break;
            cur_other[r] = 0;
        }
    }
    return *this;
}

}  // namespace bn
