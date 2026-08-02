#include "potential.h"

#include <algorithm>
#include <cmath>
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

std::vector<int> dimsOf(const std::vector<Variable>& vars) {
    std::vector<int> dims;
    dims.reserve(vars.size());
    for (const Variable& v : vars) dims.push_back(v.num_states());
    return dims;
}

std::vector<int> rowMajorStrides(const std::vector<int>& dims) {
    std::vector<int> strides(dims.size());
    int acc = 1;
    for (int i = static_cast<int>(dims.size()) - 1; i >= 0; --i) {
        strides[i] = acc;
        acc *= dims[i];
    }
    return strides;
}

float logSumExp(const std::vector<float>& values) {
    if (values.empty()) return 0.0f;
    float m = *std::max_element(values.begin(), values.end());
    if (!std::isfinite(m)) return m;  // all entries were -inf (zero probability)
    double acc = 0.0;
    for (float v : values) acc += std::exp(static_cast<double>(v) - m);
    return m + static_cast<float>(std::log(acc));
}

}  // namespace

Potential::Potential() : log_(1, 0.0f) {}  // identity: P() = 1

Potential::Potential(std::vector<Variable> variables, std::vector<float> probabilities)
    : variables_(std::move(variables)), log_(probabilities.size()) {
    const int expected = product(dimsOf(variables_));
    if (static_cast<int>(probabilities.size()) != expected) {
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
    const int expected = product(dimsOf(p.variables_));
    if (static_cast<int>(p.log_.size()) != expected) {
        throw std::invalid_argument(
            "Potential::fromLog: log-table size does not match the product of "
            "variable state counts");
    }
    return p;
}

int Potential::numEntries() const { return product(dimsOf(variables_)); }

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
    const std::vector<int> dims = dimsOf(variables_);
    const std::vector<int> strides = rowMajorStrides(dims);
    int idx = 0;
    for (std::size_t i = 0; i < variables_.size(); ++i) {
        auto it = assignment.find(variables_[i]);
        if (it == assignment.end()) {
            throw std::invalid_argument(
                "Potential::probability: missing assignment for variable " +
                variables_[i].name());
        }
        if (it->second < 0 || it->second >= dims[i]) {
            throw std::invalid_argument(
                "Potential::probability: state out of range for variable " +
                variables_[i].name());
        }
        idx += it->second * strides[i];
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
    for (const Variable& v : other.variables_) {
        if (!contains(v)) result_vars.push_back(v);
    }

    // Map every result position to a position in each operand (or -1).
    std::vector<int> map_this(result_vars.size(), -1);
    std::vector<int> map_other(result_vars.size(), -1);
    for (std::size_t r = 0; r < result_vars.size(); ++r) {
        const int pa = positionOf(result_vars[r]);
        const int pb = other.positionOf(result_vars[r]);
        if (pa >= 0 && pb >= 0 &&
            variables_[pa].num_states() != other.variables_[pb].num_states()) {
            throw std::invalid_argument(
                "Potential::operator*: shared variable has mismatched state "
                "counts");
        }
        map_this[r] = pa;
        map_other[r] = pb;
    }

    const std::vector<int> rdims = dimsOf(result_vars);
    const std::vector<int> rstride = rowMajorStrides(rdims);
    const std::vector<int> adims = dimsOf(variables_);
    const std::vector<int> astride = rowMajorStrides(adims);
    const std::vector<int> bdims = dimsOf(other.variables_);
    const std::vector<int> bstride = rowMajorStrides(bdims);

    const int total = product(rdims);
    std::vector<float> out(total);
    std::vector<int> cur(result_vars.size(), 0);
    for (int k = 0; k < total; ++k) {
        int ia = 0, ib = 0;
        for (std::size_t r = 0; r < result_vars.size(); ++r) {
            if (map_this[r] >= 0) ia += cur[r] * astride[map_this[r]];
            if (map_other[r] >= 0) ib += cur[r] * bstride[map_other[r]];
        }
        out[k] = log_[ia] + other.log_[ib];
        for (int r = static_cast<int>(result_vars.size()) - 1; r >= 0; --r) {
            if (++cur[r] < rdims[r]) break;
            cur[r] = 0;
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
    const std::vector<int> dims = dimsOf(variables_);
    const std::vector<int> stride = rowMajorStrides(dims);
    const std::vector<int> ddims = dimsOf(divisor.variables_);
    const std::vector<int> dstride = rowMajorStrides(ddims);

    std::vector<int> map(divisor.variables_.size());
    for (std::size_t i = 0; i < divisor.variables_.size(); ++i) {
        map[i] = positionOf(divisor.variables_[i]);
    }

    const int total = numEntries();
    std::vector<float> out(total);
    std::vector<int> cur(variables_.size(), 0);
    for (int k = 0; k < total; ++k) {
        int id = 0;
        for (std::size_t i = 0; i < divisor.variables_.size(); ++i) {
            id += cur[map[i]] * dstride[i];
        }
        out[k] = log_[k] - divisor.log_[id];
        for (int r = static_cast<int>(variables_.size()) - 1; r >= 0; --r) {
            if (++cur[r] < dims[r]) break;
            cur[r] = 0;
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

    const std::vector<int> dims = dimsOf(variables_);
    const std::vector<int> stride = rowMajorStrides(dims);
    std::vector<int> keep_dim, sum_dim;
    for (int p : keep_pos) keep_dim.push_back(dims[p]);
    for (int p : sum_pos) sum_dim.push_back(dims[p]);

    const int total_keep = product(keep_dim);
    const int total_sum = product(sum_dim);
    std::vector<float> out(total_keep);
    std::vector<int> cur_keep(keep_pos.size(), 0);
    std::vector<int> cur_sum(sum_pos.size(), 0);

    for (int k = 0; k < total_keep; ++k) {
        std::vector<float> group(total_sum);
        std::fill(cur_sum.begin(), cur_sum.end(), 0);
        for (int s = 0; s < total_sum; ++s) {
            int idx = 0;
            for (std::size_t i = 0; i < keep_pos.size(); ++i) {
                idx += cur_keep[i] * stride[keep_pos[i]];
            }
            for (std::size_t i = 0; i < sum_pos.size(); ++i) {
                idx += cur_sum[i] * stride[sum_pos[i]];
            }
            group[s] = log_[idx];
            for (int r = static_cast<int>(sum_pos.size()) - 1; r >= 0; --r) {
                if (++cur_sum[r] < sum_dim[r]) break;
                cur_sum[r] = 0;
            }
        }
        out[k] = logSumExp(group);
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

    const std::vector<int> dims = dimsOf(variables_);
    const std::vector<int> stride = rowMajorStrides(dims);
    const std::vector<int> rdims = dimsOf(order);
    const std::vector<int> rstride = rowMajorStrides(rdims);

    std::vector<int> map(order.size());
    for (std::size_t i = 0; i < order.size(); ++i) map[i] = positionOf(order[i]);

    const int total = numEntries();
    std::vector<float> out(total);
    std::vector<int> cur(order.size(), 0);
    for (int k = 0; k < total; ++k) {
        int src = 0;
        for (std::size_t i = 0; i < order.size(); ++i) {
            src += cur[i] * stride[map[i]];
        }
        out[k] = log_[src];
        for (int r = static_cast<int>(order.size()) - 1; r >= 0; --r) {
            if (++cur[r] < rdims[r]) break;
            cur[r] = 0;
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

    const std::vector<int> dims = dimsOf(variables_);
    const std::vector<int> stride = rowMajorStrides(dims);
    std::vector<int> keep_dim;
    for (int p : keep_pos) keep_dim.push_back(dims[p]);

    const int total_keep = product(keep_dim);
    std::vector<float> out(total_keep);
    std::vector<int> cur_keep(keep_pos.size(), 0);
    for (int k = 0; k < total_keep; ++k) {
        int idx = 0;
        for (std::size_t i = 0; i < keep_pos.size(); ++i) {
            idx += cur_keep[i] * stride[keep_pos[i]];
        }
        for (int p : fix_pos) {
            auto it = assignment.find(variables_[p]);
            if (it->second < 0 || it->second >= dims[p]) {
                throw std::invalid_argument(
                    "Potential::restrict: state out of range");
            }
            idx += it->second * stride[p];
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
    std::vector<float> values = log_;
    const float z = logSumExp(values);
    for (float& v : log_) v -= z;
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

    const std::vector<int> dims = dimsOf(variables_);
    const std::vector<int> stride = rowMajorStrides(dims);
    std::vector<int> norm_dim, other_dim;
    for (int p : norm_pos) norm_dim.push_back(dims[p]);
    for (int p : other_pos) other_dim.push_back(dims[p]);

    const int total_other = product(other_dim);
    const int total_norm = product(norm_dim);
    std::vector<int> cur_other(other_pos.size(), 0);
    std::vector<int> cur_norm(norm_pos.size(), 0);

    for (int k = 0; k < total_other; ++k) {
        std::vector<float> group(total_norm);
        std::fill(cur_norm.begin(), cur_norm.end(), 0);
        for (int s = 0; s < total_norm; ++s) {
            int idx = 0;
            for (std::size_t i = 0; i < other_pos.size(); ++i) {
                idx += cur_other[i] * stride[other_pos[i]];
            }
            for (std::size_t i = 0; i < norm_pos.size(); ++i) {
                idx += cur_norm[i] * stride[norm_pos[i]];
            }
            group[s] = log_[idx];
            for (int r = static_cast<int>(norm_pos.size()) - 1; r >= 0; --r) {
                if (++cur_norm[r] < norm_dim[r]) break;
                cur_norm[r] = 0;
            }
        }
        const float z = logSumExp(group);
        std::fill(cur_norm.begin(), cur_norm.end(), 0);
        for (int s = 0; s < total_norm; ++s) {
            int idx = 0;
            for (std::size_t i = 0; i < other_pos.size(); ++i) {
                idx += cur_other[i] * stride[other_pos[i]];
            }
            for (std::size_t i = 0; i < norm_pos.size(); ++i) {
                idx += cur_norm[i] * stride[norm_pos[i]];
            }
            log_[idx] = group[s] - z;
            for (int r = static_cast<int>(norm_pos.size()) - 1; r >= 0; --r) {
                if (++cur_norm[r] < norm_dim[r]) break;
                cur_norm[r] = 0;
            }
        }
        for (int r = static_cast<int>(other_pos.size()) - 1; r >= 0; --r) {
            if (++cur_other[r] < other_dim[r]) break;
            cur_other[r] = 0;
        }
    }
    return *this;
}

}  // namespace bn
