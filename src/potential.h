#pragma once

#include <map>
#include <vector>

#include "variable.h"

namespace bn {

/// A table-based factor (probability potential) defined over an ordered set of
/// random variables. The table stores one entry per joint assignment of its
/// variables; internally the entries are kept in log-space for numerical
/// stability, while the public API works with probabilities in normal space.
///
/// The variables are ordered; the first variable is the slowest-varying
/// dimension (row-major layout), matching the convention used across the
/// codebase. A potential with no variables is the multiplicative identity
/// (single entry equal to 1).
class Potential {
public:
    /// Default-constructs the multiplicative identity potential.
    Potential();

    /// Builds a potential over `variables` from probabilities in normal space.
    /// `probabilities` must contain product(states) entries in row-major order.
    /// @throws std::invalid_argument on a size mismatch or a negative value.
    Potential(std::vector<Variable> variables, std::vector<float> probabilities);

    /// Builds a potential directly from log-space entries. Used internally.
    static Potential fromLog(std::vector<Variable> variables,
                             std::vector<float> log_table);

    /// Variables in scope order (first = outermost / slowest-varying).
    const std::vector<Variable>& variables() const { return variables_; }
    /// Log-space table, row-major over `variables()`.
    const std::vector<float>& logTable() const { return log_; }

    int numVariables() const { return static_cast<int>(variables_.size()); }
    /// Number of table entries = product of variable state counts.
    int numEntries() const;

    bool contains(const Variable& v) const;
    bool contains(int variable_id) const;
    /// Position (0-based) of `v` in the scope, or -1 if not present.
    int positionOf(const Variable& v) const;
    int positionOfId(int variable_id) const;

    /// Probability (normal space) at a full assignment covering every variable
    /// of the potential. Extra keys in the assignment are ignored.
    float probability(const std::map<Variable, int>& assignment) const;
    /// Normal-space probabilities in row-major order.
    std::vector<float> probabilities() const;

    /// Join (multiplication) over the union of both scopes. In log-space this
    /// is simply an addition of the two broadcast tables.
    /// @throws std::invalid_argument when a shared variable has mismatched states.
    Potential operator*(const Potential& other) const;
    /// Divide by a potential whose scope is a subset of this scope, producing a
    /// potential over this scope. In log-space this is a broadcast subtraction.
    Potential operator/(const Potential& divisor) const;
    /// Sum out the given variables (they must all be in scope).
    Potential marginalize(const std::vector<Variable>& to_sum_out) const;
    /// Reorder the scope; `order` must be a permutation of the current variables.
    Potential reorder(const std::vector<Variable>& order) const;
    /// Fix a subset of variables to the given states, returning a potential
    /// over the remaining variables.
    Potential restrict(const std::map<Variable, int>& assignment) const;

    /// Normalize all entries so they sum to 1.
    Potential& normalize();
    /// Normalize only the given dimensions: for every joint assignment of the
    /// remaining variables, the entries sum to 1.
    Potential& normalizeOver(const std::vector<Variable>& to_normalize);

private:
    void computeCached();

    std::vector<Variable> variables_;
    std::vector<float> log_;  // log-space entries, row-major over variables_
    std::vector<int> dims_;     // per-variable state counts
    std::vector<int> strides_;  // row-major strides for dims_
    int num_entries_ = 1;       // product of dims_
};

}  // namespace bn
