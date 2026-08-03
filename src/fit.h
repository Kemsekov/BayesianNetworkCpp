#pragma once

#include <string>
#include <vector>

#include "inference.h"

namespace bn {

/// Fit a discrete Bayesian network from integer-coded data and return an
/// `Inference` object ready for exact queries.
///
/// Structure: a Chow-Liu tree — the maximum spanning tree over pairwise
/// empirical mutual information, which minimizes KL divergence among all
/// first-order dependency-tree approximations (Chow & Liu 1968).
/// Parameters: maximum-likelihood CPTs (relative frequencies).
///
/// @param data row-major `nrows` x `ncols` matrix of integer states; every
///             entry of column `c` must be in [0, states(c)) where
///             states(c) = max value in that column + 1.
/// @param nrows number of samples (rows)
/// @param ncols number of variables (columns)
/// @param names variable names; must be empty or contain exactly `ncols`
///              entries (empty -> "x1".."xn").
/// @throws std::invalid_argument on malformed input (negative states,
///         negative dimensions, size mismatch, wrong names count).
Inference fitBayesian(const std::vector<int>& data, int nrows, int ncols,
                      const std::vector<std::string>& names);

}  // namespace bn
