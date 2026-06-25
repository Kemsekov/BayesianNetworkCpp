#include <iostream>
#include <vector>
#include <eigen3/Eigen/Dense>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>
#include <algorithm> // Required for std::erase
#include <iomanip> // Required for std::setw and std::setprecision
#include <unordered_map> //for dict
#include <memory> //for shared_ptr

#define MAX_CONDITIONAL_PARENTS 5
using ProbTable = Eigen::Tensor<float, MAX_CONDITIONAL_PARENTS, Eigen::RowMajor>;
using Index = Eigen::DenseIndex;
using DataView = Eigen::Map<Eigen::VectorXf>;

// Custom hash structure for std::vector<int>
struct VectorHasher {
    size_t operator()(const std::vector<int>& v) const {
        size_t seed = 0;
        for (int i : v) {
            // Combine hash of each element
            seed ^= std::hash<int>{}(i) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};

class Prob;
// Now you can define your map
using ProbCache = std::unordered_map<std::vector<int>, std::shared_ptr<Prob>, VectorHasher>;

/// @brief Struct that contains full joint probability
class Prob{
    private:
    // var1_uniques,var2_uniques,...
    // if len(variables)<MAX_CONDITIONAL_PARENTS, unused dimensions
    // are on the right, like P(x1,x2,x3) where |x1|=2, |x2|=|x3|=3
    // table shape is (3,3,2,1,1) for MAX_CONDITIONAL_PARENTS=5
    ProbTable log_table;
    mutable ProbCache condition_cache = {};
    mutable ProbCache marginalize_cache = {};

    // Return the Map by value.
    // This view is no longer valid if underlying data is destroyed or freed from memory,
    // So be cautious not to store this data views at all.
    // Resulting view is proper 1-dimensional flattened version of `log_table` ndarray
    // and can be used to do fast element-wise operations
    DataView data_view() {
        return DataView(log_table.data(), log_table.size());
    }

    public:
    Prob(){}
    Prob(ProbTable& new_log_table){
        log_table=new_log_table;
    }
    
    void update(ProbTable& new_log_table){
        log_table = new_log_table;
        condition_cache={};
        marginalize_cache={};
    }

    const ProbTable& get_table(){return log_table;}

    void normalize() {
        // 1. Map the underlying buffer to a 1D Vector (no copying)
        // Map<T> wraps raw memory and presents it as an Eigen expression
        auto flat_view = data_view();

        // 2. Find max for the log-sum-exp trick (stability)
        float max_val = flat_view.maxCoeff();
        
        // 3. Compute log-partition function using fast vector operations
        // This is vectorized automatically by Eigen
        float log_z = std::log((flat_view.array() - max_val).exp().sum()) + max_val;
        
        // 4. Subtract log_z in-place across the memory
        // Because flat_view is a Map, this updates log_table directly
        flat_view.array() -= log_z;
        condition_cache={};
        marginalize_cache={};
    }

    /// @brief Returns conditional distribution over provided indices.
    /// @param variable_reduce_indices 
    /// @return 
    const Prob& condition(std::vector<int> variable_reduce_indices)
    {
        // sort this stuff to hit cache
        std::sort(variable_reduce_indices.begin(), variable_reduce_indices.end());

        if(condition_cache.find(variable_reduce_indices)!=condition_cache.end()){
            return *condition_cache[variable_reduce_indices];
        }
        auto marginalized = marginalize(variable_reduce_indices);
        ProbTable new_table = log_table-marginalized.log_table;
        auto result = std::make_shared<Prob>(new_table);
        condition_cache[variable_reduce_indices]=result;
        return *result;
    }

    /// @brief Marginalize over provided indices of variables. 
    /// It simply computes sum over given variables.
    /// @param variable_reduce_indices indices of variables to marginalize over
    /// @return New marginalized `Prob` 
    const Prob& marginalize(std::vector<int> variable_reduce_indices) {
        // sort this stuff to hit cache
        std::sort(variable_reduce_indices.begin(), variable_reduce_indices.end());

        if(marginalize_cache.find(variable_reduce_indices)!=marginalize_cache.end()){
            return *marginalize_cache[variable_reduce_indices];
        }
        auto original_dims = log_table.dimensions();
        
        // 1. Calculate the target shape where reduced dimensions become 1
        Eigen::array<Index, MAX_CONDITIONAL_PARENTS> reduced_shape = original_dims;
        for (int idx : variable_reduce_indices) {
            reduced_shape[idx] = 1;
        }

        auto exp_table = log_table.exp();
        ProbTable new_table;

        // 2. Perform the reduction
        // We sum over the indices, then immediately reshape to collapse the dimensions to size 1
        switch (variable_reduce_indices.size()) {
            case 1: {
                Eigen::array<int, 1> sum_ind = {variable_reduce_indices[0]};
                new_table = exp_table.sum(sum_ind).log().reshape(reduced_shape);
                break;
            }
            case 2: {
                Eigen::array<int, 2> sum_ind = {variable_reduce_indices[0], variable_reduce_indices[1]};
                new_table = exp_table.sum(sum_ind).log().reshape(reduced_shape);
                break;
            }
            case 3: {
                Eigen::array<int, 3> sum_ind = {variable_reduce_indices[0], variable_reduce_indices[1], variable_reduce_indices[2]};
                new_table = exp_table.sum(sum_ind).log().reshape(reduced_shape);
                break;
            }
            case 4: {
                Eigen::array<int, 4> sum_ind = {variable_reduce_indices[0], variable_reduce_indices[1], variable_reduce_indices[2], variable_reduce_indices[3]};
                new_table = exp_table.sum(sum_ind).log().reshape(reduced_shape);
                break;
            }
            case 5: {
                Eigen::array<int, 5> sum_ind = {variable_reduce_indices[0], variable_reduce_indices[1], variable_reduce_indices[2], variable_reduce_indices[3], variable_reduce_indices[4]};
                new_table = exp_table.sum(sum_ind).log().reshape(reduced_shape);
                break;
            }
            default: {
                new_table = log_table;
                break;
            }
        }
        auto result = std::make_shared<Prob>(new_table);
        marginalize_cache[variable_reduce_indices]=result;
        return *result;
    }
};


// Recursive function to print nested lists
void print_recursive(const ProbTable& t, std::vector<int>& current_coords, int dim_idx) {
    auto dims = t.dimensions();
    
    // Base case: we are at the innermost dimension (e.g., node values)
    if (dim_idx == 2) { 
        std::cout << "[";
        for (int k = 0; k < dims[2]; ++k) {
            current_coords[2] = k;
            float val = std::exp(t(Eigen::array<Index, 5>{(Index)current_coords[0], 
                                                        (Index)current_coords[1], 
                                                        (Index)current_coords[2], 0, 0}));
            std::cout << std::fixed << std::setprecision(4) << val << (k == dims[2] - 1 ? "" : ", ");
        }
        std::cout << "]";
        return;
    }

    // Recursive step: print opening bracket and iterate through dimension
    std::cout << "[";
    for (int i = 0; i < dims[dim_idx]; ++i) {
        current_coords[dim_idx] = i;
        print_recursive(t, current_coords, dim_idx + 1);
        if (i < dims[dim_idx] - 1) std::cout << ",\n "; // Add newline and indent for readability
    }
    std::cout << "]";
}

void print_prob_table_numpy(const std::string& label, Prob& p) {
    std::cout << "\n" << label << "\n";
    std::vector<int> coords = {0, 0, 0};
    print_recursive(p.get_table(), coords, 0);
    std::cout << "\n";
}

void print_prob_table(const std::string& label, Prob& p) {
    std::cout << "\n=========================================\n";
    std::cout << "  " << label << "\n";
    std::cout << "=========================================";

    auto t = p.get_table();
    auto dims = t.dimensions();

    int d0 = (int)dims[0];
    int d1 = (int)dims[1];
    int d2 = (int)dims[2];

    // Set formatting: fixed-point notation, 4 decimal places, 12 character width
    std::cout << std::fixed << std::setprecision(4);

    for (int i = 0; i < d0; ++i) {
        std::cout << "\n--- Dim0 Index: " << i << " ---\n";
        for (int j = 0; j < d1; ++j) {
            std::cout << "  [" << j << "]: ";
            for (int k = 0; k < d2; ++k) {
                float val = std::exp(t(Eigen::array<Index, 5>{(Index)i, (Index)j, (Index)k, 0, 0}));
                // std::setw(10) ensures each value takes 10 characters of space
                std::cout << std::setw(10) << val << "  ";
            }
            std::cout << "\n";
        }
    }
}
int main() {
    // 1. Initialize P(X1 | X2, X3) with shape (3, 3, 2, 1, 1)
    Prob p;
    ProbTable log_table(3, 3, 2, 1, 1);
    
    // Seed random generator
    std::srand(std::time(nullptr));

    // Fill table with valid randomized conditional distributions
    // Each parent combination (p2, p3) must have its node (x1) values sum up to 1.0
    for (int p2 = 0; p2 < 3; ++p2) {
        for (int p3 = 0; p3 < 3; ++p3) {
            // Generate raw random weights for the node states (X1 has 2 states)
            float w0 = static_cast<float>(std::rand()) / RAND_MAX;
            float w1 = static_cast<float>(std::rand()) / RAND_MAX;
            
            // Assign into log_table in log-space
            log_table(Eigen::array<Index, 5>{(Index)p2, (Index)p3, 0, 0, 0}) = w0;
            log_table(Eigen::array<Index, 5>{(Index)p2, (Index)p3, 1, 0, 0}) = w1;
        }
    }
    p.update(log_table);
    p.normalize();
    // Print original randomized state to verify each cell coordinate pair sums to 1.0 down Node_Val
    print_prob_table_numpy("Original Validized Random P(X1, X2, X3)", p);

    // 2. Marginalize X2 (axis 0) -> Yields a 2D table layout: P(X1 | X3) with shape (3, 2, 1, 1, 1)
    // Note: Marginalization changes the scale because you are computing joint structures, 
    // these values will no longer sum to 1.0 until you explicitly divide by the new partition normalization factor!
    auto no_x2 = p.marginalize({0});
    print_prob_table_numpy("Marginalized X2 -> P(X1, X3)", no_x2);

    // 3. Marginalize BOTH X2 and X3 -> Yields a 1D vector distribution layout: P(X1) with shape (2, 1, 1, 1, 1)
    auto prior_x1 = p.marginalize({0, 1});
    print_prob_table_numpy("Marginalized Both X2, X3 -> P(X1)", prior_x1);

    return 0;
}
