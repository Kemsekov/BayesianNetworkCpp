#pragma once

#include <cstddef>
#include <functional>
#include <string>
#include <utility>

namespace bn {

/// A random variable: a unique integer id, a human-readable name and a finite
/// number of mutually exclusive states. Two variables are considered equal
/// when their ids match, which makes variables usable as map / set keys even
/// if they were constructed with different names.
class Variable {
public:
    Variable() = default;
    Variable(int id, std::string name, int num_states)
        : id_(id), name_(std::move(name)), num_states_(num_states) {}

    int id() const { return id_; }
    const std::string& name() const { return name_; }
    int num_states() const { return num_states_; }

    bool operator==(const Variable& other) const { return id_ == other.id_; }
    bool operator!=(const Variable& other) const { return id_ != other.id_; }
    bool operator<(const Variable& other) const { return id_ < other.id_; }
    bool operator>(const Variable& other) const { return id_ > other.id_; }

private:
    int id_ = -1;
    std::string name_;
    int num_states_ = 0;
};

struct VariableHash {
    std::size_t operator()(const Variable& v) const {
        return std::hash<int>{}(v.id());
    }
};

}  // namespace bn
