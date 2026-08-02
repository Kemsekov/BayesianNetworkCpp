#include <gtest/gtest.h>

#include <unordered_set>

#include "variable.h"

using bn::Variable;

TEST(Variable, BasicAccessors) {
    const Variable x(0, "X", 2);
    EXPECT_EQ(x.id(), 0);
    EXPECT_EQ(x.name(), "X");
    EXPECT_EQ(x.num_states(), 2);
}

TEST(Variable, EqualityIsById) {
    const Variable x(0, "X", 2);
    const Variable same(0, "renamed", 5);  // same id -> equal
    const Variable y(1, "Y", 3);
    EXPECT_EQ(x, same);
    EXPECT_NE(x, y);
}

TEST(Variable, OrderingIsById) {
    const Variable x(0, "X", 2);
    const Variable y(1, "Y", 3);
    const Variable z(2, "Z", 4);
    EXPECT_LT(x, y);
    EXPECT_LT(y, z);
    EXPECT_GT(z, x);
}

TEST(Variable, UsableAsUnorderedKey) {
    const Variable x(0, "X", 2), y(1, "Y", 2), duplicate(0, "X", 2);
    std::unordered_set<Variable, bn::VariableHash> set;
    set.insert(x);
    set.insert(y);
    set.insert(duplicate);
    EXPECT_EQ(set.size(), 2);
    EXPECT_TRUE(set.count(x));
    EXPECT_TRUE(set.count(y));
}

TEST(Variable, UsableAsMapKey) {
    const Variable x(0, "X", 2), y(1, "Y", 2);
    std::map<Variable, int> assignment;
    assignment[x] = 0;
    assignment[y] = 1;
    EXPECT_EQ(assignment.size(), 2);
    EXPECT_EQ(assignment[x], 0);
    EXPECT_EQ(assignment[y], 1);
}
