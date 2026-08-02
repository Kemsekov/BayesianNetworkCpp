// pybind11 bindings for the C++ Bayesian network engine.
//
// Exposes Variable, Potential, Inference and JunctionTree with idiomatic
// Python names (snake_case), dict-based evidence/assignment maps, and
// exception translation (invalid_argument -> ValueError, runtime_error ->
// RuntimeError).
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <sstream>
#include <string>
#include <vector>

#include "inference.h"
#include "junction_tree.h"
#include "potential.h"
#include "variable.h"

namespace py = pybind11;

namespace {

std::string variableRepr(const bn::Variable& v) {
    return "Variable(" + std::to_string(v.id()) + ", " + v.name() + ", " +
           std::to_string(v.num_states()) + ")";
}

std::string potentialRepr(const bn::Potential& p) {
    const auto& vars = p.variables();
    std::ostringstream os;
    os << "Potential over {";
    for (std::size_t i = 0; i < vars.size(); ++i) {
        os << (i ? ", " : "") << vars[i].name();
    }
    os << "}\n";
    std::vector<int> cur(vars.size(), 0);
    std::vector<int> dims;
    for (const bn::Variable& v : vars) dims.push_back(v.num_states());
    const auto probs = p.probabilities();
    for (std::size_t k = 0; k < probs.size(); ++k) {
        os << "  P(";
        for (std::size_t i = 0; i < vars.size(); ++i) {
            os << (i ? ", " : "") << vars[i].name() << "=" << cur[i];
        }
        os << ") = " << probs[k] << "\n";
        for (int r = static_cast<int>(dims.size()) - 1; r >= 0; --r) {
            if (++cur[r] < dims[r]) break;
            cur[r] = 0;
        }
    }
    return os.str();
}

}  // namespace

PYBIND11_MODULE(_native, m) {
    m.doc() =
        "C++ Bayesian network inference engine: variable elimination and "
        "junction tree propagation.";

    py::register_exception_translator([](std::exception_ptr p) {
        try {
            if (p) std::rethrow_exception(p);
        } catch (const std::invalid_argument& e) {
            PyErr_SetString(PyExc_ValueError, e.what());
        } catch (const std::runtime_error& e) {
            PyErr_SetString(PyExc_RuntimeError, e.what());
        }
    });

    py::class_<bn::Variable>(m, "Variable")
        .def(py::init<int, std::string, int>(), py::arg("id"), py::arg("name"),
             py::arg("num_states"))
        .def_property_readonly("id", &bn::Variable::id)
        .def_property_readonly("name", &bn::Variable::name)
        .def_property_readonly("num_states", &bn::Variable::num_states)
        .def("__repr__", &variableRepr)
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def(py::self < py::self)
        .def("__hash__", [](const bn::Variable& v) { return v.id(); });

    py::class_<bn::Potential>(m, "Potential")
        .def(py::init<const std::vector<bn::Variable>&,
                      const std::vector<float>&>(),
             py::arg("variables"), py::arg("probabilities"))
        .def_property_readonly("variables", &bn::Potential::variables)
        .def_property_readonly("num_entries", &bn::Potential::numEntries)
        .def("probabilities", &bn::Potential::probabilities)
        .def("log_table", &bn::Potential::logTable)
        .def("value", &bn::Potential::probability, py::arg("assignment"))
        .def("marginalize", &bn::Potential::marginalize, py::arg("variables"))
        .def("reorder", &bn::Potential::reorder, py::arg("order"))
        .def("restrict", &bn::Potential::restrict, py::arg("assignment"))
        .def("normalize", [](bn::Potential& p) {
            p.normalize();
            return p;
        })
        .def(py::self * py::self)
        .def(py::self / py::self)
        .def("__repr__", &potentialRepr);

    py::class_<bn::Inference>(m, "Inference")
        .def(py::init<const std::vector<bn::Potential>&>(), py::arg("factors"))
        .def("full_joint", &bn::Inference::fullJoint)
        .def("marginal", &bn::Inference::marginal, py::arg("query"))
        .def("conditional", &bn::Inference::conditional, py::arg("query"),
             py::arg("evidence"))
        .def("conditional_given", &bn::Inference::conditionalGiven,
             py::arg("query"), py::arg("evidence"))
        .def("map_query", &bn::Inference::mapQuery, py::arg("query"),
             py::arg("evidence"));

    py::class_<bn::JunctionTree>(m, "JunctionTree")
        .def(py::init<const std::vector<bn::Potential>&>(), py::arg("factors"))
        .def("set_evidence", &bn::JunctionTree::setEvidence, py::arg("evidence"))
        .def("marginal", &bn::JunctionTree::marginal, py::arg("query"))
        .def("map_query", &bn::JunctionTree::mapQuery, py::arg("query"))
        .def_property_readonly("num_cliques", &bn::JunctionTree::numCliques)
        .def("clique_scope", &bn::JunctionTree::cliqueScope, py::arg("i"));
}
