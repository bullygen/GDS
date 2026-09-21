/** @file network.cpp
 * @brief Привязки сохранённого интерфейса GDS к Python.
 */
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "gds/network.hpp"

namespace py = pybind11;
using gds::ForbiddenPair;
using gds::GDSNetwork;
using gds::RunResult;
using gds::StepInfo;
using gds::StopRule;
using gds::WeightedPair;

/** Регистрирует класс сети и служебные типы в общем модуле. */
void bind_network(py::module_& m) {


    py::enum_<StopRule>(m, "StopRule")
        .value("MAX_ITERATIONS", StopRule::MaxIterations)
        .value("SATISFIED_OR_MAX_ITERATIONS", StopRule::SatisfiedOrMaxIterations)
        .export_values();

    py::class_<ForbiddenPair>(m, "ForbiddenPair")
        .def(py::init<>())
        .def_readwrite("var_a", &ForbiddenPair::var_a)
        .def_readwrite("val_a", &ForbiddenPair::val_a)
        .def_readwrite("var_b", &ForbiddenPair::var_b)
        .def_readwrite("val_b", &ForbiddenPair::val_b);

    py::class_<WeightedPair>(m, "WeightedPair")
        .def(py::init<>())
        .def_readwrite("var_a", &WeightedPair::var_a)
        .def_readwrite("val_a", &WeightedPair::val_a)
        .def_readwrite("var_b", &WeightedPair::var_b)
        .def_readwrite("val_b", &WeightedPair::val_b)
        .def_readwrite("weight", &WeightedPair::weight);

    py::class_<StepInfo>(m, "StepInfo")
        .def(py::init<>())
        .def_readwrite("flipped", &StepInfo::flipped)
        .def_readwrite("variable", &StepInfo::variable)
        .def_readwrite("neuron_index", &StepInfo::neuron_index)
        .def_readwrite("old_value", &StepInfo::old_value)
        .def_readwrite("new_value", &StepInfo::new_value)
        .def_readwrite("score", &StepInfo::score)
        .def_readwrite("iteration", &StepInfo::iteration)
        .def_readwrite("transitions", &StepInfo::transitions);

    py::class_<RunResult>(m, "RunResult")
        .def(py::init<>())
        .def_readwrite("satisfied", &RunResult::satisfied)
        .def_readwrite("stopped_by_max_iterations", &RunResult::stopped_by_max_iterations)
        .def_readwrite("iterations", &RunResult::iterations)
        .def_readwrite("transitions", &RunResult::transitions)
        .def_readwrite("state", &RunResult::state)
        .def_readwrite("guards", &RunResult::guards)
        .def_readwrite("assignment", &RunResult::assignment);

    py::class_<GDSNetwork>(m, "GDSNetwork")
        .def(
            py::init<
                int,
                const std::vector<int>&,
                double,
                double,
                double,
                double,
                double,
                double,
                std::size_t,
                StopRule,
                const std::string&,
                const std::string&,
                const std::string&,
                std::uint64_t>(),
            py::arg("num_variables"),
            py::arg("domain_sizes"),
            py::arg("beta") = 1.0,
            py::arg("w_forbidden") = 4.0,
            py::arg("gamma_same") = 5.0,
            py::arg("guard_bias") = 0.5,
            py::arg("guard_delta") = 1.0,
            py::arg("guard_phi") = 8.0,
            py::arg("max_iterations") = 10000,
            py::arg("stop_rule") = StopRule::SatisfiedOrMaxIterations,
            py::arg("functional_type") = "binary_threshold",
            py::arg("update_rule") = "guarded_discrete_stochastic",
            py::arg("init_rule") = "all_off",
            py::arg("seed") = 42)
        .def_static(
            "make_n_queens",
            &GDSNetwork::make_n_queens,
            py::arg("n"),
            py::arg("beta") = 1.0,
            py::arg("w_forbidden") = 4.0,
            py::arg("gamma_same") = 5.0,
            py::arg("guard_bias") = 0.5,
            py::arg("guard_delta") = 1.0,
            py::arg("guard_phi") = 8.0,
            py::arg("max_iterations") = 10000,
            py::arg("stop_rule") = StopRule::SatisfiedOrMaxIterations,
            py::arg("seed") = 42)
        .def_static(
            "make_graph_3_coloring",
            &GDSNetwork::make_graph_3_coloring,
            py::arg("num_nodes"),
            py::arg("edges"),
            py::arg("beta") = 1.0,
            py::arg("w_forbidden") = 4.0,
            py::arg("gamma_same") = 5.0,
            py::arg("guard_bias") = 0.5,
            py::arg("guard_delta") = 1.0,
            py::arg("guard_phi") = 8.0,
            py::arg("max_iterations") = 10000,
            py::arg("stop_rule") = StopRule::SatisfiedOrMaxIterations,
            py::arg("seed") = 42)
        .def("add_forbidden_pair", &GDSNetwork::add_forbidden_pair)
        .def("clear_forbidden_pairs", &GDSNetwork::clear_forbidden_pairs)
        .def("add_weighted_pair", &GDSNetwork::add_weighted_pair)
        .def("clear_weighted_pairs", &GDSNetwork::clear_weighted_pairs)
        .def("finalize_weights", &GDSNetwork::finalize_weights)
        .def("set_unary_bias", &GDSNetwork::set_unary_bias)
        .def("get_unary_bias", &GDSNetwork::get_unary_bias)
        .def("add_nogood", &GDSNetwork::add_nogood)
        .def("step", &GDSNetwork::step)
        .def("run", &GDSNetwork::run, py::arg("override_max_iterations") = 0)
        .def("reset_state_all_off", &GDSNetwork::reset_state_all_off)
        .def("set_state", &GDSNetwork::set_state)
        .def("set_assignment", &GDSNetwork::set_assignment)
        .def("get_state", &GDSNetwork::get_state)
        .def("get_state_matrix", &GDSNetwork::get_state_matrix)
        .def("decode_assignment", &GDSNetwork::decode_assignment)
        .def("get_guard_state", &GDSNetwork::get_guard_state)
        .def("get_inputs", &GDSNetwork::get_inputs)
        .def("get_domain_sizes", &GDSNetwork::get_domain_sizes)
        .def("get_forbidden_pairs", &GDSNetwork::get_forbidden_pairs)
        .def("get_weighted_pairs", &GDSNetwork::get_weighted_pairs)
        .def("is_satisfied", &GDSNetwork::is_satisfied)
        .def("count_conflicts", &GDSNetwork::count_conflicts)
        .def("count_active_neurons", &GDSNetwork::count_active_neurons)
        .def("get_weight", &GDSNetwork::get_weight)
        .def("save_weights", &GDSNetwork::save_weights)
        .def_static("load_weights", &GDSNetwork::load_weights)
        .def_property("beta", &GDSNetwork::get_beta, &GDSNetwork::set_beta)
        .def_property("w_forbidden", &GDSNetwork::get_w_forbidden, &GDSNetwork::set_w_forbidden)
        .def_property("gamma_same", &GDSNetwork::get_gamma_same, &GDSNetwork::set_gamma_same)
        .def_property("guard_bias", &GDSNetwork::get_guard_bias, &GDSNetwork::set_guard_bias)
        .def_property("guard_delta", &GDSNetwork::get_guard_delta, &GDSNetwork::set_guard_delta)
        .def_property("guard_phi", &GDSNetwork::get_guard_phi, &GDSNetwork::set_guard_phi)
        .def_property("max_iterations", &GDSNetwork::get_max_iterations, &GDSNetwork::set_max_iterations)
        .def_property("stop_rule", &GDSNetwork::get_stop_rule, &GDSNetwork::set_stop_rule)
        .def_property("functional_type", &GDSNetwork::get_functional_type, &GDSNetwork::set_functional_type)
        .def_property("update_rule", &GDSNetwork::get_update_rule, &GDSNetwork::set_update_rule)
        .def_property("init_rule", &GDSNetwork::get_init_rule, &GDSNetwork::set_init_rule)
        .def_property("seed", &GDSNetwork::get_seed, &GDSNetwork::set_seed)
        .def_property_readonly("num_variables", &GDSNetwork::get_num_variables)
        .def_property_readonly("num_neurons", &GDSNetwork::get_num_neurons)
        .def_property_readonly("iterations", &GDSNetwork::get_iterations)
        .def_property_readonly("transitions", &GDSNetwork::get_transitions);
}
