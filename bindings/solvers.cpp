/** @file solvers.cpp
 * @brief Прямые привязки C++-решателей без повторения алгоритмов на Python.
 */
#include "gds/solvers.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;
using namespace gds;
/** Экспортирует параметры, результаты и функции поиска с именованными аргументами. */
void bind_solvers(py::module_& m) {
    py::class_<SolverOptions>(m, "SolverOptions", "Пределы поиска и вероятности эволюционных операторов.")
        .def(py::init<>())
        .def_readwrite("seed", &SolverOptions::seed)
        .def_readwrite("iterations", &SolverOptions::iterations)
        .def_readwrite("restarts", &SolverOptions::restarts)
        .def_readwrite("repair_steps", &SolverOptions::repair_steps)
        .def_readwrite("population", &SolverOptions::population)
        .def_readwrite("generations", &SolverOptions::generations)
        .def_readwrite("mutation", &SolverOptions::mutation)
        .def_readwrite("differential_weight", &SolverOptions::differential_weight)
        .def_readwrite("crossover", &SolverOptions::crossover)
        .def_readwrite("momentum_balancing", &SolverOptions::momentum_balancing);
    py::class_<PopulationResult>(m, "PopulationResult", "Итоговая популяция и приближение границы Парето.")
        .def_readonly("population", &PopulationResult::population)
        .def_readonly("pareto_front", &PopulationResult::pareto_front)
        .def_readonly("generations", &PopulationResult::generations)
        .def_readonly("evaluations", &PopulationResult::evaluations);
    py::class_<TourResult>(m, "TourResult", "Проверенный замкнутый маршрут коммивояжёра.")
        .def_readonly("tour", &TourResult::tour).def_readonly("cost", &TourResult::cost)
        .def_readonly("feasible", &TourResult::feasible).def_readonly("iterations", &TourResult::iterations);
    m.def("greedy", &greedy, py::arg("problem"), py::arg("options") = SolverOptions{}, "Выполняет жадный проход по заявкам.");
    m.def("min_conflicts", &min_conflicts, py::arg("problem"), py::arg("initial") = std::vector<int>{}, py::arg("options") = SolverOptions{}, "Исправляет наиболее конфликтные области расписания.");
    m.def("solve_gds", &solve_gds, py::arg("problem"), py::arg("options") = SolverOptions{}, "Ищет расписание сетью GDS с независимой проверкой.");
    m.def("genetic", &genetic, py::arg("problem"), py::arg("options") = SolverOptions{}, "Выполняет генетический отбор расписаний сети GDS.");
    m.def("gde3", &gde3, py::arg("problem"), py::arg("options") = SolverOptions{}, "Выполняет многокритериальную дифференциальную эволюцию GDE3.");
    m.def("dominates", &dominates, "Проверяет доминирование с первенством допустимого решения.");
    m.def("pareto_indices", &pareto_indices, "Возвращает индексы недоминируемых решений.");
    m.def("solve_tsp", &solve_tsp, py::arg("costs"), py::arg("options") = SolverOptions{}, "Ищет замкнутый маршрут сетью GDS.");
}
