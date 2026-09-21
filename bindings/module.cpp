/** @file module.cpp
 * @brief Точка входа расширения Python; вычислительные алгоритмы остаются в C++.
 */
#include <pybind11/pybind11.h>
namespace py = pybind11;
/** Регистрирует низкоуровневый интерфейс сети. */
void bind_network(py::module_& module);
/** Регистрирует постановку, внешние длительности и проверку. */
void bind_scheduling(py::module_& module);
/** Регистрирует алгоритмы поиска. */
void bind_solvers(py::module_& module);
/** Создаёт модуль, экспортируемый в пакет gds. */
PYBIND11_MODULE(_core, module) {
    module.doc() = "Планирование на C++: сеть GDS, ограничения и многокритериальный поиск.";
    bind_network(module); bind_scheduling(module); bind_solvers(module);
}
