/** @file scheduling.cpp
 * @brief Привязки постановки задач и внешних физических моделей.
 */
#include "gds/scheduling.hpp"
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;
using namespace gds;
/** Регистрирует типы данных и вызывает C++ непосредственно из Python. */
void bind_scheduling(py::module_& m) {
    py::class_<Resource>(m, "Resource", "Ресурс группировки с ограничениями ёмкости, памяти и момента.")
        .def(py::init<>())
        .def_readwrite("name", &Resource::name)
        .def_readwrite("capacity", &Resource::capacity)
        .def_readwrite("sequence", &Resource::sequence)
        .def_readwrite("memory_capacity", &Resource::memory_capacity)
        .def_readwrite("initial_memory", &Resource::initial_memory)
        .def_readwrite("initial_momentum", &Resource::initial_momentum)
        .def_readwrite("momentum_limit", &Resource::momentum_limit)
        .def_readwrite("initial_state", &Resource::initial_state)
        .def_readwrite("initial_pointing", &Resource::initial_pointing)
        .def_readwrite("initial_level", &Resource::initial_level)
        .def_readwrite("minimum_level", &Resource::minimum_level)
        .def_readwrite("maximum_level", &Resource::maximum_level);
    py::class_<Mode>(m, "Mode", "Способ исполнения заявки на аппарате.")
        .def(py::init<>())
        .def_readwrite("resource", &Mode::resource)
        .def_readwrite("duration", &Mode::duration)
        .def_readwrite("demands", &Mode::demands)
        .def_readwrite("momentum", &Mode::momentum)
        .def_readwrite("momentum_profile", &Mode::momentum_profile)
        .def_readwrite("flows", &Mode::flows)
        .def_readwrite("memory", &Mode::memory)
        .def_readwrite("downlink_rate", &Mode::downlink_rate)
        .def_readwrite("remaining_momentum", &Mode::remaining_momentum);
    py::class_<Action>(m, "Action", "Заявка с временными окнами и альтернативами.")
        .def(py::init<>())
        .def_readwrite("name", &Action::name)
        .def_readwrite("group", &Action::group)
        .def_readwrite("modes", &Action::modes)
        .def_readwrite("windows", &Action::windows)
        .def_readwrite("suitability", &Action::suitability)
        .def_readwrite("pointing", &Action::pointing)
        .def_readwrite("priority", &Action::priority)
        .def_readwrite("optional", &Action::optional);
    py::class_<Placement>(m, "Placement", "Координаты выбранного нейрона в расписании.")
        .def(py::init<>())
        .def_readwrite("action", &Placement::action)
        .def_readwrite("mode", &Placement::mode)
        .def_readwrite("start", &Placement::start)
        .def_readwrite("end", &Placement::end)
        .def_readwrite("resource", &Placement::resource)
        .def_readwrite("duration", &Placement::duration)
        .def_readwrite("log_suitability", &Placement::log_suitability)
        .def_readwrite("omitted", &Placement::omitted);
    py::class_<DurationRequest>(m, "DurationRequest", "Запрос внешней модели с временем и полным состоянием.")
        .def(py::init<>())
        .def_readwrite("transition", &DurationRequest::transition)
        .def_readwrite("before", &DurationRequest::before)
        .def_readwrite("after", &DurationRequest::after)
        .def_readwrite("mode", &DurationRequest::mode)
        .def_readwrite("resource", &DurationRequest::resource)
        .def_readwrite("start_seconds", &DurationRequest::start_seconds)
        .def_readwrite("slot_seconds", &DurationRequest::slot_seconds)
        .def_readwrite("nominal_seconds", &DurationRequest::nominal_seconds)
        .def_readwrite("from_pointing", &DurationRequest::from_pointing)
        .def_readwrite("to_pointing", &DurationRequest::to_pointing)
        .def_readwrite("state", &DurationRequest::state)
        .def_readwrite("model_version", &DurationRequest::model_version);
    py::class_<DurationResult>(m, "DurationResult", "Статус, длительность и сертификат внешнего расчёта.")
        .def(py::init<>())
        .def_readwrite("status", &DurationResult::status)
        .def_readwrite("seconds", &DurationResult::seconds)
        .def_readwrite("risk", &DurationResult::risk)
        .def_readwrite("cost", &DurationResult::cost)
        .def_readwrite("certificate", &DurationResult::certificate)
        .def_readwrite("diagnostic", &DurationResult::diagnostic)
        .def_readwrite("end_state", &DurationResult::end_state);
    py::class_<SlotSample>(m, "SlotSample", "Выборочные оценки функции пригодности внутри слота.")
        .def(py::init<>())
        .def_readwrite("minimum", &SlotSample::minimum)
        .def_readwrite("maximum", &SlotSample::maximum)
        .def_readwrite("mean", &SlotSample::mean)
        .def_readwrite("aliasing_warning", &SlotSample::aliasing_warning);
    py::class_<Violation>(m, "Violation", "Нарушение ограничения и участвующие действия.")
        .def(py::init<>())
        .def_readwrite("kind", &Violation::kind)
        .def_readwrite("message", &Violation::message)
        .def_readwrite("actions", &Violation::actions)
        .def_readwrite("begin", &Violation::begin)
        .def_readwrite("end", &Violation::end)
        .def_readwrite("magnitude", &Violation::magnitude);
    py::class_<Segment>(m, "Segment", "Полоса действия, перехода или простоя.")
        .def(py::init<>())
        .def_readwrite("kind", &Segment::kind)
        .def_readwrite("resource", &Segment::resource)
        .def_readwrite("action", &Segment::action)
        .def_readwrite("begin", &Segment::begin)
        .def_readwrite("end", &Segment::end)
        .def_readwrite("metadata", &Segment::metadata);
    py::class_<ResourceTrace>(m, "ResourceTrace", "Изменение загрузки, памяти и момента по слотам.")
        .def(py::init<>())
        .def_readwrite("resource", &ResourceTrace::resource)
        .def_readwrite("load", &ResourceTrace::load)
        .def_readwrite("memory", &ResourceTrace::memory)
        .def_readwrite("momentum", &ResourceTrace::momentum)
        .def_readwrite("level", &ResourceTrace::level);
    py::class_<Schedule>(m, "Schedule", "Расписание с результатом независимой проверки.")
        .def(py::init<>())
        .def_readwrite("assignment", &Schedule::assignment)
        .def_readwrite("placements", &Schedule::placements)
        .def_readwrite("violations", &Schedule::violations)
        .def_readwrite("segments", &Schedule::segments)
        .def_readwrite("traces", &Schedule::traces)
        .def_readwrite("objectives", &Schedule::objectives)
        .def_readwrite("feasible", &Schedule::feasible)
        .def_readwrite("exact", &Schedule::exact)
        .def_readwrite("iterations", &Schedule::iterations)
        .def_readwrite("stop_reason", &Schedule::stop_reason);
    py::class_<Flow>(m, "Flow", "Изменение накопительного ресурса во время и при окончании действия.")
        .def(py::init<int,double,double>(), py::arg("resource"), py::arg("rate") = 0.0, py::arg("at_end") = 0.0)
        .def_readwrite("resource", &Flow::resource).def_readwrite("rate", &Flow::rate).def_readwrite("at_end", &Flow::at_end);
    m.def("angular_visibility", &angular_visibility, py::arg("pointing"), py::arg("body"), py::arg("observer"), py::arg("limit_degrees"), py::arg("at_least") = true, "Строит маску угловой видимости на C++.");
    py::class_<Window>(m, "Window", "Полуоткрытое временное окно в слотах.")
        .def(py::init<int,int>(), py::arg("begin"), py::arg("end"))
        .def_readwrite("begin", &Window::begin).def_readwrite("end", &Window::end);
    py::class_<Demand>(m, "Demand", "Потребность в дополнительном ресурсе.")
        .def(py::init<int,double>(), py::arg("resource"), py::arg("amount") = 1.0)
        .def_readwrite("resource", &Demand::resource).def_readwrite("amount", &Demand::amount);
    py::class_<Precedence>(m, "Precedence", "Зависимость начала от окончания предшественника.")
        .def(py::init<int,int,int,int>(), py::arg("before"), py::arg("after"), py::arg("min_gap") = 0, py::arg("max_gap") = -1)
        .def_readwrite("before", &Precedence::before).def_readwrite("after", &Precedence::after)
        .def_readwrite("min_gap", &Precedence::min_gap).def_readwrite("max_gap", &Precedence::max_gap);
    py::class_<Quota>(m, "Quota", "Минимум выполненных действий группы.")
        .def(py::init<std::string,int>(), py::arg("group"), py::arg("minimum"))
        .def_readwrite("group", &Quota::group).def_readwrite("minimum", &Quota::minimum);
    py::class_<Operation>(m, "Operation", "Операция работы на заданном станке.")
        .def(py::init<int,int>(), py::arg("machine"), py::arg("duration"))
        .def_readwrite("machine", &Operation::machine).def_readwrite("duration", &Operation::duration);
    py::enum_<DurationStatus>(m, "DurationStatus", "Достоверность внешнего расчёта.")
        .value("FEASIBLE", DurationStatus::Feasible).value("INFEASIBLE", DurationStatus::Infeasible)
        .value("UNKNOWN", DurationStatus::Unknown).value("TIMEOUT", DurationStatus::Timeout).value("STALE", DurationStatus::Stale);
    py::class_<Problem>(m, "Problem", "Описание задачи до компиляции областей значений.")
        .def(py::init<int,double>(), py::arg("horizon"), py::arg("slot_seconds") = 1.0)
        .def("add_resource", &Problem::add_resource, "Добавляет ресурс и возвращает его индекс.")
        .def("add_action", &Problem::add_action, "Добавляет заявку и возвращает её индекс.")
        .def("add_precedence", &Problem::add_precedence, "Добавляет зависимость действий.")
        .def("add_quota", &Problem::add_quota, "Добавляет квоту группы.")
        .def("add_contiguous_group", &Problem::add_contiguous_group, "Задаёт непрерывный блок группы.")
        .def("add_pair_suitability", &Problem::add_pair_suitability, py::arg("first"), py::arg("second"), py::arg("function"), "Задаёт парную функцию пригодности перед компиляцией.")
        .def("set_duration_provider", &Problem::set_duration_provider, py::arg("provider"), py::arg("version"), "Подключает внешний расчёт длительности действий.")
        .def("set_transition_provider", &Problem::set_transition_provider, py::arg("provider"), py::arg("version"), py::arg("initial") = false, "Подключает внешний расчёт переходов.")
        .def("set_exact_transition_provider", &Problem::set_exact_transition_provider, py::arg("provider"), py::arg("version"), "Подключает точную проверку переходов.")
        .def_property_readonly("horizon", &Problem::horizon)
        .def_property_readonly("slot_seconds", &Problem::slot_seconds)
        .def_property_readonly("resources", &Problem::resources)
        .def_property_readonly("actions", &Problem::actions)
        .def("compile", &Problem::compile, "Строит конечную модель на C++.");
    py::class_<CompiledProblem>(m, "CompiledProblem", "Скомпилированные размещения и независимая проверка.")
        .def("evaluate", &CompiledProblem::evaluate, py::arg("assignment"), py::arg("exact") = true, "Проверяет ограничения и материализует соседние переходы.")
        .def("make_network", &CompiledProblem::make_network, py::arg("seed") = 42, "Создаёт сеть по функциям пригодности и ограничениям.")
        .def_property_readonly("domains", &CompiledProblem::domains)
        .def_property_readonly("problem", &CompiledProblem::problem)
        .def("cache_statistics", &CompiledProblem::cache_statistics, "Возвращает число вызовов модели и попаданий в кэш.")
        .def("clear_cache", &CompiledProblem::clear_cache, "Удаляет кэш ответов внешней модели.");
    m.def("duration_slots", &duration_slots, "Округляет физическую длительность вверх до целого числа слотов.");
    m.def("quantize_interval", &quantize_interval, py::arg("begin"), py::arg("end"), py::arg("quantum"), py::arg("mode") = "cover", "Квантует непрерывный интервал.");
    m.def("estimate_duration", &estimate_duration, py::arg("request"), py::arg("predicate"), py::arg("max_slots"), py::arg("monotone"), "Ищет минимальную допустимую сеточную длительность.");
    m.def("radec_to_unit", &radec_to_unit, "Преобразует небесные координаты в единичный вектор.");
    m.def("angular_separation", &angular_separation, "Вычисляет угловое расстояние в градусах.");
    m.def("log_suitability", &log_suitability, "Суммирует логарифмы неотрицательных свидетельств пригодности.");
    m.def("sample_slot", &sample_slot, py::arg("function"), py::arg("begin"), py::arg("end"), py::arg("samples") = 17, py::arg("aliasing_ratio") = 4.0, "Исследует функцию в равномерных узлах слота.");
    m.def("make_jssp", &make_jssp, py::arg("machines"), py::arg("horizon"), py::arg("jobs"), "Строит ограничения цепочек операций JSSP.");
    m.def("make_parallel_jobs", &make_parallel_jobs, py::arg("machines"), py::arg("horizon"), py::arg("durations"), py::arg("capacity") = 1, "Строит независимые работы на взаимозаменяемых ресурсах.");
    m.def("make_tsp_network", &make_tsp_network, py::arg("costs"), py::arg("seed") = 42, "Строит сеть для замкнутого направленного маршрута.");
    m.def("tour_cost", &tour_cost, "Проверяет маршрут и вычисляет стоимость замыкающего цикла.");
    m.def("find_reducers", &find_reducers, "Находит кандидатов, уменьшающих норму суммарного момента.");
}
