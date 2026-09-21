/** @file solvers.hpp
 * @brief Стохастический поиск, исправление конфликтов и многокритериальная эволюция.
 */
#pragma once
#include "gds/scheduling.hpp"

namespace gds {
/** Общие конечные пределы поиска; одинаковое зерно воспроизводит ход в одной сборке. */
struct SolverOptions {
    std::uint64_t seed = 42;
    std::size_t iterations = 5000;
    int restarts = 8;
    int repair_steps = 200;
    int population = 20;
    int generations = 20;
    double mutation = 0.1;
    double differential_weight = 0.5;
    double crossover = 0.9;
    bool momentum_balancing = false;
};
/** Популяция и её недоминируемая часть; допустимость проверяется для каждого члена. */
struct PopulationResult {
    std::vector<Schedule> population;
    std::vector<Schedule> pareto_front;
    int generations = 0;
    std::size_t evaluations = 0;
};
/** Последовательно выбирает наименее конфликтное размещение каждой заявки. */
Schedule greedy(const CompiledProblem& problem, const SolverOptions& options = {});
/** Исправляет действие из наиболее конфликтной области; возвращает лучший найденный план. */
Schedule min_conflicts(const CompiledProblem& problem, const std::vector<int>& initial,
                       const SolverOptions& options = {});
/** Запускает GDS с повторными стартами и добавляет запреты отклонённых полных назначений. */
Schedule solve_gds(const CompiledProblem& problem, const SolverOptions& options = {});
/** Скрещивает расписания независимых запусков GDS с элитным отбором NSGA-II. */
PopulationResult genetic(const CompiledProblem& problem, const SolverOptions& options = {});
/** Выполняет GDE3 над нормированными координатами конечных областей размещения. */
PopulationResult gde3(const CompiledProblem& problem, const SolverOptions& options = {});
/** Проверяет строгое доминирование с первенством допустимого расписания. */
bool dominates(const Schedule& left, const Schedule& right);
/** Возвращает индексы недоминируемых расписаний; равные назначения не дублируются. */
std::vector<int> pareto_indices(const std::vector<Schedule>& schedules);

/** Маршрут, его стоимость и признак существования проверенного цикла. */
struct TourResult {
    std::vector<int> tour;
    double cost = 0.0;
    bool feasible = false;
    std::size_t iterations = 0;
};
/** Ищет коммивояжёрный цикл сетью GDS и сохраняет лучший допустимый маршрут. */
TourResult solve_tsp(const std::vector<std::vector<double>>& costs, const SolverOptions& options = {});
} // Пространство имён gds.
