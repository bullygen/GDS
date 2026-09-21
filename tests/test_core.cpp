/** @file test_core.cpp
 * @brief Автономные проверки C++ без Python и внешних тестовых библиотек.
 */
#include "gds/solvers.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>

/** Проверяет утверждение независимо от режима NDEBUG. */
void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
/** Проверяет ядро, постановку JSSP и независимую проверку общих ресурсов. */
int main() {
    try {
        auto queens = gds::GDSNetwork::make_n_queens(8);
        const auto solved = queens.run(100000);
        check(solved.satisfied && queens.count_conflicts() == 0, "Восемь ферзей не решены.");
        check(gds::duration_slots(1.01, 1) == 2, "Нарушено округление длительности.");
        auto problem = gds::make_jssp(2, 8, {{{0, 2}, {1, 2}}, {{1, 2}, {0, 2}}}).compile();
        gds::SolverOptions options; options.iterations = 2000;
        const auto schedule = gds::solve_gds(problem, options);
        check(schedule.feasible && schedule.placements.size() == 4, "Не решена задача JSSP.");
        check(problem.evaluate(schedule.assignment).feasible, "Независимая проверка отвергла план.");
        auto capacity = gds::make_parallel_jobs(1, 1, {1, 1, 1}, 2).compile();
        check(!capacity.evaluate({0, 0, 0}).feasible, "Пропущен конфликт трёх потребителей.");
        gds::DurationRequest request; request.slot_seconds = 0.5;
        auto duration = gds::estimate_duration(request, [](const gds::DurationRequest&, double t) {
            return t >= 1.1 ? gds::DurationStatus::Feasible : gds::DurationStatus::Infeasible;
        }, 10, true);
        check(duration.status == gds::DurationStatus::Feasible && duration.seconds == 1.5, "Неверный поиск длительности.");
        auto reducers = gds::find_reducers({{4, 0, 0}, {-1, 0, 0}});
        check(reducers.first == std::vector<int>{1}, "Неверно найдены уменьшающие момент наблюдения.");
        const std::vector<std::vector<double>> costs{{0, 1, 9}, {9, 0, 1}, {1, 9, 0}};
        check(gds::tour_cost({0, 1, 2}, costs) == 3, "Потеряна замыкающая дуга маршрута.");
        std::cout << "Проверки C++ завершены успешно.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n'; return 1;
    }
}
