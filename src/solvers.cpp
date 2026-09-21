/** @file solvers.cpp
 * @brief Реализация эвристик и эволюционных методов над общей C++-моделью.
 */
#include "gds/solvers.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <set>

namespace gds {
namespace {
/** Проверяет конечность и допустимые области параметров поиска. */
void validate(const SolverOptions& o) {
    if (o.iterations == 0 || o.restarts <= 0 || o.repair_steps < 0 || o.population < 4 || o.generations < 0 ||
        !std::isfinite(o.mutation) || o.mutation < 0 || o.mutation > 1 || !std::isfinite(o.crossover) || o.crossover < 0 || o.crossover > 1 ||
        !std::isfinite(o.differential_weight) || o.differential_weight <= 0 || o.differential_weight > 2)
        throw std::invalid_argument("Некорректные пределы или вероятности поиска.");
}
/** Суммирует величины нарушений; величина используется только среди недопустимых планов. */
double penalty(const Schedule& s) {
    double sum = 0; for (const auto& v : s.violations) sum += v.magnitude; return sum;
}
/** Сравнивает покомпонентные суммы нарушений одного вида, не складывая разные единицы. */
bool constraint_dominates(const Schedule& a, const Schedule& b, bool weak) {
    std::map<std::string, double> left, right;
    for (const auto& v : a.violations) left[v.kind] += v.magnitude;
    for (const auto& v : b.violations) right[v.kind] += v.magnitude;
    std::set<std::string> keys;
    for (const auto& x : left) keys.insert(x.first);
    for (const auto& x : right) keys.insert(x.first);
    bool strict = false;
    for (const auto& key : keys) { if (left[key] > right[key]) return false; strict = strict || left[key] < right[key]; }
    return weak || strict;
}
/** Сравнивает планы для однокритериального поиска с лексикографическими предпочтениями. */
bool better(const Schedule& a, const Schedule& b) {
    if (a.feasible != b.feasible) return a.feasible;
    if (a.violations.size() != b.violations.size()) return a.violations.size() < b.violations.size();
    if (penalty(a) != penalty(b)) return penalty(a) < penalty(b);
    return a.objectives < b.objectives;
}
/** Выбирает целое число из полуинтервала [0, count). */
int draw(std::mt19937_64& rng, int count) { return std::uniform_int_distribution<int>(0, count - 1)(rng); }
/** Выбирает число из полуинтервала [0, 1). */
double uniform(std::mt19937_64& rng) { return std::generate_canonical<double, 53>(rng); }
/** Строит случайное полное назначение без гарантии совместимости. */
std::vector<int> random_assignment(const std::vector<std::vector<Placement>>& domains, std::mt19937_64& rng) {
    std::vector<int> result; for (const auto& d : domains) result.push_back(draw(rng, static_cast<int>(d.size()))); return result;
}
/** Раскладывает множество планов на последовательные недоминируемые слои. */
std::vector<std::vector<int>> fronts(const std::vector<Schedule>& population) {
    std::vector<std::vector<int>> result; std::vector<int> remaining(population.size()); std::iota(remaining.begin(), remaining.end(), 0);
    while (!remaining.empty()) {
        std::vector<int> front;
        for (int i : remaining) {
            bool dominated = false;
            for (int j : remaining) if (i != j && dominates(population[j], population[i])) { dominated = true; break; }
            if (!dominated) front.push_back(i);
        }
        result.push_back(front); std::set<int> selected(front.begin(), front.end());
        remaining.erase(std::remove_if(remaining.begin(), remaining.end(), [&](int i) { return selected.count(i) != 0; }), remaining.end());
    }
    return result;
}
/** Вычисляет нормированное расстояние до соседей в пространстве критериев одного слоя. */
std::vector<double> crowding(const std::vector<Schedule>& population, const std::vector<int>& front) {
    std::vector<double> distance(population.size(), 0.0);
    if (front.empty()) return distance;
    for (std::size_t k = 0; k < population[front.front()].objectives.size(); ++k) {
        auto order = front;
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return population[a].objectives[k] < population[b].objectives[k]; });
        const double lo = population[order.front()].objectives[k], hi = population[order.back()].objectives[k];
        if (hi == lo) continue;
        distance[order.front()] = distance[order.back()] = std::numeric_limits<double>::infinity();
        for (std::size_t j = 1; j + 1 < order.size(); ++j) distance[order[j]] += (population[order[j + 1]].objectives[k] - population[order[j - 1]].objectives[k]) / (hi - lo);
    }
    return distance;
}
/** Отбирает заданное число индексов по рангу и расстоянию скученности NSGA-II. */
std::vector<int> select(const std::vector<Schedule>& population, int count) {
    std::vector<int> result;
    for (auto front : fronts(population)) {
        if (result.size() + front.size() > static_cast<std::size_t>(count)) {
            const auto distances = crowding(population, front);
            std::stable_sort(front.begin(), front.end(), [&](int a, int b) { return distances[a] > distances[b]; });
            front.resize(static_cast<std::size_t>(count) - result.size());
        }
        result.insert(result.end(), front.begin(), front.end());
        if (result.size() == static_cast<std::size_t>(count)) break;
    }
    return result;
}
/** Заполняет недоминируемую часть результата, сохраняя недопустимость явным признаком. */
void finish(PopulationResult& result) {
    for (int i : pareto_indices(result.population)) result.pareto_front.push_back(result.population[i]);
}
} // Внутреннее пространство имён.

bool dominates(const Schedule& a, const Schedule& b) {
    if (a.objectives.size() != b.objectives.size() || a.objectives.empty()) throw std::invalid_argument("Критерии должны иметь одинаковую ненулевую размерность.");
    for (double v : a.objectives) if (!std::isfinite(v)) throw std::invalid_argument("Критерии должны быть конечными.");
    for (double v : b.objectives) if (!std::isfinite(v)) throw std::invalid_argument("Критерии должны быть конечными.");
    if (a.feasible != b.feasible) return a.feasible;
    if (!a.feasible) return constraint_dominates(a, b, false);
    bool strict = false;
    for (std::size_t k = 0; k < a.objectives.size(); ++k) { if (a.objectives[k] > b.objectives[k]) return false; strict = strict || a.objectives[k] < b.objectives[k]; }
    return strict;
}
/** Возвращает индексы недоминируемых расписаний; равные назначения не дублируются. */
std::vector<int> pareto_indices(const std::vector<Schedule>& schedules) {
    if (schedules.empty()) return {};
    const auto layers = fronts(schedules); std::vector<int> result; std::set<std::pair<std::vector<int>, std::vector<double>>> seen;
    for (int i : layers.front()) if (seen.insert({schedules[i].assignment, schedules[i].objectives}).second) result.push_back(i);
    return result;
}

Schedule greedy(const CompiledProblem& problem, const SolverOptions& options) {
    validate(options); const auto domains = problem.domains(); const auto actions = problem.problem().actions();
    std::vector<int> assignment(domains.size(), -1), order(domains.size()); std::iota(order.begin(), order.end(), 0);
    std::set<int> reducers;
    if (options.momentum_balancing) {
        std::vector<Vector3> momenta; for (const auto& a : actions) momenta.push_back(a.modes.front().momentum);
        auto classified = find_reducers(momenta); reducers.insert(classified.first.begin(), classified.first.end());
    }
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
        if (reducers.count(a) != reducers.count(b)) return reducers.count(a) > reducers.count(b);
        if (actions[a].priority != actions[b].priority) return actions[a].priority > actions[b].priority;
        return domains[a].size() < domains[b].size();
    });
    std::size_t evaluations = 0;
    for (int i : order) {
        std::optional<Schedule> best;
        for (std::size_t v = 0; v < domains[i].size(); ++v) {
            assignment[i] = static_cast<int>(v); auto candidate = problem.evaluate(assignment); ++evaluations;
            if (!best || better(candidate, *best)) best = std::move(candidate);
        }
        assignment = best->assignment;
    }
    auto result = problem.evaluate(assignment); result.iterations = evaluations; result.stop_reason = "Жадный проход завершён."; return result;
}

Schedule min_conflicts(const CompiledProblem& problem, const std::vector<int>& initial, const SolverOptions& options) {
    validate(options); const auto domains = problem.domains(); std::mt19937_64 rng(options.seed);
    auto assignment = initial.empty() ? random_assignment(domains, rng) : initial;
    auto current = problem.evaluate(assignment); auto best = current; std::size_t steps = 0;
    for (; steps < static_cast<std::size_t>(options.repair_steps) && !current.feasible; ++steps) {
        std::vector<double> involvement(domains.size(), 0);
        for (const auto& v : current.violations) for (int a : v.actions) if (a >= 0 && a < static_cast<int>(involvement.size())) involvement[a] += v.magnitude;
        const double maximum = *std::max_element(involvement.begin(), involvement.end()); std::vector<int> candidates;
        for (std::size_t i = 0; i < involvement.size(); ++i) if (involvement[i] == maximum) candidates.push_back(static_cast<int>(i));
        const int variable = candidates[draw(rng, static_cast<int>(candidates.size()))];
        std::optional<Schedule> next; std::size_t ties = 0;
        for (std::size_t value = 0; value < domains[variable].size(); ++value) {
            if (static_cast<int>(value) == assignment[variable] && domains[variable].size() > 1) continue;
            auto trial = assignment; trial[variable] = static_cast<int>(value); auto evaluated = problem.evaluate(trial);
            if (!next || better(evaluated, *next)) { next = std::move(evaluated); ties = 1; }
            else if (!better(*next, evaluated) && draw(rng, static_cast<int>(++ties)) == 0) next = std::move(evaluated);
        }
        // Случайный шаг позволяет выйти из плоской области локального минимума.
        if (uniform(rng) < 0.08) {
            auto trial = assignment; trial[variable] = draw(rng, static_cast<int>(domains[variable].size())); next = problem.evaluate(trial);
        }
        current = std::move(*next); assignment = current.assignment;
        if (better(current, best)) best = current;
    }
    best.iterations = steps; best.stop_reason = best.feasible ? "Все ограничения выполнены." : "Исчерпан предел локального исправления."; return best;
}

Schedule solve_gds(const CompiledProblem& problem, const SolverOptions& options) {
    validate(options); std::optional<Schedule> best; std::vector<std::vector<int>> nogoods; std::size_t iterations = 0;
    for (int restart = 0; restart < options.restarts; ++restart) {
        auto net = problem.make_network(options.seed + static_cast<std::uint64_t>(restart));
        for (const auto& ng : nogoods) net.add_nogood(ng);
        for (std::size_t step = 0; step < options.iterations; ++step) {
            net.step(); ++iterations;
            if (!net.is_satisfied()) continue;
            auto candidate = problem.evaluate(net.decode_assignment());
            if (!best || better(candidate, *best)) best = candidate;
            if (candidate.feasible) { candidate.iterations = iterations; candidate.stop_reason = "Сеть и независимая проверка подтвердили расписание."; return candidate; }
            net.add_nogood(candidate.assignment); nogoods.push_back(candidate.assignment);
        }
        SolverOptions repair = options; repair.seed += static_cast<std::uint64_t>(restart);
        auto candidate = min_conflicts(problem, net.decode_assignment(), repair);
        if (!best || better(candidate, *best)) best = candidate;
        if (candidate.feasible) { candidate.iterations += iterations; candidate.stop_reason = "Расписание сети исправлено и проверено."; return candidate; }
    }
    best->iterations += iterations; best->stop_reason = "Предел поиска исчерпан; допустимый план не найден."; return *best;
}

PopulationResult genetic(const CompiledProblem& problem, const SolverOptions& options) {
    validate(options); std::mt19937_64 rng(options.seed); PopulationResult result; const auto domains = problem.domains();
    for (int i = 0; i < options.population; ++i) {
        auto o = options; o.seed = rng(); o.restarts = 1;
        result.population.push_back(solve_gds(problem, o)); ++result.evaluations;
    }
    for (int generation = 0; generation < options.generations; ++generation) {
        const auto layers = fronts(result.population); std::vector<int> ranks(result.population.size()); std::vector<double> distances(result.population.size());
        for (std::size_t k = 0; k < layers.size(); ++k) { const auto d = crowding(result.population, layers[k]); for (int i : layers[k]) { ranks[i] = static_cast<int>(k); distances[i] = d[i]; } }
        // Турнир выбирает меньший ранг, затем большее расстояние скученности.
        auto parent = [&]() { int a = draw(rng, options.population), b = draw(rng, options.population);
            return ranks[a] < ranks[b] || (ranks[a] == ranks[b] && distances[a] >= distances[b]) ? a : b; };
        auto pool = result.population;
        for (int i = 0; i < options.population; ++i) {
            const auto& a = result.population[parent()].assignment; const auto& b = result.population[parent()].assignment; auto child = a;
            for (std::size_t k = 0; k < child.size(); ++k) { if (uniform(rng) < 0.5) child[k] = b[k]; if (child[k] < 0 || uniform(rng) < options.mutation) child[k] = draw(rng, static_cast<int>(domains[k].size())); }
            auto repair = options; repair.seed = rng(); pool.push_back(min_conflicts(problem, child, repair)); ++result.evaluations;
        }
        result.population.clear(); for (int i : select(pool, options.population)) result.population.push_back(std::move(pool[i]));
        result.generations = generation + 1;
    }
    finish(result); return result;
}

PopulationResult gde3(const CompiledProblem& problem, const SolverOptions& options) {
    validate(options); std::mt19937_64 rng(options.seed); const auto domains = problem.domains();
    std::vector<std::vector<double>> vectors; PopulationResult result;
    // Упорядоченная область задаёт кусочно-постоянное отображение [0,1] в размещения.
    auto decode = [&](const std::vector<double>& x) { std::vector<int> a;
        for (std::size_t k = 0; k < x.size(); ++k) a.push_back(std::min(static_cast<int>(domains[k].size()) - 1, static_cast<int>(x[k] * domains[k].size())));
        return a;
    };
    for (int i = 0; i < options.population; ++i) {
        std::vector<double> x(domains.size()); for (double& v : x) v = uniform(rng);
        // Один исходный член использует эвристику балансировки из статьи о JWST.
        if (i == 0) { const auto seed = greedy(problem, options); for (std::size_t k = 0; k < x.size(); ++k) x[k] = (seed.assignment[k] + 0.5) / domains[k].size(); }
        result.population.push_back(problem.evaluate(decode(x))); vectors.push_back(std::move(x)); ++result.evaluations;
    }
    for (int generation = 0; generation < options.generations; ++generation) {
        std::vector<Schedule> pool; std::vector<std::vector<double>> pool_vectors;
        for (int i = 0; i < options.population; ++i) {
            std::vector<int> donors; while (donors.size() < 3) { const int j = draw(rng, options.population); if (j != i && std::find(donors.begin(), donors.end(), j) == donors.end()) donors.push_back(j); }
            auto trial = vectors[i]; const int forced = draw(rng, static_cast<int>(trial.size()));
            for (std::size_t k = 0; k < trial.size(); ++k) if (static_cast<int>(k) == forced || uniform(rng) < options.crossover)
                trial[k] = std::clamp(vectors[donors[0]][k] + options.differential_weight * (vectors[donors[1]][k] - vectors[donors[2]][k]), 0.0, 1.0);
            auto candidate = problem.evaluate(decode(trial)); ++result.evaluations; const auto& parent = result.population[i];
            bool keep_trial = !dominates(parent, candidate), keep_parent = !dominates(candidate, parent);
            // При равных критериях достаточно потомка; два недопустимых сравниваются по нарушению.
            if ((!candidate.feasible && !parent.feasible) || candidate.objectives == parent.objectives) {
                keep_trial = candidate.feasible == parent.feasible ? constraint_dominates(candidate, parent, true) : candidate.feasible;
                keep_parent = !keep_trial;
            }
            if (keep_parent) { pool.push_back(parent); pool_vectors.push_back(vectors[i]); }
            if (keep_trial) { pool.push_back(std::move(candidate)); pool_vectors.push_back(std::move(trial)); }
        }
        result.population.clear(); vectors.clear();
        for (int i : select(pool, options.population)) { result.population.push_back(std::move(pool[i])); vectors.push_back(std::move(pool_vectors[i])); }
        result.generations = generation + 1;
    }
    finish(result); return result;
}

TourResult solve_tsp(const std::vector<std::vector<double>>& costs, const SolverOptions& options) {
    validate(options); TourResult best; best.cost = std::numeric_limits<double>::infinity();
    for (int restart = 0; restart < options.restarts; ++restart) {
        auto net = make_tsp_network(costs, options.seed + static_cast<std::uint64_t>(restart));
        for (std::size_t i = 0; i < options.iterations; ++i) {
            net.step(); ++best.iterations;
            if (net.is_satisfied()) { const auto tour = net.decode_assignment(); const double cost = tour_cost(tour, costs);
                if (cost < best.cost) { best.cost = cost; best.tour = tour; best.feasible = true; }
                // Выход из устойчивого допустимого цикла позволяет исследовать другие перестановки.
                net.add_nogood(tour);
            }
        }
    }
    return best;
}
} // Пространство имён gds.
