/** @file scheduling.cpp
 * @brief Квантование, компиляция ограничений и расчёт проверяемого расписания.
 */
#include "gds/scheduling.hpp"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <numeric>
#include <set>
#include <sstream>

namespace gds {
namespace {
/** Проверяет условие описания задачи и выдаёт диагностическое исключение. */
void require(bool condition, const std::string& message) {
    if (!condition) throw std::invalid_argument(message);
}
/** Проверяет конечность трёх координат. */
bool finite(const Vector3& v) {
    return std::all_of(v.begin(), v.end(), [](double x) { return std::isfinite(x); });
}
/** Возвращает евклидову норму без промежуточного переполнения квадратов. */
double norm(const Vector3& v) { return std::hypot(v[0], v[1], v[2]); }
/** Возвращает суммарную потребность способа исполнения в данном ресурсе. */
double demand(const Mode& mode, int resource) {
    double result = mode.resource == resource ? 1.0 : 0.0;
    for (const auto& d : mode.demands) if (d.resource == resource) result += d.amount;
    return result;
}
/** Определяет пересечение полуоткрытых интервалов. */
bool overlap(const Placement& a, const Placement& b) {
    return !a.omitted && !b.omitted && a.start < b.end && b.start < a.end;
}
/** Проверяет размерность и значения матрицы направленных стоимостей. */
void validate_costs(const std::vector<std::vector<double>>& costs) {
    require(costs.size() >= 2, "Требуются хотя бы два города.");
    for (const auto& row : costs) {
        require(row.size() == costs.size(), "Матрица стоимостей должна быть квадратной.");
        for (double c : row) require(!std::isnan(c) && c >= 0, "Стоимость должна быть неотрицательной либо бесконечной.");
    }
}
} // Внутреннее пространство имён.

/** Консервативно округляет длительность вверх; отвергает неконечные и отрицательные числа. */
int duration_slots(double seconds, double slot_seconds) {
    require(std::isfinite(seconds) && seconds >= 0, "Длительность должна быть конечной и неотрицательной.");
    require(std::isfinite(slot_seconds) && slot_seconds > 0, "Шаг времени должен быть конечным и положительным.");
    const double n = std::ceil(seconds / slot_seconds);
    require(n <= std::numeric_limits<int>::max(), "Длительность не помещается в целое число слотов.");
    return static_cast<int>(n);
}

/** Квантует непрерывный интервал: cover покрывает, inside вписывает, nearest округляет. */
Window quantize_interval(double begin, double end, double quantum, const std::string& mode) {
    require(std::isfinite(begin) && std::isfinite(end) && begin >= 0 && end > begin,
            "Границы интервала должны быть конечными и возрастающими.");
/** Консервативно округляет длительность вверх; отвергает неконечные и отрицательные числа. */
    duration_slots(end, quantum);
    double a = begin / quantum, b = end / quantum;
    if (mode == "cover") { a = std::floor(a); b = std::ceil(b); }
    else if (mode == "inside") { a = std::ceil(a); b = std::floor(b); }
    else if (mode == "nearest") { a = std::round(a); b = std::round(b); }
    else throw std::invalid_argument("Неизвестный способ квантования.");
    require(b > a, "Интервал исчезает при выбранном квантовании.");
    return {static_cast<int>(a), static_cast<int>(b)};
}

/** Преобразует прямое восхождение и склонение в единичный вектор. */
Vector3 radec_to_unit(double ra, double dec) {
    require(std::isfinite(ra) && std::isfinite(dec) && std::abs(dec) <= 90, "Недопустимые небесные координаты.");
    constexpr double rad = 3.14159265358979323846 / 180.0;
    ra *= rad; dec *= rad;
    return {std::cos(dec) * std::cos(ra), std::cos(dec) * std::sin(ra), std::sin(dec)};
}

/** Вычисляет угол между ненулевыми векторами в градусах. */
double angular_separation(const Vector3& a, const Vector3& b) {
    require(finite(a) && finite(b) && norm(a) > 0 && norm(b) > 0, "Направления должны быть конечными ненулевыми векторами.");
    double dot = 0;
    for (int k = 0; k < 3; ++k) dot += (a[k] / norm(a)) * (b[k] / norm(b));
    return std::acos(std::clamp(dot, -1.0, 1.0)) * 180.0 / 3.14159265358979323846;
}

/** Вычисляет логарифм произведения неотрицательных функций пригодности. */
double log_suitability(const std::vector<double>& evidence) {
    double result = 0;
    for (double value : evidence) {
        require(std::isfinite(value) && value >= 0, "Пригодность должна быть конечной и неотрицательной.");
        result += std::log(value);
    }
    return result;
}

/** Строит маску угловой видимости по положениям тела и наблюдателя на каждом слоте. */
std::vector<double> angular_visibility(const Vector3& pointing, const std::vector<Vector3>& body,
                                      const std::vector<Vector3>& observer, double limit, bool at_least) {
    require(!body.empty() && (observer.empty() || observer.size() == body.size()), "Не совпадают длины рядов положений.");
    require(std::isfinite(limit) && limit >= 0 && limit <= 180, "Угловой предел должен лежать от нуля до 180 градусов.");
    std::vector<double> mask;
    for (std::size_t t = 0; t < body.size(); ++t) {
        auto relative = body[t];
        if (!observer.empty()) for (int k = 0; k < 3; ++k) relative[k] -= observer[t][k];
        const double angle = angular_separation(pointing, relative);
        mask.push_back(at_least ? angle >= limit : angle <= limit);
    }
    return mask;
}

SlotSample sample_slot(const std::function<double(double)>& f, double begin, double end, int samples, double ratio) {
    require(std::isfinite(begin) && std::isfinite(end) && end > begin && samples >= 3,
            "Нужны конечный непустой интервал и хотя бы три узла.");
    require(std::isfinite(ratio) && ratio > 0, "Порог изменчивости должен быть положительным.");
    SlotSample result{std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity(), 0, false};
    for (int i = 0; i < samples; ++i) {
        const double y = f(begin + (end - begin) * i / (samples - 1));
        require(std::isfinite(y), "Функция вернула неконечное значение.");
        result.minimum = std::min(result.minimum, y); result.maximum = std::max(result.maximum, y);
        result.mean += y * (i == 0 || i == samples - 1 ? 0.5 : 1.0) / (samples - 1);
    }
    result.aliasing_warning = (result.maximum - result.minimum) / std::max(std::abs(result.mean), 1e-12) > ratio;
    return result;
}

DurationResult estimate_duration(const DurationRequest& request, const DurationPredicate& predicate, int max_slots, bool monotone) {
    require(max_slots >= 0 && static_cast<bool>(predicate), "Нужны предикат и неотрицательная граница поиска.");
/** Консервативно округляет длительность вверх; отвергает неконечные и отрицательные числа. */
    duration_slots(0, request.slot_seconds);
    DurationResult result;
    // Неопределённый ответ немедленно останавливает поиск: его нельзя считать ложью.
    auto probe = [&](int k) { return predicate(request, k * request.slot_seconds); };
    if (!monotone) {
        for (int k = 0; k <= max_slots; ++k) {
            result.status = probe(k);
            if (result.status == DurationStatus::Feasible) { result.seconds = k * request.slot_seconds; return result; }
            if (result.status != DurationStatus::Infeasible) return result;
        }
        return result;
    }
    result.status = probe(0);
    if (result.status != DurationStatus::Infeasible || max_slots == 0) return result;
    result.status = probe(max_slots);
    if (result.status != DurationStatus::Feasible) return result;
    int lo = 0, hi = max_slots;
    while (hi - lo > 1) {
        const int mid = lo + (hi - lo) / 2;
        result.status = probe(mid);
        if (result.status == DurationStatus::Feasible) hi = mid;
        else if (result.status == DurationStatus::Infeasible) lo = mid;
        else return result;
    }
    result.status = DurationStatus::Feasible; result.seconds = hi * request.slot_seconds;
    return result;
}

/** Создаёт конечный горизонт и проверяет положительность временного шага. */
Problem::Problem(int horizon, double quantum) : horizon_(horizon), quantum_(quantum) {
    require(horizon > 0, "Горизонт должен содержать хотя бы один слот.");
/** Консервативно округляет длительность вверх; отвергает неконечные и отрицательные числа. */
    duration_slots(0, quantum);
    require(std::isfinite(horizon * quantum), "Горизонт в секундах должен быть конечным.");
}
/** Добавляет ресурс и возвращает его индекс. */
int Problem::add_resource(const Resource& resource) { resources_.push_back(resource); return static_cast<int>(resources_.size()) - 1; }
/** Добавляет заявку и возвращает её индекс. */
int Problem::add_action(const Action& action) { actions_.push_back(action); return static_cast<int>(actions_.size()) - 1; }
/** Добавляет условие предшествования с необязательной верхней границей разрыва. */
void Problem::add_precedence(const Precedence& constraint) { precedences_.push_back(constraint); }
/** Добавляет квоту на число выполненных заявок группы. */
void Problem::add_quota(const Quota& quota) { quotas_.push_back(quota); }
/** Запрещает вставку других групп внутрь блока указанной группы на одном аппарате. */
void Problem::add_contiguous_group(const std::string& group) { contiguous_.push_back(group); }
/** Задаёт парную функцию для двух действий; вызывается только при компиляции. */
void Problem::add_pair_suitability(int first, int second, PairSuitability function) {
    require(static_cast<bool>(function), "Парная функция пригодности не задана.");
    pair_functions_.emplace_back(first, second, std::move(function));
}
/** Устанавливает внешний расчёт длительности действий и версию модели. */
void Problem::set_duration_provider(DurationProvider provider, const std::string& version) {
    require(static_cast<bool>(provider) && !version.empty(), "Нужны провайдер и версия модели.");
    duration_provider_ = std::move(provider); duration_version_ = version;
}
/** Устанавливает расчёт соседних переходов, включая начальный переход при initial=true. */
void Problem::set_transition_provider(DurationProvider provider, const std::string& version, bool initial) {
    require(static_cast<bool>(provider) && !version.empty(), "Нужны провайдер и версия модели.");
    transition_provider_ = std::move(provider); transition_version_ = version; initial_transition_ = initial;
}
/** Фиксирует отдельный точный проверяющий расчёт переходов вместо приближённого. */
void Problem::set_exact_transition_provider(DurationProvider provider, const std::string& version) {
    require(static_cast<bool>(provider) && !version.empty(), "Нужны провайдер и версия модели.");
    exact_transition_provider_ = std::move(provider); exact_version_ = version;
}
/** Возвращает число слотов горизонта. */
int Problem::horizon() const { return horizon_; }
/** Возвращает шаг сетки в секундах. */
double Problem::slot_seconds() const { return quantum_; }
/** Возвращает копию ресурсов. */
std::vector<Resource> Problem::resources() const { return resources_; }
/** Возвращает копию заявок. */
std::vector<Action> Problem::actions() const { return actions_; }
/** Строит конечные области значений и парные ограничения исключительно на C++. */
CompiledProblem Problem::compile() const { return CompiledProblem(*this); }

/** Проверяет описание и строит допустимые размещения и пары. */
CompiledProblem::CompiledProblem(const Problem& problem) : problem_(problem) {
    const auto& p = problem_;
    require(!p.resources_.empty() && !p.actions_.empty(), "Задача должна содержать ресурсы и действия.");
    for (const auto& r : p.resources_) {
        require(!r.name.empty() && std::isfinite(r.capacity) && r.capacity > 0, "Ресурсу нужны имя и положительная ёмкость.");
        require(!r.sequence || r.capacity == 1.0, "Последовательная линия должна иметь единичную ёмкость.");
        require(std::isfinite(r.memory_capacity) && r.memory_capacity >= 0 && std::isfinite(r.initial_memory)
                && r.initial_memory >= 0 && r.initial_memory <= r.memory_capacity, "Некорректные пределы памяти.");
        require(finite(r.initial_momentum) && std::isfinite(r.momentum_limit) && r.momentum_limit >= norm(r.initial_momentum),
                "Некорректные пределы момента.");
        for (double x : r.initial_state) require(std::isfinite(x), "Начальное состояние должно быть конечным.");
        require(finite(r.initial_pointing) && norm(r.initial_pointing) > 0, "Начальное направление должно быть ненулевым конечным вектором.");
        require(std::isfinite(r.initial_level) && std::isfinite(r.minimum_level) && std::isfinite(r.maximum_level)
                && r.minimum_level <= r.initial_level && r.initial_level <= r.maximum_level, "Некорректные пределы накопительного ресурса.");
    }
    for (const auto& c : p.precedences_) {
        require(c.before >= 0 && c.after >= 0 && c.before < static_cast<int>(p.actions_.size())
                && c.after < static_cast<int>(p.actions_.size()) && c.before != c.after,
                "Недопустимые индексы предшествования.");
        require(c.min_gap >= 0 && (c.max_gap < 0 || c.max_gap >= c.min_gap), "Некорректные границы разрыва.");
    }
    for (const auto& q : p.quotas_) require(!q.group.empty() && q.minimum >= 0, "Некорректная квота.");
    for (const auto& f : p.pair_functions_) require(std::get<0>(f) >= 0 && std::get<1>(f) >= 0
        && std::get<0>(f) < static_cast<int>(p.actions_.size()) && std::get<1>(f) < static_cast<int>(p.actions_.size())
        && std::get<0>(f) != std::get<1>(f), "Недопустимые индексы парной функции.");
    for (std::size_t i = 0; i < p.actions_.size(); ++i) {
        const auto& a = p.actions_[i];
        require(!a.name.empty() && !a.modes.empty(), "Действию нужны имя и способ исполнения.");
        require(std::isfinite(a.priority) && a.priority >= 0 && finite(a.pointing) && norm(a.pointing) > 0,
                "Некорректные приоритет или направление.");
        require(a.suitability.empty() || a.suitability.size() == static_cast<std::size_t>(p.horizon_), "Длина пригодности не равна горизонту.");
        for (double v : a.suitability) require(std::isfinite(v) && v >= 0, "Пригодность должна быть конечной и неотрицательной.");
        for (auto w : a.windows) require(w.begin >= 0 && w.end > w.begin && w.end <= p.horizon_, "Окно выходит за горизонт.");
        std::vector<Placement> domain;
        for (std::size_t m = 0; m < a.modes.size(); ++m) {
            const auto& mode = a.modes[m];
            require(mode.resource >= 0 && mode.resource < static_cast<int>(p.resources_.size()), "Неизвестный основной ресурс.");
            require(std::isfinite(mode.duration) && mode.duration > 0 && finite(mode.momentum), "Некорректная длительность или момент.");
            require(mode.momentum_profile.empty() || mode.momentum_profile.size() == static_cast<std::size_t>(p.horizon_), "Длина профиля момента не равна горизонту.");
            for (const auto& h : mode.momentum_profile) require(finite(h), "Профиль момента должен быть конечным.");
            for (const auto& f : mode.flows) require(f.resource >= 0 && f.resource < static_cast<int>(p.resources_.size())
                    && std::isfinite(f.rate) && std::isfinite(f.at_end), "Некорректный поток накопительного ресурса.");
            require(std::isfinite(mode.memory) && mode.memory >= 0 && std::isfinite(mode.downlink_rate) && mode.downlink_rate >= 0
                    && std::isfinite(mode.remaining_momentum) && mode.remaining_momentum >= 0 && mode.remaining_momentum <= 1,
                    "Некорректные параметры памяти или разгрузки момента.");
            for (const auto& d : mode.demands) require(d.resource >= 0 && d.resource < static_cast<int>(p.resources_.size())
                    && std::isfinite(d.amount) && d.amount >= 0, "Некорректная потребность в ресурсе.");
            for (int t = 0; t < p.horizon_; ++t) {
                DurationRequest request;
                request.after = static_cast<int>(i); request.mode = static_cast<int>(m); request.resource = mode.resource;
                request.start_seconds = t * p.quantum_; request.slot_seconds = p.quantum_; request.nominal_seconds = mode.duration;
                request.state = p.resources_[mode.resource].initial_state; request.to_pointing = a.pointing;
                request.model_version = p.duration_version_;
                const auto duration = query(request, false);
                if (duration.status != DurationStatus::Feasible || duration.seconds <= 0) continue;
                const int length = duration_slots(duration.seconds, p.quantum_);
                if (length > p.horizon_ - t) continue;
                const int end = t + length;
                if (!a.windows.empty() && !std::any_of(a.windows.begin(), a.windows.end(),
                    [&](Window w) { return t >= w.begin && end <= w.end; })) continue;
                double score = 0;
                bool allowed = true;
                for (int k = t; k < end && !a.suitability.empty(); ++k) {
                    if (a.suitability[k] == 0) { allowed = false; break; }
                    score += std::log(a.suitability[k]) / length;
                }
                for (std::size_t r = 0; r < p.resources_.size(); ++r)
                    if (demand(mode, static_cast<int>(r)) > p.resources_[r].capacity) allowed = false;
                if (allowed) domain.push_back({static_cast<int>(i), static_cast<int>(m), t, end, mode.resource, duration.seconds, score, false});
            }
        }
        if (a.optional) domain.push_back({static_cast<int>(i), -1, -1, -1, -1, 0, 0, true});
        require(!domain.empty(), "Пустая область допустимых размещений действия: " + a.name);
        domains_.push_back(std::move(domain));
    }
    // Парные запреты точны для пересечений и предшествования. Общие ёмкости
    // и зависящие от истории переходы проверяются по полному расписанию.
    for (std::size_t i = 0; i < domains_.size(); ++i) for (std::size_t j = i + 1; j < domains_.size(); ++j)
        for (std::size_t u = 0; u < domains_[i].size(); ++u) for (std::size_t v = 0; v < domains_[j].size(); ++v) {
            const auto& a = domains_[i][u]; const auto& b = domains_[j][v]; bool bad = false;
            if (overlap(a, b)) for (std::size_t r = 0; r < p.resources_.size(); ++r)
                if (demand(p.actions_[i].modes[a.mode], static_cast<int>(r)) + demand(p.actions_[j].modes[b.mode], static_cast<int>(r)) > p.resources_[r].capacity) bad = true;
            for (const auto& c : p.precedences_) if ((c.before == static_cast<int>(i) && c.after == static_cast<int>(j)) || (c.before == static_cast<int>(j) && c.after == static_cast<int>(i))) {
                const auto& before = c.before == static_cast<int>(i) ? a : b;
                const auto& after = c.after == static_cast<int>(j) ? b : a;
                if (!after.omitted) {
                    const int gap = after.start - before.end;
                    if (before.omitted || gap < c.min_gap || (c.max_gap >= 0 && gap > c.max_gap)) bad = true;
                }
            }
            double log_preference = 0;
            if (!a.omitted && !b.omitted) for (const auto& f : p.pair_functions_) {
                const int first = std::get<0>(f), second = std::get<1>(f);
                if ((first == static_cast<int>(i) && second == static_cast<int>(j)) || (first == static_cast<int>(j) && second == static_cast<int>(i))) {
                    const double suitability = first == static_cast<int>(i) ? std::get<2>(f)(a, b) : std::get<2>(f)(b, a);
                    require(std::isfinite(suitability) && suitability >= 0, "Парная пригодность должна быть конечной и неотрицательной.");
                    if (suitability == 0) bad = true; else log_preference += std::log(suitability);
                }
            }
            if (bad) forbidden_.push_back({static_cast<int>(i), static_cast<int>(u), static_cast<int>(j), static_cast<int>(v)});
            else if (log_preference != 0) preferences_.push_back({static_cast<int>(i), static_cast<int>(u), static_cast<int>(j), static_cast<int>(v), log_preference});
        }
}

/** Выполняет кэшируемый вызов провайдера с полным состоянием в ключе. */
DurationResult CompiledProblem::query(const DurationRequest& request, bool exact) const {
    const auto& provider = request.transition ? (exact && problem_.exact_transition_provider_ ? problem_.exact_transition_provider_ : problem_.transition_provider_) : problem_.duration_provider_;
    if (!provider) { DurationResult r; r.status = DurationStatus::Feasible; r.seconds = request.nominal_seconds; return r; }
    std::ostringstream key;
    key << std::setprecision(17) << request.transition << ':' << exact << ':' << request.before << ':' << request.after << ':' << request.mode
        << ':' << request.resource << ':' << request.start_seconds << ':' << request.slot_seconds << ':' << request.nominal_seconds << ':' << request.model_version;
    for (double v : request.from_pointing) key << ':' << v;
    for (double v : request.to_pointing) key << ':' << v;
    key << ":состояние:" << request.state.size();
    for (double v : request.state) key << ':' << v;
    const auto found = cache_.find(key.str());
    if (found != cache_.end()) { ++hits_; return found->second; }
    ++calls_;
    auto result = provider(request);
    if (result.status == DurationStatus::Feasible) {
/** Консервативно округляет длительность вверх; отвергает неконечные и отрицательные числа. */
        duration_slots(result.seconds, request.slot_seconds);
        require(std::isfinite(result.risk) && result.risk >= 0 && std::isfinite(result.cost) && result.cost >= 0,
                "Внешняя модель вернула некорректные риск или стоимость.");
        for (double x : result.end_state) require(std::isfinite(x), "Внешняя модель вернула неконечное состояние.");
    }
    // Временные ошибки не кэшируются: повторный вызов может завершиться успешно.
    if (result.status == DurationStatus::Feasible || result.status == DurationStatus::Infeasible) cache_[key.str()] = result;
    return result;
}

/** Независимо проверяет назначения, накопительные ресурсы и фактическую последовательность переходов. */
Schedule CompiledProblem::evaluate(const std::vector<int>& assignment, bool exact) const {
    require(assignment.size() == domains_.size(), "Число назначений не равно числу действий.");
    const auto& p = problem_;
    Schedule out; out.assignment = assignment; out.exact = exact; out.objectives.assign(5, 0.0);
    // Каждое нарушение хранит участников, чтобы исправление охватывало всю причину.
    auto violation = [&](std::string kind, std::string message, std::vector<int> actions, int begin, int end, double magnitude = 1.0) {
        out.violations.push_back({std::move(kind), std::move(message), std::move(actions), begin, end, magnitude});
    };
    for (std::size_t i = 0; i < assignment.size(); ++i) {
        require(assignment[i] >= -1 && assignment[i] < static_cast<int>(domains_[i].size()), "Индекс размещения вне области значений.");
        if (assignment[i] < 0) { violation("unassigned", "Действие не назначено.", {static_cast<int>(i)}, -1, -1); continue; }
        const auto place = domains_[i][assignment[i]]; out.placements.push_back(place);
        if (place.omitted) out.objectives[0] += p.actions_[i].priority;
        else out.objectives[3] -= place.log_suitability;
    }
    for (const auto& pair : forbidden_) if (assignment[pair.var_a] == pair.val_a && assignment[pair.var_b] == pair.val_b)
        violation("pair", "Нарушены совместимость ресурсов или предшествование.", {pair.var_a, pair.var_b},
                  std::min(domains_[pair.var_a][pair.val_a].start, domains_[pair.var_b][pair.val_b].start),
                  std::max(domains_[pair.var_a][pair.val_a].end, domains_[pair.var_b][pair.val_b].end));
    for (const auto& pair : preferences_) if (assignment[pair.var_a] == pair.val_a && assignment[pair.var_b] == pair.val_b)
        out.objectives[3] -= pair.weight;
    for (const auto& q : p.quotas_) {
        int count = 0; std::vector<int> members;
        for (const auto& x : out.placements) if (p.actions_[x.action].group == q.group) { members.push_back(x.action); count += !x.omitted; }
        if (count < q.minimum) violation("quota", "Не выполнена квота группы " + q.group, members, 0, p.horizon_, q.minimum - count);
    }
    for (std::size_t r = 0; r < p.resources_.size(); ++r) {
        const auto& resource = p.resources_[r];
        ResourceTrace trace; trace.resource = static_cast<int>(r);
        double memory = resource.initial_memory, level = resource.initial_level; Vector3 momentum = resource.initial_momentum;
        std::vector<int> history;
        for (int t = 0; t < p.horizon_; ++t) {
            double load = 0, released = 0, recorded = 0, level_delta = 0; std::vector<int> active;
            Vector3 delta{0, 0, 0}; double remaining = 1.0;
            for (const auto& x : out.placements) if (!x.omitted) {
                const auto& m = p.actions_[x.action].modes[x.mode];
                if (x.start <= t && t < x.end) {
                    const double d = demand(m, static_cast<int>(r)); load += d;
                    if (d > 0) active.push_back(x.action);
                    if (x.resource == static_cast<int>(r)) released += m.downlink_rate * p.quantum_;
                    for (const auto& f : m.flows) if (f.resource == static_cast<int>(r)) {
                        level_delta += f.rate * p.quantum_;
                        if (std::find(history.begin(), history.end(), x.action) == history.end()) history.push_back(x.action);
                    }
                }
                if (x.end == t + 1) for (const auto& f : m.flows) if (f.resource == static_cast<int>(r)) {
                    level_delta += f.at_end;
                    if (std::find(history.begin(), history.end(), x.action) == history.end()) history.push_back(x.action);
                }
                if (x.resource == static_cast<int>(r) && x.end == t + 1) {
                    history.push_back(x.action); recorded += m.memory; remaining *= m.remaining_momentum;
                    for (int k = 0; k < 3; ++k) delta[k] += m.momentum_profile.empty() ? m.momentum[k] : m.momentum_profile[x.start][k];
                }
            }
            memory = std::max(0.0, memory - released) + recorded;
            level += level_delta;
            for (int k = 0; k < 3; ++k) momentum[k] = remaining * momentum[k] + delta[k];
            trace.load.push_back(load); trace.memory.push_back(memory); trace.momentum.push_back(momentum); trace.level.push_back(level);
            if (level < resource.minimum_level || level > resource.maximum_level)
                violation("cumulative", "Накопительный ресурс вышел за пределы: " + resource.name, history, t, t + 1,
                          std::max(resource.minimum_level - level, level - resource.maximum_level));
            if (load > resource.capacity + 1e-12) violation("capacity", "Превышена ёмкость ресурса " + resource.name, active, t, t + 1, load - resource.capacity);
            if (memory > resource.memory_capacity + 1e-12) violation("memory", "Переполнена память ресурса " + resource.name, history, t, t + 1, memory - resource.memory_capacity);
            if (norm(momentum) > resource.momentum_limit + 1e-12) violation("momentum", "Превышена норма момента ресурса " + resource.name, history, t, t + 1, norm(momentum) - resource.momentum_limit);
        }
        double peak = norm(resource.initial_momentum);
        for (auto h : trace.momentum) peak = std::max(peak, norm(h));
        out.objectives[2] += peak;
        out.traces.push_back(std::move(trace));
        std::vector<Placement> sequence;
        for (const auto& x : out.placements) if (!x.omitted && x.resource == static_cast<int>(r)) sequence.push_back(x);
        std::sort(sequence.begin(), sequence.end(), [](const Placement& a, const Placement& b) { return std::tie(a.start, a.action) < std::tie(b.start, b.action); });
        if (resource.sequence) for (const auto& group : p.contiguous_) {
            int first = -1, last = -1;
            for (std::size_t k = 0; k < sequence.size(); ++k) if (p.actions_[sequence[k].action].group == group) { if (first < 0) first = static_cast<int>(k); last = static_cast<int>(k); }
            for (int k = first + 1; first >= 0 && k < last; ++k) if (p.actions_[sequence[k].action].group != group)
                violation("contiguous", "Внутрь непрерывной группы вставлено другое действие.", {sequence[first].action, sequence[k].action, sequence[last].action}, sequence[first].start, sequence[last].end);
        }
        std::vector<double> state = resource.initial_state;
        int previous = -1; double previous_end = 0; Vector3 pointing = resource.initial_pointing;
        for (const auto& x : sequence) {
            const auto& a = p.actions_[x.action]; const auto& mode = a.modes[x.mode];
            double ready = previous_end;
            bool chain_valid = true;
            if (resource.sequence && (p.transition_provider_ || p.exact_transition_provider_) && (previous >= 0 || p.initial_transition_)) {
                DurationRequest request;
                request.transition = true; request.before = previous; request.after = x.action; request.mode = x.mode; request.resource = static_cast<int>(r);
                request.start_seconds = previous_end; request.slot_seconds = p.quantum_; request.state = state;
                request.from_pointing = pointing; request.to_pointing = a.pointing;
                request.model_version = exact && p.exact_transition_provider_ ? p.exact_version_ : p.transition_version_;
                const auto transition = query(request, exact);
                ready = previous_end + duration_slots(transition.status == DurationStatus::Feasible ? transition.seconds : 0, p.quantum_) * p.quantum_;
                if (transition.status != DurationStatus::Feasible || ready > x.start * p.quantum_ + 1e-12) {
                    std::vector<int> affected;
                    // Полный префикс, а не произвольная пара, определяет физическое состояние.
                    for (const auto& y : sequence) { affected.push_back(y.action); if (y.action == x.action) break; }
                    violation("transition", "Внешний переход не подтверждён либо не помещается в разрыв: " + transition.diagnostic,
                              affected, static_cast<int>(previous_end / p.quantum_), x.start, std::max(1.0, (ready - x.start * p.quantum_) / p.quantum_));
                    chain_valid = false;
                } else {
                    out.segments.push_back({"transition", static_cast<int>(r), x.action, previous_end, previous_end + transition.seconds, transition});
                    if (!transition.end_state.empty()) state = transition.end_state;
                    out.objectives[4] += transition.seconds + transition.cost + transition.risk;
                }
            }
            if (resource.sequence && ready < x.start * p.quantum_) {
                out.segments.push_back({"idle", static_cast<int>(r), -1, ready, x.start * p.quantum_, {}});
                out.objectives[1] += x.start * p.quantum_ - ready;
            }
            DurationRequest request;
            request.after = x.action; request.mode = x.mode; request.resource = static_cast<int>(r);
            request.start_seconds = x.start * p.quantum_; request.slot_seconds = p.quantum_; request.nominal_seconds = mode.duration;
            request.state = resource.sequence ? state : resource.initial_state; request.to_pointing = a.pointing; request.model_version = p.duration_version_;
            const auto actual = query(request, exact);
            if (actual.status != DurationStatus::Feasible || actual.seconds <= 0 || actual.seconds > (x.end - x.start) * p.quantum_ + 1e-12) {
                violation("duration", "Точная длительность действия не подтверждена или превышает резерв.", {x.action}, x.start, x.end);
                chain_valid = false;
            }
            out.segments.push_back({"action", static_cast<int>(r), x.action, x.start * p.quantum_, x.end * p.quantum_, actual});
            if (!actual.end_state.empty()) state = actual.end_state;
            previous = x.action; previous_end = x.end * p.quantum_; pointing = a.pointing;
            if (resource.sequence && !chain_valid) {
                // После неизвестного состояния последующие физические сертификаты недостоверны.
                break;
            }
        }
        if (resource.sequence && previous_end < p.horizon_ * p.quantum_) {
            out.objectives[1] += p.horizon_ * p.quantum_ - previous_end;
            out.segments.push_back({"idle", static_cast<int>(r), -1, previous_end, p.horizon_ * p.quantum_, {}});
        }
    }
    out.feasible = out.violations.empty();
    return out;
}

/** Создаёт GDS-сеть, сохраняя соответствие нейрона элементу области значений. */
GDSNetwork CompiledProblem::make_network(std::uint64_t seed) const {
    std::vector<int> sizes; for (const auto& domain : domains_) sizes.push_back(static_cast<int>(domain.size()));
    GDSNetwork net(static_cast<int>(sizes.size()), sizes, 1, 4, 5, 0.5, 1, 8, 10000,
                   StopRule::SatisfiedOrMaxIterations, "binary_threshold", "guarded_discrete_stochastic", "all_off", seed);
    for (const auto& pair : forbidden_) net.add_forbidden_pair(pair.var_a, pair.val_a, pair.var_b, pair.val_b);
    for (const auto& pair : preferences_) net.add_weighted_pair(pair.var_a, pair.val_a, pair.var_b, pair.val_b, std::tanh(pair.weight));
    for (std::size_t i = 0; i < domains_.size(); ++i) for (std::size_t v = 0; v < domains_[i].size(); ++v) {
        const auto& x = domains_[i][v];
        const double score = x.omitted ? -problem_.actions_[i].priority : x.log_suitability;
        net.set_unary_bias(static_cast<int>(i), static_cast<int>(v), std::tanh(score));
    }
    net.finalize_weights(); return net;
}
/** Возвращает области значений с координатами время—действие—ресурс. */
std::vector<std::vector<Placement>> CompiledProblem::domains() const { return domains_; }
/** Возвращает число вычислений внешних моделей и попаданий в кэш. */
std::pair<std::size_t, std::size_t> CompiledProblem::cache_statistics() const { return {calls_, hits_}; }
/** Очищает кэш после внешнего изменения модели или телеметрии. */
void CompiledProblem::clear_cache() const { cache_.clear(); calls_ = hits_ = 0; }
/** Возвращает исходную постановку для просмотра. */
Problem CompiledProblem::problem() const { return problem_; }

Problem make_jssp(int machines, int horizon, const std::vector<std::vector<Operation>>& jobs) {
    require(machines > 0 && !jobs.empty(), "Нужны станки и непустое множество работ.");
    Problem p(horizon);
    for (int r = 0; r < machines; ++r) { Resource resource; resource.name = "Станок " + std::to_string(r); resource.sequence = true; p.add_resource(resource); }
    for (std::size_t j = 0; j < jobs.size(); ++j) {
        require(!jobs[j].empty(), "Работа должна содержать операции."); int previous = -1;
        for (std::size_t k = 0; k < jobs[j].size(); ++k) {
            const auto op = jobs[j][k]; require(op.machine >= 0 && op.machine < machines && op.duration > 0, "Некорректная операция.");
            Action a; a.name = "Работа " + std::to_string(j) + ", операция " + std::to_string(k);
            Mode m; m.resource = op.machine; m.duration = op.duration; a.modes.push_back(m);
            const int index = p.add_action(a); if (previous >= 0) p.add_precedence({previous, index, 0, -1}); previous = index;
        }
    }
    return p;
}
/** Переводит независимые работы на взаимозаменяемых аппаратах в задачу ограничений. */
Problem make_parallel_jobs(int machines, int horizon, const std::vector<int>& durations, int capacity) {
    require(machines > 0 && capacity > 0 && !durations.empty(), "Нужны аппараты, положительная ёмкость и работы.");
    Problem p(horizon);
    for (int r = 0; r < machines; ++r) { Resource resource; resource.name = "Аппарат " + std::to_string(r); resource.capacity = capacity; resource.sequence = capacity == 1; p.add_resource(resource); }
    for (std::size_t i = 0; i < durations.size(); ++i) {
        require(durations[i] > 0, "Длительность работы должна быть положительной."); Action a; a.name = "Работа " + std::to_string(i);
        for (int r = 0; r < machines; ++r) { Mode mode; mode.resource = r; mode.duration = durations[i]; a.modes.push_back(mode); }
        p.add_action(a);
    }
    return p;
}
/** Строит CSP-сеть замкнутого коммивояжёра; бесконечность обозначает отсутствие дуги. */
GDSNetwork make_tsp_network(const std::vector<std::vector<double>>& costs, std::uint64_t seed) {
    validate_costs(costs); const int n = static_cast<int>(costs.size());
    GDSNetwork net(n, std::vector<int>(n, n)); net.set_seed(seed);
    double scale = 1; for (const auto& row : costs) for (double c : row) if (std::isfinite(c)) scale = std::max(scale, c);
    for (int i = 0; i < n; ++i) for (int j = i + 1; j < n; ++j) for (int city = 0; city < n; ++city) net.add_forbidden_pair(i, city, j, city);
    for (int pos = 0; pos < n; ++pos) for (int a = 0; a < n; ++a) for (int b = 0; b < n; ++b) if (a != b) {
        if (std::isfinite(costs[a][b])) net.add_weighted_pair(pos, a, (pos + 1) % n, b, -costs[a][b] / scale);
        else net.add_forbidden_pair(pos, a, (pos + 1) % n, b);
    }
    net.finalize_weights(); return net;
}
/** Проверяет перестановку городов и вычисляет стоимость замкнутого маршрута. */
double tour_cost(const std::vector<int>& tour, const std::vector<std::vector<double>>& costs) {
    validate_costs(costs); require(tour.size() == costs.size(), "Маршрут должен содержать все города."); std::set<int> seen;
    for (int c : tour) require(c >= 0 && c < static_cast<int>(costs.size()) && seen.insert(c).second, "Маршрут должен быть перестановкой городов.");
    double sum = 0; for (std::size_t i = 0; i < tour.size(); ++i) sum += costs[tour[i]][tour[(i + 1) % tour.size()]]; return sum;
}
/** Делит кандидатов на уменьшающие и увеличивающие норму суммарного момента по статье 2007 года. */
std::pair<std::vector<int>, std::vector<int>> find_reducers(const std::vector<Vector3>& momenta) {
    Vector3 total{0, 0, 0}; for (auto v : momenta) { require(finite(v), "Момент должен быть конечным."); for (int k = 0; k < 3; ++k) total[k] += v[k]; }
    std::pair<std::vector<int>, std::vector<int>> result;
    for (std::size_t i = 0; i < momenta.size(); ++i) { Vector3 next = total; for (int k = 0; k < 3; ++k) next[k] += momenta[i][k];
        (norm(next) < norm(total) ? result.first : result.second).push_back(static_cast<int>(i)); }
    return result;
}
} // Пространство имён gds.
