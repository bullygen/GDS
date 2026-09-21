/** @file network.cpp
 * @brief Дискретная стохастическая сеть с охранными нейронами для конечных ограничений.
 */
#include "gds/network.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_set>

namespace gds {
namespace {

/** Кодирует неупорядоченную пару индексов для устранения повторных запретов. */
std::uint64_t make_pair_key(int a, int b) {
    const std::uint64_t lo = static_cast<std::uint32_t>(std::min(a, b));
    const std::uint64_t hi = static_cast<std::uint32_t>(std::max(a, b));
    return (hi << 32ULL) | lo;
}

}  // Внутреннее пространство имён.

/** Создаёт двоичную сеть с заданными областями значений и параметрами охранных нейронов. */
GDSNetwork::GDSNetwork(
    int num_variables,
    const std::vector<int>& domain_sizes,
    double beta,
    double w_forbidden,
    double gamma_same,
    double guard_bias,
    double guard_delta,
    double guard_phi,
    std::size_t max_iterations,
    StopRule stop_rule,
    const std::string& functional_type,
    const std::string& update_rule,
    const std::string& init_rule,
    std::uint64_t seed
)
    : num_variables_(num_variables),
      num_neurons_(0),
      domain_sizes_(domain_sizes),
      functional_type_(functional_type),
      update_rule_(update_rule),
      init_rule_(init_rule),
      stop_rule_(stop_rule),
      beta_(beta),
      w_forbidden_(w_forbidden),
      gamma_same_(gamma_same),
      guard_bias_(guard_bias),
      guard_delta_(guard_delta),
      guard_phi_(guard_phi),
      max_iterations_(max_iterations),
      seed_(seed),
      rng_(seed) {
    if (num_variables_ <= 0) {
        throw std::runtime_error("Число переменных должно быть положительным.");
    }
    if (static_cast<int>(domain_sizes_.size()) != num_variables_) {
        throw std::runtime_error("Число областей значений не совпадает с числом переменных.");
    }
    for (int size : domain_sizes_) {
        if (size <= 0) {
            throw std::runtime_error("Каждая область значений должна быть непустой.");
        }
    }
    if (!std::isfinite(beta_) || !std::isfinite(w_forbidden_) || w_forbidden_ <= 0 ||
        !std::isfinite(gamma_same_) || gamma_same_ <= 0 || !std::isfinite(guard_bias_) ||
        !std::isfinite(guard_delta_) || guard_delta_ <= 0 || !std::isfinite(guard_phi_) || guard_phi_ <= 0 || max_iterations_ == 0)
        throw std::invalid_argument("Некорректные числовые параметры сети.");
    set_functional_type(functional_type_); set_update_rule(update_rule_); set_init_rule(init_rule_);
    set_stop_rule(stop_rule_);
    rebuild_offsets();
    unary_bias_.assign(num_neurons_, 0.0);
    state_.assign(num_neurons_, 0);
    guard_state_.assign(num_variables_, 0);
    inputs_.assign(num_neurons_, 0.0);
    row_counts_.assign(num_variables_, 0);
    rebuild_neighbors();
    recompute_guards();
    recompute_inputs();
}

/** Строит смещения строк, проверяя переполнение целого индекса. */
void GDSNetwork::rebuild_offsets() {
    offsets_.assign(num_variables_ + 1, 0);
    num_neurons_ = 0;
    for (int i = 0; i < num_variables_; ++i) {
        offsets_[i] = num_neurons_;
        if (domain_sizes_[i] > std::numeric_limits<int>::max() - num_neurons_)
            throw std::invalid_argument("Слишком много нейронов.");
        num_neurons_ += domain_sizes_[i];
    }
    offsets_[num_variables_] = num_neurons_;
}

/** Преобразует координаты переменная—значение в плоский индекс с проверкой границ. */
int GDSNetwork::flatten(int var, int val) const {
    validate_var_val(var, val);
    return offsets_[var] + val;
}

/** Восстанавливает координаты переменная—значение по плоскому индексу. */
std::pair<int, int> GDSNetwork::unflatten(int idx) const {
    if (idx < 0 || idx >= num_neurons_) {
        throw std::runtime_error("Индекс нейрона вне границ.");
    }
    const auto it = std::upper_bound(offsets_.begin(), offsets_.end(), idx);
    const int var = static_cast<int>(std::distance(offsets_.begin(), it)) - 1;
    return {var, idx - offsets_[var]};
}

/** Проверяет принадлежность переменной и значения заданным областям. */
void GDSNetwork::validate_var_val(int var, int val) const {
    if (var < 0 || var >= num_variables_) {
        throw std::runtime_error("Индекс переменной вне границ.");
    }
    if (val < 0 || val >= domain_sizes_[var]) {
        throw std::runtime_error("Индекс значения вне границ.");
    }
}

/** Удаляет все парные запреты и перестраивает входы. */
void GDSNetwork::clear_forbidden_pairs() {
    forbidden_pairs_.clear();
    finalize_weights();
}

/** Добавляет жёсткий парный запрет; после изменения списка требуется finalize_weights(). */
void GDSNetwork::add_forbidden_pair(int var_a, int val_a, int var_b, int val_b) {
    validate_var_val(var_a, val_a);
    validate_var_val(var_b, val_b);
    if (var_a == var_b && val_a == val_b) {
        throw std::invalid_argument("Самозапрет нейрона задавайте исключением значения из области.");
    }
    forbidden_pairs_.push_back({var_a, val_a, var_b, val_b});
}

/** Удаляет мягкие связи и перестраивает входы. */
void GDSNetwork::clear_weighted_pairs() {
    weighted_pairs_.clear();
    finalize_weights();
}

/** Добавляет симметричную мягкую связь; после изменения требуется finalize_weights(). */
void GDSNetwork::add_weighted_pair(int var_a, int val_a, int var_b, int val_b, double weight) {
    validate_var_val(var_a, val_a);
    validate_var_val(var_b, val_b);
    if (var_a == var_b && val_a == val_b) {
        throw std::invalid_argument("Для индивидуального веса используйте set_unary_bias.");
    }
    if (!std::isfinite(weight)) throw std::invalid_argument("Вес должен быть конечным.");
    weighted_pairs_.push_back({var_a, val_a, var_b, val_b, weight});
}

/** Проверяет наличие явно заданного запрета для пары плоских индексов. */
bool GDSNetwork::has_explicit_forbidden_idx(int idx_a, int idx_b) const {
    for (const auto& p : forbidden_pairs_) {
        const int a = flatten(p.var_a, p.val_a);
        const int b = flatten(p.var_b, p.val_b);
        if ((a == idx_a && b == idx_b) || (a == idx_b && b == idx_a)) {
            return true;
        }
    }
    return false;
}

/** Строит разреженные списки межстрочных связей и связи взаимного торможения в строках. */
void GDSNetwork::rebuild_neighbors() {
    neighbors_.assign(num_neurons_, {});

    for (int var = 0; var < num_variables_; ++var) {
        const int begin = offsets_[var];
        const int end = offsets_[var + 1];
        for (int a = begin; a < end; ++a) {
            for (int b = a + 1; b < end; ++b) {
                neighbors_[a].push_back({b, -gamma_same_});
                neighbors_[b].push_back({a, -gamma_same_});
            }
        }
    }

    std::unordered_set<std::uint64_t> seen;
    for (const auto& pair : forbidden_pairs_) {
        const int a = flatten(pair.var_a, pair.val_a);
        const int b = flatten(pair.var_b, pair.val_b);
        if (a == b) {
            continue;
        }
        const auto key = make_pair_key(a, b);
        if (seen.find(key) != seen.end()) {
            continue;
        }
        seen.insert(key);
        neighbors_[a].push_back({b, -w_forbidden_});
        neighbors_[b].push_back({a, -w_forbidden_});
    }

    for (const auto& pair : weighted_pairs_) {
        const int a = flatten(pair.var_a, pair.val_a);
        const int b = flatten(pair.var_b, pair.val_b);
        if (a == b) {
            continue;
        }
        neighbors_[a].push_back({b, pair.weight});
        neighbors_[b].push_back({a, pair.weight});
    }
}

/** Пересчитывает число активных нейронов и охранное состояние каждой строки. */
void GDSNetwork::recompute_guards() {
    std::fill(row_counts_.begin(), row_counts_.end(), 0);
    for (int var = 0; var < num_variables_; ++var) {
        for (int idx = offsets_[var]; idx < offsets_[var + 1]; ++idx) {
            row_counts_[var] += state_[idx];
        }
        const double xg = guard_bias_ - guard_delta_ * static_cast<double>(row_counts_[var]);
        guard_state_[var] = xg >= 0.0 ? 1 : 0;
    }
}

/** Пересчитывает входы из сдвигов, охранных сигналов, связей и запретов полного назначения. */
void GDSNetwork::recompute_inputs() {
    inputs_.assign(num_neurons_, 0.0);
    for (int var = 0; var < num_variables_; ++var) {
        const double guard_term = guard_phi_ * static_cast<double>(guard_state_[var]);
        for (int idx = offsets_[var]; idx < offsets_[var + 1]; ++idx) {
            inputs_[idx] = beta_ + unary_bias_[idx] + guard_term;
        }
    }
    for (int idx = 0; idx < num_neurons_; ++idx) {
        if (state_[idx] == 0) {
            continue;
        }
        for (const auto& [other, weight] : neighbors_[idx]) {
            inputs_[other] += weight;
        }
    }
    for (const auto& ng : nogoods_) {
        int active = 0;
        for (int v = 0; v < num_variables_; ++v) active += state_[flatten(v, ng[v])];
        for (int v = 0; v < num_variables_; ++v) {
            const int idx = flatten(v, ng[v]);
            if (active - state_[idx] == num_variables_ - 1) inputs_[idx] -= w_forbidden_;
        }
    }
}

/** Задаёт конечную индивидуальную добавку и пересчитывает входы. */
void GDSNetwork::set_unary_bias(int variable, int value, double bias) {
    if (!std::isfinite(bias)) throw std::invalid_argument("Сдвиг должен быть конечным.");
    unary_bias_[flatten(variable, value)] = bias;
    recompute_inputs();
}
void GDSNetwork::set_unary_biases(const std::vector<double>& biases) {
    if (biases.size() != unary_bias_.size()) throw std::invalid_argument("Неверное число сдвигов.");
    for (double bias : biases)
        if (!std::isfinite(bias)) throw std::invalid_argument("Сдвиг должен быть конечным.");
    unary_bias_ = biases;
    recompute_inputs();
}
/** Возвращает сдвиг указанного нейрона. */
double GDSNetwork::get_unary_bias(int variable, int value) const { return unary_bias_[flatten(variable, value)]; }
/** Добавляет запрет целого назначения, не запрещая его допустимые подмножества. */
void GDSNetwork::add_nogood(const std::vector<int>& assignment) {
    if (assignment.size() != domain_sizes_.size()) throw std::invalid_argument("Неверная длина запрета.");
    for (int i = 0; i < num_variables_; ++i) validate_var_val(i, assignment[i]);
    if (std::find(nogoods_.begin(), nogoods_.end(), assignment) == nogoods_.end()) nogoods_.push_back(assignment);
    recompute_inputs();
}

/** Перестраивает списки соседей, охранные состояния и входы нейронов. */
void GDSNetwork::finalize_weights() {
    rebuild_neighbors();
    recompute_guards();
    recompute_inputs();
}

/** Прибавляет изменение охранного сигнала ко всем нейронам строки. */
void GDSNetwork::apply_guard_change(int var, int old_guard, int new_guard) {
    if (old_guard == new_guard) {
        return;
    }
    const double delta = static_cast<double>(new_guard - old_guard) * guard_phi_;
    for (int idx = offsets_[var]; idx < offsets_[var + 1]; ++idx) {
        inputs_[idx] += delta;
    }
}

/** Возвращает сохранённое число активных нейронов строки. */
int GDSNetwork::row_active_count(int var) const {
    return row_counts_[var];
}

/** Меняет двоичное состояние, обновляет входы соседей и охранный сигнал строки. */
void GDSNetwork::flip_neuron(int idx) {
    const int old_value = state_[idx];
    const int new_value = 1 - old_value;
    const int delta = new_value - old_value;
    state_[idx] = new_value;
    ++transitions_;

    const auto [var, _val] = unflatten(idx);
    row_counts_[var] += delta;

    for (const auto& [other, weight] : neighbors_[idx]) {
        inputs_[other] += weight * static_cast<double>(delta);
    }

    const int old_guard = guard_state_[var];
    const double xg = guard_bias_ - guard_delta_ * static_cast<double>(row_counts_[var]);
    const int new_guard = xg >= 0.0 ? 1 : 0;
    guard_state_[var] = new_guard;
    apply_guard_change(var, old_guard, new_guard);
    if (!nogoods_.empty()) recompute_inputs();
}

/** Равновероятно выбирает индекс переменной для следующего обновления. */
int GDSNetwork::choose_variable() {
    std::uniform_int_distribution<int> dist(0, num_variables_ - 1);
    return dist(rng_);
}

/** Выбирает наиболее несогласованный нейрон; равные оценки разрешает случайно. */
int GDSNetwork::pick_best_inconsistent_neuron(int var, double& best_score) {
    best_score = -std::numeric_limits<double>::infinity();
    std::vector<int> candidates;
    for (int idx = offsets_[var]; idx < offsets_[var + 1]; ++idx) {
        const int y = state_[idx];
        const double x = inputs_[idx];
        double score = -std::numeric_limits<double>::infinity();
        if (y == 0 && x >= 0.0) {
            score = x;
        } else if (y == 1 && x < 0.0) {
            score = -x;
        }
        if (!std::isfinite(score)) {
            continue;
        }
        if (score > best_score + 1e-12) {
            best_score = score;
            candidates.clear();
            candidates.push_back(idx);
        } else if (std::fabs(score - best_score) <= 1e-12) {
            candidates.push_back(idx);
        }
    }
    if (candidates.empty()) {
        best_score = 0.0;
        return -1;
    }
    std::uniform_int_distribution<std::size_t> dist(0, candidates.size() - 1);
    return candidates[dist(rng_)];
}

/** Выбирает случайную переменную и меняет наиболее несогласованный нейрон её строки. */
StepInfo GDSNetwork::step() {
    StepInfo info;
    ++iterations_;
    info.iteration = iterations_;

    const int var = choose_variable();
    info.variable = var;

    double best_score = 0.0;
    const int idx = pick_best_inconsistent_neuron(var, best_score);
    info.neuron_index = idx;
    info.score = best_score;
    info.transitions = transitions_;

    if (idx < 0) {
        return info;
    }

    info.old_value = state_[idx];
    flip_neuron(idx);
    info.new_value = state_[idx];
    info.flipped = true;
    info.transitions = transitions_;
    return info;
}

/** Выполняет шаги до выполнения ограничений либо до заданного предела итераций. */
RunResult GDSNetwork::run(std::size_t override_max_iterations) {
    const std::size_t limit = override_max_iterations == 0 ? max_iterations_ : override_max_iterations;
    if (stop_rule_ == StopRule::SatisfiedOrMaxIterations && is_satisfied()) {
        return {true, false, iterations_, transitions_, state_, guard_state_, decode_assignment()};
    }
    const std::size_t target = iterations_ + limit;
    while (iterations_ < target) {
        step();
        if (stop_rule_ == StopRule::SatisfiedOrMaxIterations && is_satisfied()) {
            return {true, false, iterations_, transitions_, state_, guard_state_, decode_assignment()};
        }
    }
    const bool satisfied = is_satisfied();
    return {satisfied, true, iterations_, transitions_, state_, guard_state_, decode_assignment()};
}

/** Обнуляет состояния и счётчики; состояние генератора случайных чисел сохраняется. */
void GDSNetwork::reset_state_all_off() {
    std::fill(state_.begin(), state_.end(), 0);
    iterations_ = 0;
    transitions_ = 0;
    recompute_guards();
    recompute_inputs();
}

/** Устанавливает проверенное двоичное состояние и пересчитывает охранные нейроны. */
void GDSNetwork::set_state(const std::vector<int>& flat_state) {
    if (static_cast<int>(flat_state.size()) != num_neurons_) {
        throw std::runtime_error("Неверная длина состояния.");
    }
    for (int value : flat_state) {
        if (value != 0 && value != 1) {
            throw std::runtime_error("Состояния нейронов должны быть нулём или единицей.");
        }
    }
    state_ = flat_state;
    recompute_guards();
    recompute_inputs();
}

/** Активирует выбранное значение каждой переменной; минус один оставляет строку пустой. */
void GDSNetwork::set_assignment(const std::vector<int>& assignment) {
    if (static_cast<int>(assignment.size()) != num_variables_) {
        throw std::runtime_error("Длина назначения не равна числу переменных.");
    }
    std::vector<int> flat(num_neurons_, 0);
    for (int var = 0; var < num_variables_; ++var) {
        const int value = assignment[var];
        if (value < -1 || value >= domain_sizes_[var]) {
            throw std::runtime_error("Значение назначения вне границ.");
        }
        if (value >= 0) {
            flat[flatten(var, value)] = 1;
        }
    }
    set_state(flat);
}

/** Возвращает плоский массив состояний основных нейронов. */
std::vector<int> GDSNetwork::get_state() const {
    return state_;
}

/** Возвращает состояния с разбиением по областям значений переменных. */
std::vector<std::vector<int>> GDSNetwork::get_state_matrix() const {
    std::vector<std::vector<int>> out(num_variables_);
    for (int var = 0; var < num_variables_; ++var) {
        out[var].reserve(domain_sizes_[var]);
        for (int idx = offsets_[var]; idx < offsets_[var + 1]; ++idx) {
            out[var].push_back(state_[idx]);
        }
    }
    return out;
}

/** Возвращает индексы единственных активных значений; неоднозначную строку обозначает минус один. */
std::vector<int> GDSNetwork::decode_assignment() const {
    std::vector<int> assignment(num_variables_, -1);
    for (int var = 0; var < num_variables_; ++var) {
        int found = -1;
        for (int val = 0; val < domain_sizes_[var]; ++val) {
            const int idx = offsets_[var] + val;
            if (state_[idx] == 1) {
                if (found != -1) {
                    assignment[var] = -1;
                    found = -2;
                    break;
                }
                found = val;
            }
        }
        if (found >= 0) {
            assignment[var] = found;
        }
    }
    return assignment;
}

/** Возвращает состояния охранных нейронов строк. */
std::vector<int> GDSNetwork::get_guard_state() const {
    return guard_state_;
}

/** Возвращает текущие входы основных нейронов с учётом всех связей. */
std::vector<double> GDSNetwork::get_inputs() const {
    return inputs_;
}

/** Возвращает размеры конечных областей значений. */
std::vector<int> GDSNetwork::get_domain_sizes() const {
    return domain_sizes_;
}

/** Возвращает явно заданные парные запреты. */
std::vector<ForbiddenPair> GDSNetwork::get_forbidden_pairs() const {
    return forbidden_pairs_;
}

/** Возвращает явно заданные мягкие связи. */
std::vector<WeightedPair> GDSNetwork::get_weighted_pairs() const {
    return weighted_pairs_;
}

/** Проверяет единственность выбора, охранные нейроны, парные и полные запреты. */
bool GDSNetwork::is_satisfied() const {
    const auto assignment = decode_assignment();
    if (std::any_of(assignment.begin(), assignment.end(), [](int x) { return x < 0; })) {
        return false;
    }
    if (std::any_of(guard_state_.begin(), guard_state_.end(), [](int x) { return x != 0; })) {
        return false;
    }
    return count_conflicts() == 0;
}

/** Подсчитывает многократные активации и нарушенные жёсткие запреты. */
std::size_t GDSNetwork::count_conflicts() const {
    std::size_t conflicts = 0;
    for (int var = 0; var < num_variables_; ++var) {
        const int active = row_counts_[var];
        if (active > 1) {
            conflicts += static_cast<std::size_t>(active - 1);
        }
    }
    std::unordered_set<std::uint64_t> seen;
    for (const auto& pair : forbidden_pairs_) {
        const int a = flatten(pair.var_a, pair.val_a);
        const int b = flatten(pair.var_b, pair.val_b);
        const auto key = make_pair_key(a, b);
        if (seen.find(key) != seen.end()) {
            continue;
        }
        seen.insert(key);
        if (state_[a] == 1 && state_[b] == 1) {
            ++conflicts;
        }
    }
    for (const auto& ng : nogoods_) {
        bool active = true;
        for (int v = 0; v < num_variables_; ++v) active = active && state_[flatten(v, ng[v])] == 1;
        if (active) ++conflicts;
    }
    return conflicts;
}

/** Подсчитывает активные основные нейроны. */
std::size_t GDSNetwork::count_active_neurons() const {
    return static_cast<std::size_t>(std::count(state_.begin(), state_.end(), 1));
}

/** Возвращает суммарный симметричный вес связи двух основных нейронов. */
double GDSNetwork::get_weight(int var_a, int val_a, int var_b, int val_b) const {
    const int a = flatten(var_a, val_a);
    const int b = flatten(var_b, val_b);
    if (a == b) {
        return 0.0;
    }
    double weight = 0.0;
    if (var_a == var_b) {
        weight += -gamma_same_;
    }
    if (has_explicit_forbidden_idx(a, b)) {
        weight += -w_forbidden_;
    }
    for (const auto& pair : weighted_pairs_) {
        const int wa = flatten(pair.var_a, pair.val_a);
        const int wb = flatten(pair.var_b, pair.val_b);
        if ((wa == a && wb == b) || (wa == b && wb == a)) {
            weight += pair.weight;
        }
    }
    return weight;
}

/** Сохраняет параметры, ограничения, состояние и генератор в текстовый формат GDSW2. */
void GDSNetwork::save_weights(const std::string& path) const {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Не удалось открыть файл для записи: " + path);
    out << std::setprecision(17) << "GDSW2\n" << num_variables_ << '\n';
    for (int n : domain_sizes_) out << n << ' ';
    out << '\n' << beta_ << ' ' << w_forbidden_ << ' ' << gamma_same_ << ' ' << guard_bias_ << ' '
        << guard_delta_ << ' ' << guard_phi_ << ' ' << max_iterations_ << ' ' << static_cast<int>(stop_rule_) << ' '
        << seed_ << ' ' << iterations_ << ' ' << transitions_ << '\n';
    for (int v : state_) out << v << ' ';
    out << '\n';
    for (double v : unary_bias_) out << v << ' ';
    out << '\n' << forbidden_pairs_.size() << '\n';
    for (auto p : forbidden_pairs_) out << p.var_a << ' ' << p.val_a << ' ' << p.var_b << ' ' << p.val_b << '\n';
    out << weighted_pairs_.size() << '\n';
    for (auto p : weighted_pairs_) out << p.var_a << ' ' << p.val_a << ' ' << p.var_b << ' ' << p.val_b << ' ' << p.weight << '\n';
    out << nogoods_.size() << '\n';
    for (const auto& ng : nogoods_) { for (int v : ng) out << v << ' '; out << '\n'; }
    out << rng_ << '\n';
    if (!out) throw std::runtime_error("Ошибка записи состояния сети.");
}

/** Загружает проверенное состояние GDSW2 для точного продолжения поиска. */
GDSNetwork GDSNetwork::load_weights(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Не удалось открыть файл: " + path);
    in.exceptions(std::ios::failbit | std::ios::badbit);
    std::string magic; in >> magic;
    if (magic != "GDSW2") throw std::invalid_argument("Неподдерживаемый формат состояния сети.");
    int n; in >> n;
    if (n <= 0 || n > 1000000) throw std::invalid_argument("Некорректное число переменных в файле.");
    std::vector<int> sizes(n); for (int& v : sizes) in >> v;
    double beta, forbidden, same, bias, delta, phi;
    std::size_t limit, iterations, transitions; int stop; std::uint64_t seed;
    in >> beta >> forbidden >> same >> bias >> delta >> phi >> limit >> stop >> seed >> iterations >> transitions;
    GDSNetwork net(n, sizes, beta, forbidden, same, bias, delta, phi, limit,
                   static_cast<StopRule>(stop), "binary_threshold", "guarded_discrete_stochastic", "all_off", seed);
    std::vector<int> state(net.num_neurons_); for (int& v : state) in >> v;
    for (double& v : net.unary_bias_) { in >> v; if (!std::isfinite(v)) throw std::invalid_argument("Неконечный сдвиг в файле."); }
    std::size_t count; in >> count;
    for (std::size_t i = 0; i < count; ++i) { ForbiddenPair p; in >> p.var_a >> p.val_a >> p.var_b >> p.val_b; net.add_forbidden_pair(p.var_a,p.val_a,p.var_b,p.val_b); }
    in >> count;
    for (std::size_t i = 0; i < count; ++i) { WeightedPair p; in >> p.var_a >> p.val_a >> p.var_b >> p.val_b >> p.weight; net.add_weighted_pair(p.var_a,p.val_a,p.var_b,p.val_b,p.weight); }
    in >> count;
    for (std::size_t i = 0; i < count; ++i) { std::vector<int> ng(n); for (int& v : ng) in >> v; net.add_nogood(ng); }
    in >> net.rng_;
    net.finalize_weights(); net.set_state(state); net.iterations_ = iterations; net.transitions_ = transitions;
    return net;
}

/** Возвращает число переменных. */
int GDSNetwork::get_num_variables() const {
    return num_variables_;
}

/** Возвращает число основных нейронов. */
int GDSNetwork::get_num_neurons() const {
    return num_neurons_;
}

/** Возвращает число выполненных итераций. */
std::size_t GDSNetwork::get_iterations() const {
    return iterations_;
}

/** Возвращает число изменений состояния. */
std::size_t GDSNetwork::get_transitions() const {
    return transitions_;
}

/** Возвращает зерно генератора случайных чисел. */
std::uint64_t GDSNetwork::get_seed() const {
    return seed_;
}

/** Возвращает вид функции активации. */
std::string GDSNetwork::get_functional_type() const {
    return functional_type_;
}

/** Возвращает правило стохастического обновления. */
std::string GDSNetwork::get_update_rule() const {
    return update_rule_;
}

/** Возвращает правило начального состояния. */
std::string GDSNetwork::get_init_rule() const {
    return init_rule_;
}

/** Возвращает правило остановки. */
StopRule GDSNetwork::get_stop_rule() const {
    return stop_rule_;
}

/** Возвращает общий сдвиг входов. */
double GDSNetwork::get_beta() const {
    return beta_;
}

/** Возвращает модуль штрафа за запрет. */
double GDSNetwork::get_w_forbidden() const {
    return w_forbidden_;
}

/** Возвращает модуль взаимного торможения в строке. */
double GDSNetwork::get_gamma_same() const {
    return gamma_same_;
}

/** Возвращает сдвиг охранного нейрона. */
double GDSNetwork::get_guard_bias() const {
    return guard_bias_;
}

/** Возвращает силу входа строки в охранный нейрон. */
double GDSNetwork::get_guard_delta() const {
    return guard_delta_;
}

/** Возвращает силу обратного охранного сигнала. */
double GDSNetwork::get_guard_phi() const {
    return guard_phi_;
}

/** Возвращает предел числа итераций. */
std::size_t GDSNetwork::get_max_iterations() const {
    return max_iterations_;
}

/** Устанавливает общий сдвиг входов с проверкой допустимости параметра. */
void GDSNetwork::set_beta(double value) {
    if (!std::isfinite(value)) throw std::invalid_argument("Некорректный параметр сети.");
    beta_ = value;
    recompute_inputs();
}

/** Устанавливает модуль штрафа за запрет с проверкой допустимости параметра. */
void GDSNetwork::set_w_forbidden(double value) {
    if (!std::isfinite(value) || value <= 0) throw std::invalid_argument("Некорректный параметр сети.");
    w_forbidden_ = value;
    finalize_weights();
}

/** Устанавливает модуль взаимного торможения в строке с проверкой допустимости параметра. */
void GDSNetwork::set_gamma_same(double value) {
    if (!std::isfinite(value) || value <= 0) throw std::invalid_argument("Некорректный параметр сети.");
    gamma_same_ = value;
    finalize_weights();
}

/** Устанавливает сдвиг охранного нейрона с проверкой допустимости параметра. */
void GDSNetwork::set_guard_bias(double value) {
    if (!std::isfinite(value)) throw std::invalid_argument("Некорректный параметр сети.");
    guard_bias_ = value;
    recompute_guards();
    recompute_inputs();
}

/** Устанавливает силу входа строки в охранный нейрон с проверкой допустимости параметра. */
void GDSNetwork::set_guard_delta(double value) {
    if (!std::isfinite(value) || value <= 0) throw std::invalid_argument("Некорректный параметр сети.");
    guard_delta_ = value;
    recompute_guards();
    recompute_inputs();
}

/** Устанавливает силу обратного охранного сигнала с проверкой допустимости параметра. */
void GDSNetwork::set_guard_phi(double value) {
    if (!std::isfinite(value) || value <= 0) throw std::invalid_argument("Некорректный параметр сети.");
    guard_phi_ = value;
    recompute_inputs();
}

/** Устанавливает предел числа итераций с проверкой допустимости параметра. */
void GDSNetwork::set_max_iterations(std::size_t value) {
    if (value == 0) throw std::invalid_argument("Число итераций должно быть положительным.");
    max_iterations_ = value;
}

/** Устанавливает правило остановки с проверкой допустимости параметра. */
void GDSNetwork::set_stop_rule(StopRule rule) {
    if (rule != StopRule::MaxIterations && rule != StopRule::SatisfiedOrMaxIterations) throw std::invalid_argument("Неизвестное правило остановки.");
    stop_rule_ = rule;
}

/** Устанавливает вид функции активации с проверкой допустимости параметра. */
void GDSNetwork::set_functional_type(const std::string& value) {
    if (value != "binary_threshold") throw std::invalid_argument("Неподдерживаемый режим: " + value);
    functional_type_ = value;
}

/** Устанавливает правило стохастического обновления с проверкой допустимости параметра. */
void GDSNetwork::set_update_rule(const std::string& value) {
    if (value != "guarded_discrete_stochastic") throw std::invalid_argument("Неподдерживаемый режим: " + value);
    update_rule_ = value;
}

/** Устанавливает правило начального состояния с проверкой допустимости параметра. */
void GDSNetwork::set_init_rule(const std::string& value) {
    if (value != "all_off") throw std::invalid_argument("Неподдерживаемый режим: " + value);
    init_rule_ = value;
}

/** Устанавливает зерно генератора случайных чисел с проверкой допустимости параметра. */
void GDSNetwork::set_seed(std::uint64_t seed) {
    seed_ = seed;
    rng_.seed(seed_);
}

/** Строит ограничения размещения n ферзей: столбцы и обе диагонали. */
GDSNetwork GDSNetwork::make_n_queens(
    int n,
    double beta,
    double w_forbidden,
    double gamma_same,
    double guard_bias,
    double guard_delta,
    double guard_phi,
    std::size_t max_iterations,
    StopRule stop_rule,
    std::uint64_t seed
) {
    if (n <= 0) throw std::invalid_argument("Размер доски должен быть положительным.");
    std::vector<int> domain_sizes(static_cast<std::size_t>(n), n);
    GDSNetwork net(
        n,
        domain_sizes,
        beta,
        w_forbidden,
        gamma_same,
        guard_bias,
        guard_delta,
        guard_phi,
        max_iterations,
        stop_rule,
        "binary_threshold",
        "guarded_discrete_stochastic",
        "all_off",
        seed
    );

    for (int row_a = 0; row_a < n; ++row_a) {
        for (int row_b = row_a + 1; row_b < n; ++row_b) {
            const int d = row_b - row_a;
            for (int col = 0; col < n; ++col) {
                net.add_forbidden_pair(row_a, col, row_b, col);
                if (col + d < n) {
                    net.add_forbidden_pair(row_a, col, row_b, col + d);
                }
                if (col - d >= 0) {
                    net.add_forbidden_pair(row_a, col, row_b, col - d);
                }
            }
        }
    }
    net.finalize_weights();
    net.reset_state_all_off();
    return net;
}

/** Строит сеть правильной раскраски графа тремя цветами. */
GDSNetwork GDSNetwork::make_graph_3_coloring(
    int num_nodes,
    const std::vector<std::pair<int, int>>& edges,
    double beta,
    double w_forbidden,
    double gamma_same,
    double guard_bias,
    double guard_delta,
    double guard_phi,
    std::size_t max_iterations,
    StopRule stop_rule,
    std::uint64_t seed
) {
    if (num_nodes <= 0) throw std::invalid_argument("Число вершин должно быть положительным.");
    std::vector<int> domain_sizes(static_cast<std::size_t>(num_nodes), 3);
    GDSNetwork net(
        num_nodes,
        domain_sizes,
        beta,
        w_forbidden,
        gamma_same,
        guard_bias,
        guard_delta,
        guard_phi,
        max_iterations,
        stop_rule,
        "binary_threshold",
        "guarded_discrete_stochastic",
        "all_off",
        seed
    );

    for (const auto& [u, v] : edges) {
        if (u < 0 || u >= num_nodes || v < 0 || v >= num_nodes) {
            throw std::runtime_error("Ребро содержит неизвестную вершину.");
        }
        if (u == v) throw std::invalid_argument("Граф с петлёй не допускает правильной раскраски.");
        for (int color = 0; color < 3; ++color) {
            net.add_forbidden_pair(u, color, v, color);
        }
    }
    net.finalize_weights();
    net.reset_state_all_off();
    return net;
}

}  // Пространство имён gds.
