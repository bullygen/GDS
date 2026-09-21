/** @file network.hpp
 * @brief Дискретная стохастическая сеть с охранными нейронами для конечных ограничений.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace gds {

/** Критерий остановки динамики: предел либо допустимость и предел. */
enum class StopRule {
    MaxIterations = 0,
    SatisfiedOrMaxIterations = 1,
};

/** Жёсткий запрет совместной активации двух различных нейронов. */
struct ForbiddenPair {
    int var_a;
    int val_a;
    int var_b;
    int val_b;
};

/** Дополнительная симметричная связь двух различных нейронов. */
struct WeightedPair {
    int var_a;
    int val_a;
    int var_b;
    int val_b;
    double weight;
};

/** Диагностика одной попытки стохастического обновления. */
struct StepInfo {
    bool flipped = false;
    int variable = -1;
    int neuron_index = -1;
    int old_value = 0;
    int new_value = 0;
    double score = 0.0;
    std::size_t iteration = 0;
    std::size_t transitions = 0;
};

/** Итог динамики с состояниями и декодированным назначением. */
struct RunResult {
    bool satisfied = false;
    bool stopped_by_max_iterations = false;
    std::size_t iterations = 0;
    std::size_t transitions = 0;
    std::vector<int> state;
    std::vector<int> guards;
    std::vector<int> assignment;
};

/** Дискретная стохастическая сеть с охранными нейронами для конечной CSP. */
class GDSNetwork {
public:
    /** Создаёт двоичную сеть с заданными областями значений и параметрами охранных нейронов. */
    GDSNetwork(
        int num_variables,
        const std::vector<int>& domain_sizes,
        double beta = 1.0,
        double w_forbidden = 4.0,
        double gamma_same = 5.0,
        double guard_bias = 0.5,
        double guard_delta = 1.0,
        double guard_phi = 8.0,
        std::size_t max_iterations = 10000,
        StopRule stop_rule = StopRule::SatisfiedOrMaxIterations,
        const std::string& functional_type = "binary_threshold",
        const std::string& update_rule = "guarded_discrete_stochastic",
        const std::string& init_rule = "all_off",
        std::uint64_t seed = 42
    );

    /** Строит ограничения размещения n ферзей: столбцы и обе диагонали. */
    static GDSNetwork make_n_queens(
        int n,
        double beta = 1.0,
        double w_forbidden = 4.0,
        double gamma_same = 5.0,
        double guard_bias = 0.5,
        double guard_delta = 1.0,
        double guard_phi = 8.0,
        std::size_t max_iterations = 10000,
        StopRule stop_rule = StopRule::SatisfiedOrMaxIterations,
        std::uint64_t seed = 42
    );

    /** Строит сеть правильной раскраски графа тремя цветами. */
    static GDSNetwork make_graph_3_coloring(
        int num_nodes,
        const std::vector<std::pair<int, int>>& edges,
        double beta = 1.0,
        double w_forbidden = 4.0,
        double gamma_same = 5.0,
        double guard_bias = 0.5,
        double guard_delta = 1.0,
        double guard_phi = 8.0,
        std::size_t max_iterations = 10000,
        StopRule stop_rule = StopRule::SatisfiedOrMaxIterations,
        std::uint64_t seed = 42
    );

    /** Добавляет жёсткий парный запрет; после изменения списка требуется finalize_weights(). */
    void add_forbidden_pair(int var_a, int val_a, int var_b, int val_b);
    /** Удаляет все парные запреты и перестраивает входы. */
    void clear_forbidden_pairs();
    /** Добавляет симметричную мягкую связь; после изменения требуется finalize_weights(). */
    void add_weighted_pair(int var_a, int val_a, int var_b, int val_b, double weight);
    /** Удаляет мягкие связи и перестраивает входы. */
    void clear_weighted_pairs();
    /** Перестраивает списки соседей, охранные состояния и входы нейронов. */
    void finalize_weights();

    /** Задаёт индивидуальную добавку к входу нейрона. */
    void set_unary_bias(int variable, int value, double bias);
    /** Возвращает индивидуальную добавку к входу нейрона. */
    double get_unary_bias(int variable, int value) const;
    /** Запрещает полное назначение после независимой проверки сложного ограничения. */
    void add_nogood(const std::vector<int>& assignment);

    /** Выбирает случайную переменную и меняет наиболее несогласованный нейрон её строки. */
    StepInfo step();
    /** Выполняет шаги до выполнения ограничений либо до заданного предела итераций. */
    RunResult run(std::size_t override_max_iterations = 0);

    /** Обнуляет состояния и счётчики; состояние генератора случайных чисел сохраняется. */
    void reset_state_all_off();
    /** Устанавливает проверенное двоичное состояние и пересчитывает охранные нейроны. */
    void set_state(const std::vector<int>& flat_state);
    /** Активирует выбранное значение каждой переменной; минус один оставляет строку пустой. */
    void set_assignment(const std::vector<int>& assignment);
    /** Возвращает плоский массив состояний основных нейронов. */
    std::vector<int> get_state() const;
    /** Возвращает состояния с разбиением по областям значений переменных. */
    std::vector<std::vector<int>> get_state_matrix() const;
    /** Возвращает индексы единственных активных значений; неоднозначную строку обозначает минус один. */
    std::vector<int> decode_assignment() const;
    /** Возвращает состояния охранных нейронов строк. */
    std::vector<int> get_guard_state() const;
    /** Возвращает текущие входы основных нейронов с учётом всех связей. */
    std::vector<double> get_inputs() const;
    /** Возвращает размеры конечных областей значений. */
    std::vector<int> get_domain_sizes() const;
    /** Возвращает явно заданные парные запреты. */
    std::vector<ForbiddenPair> get_forbidden_pairs() const;
    /** Возвращает явно заданные мягкие связи. */
    std::vector<WeightedPair> get_weighted_pairs() const;

    /** Проверяет единственность выбора, охранные нейроны, парные и полные запреты. */
    bool is_satisfied() const;
    /** Подсчитывает многократные активации и нарушенные жёсткие запреты. */
    std::size_t count_conflicts() const;
    /** Подсчитывает активные основные нейроны. */
    std::size_t count_active_neurons() const;

    /** Возвращает суммарный симметричный вес связи двух основных нейронов. */
    double get_weight(int var_a, int val_a, int var_b, int val_b) const;

    /** Сохраняет параметры, ограничения, состояние и генератор в текстовый формат GDSW2. */
    void save_weights(const std::string& path) const;
    /** Загружает проверенное состояние GDSW2 для точного продолжения поиска. */
    static GDSNetwork load_weights(const std::string& path);

    /** Возвращает число переменных. */
    int get_num_variables() const;
    /** Возвращает число основных нейронов. */
    int get_num_neurons() const;
    /** Возвращает число выполненных итераций. */
    std::size_t get_iterations() const;
    /** Возвращает число изменений состояния. */
    std::size_t get_transitions() const;
    /** Возвращает зерно генератора случайных чисел. */
    std::uint64_t get_seed() const;

    /** Возвращает вид функции активации. */
    std::string get_functional_type() const;
    /** Возвращает правило стохастического обновления. */
    std::string get_update_rule() const;
    /** Возвращает правило начального состояния. */
    std::string get_init_rule() const;
    /** Возвращает правило остановки. */
    StopRule get_stop_rule() const;

    /** Возвращает общий сдвиг входов. */
    double get_beta() const;
    /** Возвращает модуль штрафа за запрет. */
    double get_w_forbidden() const;
    /** Возвращает модуль взаимного торможения в строке. */
    double get_gamma_same() const;
    /** Возвращает сдвиг охранного нейрона. */
    double get_guard_bias() const;
    /** Возвращает силу входа строки в охранный нейрон. */
    double get_guard_delta() const;
    /** Возвращает силу обратного охранного сигнала. */
    double get_guard_phi() const;
    /** Возвращает предел числа итераций. */
    std::size_t get_max_iterations() const;

    /** Устанавливает общий сдвиг входов с проверкой допустимости параметра. */
    void set_beta(double value);
    /** Устанавливает модуль штрафа за запрет с проверкой допустимости параметра. */
    void set_w_forbidden(double value);
    /** Устанавливает модуль взаимного торможения в строке с проверкой допустимости параметра. */
    void set_gamma_same(double value);
    /** Устанавливает сдвиг охранного нейрона с проверкой допустимости параметра. */
    void set_guard_bias(double value);
    /** Устанавливает силу входа строки в охранный нейрон с проверкой допустимости параметра. */
    void set_guard_delta(double value);
    /** Устанавливает силу обратного охранного сигнала с проверкой допустимости параметра. */
    void set_guard_phi(double value);
    /** Устанавливает предел числа итераций с проверкой допустимости параметра. */
    void set_max_iterations(std::size_t value);
    /** Устанавливает правило остановки с проверкой допустимости параметра. */
    void set_stop_rule(StopRule rule);
    /** Устанавливает вид функции активации с проверкой допустимости параметра. */
    void set_functional_type(const std::string& value);
    /** Устанавливает правило стохастического обновления с проверкой допустимости параметра. */
    void set_update_rule(const std::string& value);
    /** Устанавливает правило начального состояния с проверкой допустимости параметра. */
    void set_init_rule(const std::string& value);
    /** Устанавливает зерно генератора случайных чисел с проверкой допустимости параметра. */
    void set_seed(std::uint64_t seed);

private:
    /** Преобразует координаты переменная—значение в плоский индекс с проверкой границ. */
    int flatten(int var, int val) const;
    /** Восстанавливает координаты переменная—значение по плоскому индексу. */
    std::pair<int, int> unflatten(int idx) const;
    /** Проверяет принадлежность переменной и значения заданным областям. */
    void validate_var_val(int var, int val) const;
    /** Строит смещения строк, проверяя переполнение целого индекса. */
    void rebuild_offsets();
    /** Строит разреженные списки межстрочных связей и связи взаимного торможения в строках. */
    void rebuild_neighbors();
    /** Пересчитывает число активных нейронов и охранное состояние каждой строки. */
    void recompute_guards();
    /** Пересчитывает входы из сдвигов, охранных сигналов, связей и запретов полного назначения. */
    void recompute_inputs();
    /** Прибавляет изменение охранного сигнала ко всем нейронам строки. */
    void apply_guard_change(int var, int old_guard, int new_guard);
    /** Равновероятно выбирает индекс переменной для следующего обновления. */
    int choose_variable();
    /** Выбирает наиболее несогласованный нейрон; равные оценки разрешает случайно. */
    int pick_best_inconsistent_neuron(int var, double& best_score);
    /** Меняет двоичное состояние, обновляет входы соседей и охранный сигнал строки. */
    void flip_neuron(int idx);
    /** Возвращает сохранённое число активных нейронов строки. */
    int row_active_count(int var) const;
    /** Проверяет наличие явно заданного запрета для пары плоских индексов. */
    bool has_explicit_forbidden_idx(int idx_a, int idx_b) const;

    int num_variables_;
    int num_neurons_;
    std::vector<int> domain_sizes_;
    std::vector<int> offsets_;

    std::string functional_type_;
    std::string update_rule_;
    std::string init_rule_;
    StopRule stop_rule_;

    double beta_;
    double w_forbidden_;
    double gamma_same_;
    double guard_bias_;
    double guard_delta_;
    double guard_phi_;
    std::size_t max_iterations_;

    std::vector<ForbiddenPair> forbidden_pairs_;
    std::vector<WeightedPair> weighted_pairs_;
    std::vector<std::vector<std::pair<int, double>>> neighbors_;
    std::vector<int> state_;
    std::vector<int> guard_state_;
    std::vector<double> inputs_;
    std::vector<int> row_counts_;
    std::vector<double> unary_bias_;
    std::vector<std::vector<int>> nogoods_;

    std::size_t iterations_ = 0;
    std::size_t transitions_ = 0;
    std::uint64_t seed_;
    std::mt19937_64 rng_;
};

}  // Пространство имён gds.
