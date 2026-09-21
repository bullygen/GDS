/** @file scheduling.hpp
 * @brief Постановка конечных задач планирования и независимая проверка расписаний.
 */
#pragma once
#include "gds/network.hpp"
#include <array>
#include <functional>
#include <map>
#include <memory>
#include <optional>

namespace gds {
/** Вектор момента или направления в трёхмерном евклидовом пространстве. */
using Vector3 = std::array<double, 3>;

/** Полуоткрытое окно [begin, end) в целых временных слотах. */
struct Window { int begin = 0; int end = 0; };

/** Требование к возобновляемому ресурсу на всём интервале действия. */
struct Demand { int resource = 0; double amount = 1.0; };

/** Изменение накопительного ресурса: скорость во время действия и скачок при окончании. */
struct Flow { int resource = 0; double rate = 0.0; double at_end = 0.0; };

/** Ресурс группировки: аппарат, прибор, антенна либо общий канал связи. */
struct Resource {
    std::string name;
    double capacity = 1.0;
    bool sequence = false; ///< Истина для последовательной линии одного аппарата.
    double memory_capacity = 1e100;
    double initial_memory = 0.0;
    Vector3 initial_momentum{0, 0, 0};
    double momentum_limit = 1e100; ///< Ограничение евклидовой нормы.
    std::vector<double> initial_state; ///< Состояние внешней модели.
    Vector3 initial_pointing{1, 0, 0};
    double initial_level = 0.0;
    double minimum_level = -1e100;
    double maximum_level = 1e100;
};

/** Способ исполнения действия на одном аппарате с дополнительными ресурсами. */
struct Mode {
    int resource = 0;
    double duration = 1.0; ///< Длительность в секундах до квантования.
    std::vector<Demand> demands;
    Vector3 momentum{0, 0, 0}; ///< Приращение в момент окончания.
    std::vector<Vector3> momentum_profile; ///< Приращение в зависимости от слота начала.
    std::vector<Flow> flows;
    double memory = 0.0; ///< Объём данных, записываемых в момент окончания.
    double downlink_rate = 0.0; ///< Скорость освобождения памяти в секунду.
    double remaining_momentum = 1.0; ///< Доля момента после разгрузки.
};

/** Заявка с окнами видимости, приоритетом и альтернативными способами исполнения. */
struct Action {
    std::string name;
    std::string group;
    std::vector<Mode> modes;
    std::vector<Window> windows;
    std::vector<double> suitability; ///< Значения по слотам; пустой массив означает единицу.
    Vector3 pointing{1, 0, 0};
    double priority = 1.0;
    bool optional = false;
};

/** Один нейрон: действие, начальный слот, способ исполнения и длительность. */
struct Placement {
    int action = -1;
    int mode = -1;
    int start = -1;
    int end = -1;
    int resource = -1;
    double duration = 0.0;
    double log_suitability = 0.0;
    bool omitted = false;
};

/** Отношение между окончанием первого и началом второго действия. */
struct Precedence {
    int before = 0;
    int after = 0;
    int min_gap = 0;
    int max_gap = -1; ///< Отрицательное значение снимает верхнюю границу.
};

/** Минимальное число выполненных заявок одной группы. */
struct Quota { std::string group; int minimum = 1; };

/** Функция парной пригодности: нуль запрещает пару, положительное число задаёт предпочтение. */
using PairSuitability = std::function<double(const Placement&, const Placement&)>;

/** Статус ответа внешней модели; неопределённость не означает допустимость. */
enum class DurationStatus { Feasible, Infeasible, Unknown, Timeout, Stale };

/** Запрос длительности действия либо перехода; before=-1 означает начальную ориентацию. */
struct DurationRequest {
    bool transition = false;
    int before = -1;
    int after = -1;
    int mode = 0;
    int resource = -1;
    double start_seconds = 0.0;
    double slot_seconds = 1.0;
    double nominal_seconds = 0.0;
    Vector3 from_pointing{1, 0, 0};
    Vector3 to_pointing{1, 0, 0};
    std::vector<double> state;
    std::string model_version;
};

/** Ответ внешнего расчёта; метаданные не восстанавливаются из весов сети. */
struct DurationResult {
    DurationStatus status = DurationStatus::Unknown;
    double seconds = 0.0;
    double risk = 0.0;
    double cost = 0.0;
    std::string certificate;
    std::string diagnostic;
    std::vector<double> end_state; ///< Пустой вектор означает неизменное состояние.
};

/** Единый контракт прямого расчёта внешней длительности. */
using DurationProvider = std::function<DurationResult(const DurationRequest&)>;
/** Предикат допустимости при заданной длительности в секундах. */
using DurationPredicate = std::function<DurationStatus(const DurationRequest&, double)>;

/** Ищет минимальную допустимую сеточную длительность; двоичный поиск требует монотонности. */
DurationResult estimate_duration(const DurationRequest& request, const DurationPredicate& predicate,
                                 int max_slots, bool monotone);
/** Консервативно округляет длительность вверх; отвергает неконечные и отрицательные числа. */
int duration_slots(double seconds, double slot_seconds);
/** Квантует непрерывный интервал: cover покрывает, inside вписывает, nearest округляет. */
Window quantize_interval(double begin, double end, double quantum, const std::string& mode = "cover");
/** Преобразует прямое восхождение и склонение в единичный вектор. */
Vector3 radec_to_unit(double ra_degrees, double dec_degrees);
/** Вычисляет угол между ненулевыми векторами в градусах. */
double angular_separation(const Vector3& left, const Vector3& right);
/** Строит маску угловой видимости по положениям тела и наблюдателя на каждом слоте. */
std::vector<double> angular_visibility(const Vector3& pointing, const std::vector<Vector3>& body,
                                       const std::vector<Vector3>& observer, double limit_degrees,
                                       bool at_least = true);
/** Вычисляет логарифм произведения неотрицательных функций пригодности. */
double log_suitability(const std::vector<double>& evidence);

/** Результат равномерного исследования функции внутри одного слота. */
struct SlotSample { double minimum; double maximum; double mean; bool aliasing_warning; };
/** Оценивает минимум, максимум и среднее методом трапеций, без гарантии между узлами. */
SlotSample sample_slot(const std::function<double(double)>& function, double begin, double end,
                       int samples = 17, double aliasing_ratio = 4.0);

/** Настраиваемая постановка; после изменения следует заново вызвать compile(). */
class Problem {
public:
    /** Создаёт сетку с заданным числом слотов и шагом в секундах. */
    Problem(int horizon, double slot_seconds = 1.0);
    /** Добавляет ресурс и возвращает его индекс. */
    int add_resource(const Resource& resource);
    /** Добавляет заявку и возвращает её индекс. */
    int add_action(const Action& action);
    /** Добавляет условие предшествования с необязательной верхней границей разрыва. */
    void add_precedence(const Precedence& constraint);
    /** Добавляет квоту на число выполненных заявок группы. */
    void add_quota(const Quota& quota);
    /** Запрещает вставку других групп внутрь блока указанной группы на одном аппарате. */
    void add_contiguous_group(const std::string& group);
    /** Задаёт парную функцию для двух действий; вызывается только при компиляции. */
    void add_pair_suitability(int first, int second, PairSuitability function);
    /** Устанавливает внешний расчёт длительности действий и версию модели. */
    void set_duration_provider(DurationProvider provider, const std::string& version);
    /** Устанавливает расчёт соседних переходов, включая начальный переход при initial=true. */
    void set_transition_provider(DurationProvider provider, const std::string& version, bool initial = false);
    /** Фиксирует отдельный точный проверяющий расчёт переходов вместо приближённого. */
    void set_exact_transition_provider(DurationProvider provider, const std::string& version);
    /** Возвращает число слотов горизонта. */
    int horizon() const;
    /** Возвращает шаг сетки в секундах. */
    double slot_seconds() const;
    /** Возвращает копию ресурсов. */
    std::vector<Resource> resources() const;
    /** Возвращает копию заявок. */
    std::vector<Action> actions() const;
    /** Строит конечные области значений и парные ограничения исключительно на C++. */
    class CompiledProblem compile() const;
private:
    friend class CompiledProblem;
    int horizon_;
    double quantum_;
    std::vector<Resource> resources_;
    std::vector<Action> actions_;
    std::vector<Precedence> precedences_;
    std::vector<Quota> quotas_;
    std::vector<std::string> contiguous_;
    std::vector<std::tuple<int, int, PairSuitability>> pair_functions_;
    DurationProvider duration_provider_, transition_provider_, exact_transition_provider_;
    std::string duration_version_, transition_version_, exact_version_;
    bool initial_transition_ = false;
};

/** Нарушение с участниками и временной областью для локального исправления. */
struct Violation {
    std::string kind;
    std::string message;
    std::vector<int> actions;
    int begin = -1;
    int end = -1;
    double magnitude = 1.0;
};
/** Полоса диаграммы Гантта: действие, переход либо свободный интервал. */
struct Segment {
    std::string kind;
    int resource = -1;
    int action = -1;
    double begin = 0.0;
    double end = 0.0;
    DurationResult metadata;
};
/** Временной ряд ресурса после каждого слота. */
struct ResourceTrace {
    int resource = -1;
    std::vector<double> load;
    std::vector<double> memory;
    std::vector<Vector3> momentum;
    std::vector<double> level;
};
/** Проверенное расписание и минимизируемые критерии: потери, простой, момент, непригодность, переходы. */
struct Schedule {
    std::vector<int> assignment;
    std::vector<Placement> placements;
    std::vector<Violation> violations;
    std::vector<Segment> segments;
    std::vector<ResourceTrace> traces;
    std::vector<double> objectives;
    bool feasible = false;
    bool exact = false; ///< Истина при использовании окончательной модели проверки.
    std::size_t iterations = 0;
    std::string stop_reason;
};

/** Неизменяемая конечная модель; кэш внешних вычислений принадлежит этой модели. */
class CompiledProblem {
public:
    /** Выполняет независимую проверку всех ограничений и точных соседних переходов. */
    Schedule evaluate(const std::vector<int>& assignment, bool exact = true) const;
    /** Создаёт GDS-сеть, сохраняя соответствие нейрона элементу области значений. */
    GDSNetwork make_network(std::uint64_t seed = 42) const;
    /** Возвращает области значений с координатами время—действие—ресурс. */
    std::vector<std::vector<Placement>> domains() const;
    /** Возвращает число вычислений внешних моделей и попаданий в кэш. */
    std::pair<std::size_t, std::size_t> cache_statistics() const;
    /** Очищает кэш после внешнего изменения модели или телеметрии. */
    void clear_cache() const;
    /** Возвращает исходную постановку для просмотра. */
    Problem problem() const;
private:
    friend class Problem;
    /** Проверяет описание и строит допустимые размещения и пары. */
    explicit CompiledProblem(const Problem& problem);
    /** Выполняет кэшируемый вызов провайдера с полным состоянием в ключе. */
    DurationResult query(const DurationRequest& request, bool exact) const;
    Problem problem_;
    std::vector<std::vector<Placement>> domains_;
    std::vector<ForbiddenPair> forbidden_;
    std::vector<WeightedPair> preferences_;
    mutable std::map<std::string, DurationResult> cache_;
    mutable std::size_t calls_ = 0, hits_ = 0;
};

/** Операция классической задачи обслуживания работ на станках. */
struct Operation { int machine = 0; int duration = 1; };
/** Переводит цепочки операций JSSP в конечную задачу ограничений. */
Problem make_jssp(int machines, int horizon, const std::vector<std::vector<Operation>>& jobs);
/** Переводит независимые работы на взаимозаменяемых аппаратах в задачу ограничений. */
Problem make_parallel_jobs(int machines, int horizon, const std::vector<int>& durations, int capacity = 1);
/** Строит CSP-сеть замкнутого коммивояжёра; бесконечность обозначает отсутствие дуги. */
GDSNetwork make_tsp_network(const std::vector<std::vector<double>>& costs, std::uint64_t seed = 42);
/** Проверяет перестановку городов и вычисляет стоимость замкнутого маршрута. */
double tour_cost(const std::vector<int>& tour, const std::vector<std::vector<double>>& costs);
/** Делит кандидатов на уменьшающие и увеличивающие норму суммарного момента по статье 2007 года. */
std::pair<std::vector<int>, std::vector<int>> find_reducers(const std::vector<Vector3>& momenta);
} // Пространство имён gds.
