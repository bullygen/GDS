"""Проверки математических контрактов, внешних моделей и установленного интерфейса."""
import itertools
import math

import pytest

import gds


def resource(name="Аппарат", capacity=1, sequence=True):
    """Создаёт ресурс для небольших независимо проверяемых задач."""
    value = gds.Resource()
    value.name, value.capacity, value.sequence = name, capacity, sequence
    return value


def action(name, duration=1, machine=0, window=None, **fields):
    """Создаёт действие с одним способом исполнения и явным временным окном."""
    mode = gds.Mode()
    mode.resource, mode.duration = machine, duration
    for key, value in fields.items():
        setattr(mode, key, value)
    value = gds.Action()
    value.name, value.modes = name, [mode]
    value.windows = [] if window is None else [gds.Window(*window)]
    return value


def result(seconds=0, status=gds.DurationStatus.FEASIBLE, state=None):
    """Возвращает детерминированный ответ демонстрационной внешней модели."""
    value = gds.DurationResult()
    value.seconds, value.status = seconds, status
    value.end_state = [] if state is None else state
    value.certificate = "проверка"
    return value


def options():
    """Ограничивает время тестового поиска небольшими воспроизводимыми пределами."""
    value = gds.SolverOptions()
    value.iterations, value.restarts, value.repair_steps = 1000, 3, 80
    value.population, value.generations = 6, 3
    return value


@pytest.mark.parametrize("size", [1, 4, 8])
def test_queens(size):
    """Сверяет найденных ферзей непосредственно по столбцам и диагоналям."""
    network = gds.GDSNetwork.make_n_queens(size)
    answer = network.run(100000)
    assert answer.satisfied
    for i in range(size):
        for j in range(i + 1, size):
            assert answer.assignment[i] != answer.assignment[j]
            assert abs(answer.assignment[i] - answer.assignment[j]) != j - i


def test_unsatisfiable_and_nogood():
    """Проверяет отсутствие ложного успеха и точность запрета полного назначения."""
    assert not gds.GDSNetwork.make_n_queens(2).run(1000).satisfied
    network = gds.GDSNetwork(3, [2, 2, 2])
    network.add_nogood([0, 0, 0])
    for values in itertools.product(range(2), repeat=3):
        network.set_assignment(values)
        assert network.is_satisfied() == (values != (0, 0, 0))


def test_incremental_inputs_and_persistence(tmp_path):
    """Сверяет инкрементальные входы с полным пересчётом и точным продолжением файла."""
    network = gds.GDSNetwork.make_n_queens(4)
    network.set_unary_bias(1, 2, 0.1234567890123456)
    network.add_weighted_pair(0, 0, 3, 2, -0.25)
    network.finalize_weights()
    for _ in range(50):
        network.step()
        inputs = network.get_inputs()
        network.set_state(network.get_state())
        assert network.get_inputs() == pytest.approx(inputs)
    filename = str(tmp_path / "network.gdsw")
    network.save_weights(filename)
    restored = gds.GDSNetwork.load_weights(filename)
    assert restored.get_state() == network.get_state()
    assert restored.get_unary_bias(1, 2) == network.get_unary_bias(1, 2)
    for _ in range(100):
        original_step, restored_step = network.step(), restored.step()
        assert original_step.neuron_index == restored_step.neuron_index
        assert restored.get_inputs() == pytest.approx(network.get_inputs())
        assert restored.get_state() == network.get_state()
    (tmp_path / "broken").write_text("GDSW2\n3\n", encoding="utf-8")
    with pytest.raises(Exception):
        gds.GDSNetwork.load_weights(str(tmp_path / "broken"))


def test_jssp_exhaustive_equivalence():
    """Сопоставляет все назначения маленькой JSSP с независимыми условиями интервалов."""
    compiled = gds.make_jssp(2, 4, [[gds.Operation(0, 1), gds.Operation(1, 1)], [gds.Operation(0, 2)]]).compile()
    domains = compiled.domains
    for indices in itertools.product(*(range(len(d)) for d in domains)):
        a, b, c = [d[i] for d, i in zip(domains, indices)]
        expected = a.end <= b.start and not (a.start < c.end and c.start < a.end)
        assert compiled.evaluate(indices).feasible == expected
        network = compiled.make_network()
        network.set_assignment(indices)
        assert network.is_satisfied() == expected


def test_cumulative_capacity_requires_global_verifier():
    """Три потребителя нарушают ёмкость два, хотя каждая пара допустима."""
    compiled = gds.make_parallel_jobs(1, 1, [1, 1, 1], capacity=2).compile()
    network = compiled.make_network()
    network.set_assignment([0, 0, 0])
    assert network.is_satisfied()
    schedule = compiled.evaluate([0, 0, 0])
    assert not schedule.feasible
    assert schedule.traces[0].load == [3]
    assert any(v.kind == "capacity" and v.actions == [0, 1, 2] for v in schedule.violations)
    limited = options()
    limited.iterations, limited.restarts, limited.repair_steps = 30, 1, 5
    assert not gds.solve_gds(compiled, limited).feasible


def test_constellation_shared_resource():
    """Независимые аппараты могут работать одновременно, общий канал запрещает это."""
    problem = gds.Problem(3)
    problem.add_resource(resource("Первый"))
    problem.add_resource(resource("Второй"))
    channel = problem.add_resource(resource("Общий канал", sequence=False))
    problem.add_action(action("Передача А", machine=0, window=(0, 1), demands=[gds.Demand(channel)]))
    problem.add_action(action("Передача Б", machine=1, window=(0, 1), demands=[gds.Demand(channel)]))
    assert not problem.compile().evaluate([0, 0]).feasible


def test_resource_traces_and_momentum_profile():
    """Проверяет порядок освобождения памяти, записи данных, разгрузки и накопления."""
    problem = gds.Problem(3)
    sat = resource()
    sat.memory_capacity, sat.initial_memory = 5, 1
    sat.initial_momentum, sat.momentum_limit = [1, 0, 0], 3
    sat.initial_level, sat.minimum_level, sat.maximum_level = 10, 0, 10
    problem.add_resource(sat)
    problem.add_action(action("Съёмка", window=(0, 1), memory=4, momentum_profile=[[2, 0, 0], [1, 0, 0], [0, 0, 0]], flows=[gds.Flow(0, -2, 0)]))
    problem.add_action(action("Передача", window=(1, 2), downlink_rate=3, remaining_momentum=0, momentum=[1, 0, 0]))
    schedule = problem.compile().evaluate([0, 0])
    assert schedule.feasible
    trace = schedule.traces[0]
    assert trace.memory == [5, 2, 2]
    assert trace.momentum == [[3, 0, 0], [1, 0, 0], [1, 0, 0]]
    assert trace.level == [8, 8, 8]


def test_visibility_and_quantization():
    """Проверяет маску по всему действию и физическое округление длительности."""
    assert gds.duration_slots(2.01, 1) == 3
    assert gds.duration_slots(0, 1) == 0
    assert gds.angular_separation([1, 0, 0], [0, 1, 0]) == pytest.approx(90)
    assert gds.radec_to_unit(90, 0) == pytest.approx([0, 1, 0])
    mask = gds.angular_visibility([1, 0, 0], [[0, 1, 0], [1, 0, 0], [0, 1, 0]], [], 45)
    assert mask == [1, 0, 1]
    problem = gds.Problem(3)
    problem.add_resource(resource())
    observation = action("Наблюдение", duration=2)
    observation.suitability = mask
    problem.add_action(observation)
    with pytest.raises(ValueError, match="Пустая область"):
        problem.compile()
    interval = gds.quantize_interval(0.1, 2.9, 1, "inside")
    assert (interval.begin, interval.end) == (1, 2)
    assert gds.sample_slot(lambda x: x * x, 0, 1, 1001).mean == pytest.approx(1 / 3, abs=1e-6)
    assert gds.log_suitability([0.5, 2]) == 0


@pytest.mark.parametrize("seconds,quantum", [(-1, 1), (math.nan, 1), (math.inf, 1), (1, 0), (1, math.nan)])
def test_invalid_duration(seconds, quantum):
    """Некорректные единицы и длительности не должны проникать в индексирование."""
    with pytest.raises(ValueError):
        gds.duration_slots(seconds, quantum)


def test_duration_estimator():
    """Сопоставляет двоичный и немонотонный поиск и сохранение неизвестного статуса."""
    request = gds.DurationRequest()
    request.slot_seconds = 0.5
    calls = []

    def monotone(_, duration):
        """Разрешает длительность от 1,1 секунды и записывает пробные значения."""
        calls.append(duration)
        return gds.DurationStatus.FEASIBLE if duration >= 1.1 else gds.DurationStatus.INFEASIBLE

    assert gds.estimate_duration(request, monotone, 20, True).seconds == 1.5
    assert len(calls) < 10
    nonmonotone = lambda _, t: gds.DurationStatus.FEASIBLE if t == 1 else gds.DurationStatus.INFEASIBLE
    assert gds.estimate_duration(request, nonmonotone, 6, False).seconds == 1
    unknown = lambda _, t: gds.DurationStatus.UNKNOWN
    assert gds.estimate_duration(request, unknown, 6, True).status == gds.DurationStatus.UNKNOWN


def test_external_action_duration():
    """Строит области значений из внешней длительности и сохраняет сертификат."""
    problem = gds.Problem(6, 2)
    problem.add_resource(resource())
    problem.add_action(action("Внешняя операция"))
    problem.set_duration_provider(lambda request: result(3.1), "модель-1")
    compiled = problem.compile()
    assert all(x.end - x.start == 2 for x in compiled.domains[0])
    schedule = compiled.evaluate([0])
    assert schedule.feasible
    assert next(s for s in schedule.segments if s.kind == "action").metadata.seconds == 3.1


def test_adjacency_not_all_pairs_and_state_cache():
    """Запрет A→C не должен исключать допустимую последовательность A→B→C."""
    problem = gds.Problem(6)
    sat = resource()
    sat.initial_state = [0]
    problem.add_resource(sat)
    for i, start in enumerate([0, 2, 4]):
        problem.add_action(action(str(i), window=(start, start + 1)))
    requests = []

    def transition(request):
        """Проверяет соседство и передаёт изменённое состояние следующему вызову."""
        requests.append((request.before, request.after, request.state))
        if request.before == 0 and request.after == 2:
            return result(status=gds.DurationStatus.INFEASIBLE)
        return result(0.25, state=[request.state[0] + 1])

    problem.set_transition_provider(transition, "версия-1")
    compiled = problem.compile()
    schedule = compiled.evaluate([0, 0, 0])
    assert schedule.feasible
    assert requests == [(0, 1, [0]), (1, 2, [1])]
    assert len([s for s in schedule.segments if s.kind == "transition"]) == 2
    compiled.evaluate([0, 0, 0])
    assert len(requests) == 2
    assert compiled.cache_statistics() == (2, 2)
    compiled.clear_cache()
    compiled.evaluate([0, 0, 0])
    assert len(requests) == 4


@pytest.mark.parametrize("status", [gds.DurationStatus.UNKNOWN, gds.DurationStatus.TIMEOUT, gds.DurationStatus.STALE, gds.DurationStatus.INFEASIBLE])
def test_external_failure_is_not_success(status):
    """Каждый отказ внешней проверки оставляет расписание недопустимым."""
    problem = gds.Problem(4)
    problem.add_resource(resource())
    problem.add_action(action("А", window=(0, 1)))
    problem.add_action(action("Б", window=(2, 3)))
    problem.set_transition_provider(lambda request: result(status=status), "модель")
    schedule = problem.compile().evaluate([0, 0])
    assert not schedule.feasible
    assert any(v.kind == "transition" for v in schedule.violations)


def test_exact_rejection_and_repair():
    """Точная модель опровергает приближение, после чего поиск находит больший разрыв."""
    problem = gds.Problem(6)
    problem.add_resource(resource())
    problem.add_action(action("А", window=(0, 1)))
    problem.add_action(action("Б", window=(1, 6)))
    problem.add_precedence(gds.Precedence(0, 1))
    problem.set_transition_provider(lambda request: result(0), "приближённая")
    problem.set_exact_transition_provider(lambda request: result(2), "точная")
    compiled = problem.compile()
    assert compiled.evaluate([0, 0], exact=False).feasible
    assert not compiled.evaluate([0, 0]).feasible
    schedule = gds.solve_gds(compiled, options())
    assert schedule.feasible
    assert schedule.placements[1].start >= 3


def test_optional_quota_precedence_and_contiguous():
    """Проверяет пропуск необязательной заявки, квоту и непрерывность группы."""
    problem = gds.Problem(3)
    problem.add_resource(resource())
    for i in range(3):
        value = action(str(i), window=(i, i + 1))
        value.group = "Группа" if i != 1 else "Другая"
        value.optional = i == 1
        problem.add_action(value)
    problem.add_contiguous_group("Группа")
    compiled = problem.compile()
    assert not compiled.evaluate([0, 0, 0]).feasible
    assert compiled.evaluate([0, 1, 0]).feasible
    problem.add_quota(gds.Quota("Другая", 1))
    assert not problem.compile().evaluate([0, 1, 0]).feasible


@pytest.mark.parametrize("solver", [gds.greedy, gds.solve_gds])
def test_schedule_solvers(solver):
    """Проверяет ресурсы результата непосредственно по интервалам выбранных действий."""
    compiled = gds.make_parallel_jobs(2, 6, [2, 2, 2]).compile()
    schedule = solver(compiled, options())
    assert schedule.feasible
    for a, b in itertools.combinations(schedule.placements, 2):
        assert a.resource != b.resource or a.end <= b.start or b.end <= a.start


def test_local_repair():
    """Исправляет заведомо совпадающие интервалы трёх работ."""
    compiled = gds.make_parallel_jobs(1, 6, [2, 2, 2]).compile()
    assert not compiled.evaluate([0, 0, 0]).feasible
    assert gds.min_conflicts(compiled, [0, 0, 0], options()).feasible


@pytest.mark.parametrize("solver", [gds.genetic, gds.gde3])
def test_evolution_and_pareto(solver):
    """Проверяет размер популяции, воспроизводимость и недоминируемость выданного фронта."""
    compiled = gds.make_parallel_jobs(2, 5, [1, 2, 1]).compile()
    first, second = solver(compiled, options()), solver(compiled, options())
    assert len(first.population) == 6 and first.generations == 3
    assert [s.assignment for s in first.population] == [s.assignment for s in second.population]
    assert first.pareto_front and any(s.feasible for s in first.population)
    for member in first.pareto_front:
        assert not any(gds.dominates(other, member) for other in first.population)
        assert compiled.evaluate(member.assignment).feasible == member.feasible


def test_pareto_known_vectors():
    """Сопоставляет фронт с вручную заданными несравнимыми векторами критериев."""
    values = []
    for x in [[0, 2], [1, 1], [2, 0], [2, 2]]:
        value = gds.Schedule()
        value.feasible, value.objectives = True, x
        values.append(value)
    assert gds.pareto_indices(values) == [0, 1, 2]


def test_directed_tsp_and_missing_edges():
    """Проверяет стоимость замыкающей дуги и запрет отсутствующего ребра."""
    costs = [[0, 1, math.inf], [math.inf, 0, 1], [1, math.inf, 0]]
    assert gds.tour_cost([0, 1, 2], costs) == 3
    assert math.isinf(gds.tour_cost([0, 2, 1], costs))
    tour = gds.solve_tsp(costs, options())
    assert tour.feasible and tour.cost == 3
    with pytest.raises(ValueError):
        gds.tour_cost([0, 0, 1], costs)


def test_invalid_network_parameters():
    """Неподдерживаемые режимы и неконечные веса должны отвергаться явно."""
    with pytest.raises(ValueError):
        gds.GDSNetwork.make_n_queens(-1)
    with pytest.raises(ValueError):
        gds.GDSNetwork(1, [1], functional_type="несуществующая")
    network = gds.GDSNetwork(1, [2])
    with pytest.raises(ValueError):
        network.set_unary_bias(0, 0, math.nan)
    with pytest.raises(ValueError):
        network.beta = math.inf


def test_pair_suitability_compiled_once():
    """Проверяет перевод парной функции в вес и запрет без вызовов при поиске."""
    problem = gds.make_parallel_jobs(1, 3, [1, 1])
    calls = []

    def suitability(a, b):
        """Разрешает прямой порядок и даёт постоянный мягкий множитель."""
        calls.append((a.start, b.start))
        return 0.5 if a.end <= b.start else 0.0

    problem.add_pair_suitability(0, 1, suitability)
    compiled = problem.compile()
    count = len(calls)
    assert count == 9
    assert compiled.evaluate([0, 1]).feasible
    assert not compiled.evaluate([1, 0]).feasible
    assert compiled.evaluate([0, 1]).objectives[3] == pytest.approx(-math.log(0.5))
    assert compiled.make_network().get_weight(0, 0, 1, 1) < 0
    assert gds.solve_gds(compiled, options()).feasible
    assert len(calls) == count


def test_initial_transition_and_unknown_not_cached():
    """Проверяет начальную ориентацию и повторный вызов после неопределённого ответа."""
    problem = gds.Problem(5)
    sat = resource()
    sat.initial_pointing = [0, 1, 0]
    problem.add_resource(sat)
    problem.add_action(action("Первое", window=(2, 3)))
    calls = []

    def provider(request):
        """Первый вызов остаётся неизвестным, второй подтверждает начальный переход."""
        calls.append(request.before)
        assert request.from_pointing == [0, 1, 0]
        return result(1, gds.DurationStatus.UNKNOWN if len(calls) == 1 else gds.DurationStatus.FEASIBLE)

    problem.set_transition_provider(provider, "начальная-1", initial=True)
    compiled = problem.compile()
    assert not compiled.evaluate([0]).feasible
    assert compiled.evaluate([0]).feasible
    assert calls == [-1, -1]


def test_duration_grows_with_history():
    """Отвергает действие, чья точная длительность после первого действия больше резерва."""
    problem = gds.Problem(5)
    sat = resource()
    sat.initial_state = [0]
    problem.add_resource(sat)
    problem.add_action(action("Первое", window=(0, 1)))
    problem.add_action(action("Второе", window=(2, 3)))

    def provider(request):
        """Каждая операция повышает состояние, которое удлиняет следующую."""
        return result(1 + request.state[0], state=[request.state[0] + 1])

    problem.set_duration_provider(provider, "с-историей-1")
    compiled = problem.compile()
    schedule = compiled.evaluate([0, 0])
    assert not schedule.feasible
    assert any(v.kind == "duration" for v in schedule.violations)


def test_constraint_vector_dominance():
    """Разноимённые нарушения нельзя взаимно компенсировать скалярной суммой."""
    left, right = gds.Schedule(), gds.Schedule()
    left.objectives, right.objectives = [0], [0]
    memory, momentum = gds.Violation(), gds.Violation()
    memory.kind, memory.magnitude = "memory", 1
    momentum.kind, momentum.magnitude = "momentum", 100
    left.violations, right.violations = [memory], [momentum]
    assert not gds.dominates(left, right)
    assert not gds.dominates(right, left)


def test_external_exception_propagates():
    """Ошибка адаптера должна передаваться пользователю, сохраняя исходную причину."""
    problem = gds.make_parallel_jobs(1, 3, [1])

    def broken(request):
        """Имитирует сбой внешней модели."""
        raise RuntimeError("Нет связи с моделью")

    problem.set_duration_provider(broken, "ошибка")
    with pytest.raises(RuntimeError, match="Нет связи"):
        problem.compile()
