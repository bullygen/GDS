# GDS — библиотека построения расписаний

GDS решает конечные задачи планирования на C++17 и предоставляет интерфейс
сценариев на Python через pybind11. Нейрон соответствует размещению действия:
начальному временному слоту, способу исполнения и аппарату. Библиотека строит
области размещения, функции пригодности и ограничения, ищет расписание, затем
независимо проверяет общие ресурсы и внешние длительности.

Реализованы сеть GDS с охранными нейронами, жадный поиск, исправление наиболее
конфликтных областей, генетический отбор расписаний сети с ранжированием NSGA-II
и многокритериальная дифференциальная эволюция GDE3. Имеются C++-построители JSSP,
распределения работ между аппаратами и замкнутой задачи коммивояжёра.

## С чего начать

Понадобятся Python версии не ниже 3.10, компилятор C++17 и CMake версии не ниже
3.18. В Linux нужны также заголовочные файлы Python. В Windows используется
компилятор из Visual Studio с поддержкой C++; в macOS — инструменты разработчика Xcode.

```bash
git clone https://github.com/bullygen/GDS.git
cd GDS
python3 -m venv .venv
source .venv/bin/activate
python -m pip install ".[test,plot]"
python tutorials/01_quick_start.py
python -m pytest
```

В Windows окружение включается командой `.venv\Scripts\activate`.

```python
import gds

# Две независимые линии обслуживают три работы длительностями 2, 3 и 2 слота.
problem = gds.make_parallel_jobs(machines=2, horizon=8, durations=[2, 3, 2])
compiled = problem.compile()
schedule = gds.solve_gds(compiled)
print(schedule.feasible, schedule.stop_reason)
for placement in schedule.placements:
    print(placement.action, placement.resource, placement.start, placement.end)
```

Положительный результат — только `schedule.feasible == True`. Исчерпание
предела эвристического поиска не доказывает неразрешимость задачи.

## Самостоятельная библиотека C++

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 2
ctest --test-dir build --output-on-failure
cmake --install build --prefix "$PWD/build/install"
```

Внешний проект подключает `find_package(GDS CONFIG REQUIRED)` и цель `GDS::gds`.
Для сборки ядра Python и pybind11 не нужны.

## Документация и примеры

- [Сборка, установка и первая задача](docs/quick_start.md).
- [Математическая постановка и алгоритмы](docs/algorithms.md).
- [Интерфейс и семантика данных](docs/api.md).
- [Внешние длительности и физическая проверка](docs/external_durations.md).
- [Последовательный курс примеров](tutorials/README.md).
- [Проверки и область подтверждённой работоспособности](docs/validation.md).
- [Планировщик хакатона Cosmo B: физика C++, GDE3, события и отчёты](hackathon/solver/README.md).

Расчётная часть находится в `include/gds` и `src`, привязки — в `bindings`.
Пакет `gds` только экспортирует C++-интерфейс. Отдельный необязательный пакет
`gds_plot` строит графики средствами Python; он не участвует в поиске расписаний.
Исходные статьи и рабочее задание сохраняются локально и исключены из публикации.
