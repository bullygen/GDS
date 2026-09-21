# Сборка и первое применение

## Получение исходного кода

Рабочий репозиторий находится по адресу <https://github.com/bullygen/GDS>.
Клонирование не требует исходных статей: все необходимые исходники, проверки и
учебные примеры входят в репозиторий. На Linux установите компилятор C++, CMake,
средства создания виртуального окружения и заголовки Python средствами своей
операционной системы. Например, пакеты Ubuntu называются `g++`, `cmake`,
`python3-venv` и `python3-dev`.

```bash
git clone https://github.com/bullygen/GDS.git
cd GDS
python3 -m venv .venv
source .venv/bin/activate
python -m pip install ".[test,plot]"
python tutorials/01_quick_start.py
```

Установочное имя проекта — `asc-gds`, импортируемое имя — `gds`. Команда `pip`
собирает расширение из C++-исходников. Отдельная установка NumPy для ядра не
нужна. Дополнение `plot` устанавливает Matplotlib, а `test` — pytest.
Сборка использует [официальную схему pybind11 с scikit-build-core](https://pybind11.readthedocs.io/en/stable/compiling.html).

Для разработки удобно выполнить `python -m pip install -e ".[test,plot]"`.
После изменения C++-файлов повторите эту команду: автоматическая пересборка
при импорте в настройках проекта не включена. После изменения только
Python-пакетов повторная сборка не требуется.

Если зависимости уже установлены, можно собирать без доступа к сети:

```bash
python -m pip install --no-build-isolation -e .
```

До этого в окружении должны присутствовать `scikit-build-core` и `pybind11`.
Параметр `--no-build-isolation` не устанавливает отсутствующие средства сборки.

## Проверка установки

```bash
python -m pytest -q
python tutorials/03_jssp_and_tsp.py
python tutorials/05_external_durations.py
python tutorials/07_multiobjective.py
python tutorials/09_visualization.py
```

Последний пример создаёт каталог `outputs`. Статьи, промежуточные сборки,
виртуальное окружение и рисунки не входят в Git. Для программного принятия
результата всегда проверяйте `schedule.feasible` и используйте точную оценку
`compiled.evaluate(assignment)`. Вызов `evaluate(..., exact=False)` предназначен
для исследования приближённой модели; такой результат имеет `exact=False`.

## Самостоятельная сборка C++

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 2
ctest --test-dir build --output-on-failure
cmake --install build --prefix "$PWD/build/install"
```

Режим по умолчанию не ищет Python и pybind11. Заголовки устанавливаются в
`include/gds`, библиотека и описание её импортируемой цели — в каталог библиотек
выбранного префикса. Во внешнем проекте используйте:

```cmake
cmake_minimum_required(VERSION 3.18)
project(MySchedule LANGUAGES CXX)
find_package(GDS CONFIG REQUIRED)
add_executable(example main.cpp)
target_link_libraries(example PRIVATE GDS::gds)
```

```cpp
/** @file main.cpp
 * @brief Первое применение установленной библиотеки без Python.
 */
#include <gds/solvers.hpp>
#include <iostream>

/** Строит расписание трёх независимых работ на двух аппаратах. */
int main() {
    auto problem = gds::make_parallel_jobs(2, 8, {2, 3, 2}).compile();
    auto schedule = gds::solve_gds(problem);
    std::cout << schedule.stop_reason << '\n';
    return schedule.feasible ? 0 : 1;
}
```

Передайте путь установки при конфигурировании внешнего проекта:
`cmake -S . -B build -DCMAKE_PREFIX_PATH=/полный/путь/к/GDS/build/install`.
Не включайте `.cpp`-файлы библиотеки непосредственно в пользовательскую программу.

## Сборка дистрибутивов

```bash
python -m pip install build
python -m build
```

В `dist` появятся исходный архив и двоичный пакет текущей платформы. Колесо,
собранное для CPython 3.12 на Linux, не является универсальным пакетом для
других версий Python и операционных систем. Исходный архив содержит всё
необходимое для повторной сборки, кроме устанавливаемых средств сборки.

## Типичные затруднения

Сообщение об отсутствии `Python.h` означает, что не установлены заголовочные
файлы выбранного Python. Ошибка поиска `pybind11Config.cmake` при ручной сборке
расширения устраняется передачей `-Dpybind11_DIR="$(python -m pybind11 --cmakedir)"`
и `-DGDS_BUILD_PYTHON=ON`. При обычной установке через `pip` путь определяется
средствами сборки.

Сообщение «Пустая область допустимых размещений» появляется до запуска сети.
Проверьте горизонт, длительность, полное попадание действия в окно, доступную
ёмкость и нулевые значения пригодности. У необязательной заявки область может
содержать единственное значение пропуска. Не увеличивайте число итераций для
исправления неверной постановки.
