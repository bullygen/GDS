// Блочные обновления сетки GDS. Динамические охранные условия вычисляются
// заново на всём изменяемом горизонте, без линейной аппроксимации физики.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <gds/network.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <string>
#include <vector>

namespace py = pybind11;
using Plan = std::vector<int>; // sat * horizon + step: 0 idle, 1 calibration, 2 + job.
struct Satellite {
    std::string id;
    double capacity, energy, temp, base, heater, calibration, downlink, relay;
    int age;
    std::vector<double> solar, target;
    std::vector<bool> available, d, r;
};
struct Job {
    std::string id;
    int kind, release, deadline, work, remaining, completed;
    std::vector<int> eligible;
};
struct Evaluation {
    // binary, shared capacity, energy, thermal, calibration, target, continuity
    std::array<int, 7> violations{};
    std::vector<int> remaining, completed;
    std::vector<double> energy, temperature;
    std::vector<int> age;
    int first_bad_job = -1;
    int total() const { return std::accumulate(violations.begin(), violations.end(), 0); }
};

class CompiledNetwork {
    int n, horizon, start, valid, parallel;
    double reserve, efficiency_in, efficiency_out, decay, gain, heater_below;
    double temp_min, temp_max, charge_min, charge_max;
    std::vector<Satellite> sats;
    std::vector<Job> jobs;
    std::vector<std::vector<std::pair<int, int>>> domains;
    std::vector<int> active_sat; // Работа, начатая до текущей границы.
    Plan baseline;
    std::vector<std::string> abandoned;
    int pos(int s, int t) const { return s * horizon + t; }

    Evaluation simulate(const Plan& plan, bool trace) const {
        if (plan.size() != static_cast<std::size_t>(n * horizon))
            throw std::invalid_argument("Invalid plan dimensions");
        Evaluation out;
        std::vector<double> energy, temp;
        std::vector<int> ages, last_step(jobs.size(), -2), last_sat(jobs.size(), -1);
        std::vector<int> work_count(jobs.size(), 0);
        for (const auto& s : sats) { energy.push_back(s.energy); temp.push_back(s.temp); ages.push_back(s.age); }
        for (std::size_t j = 0; j < jobs.size(); ++j) {
            out.remaining.push_back(jobs[j].remaining);
            out.completed.push_back(jobs[j].completed);
            if (active_sat[j] >= 0) { last_step[j] = start - 1; last_sat[j] = active_sat[j]; }
        }
        if (trace) {
            out.energy.resize(n * (horizon - start + 1));
            out.temperature.resize(out.energy.size()); out.age.resize(out.energy.size());
        }
        auto save = [&](int t) {
            if (!trace) return;
            for (int s = 0; s < n; ++s) {
                auto p = s * (horizon - start + 1) + t - start;
                out.energy[p] = energy[s]; out.temperature[p] = temp[s]; out.age[p] = ages[s];
            }
        };
        save(start);
        for (int t = start; t < horizon; ++t) {
            int downlinks = 0;
            std::vector<bool> used(jobs.size(), false);
            for (int s = 0; s < n; ++s) {
                const auto& sat = sats[s];
                int code = plan[pos(s, t)];
                if (code < 0 || code >= static_cast<int>(jobs.size()) + 2)
                    throw std::invalid_argument("Unknown command code");
                int j = code - 2;
                bool okay = true;
                double power = code == 1 ? sat.calibration : 0.;
                auto fail = [&](int component) {
                    if (okay) {
                        ++out.violations[component];
                        if (j >= 0 && out.first_bad_job < 0) out.first_bad_job = j;
                    }
                    okay = false;
                };
                if (code && !sat.available[t]) fail(0);
                if (code >= 2) {
                    const auto& job = jobs[j];
                    power = job.kind == 2 ? sat.downlink : sat.relay;
                    if (out.remaining[j] <= 0 || t < job.release || t >= job.deadline) fail(5);
                    if (std::find(job.eligible.begin(), job.eligible.end(), s) == job.eligible.end()
                        || !(job.kind == 2 ? sat.d[t] : sat.r[t])) fail(0);
                    if (ages[s] >= valid) fail(4);
                }
                double heater = temp[s] < heater_below ? sat.heater : 0.;
                auto transition = [&](double payload) {
                    double load = sat.base + heater + payload;
                    double delta = (sat.solar[t] - load) * 300. / 3600.;
                    if (delta >= 0.) delta *= (charge_min <= temp[s] && temp[s] <= charge_max) ? efficiency_in : 0.;
                    else delta /= efficiency_out;
                    double equilibrium = sat.target[t] + gain * load;
                    return std::pair<double, double>{energy[s] + delta, equilibrium + (temp[s] - equilibrium) * decay};
                };
                auto next = transition(power);
                if (code) {
                    if (energy[s] < sat.capacity * reserve / 100. - 1e-9
                        || next.first < sat.capacity * reserve / 100. - 1e-9) fail(2);
                    if (temp[s] < temp_min || temp[s] > temp_max || next.second < temp_min || next.second > temp_max) fail(3);
                }
                if (code >= 2 && okay) {
                    if (used[j] || (jobs[j].kind == 2 && downlinks >= parallel)) fail(1);
                }
                if (!okay) { next = transition(0.); code = 0; }
                energy[s] = std::clamp(next.first, 0., sat.capacity);
                temp[s] = next.second;
                ages[s] = code == 1 ? 0 : ages[s] + 1;
                if (code >= 2) {
                    if ((work_count[j] || jobs[j].remaining < jobs[j].work)
                        && (last_step[j] != t - 1 || last_sat[j] != s)) ++out.violations[6];
                    last_step[j] = t; last_sat[j] = s; ++work_count[j];
                    used[j] = true;
                    downlinks += jobs[j].kind == 2;
                    if (--out.remaining[j] == 0) out.completed[j] = t + 1;
                }
            }
            save(t + 1);
        }
        for (std::size_t j = 0; j < jobs.size(); ++j)
            if (work_count[j] && out.remaining[j] != 0) ++out.violations[5];
        return out;
    }

public:
    CompiledNetwork(py::dict scenario, py::dict states, py::list registry, int boundary, py::dict running) {
        auto time = scenario["time"].cast<py::dict>();
        horizon = time["steps"].cast<int>(); start = boundary;
        if (start < 0 || start > horizon) throw std::invalid_argument("Invalid boundary");
        auto model = scenario["model"].cast<py::dict>();
        valid = model["calibration_valid_steps"].cast<int>(); parallel = model["downlink_parallel_limit"].cast<int>();
        reserve = model["reserve_soc_pct"].cast<double>();
        efficiency_in = model["charge_efficiency"].cast<double>(); efficiency_out = model["discharge_efficiency"].cast<double>();
        decay = std::exp(-300. / model["thermal_tau_s"].cast<double>());
        gain = model["thermal_gain_c_per_w"].cast<double>(); heater_below = model["heater_below_c"].cast<double>();
        temp_min = model["payload_min_c"].cast<double>(); temp_max = model["payload_max_c"].cast<double>();
        charge_min = model["charge_min_c"].cast<double>(); charge_max = model["charge_max_c"].cast<double>();
        auto environment = scenario["environment"].cast<py::dict>();
        for (auto item : scenario["satellites"].cast<py::list>()) {
            auto v = py::cast<py::dict>(item);
            Satellite sat; sat.id = v["id"].cast<std::string>();
            auto state = states[py::str(sat.id)].cast<py::dict>();
            sat.capacity = v["capacity_wh"].cast<double>(); sat.energy = state["energy_wh"].cast<double>();
            sat.temp = state["temp_c"].cast<double>(); sat.age = state["calibration_age_steps"].cast<int>();
            sat.base = v["base_w"].cast<double>(); sat.heater = v["heater_w"].cast<double>();
            sat.calibration = v["calibration_w"].cast<double>(); sat.downlink = v["downlink_w"].cast<double>();
            sat.relay = v["relay_w"].cast<double>();
            auto env = environment[py::str(sat.id)].cast<py::dict>();
            sat.solar = env["solar_w"].cast<std::vector<double>>(); sat.target = env["thermal_target_c"].cast<std::vector<double>>();
            sat.d = env["downlink_available"].cast<std::vector<bool>>(); sat.r = env["relay_available"].cast<std::vector<bool>>();
            sat.available.assign(horizon, true);
            for (auto f : scenario["failures"].cast<py::list>()) {
                auto failure = py::cast<py::dict>(f);
                if (failure["satellite_id"].cast<std::string>() == sat.id)
                    for (int t = failure["start_step"].cast<int>(); t < failure["end_step"].cast<int>(); ++t) sat.available[t] = false;
            }
            sats.push_back(std::move(sat));
        }
        std::sort(sats.begin(), sats.end(), [](const auto& a, const auto& b) { return a.id < b.id; });
        n = static_cast<int>(sats.size());
        auto satellite_index = [&](std::string id) {
            for (int s = 0; s < n; ++s) if (sats[s].id == id) return s;
            throw std::invalid_argument("Unknown eligible satellite");
        };
        for (auto item : registry) {
            auto v = py::cast<py::dict>(item);
            Job j; j.id = v["id"].cast<std::string>(); j.kind = v["kind"].cast<std::string>() == "downlink" ? 2 : 3;
            j.release = v["release_step"].cast<int>(); j.deadline = v["deadline_step"].cast<int>();
            j.work = v["work_steps"].cast<int>(); j.remaining = v["remaining_steps"].cast<int>();
            j.completed = v["completed_step"].is_none() ? -1 : v["completed_step"].cast<int>();
            for (auto sid : v["eligible_satellites"].cast<py::list>()) j.eligible.push_back(satellite_index(py::cast<std::string>(sid)));
            active_sat.push_back(running.contains(py::str(j.id)) ? satellite_index(running[py::str(j.id)].cast<std::string>()) : -1);
            jobs.push_back(std::move(j));
        }
        baseline.assign(n * horizon, 0);
        // Незавершённый непрерывный блок фиксируется до конца; если событие
        // сделало продолжение невозможным, он остаётся незавершённым в истории.
        for (std::size_t j = 0; j < jobs.size(); ++j) if (active_sat[j] >= 0 && jobs[j].remaining) {
            if (start + jobs[j].remaining > jobs[j].deadline) { abandoned.push_back(jobs[j].id); continue; }
            for (int t = start; t < start + jobs[j].remaining; ++t) baseline[pos(active_sat[j], t)] = static_cast<int>(j) + 2;
        }
        while (true) {
            auto e = simulate(baseline, false);
            if (!e.total()) break;
            int bad = e.first_bad_job;
            if (bad < 0) throw std::runtime_error("Invalid continuation baseline");
            for (auto& code : baseline) if (code == bad + 2) code = 0;
            abandoned.push_back(jobs[bad].id);
        }
        domains.resize(jobs.size());
        for (std::size_t j = 0; j < jobs.size(); ++j) {
            const auto& job = jobs[j];
            if (job.remaining != job.work || job.completed >= 0 || job.work > valid) continue;
            for (int s : job.eligible) for (int t = std::max(start, job.release); t + job.work <= job.deadline; ++t) {
                bool okay = true;
                for (int u = t; u < t + job.work; ++u)
                    if (!sats[s].available[u] || !(job.kind == 2 ? sats[s].d[u] : sats[s].r[u])) { okay = false; break; }
                if (okay) domains[j].emplace_back(s, t);
            }
        }
    }

    py::dict inspect(const Plan& plan) const {
        auto e = simulate(plan, true);
        py::dict result;
        result["U"] = e.total(); result["violations"] = e.violations;
        result["remaining"] = e.remaining; result["completed"] = e.completed;
        result["energy_wh"] = e.energy; result["temp_c"] = e.temperature; result["calibration_age_steps"] = e.age;
        return result;
    }

    py::dict decode(const std::vector<double>& genes, int attempts) const {
        if (genes.size() != 3 * jobs.size() || attempts < 1) throw std::invalid_argument("Invalid genome or attempt limit");
        for (double gene : genes) if (!std::isfinite(gene) || gene < 0. || gene > 1.) throw std::invalid_argument("Gene outside [0,1]");
        Plan plan = baseline;
        std::vector<int> order(jobs.size()); std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return genes[3*a] < genes[3*b]; });
        int proposals = 0, accepted = 0;
        std::array<int, 7> guards{};
        std::vector<int> history;
        for (int j : order) {
            const auto& options = domains[j];
            if (options.empty() || genes[3*j+2] < 0.08) continue;
            int first = std::min(static_cast<int>(options.size()) - 1, static_cast<int>(genes[3*j+1] * options.size()));
            for (int attempt = 0; attempt < std::min(attempts, static_cast<int>(options.size())); ++attempt) {
                // Равномерно распределённые пробы охватывают все аппараты и окна.
                int index = (first + static_cast<int>((static_cast<long long>(attempt) * options.size()) / std::min(attempts, static_cast<int>(options.size())))) % options.size();
                auto [s, t] = options[index];
                bool empty = true;
                for (int u = t; u < t + jobs[j].work; ++u) if (plan[pos(s, u)]) { empty = false; break; }
                if (!empty) continue; // Непересечение блоков — структурный W-фактор.
                for (int u = t; u < t + jobs[j].work; ++u) plan[pos(s, u)] = j + 2;
                ++proposals;
                auto e = simulate(plan, false);
                history.push_back(e.total());
                for (int c = 0; c < 7; ++c) guards[c] += e.violations[c];
                int calibration = -1;
                if (e.violations[4]) {
                    // Калибровка является отдельным нейроном перед непрерывным блоком.
                    // Ищем последний свободный слот, в котором срока калибровки хватит
                    // до последнего рабочего слота. Тепловой допуск проверяет U.
                    for (int u = t - 1; u >= start && t + jobs[j].work - 2 - u < valid; --u) {
                        if (!plan[pos(s, u)] && sats[s].available[u]) {
                            plan[pos(s, u)] = 1; ++proposals;
                            e = simulate(plan, false); history.push_back(e.total());
                            for (int c = 0; c < 7; ++c) guards[c] += e.violations[c];
                            if (!e.total()) { calibration = u; break; }
                            plan[pos(s, u)] = 0;
                        }
                    }
                }
                if (!e.total()) { ++accepted; break; }
                if (calibration >= 0) plan[pos(s, calibration)] = 0;
                for (int u = t; u < t + jobs[j].work; ++u) plan[pos(s, u)] = 0;
            }
        }
        auto e = simulate(plan, false);
        if (e.total()) throw std::runtime_error("GDS guard failed to preserve U=0");
        py::dict result;
        result["plan"] = plan; result["completed"] = e.completed;
        result["U"] = e.total(); result["proposals"] = proposals; result["accepted_blocks"] = accepted;
        result["guard_violations"] = guards; result["proposal_U"] = history;
        return result;
    }

    py::dict description() const {
        py::dict result;
        std::vector<std::string> ids; for (const auto& s : sats) ids.push_back(s.id);
        std::vector<std::string> job_ids; for (const auto& j : jobs) job_ids.push_back(j.id);
        // b[m,t]=1 соответствует запрещённому действию. Заявки дополнительно
        // фильтруются собственными release/deadline/eligible в domains.
        std::vector<int> b(3 * n * horizon);
        for (int s = 0; s < n; ++s) for (int t = 0; t < horizon; ++t) {
            b[(3*s)*horizon+t] = !sats[s].available[t];
            b[(3*s+1)*horizon+t] = !sats[s].available[t] || !sats[s].d[t];
            b[(3*s+2)*horizon+t] = !sats[s].available[t] || !sats[s].r[t];
        }
        result["satellite_ids"] = ids; result["job_ids"] = job_ids;
        result["horizon"] = horizon; result["start"] = start;
        result["b"] = b; result["domains"] = domains; result["abandoned_jobs"] = abandoned;
        return result;
    }

    void save_network(const Plan& plan, const std::string& path) const {
        if (simulate(plan, false).total()) throw std::invalid_argument("Cannot export an invalid network state");
        // Основная библиотека хранит one-hot строку (idle,c,d,r) на каждый (s,t).
        // Нулевой idle-столбец не входит в требуемую матрицу y размером 3N x T.
        gds::GDSNetwork grid(n * horizon, std::vector<int>(n * horizon, 4));
        std::vector<int> assignment(n * horizon);
        std::vector<double> biases(4 * n * horizon, 0.);
        for (int s = 0; s < n; ++s) for (int t = 0; t < horizon; ++t) {
            int p = pos(s, t), code = plan[p];
            assignment[p] = code < 2 ? code : jobs[code - 2].kind;
            if (!sats[s].available[t]) for (int a = 1; a < 4; ++a) biases[4*p+a] = -4.;
            if (!sats[s].d[t]) biases[4*p+2] = -4.;
            if (!sats[s].r[t]) biases[4*p+3] = -4.;
        }
        grid.set_unary_biases(biases);
        grid.set_assignment(assignment);
        grid.save_weights(path);
    }
};

PYBIND11_MODULE(_native, m) {
    m.doc() = "Compiled block GDS guards with full-horizon resource simulation";
    py::class_<CompiledNetwork>(m, "CompiledNetwork")
        .def(py::init<py::dict, py::dict, py::list, int, py::dict>())
        .def("decode", &CompiledNetwork::decode)
        .def("inspect", &CompiledNetwork::inspect)
        .def("description", &CompiledNetwork::description)
        .def("save_network", &CompiledNetwork::save_network);
}
