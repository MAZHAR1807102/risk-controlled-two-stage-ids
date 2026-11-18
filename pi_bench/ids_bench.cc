// Standalone two-stage IDS benchmark for real gateway hardware (Raspberry Pi / Orange Pi).
//
// Same detection path as ns3/IoT_TwoStage_IDS_v2.cc (inline scaler -> Stage 1 Gatekeeper
// -> Stage 2), without the ns-3 simulator: flows are replayed back to back from a labelled
// traffic file. The per-packet CSV has the same columns as the ns-3 version, so the same
// analysis scripts apply.
//
// Build:  see build.sh      Run:  see run_all.sh
// Usage:
//   ./ids_bench --data=traffic.csv --params=ids_params.h --l1=gk.json --l2=brain.json
//               --mode=two-stage|single-stage|adaptive --nthread1=1 --nthread2=0
//               [--rows=N] [--warmup=200] [--tau=x] [--out=packets.csv]

#include <xgboost/c_api.h>

#include <algorithm>
#include <chrono>
#include <sys/resource.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace
{

// Feature count comes from the params file (46 for CICIoT2023, 45 for CICIoMT2024);
// the class count comes from the Stage 2 model output. Class 0 is Benign in both datasets.
constexpr int MAX_FEATURES = 128;
int N_FEATURES = 0;
constexpr int BENIGN_CLASS = 0;

#define XGB_CHECK(call)                                                                  \
    do                                                                                   \
    {                                                                                    \
        if ((call) != 0)                                                                 \
        {                                                                                \
            std::cerr << "XGBoost error: " << XGBGetLastError() << std::endl;            \
            std::exit(1);                                                                \
        }                                                                                \
    } while (0)

[[noreturn]] void
Fail(const std::string& msg)
{
    std::cerr << "error: " << msg << std::endl;
    std::exit(1);
}

using Clock = std::chrono::steady_clock;

int64_t
ElapsedNs(Clock::time_point a, Clock::time_point b)
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count();
}

struct Flow
{
    float features[MAX_FEATURES];
    int label;
};

struct Record
{
    int label;
    float pAttack;
    int predicted;
    bool inspected;
    int64_t scaleNs, stage1Ns, stage2Ns;
    bool direct;
};

// ---- Configuration -----------------------------------------------------------------------
float g_means[MAX_FEATURES];
float g_scales[MAX_FEATURES];
float g_tau = 0.5f;
bool g_singleStage = false;
bool g_adaptive = false;

// Adaptive routing state (identical policy to the ns-3 version, Eq. (19) of the paper).
double g_alpha = 0.01, g_hysteresis = 0.2, g_bypassPrior = 0.77;
uint32_t g_minDwell = 200;
bool g_direct = false;
double g_ewmaBypass = -1, g_ewmaBenign = -1, g_ewmaT1 = -1, g_ewmaT2 = -1;
uint32_t g_sinceSwitch = 0, g_switches = 0;

std::vector<float>
ParseArray(const std::string& text, const std::string& name)
{
    auto pos = text.find(name);
    if (pos == std::string::npos)
    {
        Fail("missing " + name + " in params file");
    }
    auto open = text.find('{', pos);
    auto close = text.find('}', open);
    std::stringstream ss(text.substr(open + 1, close - open - 1));
    std::vector<float> out;
    std::string tok;
    while (std::getline(ss, tok, ','))
    {
        out.push_back(std::stof(tok));
    }
    return out;
}

void
LoadParams(const std::string& path, double tauOverride)
{
    std::ifstream f(path);
    if (!f)
    {
        Fail("cannot open params file " + path);
    }
    std::stringstream buf;
    buf << f.rdbuf();
    std::string text = buf.str();
    auto means = ParseArray(text, "SCALER_MEANS");
    auto scales = ParseArray(text, "SCALER_SCALES");
    if (means.empty() || means.size() != scales.size() || means.size() > MAX_FEATURES)
    {
        Fail("params file must hold the same number of means and scales");
    }
    N_FEATURES = static_cast<int>(means.size());
    std::copy(means.begin(), means.end(), g_means);
    std::copy(scales.begin(), scales.end(), g_scales);
    if (tauOverride >= 0)
    {
        g_tau = static_cast<float>(tauOverride);
        return;
    }
    auto pos = text.find("GATEKEEPER_TAU");
    if (pos == std::string::npos)
    {
        Fail("no GATEKEEPER_TAU in params; pass --tau");
    }
    g_tau = std::stof(text.substr(text.find('=', pos) + 1));
}

std::vector<Flow>
LoadFlows(const std::string& path, uint32_t maxRows)
{
    std::ifstream file(path);
    if (!file)
    {
        Fail("cannot open traffic file " + path);
    }
    std::vector<Flow> flows;
    std::string line;
    std::getline(file, line); // header
    while (std::getline(file, line) && flows.size() < maxRows)
    {
        std::stringstream ss(line);
        std::string value;
        Flow flow{};
        int col = 0;
        while (std::getline(ss, value, ',') && col <= N_FEATURES)
        {
            if (col < N_FEATURES)
            {
                flow.features[col] = std::stof(value);
            }
            else
            {
                flow.label = std::stoi(value);
            }
            ++col;
        }
        if (col <= N_FEATURES)
        {
            Fail("row " + std::to_string(flows.size()) + " has no label column");
        }
        flows.push_back(flow);
    }
    return flows;
}

BoosterHandle
LoadBooster(const std::string& path, const std::string& nthread)
{
    BoosterHandle h;
    XGB_CHECK(XGBoosterCreate(nullptr, 0, &h));
    XGB_CHECK(XGBoosterLoadModel(h, path.c_str()));
    if (nthread != "0")
    {
        XGB_CHECK(XGBoosterSetParam(h, "nthread", nthread.c_str()));
    }
    return h;
}

// Returns the output row; *outLen receives its length (1 for Stage 1, classes for Stage 2).
const float*
Predict(BoosterHandle booster, const float* row, bst_ulong* outLen = nullptr)
{
    static const char* kConfig = "{\"type\": 0, \"training\": false, \"iteration_begin\": 0, "
                                 "\"iteration_end\": 0, \"strict_shape\": false, "
                                 "\"missing\": NaN}";
    char iface[160];
    std::snprintf(iface,
                  sizeof(iface),
                  "{\"data\": [%llu, true], \"shape\": [1, %d], \"typestr\": \"<f4\", "
                  "\"version\": 3}",
                  static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(row)),
                  N_FEATURES);
    const bst_ulong* shape;
    bst_ulong dim;
    const float* result;
    XGB_CHECK(XGBoosterPredictFromDense(booster, iface, kConfig, nullptr, &shape, &dim, &result));
    if (outLen)
    {
        *outLen = shape[dim - 1];
    }
    return result;
}

int
ArgMax(const float* p, bst_ulong n)
{
    return static_cast<int>(std::max_element(p, p + n) - p);
}

void
Ewma(double& state, double x)
{
    state = state < 0 ? x : g_alpha * x + (1 - g_alpha) * state;
}

void
UpdateAdaptive(const Record& rec)
{
    if (!rec.direct)
    {
        Ewma(g_ewmaBypass, rec.inspected ? 0.0 : 1.0);
        Ewma(g_ewmaT1, rec.stage1Ns / 1e3);
    }
    if (rec.inspected)
    {
        Ewma(g_ewmaT2, rec.stage2Ns / 1e3);
    }
    Ewma(g_ewmaBenign, (!rec.inspected || rec.predicted == BENIGN_CLASS) ? 1.0 : 0.0);
    if (++g_sinceSwitch < g_minDwell || g_ewmaT1 < 0 || g_ewmaT2 < 0)
    {
        return;
    }
    double bypass = g_direct ? g_ewmaBenign * g_bypassPrior : g_ewmaBypass;
    double saved = bypass * g_ewmaT2;
    if ((!g_direct && saved < g_ewmaT1 * (1 - g_hysteresis)) ||
        (g_direct && saved > g_ewmaT1 * (1 + g_hysteresis)))
    {
        g_direct = !g_direct;
        g_sinceSwitch = 0;
        ++g_switches;
    }
}

double
Percentile(std::vector<int64_t> v, double q)
{
    if (v.empty())
    {
        return 0;
    }
    std::sort(v.begin(), v.end());
    return v[static_cast<size_t>(q * (v.size() - 1))] / 1000.0;
}

// User + system CPU time of the whole process (all threads), in ms.
double
ProcessCpuMs()
{
    rusage u{};
    getrusage(RUSAGE_SELF, &u);
    return (u.ru_utime.tv_sec + u.ru_stime.tv_sec) * 1e3 + (u.ru_utime.tv_usec + u.ru_stime.tv_usec) / 1e3;
}

// Reads "--key=value" arguments into a map.
std::map<std::string, std::string>
ParseArgs(int argc, char* argv[])
{
    std::map<std::string, std::string> args;
    for (int i = 1; i < argc; ++i)
    {
        std::string a = argv[i];
        auto eq = a.find('=');
        if (a.rfind("--", 0) != 0 || eq == std::string::npos)
        {
            Fail("bad argument " + a + " (expected --key=value)");
        }
        args[a.substr(2, eq - 2)] = a.substr(eq + 1);
    }
    return args;
}

std::string
Get(const std::map<std::string, std::string>& args, const std::string& key, const std::string& def)
{
    auto it = args.find(key);
    return it == args.end() ? def : it->second;
}

} // namespace

int
main(int argc, char* argv[])
{
    auto args = ParseArgs(argc, argv);
    const std::string mode = Get(args, "mode", "two-stage");
    if (mode != "two-stage" && mode != "single-stage" && mode != "adaptive")
    {
        Fail("bad --mode");
    }
    g_singleStage = (mode == "single-stage");
    g_adaptive = (mode == "adaptive");
    const std::string nthread = Get(args, "nthread", "1");
    const std::string nthread1 = Get(args, "nthread1", nthread);
    const std::string nthread2 = Get(args, "nthread2", nthread);
    const uint32_t warmup = std::stoul(Get(args, "warmup", "200"));
    const uint32_t rows = std::stoul(Get(args, "rows", "1000000000"));
    const std::string out = Get(args, "out", "");

    LoadParams(Get(args, "params", "ids_params.h"), std::stod(Get(args, "tau", "-1")));
    auto flows = LoadFlows(Get(args, "data", "traffic.csv"), rows);
    BoosterHandle stage1 = LoadBooster(Get(args, "l1", "layer1_gatekeeper.json"), nthread1);
    BoosterHandle stage2 = LoadBooster(Get(args, "l2", "layer2_multiclass.json"), nthread2);
    std::cout << "Loaded " << flows.size() << " flows; mode=" << mode << " tau=" << g_tau
              << " nthread1=" << nthread1 << " nthread2=" << nthread2 << std::endl;

    std::vector<Record> recs;
    recs.reserve(flows.size());
    // Process CPU time (user + system, all threads) and wall time of the whole inference loop.
    // The per-flow timers below measure elapsed (wall-clock) time; these two totals let us
    // report real CPU consumption next to it.
    const double cpu0 = ProcessCpuMs();
    const auto wall0 = Clock::now();
    for (const Flow& flow : flows)
    {
        Record rec{flow.label, -1.0f, BENIGN_CLASS, false, 0, 0, 0, false};
        auto t0 = Clock::now();
        float scaled[MAX_FEATURES];
        for (int i = 0; i < N_FEATURES; ++i)
        {
            scaled[i] = (flow.features[i] - g_means[i]) / g_scales[i];
        }
        rec.scaleNs = ElapsedNs(t0, Clock::now());

        bool inspect = true;
        rec.direct = g_adaptive && g_direct;
        if (!g_singleStage && !rec.direct)
        {
            auto s = Clock::now();
            rec.pAttack = Predict(stage1, scaled)[0];
            rec.stage1Ns = ElapsedNs(s, Clock::now());
            inspect = rec.pAttack >= g_tau;
        }
        if (inspect)
        {
            auto s = Clock::now();
            bst_ulong n = 0;
            const float* probs = Predict(stage2, scaled, &n);
            rec.predicted = ArgMax(probs, n);
            rec.stage2Ns = ElapsedNs(s, Clock::now());
            rec.inspected = true;
        }
        if (g_adaptive)
        {
            UpdateAdaptive(rec);
        }
        recs.push_back(rec);
    }

    const double loopCpuMs = ProcessCpuMs() - cpu0;
    const double loopWallMs = ElapsedNs(wall0, Clock::now()) / 1e6;

    // Summary (warm-up excluded).
    std::vector<int64_t> total;
    uint64_t attacks = 0, missed = 0, inspected = 0, correct = 0, direct = 0;
    double sumMs = 0;
    for (size_t i = std::min<size_t>(warmup, recs.size()); i < recs.size(); ++i)
    {
        const Record& r = recs[i];
        int64_t t = r.scaleNs + r.stage1Ns + r.stage2Ns;
        total.push_back(t);
        sumMs += t / 1e6;
        inspected += r.inspected;
        correct += (r.predicted == r.label);
        direct += r.direct;
        if (r.label != BENIGN_CLASS)
        {
            ++attacks;
            missed += !r.inspected;
        }
    }
    std::cout << "packets measured: " << total.size() << "\n"
              << "inspected by Stage 2: " << 100.0 * inspected / total.size() << "%\n"
              << "attacks missed by Gatekeeper: " << missed << "/" << attacks << "\n"
              << "accuracy: " << 100.0 * correct / total.size() << "%\n"
              << "summed per-flow inference time (wall clock): " << sumMs << " ms\n"
              << "loop wall time (all flows): " << loopWallMs << " ms\n"
              << "loop process CPU time, user+sys (all flows): " << loopCpuMs << " ms\n"
              << "per-packet total us  p50=" << Percentile(total, 0.5)
              << " p95=" << Percentile(total, 0.95) << " p99=" << Percentile(total, 0.99) << "\n";
    if (g_adaptive)
    {
        std::cout << "adaptive: " << g_switches << " switches, "
                  << 100.0 * direct / total.size() << "% routed directly\n";
    }

    if (!out.empty())
    {
        std::ofstream f(out);
        f << "idx,label,p_attack,predicted,inspected,scale_ns,stage1_ns,stage2_ns,direct\n";
        for (size_t i = 0; i < recs.size(); ++i)
        {
            const Record& r = recs[i];
            f << i << ',' << r.label << ',' << r.pAttack << ',' << r.predicted << ','
              << r.inspected << ',' << r.scaleNs << ',' << r.stage1Ns << ',' << r.stage2Ns << ','
              << r.direct << '\n';
        }
    }
    XGBoosterFree(stage1);
    XGBoosterFree(stage2);
    return 0;
}
