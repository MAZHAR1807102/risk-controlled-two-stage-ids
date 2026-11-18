// Two-stage XGBoost IDS on an ns-3 edge gateway.
//
// Features:
//   * ground truth is read from the label column of the traffic file
//   * scaler + tau are loaded at runtime from an ids_params.h-format file, so any
//     trained model pair can be timed by the same binary
//   * --mode=single-stage actually runs the baseline instead of extrapolating it
//   * per-packet nanosecond timings for scaling, Stage 1 and Stage 2 are written to CSV
//     (so p50/p95/p99 can be reported), after a warm-up period
//   * --api=inplace uses XGBoosterPredictFromDense (no per-packet DMatrix)
//   * single-threaded inference by default (--nthread), overridable per stage
//     (--nthread1 / --nthread2)
//   * --mode=adaptive bypasses the Gatekeeper while it is not paying for itself
//     (e.g. during a flood) and re-enables it when benign traffic returns
//
// Example:
//   ./ns3 run "IoT_TwoStage_IDS_v2 --data=traffic.csv --params=ids_params.h
//              --l1=layer1_gatekeeper.ubj --l2=layer2_multiclass.ubj
//              --mode=two-stage --api=inplace --out=packets.csv"

#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/internet-module.h"
#include "ns3/network-module.h"
#include "ns3/point-to-point-module.h"

#include <xgboost/c_api.h>

#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("IoT_TwoStage_IDS_v2");

namespace
{

constexpr int N_FEATURES = 46;
constexpr int N_CLASSES = 8;
constexpr int BENIGN_CLASS = 0; // classes are alphabetical: Benign, BruteForce, DDoS, ...

#define XGB_CHECK(call)                                                                  \
    do                                                                                   \
    {                                                                                    \
        if ((call) != 0)                                                                 \
        {                                                                                \
            NS_FATAL_ERROR("XGBoost error: " << XGBGetLastError());                      \
        }                                                                                \
    } while (0)

using Clock = std::chrono::steady_clock;

int64_t
ElapsedNs(Clock::time_point a, Clock::time_point b)
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count();
}

struct Flow
{
    float features[N_FEATURES];
    int label;
};

struct PacketRecord
{
    int label;
    float pAttack;  // Stage 1 output (-1 in single-stage mode)
    int predicted;  // final class
    bool inspected; // went through Stage 2
    int64_t scaleNs, stage1Ns, stage2Ns;
    bool direct = false; // adaptive mode: Gatekeeper skipped, sent straight to Stage 2
    float estBenign = -1; // adaptive mode: router's running estimate of the benign share
    float estBypass = -1; // adaptive mode: estimated bypass rate used for the decision
};

// ---- Configuration and state --------------------------------------------------------
float g_means[N_FEATURES];
float g_scales[N_FEATURES];
float g_tau = 0.5f;
bool g_singleStage = false;
bool g_adaptive = false;
bool g_inplace = true;
uint32_t g_warmup = 200;
std::string g_nthread = "1"; // "0" = XGBoost default (all cores)
std::string g_nthread1;      // per-stage overrides; empty = use g_nthread
std::string g_nthread2;

// Adaptive routing. The Gatekeeper is worth running while the Stage 2 time it saves
// (bypass rate x Stage 2 cost) exceeds its own cost. Both sides are tracked online
// with EWMAs; hysteresis and a minimum dwell prevent flapping.
double g_alpha = 0.01;       // EWMA weight (~100-packet memory)
double g_hysteresis = 0.2;   // switch only when the margin exceeds 20% of Stage 1 cost
uint32_t g_minDwell = 200;   // packets between switches
double g_bypassPrior = 0.77; // benign bypass rate of the Gatekeeper (from validation)
bool g_direct = false;
double g_ewmaBypass = -1, g_ewmaBenign = -1, g_ewmaT1 = -1, g_ewmaT2 = -1;
uint32_t g_sinceSwitch = 0;
uint32_t g_switches = 0;

BoosterHandle g_stage1;
BoosterHandle g_stage2;
std::vector<Flow> g_flows;
std::vector<PacketRecord> g_records;
uint32_t g_next = 0;

// Parses "NAME[46] = { a, b, ... }" and "GATEKEEPER_TAU = x" from an ids_params.h file.
std::vector<float>
ParseArray(const std::string& text, const std::string& name)
{
    auto pos = text.find(name);
    NS_ABORT_MSG_IF(pos == std::string::npos, "missing " << name << " in params file");
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
    NS_ABORT_MSG_IF(!f, "cannot open params file " << path);
    std::stringstream buf;
    buf << f.rdbuf();
    std::string text = buf.str();

    auto means = ParseArray(text, "SCALER_MEANS");
    auto scales = ParseArray(text, "SCALER_SCALES");
    NS_ABORT_MSG_IF(means.size() != N_FEATURES || scales.size() != N_FEATURES,
                    "params file must hold 46 means and 46 scales");
    std::copy(means.begin(), means.end(), g_means);
    std::copy(scales.begin(), scales.end(), g_scales);

    if (tauOverride >= 0)
    {
        g_tau = static_cast<float>(tauOverride);
    }
    else
    {
        auto pos = text.find("GATEKEEPER_TAU");
        NS_ABORT_MSG_IF(pos == std::string::npos, "no GATEKEEPER_TAU in params; pass --tau");
        g_tau = std::stof(text.substr(text.find('=', pos) + 1));
    }
}

// CSV: 46 raw features, integer class_id, then any extra columns (ignored).
void
LoadFlows(const std::string& path, uint32_t maxRows)
{
    std::ifstream file(path);
    NS_ABORT_MSG_IF(!file, "cannot open traffic file " << path);
    std::string line;
    std::getline(file, line); // header
    while (std::getline(file, line) && g_flows.size() < maxRows)
    {
        std::stringstream ss(line);
        std::string value;
        Flow flow;
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
        NS_ABORT_MSG_IF(col <= N_FEATURES, "row " << g_flows.size() << " has no label column");
        g_flows.push_back(flow);
    }
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

void
Ewma(double& state, double x)
{
    state = state < 0 ? x : g_alpha * x + (1 - g_alpha) * state;
}

void
UpdateAdaptive(const PacketRecord& rec)
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
    // Share of traffic that ends up classified benign (bypassed or Stage 2 says Benign).
    Ewma(g_ewmaBenign, (!rec.inspected || rec.predicted == BENIGN_CLASS) ? 1.0 : 0.0);

    if (++g_sinceSwitch < g_minDwell || g_ewmaT1 < 0 || g_ewmaT2 < 0)
    {
        return;
    }
    // While direct, the bypass rate cannot be observed, so estimate it from the benign share.
    double bypass = g_direct ? g_ewmaBenign * g_bypassPrior : g_ewmaBypass;
    double saved = bypass * g_ewmaT2; // Stage 2 time the Gatekeeper saves per packet
    if ((!g_direct && saved < g_ewmaT1 * (1 - g_hysteresis)) ||
        (g_direct && saved > g_ewmaT1 * (1 + g_hysteresis)))
    {
        g_direct = !g_direct;
        g_sinceSwitch = 0;
        ++g_switches;
    }
}

// One-row prediction through either the in-place API or a per-packet DMatrix. Returns a pointer into XGBoost-owned memory.
const float*
Predict(BoosterHandle booster, const float* row)
{
    static const char* kConfig = "{\"type\": 0, \"training\": false, \"iteration_begin\": 0, "
                                 "\"iteration_end\": 0, \"strict_shape\": false, "
                                 "\"missing\": NaN}";
    const bst_ulong* shape;
    bst_ulong dim;
    const float* result;
    if (g_inplace)
    {
        char iface[160];
        std::snprintf(iface,
                      sizeof(iface),
                      "{\"data\": [%llu, true], \"shape\": [1, %d], \"typestr\": \"<f4\", "
                      "\"version\": 3}",
                      static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(row)),
                      N_FEATURES);
        XGB_CHECK(XGBoosterPredictFromDense(booster, iface, kConfig, nullptr, &shape, &dim, &result));
    }
    else
    {
        DMatrixHandle dm;
        XGB_CHECK(XGDMatrixCreateFromMat(row, 1, N_FEATURES, NAN, &dm));
        XGB_CHECK(XGBoosterPredictFromDMatrix(booster, dm, kConfig, &shape, &dim, &result));
        XGB_CHECK(XGDMatrixFree(dm));
    }
    return result;
}

int
ArgMax(const float* p)
{
    int best = 0;
    for (int k = 1; k < N_CLASSES; ++k)
    {
        if (p[k] > p[best])
        {
            best = k;
        }
    }
    return best;
}

void
SinkRxInterceptor(Ptr<const Packet> packet, const Address& address)
{
    if (g_next >= g_flows.size())
    {
        return;
    }
    const Flow& flow = g_flows[g_next++];
    PacketRecord rec{flow.label, -1.0f, BENIGN_CLASS, false, 0, 0, 0};

    // Inline StandardScaler.
    auto t0 = Clock::now();
    float scaled[N_FEATURES];
    for (int i = 0; i < N_FEATURES; ++i)
    {
        scaled[i] = (flow.features[i] - g_means[i]) / g_scales[i];
    }
    auto t1 = Clock::now();
    rec.scaleNs = ElapsedNs(t0, t1);

    bool inspect = true;
    rec.direct = g_adaptive && g_direct;
    if (!g_singleStage && !rec.direct)
    {
        auto s = Clock::now();
        rec.pAttack = Predict(g_stage1, scaled)[0];
        rec.stage1Ns = ElapsedNs(s, Clock::now());
        inspect = rec.pAttack >= g_tau;
    }

    if (inspect)
    {
        auto s = Clock::now();
        rec.predicted = ArgMax(Predict(g_stage2, scaled));
        rec.stage2Ns = ElapsedNs(s, Clock::now());
        rec.inspected = true;
    }
    if (g_adaptive)
    {
        UpdateAdaptive(rec);
        rec.estBenign = static_cast<float>(g_ewmaBenign);
        rec.estBypass = static_cast<float>(g_direct ? g_ewmaBenign * g_bypassPrior : g_ewmaBypass);
    }
    g_records.push_back(rec);
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

void
PrintSummary(const std::string& outPath)
{
    // Discard warm-up packets from the statistics (they include cache/allocator warm-up).
    std::vector<PacketRecord> recs(g_records.begin() + std::min<size_t>(g_warmup, g_records.size()),
                                   g_records.end());
    std::vector<int64_t> total, s1, s2;
    uint64_t attacks = 0, missed = 0, inspected = 0, correct = 0, benign = 0, benignBypassed = 0;
    for (const auto& r : recs)
    {
        total.push_back(r.scaleNs + r.stage1Ns + r.stage2Ns);
        if (!g_singleStage && !r.direct)
        {
            s1.push_back(r.stage1Ns);
        }
        if (r.inspected)
        {
            s2.push_back(r.stage2Ns);
            ++inspected;
        }
        correct += (r.predicted == r.label);
        if (r.label == BENIGN_CLASS)
        {
            ++benign;
            benignBypassed += !r.inspected;
        }
        else
        {
            ++attacks;
            missed += !r.inspected;
        }
    }
    double sumMs = 0;
    for (auto t : total)
    {
        sumMs += t / 1e6;
    }

    std::cout << "\n================ IDS v2 RESULTS ================\n"
              << "mode=" << (g_singleStage ? "single-stage" : g_adaptive ? "adaptive" : "two-stage")
              << " api=" << (g_inplace ? "inplace" : "dmatrix") << " nthread1=" << g_nthread1
              << " nthread2=" << g_nthread2 << " tau=" << g_tau << "\n"
              << "packets measured: " << recs.size() << " (warm-up excluded: " << g_warmup << ")\n"
              << "inspected by Stage 2: " << inspected << " ("
              << 100.0 * inspected / recs.size() << "%)\n"
              << "benign bypassed: " << benignBypassed << "/" << benign << "\n"
              << "attacks missed by Gatekeeper: " << missed << "/" << attacks << "\n"
              << "accuracy: " << 100.0 * correct / recs.size() << "%\n"
              << "total CPU time: " << sumMs << " ms\n"
              << "per-packet total us  p50=" << Percentile(total, 0.5)
              << " p95=" << Percentile(total, 0.95) << " p99=" << Percentile(total, 0.99) << "\n";
    if (!s1.empty())
    {
        std::cout << "Stage 1 us           p50=" << Percentile(s1, 0.5)
                  << " p95=" << Percentile(s1, 0.95) << " p99=" << Percentile(s1, 0.99) << "\n";
    }
    if (g_adaptive)
    {
        uint64_t direct = 0;
        for (const auto& r : recs)
        {
            direct += r.direct;
        }
        std::cout << "adaptive: " << g_switches << " switches, " << 100.0 * direct / recs.size()
                  << "% of packets routed directly to Stage 2\n";
    }
    std::cout << "Stage 2 us           p50=" << Percentile(s2, 0.5) << " p95=" << Percentile(s2, 0.95)
              << " p99=" << Percentile(s2, 0.99) << "\n";

    if (!outPath.empty())
    {
        std::ofstream out(outPath);
        out << "idx,label,p_attack,predicted,inspected,scale_ns,stage1_ns,stage2_ns,direct,est_benign,est_bypass\n";
        for (size_t i = 0; i < g_records.size(); ++i)
        {
            const auto& r = g_records[i];
            out << i << ',' << r.label << ',' << r.pAttack << ',' << r.predicted << ','
                << r.inspected << ',' << r.scaleNs << ',' << r.stage1Ns << ',' << r.stage2Ns
                << ',' << r.direct << ',' << r.estBenign << ',' << r.estBypass << '\n';
        }
        std::cout << "per-packet log: " << outPath << "\n";
    }
}

} // namespace

int
main(int argc, char* argv[])
{
    std::string data = "traffic.csv";
    std::string params = "ids_params.h";
    std::string l1 = "layer1_gatekeeper.json";
    std::string l2 = "layer2_multiclass.json";
    std::string mode = "two-stage";
    std::string api = "inplace";
    std::string out;
    uint32_t rows = 20000;
    double tau = -1;

    CommandLine cmd(__FILE__);
    cmd.AddValue("data", "Labelled traffic CSV (46 features + class_id)", data);
    cmd.AddValue("params", "ids_params.h with scaler and GATEKEEPER_TAU", params);
    cmd.AddValue("l1", "Stage 1 (Gatekeeper) model", l1);
    cmd.AddValue("l2", "Stage 2 (multiclass) model", l2);
    cmd.AddValue("mode", "two-stage | single-stage | adaptive", mode);
    cmd.AddValue("api", "inplace | dmatrix", api);
    cmd.AddValue("tau", "Override the Gatekeeper threshold", tau);
    cmd.AddValue("rows", "Max flows to replay", rows);
    cmd.AddValue("warmup", "Packets excluded from latency statistics", g_warmup);
    cmd.AddValue("out", "Per-packet CSV log", out);
    cmd.AddValue("nthread", "XGBoost threads per prediction (0 = all cores)", g_nthread);
    cmd.AddValue("nthread1", "Stage 1 threads (default: --nthread)", g_nthread1);
    cmd.AddValue("nthread2", "Stage 2 threads (default: --nthread)", g_nthread2);
    cmd.AddValue("alpha", "Adaptive mode: EWMA weight", g_alpha);
    cmd.AddValue("hysteresis", "Adaptive mode: relative switching margin", g_hysteresis);
    cmd.AddValue("dwell", "Adaptive mode: minimum packets between switches", g_minDwell);
    cmd.AddValue("bypassPrior", "Adaptive mode: Gatekeeper benign bypass rate", g_bypassPrior);
    cmd.Parse(argc, argv);

    NS_ABORT_MSG_IF(mode != "two-stage" && mode != "single-stage" && mode != "adaptive",
                    "bad --mode");
    NS_ABORT_MSG_IF(api != "inplace" && api != "dmatrix", "bad --api");
    g_singleStage = (mode == "single-stage");
    g_adaptive = (mode == "adaptive");
    g_nthread1 = g_nthread1.empty() ? g_nthread : g_nthread1;
    g_nthread2 = g_nthread2.empty() ? g_nthread : g_nthread2;
    g_inplace = (api == "inplace");

    LoadParams(params, tau);
    LoadFlows(data, rows);
    g_stage1 = LoadBooster(l1, g_nthread1);
    g_stage2 = LoadBooster(l2, g_nthread2);
    g_records.reserve(g_flows.size());
    std::cout << "Loaded " << g_flows.size() << " labelled flows; tau=" << g_tau << "\n";

    // Topology: IoT robot (node 0) -> edge gateway (node 1). One packet = one flow record.
    NodeContainer nodes;
    nodes.Create(2);
    PointToPointHelper p2p;
    p2p.SetDeviceAttribute("DataRate", StringValue("1Gbps"));
    p2p.SetChannelAttribute("Delay", StringValue("1ms"));
    NetDeviceContainer devices = p2p.Install(nodes);
    InternetStackHelper stack;
    stack.Install(nodes);
    Ipv4AddressHelper address;
    address.SetBase("10.1.1.0", "255.255.255.0");
    Ipv4InterfaceContainer interfaces = address.Assign(devices);

    uint16_t port = 9;
    PacketSinkHelper sinkHelper("ns3::UdpSocketFactory",
                                InetSocketAddress(Ipv4Address::GetAny(), port));
    ApplicationContainer sinkApp = sinkHelper.Install(nodes.Get(1));
    sinkApp.Start(Seconds(0.0));
    sinkApp.Stop(Seconds(10.0));
    sinkApp.Get(0)->GetObject<PacketSink>()->TraceConnectWithoutContext(
        "Rx",
        MakeCallback(&SinkRxInterceptor));

    OnOffHelper client("ns3::UdpSocketFactory", InetSocketAddress(interfaces.GetAddress(1), port));
    client.SetAttribute("OnTime", StringValue("ns3::ConstantRandomVariable[Constant=1]"));
    client.SetAttribute("OffTime", StringValue("ns3::ConstantRandomVariable[Constant=0]"));
    client.SetAttribute("DataRate", StringValue("100Mbps"));
    client.SetAttribute("PacketSize", UintegerValue(1024));
    // Enough packets for every flow record: 100 Mbps / 8 kbit per packet ~ 12k packets/s.
    client.SetAttribute("MaxBytes", UintegerValue(static_cast<uint64_t>(g_flows.size()) * 1024));
    ApplicationContainer clientApp = client.Install(nodes.Get(0));
    clientApp.Start(Seconds(1.0));
    clientApp.Stop(Seconds(9.0));

    Simulator::Run();
    Simulator::Destroy();

    NS_ABORT_MSG_IF(g_records.size() < g_flows.size(),
                    "only " << g_records.size() << " of " << g_flows.size()
                            << " flows were delivered; raise the OnOff duration or rate");
    PrintSummary(out);

    XGBoosterFree(g_stage1);
    XGBoosterFree(g_stage2);
    return 0;
}
