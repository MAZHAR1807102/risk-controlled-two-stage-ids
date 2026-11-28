// Flow-record gateway under flooding: packets are forwarded at once, the IDS checks flow records.
//
// This replaces the "one packet = one inspection request" abstraction of IoT_IDS_Overload.cc.
// The gateway has one CPU core that does two kinds of work:
//   1. Data plane (highest priority, per packet): firewall lookup, flow-meter update, forwarding.
//      Packets of the authenticated robot are served before other packets (strict priority,
//      non-preemptive). The cost of every packet is the measured wall-clock time of the real
//      C++ meter update and firewall lookup below, plus an assumed network-stack cost per packet
//      (--fwdNs forwarded, --dropNs dropped) and --authNs for robot packets.
//   2. IDS (lower priority, preempted by the data plane, resumes where it stopped): one check per
//      flow record. The service time is the measured inference time of a labelled record (one thread).
// Flow meter: one flow per source. As in the CIC CSV files, a record is a window of W packets of a
// flow: W = 100 for flood traffic, W = 10 otherwise (--wFlood, --wOther). With --activeTimeout=t, a
// window that has run for t seconds is also closed (records with fewer packets; the models were
// trained on full windows, so this is a sensitivity run only).
// Mitigation: once a source has at least --detectK verdicts and at least --detectRatio of them say
// attack, a block rule is installed --mitigationMs
// later; from then on, the firewall drops the source's packets. The robot is never blocked.
// Security delay of an attack flow (one bot): D = formation + queue + inference + mitigation, from the
// flow's first packet to its block rule; the attack packets forwarded before the rule are "leaked".
//
// Output: one summary line appended to --summary; per-bot detail with --bots=<file>.

#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/internet-module.h"
#include "ns3/network-module.h"
#include "ns3/point-to-point-module.h"

#include <xgboost/c_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <deque>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("IoT_IDS_FlowMeter");

namespace
{

constexpr int N_FEATURES = 46;
constexpr int BENIGN_CLASS = 0;
enum SenderClass : uint8_t
{
    ROBOT = 0,
    DEVICE = 1,
    ATTACKER = 2
};
const char* kClassName[] = {"robot", "device", "attacker"};
const uint32_t ROBOT_SRC = 1;
double g_t1Scale = 1.0, g_t2Scale = 1.0; // per-stage time multipliers (see Inspect)

#define XGB_CHECK(call)                                                                  \
    do                                                                                   \
    {                                                                                    \
        if ((call) != 0)                                                                 \
        {                                                                                \
            NS_FATAL_ERROR("XGBoost error: " << XGBGetLastError());                      \
        }                                                                                \
    } while (0)

// ---- Packet tag: source id and true sender class -------------------------------------------
class IdsTag : public Tag
{
  public:
    static TypeId GetTypeId()
    {
        static TypeId tid = TypeId("ns3::IdsTagFm").SetParent<Tag>().AddConstructor<IdsTag>();
        return tid;
    }

    TypeId GetInstanceTypeId() const override
    {
        return GetTypeId();
    }

    uint32_t GetSerializedSize() const override
    {
        return 6;
    }

    void Serialize(TagBuffer i) const override
    {
        i.WriteU32(src);
        i.WriteU8(cls);
        i.WriteU8(forged);
    }

    void Deserialize(TagBuffer i) override
    {
        src = i.ReadU32();
        cls = i.ReadU8();
        forged = i.ReadU8();
    }

    void Print(std::ostream& os) const override
    {
        os << src << "," << int(cls);
    }

    uint32_t src = 0;
    uint8_t cls = 0;
    uint8_t forged = 0; // claims to be the robot, but its HMAC check fails
};

// ---- Sender: constant-rate or Poisson UDP packets, each from one of nSrc sources ----------
class TaggedSender : public Application
{
  public:
    void Setup(Address peer, double rate, bool poisson, uint8_t cls, uint32_t srcBase, uint32_t nSrc,
               uint32_t size, bool forged = false)
    {
        m_forged = forged;
        m_peer = peer;
        m_rate = rate;
        m_poisson = poisson;
        m_cls = cls;
        m_srcBase = srcBase;
        m_nSrc = nSrc;
        m_size = size;
    }

  private:
    void StartApplication() override
    {
        if (m_rate <= 0)
        {
            return;
        }
        m_sock = Socket::CreateSocket(GetNode(), UdpSocketFactory::GetTypeId());
        m_sock->Bind();
        m_sock->Connect(m_peer);
        m_exp = CreateObject<ExponentialRandomVariable>();
        m_exp->SetAttribute("Mean", DoubleValue(1.0 / m_rate));
        m_uni = CreateObject<UniformRandomVariable>();
        SendOne();
    }

    void StopApplication() override
    {
        Simulator::Cancel(m_event);
        if (m_sock)
        {
            m_sock->Close();
        }
    }

    void SendOne()
    {
        Ptr<Packet> p = Create<Packet>(m_size);
        IdsTag tag;
        tag.cls = m_cls;
        tag.forged = m_forged;
        tag.src = m_srcBase + (m_nSrc > 1 ? m_uni->GetInteger(0, m_nSrc - 1) : 0);
        p->AddPacketTag(tag);
        m_sock->Send(p);
        double gap = m_poisson ? m_exp->GetValue() : 1.0 / m_rate;
        m_event = Simulator::Schedule(Seconds(gap), &TaggedSender::SendOne, this);
    }

    Address m_peer;
    double m_rate = 0;
    bool m_poisson = false;
    uint8_t m_cls = 0;
    bool m_forged = false;
    uint32_t m_srcBase = 0, m_nSrc = 1, m_size = 512;
    Ptr<Socket> m_sock;
    EventId m_event;
    Ptr<ExponentialRandomVariable> m_exp;
    Ptr<UniformRandomVariable> m_uni;
};

// ---- IDS models ------------------------------------------------------------------------------
float g_means[N_FEATURES], g_scales[N_FEATURES], g_tau = 0.5f;
BoosterHandle g_stage1, g_stage2;
bool g_singleStage = false;
std::vector<std::vector<float>> g_benignPool, g_floodPool;
size_t g_benignNext = 0, g_floodNext = 0;

std::vector<float>
ParseArray(const std::string& text, const std::string& name)
{
    auto pos = text.find(name);
    NS_ABORT_MSG_IF(pos == std::string::npos, "missing " << name);
    auto open = text.find('{', pos), close = text.find('}', open);
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
    NS_ABORT_MSG_IF(!f, "cannot open " << path);
    std::stringstream b;
    b << f.rdbuf();
    auto m = ParseArray(b.str(), "SCALER_MEANS"), s = ParseArray(b.str(), "SCALER_SCALES");
    NS_ABORT_MSG_IF(m.size() != N_FEATURES || s.size() != N_FEATURES, "expected 46 features");
    std::copy(m.begin(), m.end(), g_means);
    std::copy(s.begin(), s.end(), g_scales);
    auto pos = b.str().find("GATEKEEPER_TAU");
    g_tau = std::stof(b.str().substr(b.str().find('=', pos) + 1));
    if (tauOverride > 0)
    {
        g_tau = static_cast<float>(tauOverride);
    }
}

template <typename Pred>
void
LoadPool(const std::string& path, Pred keep, std::vector<std::vector<float>>& pool)
{
    std::ifstream f(path);
    NS_ABORT_MSG_IF(!f, "cannot open " << path);
    std::string line;
    std::getline(f, line);
    while (std::getline(f, line))
    {
        std::stringstream ss(line);
        std::string v;
        std::vector<float> row;
        for (int c = 0; c <= N_FEATURES && std::getline(ss, v, ','); ++c)
        {
            row.push_back(std::stof(v));
        }
        if (row.size() == N_FEATURES + 1 && keep(static_cast<int>(row[N_FEATURES])))
        {
            row.pop_back();
            pool.push_back(row);
        }
    }
    NS_ABORT_MSG_IF(pool.empty(), "empty pool from " << path);
}

BoosterHandle
LoadBooster(const std::string& path)
{
    BoosterHandle h;
    XGB_CHECK(XGBoosterCreate(nullptr, 0, &h));
    XGB_CHECK(XGBoosterLoadModel(h, path.c_str()));
    XGB_CHECK(XGBoosterSetParam(h, "nthread", "1"));
    return h;
}

const float*
Predict(BoosterHandle b, const float* row, bst_ulong* n)
{
    static const char* cfg = "{\"type\": 0, \"training\": false, \"iteration_begin\": 0, "
                             "\"iteration_end\": 0, \"strict_shape\": false, \"missing\": NaN}";
    char iface[160];
    std::snprintf(iface, sizeof(iface),
                  "{\"data\": [%llu, true], \"shape\": [1, %d], \"typestr\": \"<f4\", \"version\": 3}",
                  static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(row)), N_FEATURES);
    const bst_ulong* shape;
    bst_ulong dim;
    const float* out;
    XGB_CHECK(XGBoosterPredictFromDense(b, iface, cfg, nullptr, &shape, &dim, &out));
    *n = shape[dim - 1];
    return out;
}

int
Inspect(const std::vector<float>& rec, int64_t* ns)
{
    auto t0 = std::chrono::steady_clock::now();
    float x[N_FEATURES];
    for (int i = 0; i < N_FEATURES; ++i)
    {
        x[i] = (rec[i] - g_means[i]) / g_scales[i];
    }
    bst_ulong n;
    int verdict = BENIGN_CLASS;
    bool inspect = true;
    if (!g_singleStage)
    {
        inspect = Predict(g_stage1, x, &n)[0] >= g_tau;
    }
    auto t1 = std::chrono::steady_clock::now();
    if (inspect)
    {
        const float* p = Predict(g_stage2, x, &n);
        verdict = static_cast<int>(std::max_element(p, p + n) - p);
    }
    auto t2 = std::chrono::steady_clock::now();
    // --t1Scale / --t2Scale emulate a faster or slower engine per stage (e.g. ONNX Runtime).
    *ns = static_cast<int64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() * g_t1Scale +
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() * g_t2Scale);
    return verdict;
}

// ---- Flow meter (real per-packet work, timed) ---------------------------------------------
// Running statistics of the kind the CIC features are built from: packet count, size and
// inter-arrival mean/variance (Welford), min/max, protocol and flag counters.
struct FlowState
{
    uint32_t n = 0;            // packets in the current window
    double tWindowStart = 0;   // first packet of the current window (s)
    double tFlowStart = -1;    // first packet of the flow (s)
    double tLast = 0;
    double sizeMean = 0, sizeM2 = 0, iatMean = 0, iatM2 = 0, sizeMin = 1e9, sizeMax = 0, tot = 0;
    uint32_t flags[8] = {0};
    uint32_t proto[6] = {0};
    uint8_t cls = 0;
    uint64_t arrived = 0, forwarded = 0;   // whole flow
    uint32_t verdicts = 0, attackVerdicts = 0;
    bool blocked = false;
    // timing of the record that led to the block
    double tBlockRecEmit = -1, tBlockRecStart = -1, tBlockRecEnd = -1, tBlock = -1;
};

std::unordered_map<uint32_t, FlowState> g_flows;
std::unordered_set<uint32_t> g_blockTable;
volatile double g_sink = 0; // keeps the meter arithmetic from being optimised away

void
MeterUpdate(FlowState& f, double now, uint32_t size, uint32_t salt)
{
    double iat = f.n > 0 ? now - f.tLast : 0.0;
    f.tLast = now;
    f.n++;
    double d = size - f.sizeMean;
    f.sizeMean += d / f.n;
    f.sizeM2 += d * (size - f.sizeMean);
    double e = iat - f.iatMean;
    f.iatMean += e / f.n;
    f.iatM2 += e * (iat - f.iatMean);
    f.sizeMin = std::min(f.sizeMin, double(size));
    f.sizeMax = std::max(f.sizeMax, double(size));
    f.tot += size;
    f.flags[salt & 7]++;
    f.proto[salt % 6]++;
    g_sink = g_sink + f.sizeM2 * 1e-30;
}

// ---- Gateway: one core, data plane preempts the IDS ----------------------------------------
struct Pkt
{
    Time arrival;
    uint32_t src;
    uint8_t cls;
    bool forged; // claims priority; still waits for its HMAC check
};

struct Record
{
    uint32_t src;
    uint8_t cls;
    double tEmit;
};

uint32_t g_wFlood = 100, g_wOther = 10, g_detectK = 3;
double g_detectRatio = 0.8;
double g_activeTimeout = 0, g_mitigationS = 0.001, g_authNs = 500, g_timeScale = 1.0;
double g_fwdNs = 5000, g_dropNs = 1000; // network-stack cost per forwarded / dropped packet (assumed)
uint32_t g_capDpHigh = 1000, g_capDpLow = 10000, g_capRecHigh = 100, g_capRec = 1000;
bool g_drr = true, g_dpPriority = true, g_sharedCore = true;
// --authInQueue=1: the HMAC check runs in software inside the data plane, so every packet that claims
// the robot's identity enters the priority queue and is demoted only after its check fails.
// --authInQueue=0: the priority class is known on arrival (dedicated interface or line-rate check).
bool g_authInQueue = false;
uint64_t g_authFailed = 0;
uint32_t g_devBase = 100, g_nDevices = 50;

std::deque<Pkt> g_dpHigh, g_dpLow;
std::deque<Record> g_recHigh, g_recDev, g_recLow;
bool g_drrTurnDev = true;
bool g_dpBusy = false;
// IDS job state (one record at a time)
bool g_idsActive = false, g_idsRunning = false;
double g_idsRemainingNs = 0;
Time g_idsEnd;
EventId g_idsEvent;
Record g_idsRec;
int g_idsVerdict = 0;
double g_idsStart = 0;

double g_dpBusyNs = 0, g_idsBusyNs = 0;
uint64_t g_recEmitted[3] = {0}, g_recInspected[3] = {0}, g_recDropped[3] = {0}, g_recTimeout = 0;
uint64_t g_robotFalseAlarms = 0;
std::vector<double> g_dpNs, g_infNs, g_recWaitMs;

struct Stats
{
    uint64_t arrived = 0, forwarded = 0, dropDp = 0, dropBlocked = 0;
    std::vector<double> delayMs;
};

Stats g_stats[3];

void StartDp();
void StartIds();

void
EmitRecord(uint32_t src, FlowState& f, double now)
{
    Record r{src, f.cls, now};
    g_recEmitted[f.cls]++;
    f.n = 0;
    f.sizeMean = f.sizeM2 = f.iatMean = f.iatM2 = f.tot = 0;
    f.sizeMin = 1e9;
    f.sizeMax = 0;
    bool registered = src >= g_devBase && src < g_devBase + g_nDevices;
    std::deque<Record>& q = f.cls == ROBOT ? g_recHigh : ((g_drr && registered) ? g_recDev : g_recLow);
    if (q.size() >= (f.cls == ROBOT ? g_capRecHigh : g_capRec))
    {
        g_recDropped[f.cls]++;
        return;
    }
    q.push_back(r);
    StartIds();
}

void
InstallBlock(uint32_t src)
{
    g_blockTable.insert(src);
    FlowState& f = g_flows[src];
    f.blocked = true;
    f.tBlock = Simulator::Now().GetSeconds();
}

void
IdsDone()
{
    g_idsActive = g_idsRunning = false;
    FlowState& f = g_flows[g_idsRec.src];
    g_recInspected[g_idsRec.cls]++;
    f.verdicts++;
    bool attack = g_idsVerdict != BENIGN_CLASS;
    if (attack && g_idsRec.cls == ROBOT)
    {
        g_robotFalseAlarms++;
    }
    if (attack && g_idsRec.src != ROBOT_SRC)
    {
        f.attackVerdicts++;
        if (f.verdicts >= g_detectK && f.attackVerdicts >= g_detectRatio * f.verdicts && !f.blocked &&
            f.tBlockRecEmit < 0)
        {
            f.tBlockRecEmit = g_idsRec.tEmit;
            f.tBlockRecStart = g_idsStart;
            f.tBlockRecEnd = Simulator::Now().GetSeconds();
            Simulator::Schedule(Seconds(g_mitigationS), &InstallBlock, g_idsRec.src);
        }
    }
    StartIds();
}

void
StartIds()
{
    if (g_idsActive || (g_sharedCore && g_dpBusy))
    {
        return;
    }
    std::deque<Record>* q = nullptr;
    if (!g_recHigh.empty())
    {
        q = &g_recHigh;
    }
    else if (!g_recDev.empty() || !g_recLow.empty())
    {
        bool dev = !g_recDev.empty() && (g_drrTurnDev || g_recLow.empty());
        q = dev ? &g_recDev : &g_recLow;
        g_drrTurnDev = !dev;
    }
    if (!q)
    {
        return;
    }
    g_idsRec = q->front();
    q->pop_front();
    g_recWaitMs.push_back((Simulator::Now().GetSeconds() - g_idsRec.tEmit) * 1e3);
    const std::vector<float>& rec = g_idsRec.cls == ATTACKER
                                        ? g_floodPool[g_floodNext++ % g_floodPool.size()]
                                        : g_benignPool[g_benignNext++ % g_benignPool.size()];
    int64_t ns;
    g_idsVerdict = Inspect(rec, &ns);
    double service = ns * g_timeScale;
    g_infNs.push_back(service);
    g_idsBusyNs += service;
    g_idsActive = g_idsRunning = true;
    g_idsStart = Simulator::Now().GetSeconds();
    g_idsEnd = Simulator::Now() + NanoSeconds(static_cast<int64_t>(service));
    g_idsEvent = Simulator::Schedule(NanoSeconds(static_cast<int64_t>(service)), &IdsDone);
}

void
PreemptIds()
{
    if (g_sharedCore && g_idsRunning)
    {
        Simulator::Cancel(g_idsEvent);
        g_idsRemainingNs = (g_idsEnd - Simulator::Now()).GetNanoSeconds();
        g_idsRunning = false;
    }
}

void
ResumeIds()
{
    if (g_idsActive && !g_idsRunning)
    {
        g_idsRunning = true;
        g_idsEnd = Simulator::Now() + NanoSeconds(static_cast<int64_t>(g_idsRemainingNs));
        g_idsEvent = Simulator::Schedule(NanoSeconds(static_cast<int64_t>(g_idsRemainingNs)), &IdsDone);
    }
    else
    {
        StartIds();
    }
}

void
DpDone(Pkt p, bool forwarded)
{
    Stats& st = g_stats[p.cls];
    if (forwarded)
    {
        st.forwarded++;
        st.delayMs.push_back((Simulator::Now() - p.arrival).GetSeconds() * 1e3);
        g_flows[p.src].forwarded++;
    }
    else
    {
        st.dropBlocked++;
    }
    g_dpBusy = false;
    if (!g_dpHigh.empty() || !g_dpLow.empty())
    {
        StartDp();
    }
    else
    {
        ResumeIds();
    }
}

void
DemoteDone(Pkt p)
{
    p.forged = false;
    if (g_dpLow.size() >= g_capDpLow)
    {
        g_stats[p.cls].dropDp++;
    }
    else
    {
        g_dpLow.push_back(p);
    }
    g_dpBusy = false;
    StartDp();
}

void
StartDp()
{
    if (g_dpBusy || (g_dpHigh.empty() && g_dpLow.empty()))
    {
        return;
    }
    bool fromHigh = !g_dpHigh.empty();
    std::deque<Pkt>& q = fromHigh ? g_dpHigh : g_dpLow;
    Pkt p = q.front();
    q.pop_front();
    PreemptIds();
    g_dpBusy = true;
    if (fromHigh && p.forged)
    {
        // HMAC check fails: the packet loses its priority and joins the normal queue.
        double ans = g_authNs * g_timeScale;
        g_dpNs.push_back(ans);
        g_dpBusyNs += ans;
        g_authFailed++;
        Simulator::Schedule(NanoSeconds(static_cast<int64_t>(ans)), &DemoteDone, p);
        return;
    }
    double now = Simulator::Now().GetSeconds();
    // ---- real per-packet work, timed: firewall lookup, then meter update ----
    auto t0 = std::chrono::steady_clock::now();
    bool blocked = g_blockTable.count(p.src) > 0;
    FlowState* fp = nullptr;
    if (!blocked)
    {
        fp = &g_flows[p.src];
        if (fp->tFlowStart < 0)
        {
            fp->tFlowStart = now;
            fp->cls = p.cls;
        }
        if (fp->n == 0)
        {
            fp->tWindowStart = now;
        }
        MeterUpdate(*fp, now, 512, p.src);
    }
    double ns = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0)
                    .count();
    ns = ns * g_timeScale + (blocked ? g_dropNs : g_fwdNs) + (p.cls == ROBOT ? g_authNs : 0);
    g_dpNs.push_back(ns);
    g_dpBusyNs += ns;
    if (fp)
    {
        uint32_t w = p.cls == ATTACKER ? g_wFlood : g_wOther;
        if (fp->n >= w)
        {
            EmitRecord(p.src, *fp, now);
        }
    }
    Simulator::Schedule(NanoSeconds(static_cast<int64_t>(ns)), &DpDone, p, !blocked);
}

void
GatewayRx(Ptr<const Packet> pkt, const Address&)
{
    IdsTag tag;
    if (!pkt->PeekPacketTag(tag))
    {
        return;
    }
    Stats& st = g_stats[tag.cls];
    st.arrived++;
    g_flows[tag.src].arrived++;
    // authenticated robot channel; forged claims share it only when the check runs in the queue
    bool high = g_dpPriority && (tag.cls == ROBOT || (tag.forged && g_authInQueue));
    std::deque<Pkt>& q = high ? g_dpHigh : g_dpLow;
    if (q.size() >= (high ? g_capDpHigh : g_capDpLow))
    {
        st.dropDp++;
        return;
    }
    q.push_back({Simulator::Now(), tag.src, tag.cls, tag.forged != 0});
    StartDp();
}

// Active timeout: close windows that have been open for longer than g_activeTimeout.
void
SweepTimeouts()
{
    double now = Simulator::Now().GetSeconds();
    for (auto& kv : g_flows)
    {
        FlowState& f = kv.second;
        if (!f.blocked && f.n > 0 && now - f.tWindowStart >= g_activeTimeout)
        {
            g_recTimeout++;
            EmitRecord(kv.first, f, now);
        }
    }
    Simulator::Schedule(Seconds(std::max(0.01, g_activeTimeout / 10)), &SweepTimeouts);
}

double
Pct(std::vector<double> v, double q)
{
    if (v.empty())
    {
        return NAN;
    }
    std::sort(v.begin(), v.end());
    return v[static_cast<size_t>(q * (v.size() - 1))];
}

double
Mean(const std::vector<double>& v)
{
    if (v.empty())
    {
        return NAN;
    }
    double s = 0;
    for (double x : v)
    {
        s += x;
    }
    return s / v.size();
}

} // namespace

NS_OBJECT_ENSURE_REGISTERED(IdsTag);

int
main(int argc, char* argv[])
{
    std::string params = "ids_params.h", l1 = "layer1_gatekeeper.ubj", l2 = "layer2_multiclass.ubj";
    std::string benignCsv = "test_benign90_20000.csv", floodCsv = "test_natural_20000.csv";
    std::string mode = "two-stage", summary = "flowmeter_summary.csv", botsOut, label;
    double robotRate = 100, deviceRate = 300, attackRate = 1000, simTime = 10, floodStart = 2,
           floodStop = 8, tau = 0;
    uint32_t nDevices = 50, nBots = 50, seed = 1;
    double mitigationMs = 1.0;

    CommandLine cmd(__FILE__);
    cmd.AddValue("params", "ids_params.h", params);
    cmd.AddValue("l1", "Stage 1 model", l1);
    cmd.AddValue("l2", "Stage 2 model", l2);
    cmd.AddValue("tau", "Gatekeeper threshold (0 = value in ids_params.h)", tau);
    cmd.AddValue("benign", "Traffic CSV with benign records (class 0)", benignCsv);
    cmd.AddValue("flood", "Traffic CSV with flood records (DDoS/DoS/Mirai)", floodCsv);
    cmd.AddValue("mode", "two-stage | single-stage", mode);
    cmd.AddValue("wFlood", "Packets per record for flood flows", g_wFlood);
    cmd.AddValue("wOther", "Packets per record for other flows", g_wOther);
    cmd.AddValue("activeTimeout", "Close a window after this many seconds (0 = off)", g_activeTimeout);
    cmd.AddValue("detectK", "Verdicts of a source needed before it can be blocked", g_detectK);
    cmd.AddValue("detectRatio", "Share of attack verdicts needed to block a source", g_detectRatio);
    cmd.AddValue("mitigationMs", "Delay to install a block rule after the verdict (ms)", mitigationMs);
    cmd.AddValue("authNs", "Authentication cost per robot packet (ns)", g_authNs);
    cmd.AddValue("fwdNs", "Network-stack cost of forwarding one packet (ns, assumed)", g_fwdNs);
    cmd.AddValue("dropNs", "Cost of dropping one packet at the firewall (ns, assumed)", g_dropNs);
    cmd.AddValue("dpPriority", "Robot packets first in the data plane", g_dpPriority);
    cmd.AddValue("drr", "Fair round robin of records: registered devices vs unknown sources", g_drr);
    cmd.AddValue("sharedCore", "Data plane and IDS share one core (data plane preempts)", g_sharedCore);
    cmd.AddValue("timeScale", "Multiply all measured times (slower board)", g_timeScale);
    cmd.AddValue("robotRate", "Robot packets per second", robotRate);
    cmd.AddValue("deviceRate", "Other benign devices, packets per second (total)", deviceRate);
    cmd.AddValue("nDevices", "Number of other benign devices", nDevices);
    cmd.AddValue("attackRate", "Flood packets per second (total)", attackRate);
    cmd.AddValue("nBots", "Botnet size (one flow per bot)", nBots);
    cmd.AddValue("simTime", "Simulation time (s)", simTime);
    cmd.AddValue("floodStart", "Flood start (s)", floodStart);
    cmd.AddValue("floodStop", "Flood stop (s)", floodStop);
    cmd.AddValue("seed", "Run number for random streams", seed);
    cmd.AddValue("summary", "Summary CSV (appended)", summary);
    cmd.AddValue("bots", "Per-bot detail CSV (empty = none)", botsOut);
    cmd.AddValue("label", "Setup name written to the summary", label);
    cmd.AddValue("capRec", "IDS record queue size, per queue (records)", g_capRec);
    cmd.AddValue("capDpLow", "Data-plane queue size for non-robot packets (packets)", g_capDpLow);
    bool robotPoisson = false;
    cmd.AddValue("robotPoisson", "Robot sends Poisson instead of periodic packets", robotPoisson);
    double forgedRate = 0;
    uint32_t forgedBots = 50;
    cmd.AddValue("forgedRate", "Packets per second that claim to be the robot (fail the HMAC check)", forgedRate);
    cmd.AddValue("forgedBots", "Sources of the forged-robot flood", forgedBots);
    cmd.AddValue("authInQueue", "HMAC check inside the data plane (1) or class known on arrival (0)", g_authInQueue);
    cmd.AddValue("t1Scale", "Multiply the measured scaling + Stage 1 time", g_t1Scale);
    cmd.AddValue("t2Scale", "Multiply the measured Stage 2 time", g_t2Scale);
    cmd.Parse(argc, argv);
    g_mitigationS = mitigationMs / 1e3;
    g_devBase = 100;
    g_nDevices = nDevices;

    RngSeedManager::SetRun(seed);
    g_singleStage = (mode == "single-stage");
    LoadParams(params, tau);
    g_stage1 = LoadBooster(l1);
    g_stage2 = LoadBooster(l2);
    LoadPool(benignCsv, [](int c) { return c == BENIGN_CLASS; }, g_benignPool);
    LoadPool(floodCsv, [](int c) { return c == 2 || c == 3 || c == 4; }, g_floodPool);
    for (int i = 0; i < 300; ++i) // warm-up
    {
        int64_t ns;
        Inspect(g_benignPool[i % g_benignPool.size()], &ns);
        Inspect(g_floodPool[i % g_floodPool.size()], &ns);
    }
    g_flows.reserve(4096);

    NodeContainer senders, gateway;
    senders.Create(3);
    gateway.Create(1);
    InternetStackHelper stack;
    stack.Install(senders);
    stack.Install(gateway);
    PointToPointHelper p2p;
    p2p.SetQueue("ns3::DropTailQueue", "MaxSize", StringValue("100000p"));
    Ipv4AddressHelper addr;
    Ipv4Address gwAddr;
    const char* rates[] = {"100Mbps", "100Mbps", "10Gbps"};
    for (uint32_t i = 0; i < 3; ++i)
    {
        p2p.SetDeviceAttribute("DataRate", StringValue(rates[i]));
        p2p.SetChannelAttribute("Delay", StringValue("1ms"));
        NetDeviceContainer d = p2p.Install(senders.Get(i), gateway.Get(0));
        std::ostringstream net;
        net << "10.1." << i + 1 << ".0";
        addr.SetBase(net.str().c_str(), "255.255.255.0");
        Ipv4InterfaceContainer ifc = addr.Assign(d);
        gwAddr = ifc.GetAddress(1);
    }
    Ipv4GlobalRoutingHelper::PopulateRoutingTables();

    uint16_t port = 9;
    PacketSinkHelper sinkHelper("ns3::UdpSocketFactory", InetSocketAddress(Ipv4Address::GetAny(), port));
    ApplicationContainer sink = sinkHelper.Install(gateway.Get(0));
    sink.Start(Seconds(0));
    sink.Stop(Seconds(simTime + 1));
    sink.Get(0)->GetObject<PacketSink>()->TraceConnectWithoutContext("Rx", MakeCallback(&GatewayRx));

    auto addSender = [&](uint32_t node, double rate, bool poisson, uint8_t cls, uint32_t srcBase,
                         uint32_t nSrc, double start, double stop, bool forged = false) {
        Ptr<TaggedSender> app = CreateObject<TaggedSender>();
        app->Setup(InetSocketAddress(gwAddr, port), rate, poisson, cls, srcBase, nSrc, 512, forged);
        senders.Get(node)->AddApplication(app);
        app->SetStartTime(Seconds(start));
        app->SetStopTime(Seconds(stop));
    };
    addSender(0, robotRate, robotPoisson, ROBOT, ROBOT_SRC, 1, 0.5, simTime);
    addSender(1, deviceRate, true, DEVICE, g_devBase, nDevices, 0.5, simTime);
    addSender(2, attackRate, true, ATTACKER, 100000, nBots, floodStart, floodStop);
    addSender(2, forgedRate, true, ATTACKER, 200000, forgedBots, floodStart, floodStop, true);

    if (g_activeTimeout > 0)
    {
        Simulator::Schedule(Seconds(0.5), &SweepTimeouts);
    }
    Simulator::Stop(Seconds(simTime + 1));
    Simulator::Run();
    Simulator::Destroy();

    // ---- Security delay of every bot flow ----------------------------------------------------
    std::vector<double> dTot, dForm, dQueue, dInf, dMit, leakShare;
    uint64_t bots = 0, botsBlocked = 0, atkArrived = 0, atkForwarded = 0, devBlocked = 0;
    std::ofstream bo;
    if (!botsOut.empty())
    {
        bo.open(botsOut);
        bo << "src,arrived,forwarded,blocked,t_first,d_total_ms,d_formation_ms,d_queue_ms,"
              "d_inference_ms,d_mitigation_ms\n";
    }
    for (auto& kv : g_flows)
    {
        const FlowState& f = kv.second;
        if (f.cls == DEVICE && f.blocked)
        {
            devBlocked++;
        }
        if (f.cls != ATTACKER || f.tFlowStart < 0)
        {
            continue;
        }
        bots++;
        atkArrived += f.arrived;
        atkForwarded += f.forwarded;
        leakShare.push_back(f.arrived ? double(f.forwarded) / f.arrived : 0);
        double tot = NAN, form = NAN, qu = NAN, inf = NAN, mit = NAN;
        if (f.blocked)
        {
            botsBlocked++;
            tot = (f.tBlock - f.tFlowStart) * 1e3;
            form = (f.tBlockRecEmit - f.tFlowStart) * 1e3;
            qu = (f.tBlockRecStart - f.tBlockRecEmit) * 1e3;
            inf = (f.tBlockRecEnd - f.tBlockRecStart) * 1e3;
            mit = (f.tBlock - f.tBlockRecEnd) * 1e3;
            dTot.push_back(tot);
            dForm.push_back(form);
            dQueue.push_back(qu);
            dInf.push_back(inf);
            dMit.push_back(mit);
        }
        if (bo)
        {
            bo << kv.first << "," << f.arrived << "," << f.forwarded << "," << f.blocked << ","
               << f.tFlowStart << "," << tot << "," << form << "," << qu << "," << inf << "," << mit << "\n";
        }
    }

    bool header = !std::ifstream(summary).good();
    std::ofstream out(summary, std::ios::app);
    if (header)
    {
        out << "label,mode,seed,attack_rate,n_bots,w_flood,w_other,active_timeout,detect_k,detect_ratio,mitigation_ms,"
               "shared_core,time_scale,fwd_ns";
        for (const char* c : kClassName)
        {
            out << "," << c << "_arrived," << c << "_forwarded," << c << "_drop_dp," << c << "_drop_blocked,"
                << c << "_p50_ms," << c << "_p99_ms," << c << "_max_ms";
        }
        out << ",dp_busy,ids_busy,dp_ns_mean,dp_ns_p999,dp_ns_max,inf_ms_mean,inf_ms_p99,rec_wait_ms_p50,"
               "rec_wait_ms_p99,rec_wait_ms_max,rec_emitted_attack,rec_inspected_attack,rec_dropped_attack,"
               "rec_dropped_benign,rec_timeout,robot_false_alarms,devices_blocked,bots,bots_blocked,"
               "attack_leak_share,d_total_ms_p50,d_total_ms_p95,d_total_ms_max,d_formation_ms_mean,"
               "d_queue_ms_mean,d_inference_ms_mean,d_mitigation_ms_mean,forged_rate,auth_in_queue,auth_failed,"
               "t1_scale,t2_scale\n";
    }
    out << label << "," << mode << "," << seed << "," << attackRate << "," << nBots << "," << g_wFlood << ","
        << g_wOther << "," << g_activeTimeout << "," << g_detectK << "," << g_detectRatio << "," << mitigationMs << "," << g_sharedCore
        << "," << g_timeScale << "," << g_fwdNs;
    for (const Stats& st : g_stats)
    {
        out << "," << st.arrived << "," << st.forwarded << "," << st.dropDp << "," << st.dropBlocked << ","
            << Pct(st.delayMs, 0.5) << "," << Pct(st.delayMs, 0.99) << "," << Pct(st.delayMs, 1.0);
    }
    double busyDen = simTime * 1e9;
    out << "," << g_dpBusyNs / busyDen << "," << g_idsBusyNs / busyDen << "," << Mean(g_dpNs) << ","
        << Pct(g_dpNs, 0.999) << "," << Pct(g_dpNs, 1.0) << "," << Mean(g_infNs) / 1e6 << ","
        << Pct(g_infNs, 0.99) / 1e6 << "," << Pct(g_recWaitMs, 0.5) << "," << Pct(g_recWaitMs, 0.99) << ","
        << Pct(g_recWaitMs, 1.0) << "," << g_recEmitted[ATTACKER] << "," << g_recInspected[ATTACKER] << ","
        << g_recDropped[ATTACKER] << "," << g_recDropped[ROBOT] + g_recDropped[DEVICE] << "," << g_recTimeout
        << "," << g_robotFalseAlarms << "," << devBlocked << "," << bots << "," << botsBlocked << ","
        << (atkArrived ? double(atkForwarded) / atkArrived : NAN) << "," << Pct(dTot, 0.5) << ","
        << Pct(dTot, 0.95) << "," << Pct(dTot, 1.0) << "," << Mean(dForm) << "," << Mean(dQueue) << ","
        << Mean(dInf) << "," << Mean(dMit) << "," << forgedRate << "," << g_authInQueue << "," << g_authFailed
        << "," << g_t1Scale << "," << g_t2Scale << "\n";

    const Stats& r = g_stats[ROBOT];
    std::cout << label << " " << mode << " rate=" << attackRate << " bots=" << nBots << " | robot "
              << r.forwarded << "/" << r.arrived << " max " << Pct(r.delayMs, 1.0) << " ms | dp_busy "
              << g_dpBusyNs / busyDen << " ids_busy " << g_idsBusyNs / busyDen << " | bots blocked "
              << botsBlocked << "/" << bots << " D_total p50 " << Pct(dTot, 0.5) << " ms | leak "
              << (atkArrived ? double(atkForwarded) / atkArrived : NAN) << " | rec dropped "
              << g_recDropped[ATTACKER] << std::endl;
    XGBoosterFree(g_stage1);
    XGBoosterFree(g_stage2);
    return 0;
}
