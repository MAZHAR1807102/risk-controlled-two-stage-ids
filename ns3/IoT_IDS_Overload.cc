// Gateway overload under flooding: does the IDS itself become the bottleneck?
//
// Topology (ns-3): robot, benign IoT devices and an attacker botnet send UDP packets to an
// edge gateway over point-to-point links. The gateway runs the real IDS (XGBoost C API) on
// one flow record per packet. The measured wall-clock inference time of each packet is used
// as its *simulated* service time, so the gateway behaves as a single-server queue whose
// speed is the speed of the IDS. A finite buffer drops packets when the IDS cannot keep up.
//
// Switches:
//   --mode=two-stage|single-stage   IDS design
//   --priority=1                    robot packets (known source) are served first
//   --block=1                       a source is blocked once it has at least --detectK
//                                   verdicts and at least --detectRatio of them say attack
//                                   (the robot's source is whitelisted)
//   --spoofed=1                     attacker uses a new random source per packet
//   --drr=1                         normal queue split into registered devices and unknown
//                                   sources, served in turn (fair share for known devices)
//   --spoofRobot=f                  share f of flood packets claim the robot's source
//   --authPriority=1                priority only for packets that really come from the robot
//                                   (models an authenticated robot channel)
//   --prioCap=r                     token bucket on the priority queue (packets/s, 0 = off)
//   --timeScale=x                   multiply service times (e.g. 4 for a slower board)
//   --authNs=n                      cost of authenticating one packet, added to every check
//                                   when --authPriority=1 (measured HMAC-SHA256: ~500 ns)
//   --sMaxMs=x                      hard time limit per check (0 = off). A check that would take
//                                   longer is cut at x ms and the flow gets the default verdict
//                                   'suspicious' (fail-safe), as on a real-time gateway
//
// Output: one summary line appended to --summary, plus per-run traces (--trace prefix):
//   <prefix>_robot.csv  time_s,delay_ms (one row per served robot packet)
//   <prefix>_queue.csv  time_s,q_high,q_low (sampled every 10 ms)

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
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("IoT_IDS_Overload");

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

#define XGB_CHECK(call)                                                                  \
    do                                                                                   \
    {                                                                                    \
        if ((call) != 0)                                                                 \
        {                                                                                \
            NS_FATAL_ERROR("XGBoost error: " << XGBGetLastError());                      \
        }                                                                                \
    } while (0)

// ---- Packet tag carrying the (claimed) source id and sender class -------------------------
class IdsTag : public Tag
{
  public:
    static TypeId GetTypeId()
    {
        static TypeId tid = TypeId("ns3::IdsTag").SetParent<Tag>().AddConstructor<IdsTag>();
        return tid;
    }

    TypeId GetInstanceTypeId() const override
    {
        return GetTypeId();
    }

    uint32_t GetSerializedSize() const override
    {
        return 5;
    }

    void Serialize(TagBuffer i) const override
    {
        i.WriteU32(src);
        i.WriteU8(cls);
    }

    void Deserialize(TagBuffer i) override
    {
        src = i.ReadU32();
        cls = i.ReadU8();
    }

    void Print(std::ostream& os) const override
    {
        os << src << "," << int(cls);
    }

    uint32_t src = 0;
    uint8_t cls = 0;
};

// ---- Sender application: constant-rate or Poisson UDP packets with an IdsTag ---------------
class TaggedSender : public Application
{
  public:
    void Setup(Address peer, double rate, bool poisson, uint8_t cls, uint32_t srcBase, uint32_t nSrc,
               uint32_t size, double spoofFrac = 0, uint32_t spoofSrc = 0)
    {
        m_spoofFrac = spoofFrac;
        m_spoofSrc = spoofSrc;
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
        tag.src = m_srcBase + (m_nSrc > 1 ? m_uni->GetInteger(0, m_nSrc - 1) : 0);
        if (m_spoofFrac > 0 && m_uni->GetValue() < m_spoofFrac)
        {
            tag.src = m_spoofSrc; // claims to be the robot
        }
        p->AddPacketTag(tag);
        m_sock->Send(p);
        double gap = m_poisson ? m_exp->GetValue() : 1.0 / m_rate;
        m_event = Simulator::Schedule(Seconds(gap), &TaggedSender::SendOne, this);
    }

    Address m_peer;
    double m_rate = 0;
    bool m_poisson = false;
    uint8_t m_cls = 0;
    uint32_t m_srcBase = 0, m_nSrc = 1, m_size = 512, m_spoofSrc = 0;
    double m_spoofFrac = 0;
    Ptr<Socket> m_sock;
    EventId m_event;
    Ptr<ExponentialRandomVariable> m_exp;
    Ptr<UniformRandomVariable> m_uni;
};

// ---- IDS -----------------------------------------------------------------------------------
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
LoadParams(const std::string& path)
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
}

// Labelled traffic CSV: 46 features, class_id, raw label. keep(class_id) selects rows.
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

// Runs the IDS on one flow; returns the verdict and the wall-clock time in ns.
int
Inspect(const std::vector<float>& flow, int64_t* ns)
{
    auto t0 = std::chrono::steady_clock::now();
    float x[N_FEATURES];
    for (int i = 0; i < N_FEATURES; ++i)
    {
        x[i] = (flow[i] - g_means[i]) / g_scales[i];
    }
    bst_ulong n;
    int verdict = BENIGN_CLASS;
    bool inspect = true;
    if (!g_singleStage)
    {
        inspect = Predict(g_stage1, x, &n)[0] >= g_tau;
    }
    if (inspect)
    {
        const float* p = Predict(g_stage2, x, &n);
        verdict = static_cast<int>(std::max_element(p, p + n) - p);
    }
    *ns = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0)
              .count();
    return verdict;
}

// ---- Gateway queue -------------------------------------------------------------------------
struct Job
{
    Time arrival;
    uint32_t src;
    uint8_t cls;
};

bool g_priority = false, g_block = false;
uint32_t g_capHigh = 100, g_capLow = 1000, g_detectK = 5;
double g_detectRatio = 0.8;
double g_timeScale = 1.0;
double g_authNs = 0, g_sMaxNs = 0;
uint64_t g_timeouts = 0;
const uint32_t ROBOT_SRC = 1; // whitelisted: never blocked
std::deque<Job> g_qHigh, g_qLow, g_qDev; // g_qDev used only with --drr
bool g_drr = false, g_authPriority = false, g_drrTurnDev = true;
double g_prioCap = 0, g_prioBurst = 20, g_tokens = 20;
Time g_lastRefill;
uint32_t g_devBase = 100, g_nDevices = 50;
bool g_busy = false;
double g_busyNs = 0;
std::set<uint32_t> g_blocked;
std::unordered_map<uint32_t, std::pair<uint32_t, uint32_t>> g_verdicts; // src -> (total, attack)
std::set<uint32_t> g_deviceSrcBlocked; // benign devices blocked by mistake

struct Stats
{
    uint64_t arrived = 0, served = 0, dropQueue = 0, dropBlocked = 0, flaggedAttack = 0;
    std::vector<double> delayMs;
};

Stats g_stats[3];
std::ofstream g_robotTrace, g_queueTrace;

void StartService();

void
FinishService(Job job, int verdict)
{
    Stats& st = g_stats[job.cls];
    double d = (Simulator::Now() - job.arrival).GetSeconds() * 1e3;
    st.served++;
    st.delayMs.push_back(d);
    if (job.cls == ROBOT && g_robotTrace)
    {
        g_robotTrace << Simulator::Now().GetSeconds() << "," << d << "\n";
    }
    if (verdict != BENIGN_CLASS)
    {
        st.flaggedAttack++;
    }
    if (g_block && job.src != ROBOT_SRC)
    {
        auto& v = g_verdicts[job.src];
        v.first++;
        v.second += (verdict != BENIGN_CLASS);
        if (v.first >= g_detectK && v.second >= g_detectRatio * v.first &&
            g_blocked.insert(job.src).second && job.cls == DEVICE)
        {
            g_deviceSrcBlocked.insert(job.src);
        }
    }
    g_busy = false;
    StartService();
}

void
StartService()
{
    if (g_busy)
    {
        return;
    }
    if (g_qHigh.empty() && g_qLow.empty() && g_qDev.empty())
    {
        return;
    }
    std::deque<Job>* q = &g_qHigh;
    if (g_qHigh.empty())
    {
        // Round robin between registered devices and unknown sources (when --drr is on).
        bool dev = !g_qDev.empty() && (g_drrTurnDev || g_qLow.empty());
        q = dev ? &g_qDev : &g_qLow;
        g_drrTurnDev = !dev;
    }
    Job job = q->front();
    q->pop_front();
    const std::vector<float>& flow = job.cls == ATTACKER
                                         ? g_floodPool[g_floodNext++ % g_floodPool.size()]
                                         : g_benignPool[g_benignNext++ % g_benignPool.size()];
    int64_t ns;
    int verdict = Inspect(flow, &ns);
    double service = ns * g_timeScale + (g_authPriority ? g_authNs : 0);
    if (g_sMaxNs > 0 && service > g_sMaxNs)
    {
        service = g_sMaxNs;            // check cut at the time limit
        verdict = BENIGN_CLASS + 1;    // default verdict: suspicious (fail-safe)
        g_timeouts++;
    }
    g_busyNs += service;
    g_busy = true;
    Simulator::Schedule(NanoSeconds(static_cast<int64_t>(service)), &FinishService, job, verdict);
}

void
GatewayRx(Ptr<const Packet> p, const Address&)
{
    IdsTag tag;
    if (!p->PeekPacketTag(tag))
    {
        return;
    }
    Stats& st = g_stats[tag.cls];
    st.arrived++;
    if (g_block && g_blocked.count(tag.src))
    {
        st.dropBlocked++; // dropped by the firewall rule before reaching the IDS
        return;
    }
    bool claimsRobot = g_authPriority ? (tag.cls == ROBOT) : (tag.src == ROBOT_SRC);
    bool high = g_priority && claimsRobot;
    if (high && g_prioCap > 0)
    {
        Time now = Simulator::Now();
        g_tokens = std::min(g_prioBurst, g_tokens + (now - g_lastRefill).GetSeconds() * g_prioCap);
        g_lastRefill = now;
        if (g_tokens >= 1)
        {
            g_tokens -= 1;
        }
        else
        {
            high = false; // over the cap: demoted to the normal queue
        }
    }
    bool registeredDevice = tag.src >= g_devBase && tag.src < g_devBase + g_nDevices;
    std::deque<Job>& q = high ? g_qHigh : ((g_drr && registeredDevice) ? g_qDev : g_qLow);
    if (q.size() >= (high ? g_capHigh : g_capLow))
    {
        st.dropQueue++;
        return;
    }
    q.push_back({Simulator::Now(), tag.src, tag.cls});
    StartService();
}

void
SampleQueue()
{
    if (g_queueTrace)
    {
        g_queueTrace << Simulator::Now().GetSeconds() << "," << g_qHigh.size() << ","
                     << g_qLow.size() + g_qDev.size() << "\n";
    }
    Simulator::Schedule(MilliSeconds(10), &SampleQueue);
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

} // namespace

NS_OBJECT_ENSURE_REGISTERED(IdsTag);

int
main(int argc, char* argv[])
{
    std::string params = "ids_params.h", l1 = "layer1_gatekeeper.ubj", l2 = "layer2_multiclass.ubj";
    std::string benignCsv = "test_benign90_20000.csv", floodCsv = "test_natural_20000.csv";
    std::string mode = "two-stage", summary = "overload_summary.csv", trace;
    double robotRate = 100, deviceRate = 300, attackRate = 1000, simTime = 10, floodStart = 2,
           floodStop = 8;
    uint32_t nDevices = 50, nBots = 1000;
    bool spoofed = false;
    uint32_t seed = 1;

    CommandLine cmd(__FILE__);
    cmd.AddValue("params", "ids_params.h", params);
    cmd.AddValue("l1", "Stage 1 model", l1);
    cmd.AddValue("l2", "Stage 2 model", l2);
    cmd.AddValue("benign", "Traffic CSV with benign flows (class 0)", benignCsv);
    cmd.AddValue("flood", "Traffic CSV with flood flows (DDoS/DoS/Mirai)", floodCsv);
    cmd.AddValue("mode", "two-stage | single-stage", mode);
    cmd.AddValue("priority", "Serve robot packets first", g_priority);
    cmd.AddValue("block", "Block a source after detectK attack verdicts", g_block);
    cmd.AddValue("detectK", "Minimum verdicts before a source can be blocked", g_detectK);
    cmd.AddValue("detectRatio", "Share of attack verdicts needed to block a source", g_detectRatio);
    cmd.AddValue("spoofed", "Attacker uses a fresh random source per packet", spoofed);
    double spoofRobot = 0;
    cmd.AddValue("drr", "Fair round robin between registered devices and unknown sources", g_drr);
    cmd.AddValue("spoofRobot", "Share of flood packets that claim the robot's source", spoofRobot);
    cmd.AddValue("authPriority", "Priority only for packets really sent by the robot", g_authPriority);
    cmd.AddValue("prioCap", "Token-bucket rate cap of the priority queue (packets/s, 0 = off)", g_prioCap);
    cmd.AddValue("robotRate", "Robot packets per second", robotRate);
    cmd.AddValue("deviceRate", "Other benign devices, packets per second (total)", deviceRate);
    cmd.AddValue("attackRate", "Flood packets per second", attackRate);
    cmd.AddValue("nBots", "Botnet size", nBots);
    cmd.AddValue("simTime", "Simulation time (s)", simTime);
    cmd.AddValue("floodStart", "Flood start (s)", floodStart);
    cmd.AddValue("floodStop", "Flood stop (s)", floodStop);
    cmd.AddValue("capHigh", "Robot queue capacity (packets)", g_capHigh);
    cmd.AddValue("capLow", "Shared queue capacity (packets)", g_capLow);
    cmd.AddValue("timeScale", "Multiply measured service times", g_timeScale);
    cmd.AddValue("authNs", "Authentication cost per packet (ns), with --authPriority", g_authNs);
    double sMaxMs = 0;
    cmd.AddValue("sMaxMs", "Hard time limit per check (ms, 0 = off)", sMaxMs);
    cmd.AddValue("seed", "Run number for random streams", seed);
    cmd.AddValue("summary", "Summary CSV (appended)", summary);
    cmd.AddValue("trace", "Prefix for per-run trace files (empty = none)", trace);
    cmd.Parse(argc, argv);
    g_sMaxNs = sMaxMs * 1e6;

    RngSeedManager::SetRun(seed);
    g_singleStage = (mode == "single-stage");
    LoadParams(params);
    g_stage1 = LoadBooster(l1);
    g_stage2 = LoadBooster(l2);
    LoadPool(benignCsv, [](int c) { return c == BENIGN_CLASS; }, g_benignPool);
    LoadPool(floodCsv, [](int c) { return c == 2 || c == 3 || c == 4; }, g_floodPool); // DDoS, DoS, Mirai
    // Warm-up so that first-call costs do not land in the simulation.
    for (int i = 0; i < 300; ++i)
    {
        int64_t ns;
        Inspect(g_benignPool[i % g_benignPool.size()], &ns);
        Inspect(g_floodPool[i % g_floodPool.size()], &ns);
    }
    if (!trace.empty())
    {
        g_robotTrace.open(trace + "_robot.csv");
        g_robotTrace << "time_s,delay_ms\n";
        g_queueTrace.open(trace + "_queue.csv");
        g_queueTrace << "time_s,q_high,q_low\n";
    }

    // Topology: robot, devices and attacker each have a point-to-point link to the gateway.
    NodeContainer senders, gateway;
    senders.Create(3);
    gateway.Create(1);
    InternetStackHelper stack;
    stack.Install(senders);
    stack.Install(gateway);
    PointToPointHelper p2p;
    p2p.SetQueue("ns3::DropTailQueue", "MaxSize", StringValue("10000p"));
    Ipv4AddressHelper addr;
    Ipv4Address gwAddr;
    const char* rates[] = {"100Mbps", "100Mbps", "1Gbps"};
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
                         uint32_t nSrc, double start, double stop) {
        Ptr<TaggedSender> app = CreateObject<TaggedSender>();
        app->Setup(InetSocketAddress(gwAddr, port), rate, poisson, cls, srcBase, nSrc, 512);
        senders.Get(node)->AddApplication(app);
        app->SetStartTime(Seconds(start));
        app->SetStopTime(Seconds(stop));
    };
    addSender(0, robotRate, false, ROBOT, ROBOT_SRC, 1, 0.5, simTime);              // robot control loop
    addSender(1, deviceRate, true, DEVICE, g_devBase, nDevices, 0.5, simTime);     // other benign IoT
    g_nDevices = nDevices;
    {
        Ptr<TaggedSender> app = CreateObject<TaggedSender>(); // flood (optionally spoofing the robot)
        app->Setup(InetSocketAddress(gwAddr, port), attackRate, true, ATTACKER, 100000,
                   spoofed ? 100000000 : nBots, 512, spoofRobot, ROBOT_SRC);
        senders.Get(2)->AddApplication(app);
        app->SetStartTime(Seconds(floodStart));
        app->SetStopTime(Seconds(floodStop));
    }

    Simulator::Schedule(Seconds(0), &SampleQueue);
    Simulator::Stop(Seconds(simTime + 1));
    Simulator::Run();
    Simulator::Destroy();

    // ---- Summary ------------------------------------------------------------------------
    bool header = !std::ifstream(summary).good();
    std::ofstream out(summary, std::ios::app);
    if (header)
    {
        out << "mode,priority,block,spoofed,time_scale,robot_rate,device_rate,attack_rate";
        for (const char* c : kClassName)
        {
            out << "," << c << "_arrived," << c << "_served," << c << "_drop_queue," << c
                << "_drop_blocked," << c << "_p50_ms," << c << "_p99_ms," << c << "_max_ms";
        }
        out << ",blocked_sources,devices_blocked,attacker_flagged,busy_frac,drr,spoof_robot,auth_priority,"
               "prio_cap,auth_ns,smax_ms,timeouts\n";
    }
    out << mode << "," << g_priority << "," << g_block << "," << spoofed << "," << g_timeScale << ","
        << robotRate << "," << deviceRate << "," << attackRate;
    for (const Stats& st : g_stats)
    {
        out << "," << st.arrived << "," << st.served << "," << st.dropQueue << "," << st.dropBlocked
            << "," << Pct(st.delayMs, 0.5) << "," << Pct(st.delayMs, 0.99) << ","
            << Pct(st.delayMs, 1.0);
    }
    out << "," << g_blocked.size() << "," << g_deviceSrcBlocked.size() << ","
        << g_stats[ATTACKER].flaggedAttack << "," << g_busyNs / 1e9 / simTime << "," << g_drr << ","
        << spoofRobot << "," << g_authPriority << "," << g_prioCap << "," << g_authNs << "," << sMaxMs
        << "," << g_timeouts << "\n";

    const Stats& r = g_stats[ROBOT];
    std::cout << mode << " prio=" << g_priority << " block=" << g_block << " spoofed=" << spoofed
              << " attack=" << attackRate << "pps | robot served " << r.served << "/" << r.arrived
              << ", dropped " << r.dropQueue << ", p99 " << Pct(r.delayMs, 0.99) << " ms | attacker "
              << g_stats[ATTACKER].arrived << " arrived, " << g_stats[ATTACKER].dropBlocked
              << " blocked | devices blocked " << g_deviceSrcBlocked.size() << std::endl;
    XGBoosterFree(g_stage1);
    XGBoosterFree(g_stage2);
    return 0;
}
