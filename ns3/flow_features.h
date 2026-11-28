// Flow-meter feature extractor (C++), the exact 45-feature pMN record vector of src/flow_records.py.
// A gateway that sees packets on the wire can compute every classifier input from these per-packet fields,
// so the IDS never needs a precomputed feature row. Validated against the Python extractor by
// src/cpp_feature_check.py (data/CICIoMT2024/cpp/*.ref.csv). No timing or context features (pMN set).
#pragma once
#include <cmath>
#include <cstdint>
#include <vector>

namespace flowfeat
{
constexpr int NF = 45;

// One packet as the gateway sees it. proto = IP protocol number (6 TCP, 17 UDP, 1 ICMP, 2 IGMP, 0 = ARP);
// flags = TCP flag byte (0 otherwise); fwd = sent by the flow's first sender.
struct Pkt
{
    bool fwd;
    int len;
    int hdr;
    int proto; // IP protocol number (Protocol Type); 0 = ARP
    int l4;    // protocol whose L4 header parsed (6/17/1/2), else 0 (ARP or truncated)
    int ttl;
    int flags;
    int sport;
    int dport;
};

inline bool
inset(int a, int b, std::initializer_list<int> s)
{
    for (int v : s)
    {
        if (a == v || b == v)
        {
            return true;
        }
    }
    return false;
}

// Fill out[0..44] in the feature order of runs/pkt_MN_sel/ids_params.h.
inline void
fill(const std::vector<Pkt>& pk, float* out)
{
    const int n = static_cast<int>(pk.size());
    const int FLAG[7] = {0x01, 0x02, 0x04, 0x08, 0x10, 0x40, 0x80}; // fin syn rst psh ack ece cwr
    double hdr = 0, ttl = 0;
    int fc[7] = {0};
    double tot = 0, mn = 1e18, mx = 0;
    int ni = 0, no = 0;
    double ind[7] = {0}; // TCP UDP ICMP ARP IGMP IPv LLC, summed over packets then averaged
    std::vector<double> li, lo;
    for (const Pkt& p : pk)
    {
        hdr += p.hdr;
        ttl += p.ttl;
        for (int i = 0; i < 7; ++i)
        {
            if (p.flags & FLAG[i])
            {
                fc[i]++;
            }
        }
        bool parp = p.proto == 0;
        ind[0] += p.l4 == 6;     // TCP
        ind[1] += p.l4 == 17;    // UDP
        ind[2] += p.l4 == 1;     // ICMP
        ind[3] += parp;          // ARP
        ind[4] += p.l4 == 2;     // IGMP
        ind[5] += parp ? 0 : 1;  // IPv
        ind[6] += parp ? 0 : 1;  // LLC
        double L = p.len;
        tot += L;
        mn = std::min(mn, L);
        mx = std::max(mx, L);
        if (p.fwd)
        {
            lo.push_back(L);
            no++;
        }
        else
        {
            li.push_back(L);
            ni++;
        }
    }
    const Pkt& p0 = pk[0];
    bool arp = p0.proto == 0;
    double avg = tot / n;
    double var = 0;
    for (const Pkt& p : pk)
    {
        var += (p.len - avg) * (p.len - avg);
    }
    var /= n; // population
    auto meanvar = [](const std::vector<double>& v, double& m, double& vv) {
        m = vv = 0;
        if (v.empty())
        {
            return;
        }
        for (double x : v)
        {
            m += x;
        }
        m /= v.size();
        for (double x : v)
        {
            vv += (x - m) * (x - m);
        }
        vv /= v.size(); // population
    };
    double mi, vi, mo, vo;
    meanvar(li, mi, vi);
    meanvar(lo, mo, vo);
    int k = std::min(li.size(), lo.size());
    double cov = 0;
    if (k > 1)
    {
        double ai = 0, ao = 0;
        for (int i = 0; i < k; ++i)
        {
            ai += li[i];
            ao += lo[i];
        }
        ai /= k;
        ao /= k;
        for (int i = 0; i < k; ++i)
        {
            cov += (li[i] - ai) * (lo[i] - ao);
        }
        cov /= (k - 1); // sample covariance, as numpy.cov default
    }

    int i = 0;
    out[i++] = hdr / n;                 // Header_Length
    out[i++] = p0.proto;               // Protocol Type
    out[i++] = ttl / n;                // Duration (mean TTL)
    for (int f = 0; f < 7; ++f)
    {
        out[i++] = double(fc[f]) / n;  // *_flag_number
    }
    for (int f = 0; f < 7; ++f)
    {
        out[i++] = fc[f];              // *_count
    }
    out[i++] = inset(p0.sport, p0.dport, {80, 8080});   // HTTP
    out[i++] = inset(p0.sport, p0.dport, {443});        // HTTPS
    out[i++] = inset(p0.sport, p0.dport, {53});         // DNS
    out[i++] = inset(p0.sport, p0.dport, {23});         // Telnet
    out[i++] = inset(p0.sport, p0.dport, {25});         // SMTP
    out[i++] = inset(p0.sport, p0.dport, {22});         // SSH
    out[i++] = inset(p0.sport, p0.dport, {194, 6667});  // IRC
    out[i++] = inset(p0.sport, p0.dport, {67, 68});     // DHCP
    out[i++] = inset(p0.sport, p0.dport, {1883, 8883}); // MQTT
    for (int f = 0; f < 7; ++f)
    {
        out[i++] = ind[f] / n;         // TCP UDP ICMP ARP IGMP IPv LLC (mean over packets)
    }
    (void)arp;
    out[i++] = tot;                    // Tot sum
    out[i++] = mn;                     // Min
    out[i++] = mx;                     // Max
    out[i++] = avg;                    // AVG
    out[i++] = std::sqrt(var);         // Std
    out[i++] = n;                      // Number
    out[i++] = std::sqrt(mi + mo);     // Magnitue
    out[i++] = std::sqrt(vi + vo);     // Radius
    out[i++] = cov;                    // Covariance
    out[i++] = vo > 0 ? vi / vo : 0;   // Variance
    out[i++] = double(ni) * no;        // Weight
    out[i++] = double(no) / n;         // fwd_share
}
} // namespace flowfeat
