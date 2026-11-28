// Standalone check that the C++ flow-meter extractor (ns3/flow_features.h) reproduces the Python features
// and that the two-stage IDS scores the packet-derived vectors. Reads the per-packet CSV from
// src/export_packets.py, rebuilds each record's 45 features in C++, scales with ids_params.h, runs stage 1
// and stage 2, and writes rec_id, p_attack, verdict and the 45 features. src/cpp_feature_check.py compares
// this against the Python reference. Build: see src/cpp_feature_check.py (clang++ with -lxgboost).
#include "flow_features.h"

#include <xgboost/c_api.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using flowfeat::NF;
using flowfeat::Pkt;

static float g_means[NF], g_scales[NF], g_tau = 0.5f;

static void
LoadParams(const std::string& path)
{
    std::ifstream f(path);
    std::stringstream ss;
    ss << f.rdbuf();
    std::string t = ss.str();
    auto arr = [&](const std::string& key, float* out) {
        size_t p = t.find(key);
        p = t.find('{', p) + 1;
        size_t e = t.find('}', p);
        std::stringstream v(t.substr(p, e - p));
        std::string tok;
        int i = 0;
        while (std::getline(v, tok, ','))
        {
            out[i++] = std::stof(tok);
        }
    };
    arr("SCALER_MEANS", g_means);
    arr("SCALER_SCALES", g_scales);
    size_t p = t.find("GATEKEEPER_TAU");
    g_tau = std::stof(t.substr(t.find('=', p) + 1));
}

static BoosterHandle
Load(const std::string& path)
{
    BoosterHandle h;
    XGBoosterCreate(nullptr, 0, &h);
    XGBoosterLoadModel(h, path.c_str());
    XGBoosterSetParam(h, "nthread", "1");
    return h;
}

static const float*
Predict(BoosterHandle b, const float* row, bst_ulong* n)
{
    char iface[256];
    std::snprintf(iface, sizeof(iface),
                  "{\"data\":[%llu,true],\"shape\":[1,%d],\"typestr\":\"<f4\",\"version\":3}",
                  (unsigned long long)(uintptr_t)row, NF);
    const char* cfg = "{\"type\":0,\"training\":false,\"iteration_begin\":0,\"iteration_end\":0,"
                      "\"strict_shape\":false,\"missing\":NaN}";
    const bst_ulong* shape;
    bst_ulong dim;
    const float* out;
    XGBoosterPredictFromDense(b, iface, cfg, nullptr, &shape, &dim, &out);
    *n = dim == 2 ? shape[1] : 1;
    return out;
}

int
main(int argc, char** argv)
{
    if (argc < 5)
    {
        std::cerr << "usage: pcap_feature_check packets.csv params.h l1.ubj l2.ubj out.csv\n";
        return 1;
    }
    LoadParams(argv[2]);
    BoosterHandle s1 = Load(argv[3]), s2 = Load(argv[4]);
    std::ifstream in(argv[1]);
    std::ofstream out(argv[5]);
    std::string line;
    std::getline(in, line); // header
    out << "rec_id,p_attack,verdict";
    for (int i = 0; i < NF; ++i)
    {
        out << ",f" << i;
    }
    out << "\n";

    long cur = -1;
    std::vector<Pkt> rec;
    auto flush = [&]() {
        if (rec.empty())
        {
            return;
        }
        float feat[NF], x[NF];
        flowfeat::fill(rec, feat);
        for (int i = 0; i < NF; ++i)
        {
            x[i] = (feat[i] - g_means[i]) / g_scales[i];
        }
        bst_ulong n;
        float pa = Predict(s1, x, &n)[0];
        int verdict = 0;
        if (pa >= g_tau)
        {
            const float* p = Predict(s2, x, &n);
            int best = 0;
            for (bst_ulong j = 1; j < n; ++j)
            {
                if (p[j] > p[best])
                {
                    best = j;
                }
            }
            verdict = best;
        }
        out << cur << ',' << pa << ',' << verdict;
        for (int i = 0; i < NF; ++i)
        {
            out << ',' << feat[i];
        }
        out << "\n";
        rec.clear();
    };

    while (std::getline(in, line))
    {
        std::stringstream ss(line);
        std::string tok;
        long vals[10];
        int c = 0;
        while (std::getline(ss, tok, ',') && c < 10)
        {
            vals[c++] = std::stol(tok);
        }
        long rid = vals[0];
        if (rid != cur)
        {
            flush();
            cur = rid;
        }
        rec.push_back({(bool)vals[1], (int)vals[2], (int)vals[3], (int)vals[4], (int)vals[5],
                       (int)vals[6], (int)vals[7], (int)vals[8], (int)vals[9]});
    }
    flush();
    return 0;
}
