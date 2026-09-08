#include "nesdis_bisector.hh"
#include "nesdis_options.hh"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/global_control.h>
#include <tbb/task_arena.h>
#ifdef MESHFEM_WITH_SCOTCH
#include <scotch.h>
#endif

namespace MeshFEM::CholmodParallelNesdis {
namespace {
#ifdef MESHFEM_WITH_SCOTCH
void check(int code, const char *operation) {
    if (code) throw std::runtime_error(std::string("Scotch: ") + operation);
}
struct Context {
    SCOTCH_Context data;
    Context() { check(SCOTCH_contextInit(&data), "context init"); }
    ~Context() { SCOTCH_contextExit(&data); }
};
struct Strategy {
    SCOTCH_Strat data;
    Strategy() { check(SCOTCH_stratInit(&data), "strategy init"); }
    ~Strategy() { SCOTCH_stratExit(&data); }
};
struct Graph {
    SCOTCH_Graph data;
    Graph() { check(SCOTCH_graphInit(&data), "graph init"); }
    ~Graph() { SCOTCH_graphExit(&data); }
};
struct Workspace {
    Context context;
    Strategy strategy;
    int threads;
    Workspace(int count, const std::string &strat) : threads(count) {
        check(SCOTCH_contextRandomClone(&context.data), "private RNG");
        check(SCOTCH_contextOptionSetNum(&context.data, SCOTCH_OPTIONNUMDETERMINISTIC, 0), "parallel matching");
        check(SCOTCH_contextThreadSpawn(&context.data, threads, nullptr), "thread spawn");
        if (strat == "recursive" || strat == "default") {
            check(SCOTCH_stratGraphPartOvlBuild(&strategy.data,
                  strat == "recursive" ? SCOTCH_STRATRECURSIVE : SCOTCH_STRATQUALITY, 2, 0.2), "strategy build");
        }
        else {
            std::string text = strat;
            if (strat == "fast" || strat == "fast2" || strat == "fast4") {
                const std::string candidate = "m{vert=100,rat=0.7,low=h{pass=1},asc=f{bal=0.2}}";
                text = "r{sep=" + candidate;
                int attempts = strat == "fast4" ? 4 : strat == "fast2" ? 2 : 1;
                for (int k = 1; k < attempts; ++k) text += "|" + candidate;
                text += "}";
            }
            check(SCOTCH_stratGraphPartOvl(&strategy.data, text.c_str()), "strategy parse");
        }
    }
};

template<class Int>
bool fits_scotch(const Int *values, size_t n) {
    if constexpr (sizeof(Int) > sizeof(SCOTCH_Num))
        for (size_t j = 0; j < n; ++j)
            if (values[j] > std::numeric_limits<SCOTCH_Num>::max()) return false;
    return true;
}
#endif
}

struct NesdisBisector::Impl {
    int levels;
    bool trace, verify;
    int threads = 1, seed = 0;
#ifdef MESHFEM_WITH_SCOTCH
    std::string strategy;
    tbb::enumerable_thread_specific<std::unique_ptr<Workspace>> workspaces;
#endif
    explicit Impl(const NesdisOptions &options)
        : levels(options.scotch_levels), trace(options.scotch_trace), verify(options.scotch_verify) {
        if (!levels) return;
#ifndef MESHFEM_WITH_SCOTCH
        throw std::runtime_error("CHOLMOD_NESDIS_SCOTCH_LEVELS requires MESHFEM_WITH_SCOTCH");
#else
        size_t limit = std::min(tbb::global_control::active_value(tbb::global_control::max_allowed_parallelism),
                               (size_t)tbb::this_task_arena::max_concurrency());
        if (options.threads) limit = std::min(limit, (size_t)options.threads);
        threads = options.scotch_threads ? options.scotch_threads : (int)limit;
        threads = (int)std::min((size_t)threads, limit);
        seed = options.scotch_seed;
        strategy = options.scotch_strategy;
        // Initialize Scotch's global RNG before any concurrent private clones,
        // and validate the strategy before starting the recursive traversal.
        static std::once_flag rng_init;
        std::call_once(rng_init, [] {
            Context context;
            check(SCOTCH_contextRandomClone(&context.data), "initialize RNG");
        });
        workspaces.local() = std::make_unique<Workspace>(1, strategy);
#endif
    }

    template<class Int>
    int64_t bisect(cholmod_sparse *matrix, Int *weights, Int *edge_weights,
                   Int *part, int depth, cholmod_common *common) {
        auto start = std::chrono::steady_clock::now();
        bool use_scotch = false;
        int thread_count = 1;
        int64_t separator;
#ifdef MESHFEM_WITH_SCOTCH
        const auto *p = static_cast<const Int *>(matrix->p);
        const auto *i = static_cast<const Int *>(matrix->i);
        size_t n = matrix->nrow, nz = p[n];
        use_scotch = depth < levels && n <= (size_t)std::numeric_limits<SCOTCH_Num>::max() &&
                     nz <= (size_t)std::numeric_limits<SCOTCH_Num>::max() && fits_scotch(weights, n);
        // Scotch also stores aggregate graph weights in SCOTCH_Num.
        if (use_scotch) {
            uint64_t total = 0;
            for (size_t v = 0; v < n; ++v) total += weights[v];
            use_scotch = total <= (uint64_t)std::numeric_limits<SCOTCH_Num>::max();
        }
        if (use_scotch) {
            thread_count = threads;
            for (int level = 0; level < depth && thread_count > 1; ++level) thread_count = std::max(1, thread_count / 2);
            auto &local = workspaces.local();
            if (!local || local->threads != thread_count) local = std::make_unique<Workspace>(thread_count, strategy);
            std::vector<SCOTCH_Num> sp, si, sw, output(n);
            const SCOTCH_Num *gp, *gi, *gw;
            if constexpr (std::is_same_v<Int, SCOTCH_Num>) { gp = p; gi = i; gw = weights; }
            else {
                sp.assign(p, p + n + 1); si.assign(i, i + nz); sw.assign(weights, weights + n);
                gp = sp.data(); gi = si.data(); gw = sw.data();
            }
            Graph graph, bound;
            check(SCOTCH_graphBuild(&graph.data, 0, (SCOTCH_Num)n, gp, gp + 1, gw,
                                   nullptr, (SCOTCH_Num)nz, gi, nullptr), "graph build");
            check(SCOTCH_contextBindGraph(&local->context.data, &graph.data, &bound.data), "context bind");
            SCOTCH_contextRandomSeed(&local->context.data, seed);
            check(SCOTCH_graphPartOvl(&bound.data, 2, &local->strategy.data, output.data()), "vertex separator");
            int64_t counts[3]{};
            separator = 0;
            for (size_t v = 0; v < n; ++v) {
                int label = output[v] == -1 ? 2 : output[v];
                if (label < 0 || label > 2) throw std::runtime_error("Scotch returned invalid partition label");
                part[v] = (Int)label; ++counts[label];
                if (label == 2) separator += weights[v];
            }
            if (verify)
                for (size_t v = 0; v < n; ++v)
                    for (Int k = p[v]; k < p[v + 1]; ++k)
                        if (part[v] != 2 && part[i[k]] != 2 && part[v] != part[i[k]])
                            throw std::runtime_error("Scotch separator has an edge crossing interiors");
            // Match cholmod_metis_bisector's convention for disconnected graphs:
            // an empty separator must not make partition() discard the split.
            if (!separator && n) {
                size_t lightest = 0;
                for (size_t v = 0; v < n; ++v) if (weights[v] <= weights[lightest]) lightest = v;
                --counts[part[lightest]];
                part[lightest] = 2;
                separator = weights[lightest];
            }
            if (!counts[0] || !counts[1]) {
                separator = 0;
                for (size_t v = 0; v < n; ++v) { part[v] = 2; separator += weights[v]; }
            }
            common->status = CHOLMOD_OK;
        }
        else
#endif
        {
            if constexpr (std::is_same_v<Int, int32_t>) separator = cholmod_metis_bisector(matrix, weights, edge_weights, part, common);
            else separator = cholmod_l_metis_bisector(matrix, weights, edge_weights, part, common);
        }
        if (trace && depth < std::max(levels, 1))
            std::fprintf(stderr, "NESDIS_BISECTOR {\"scotch\":%s,\"depth\":%d,\"threads\":%d,\"vertices\":%zu,\"separator_weight\":%lld,\"seconds\":%.9g}\n",
                         use_scotch ? "true" : "false", depth, thread_count, matrix->nrow, (long long)separator,
                         std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
        return separator;
    }
};

NesdisBisector::NesdisBisector() : NesdisBisector(NesdisOptions{}) { }
NesdisBisector::NesdisBisector(const NesdisOptions &options) : m_impl(std::make_unique<Impl>(options)) { }
NesdisBisector::~NesdisBisector() = default;
int64_t NesdisBisector::operator()(cholmod_sparse *a, int32_t *w, int32_t *ew, int32_t *part, int depth, cholmod_common *common) {
    return m_impl->bisect(a, w, ew, part, depth, common);
}
int64_t NesdisBisector::operator()(cholmod_sparse *a, int64_t *w, int64_t *ew, int64_t *part, int depth, cholmod_common *common) {
    return m_impl->bisect(a, w, ew, part, depth, common);
}
}
