#ifndef MESHFEMSPARSE_NESDIS_OPTIONS_HH
#define MESHFEMSPARSE_NESDIS_OPTIONS_HH

#include "hierarchical_camd.hh"
#include <charconv>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

namespace MeshFEM::CholmodParallelNesdis {

// Private, immutable per-call settings. Parse before allocating graph/workspace
// storage; recursive tasks never consult the environment.
struct NesdisOptions {
    template<class Int = int>
    static Int read_integer(const char *name, Int fallback, Int minimum = 0) {
        const char *text = std::getenv(name);
        if (!text) return fallback;
        Int value;
        const char *end = text + std::strlen(text);
        auto parsed = std::from_chars(text, end, value);
        if (parsed.ec != std::errc() || parsed.ptr != end || value < minimum)
            throw std::invalid_argument(std::string("invalid ") + name);
        return value;
    }

    int threads = read_integer("CHOLMOD_NESDIS_NUM_THREADS", 0);
    int64_t serial_subtree_size = read_integer<int64_t>("CHOLMOD_NESDIS_SERIAL_SUBTREE_SIZE", 2000, 1);
    HierarchicalCAMDOptions camd{
        read_integer("CHOLMOD_NESDIS_CAMD_CUT_DEPTH", 4, -1),
        read_integer("CHOLMOD_NESDIS_CAMD_VERIFY", 0) != 0,
        read_integer("CHOLMOD_NESDIS_CAMD_TRACE", 0) != 0,
        read_integer("CHOLMOD_NESDIS_CAMD_SKIP_UPPER", 0) != 0};
    bool trace_tree = read_integer("CHOLMOD_NESDIS_CAMD_TRACE_TREE", 0) != 0;
    int scotch_levels = read_integer("CHOLMOD_NESDIS_SCOTCH_LEVELS", 0);
    int scotch_threads = read_integer("CHOLMOD_NESDIS_SCOTCH_THREADS", 0, 1);
    int scotch_seed = read_integer("CHOLMOD_NESDIS_SCOTCH_SEED", 0);
    bool scotch_trace = read_integer("CHOLMOD_NESDIS_SCOTCH_TRACE", 0) != 0;
    bool scotch_verify = read_integer("CHOLMOD_NESDIS_SCOTCH_VERIFY", 0) != 0;
    std::string scotch_strategy = [] {
        const char *text = std::getenv("CHOLMOD_NESDIS_SCOTCH_STRATEGY");
        return text ? text : "fast";
    }();
};

} // namespace MeshFEM::CholmodParallelNesdis
#endif
