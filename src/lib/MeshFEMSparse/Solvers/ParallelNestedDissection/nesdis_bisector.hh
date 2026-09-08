#ifndef MESHFEMSPARSE_NESDIS_BISECTOR_HH
#define MESHFEMSPARSE_NESDIS_BISECTOR_HH

#include <cholmod.h>
#include <cstdint>
#include <memory>

namespace MeshFEM::CholmodParallelNesdis {

struct NesdisOptions;

// Per-ordering configuration and thread-owned optional Scotch workspaces.
class NesdisBisector {
public:
    NesdisBisector();
    explicit NesdisBisector(const NesdisOptions &options);
    ~NesdisBisector();
    int64_t operator()(cholmod_sparse *, int32_t *, int32_t *, int32_t *, int, cholmod_common *);
    int64_t operator()(cholmod_sparse *, int64_t *, int64_t *, int64_t *, int, cholmod_common *);
private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

}
#endif
