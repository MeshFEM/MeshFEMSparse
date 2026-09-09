#include "hierarchical_camd.hh"
#include "cholmod_nesdis_parallel.hh"
#include "nesdis_bisector.hh"
#include "nesdis_options.hh"
#include "cholmod_internal_excerpts.hh"
#include <MeshFEMCore/GlobalBenchmark.hh>

#include <algorithm>
#include <chrono>
#include <limits>
#include <memory>
#include <sstream>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <tbb/enumerable_thread_specific.h>
#include <tbb/global_control.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_invoke.h>
#include <tbb/task_arena.h>
#include <tbb/task_group.h>

namespace MeshFEM::CholmodParallelNesdis {

//------------------------------------------------------------------------------
// recursive nested-dissection work state
//------------------------------------------------------------------------------
// Here we decompose the state used by cholmod_nested_dissection into shared and
// local components. The former consists of all data that can be safely accessed
// concurrently by multiple threads, while the latter must be task- or
// thread-local.

struct nesdis_failure {
    explicit nesdis_failure(int status_ = CHOLMOD_OUT_OF_MEMORY)
        : status (status_) { }

    int status;
};

template<class Int>
struct nesdis_local {
    nesdis_local(Int n_, Int csize_, cholmod_common *Common_);
    nesdis_local(const nesdis_local &) = delete;
    nesdis_local &operator=(const nesdis_local &) = delete;
    ~nesdis_local() { release (); }

    void release () ;

    Int n ;
    Int csize ;
    cholmod_sparse *C ;

    Int *Imap ;
    Int *Hash ;
    Int *Cmap ;
    Int *Cp ;
    Int *Ci ;
    Int *Cew ;
    Int *Cnw ;
    Int *Part ;
    Int *WorkLocal ;

    cholmod_common CommonLocal ;
    cholmod_common *Common ;
};

template<class Int>
struct nesdis_shared {
    explicit nesdis_shared(const NesdisOptions &options) : Bisector(options) { }
    NesdisBisector Bisector;
    Int n ;
    Int csize ;
    Int nd_compress ;
    Int nd_small ;
    double nd_oksep ;

    cholmod_sparse *B ;
    Int *Bp ;
    Int *Bi ;
    Int *Bnz ;
    Int *Flag ;
    Int *Bnw ;
    Int *CParent ;

    tbb::task_group TaskGroup ;
    cholmod_common *Common ;
    bool UseParallel = false;
    int MaxParallelDepth = 0;
    Int SerialSubtreeSize;
    tbb::enumerable_thread_specific<std::unique_ptr<nesdis_local<Int>>> LocalWorkspaces ;
};

template<class Int>
inline nesdis_local<Int>::nesdis_local
(
    Int n_,
    Int csize_,
    cholmod_common *Common_
)
{
    Int j ;

    n = n_ ;
    csize = csize_ ;
    CommonLocal = *Common_ ;
    Common = &(CommonLocal) ;
    C = NULL ;
    Cew = NULL ;
    WorkLocal = NULL ;

    C = CholmodApi<Int>::allocate_sparse (n, n, csize, FALSE, TRUE, 0,
            CHOLMOD_PATTERN, Common) ;
    if (Common->method [Common->current].nd_compress)
        Cew = (Int *) CholmodApi<Int>::malloc (csize, sizeof (Int), Common) ;
    WorkLocal = (Int *) CholmodApi<Int>::malloc (n, 5*sizeof (Int), Common) ;

    if (Common->status < CHOLMOD_OK)
    {
        int status = Common->status ;
        release () ;
        throw nesdis_failure (status) ;
    }

    Part = WorkLocal ;
    Cnw  = Part + n ;
    Imap = Cnw + n ;
    Hash = Imap + n ;
    Cmap = Hash + n ;
    Cp = (Int *) C->p ;
    Ci = (Int *) C->i ;
    if (Cew) for (j = 0 ; j < csize ; j++)
    {
        Cew [j] = 1 ;
    }
}


template<class Int>
inline void nesdis_local<Int>::release
(
)
{
    if (Common == NULL)
    {
        return ;
    }
    if (C != NULL)
    {
        C->ncol = n ;
        CholmodApi<Int>::free_sparse (&C, Common) ;
    }
    CholmodApi<Int>::free (csize, sizeof (Int), Cew, Common) ;
    CholmodApi<Int>::free (5*n, sizeof (Int), WorkLocal, Common) ;
    Cew = NULL ;
    WorkLocal = NULL ;
    Part = NULL ;
    Cnw = NULL ;
    Imap = NULL ;
    Hash = NULL ;
    Cmap = NULL ;
    Cp = NULL ;
    Ci = NULL ;
}

//------------------------------------------------------------------------------
// Component discovery and traversal order
//------------------------------------------------------------------------------

template<class Int>
struct NesdisComponent {
    Int representative;
    std::vector<Int> vertices;
};

// Discover live components and compact their adjacency lists. Return groups in
// the old stack's processing order, with vertices in the old graph builder's BFS
// order. When nd_components is false, multiple components form one group per side.
template<class Int>
static std::vector<NesdisComponent<Int>> find_components_on_side(
    cholmod_sparse *B, const Int *Map, Int cn, Int cnode, const Int *Part,
    int part, Int *Bnz, Int *CParent, Int *State, Int mark,
    Int *Queue, bool separate)
{
    const auto *Bp = static_cast<const Int *>(B->p);
    auto *Bi = static_cast<Int *>(B->i);
    std::vector<NesdisComponent<Int>> result;

    NesdisComponent<Int> group{EMPTY, {}};
    struct Layer { size_t begin, end; bool last; };
    std::vector<Layer> layers;
    std::vector<size_t> heads;
    for (Int cj = 0; cj < cn; ++cj) {
        if (Part && Part[cj] != part) continue;
        const Int snode = Map ? Map[cj] : cj;
        const Int state = State[snode];
        if (state < EMPTY || state == mark) continue;
        ASSERT(CParent[snode] == -2);
        if (separate || group.representative == EMPTY) CParent[snode] = cnode;
        if (group.representative == EMPTY) group.representative = snode;

        Queue[0] = snode;
        State[snode] = mark;
        Int sn = 1, sj = 0;
        if (!separate) heads.push_back(layers.size());
        while (sj < sn) {
            const Int layer_begin = sj, layer_end = sn;
            for (; sj < layer_end; ++sj) {
                const Int j = Queue[sj], pstart = Bp[j], pend = pstart + Bnz[j];
                Int pdest = pstart;
                for (Int p = pstart; p < pend; ++p) {
                    const Int i = Bi[p];
                    ASSERT(i != j); // Graph construction excludes diagonals.
                    const Int neighbor_state = State[i];
                    if (neighbor_state < EMPTY) continue;
                    Bi[pdest++] = i;
                    if (neighbor_state != mark) {
                        Queue[sn++] = i;
                        State[i] = mark;
                    }
                }
                Bnz[j] = pdest - pstart;
            }
            if (!separate)
                layers.push_back({group.vertices.size() + (size_t)layer_begin,
                                  group.vertices.size() + (size_t)layer_end, sj == sn});
        }
        if (separate) result.push_back({snode, std::vector<Int>(Queue, Queue + sn)});
        else group.vertices.insert(group.vertices.end(), Queue, Queue + sn);
    }
    if (!separate && group.representative != EMPTY) {
        if (heads.size() > 1) {
            // The old builder seeds one BFS with component representatives
            // in reverse discovery order. Merge their saved layers to match
            // that numbering, without traversing any adjacency again.
            std::reverse(heads.begin(), heads.end());
            std::vector<Int> ordered;
            ordered.reserve(group.vertices.size());
            for (size_t k = 0; k < heads.size(); ++k) {
                const size_t index = heads[k];
                const auto &layer = layers[index];
                ordered.insert(ordered.end(), group.vertices.begin() + layer.begin,
                               group.vertices.begin() + layer.end);
                if (!layer.last) heads.push_back(index + 1);
            }
            group.vertices = std::move(ordered);
        }
        result.push_back(std::move(group));
    }
    std::reverse(result.begin(), result.end());
    return result;
}

template<class Int>
static std::vector<NesdisComponent<Int>> find_components(
    cholmod_sparse *B, const Int *Map, Int cn, Int cnode, const Int *Part,
    Int *Bnz, Int *CParent, Int *State,
    Int *Queue, cholmod_common *Common, bool parallel = false)
{
    // States below EMPTY retain CHOLMOD's removed-node encoding. Each discovery
    // uses its removed parent separator's ID + 1 as a fresh nonnegative tag;
    // that vertex cannot be a separator again. Initial discovery uses 0 when
    // there is no dense-node parent. Tags need not increase along the tree.
    // Concurrent searches touch disjoint live vertices; shared removed states
    // remain unchanged, so neither atomics nor a shared counter are needed.
    ASSERT(cnode >= EMPTY && cnode < (Int)B->nrow);
    const Int mark = cnode + 1;
    const bool separate = Common->method[Common->current].nd_components;
    auto find_side = [&](int part, Int *queue) {
        return find_components_on_side<Int>(B, Map, cn, cnode, Part, part,
                Bnz, CParent, State, mark, queue, separate);
    };
    if (!Part) return find_side(0, Queue);

    std::vector<NesdisComponent<Int>> sides[2];
    if (parallel) {
        const Int side0_size = (Int)std::count(Part, Part + cn, 0);
        // Sides touch disjoint vertices and use disjoint queue slices. Isolation
        // prevents another component task from reusing the parent's TLS scratch
        // while this thread waits; helpers never obtain their own TLS workspace.
        tbb::this_task_arena::isolate([&] {
            tbb::parallel_invoke(
                [&] { sides[0] = find_side(0, Queue); },
                [&] { sides[1] = find_side(1, Queue + side0_size); });
        });
    }
    else {
        sides[1] = find_side(1, Queue);
        sides[0] = find_side(0, Queue);
    }
    auto &result = sides[0];
    result.reserve(result.size() + sides[1].size());
    for (auto &component : sides[1]) result.push_back(std::move(component));
    return std::move(result);
}

template<class Int>
static void nesdis_process_component(
    nesdis_shared<Int> *S, nesdis_local<Int> *L,
    const NesdisComponent<Int> &component, Int depth);

template<class Int>
static void nesdis_process_components(
    nesdis_shared<Int> *S, nesdis_local<Int> *L,
    std::vector<NesdisComponent<Int>> components, Int depth)
{
    if (S->UseParallel && depth < S->MaxParallelDepth) {
        // Tasks own their vertex lists; the first group stays on this thread.
        for (size_t k = 1; k < components.size(); ++k) {
            S->TaskGroup.run([S, component = std::move(components[k]), depth]() {
                auto &local = S->LocalWorkspaces.local();
                if (!local) local = std::make_unique<nesdis_local<Int>>(S->n, S->csize, S->Common);
                nesdis_process_component<Int>(S, local.get(), component, depth);
            });
        }
        if (!components.empty()) nesdis_process_component<Int>(S, L, components.front(), depth);
        return;
    }
    for (const auto &component : components)
        nesdis_process_component<Int>(S, L, component, depth);
}

template<class Int>
static int nesdis_max_parallel_depth
(
    Int n,
    Int target_size
)
{
    if (n <= target_size)
    {
        return 0 ;
    }

    int depth = 0 ;
    for (Int subtree_size = n ; subtree_size > target_size ; subtree_size = subtree_size / 2 + subtree_size % 2)
    {
        ++depth ;
    }
    return depth + 1 ;
}

template<class Int>
static void nesdis_process_parallel
(
    nesdis_shared<Int> *S,
    nesdis_local<Int> *MainLocal,
    std::vector<NesdisComponent<Int>> components,
    int max_parallel_depth
)
{
    S->UseParallel = TRUE ;
    S->MaxParallelDepth = max_parallel_depth ;

    try {
        // Initial connected components are independent, just like separator children.
        nesdis_process_components<Int> (S, MainLocal, std::move(components), 0) ;
    }
    catch (...) {
        S->TaskGroup.cancel() ;
        S->TaskGroup.wait() ;
        throw ;
    }
    S->TaskGroup.wait();
}

//------------------------------------------------------------------------------
// nesdis_process_component
//------------------------------------------------------------------------------


template<class Int>
static void nesdis_process_component
(
    nesdis_shared<Int> *S,
    nesdis_local<Int> *L,
    const NesdisComponent<Int> &component,
    Int depth
)
{
    Int *Bp = S->Bp ;
    Int *Bi = S->Bi ;
    Int *Bnz = S->Bnz ;
    Int *Imap = L->Imap ;
    const Int *Map = component.vertices.data() ;
    Int *Flag = S->Flag ;
    Int *Hash = L->Hash ;
    Int *Cmap = L->Cmap ;
    Int *Cp = L->Cp ;
    Int *Ci = L->Ci ;
    Int *Cew = L->Cew ;
    Int *Bnw = S->Bnw ;
    Int *Cnw = L->Cnw ;
    Int *Part = L->Part ;
    Int *CParent = S->CParent ;
    cholmod_sparse *B = S->B ;
    cholmod_sparse *C = L->C ;
    cholmod_common *Common = L->Common ;
    const bool compress = S->nd_compress != 0 ;

    using UInt = std::make_unsigned_t<Int> ;
    Int cnode = component.representative, cn = (Int)component.vertices.size();
    Int i, j, cj, ci, cnz, total_weight, sepsize, parent;
    DEBUG (Int p) ;
    DEBUG (Int cnt) ;

    ASSERT(cn > 0 && cnode >= 0 && cnode < S->n);
    if (cn < S->nd_small) {
        for (Int v : component.vertices) Flag[v] = FLIP(cnode);
        return;
    }
    {
        //----------------------------------------------------------------------
        // create the subgraph for this connected component C
        //----------------------------------------------------------------------

        cnz = 0 ;
        total_weight = 0 ;
        for (cj = 0 ; cj < cn ; cj++)
        {
            j = Map [cj] ;
            Imap[j] = cj;
            Cp [cj] = cnz ;
            Cnw [cj] = Bnw [j] ;
            ASSERT (Cnw [cj] >= 0) ;
            total_weight += Cnw [cj] ;
            cnz += Bnz[j];
        }
        Cp [cn] = cnz ;
        auto fill_column = [&](Int column) {
            const Int vertex = Map[column], start = Bp[vertex];
            UInt hash = column;
            for (Int k = 0; k < Bnz[vertex]; ++k) {
                const Int neighbor = Bi[start + k];
                // Construction excludes self-edges; discovery prunes deleted vertices.
                ASSERT(neighbor != vertex && Flag[neighbor] >= EMPTY);
                const Int local = Imap[neighbor];
                ASSERT(local >= 0 && local < cn && local != column && Cp[column] + k < S->csize);
                Ci[Cp[column] + k] = local;
                if (compress) hash += local;
            }
            if (compress) {
                hash %= S->csize ;
                Hash[column] = (Int)hash;
            }
        };
        const bool parallel_construction = !compress && S->UseParallel && depth < S->MaxParallelDepth &&
                                           cn > S->SerialSubtreeSize;
        if (parallel_construction) {
            // Imap is complete and Cp gives disjoint output slices. Protect the
            // parent's TLS scratch while waiting, just as in component discovery.
            tbb::this_task_arena::isolate([&] {
                tbb::parallel_for(tbb::blocked_range<Int>(0, cn, 128), [&](const auto &range) {
                    for (Int column = range.begin(); column < range.end(); ++column) fill_column(column);
                });
            });
        }
        else for (Int column = 0; column < cn; ++column) fill_column(column);
        C->nrow = cn ;
        C->ncol = cn ;

        #ifndef NDEBUG
        for (cj = 0 ; cj < cn ; cj++)
        {
            j = Map [cj] ;
            PRINT2 (("----------------------------C column cj: " ID " j: " ID "\n",
                cj, j)) ;
            ASSERT (j >= 0 && j < S->n) ;
            ASSERT (Flag [j] >= EMPTY) ;
            for (p = Cp [cj] ; p < Cp [cj+1] ; p++)
            {
                ci = Ci [p] ;
                i = Map [ci] ;
                PRINT3 (("ci: " ID " i: " ID "\n", ci, i)) ;
                ASSERT (ci != cj && ci >= 0 && ci < cn) ;
                ASSERT (i != j && i >= 0 && i < S->n) ;
                ASSERT (Flag [i] >= EMPTY) ;
            }
        }
        #endif

        ASSERT(cn >= S->nd_small);
        {
            PRINT0 ((" cut\n")) ;

            sepsize = partition<Int> (
                #ifndef NDEBUG
                S->csize,
                #endif
                compress, depth, Hash, C, Cnw, Cew,
                Cmap, Part, Common, S->Bisector) ;

            if (sepsize < 0)
            {
                C->ncol = S->n ;
                throw nesdis_failure (Common->status) ;
            }

            if (compress) for (ci = 0 ; ci < cn ; ci++)
            {
                if (Hash [ci] < EMPTY)
                {
                    cj = FLIP (Hash [ci]) ;
                    PRINT2 (("In C, " ID " absorbed into " ID " (wgt now " ID ")\n",
                            ci, cj, Cnw [cj])) ;
                    i = Map [ci] ;
                    j = Map [cj] ;
                    PRINT2 (("In B, " ID " (wgt " ID ") => " ID " (wgt " ID ")\n",
                                i, Bnw [i], j, Bnw [j], Cnw [cj])) ;
                    Bnw [i] = 0 ;
                    Bnw [j] = Cnw [cj] ;
                    Flag [i] = FLIP (j) ;
                }
            }

            // Only inspect this component: other tasks may modify Bnw elsewhere.
            DEBUG (for (cnt = 0, cj = 0 ; cj < cn ; cj++) cnt += Bnw [Map [cj]]) ;
            ASSERT (cnt == total_weight) ;
        }

        ASSERT (sepsize >= 0 && sepsize <= total_weight) ;

        PRINT0 (("sepsize %d tot %d : %8.4f ", sepsize, total_weight,
            ((double) sepsize) / ((double) total_weight))) ;

        if (sepsize == total_weight || sepsize == 0 ||
            sepsize > S->nd_oksep * total_weight)
        {
            PRINT2 (("cnode %d sepsize zero or all of graph: " ID "\n",
                cnode, sepsize)) ;
            for (cj = 0 ; cj < cn ; cj++)
            {
                j = Map [cj] ;
                Flag [j] = FLIP (cnode) ;
                PRINT2 (("      node cj: " ID " j: " ID " ordered\n", cj, j)) ;
            }
            ASSERT (Flag [cnode] == FLIP (cnode)) ;
            ASSERT (cnode != EMPTY && Flag [cnode] < EMPTY) ;
            PRINT0 (("discarded\n")) ;

        }
        else
        {
            PRINT0 (("sepsize not tiny: " ID "\n", sepsize)) ;
            parent = CParent [cnode] ;
            ASSERT (parent >= EMPTY && parent < S->n) ;
            CParent [cnode] = -2 ;
            cnode = EMPTY ;
            for (cj = 0 ; cj < cn ; cj++)
            {
                j = Map [cj] ;
                if (Part [cj] == 2)
                {
                    PRINT2 (("node cj: " ID " j: " ID " ordered\n", cj, j)) ;
                    if (cnode == EMPTY)
                    {
                        PRINT2(("------------new cnode: cj " ID " j " ID "\n",
                                    cj, j)) ;
                        cnode = j ;
                    }
                    Flag [j] = FLIP (cnode) ;
                }
                else
                {
                    PRINT2 (("      node cj: " ID " j: " ID " not ordered\n",
                                cj, j)) ;
                }
            }
            ASSERT (cnode != EMPTY && Flag [cnode] < EMPTY) ;
            ASSERT (CParent [cnode] == -2) ;
            CParent [cnode] = parent ;

            const bool parallel_discovery = S->UseParallel && depth < S->MaxParallelDepth;
            auto children = find_components<Int>(B, Map, cn, cnode, Part, Bnz,
                    CParent, Flag, Imap, Common, parallel_discovery);
            nesdis_process_components<Int>(S, L, std::move(children), depth + 1);
        }
    }
}


//------------------------------------------------------------------------------
// cholmod_nested_dissection
//------------------------------------------------------------------------------

// This method uses a node bisector, applied recursively. Once the graph is partitioned, it calls a
// constrained min degree code (CAMD or CSYMAMD for A+A', and CCOLAMD for A*A')
// to order all the nodes in the graph - but obeying the constraints determined
// by the separators.  This routine is similar to METIS_NodeND, except for how
// it treats the leaf nodes.  METIS_NodeND orders the leaves of the separator
// tree with MMD, ignoring the rest of the matrix when ordering a single leaf.
// After partitioning, symmetric inputs use hierarchical CAMD by default, or
// global CAMD/CSYMAMD when requested. Unsymmetric inputs retain CCOLAMD.
//
// This function also returns a postordered separator tree (CParent), and a
// mapping of nodes in the graph to nodes in the separator tree (Cmember).
//
// workspace: Flag (nrow), Head (nrow+1), Iwork (4*nrow + (ncol if unsymmetric))
//      Allocates a mutable full graph B and node weights Bnw.
//      Each local workspace owns O(n + nnz(B)) storage; tasks own vertex lists.

// Copy without symmetrizing, filtering optional diagonals. Keep the original
// column offsets and store live lengths in the caller's Bnz workspace, avoiding
// a separate compaction pass. As in discovery, B->nz stays null (Bnz is external).
template<class Int>
static cholmod_sparse *copy_full_graph(const cholmod_sparse &graph, Int *Bnz, cholmod_common *Common)
{
    const auto *Ap = static_cast<const Int *>(graph.p);
    const auto *Ai = static_cast<const Int *>(graph.i);
    auto *B = CholmodApi<Int>::allocate_sparse(graph.nrow, graph.ncol,
            Ap[graph.ncol], TRUE, TRUE, 0, CHOLMOD_PATTERN, Common);
    if (!B) return nullptr;
    auto *Bp = static_cast<Int *>(B->p);
    auto *Bi = static_cast<Int *>(B->i);
    try {
        tbb::parallel_for(tbb::blocked_range<size_t>(0, graph.ncol, 256), [&](const auto &range) {
            for (size_t j = range.begin(); j < range.end(); ++j) {
                Bp[j] = Ap[j];
                Int dest = Ap[j];
                for (Int k = Ap[j]; k < Ap[j + 1]; ++k)
                    if (Ai[k] != Int(j)) Bi[dest++] = Ai[k];
                Bnz[j] = dest - Ap[j];
            }
        });
        Bp[graph.ncol] = Ap[graph.ncol];
        B->packed = FALSE;
    }
    catch (...) {
        CholmodApi<Int>::free_sparse(&B, Common);
        throw;
    }
    return B;
}

// Build a fresh ND forest, or, when ReusedTree is supplied, only flatten its
// active nodes into postordered CParent/Cmember arrays (no bisectors). ReusedTree
// must already be valid for the current graph and cover all A->nrow vertices.
// Unless NDOnly is true, then compute Perm with the configured constrained
// ordering (hierarchical/global CAMD, CSYMAMD, CCOLAMD, or natural component order).
// NDOnly callers consume the forest outputs; NDSeconds excludes final ordering.
// Without FullGraph, symmetric A is expanded and unsymmetric A orders A(:,fset)*A(:,fset)'
// (all columns if fset is null). FullGraph instead supplies the same symmetric
// graph directly: square, sorted, packed, duplicate-free CSC with both triangles,
// stype == 0, optional diagonals, and dimension A->nrow; A->stype must be nonzero.
// Inputs/Common must use Int indices, Common must be initialized and exclusive
// to this call, and all three output arrays must have room for A->nrow entries.
template<class Int>
static int64_t nested_dissection_impl // returns # of components, or -1 if error
(
    // input:
    cholmod_sparse *A,  // matrix to order
    Int *fset,          // subset of 0:(A->ncol)-1
    size_t fsize,       // size of fset
    // output:
    Int *Perm,          // size A->nrow, output permutation
    Int *CParent,       // size A->nrow.  On output, CParent [c] is the parent
                        // of component c, or EMPTY if c is a root, and where
                        // c is in the range 0 to # of components minus 1
    Int *Cmember,       // size A->nrow.  Cmember [j] = c if node j of A is
                        // in component c
    cholmod_common *Common,
    const NesdisOptions &options,
    // Borrowed full symmetric, sorted, packed graph; optional diagonals are
    // filtered from the private ND copy and ignored by the CAMD wrappers.
    const cholmod_sparse *FullGraph = nullptr,
    const PersistentSeparatorTree<Int> *ReusedTree = nullptr,
    bool NDOnly = false, double *NDSeconds = nullptr
)
{
    const auto nesdis_start = std::chrono::steady_clock::now();

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------


    double prune_dense, nd_oksep ;
    Int *Bp, *Bi, *Bnz, *Flag, *Head, *Next, *Bnw, *Iwork,
        *Ipost, *NewParent, *Post ;
    Int n, bnz, i, j, k, cnode, cdense, c,
        parent, ncomponents, threshold, ndense,
        nd_compress, nd_camd, csize, jnext, nd_small,
        nchild, child = EMPTY ;
    cholmod_sparse *B ;
    DEBUG (Int cnt) ;

    RETURN_IF_NULL_COMMON (EMPTY) ;
    RETURN_IF_NULL (A, EMPTY) ;
    RETURN_IF_NULL (Perm, EMPTY) ;
    RETURN_IF_NULL (CParent, EMPTY) ;
    RETURN_IF_NULL (Cmember, EMPTY) ;
    RETURN_IF_XTYPE_INVALID (A, CHOLMOD_PATTERN, CHOLMOD_ZOMPLEX, EMPTY) ;
    Common->status = CHOLMOD_OK ;

    //--------------------------------------------------------------------------
    // quick return
    //--------------------------------------------------------------------------

    n = A->nrow ;
    if (n == 0)
    {
        return (1) ;
    }

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    // get ordering parameters
    prune_dense = Common->method [Common->current].prune_dense ;
    nd_compress = Common->method [Common->current].nd_compress ;
    nd_oksep = Common->method [Common->current].nd_oksep ;
    nd_oksep = MAX (0, nd_oksep) ;
    nd_oksep = MIN (1, nd_oksep) ;
    nd_camd = Common->method [Common->current].nd_camd ;
    nd_small = Common->method [Common->current].nd_small ;
    nd_small = MAX (4, nd_small) ;

    PRINT0 (("nd_components %d nd_small %d nd_oksep %g\n",
        Common->method [Common->current].nd_components,
        nd_small, nd_oksep)) ;

    //--------------------------------------------------------------------------
    // allocate workspace
    //--------------------------------------------------------------------------

    // s = 4*nrow + uncol
    size_t uncol = (A->stype == 0) ? A->ncol : 0 ;
    int ok = TRUE ;
    size_t s = mult_size_t (A->nrow, 4, &ok) ;
    s = add_size_t (s, uncol, &ok) ;
    if (!ok)
    {
        ERROR (CHOLMOD_TOO_LARGE, "problem too large") ;
        return (EMPTY) ;
    }

    CholmodApi<Int>::allocate_work (A->nrow, s, 0, Common) ;
    if (Common->status < CHOLMOD_OK)
    {
        return (EMPTY) ;
    }

    //--------------------------------------------------------------------------
    // get workspace
    //--------------------------------------------------------------------------

    Flag = (Int *) Common->Flag ;       // size n
    Head = (Int *) Common->Head ;       // size n+1, all equal to -1

    Iwork = (Int *) Common->Iwork ;
    Bnz = Iwork ;               // size n

    Bnw = NULL ;

    if (Common->status < CHOLMOD_OK)
    {
        return (EMPTY) ;
    }

    //--------------------------------------------------------------------------
    // convert B to symmetric form with both upper/lower parts present
    //--------------------------------------------------------------------------

    double partition_seconds = 0;
    if (ReusedTree) {
        // Note: ND is skipped when ReusedTree is passed!
        ncomponents = ReusedTree->flatten(CParent, Cmember);
    }
    else {
    // B = A+A', A*A', or A(:,f)*A(:,f)', upper and lower parts present

    if (FullGraph)
    {
        B = copy_full_graph<Int>(*FullGraph, Bnz, Common);
    }
    else if (A->stype)
    {
        // Add the upper/lower part to a symmetric lower/upper matrix by
        // converting to unsymmetric mode
        // workspace: Iwork (nrow)
        B = CholmodApi<Int>::copy (A, 0, -1, Common) ;
    }
    else
    {
        // B = A*A' or A(:,f)*A(:,f)', no diagonal
        // workspace: Flag (nrow), Iwork (max (nrow,ncol))
        B = CholmodApi<Int>::aat (A, fset, fsize, -1, Common) ;
    }

    if (Common->status < CHOLMOD_OK)
    {
        return (EMPTY) ;
    }
    Bp = (Int *) B->p ;
    Bi = (Int *) B->i ;
    ASSERT ((Int) (B->nrow) == n && (Int) (B->ncol) == n) ;

    //--------------------------------------------------------------------------
    // initializations
    //--------------------------------------------------------------------------

    // All nodes start out live and unvisited.
    Common->mark = EMPTY ;
    clear_common_flag<Int> (Common) ;
    ASSERT (Flag == Common->Flag) ;

    for (j = 0 ; j < n ; j++)
    {
        CParent [j] = -2 ;
    }

    // prune dense nodes from B
    if (std::isnan (prune_dense) || prune_dense < 0)
    {
        // only remove completely dense nodes
        threshold = n-2 ;
    }
    else
    {
        // remove nodes with degree more than threshold
        threshold = (Int) (MAX (16, prune_dense * sqrt ((double) (n)))) ;
        threshold = MIN (n, threshold) ;
    }
    ndense = 0 ;
    cnode = EMPTY ;
    cdense = EMPTY ;

    bnz = 0;
    for (j = 0 ; j < n ; j++)
    {
        // FullGraph's copy has gaps and already supplied the live lengths.
        if (!FullGraph) Bnz [j] = Bp [j+1] - Bp [j] ;
        bnz += Bnz[j];
        if (Bnz[j] > threshold)
        {
            // node j is dense, prune it from B
            PRINT2 (("j is dense %d\n", j)) ;
            ndense++ ;
            if (cnode == EMPTY)
            {
                // first dense node found becomes root of this component,
                // which contains all of the dense nodes found here
                cdense = j ;
                cnode = j ;
                CParent [cnode] = EMPTY ;
            }
            Flag [j] = FLIP (cnode) ;
        }
    }
    csize = MAX (n, bnz) ;
    B->packed = FALSE ;
    ASSERT (B->nz == NULL) ;

    if (ndense == n)
    {
        // all nodes removed: Perm is identity, all nodes in component zero,
        // and the separator tree has just one node.
        PRINT2 (("all nodes are dense\n")) ;
        for (k = 0 ; k < n ; k++)
        {
            Perm [k] = k ;
            Cmember [k] = 0 ;
        }
        CParent [0] = EMPTY ;
        CholmodApi<Int>::free_sparse (&B, Common) ;
        Common->mark = EMPTY ;
        clear_common_flag<Int> (Common) ;
        if (NDSeconds) *NDSeconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - nesdis_start).count();
        return (1) ;
    }

    Bnw = (Int *) CholmodApi<Int>::malloc (n, sizeof (Int), Common) ;

    if (Common->status < CHOLMOD_OK)
    {
        // out of memory
        CholmodApi<Int>::free_sparse (&B, Common) ;
        CholmodApi<Int>::free (n, sizeof (Int), Bnw, Common) ;
        Common->mark = EMPTY ;
        clear_common_flag<Int> (Common) ;
        PRINT2 (("out of memory for Bnw\n")) ;
        return (EMPTY) ;
    }

    const auto partition_start = std::chrono::steady_clock::now();
    try
    {
    nesdis_local<Int> L (n, csize, Common) ;

    // create initial unit node weights
    for (j = 0 ; j < n ; j++)
    {
        Bnw [j] = 1 ;
    }
    // Discover initial component groups and retain their vertex lists.
    // workspace: Flag (nrow), Iwork (nrow); use Imap as workspace for Queue [
    auto components = find_components<Int> (B, NULL, n, cnode, NULL,
            Bnz, CParent,
            Flag, L.Imap, Common) ;
    // done using Imap as workspace for Queue ]

    nesdis_shared<Int> S (options) ;
    S.n = n ;
    S.csize = csize ;
    S.nd_compress = nd_compress ;
    S.nd_small = nd_small ;
    S.nd_oksep = nd_oksep ;
    S.B = B ;
    S.Bp = Bp ;
    S.Bi = Bi ;
    S.Bnz = Bnz ;
    S.Flag = Flag ;
    S.Bnw = Bnw ;
    S.CParent = CParent ;
    S.Common = Common ;
    S.SerialSubtreeSize = (Int)std::max<int64_t>(nd_small,
        std::min<int64_t>(options.serial_subtree_size, std::numeric_limits<Int>::max()));
    int max_parallel_depth = nesdis_max_parallel_depth<Int> (n, S.SerialSubtreeSize) ;
    if (max_parallel_depth > 0) {
        nesdis_process_parallel<Int> (&S, &L, std::move(components), max_parallel_depth) ;
    }
    else
    {
        nesdis_process_components<Int> (&S, &L, std::move(components), 0) ;
    }
    }
    catch (...)
    {
        CholmodApi<Int>::free_sparse (&B, Common) ;
        CholmodApi<Int>::free (n, sizeof (Int), Bnw, Common) ;
        Common->mark = EMPTY ;
        clear_common_flag<Int> (Common) ;
        throw ;
    }

    partition_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - partition_start).count();

    //--------------------------------------------------------------------------
    // place nodes removed via compression into their proper component
    //--------------------------------------------------------------------------

    // All vertices are removed; Flag links absorbed vertices to representatives.

    for (i = 0 ; i < n ; i++)
    {
        // find the repnode cnode that contains node i
        j = FLIP (Flag [i]) ;
        PRINT2 (("\nfind component for " ID ", in: " ID "\n", i, j)) ;
        ASSERT (j >= 0 && j < n) ;
        DEBUG (cnt = 0) ;
        while (CParent [j] == -2)
        {
            j = FLIP (Flag [j]) ;
            PRINT2 (("    walk up to " ID " ", j)) ;
            ASSERT (j >= 0 && j < n) ;
            PRINT2 ((" CParent " ID "\n", CParent [j])) ;
            ASSERT (cnt < n) ;
            DEBUG (cnt++) ;
        }
        cnode = j ;
        ASSERT (cnode >= 0 && cnode < n) ;
        ASSERT (CParent [cnode] >= EMPTY && CParent [cnode] < n) ;
        PRINT2 (("i " ID " is in component with cnode " ID "\n", i, cnode)) ;
        ASSERT (Flag [cnode] == FLIP (cnode)) ;

        // Mark all nodes along the path from i to cnode as being in the
        // component whos repnode is cnode.  Perform path compression.
        j = FLIP (Flag [i]) ;
        Flag [i] = FLIP (cnode) ;
        DEBUG (cnt = 0) ;
        while (CParent [j] == -2)
        {
            ASSERT (j >= 0 && j < n) ;
            jnext = FLIP (Flag [j]) ;
            PRINT2 (("    " ID " walk " ID " set cnode to " ID "\n", i, j, cnode)) ;
            ASSERT (cnt < n) ;
            DEBUG (cnt++) ;
            Flag [j] = FLIP (cnode) ;
            j = jnext ;
        }
    }

    // At this point, all nodes fall into Types 1 or 2, as defined above.

    #ifndef NDEBUG
    for (j = 0 ; j < n ; j++)
    {
        PRINT2 (("j %d CParent %d  ", j, CParent [j])) ;
        if (CParent [j] >= EMPTY && CParent [j] < n)
        {
            // case 1: j is a repnode of a component
            cnode = j ;
            PRINT2 ((" a repnode\n")) ;
        }
        else
        {
            // case 2: j is not a repnode of a component
            cnode = FLIP (Flag [j]) ;
            PRINT2 ((" repnode is %d\n", cnode)) ;
            ASSERT (cnode >= 0 && cnode < n) ;
            ASSERT (CParent [cnode] >= EMPTY && CParent [cnode] < n) ;
        }
        ASSERT (Flag [cnode] == FLIP (cnode)) ;
        // case 3 no longer holds
    }
    #endif

    //--------------------------------------------------------------------------
    // free workspace
    //--------------------------------------------------------------------------

    CholmodApi<Int>::free_sparse (&B, Common) ;
    CholmodApi<Int>::free (n, sizeof (Int), Bnw, Common) ;

    //--------------------------------------------------------------------------
    // handle dense nodes
    //--------------------------------------------------------------------------

    // The separator tree has nodes with either no children or two or more
    // children - with one exception.  There may exist a single root node with
    // exactly one child, which holds the dense rows/columns of the matrix.
    // Delete this node if it exists.

    if (ndense > 0)
    {
        ASSERT (CParent [cdense] == EMPTY) ;    // cdense has no parent
        // find the children of cdense
        nchild = 0 ;
        for (j = 0 ; j < n ; j++)
        {
            if (CParent [j] == cdense)
            {
                nchild++ ;
                child = j ;
            }
        }
        if (nchild == 1)
        {
            // the cdense node has just one child; merge the two nodes
            PRINT1 (("root has one child\n")) ;
            CParent [cdense] = -2 ;             // cdense is deleted
            CParent [child] = EMPTY ;           // child becomes a root
            for (j = 0 ; j < n ; j++)
            {
                if (Flag [j] == FLIP (cdense))
                {
                    // j is a dense node
                    PRINT1 (("dense %d\n", j)) ;
                    Flag [j] = FLIP (child) ;
                }
            }
        }
    }

    //--------------------------------------------------------------------------
    // postorder the components
    //--------------------------------------------------------------------------

    DEBUG (for (cnt = 0, j = 0 ; j < n ; j++) if (CParent [j] != -2) cnt++) ;

    // use Cmember as workspace for Post [
    Post = Cmember ;

    // cholmod_postorder uses Head and Iwork [0..2n].  It does not use Flag,
    // which here holds the mapping of nodes to repnodes.  It ignores all nodes
    // for which CParent [j] < -1, so it operates just on the repnodes.
    // workspace: Head (n), Iwork (2*n)
    ncomponents = CholmodApi<Int>::postorder (CParent, n, NULL, Post, Common) ;
    ASSERT (cnt == ncomponents) ;

    // use Iwork [0..n-1] as workspace for Ipost (
    Ipost = Iwork ;
    DEBUG (for (j = 0 ; j < n ; j++) Ipost [j] = EMPTY) ;

    // compute inverse postorder
    for (c = 0 ; c < ncomponents ; c++)
    {
        cnode = Post [c] ;
        ASSERT (cnode >= 0 && cnode < n) ;
        Ipost [cnode] = c ;
        ASSERT (Head [c] == EMPTY) ;
    }

    // adjust the parent array
    // Iwork [n..2n-1] used for NewParent [
    NewParent = Iwork + n ;
    for (c = 0 ; c < ncomponents ; c++)
    {
        parent = CParent [Post [c]] ;
        NewParent [c] = (parent == EMPTY) ? EMPTY : (Ipost [parent]) ;
    }
    for (c = 0 ; c < ncomponents ; c++)
    {
        CParent [c] = NewParent [c] ;
    }

    // Iwork [n..2n-1] no longer needed for NewParent ]
    // Cmember no longer needed for Post ]

    #ifndef NDEBUG
    // count the number of children of each node
    for (c = 0 ; c < ncomponents ; c++)
    {
        Cmember [c] = 0 ;
    }
    for (c = 0 ; c < ncomponents ; c++)
    {
        if (CParent [c] != EMPTY) Cmember [CParent [c]]++ ;
    }
    for (c = 0 ; c < ncomponents ; c++)
    {
        // a node is either a leaf, or has 2 or more children
        ASSERT (Cmember [c] == 0 || Cmember [c] >= 2) ;
    }
    #endif

    //--------------------------------------------------------------------------
    // place each node in its component
    //--------------------------------------------------------------------------

    for (j = 0 ; j < n ; j++)
    {
        // node j is in the cth component, whose repnode is cnode
        cnode = FLIP (Flag [j]) ;
        PRINT2 (("j " ID "  flag " ID " cnode " ID "\n",
                    j, Flag [j], FLIP (Flag [j]))) ;
        ASSERT (cnode >= 0 && cnode < n) ;
        c = Ipost [cnode] ;
        ASSERT (c >= 0 && c < ncomponents) ;
        Cmember [j] = c ;
    }

    // Flag no longer needed for the node-to-component mapping

    // done using Iwork [0..n-1] as workspace for Ipost )

    //--------------------------------------------------------------------------
    // clear the Flag array
    //--------------------------------------------------------------------------

    Common->mark = EMPTY ;
    clear_common_flag<Int> (Common) ;

    } // full ND; reused trees already provide postordered membership

    if (NDSeconds) *NDSeconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - nesdis_start).count();
    if (NDOnly) return ncomponents;

    //--------------------------------------------------------------------------
    // find the permutation
    //--------------------------------------------------------------------------

    PRINT1 (("nd_camd: %d A->stype %d\n", nd_camd, A->stype)) ;

    const bool camd_trace = options.camd.trace;
    if (options.trace_tree)
        trace_camd_tree((size_t)n, CParent, ncomponents, Cmember);

    if (nd_camd)
    {
        BENCHMARK_SCOPED_TIMER_SECTION t_camd("CAMD");

        //----------------------------------------------------------------------
        // apply camd, csymamd, or ccolamd using the Cmember constraints
        //----------------------------------------------------------------------

        if (A->stype != 0)
        {
            // ordering A+A', so fset and fsize are ignored.
            // Add the upper/lower part to a symmetric lower/upper matrix by
            // converting to unsymmetric mode
            // workspace: Iwork (nrow)
            cholmod_sparse original_graph;
            if (FullGraph) {
                original_graph = *FullGraph;
                B = &original_graph;
            }
            else B = CholmodApi<Int>::copy (A, 0, -1, Common) ;
            if (Common->status < CHOLMOD_OK)
            {
                PRINT0 (("make symmetric failed\n")) ;
                return (EMPTY) ;
            }
            ASSERT ((Int) (B->nrow) == n && (Int) (B->ncol) == n) ;
            PRINT2 (("nested dissection (2)\n")) ;
            bool hierarchical = false;
            auto camd_start = std::chrono::steady_clock::now();
            if (nd_camd == 1)
            {
                // B is the original graph (borrowed or copied): compression and
                // dense-node pruning affect membership, not this CAMD input.
                try {
                    hierarchical = options.camd.cut_depth >= 0 && ncomponents > 1;
                    if (hierarchical) {
                        auto result = hierarchical_camd(*B, CParent, ncomponents, Cmember, *Common, options.camd);
                        std::copy(result.permutation.begin(), result.permutation.end(), Perm);
                        // Local CAMD estimates omit boundary interactions; summing
                        // them would not estimate the full factor's nnz or flops.
                        Common->lnz = Common->fl = -1;
                        ok = TRUE;
                    }
                    else {
                        B->stype = -1;
                        ok = CholmodApi<Int>::camd(B, NULL, 0, Cmember, Perm, Common);
                    }
                }
                catch (...) {
                    if (!FullGraph) CholmodApi<Int>::free_sparse(&B, Common);
                    throw;
                }
            }
            else if (nd_camd == 2)
            {
                B->stype = -1 ;
                // workspace:  Head (nrow+1), Iwork (nrow) if symmetric-upper
                ok = CholmodApi<Int>::csymamd (B, Cmember, Perm, Common) ;
            }
            else
            {
                B->stype = -1 ;
                // workspace: Head (nrow), Iwork (4*nrow)
                ok = CholmodApi<Int>::camd (B, NULL, 0, Cmember, Perm, Common) ;
            }
            if (camd_trace)
                std::fprintf(stderr, "CAMD_TOTAL {\"hierarchical\":%s,\"vertices\":%lld,\"seconds\":%.9g,\"ok\":%d}\n",
                    hierarchical ? "true" : "false", (long long)n,
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - camd_start).count(), ok);
            if (!FullGraph) CholmodApi<Int>::free_sparse (&B, Common) ;
            if (!ok)
            {
                // failed
                PRINT0 (("camd/csymamd failed\n")) ;
                return (EMPTY) ;
            }
        }
        else
        {
            // ordering A*A' or A(:,f)*A(:,f)'
            // workspace: Iwork (nrow if no fset; MAX(nrow,ncol) if fset)
            if (!CholmodApi<Int>::ccolamd (A, fset, fsize, Cmember, Perm, Common))
            {
                // ccolamd failed
                PRINT2 (("ccolamd failed\n")) ;
                return (EMPTY) ;
            }
        }

    }
    else
    {

        //----------------------------------------------------------------------
        // natural ordering of each component
        //----------------------------------------------------------------------

        // use Iwork [0..n-1] for Next [
        Next = Iwork  ;

        //----------------------------------------------------------------------
        // place the nodes in link lists, one list per component
        //----------------------------------------------------------------------

        // do so in reverse order, to preserve original ordering
        for (j = n-1 ; j >= 0 ; j--)
        {
            // node j is in the cth component
            c = Cmember [j] ;
            ASSERT (c >= 0 && c < ncomponents) ;
            // place node j in link list for component c
            Next [j] = Head [c] ;
            Head [c] = j ;
        }

        //----------------------------------------------------------------------
        // order each node in each component
        //----------------------------------------------------------------------

        k = 0 ;
        for (c = 0 ; c < ncomponents ; c++)
        {
            for (j = Head [c] ; j != EMPTY ; j = Next [j])
            {
                Perm [k++] = j ;
            }
            Head [c] = EMPTY ;
        }
        ASSERT (k == n) ;

        // done using Iwork [0..n-1] for Next ]
    }

    //--------------------------------------------------------------------------
    // clear workspace and return number of components
    //--------------------------------------------------------------------------

    if (camd_trace)
        std::fprintf(stderr, "NESDIS_TOTAL {\"seconds\":%.9g,\"partition_seconds\":%.9g}\n",
            std::chrono::duration<double>(std::chrono::steady_clock::now() - nesdis_start).count(), partition_seconds);
    return (ncomponents) ;
}



// A fresh CHOLMOD context is required for each concurrently repaired graph;
// copying Common would alias its Flag/Head/Iwork storage.
template<class Int>
struct SubtreeCommon {
    cholmod_common value;
    explicit SubtreeCommon(const cholmod_common &source) {
        if constexpr (sizeof(Int) == 4) cholmod_start(&value);
        else cholmod_l_start(&value);
        value.current = 0;
        value.method[0] = source.method[source.current];
        value.error_handler = source.error_handler;
        value.print = source.print;
        value.metis_memory = source.metis_memory;
        value.metis_dswitch = source.metis_dswitch;
        value.metis_nswitch = source.metis_nswitch;
    }
    ~SubtreeCommon() {
        if constexpr (sizeof(Int) == 4) cholmod_finish(&value);
        else cholmod_l_finish(&value);
    }
};

template<class Int>
static PersistentSeparatorTree<Int> compute_nd_subtree(const cholmod_sparse &graph,
    const std::vector<Int> &vertices, const std::vector<Int> &local_index,
    const cholmod_common &common, const NesdisOptions &options)
{
    const auto *p = static_cast<const Int *>(graph.p);
    const auto *i = static_cast<const Int *>(graph.i);
    const size_t n = vertices.size();
    std::vector<Int> sub_p(n + 1), sub_i;
    for (size_t v = 0; v < n; ++v) {
        for (Int k = p[vertices[v]]; k < p[vertices[v] + 1]; ++k) {
            Int local = local_index[i[k]];
            // local_index covers all dirty regions. Test the inverse map too,
            // excluding ancestor separators, other regions, and diagonals.
            if (local >= 0 && size_t(local) < n && local != Int(v) && vertices[local] == i[k])
                sub_i.push_back(local);
        }
        sub_p[v + 1] = Int(sub_i.size());
        // The explicit graph API is sorted, but cholmod_aat can produce
        // unsorted columns when the stateful matrix API orders A*A'.
        if (!graph.sorted) std::sort(sub_i.begin() + sub_p[v], sub_i.end());
    }
    cholmod_sparse induced{};
    induced.nrow = induced.ncol = n;
    induced.nzmax = sub_i.size();
    if (sub_i.empty()) sub_i.push_back(0);
    induced.p = sub_p.data(); induced.i = sub_i.data();
    induced.packed = induced.sorted = 1;
    induced.itype = graph.itype;
    induced.xtype = CHOLMOD_PATTERN;
    SubtreeCommon<Int> local(common);
    std::vector<Int> perm(n), parent(n), member(n);
    auto A = induced;
    A.stype = 1;
    auto nc = nested_dissection_impl<Int>(&A, nullptr, 0, perm.data(), parent.data(),
        member.data(), &local.value, options, &induced, nullptr, true);
    if (nc < 0) throw nesdis_failure(local.value.status);
    return PersistentSeparatorTree<Int>::from_cholmod(n, Int(nc), parent.data(), member.data());
}

static std::string partition_settings(const cholmod_common &common, const NesdisOptions &options) {
    const auto &m = common.method[common.current];
    std::ostringstream key;
    key << std::setprecision(17) << m.prune_dense << ' ' << m.nd_compress << ' '
        << m.nd_oksep << ' ' << m.nd_small << ' ' << m.nd_components << ' '
        << options.scotch_levels << ' ' << options.scotch_threads << ' '
        << options.scotch_seed << ' ' << options.scotch_strategy;
    return key.str();
}

// Stateful driver with the same matrix/FullGraph and workspace contracts as
// nested_dissection_impl. The caller owns reuse exclusively, enables a positive
// reuse period, and preserves vertex identities across analyses (reset after
// relabeling). Missing/incompatible history, period expiry, or a global separator
// violation triggers full ND; otherwise repair only dirty subtrees with NDOnly.
// In either case, run final constrained ordering on the current graph; repaired
// or unchanged forests are flattened through ReusedTree without repeating ND.
// Commit history only after success; an empty input clears the history.
template<class Int>
static int64_t nested_dissection_temporal(cholmod_sparse *A, Int *fset, size_t fsize,
    Int *Perm, Int *CParent, Int *Cmember, cholmod_common *Common,
    const NesdisOptions &options, const cholmod_sparse *FullGraph, TemporalReuseState<Int> &reuse)
{
    BENCHMARK_SCOPED_TIMER_SECTION tfull("nested_dissection_temporal");
    if (!A || !Perm || !CParent || !Cmember) {
        Common->status = CHOLMOD_INVALID;
        return EMPTY;
    }
    if (A->nrow == 0) {
        reuse.reset();
        Common->status = CHOLMOD_OK;
        return 1;
    }
    Common->status = CHOLMOD_OK;
    const auto start = std::chrono::steady_clock::now();
    auto settings = partition_settings(*Common, options);
    const bool have_tree = !reuse.tree.roots.empty() && reuse.tree.member.size() == A->nrow;
    bool full = !have_tree || settings != reuse.partition_settings ||
        reuse.incremental_analyses_since_rebuild >= reuse.temporal_reuse_period;
    // Own the converted graph only when the caller did not supply a full graph.
    auto deleter = [Common](cholmod_sparse *g) { CholmodApi<Int>::free_sparse(&g, Common); };
    std::unique_ptr<cholmod_sparse, decltype(deleter)> owned(nullptr, deleter);
    const cholmod_sparse *graph = FullGraph;
    typename PersistentSeparatorTree<Int>::DirtySubtrees dirty;
    if (!full) {
        if (!graph) {
            owned.reset(A->stype ? CholmodApi<Int>::copy(A, 0, -1, Common)
                                : CholmodApi<Int>::aat(A, fset, fsize, -1, Common));
            if (!owned || Common->status < CHOLMOD_OK) return EMPTY;
            graph = owned.get();
        }
        dirty = reuse.tree.dirty_subtrees(static_cast<const Int *>(graph->p),
                                          static_cast<const Int *>(graph->i));
        full = dirty.full_rebuild;
    }
    TemporalReuseStatistics statistics;
    statistics.full_rebuild = full;
    statistics.graph_vertices = A->nrow;
    PersistentSeparatorTree<Int> candidate;
    int64_t nc;
    double ordering_nd_seconds = 0;
    if (full) {
        BENCHMARK_SCOPED_TIMER_SECTION tfull_recompute("full ND recompute");
        const double preparation_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start).count();
        {
            BENCHMARK_SCOPED_TIMER_SECTION timer("nested_dissection_impl");
            nc = nested_dissection_impl<Int>(A, fset, fsize, Perm, CParent, Cmember,
                Common, options, FullGraph, nullptr, false, &ordering_nd_seconds);
        }
        if (nc < 0) return nc;
        const auto capture_start = std::chrono::steady_clock::now();
        candidate = PersistentSeparatorTree<Int>::from_cholmod(A->nrow, Int(nc), CParent, Cmember);
        statistics.repartitioned_vertices = A->nrow;
        statistics.nd_seconds = preparation_seconds + ordering_nd_seconds +
            std::chrono::duration<double>(std::chrono::steady_clock::now() - capture_start).count();
    }
    else {
        std::vector<std::vector<Int>> regions;
        if (!dirty.roots.empty()) {
            std::vector<Int> local_index;
            {
                BENCHMARK_SCOPED_TIMER_SECTION timer("temporal ND region construction");
                typename PersistentSeparatorTree<Int>::SubtreeRegionLabels region_labels;
                {
                    // BENCHMARK_SCOPED_TIMER_SECTION label_timer("label and reserve regions");
                    region_labels = reuse.tree.label_subtree_regions(dirty.roots, regions);
                }
                {
                    // BENCHMARK_SCOPED_TIMER_SECTION gather_timer("gather vertices and inverse map");
                    reuse.tree.gather_subtree_regions(region_labels, regions, local_index);
                }
                for (const auto &region : regions) statistics.repartitioned_vertices += region.size();
            }
            std::vector<PersistentSeparatorTree<Int>> replacements;
            {
                // Keep the global section timer on the calling thread; worker
                // tasks must not mutate the shared benchmark section stack.
                BENCHMARK_SCOPED_TIMER_SECTION timer("temporal ND region recomputation");
                replacements.resize(regions.size());
                tbb::parallel_for(size_t(0), regions.size(), [&](size_t r) {
                    auto subtree_options = options;
                    // Scotch's level limit refers to depth in the persistent tree,
                    // while recursion on the induced graph starts at depth zero.
                    subtree_options.scotch_levels = int(std::max<int64_t>(0,
                        int64_t(options.scotch_levels) - reuse.tree.depth[dirty.roots[r]]));
                    replacements[r] = compute_nd_subtree<Int>(*graph, regions[r], local_index, *Common, subtree_options);
                });
            }
            {
                BENCHMARK_SCOPED_TIMER_SECTION timer("temporal ND region splicing");
                candidate = reuse.tree;
                for (size_t r = 0; r < regions.size(); ++r)
                    candidate.replace(dirty.roots[r], replacements[r], regions[r]);
                candidate.update_depths();
            }
        }
        statistics.recomputed_subtrees = regions.size();
        const auto *tree = regions.empty() ? &reuse.tree : &candidate;
        const double repair_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start).count();
        {
            BENCHMARK_SCOPED_TIMER_SECTION timer("nested_dissection_impl");
            nc = nested_dissection_impl<Int>(A, fset, fsize, Perm, CParent, Cmember,
                Common, options, FullGraph, tree, false, &ordering_nd_seconds);
        }
        if (nc < 0) return nc;
        statistics.nd_seconds = repair_seconds + ordering_nd_seconds;
    }
    // Commit only after the final CAMD/permutation stage has succeeded.
    if (full || !dirty.roots.empty()) reuse.tree = std::move(candidate);
    reuse.incremental_analyses_since_rebuild = full ? 0 : reuse.incremental_analyses_since_rebuild + 1;
    reuse.partition_settings = std::move(settings);
    reuse.statistics = statistics;
    return nc;
}

template<class Int>
static int64_t nested_dissection_checked(cholmod_sparse *A, Int *fset, size_t fsize,
                                         Int *Perm, Int *CParent, Int *Cmember,
                                         cholmod_common *Common,
                                         const cholmod_sparse *FullGraph = nullptr,
                                         TemporalReuseState<Int> *reuse = nullptr) noexcept {
    if (!Common) return EMPTY;
    // The C and full-graph APIs share one exception boundary, including optional
    // diagnostics. Internal catches only clean up owned storage before rethrowing.
    try {
        const NesdisOptions options;
        std::unique_ptr<tbb::global_control> thread_limit;
        if (options.threads) thread_limit = std::make_unique<tbb::global_control>(
            tbb::global_control::max_allowed_parallelism, options.threads);
        if (reuse && reuse->temporal_reuse_period)
            return nested_dissection_temporal<Int>(A, fset, fsize, Perm, CParent, Cmember,
                                                   Common, options, FullGraph, *reuse);
        // Only time caller-thread invocations: ND-only subtree calls run in
        // parallel, while benchmark sections share a global section stack.
        BENCHMARK_SCOPED_TIMER_SECTION timer("nested_dissection_impl");
        auto result = nested_dissection_impl<Int>(A, fset, fsize, Perm, CParent, Cmember, Common, options, FullGraph);
        if (reuse && result >= 0) reuse->reset();
        return result;
    }
    catch (const nesdis_failure &failure) {
        Common->status = failure.status < CHOLMOD_OK ? failure.status : CHOLMOD_INVALID;
    }
    catch (const std::bad_alloc &) { Common->status = CHOLMOD_OUT_OF_MEMORY; }
    catch (const std::exception &e) {
        Common->status = CHOLMOD_INVALID;
        std::fprintf(stderr, "nested dissection: %s\n", e.what());
    }
    catch (...) { Common->status = CHOLMOD_INVALID; }
    return EMPTY;
}

template<class Int>
int64_t nested_dissection_from_graph(const cholmod_sparse &graph,
    Int *Perm, Int *CParent, Int *Cmember, cholmod_common *Common, TemporalReuseState<Int> *reuse)
{
    if (!Common) return EMPTY;
    const int itype = sizeof(Int) == sizeof(int32_t) ? CHOLMOD_INT : CHOLMOD_LONG;
    if (graph.nrow != graph.ncol || graph.nrow > (size_t)std::numeric_limits<Int>::max() ||
        graph.stype != 0 || !graph.packed || !graph.sorted ||
        graph.itype != itype || Common->itype != itype || !graph.p ||
        (!graph.i && graph.nzmax != 0)) {
        Common->status = CHOLMOD_INVALID;
        return EMPTY;
    }
    // Only this explicit graph API bypasses A*A' semantics for stype == 0.
    auto A = graph;
    A.stype = 1;
    return nested_dissection_checked<Int>(&A, nullptr, 0, Perm, CParent, Cmember, Common, &graph, reuse);
}

template int64_t nested_dissection_from_graph<int32_t>(const cholmod_sparse &,
    int32_t *, int32_t *, int32_t *, cholmod_common *, TemporalReuseState<int32_t> *);
template int64_t nested_dissection_from_graph<int64_t>(const cholmod_sparse &,
    int64_t *, int64_t *, int64_t *, cholmod_common *, TemporalReuseState<int64_t> *);

template<class Int>
int64_t nested_dissection(cholmod_sparse *A, Int *fset, size_t fsize,
    Int *Perm, Int *CParent, Int *Cmember, cholmod_common *Common, TemporalReuseState<Int> &reuse) {
    return nested_dissection_checked<Int>(A, fset, fsize, Perm, CParent, Cmember, Common, nullptr, &reuse);
}
template int64_t nested_dissection<int32_t>(cholmod_sparse *, int32_t *, size_t,
    int32_t *, int32_t *, int32_t *, cholmod_common *, TemporalReuseState<int32_t> &);
template int64_t nested_dissection<int64_t>(cholmod_sparse *, int64_t *, size_t,
    int64_t *, int64_t *, int64_t *, cholmod_common *, TemporalReuseState<int64_t> &);

} // namespace MeshFEM::CholmodParallelNesdis

extern "C" int64_t cholmod_nested_dissection_parallel(cholmod_sparse *A, int32_t *fset,
                                                       size_t fsize, int32_t *Perm,
                                                       int32_t *CParent, int32_t *Cmember,
                                                       cholmod_common *Common) {
    return MeshFEM::CholmodParallelNesdis::nested_dissection_checked<int32_t>(
        A, fset, fsize, Perm, CParent, Cmember, Common);
}

extern "C" int64_t cholmod_l_nested_dissection_parallel(cholmod_sparse *A, int64_t *fset,
                                                         size_t fsize, int64_t *Perm,
                                                         int64_t *CParent, int64_t *Cmember,
                                                         cholmod_common *Common) {
    return MeshFEM::CholmodParallelNesdis::nested_dissection_checked<int64_t>(
        A, fset, fsize, Perm, CParent, Cmember, Common);
}
