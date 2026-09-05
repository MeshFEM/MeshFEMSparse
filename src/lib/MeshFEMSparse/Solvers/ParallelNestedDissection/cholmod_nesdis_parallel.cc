#include "cholmod_internal_excerpts.hh"

#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include <tbb/enumerable_thread_specific.h>
#include <tbb/global_control.h>
#include <tbb/task_group.h>

#define CHOLMOD_NESDIS_USE_THREADS 1

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
    ~nesdis_local() { release (); }

    void release () ;

    Int n ;
    Int csize ;
    cholmod_sparse *C ;

    Int *Imap ;
    Int *Map ;
    Int *Mark ;
    Int mark ;
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
#ifdef CHOLMOD_NESDIS_USE_THREADS
    int UseParallel ;
    int MaxParallelDepth ;
    tbb::enumerable_thread_specific<std::unique_ptr<nesdis_local<Int>>> LocalWorkspaces ;
#endif
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
    Cew  = (Int *) CholmodApi<Int>::malloc (csize, sizeof (Int), Common) ;
    WorkLocal = (Int *) CholmodApi<Int>::malloc (n, 7*sizeof (Int), Common) ;

    if (Common->status < CHOLMOD_OK)
    {
        int status = Common->status ;
        release () ;
        throw nesdis_failure (status) ;
    }

    Part = WorkLocal ;
    Cnw  = Part + n ;
    Mark = Cnw + n ;
    Map  = Mark + n ;
    Imap = Map + n ;
    Hash = Imap + n ;
    Cmap = Hash + n ;
    Cp = (Int *) C->p ;
    Ci = (Int *) C->i ;
    mark = 0 ;

    for (j = 0 ; j < n ; j++)
    {
        Mark [j] = EMPTY ;
    }
    for (j = 0 ; j < csize ; j++)
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
    CholmodApi<Int>::free (7*n, sizeof (Int), WorkLocal, Common) ;
    Cew = NULL ;
    WorkLocal = NULL ;
    Part = NULL ;
    Cnw = NULL ;
    Mark = NULL ;
    Map = NULL ;
    Imap = NULL ;
    Hash = NULL ;
    Cmap = NULL ;
    Cp = NULL ;
    Ci = NULL ;
}

//------------------------------------------------------------------------------
// nesdis threaded helpers
//------------------------------------------------------------------------------

template<class Int>
static void nesdis_process_recursive
(
    nesdis_shared<Int> *S,
    nesdis_local<Int> *L,
    Int Cstack [ ],
    Int *top,
    Int depth
);

#ifdef CHOLMOD_NESDIS_USE_THREADS
template<class Int, class F>
static void foreach_group(const Int *ChildStack, const Int child_top, const F &f) {
    Int group_end = child_top;
    while (group_end >= 0) {
        Int group_start = group_end;
        while (group_start >= 0 && ChildStack [group_start] >= 0) group_start--;
        ASSERT (group_start >= 0);
        f(group_start, group_end);
        group_end = group_start - 1 ;
    }
}

template<class Int>
static void nesdis_process_child_groups_parallel
(
    nesdis_shared<Int> *S,
    nesdis_local<Int> *L,
    Int *ChildStack,
    Int child_top,
    Int depth
)
{
    auto run_sequential = [ChildStack, S, L, depth](Int group_start, Int group_end) {
        Int local_top = group_end - group_start;
        nesdis_process_recursive<Int>(S, L, ChildStack + group_start, &local_top, depth);
    };

    if (depth >= S->MaxParallelDepth) {
        return foreach_group<Int>(ChildStack, child_top, run_sequential);
        return;
    }

    // We run the first group sequentially on this thread after launching the other groups in parallel.
    std::pair<Int, Int> first_group;
    bool has_first_group = false;

    foreach_group<Int>(ChildStack, child_top, [ChildStack, S, depth, &has_first_group, &first_group](Int group_start, Int group_end) {
        if (!has_first_group) {
            first_group = {group_start, group_end};
            has_first_group = true;
            return;
        }
        // Note: we need to copy the stack now rather than at the start of the
        // task. Otherwise the `ChildStack` held by the parent call frame
        // may be destroyed by the time the task actually runs.
        auto task_stack_ptr = std::make_shared<std::vector<Int>>(ChildStack + group_start,
                                                                 ChildStack + group_end + 1);
        S->TaskGroup.run([S, task_stack_ptr, depth]() {
            Int local_top = task_stack_ptr->size() - 1;
            auto &Local = S->LocalWorkspaces.local();
            if (!Local) Local = std::make_unique<nesdis_local<Int>> (S->n, S->csize, S->Common);
            nesdis_process_recursive<Int>(S, Local.get(), task_stack_ptr->data(), &local_top, depth);
        });
    });

    if (has_first_group)
        run_sequential(first_group.first, first_group.second);
}

static int nesdis_requested_num_threads
(
)
{
    const char *env_threads = std::getenv ("CHOLMOD_NESDIS_NUM_THREADS") ;
    if (env_threads != NULL)
    {
        int requested_threads = std::atoi (env_threads) ;
        if (requested_threads > 0)
        {
            return requested_threads ;
        }
    }
    return 0 ;
}

template<class Int>
static Int nesdis_serial_subtree_size
(
    Int nd_small
)
{
    Int target_size = 2000 ;

    const char *env_size = std::getenv ("CHOLMOD_NESDIS_SERIAL_SUBTREE_SIZE") ;
    if (env_size != NULL)
    {
        long requested_size = std::atol (env_size) ;
        if (requested_size > 0)
        {
            target_size = (Int) requested_size ;
        }
    }

    return MAX (target_size, nd_small) ;
}

template<class Int>
static int nesdis_max_parallel_depth
(
    Int n,
    Int nd_small
)
{
    Int target_size = nesdis_serial_subtree_size<Int> (nd_small) ;
    if (n <= target_size)
    {
        return 0 ;
    }

    int depth = 0 ;
    for (Int subtree_size = n ; subtree_size > target_size ; subtree_size = (subtree_size + 1) / 2)
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
    Int Cstack [ ],
    Int *top,
    int max_parallel_depth
)
{
    S->UseParallel = TRUE ;
    S->MaxParallelDepth = max_parallel_depth ;

    try {
        nesdis_process_recursive<Int> (S, MainLocal, Cstack, top, 0) ;
    }
    catch (...) {
        S->TaskGroup.cancel() ;
        S->TaskGroup.wait() ;
        throw ;
    }
    S->TaskGroup.wait();
}

#endif

//------------------------------------------------------------------------------
// nesdis_process_recursive
//------------------------------------------------------------------------------


template<class Int>
static void nesdis_process_recursive
(
    nesdis_shared<Int> *S,
    nesdis_local<Int> *L,
    Int Cstack [ ],
    Int *top,
    Int depth
)
{
    Int *Bp = S->Bp ;
    Int *Bi = S->Bi ;
    Int *Bnz = S->Bnz ;
    Int *Imap = L->Imap ;
    Int *Map = L->Map ;
    Int *Mark = L->Mark ;
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

    using UInt = std::make_unsigned_t<Int> ;
    Int cnode, cn, mark, i, j, cj, ci, cnz, pstart, pdest, pend, p,
        total_weight, sepsize, parent, child_top ;
    UInt hash ;
    DEBUG (Int cnt) ;

    while (*top >= 0)
    {

        //----------------------------------------------------------------------
        // get node(s) from the top of the Cstack
        //----------------------------------------------------------------------

        mark = local_clear_mark<Int> (NULL, 0, Mark, &(L->mark), S->n) ;
        DEBUG (for (i = 0 ; i < S->n ; i++) Imap [i] = EMPTY) ;

        cnode = EMPTY ;
        cn = 0 ;
        while (cnode == EMPTY)
        {
            i = Cstack [(*top)--] ;
            Int raw_i = i ;

            if (i < 0)
            {
                i = FLIP (i) ;
                cnode = i ;
            }

            if (i < 0 || i >= S->n || Flag [i] < EMPTY)
            {
                fprintf (stderr,
                    "CHOLMOD_NESDIS_BAD_POP depth=%lld top_after=%lld "
                    "raw=%lld node=%lld n=%lld flag=%lld\n",
                    (long long) depth, (long long) *top,
                    (long long) raw_i, (long long) i, (long long) S->n,
                    (long long) ((i >= 0 && i < S->n) ? Flag [i] : EMPTY - 1)) ;
                throw nesdis_failure (CHOLMOD_INVALID) ;
            }

            ASSERT (i >= 0 && i < S->n && Flag [i] >= EMPTY) ;

            Map [cn] = i ;
            Mark [i] = mark ;
            Imap [i] = cn ;
            cn++ ;
        }

        ASSERT (cnode != EMPTY) ;

        //----------------------------------------------------------------------
        // create the subgraph for this connected component C
        //----------------------------------------------------------------------

        cnz = 0 ;
        total_weight = 0 ;
        for (cj = 0 ; cj < cn ; cj++)
        {
            j = Map [cj] ;
            ASSERT (Mark [j] == mark) ;
            Cp [cj] = cnz ;
            Cnw [cj] = Bnw [j] ;
            ASSERT (Cnw [cj] >= 0) ;
            total_weight += Cnw [cj] ;
            pstart = Bp [j] ;
            pdest = pstart ;
            pend = pstart + Bnz [j] ;
            hash = cj ;
            for (p = pstart ; p < pend ; p++)
            {
                i = Bi [p] ;
                if (i != j && Flag [i] >= EMPTY)
                {
                    Bi [pdest++] = i ;
                    if (Mark [i] != mark)
                    {
                        Map [cn] = i ;
                        Mark [i] = mark ;
                        Imap [i] = cn ;
                        cn++ ;
                    }
                    ci = Imap [i] ;
                    ASSERT (ci >= 0 && ci < cn && ci != cj && cnz < S->csize) ;
                    Ci [cnz++] = ci ;
                    hash += ci ;
                }
            }
            Bnz [j] = pdest - pstart ;
            hash %= S->csize ;
            Hash [cj] = (Int) hash ;
            ASSERT (Hash [cj] >= 0 && Hash [cj] < S->csize) ;
        }
        Cp [cn] = cnz ;
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

        PRINT0 (("consider cn %d nd_small %d ", cn, S->nd_small)) ;
        if (cn < S->nd_small)
        {
            PRINT0 ((" too small\n")) ;
            sepsize = total_weight ;
        }
        else
        {
            PRINT0 ((" cut\n")) ;

            sepsize = partition<Int> (
                #ifndef NDEBUG
                S->csize,
                #endif
                S->nd_compress, depth, Hash, C, Cnw, Cew,
                Cmap, Part, Common) ;

            if (sepsize < 0)
            {
                C->ncol = S->n ;
                throw nesdis_failure (Common->status) ;
            }

            for (ci = 0 ; ci < cn ; ci++)
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

            DEBUG (for (cnt = 0, j = 0 ; j < S->n ; j++) cnt += Bnw [j]) ;
            ASSERT (cnt == S->n) ;
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

#ifdef CHOLMOD_NESDIS_USE_THREADS
            if (S->UseParallel)
            {
                child_top = EMPTY ;
                std::vector<Int> ChildStackStorage ((size_t) cn) ;
                Int *ChildStack = ChildStackStorage.data () ;
                find_components<Int> (B, Map, cn, cnode, Part, Bnz,
                        CParent, ChildStack, &child_top,
                        Flag, Mark, &(L->mark), Imap, Common) ;
                nesdis_process_child_groups_parallel<Int> (S, L, ChildStack,
                        child_top, depth + 1) ;
                continue ;
            }
#endif
            child_top = EMPTY ;
            std::vector<Int> ChildStackStorage ((size_t) cn) ;
            Int *ChildStack = ChildStackStorage.data () ;
            find_components<Int> (B, Map, cn, cnode, Part, Bnz,
                    CParent, ChildStack, &child_top,
                    Flag, Mark, &(L->mark), Imap, Common) ;
            nesdis_process_recursive<Int> (S, L, ChildStack, &child_top,
                    depth + 1) ;
        }
    }
}


//------------------------------------------------------------------------------
// cholmod_nested_dissection
//------------------------------------------------------------------------------

// This method uses a node bisector, applied recursively (but using a
// non-recursive algorithm).  Once the graph is partitioned, it calls a
// constrained min degree code (CAMD or CSYMAMD for A+A', and CCOLAMD for A*A')
// to order all the nodes in the graph - but obeying the constraints determined
// by the separators.  This routine is similar to METIS_NodeND, except for how
// it treats the leaf nodes.  METIS_NodeND orders the leaves of the separator
// tree with MMD, ignoring the rest of the matrix when ordering a single leaf.
// This routine orders the whole matrix with CSYMAMD or CCOLAMD, all at once,
// when the graph partitioning is done.
//
// This function also returns a postorderd separator tree (CParent), and a
// mapping of nodes in the graph to nodes in the separator tree (Cmember).
//
// workspace: Flag (nrow), Head (nrow+1), Iwork (4*nrow + (ncol if unsymmetric))
//      Allocates a temporary matrix B=A*A' or B=A,
//      and O(nnz(A)) temporary memory space.
//      Allocates an additional 3*n*sizeof(Int) temporary workspace

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
    cholmod_common *Common
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------


    double prune_dense, nd_oksep ;
    using UInt = std::make_unsigned_t<Int> ;
    Int *Bp, *Bi, *Bnz, *Cstack, *Flag, *Head, *Next, *Bnw, *Iwork,
        *Ipost, *NewParent, *Post ;
    UInt hash ;
    Int n, bnz, top, i, j, k, cnode, cdense, p, cj, cn, ci, cnz, mark, c,
        sepsize, parent, ncomponents, threshold, ndense, pstart, pdest, pend,
        nd_compress, nd_camd, csize, jnext, nd_small, total_weight,
        nchild, local_mark, child = EMPTY ;
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

    Cstack = Perm ;             // size n, use Perm as workspace for Cstack [

    if (Common->status < CHOLMOD_OK)
    {
        return (EMPTY) ;
    }

    //--------------------------------------------------------------------------
    // convert B to symmetric form with both upper/lower parts present
    //--------------------------------------------------------------------------

    // B = A+A', A*A', or A(:,f)*A(:,f)', upper and lower parts present

    if (A->stype)
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
    bnz = CholmodApi<Int>::nnz (B, Common) ;
    ASSERT ((Int) (B->nrow) == n && (Int) (B->ncol) == n) ;
    csize = MAX (n, bnz) ;

    //--------------------------------------------------------------------------
    // initializations
    //--------------------------------------------------------------------------

    // all nodes start out unmarked and unordered (Type 4, see below)
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

    for (j = 0 ; j < n ; j++)
    {
        Bnz [j] = Bp [j+1] - Bp [j] ;
        if (Bnz [j] > threshold)
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

    try
    {
    nesdis_local<Int> L (n, csize, Common) ;

    // create initial unit node and edge weights
    for (j = 0 ; j < n ; j++)
    {
        Bnw [j] = 1 ;
    }
    for (p = 0 ; p < csize ; p++)
    {
        L.Cew [p] = 1 ;
    }

    // push the initial connnected components of B onto the Cstack
    top = EMPTY ;       // Cstack is empty
    local_mark = L.mark ;
    // workspace: Flag (nrow), Iwork (nrow); use Imap as workspace for Queue [
    find_components<Int> (B, NULL, n, cnode, NULL,
            Bnz, CParent, Cstack, &top,
            Flag, L.Mark, &local_mark, L.Imap, Common) ;
    L.mark = local_mark ;
    // done using Imap as workspace for Queue ]

    // Nodes can now be of Type 0, 1, 2, or 4 (see definition below)

    nesdis_shared<Int> S ;
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
#ifdef CHOLMOD_NESDIS_USE_THREADS
    S.UseParallel = FALSE ;
    S.MaxParallelDepth = 0 ;
#endif

#ifdef CHOLMOD_NESDIS_USE_THREADS
    int max_parallel_depth = nesdis_max_parallel_depth<Int> (n, nd_small) ;
    if (max_parallel_depth > 0) {
        std::unique_ptr<tbb::global_control> thread_limit ;
        int requested_nthreads = nesdis_requested_num_threads () ;
        if (requested_nthreads > 0)
            thread_limit = std::make_unique<tbb::global_control> (
                    tbb::global_control::max_allowed_parallelism, requested_nthreads) ;

        nesdis_process_parallel<Int> (&S, &L, Cstack, &top, max_parallel_depth) ;
    }
    else
#endif
    {
        nesdis_process_recursive<Int> (&S, &L, Cstack, &top, 0) ;
    }
    }
    catch (nesdis_failure &failure)
    {
        Common->status = (failure.status < CHOLMOD_OK) ?
            failure.status : CHOLMOD_INVALID ;
        CholmodApi<Int>::free_sparse (&B, Common) ;
        CholmodApi<Int>::free (n, sizeof (Int), Bnw, Common) ;
        Common->mark = EMPTY ;
        clear_common_flag<Int> (Common) ;
            PRINT2 (("nested dissection workspace allocation failed\n")) ;
        return (EMPTY) ;
    }
    catch (...)
    {
        Common->status = CHOLMOD_OUT_OF_MEMORY ;
        CholmodApi<Int>::free_sparse (&B, Common) ;
        CholmodApi<Int>::free (n, sizeof (Int), Bnw, Common) ;
        Common->mark = EMPTY ;
        clear_common_flag<Int> (Common) ;
            PRINT2 (("nested dissection failed with C++ exception\n")) ;
        return (EMPTY) ;
    }

    // done using Perm as workspace for Cstack ]

    //--------------------------------------------------------------------------
    // place nodes removed via compression into their proper component
    //--------------------------------------------------------------------------

    // At this point, all nodes are of Type 1, 2, or 3, as defined above.

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

    //--------------------------------------------------------------------------
    // find the permutation
    //--------------------------------------------------------------------------

    PRINT1 (("nd_camd: %d A->stype %d\n", nd_camd, A->stype)) ;

    if (nd_camd)
    {

        //----------------------------------------------------------------------
        // apply camd, csymamd, or ccolamd using the Cmember constraints
        //----------------------------------------------------------------------

        if (A->stype != 0)
        {
            // ordering A+A', so fset and fsize are ignored.
            // Add the upper/lower part to a symmetric lower/upper matrix by
            // converting to unsymmetric mode
            // workspace: Iwork (nrow)
            B = CholmodApi<Int>::copy (A, 0, -1, Common) ;
            if (Common->status < CHOLMOD_OK)
            {
                PRINT0 (("make symmetric failed\n")) ;
                return (EMPTY) ;
            }
            ASSERT ((Int) (B->nrow) == n && (Int) (B->ncol) == n) ;
            PRINT2 (("nested dissection (2)\n")) ;
            B->stype = -1 ;
            if (nd_camd == 2)
            {
                // workspace:  Head (nrow+1), Iwork (nrow) if symmetric-upper
                ok = CholmodApi<Int>::csymamd (B, Cmember, Perm, Common) ;
            }
            else
            {
                // workspace: Head (nrow), Iwork (4*nrow)
                ok = CholmodApi<Int>::camd (B, NULL, 0, Cmember, Perm, Common) ;
            }
            CholmodApi<Int>::free_sparse (&B, Common) ;
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

    return (ncomponents) ;
}



template<class Int>
static int64_t nested_dissection_checked(cholmod_sparse *A, Int *fset, size_t fsize,
                                         Int *Perm, Int *CParent, Int *Cmember,
                                         cholmod_common *Common) {
    return nested_dissection_impl<Int>(A, fset, fsize, Perm, CParent, Cmember, Common);
}

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
