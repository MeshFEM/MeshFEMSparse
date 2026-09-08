#ifndef MESHFEMSPARSE_CHOLMOD_INTERNAL_EXCERPTS_HH
#define MESHFEMSPARSE_CHOLMOD_INTERNAL_EXCERPTS_HH

//------------------------------------------------------------------------------
// CHOLMOD internal excerpts for parallel nested dissection
//------------------------------------------------------------------------------
//
// This private implementation header contains lightly adapted excerpts from
// CHOLMOD/Partition/cholmod_nesdis.c. CHOLMOD does not expose these helpers
// through its public API, but the parallel nested-dissection implementation
// needs the same graph compression, partition uncompression, and flag-clearing
// logic as the upstream routine.

#include "cholmod_nesdis_parallel.hh"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <type_traits>

namespace MeshFEM::CholmodParallelNesdis {

constexpr int TRUE = 1;
constexpr int FALSE = 0;
constexpr int EMPTY = -1;

#define FLIP(i) (-(i)-2)
#define UNFLIP(i) (((i) < EMPTY) ? FLIP (i) : (i))
#define MAX(a,b) (((a) > (b)) ? (a) : (b))
#define MIN(a,b) (((a) < (b)) ? (a) : (b))
#define IMPLIES(p,q) (!(p) || (q))
#define ASSERT(expression) assert(expression)
#ifdef NDEBUG
#define DEBUG(statement)
#else
#define DEBUG(statement) statement
#endif
#define PRINT0(params)
#define PRINT1(params)
#define PRINT2(params)
#define PRINT3(params)
#define ERROR(status,msg) CholmodApi<Int>::error(status, __FILE__, __LINE__, msg, Common)
#define RETURN_IF_NULL(A,result) do { if ((A) == NULL) { if (Common->status != CHOLMOD_OUT_OF_MEMORY) ERROR(CHOLMOD_INVALID, "argument missing"); return (result); } } while (0)
#define RETURN_IF_NULL_COMMON(result) do { if (Common == NULL) return (result); } while (0)
#define RETURN_IF_XTYPE_INVALID(A,xtype1,xtype2,result) do { if ((A)->xtype < (xtype1) || (A)->xtype > (xtype2)) { ERROR(CHOLMOD_INVALID, "invalid xtype"); return (result); } } while (0)
#define ID "%lld"

template<class Int>
struct CholmodApi;

template<>
struct CholmodApi<int32_t> {
    static int error(int status, const char *file, int line, const char *msg, cholmod_common *Common) { return cholmod_error(status, file, line, msg, Common); }
    static int allocate_work(size_t nrow, size_t iworksize, size_t xworksize, cholmod_common *Common) { return cholmod_allocate_work(nrow, iworksize, xworksize, Common); }
    static cholmod_sparse *allocate_sparse(size_t nrow, size_t ncol, size_t nzmax, int sorted, int packed, int stype, int xtype, cholmod_common *Common) { return cholmod_allocate_sparse(nrow, ncol, nzmax, sorted, packed, stype, xtype, Common); }
    static int free_sparse(cholmod_sparse **A, cholmod_common *Common) { return cholmod_free_sparse(A, Common); }
    static void *malloc(size_t n, size_t size, cholmod_common *Common) { return cholmod_malloc(n, size, Common); }
    static void *free(size_t n, size_t size, void *p, cholmod_common *Common) { return cholmod_free(n, size, p, Common); }
    static cholmod_sparse *copy(cholmod_sparse *A, int stype, int mode, cholmod_common *Common) { return cholmod_copy(A, stype, mode, Common); }
    static cholmod_sparse *aat(cholmod_sparse *A, int32_t *fset, size_t fsize, int mode, cholmod_common *Common) { return cholmod_aat(A, fset, fsize, mode, Common); }
    static int64_t nnz(cholmod_sparse *A, cholmod_common *Common) { return cholmod_nnz(A, Common); }
    static int64_t metis_bisector(cholmod_sparse *A, int32_t *Anw, int32_t *Aew, int32_t *Partition, cholmod_common *Common) { return cholmod_metis_bisector(A, Anw, Aew, Partition, Common); }
    static int64_t postorder(int32_t *Parent, size_t n, int32_t *Weight, int32_t *Post, cholmod_common *Common) { return cholmod_postorder(Parent, n, Weight, Post, Common); }
    static int camd(cholmod_sparse *A, int32_t *fset, size_t fsize, int32_t *Cmember, int32_t *Perm, cholmod_common *Common) { return cholmod_camd(A, fset, fsize, Cmember, Perm, Common); }
    static int ccolamd(cholmod_sparse *A, int32_t *fset, size_t fsize, int32_t *Cmember, int32_t *Perm, cholmod_common *Common) { return cholmod_ccolamd(A, fset, fsize, Cmember, Perm, Common); }
    static int csymamd(cholmod_sparse *A, int32_t *Cmember, int32_t *Perm, cholmod_common *Common) { return cholmod_csymamd(A, Cmember, Perm, Common); }
};

template<>
struct CholmodApi<int64_t> {
    static int error(int status, const char *file, int line, const char *msg, cholmod_common *Common) { return cholmod_l_error(status, file, line, msg, Common); }
    static int allocate_work(size_t nrow, size_t iworksize, size_t xworksize, cholmod_common *Common) { return cholmod_l_allocate_work(nrow, iworksize, xworksize, Common); }
    static cholmod_sparse *allocate_sparse(size_t nrow, size_t ncol, size_t nzmax, int sorted, int packed, int stype, int xtype, cholmod_common *Common) { return cholmod_l_allocate_sparse(nrow, ncol, nzmax, sorted, packed, stype, xtype, Common); }
    static int free_sparse(cholmod_sparse **A, cholmod_common *Common) { return cholmod_l_free_sparse(A, Common); }
    static void *malloc(size_t n, size_t size, cholmod_common *Common) { return cholmod_l_malloc(n, size, Common); }
    static void *free(size_t n, size_t size, void *p, cholmod_common *Common) { return cholmod_l_free(n, size, p, Common); }
    static cholmod_sparse *copy(cholmod_sparse *A, int stype, int mode, cholmod_common *Common) { return cholmod_l_copy(A, stype, mode, Common); }
    static cholmod_sparse *aat(cholmod_sparse *A, int64_t *fset, size_t fsize, int mode, cholmod_common *Common) { return cholmod_l_aat(A, fset, fsize, mode, Common); }
    static int64_t nnz(cholmod_sparse *A, cholmod_common *Common) { return cholmod_l_nnz(A, Common); }
    static int64_t metis_bisector(cholmod_sparse *A, int64_t *Anw, int64_t *Aew, int64_t *Partition, cholmod_common *Common) { return cholmod_l_metis_bisector(A, Anw, Aew, Partition, Common); }
    static int64_t postorder(int64_t *Parent, size_t n, int64_t *Weight, int64_t *Post, cholmod_common *Common) { return cholmod_l_postorder(Parent, n, Weight, Post, Common); }
    static int camd(cholmod_sparse *A, int64_t *fset, size_t fsize, int64_t *Cmember, int64_t *Perm, cholmod_common *Common) { return cholmod_l_camd(A, fset, fsize, Cmember, Perm, Common); }
    static int ccolamd(cholmod_sparse *A, int64_t *fset, size_t fsize, int64_t *Cmember, int64_t *Perm, cholmod_common *Common) { return cholmod_l_ccolamd(A, fset, fsize, Cmember, Perm, Common); }
    static int csymamd(cholmod_sparse *A, int64_t *Cmember, int64_t *Perm, cholmod_common *Common) { return cholmod_l_csymamd(A, Cmember, Perm, Common); }
};

template<class Int>
static void clear_common_flag(cholmod_common *Common) {
    Common->mark++;
    if (Common->mark <= 0) {
        auto *Flag = static_cast<Int *>(Common->Flag);
        for (size_t i = 0; i < Common->nrow; ++i) Flag[i] = EMPTY;
        Common->mark = 0;
    }
}

static size_t add_size_t(size_t a, size_t b, int *ok) {
    if (SIZE_MAX - a < b) { *ok = FALSE; return 0; }
    return a + b;
}

static size_t mult_size_t(size_t a, size_t b, int *ok) {
    if (a != 0 && SIZE_MAX / a < b) { *ok = FALSE; return 0; }
    return a * b;
}


//------------------------------------------------------------------------------
// partition
//------------------------------------------------------------------------------

// Find a set of nodes that partition a graph.  The graph must be symmetric
// with no diagonal entries.  To compress the graph first, compress is TRUE
// and on input Hash [j] holds the hash key for node j, which must be in the
// range 0 to csize-1. The input graph (Cp, Ci) is destroyed. Cew is all 1's
// on input and output, or NULL when compression is disabled (unit edge weights).
// Cnw [j] > 0 is the initial weight of node j. On
// output, Cnw [i] = 0 if node i is absorbed into j and the original weight
// Cnw [i] is added to Cnw [j].  If compress is FALSE, the graph is not
// compressed, Hash is not accessed, and Cnw is unmodified. The partition itself is held in
// the output array Part of size n.  Part [j] is 0, 1, or 2, depending on
// whether node j is in the left part of the graph, the right part, or the
// separator, respectively.  Note that the input graph need not be connected,
// and the output subgraphs (the three parts) may also be unconnected.
//
// Returns the size of the separator, in terms of the sum of the weights of
// the nodes.  It is guaranteed to be between 1 and the total weight of all
// the nodes.  If it is of size less than the total weight, then both the left
// and right parts are guaranteed to be non-empty (this guarantee depends on
// cholmod_metis_bisector).

template<class Int, class Bisector>
static int64_t partition    // size of separator or -1 if failure
(
    // inputs, not modified on output
    #ifndef NDEBUG
    Int csize,          // upper bound on # of edges in the graph;
                        // csize >= MAX (n, nnz(C)) must hold.
    #endif
    int compress,       // if TRUE the compress the graph first
    Int nd_level,       // nested-dissection recursion level

    // input/output
    Int Hash [ ],       // Hash [i] = hash >= 0 is the hash function for node
                        // i on input.  On output, Hash [i] = FLIP (j) if node
                        // i is absorbed into j.  Hash [i] >= 0 if i has not
                        // been absorbed.

    // input graph, compressed graph of cn nodes on output
    cholmod_sparse *C,

    // input/output
    Int Cnw [ ],        // size n.  Cnw [j] > 0 is the weight of node j on
                        // input.  On output, if node i is absorbed into
                        // node j, then Cnw [i] = 0 and the original weight of
                        // node i is added to Cnw [j].  The sum of Cnw [0..n-1]
                        // is not modified.

    // workspace
    Int Cew [ ],        // size csize, all 1's; may be NULL if !compress

    // more workspace, undefined on input and output
    Int Cmap [ ],       // size n

    // output
    Int Part [ ],       // size n, Part [j] = 0, 1, or 2.

    cholmod_common *Common,
    Bisector &bisect
)
{
    Int n, hash, head, i, j, k, p, pend, ilen, ilast, pi, piend,
        jlen, ok, cn, csep, pdest, nodes_pruned, nz, total_weight, jscattered ;
    Int *Cp, *Ci, *Next, *Hhead ;

    #ifndef NDEBUG
    Int cnt, pruned ;
    double work = 0, goodwork = 0 ;
    #endif

    //--------------------------------------------------------------------------
    // quick return for small or empty graphs
    //--------------------------------------------------------------------------

    n = C->nrow ;
    Cp = (Int *) C->p ;
    Ci = (Int *) C->i ;
    nz = Cp [n] ;

    PRINT2 (("Partition start, n " ID " nz " ID "\n", n, nz)) ;

    total_weight = 0 ;
    for (j = 0 ; j < n ; j++)
    {
        ASSERT (Cnw [j] > 0) ;
        total_weight += Cnw [j] ;
    }

    if (n <= 2)
    {
        // very small graph
        for (j = 0 ; j < n ; j++)
        {
            Part [j] = 2 ;
        }
        return (total_weight) ;
    }
    else if (nz <= 0)
    {
        // no edges, this is easy
        PRINT2 (("diagonal matrix\n")) ;
        k = n/2 ;
        for (j = 0 ; j < k ; j++)
        {
            Part [j] = 0 ;
        }
        for ( ; j < n ; j++)
        {
            Part [j] = 1 ;
        }
        // ensure the separator is not empty (required by nested dissection)
        Part [n-1] = 2 ;
        return (Cnw [n-1]) ;
    }

    #ifndef NDEBUG
    ASSERT (n > 1 && nz > 0) ;
    PRINT2 (("original graph:\n")) ;
    for (j = 0 ; j < n ; j++)
    {
        PRINT2 (("" ID ": ", j)) ;
        for (p = Cp [j] ; p < Cp [j+1] ; p++)
        {
            i = Ci [p] ;
            PRINT3 (("" ID " ", i)) ;
            ASSERT (i >= 0 && i < n && i != j) ;
        }
        PRINT2 (("hash: " ID "\n", Hash [j])) ;
    }
    DEBUG (if (Cew) for (p = 0 ; p < csize ; p++) ASSERT (Cew [p] == 1)) ;
    #endif

    nodes_pruned = 0 ;

    if (compress)
    {

        //----------------------------------------------------------------------
        // get workspace
        //----------------------------------------------------------------------

        Next = Part ;   // use Part as workspace for Next [
        Hhead = Cew ;   // use Cew as workspace for Hhead [

        //----------------------------------------------------------------------
        // create the hash buckets
        //----------------------------------------------------------------------

        for (j = 0 ; j < n ; j++)
        {
            // get the hash key for node j
            hash = Hash [j] ;
            ASSERT (hash >= 0 && hash < csize) ;
            head = Hhead [hash] ;
            if (head > EMPTY)
            {
                // hash bucket for this hash key is empty.
                head = EMPTY ;
            }
            else
            {
                // hash bucket for this hash key is not empty.  get old head
                head = FLIP (head) ;
                ASSERT (head >= 0 && head < n) ;
            }
            // node j becomes the new head of the hash bucket.  FLIP it so that
            // we can tell the difference between an empty or non-empty hash
            // bucket.
            Hhead [hash] = FLIP (j) ;
            Next [j] = head ;
            ASSERT (head >= EMPTY && head < n) ;
        }

        #ifndef NDEBUG
        for (cnt = 0, k = 0 ; k < n ; k++)
        {
            ASSERT (Hash [k] >= 0 && Hash [k] < csize) ;    // k is alive
            hash = Hash [k] ;
            ASSERT (hash >= 0 && hash < csize) ;
            head = Hhead [hash] ;
            ASSERT (head < EMPTY) ;     // hash bucket not empty
            j = FLIP (head) ;
            ASSERT (j >= 0 && j < n) ;
            if (j == k)
            {
                PRINT2 (("hash " ID ": ", hash)) ;
                for ( ; j != EMPTY ; j = Next [j])
                {
                    PRINT3 ((" " ID "", j)) ;
                    ASSERT (j >= 0 && j < n) ;
                    ASSERT (Hash [j] == hash) ;
                    cnt++ ;
                    ASSERT (cnt <= n) ;
                }
                PRINT2 (("\n")) ;
            }
        }
        ASSERT (cnt == n) ;
        #endif

        //----------------------------------------------------------------------
        // scan the non-empty hash buckets for indistinguishable nodes
        //----------------------------------------------------------------------

        // If there are no hash collisions and no compression occurs, this takes
        // O(n) time.  If no hash collisions, but some nodes are removed, this
        // takes time O(n+e) where e is the sum of the degress of the nodes
        // that are removed.  Even with many hash collisions (a rare case),
        // this algorithm has never been observed to perform more than nnz(A)
        // useless work.
        //
        // Cmap is used as workspace to mark nodes of the graph, [
        // for comparing the nonzero patterns of two nodes i and j.

        #define Cmap_MARK(i)   Cmap [i] = j
        #define Cmap_MARKED(i) (Cmap [i] == j)

        for (i = 0 ; i < n ; i++)
        {
            Cmap [i] = EMPTY ;
        }

        for (k = 0 ; k < n ; k++)
        {
            hash = Hash [k] ;
            ASSERT (hash >= FLIP (n-1) && hash < csize) ;
            if (hash < 0)
            {
                // node k has already been absorbed into some other node
                ASSERT (FLIP (Hash [k]) >= 0 && FLIP (Hash [k] < n)) ;
                continue ;
            }
            head = Hhead [hash] ;
            ASSERT (head < EMPTY || head == 1) ;
            if (head == 1)
            {
                // hash bucket is already empty
                continue ;
            }
            PRINT2 (("\n--------------------hash " ID ":\n", hash)) ;
            for (j = FLIP (head) ; j != EMPTY && Next[j] > EMPTY ; j = Next [j])
            {
                // compare j with all nodes i following it in hash bucket
                ASSERT (j >= 0 && j < n && Hash [j] == hash) ;
                p = Cp [j] ;
                pend = Cp [j+1] ;
                jlen = pend - p ;
                jscattered = FALSE ;
                DEBUG (for (i = 0 ; i < n ; i++) ASSERT (!Cmap_MARKED (i))) ;
                DEBUG (pruned = FALSE) ;
                ilast = j ;
                for (i = Next [j] ; i != EMPTY ; i = Next [i])
                {
                    ASSERT (i >= 0 && i < n && Hash [i] == hash && i != j) ;
                    pi = Cp [i] ;
                    piend = Cp [i+1] ;
                    ilen = piend - pi ;
                    DEBUG (work++) ;
                    if (ilen != jlen)
                    {
                        // i and j have different degrees
                        ilast = i ;
                        continue ;
                    }
                    // scatter the pattern of node j, if not already
                    if (!jscattered)
                    {
                        Cmap_MARK (j) ;
                        for ( ; p < pend ; p++)
                        {
                            Cmap_MARK (Ci [p]) ;
                        }
                        jscattered = TRUE ;
                        DEBUG (work += jlen) ;
                    }
                    for (ok = Cmap_MARKED (i) ; ok && pi < piend ; pi++)
                    {
                        ok = Cmap_MARKED (Ci [pi]) ;
                        DEBUG (work++) ;
                    }
                    if (ok)
                    {
                        // found it.  kill node i and merge it into j
                        PRINT2 (("found " ID " absorbed into " ID "\n", i, j)) ;
                        Hash [i] = FLIP (j) ;
                        Cnw [j] += Cnw [i] ;
                        Cnw [i] = 0 ;
                        ASSERT (ilast != i && ilast >= 0 && ilast < n) ;
                        Next [ilast] = Next [i] ; // delete i from bucket
                        nodes_pruned++ ;
                        DEBUG (goodwork += (ilen+1)) ;
                        DEBUG (pruned = TRUE) ;
                    }
                    else
                    {
                        // i and j are different
                        ilast = i ;
                    }
                }
                DEBUG (if (pruned) goodwork += jlen) ;
            }
            // empty the hash bucket, restoring Cew
            Hhead [hash] = 1 ;
        }

        DEBUG (if (((work - goodwork) / (double) nz) > 0.20) PRINT0 ((
            "work %12g good %12g nz %12g (wasted work/nz: %6.2f )\n",
            work, goodwork, (double) nz, (work - goodwork) / ((double) nz)))) ;

        // All hash buckets now empty.  Cmap no longer needed as workspace. ]
        // Cew no longer needed as Hhead; Cew is now restored to all ones. ]
        // Part no longer needed as workspace for Next. ]
    }

    // Edge weights are all one, node weights reflect node absorption
    DEBUG (if (Cew) for (p = 0 ; p < csize ; p++) ASSERT (Cew [p] == 1)) ;
    DEBUG (for (cnt = 0, j = 0 ; j < n ; j++) cnt += Cnw [j]) ;
    ASSERT (cnt == total_weight) ;

    //--------------------------------------------------------------------------
    // compress and partition the graph
    //--------------------------------------------------------------------------

    if (nodes_pruned == 0)
    {

        //----------------------------------------------------------------------
        // no pruning done at all.  Do not create the compressed graph
        //----------------------------------------------------------------------

        csep = bisect (C, Cnw, Cew, Part, (int) nd_level, Common) ;

    }
    else if (nodes_pruned == n-1)
    {

        //----------------------------------------------------------------------
        // only one node left.  This is a dense graph
        //----------------------------------------------------------------------

        PRINT2 (("completely dense graph\n")) ;
        csep = total_weight ;
        for (j = 0 ; j < n ; j++)
        {
            Part [j] = 2 ;
        }

    }
    else
    {

        //----------------------------------------------------------------------
        // compress the graph and partition the compressed graph
        //----------------------------------------------------------------------

        //----------------------------------------------------------------------
        // create the map from the uncompressed graph to the compressed graph
        //----------------------------------------------------------------------

        // Cmap [j] = k if node j is alive and the kth node of compressed graph.
        // The mapping is done monotonically (that is, k <= j) to simplify the
        // uncompression later on.  Cmap [j] = EMPTY if node j is dead.

        for (j = 0 ; j < n ; j++)
        {
            Cmap [j] = EMPTY ;
        }
        k = 0 ;
        for (j = 0 ; j < n ; j++)
        {
            if (Cnw [j] > 0)
            {
                ASSERT (k <= j) ;
                Cmap [j] = k++ ;
            }
        }
        cn = k ;            // # of nodes in compressed graph
        PRINT2 (("compressed graph from " ID " to " ID " nodes\n", n, cn)) ;
        ASSERT (cn > 1 && cn == n - nodes_pruned) ;

        //----------------------------------------------------------------------
        // create the compressed graph
        //----------------------------------------------------------------------

        k = 0 ;
        pdest = 0 ;
        for (j = 0 ; j < n ; j++)
        {
            if (Cnw [j] > 0)
            {
                // node j in the full graph is node k in the compressed graph
                ASSERT (k <= j && Cmap [j] == k) ;
                p = Cp [j] ;
                pend = Cp [j+1] ;
                Cp [k] = pdest ;
                Cnw [k] = Cnw [j] ;
                for ( ; p < pend ; p++)
                {
                    // prune dead nodes, and remap to new node numbering
                    i = Ci [p] ;
                    ASSERT (i >= 0 && i < n && i != j) ;
                    i = Cmap [i] ;
                    ASSERT (i >= EMPTY && i < cn && i != k) ;
                    if (i > EMPTY)
                    {
                        ASSERT (pdest <= p) ;
                        Ci [pdest++] = i ;
                    }
                }
                k++ ;
            }
        }
        Cp [cn] = pdest ;
        C->nrow = cn ;
        C->ncol = cn ;  // affects mem stats unless restored when C free'd

        #ifndef NDEBUG
        PRINT2 (("pruned graph (" ID "/" ID ") nodes, (" ID "/" ID ") edges\n",
                    cn, n, pdest, nz)) ;
        PRINT2 (("compressed graph:\n")) ;
        for (cnt = 0, j = 0 ; j < cn ; j++)
        {
            PRINT2 (("" ID ": ", j)) ;
            for (p = Cp [j] ; p < Cp [j+1] ; p++)
            {
                i = Ci [p] ;
                PRINT3 (("" ID " ", i)) ;
                ASSERT (i >= 0 && i < cn && i != j) ;
            }
            PRINT2 (("weight: " ID "\n", Cnw [j])) ;
            ASSERT (Cnw [j] > 0) ;
            cnt += Cnw [j] ;
        }
        ASSERT (cnt == total_weight) ;
        for (j = 0 ; j < n ; j++) PRINT2 (("Cmap [" ID "] = " ID "\n", j, Cmap[j]));
        ASSERT (k == cn) ;
        #endif

        //----------------------------------------------------------------------
        // find the separator of the compressed graph
        //----------------------------------------------------------------------

        csep = bisect (C, Cnw, Cew, Part, (int) nd_level, Common) ;

        if (csep < 0)
        {
            // failed
            return (-1) ;
        }

        PRINT2 (("Part: ")) ;
        PRINT2 (("\n")) ;

        // Cp and Ci no longer needed

        //----------------------------------------------------------------------
        // find the separator of the uncompressed graph
        //----------------------------------------------------------------------

        // expand the separator to live nodes in the uncompressed graph
        for (j = n-1 ; j >= 0 ; j--)
        {
            // do this in reverse order so that Cnw can be expanded in place
            k = Cmap [j] ;
            ASSERT (k >= EMPTY && k < n) ;
            if (k > EMPTY)
            {
                // node k in compressed graph and is node j in full graph
                ASSERT (k <= j) ;
                ASSERT (Hash [j] >= EMPTY) ;
                Part [j] = Part [k] ;
                Cnw [j] = Cnw [k] ;
            }
            else
            {
                // node j is a dead node
                Cnw [j] = 0 ;
                DEBUG (Part [j] = EMPTY) ;
                ASSERT (Hash [j] < EMPTY) ;
            }
        }

        // find the components for the dead nodes
        for (i = 0 ; i < n ; i++)
        {
            if (Hash [i] < EMPTY)
            {
                // node i has been absorbed into node j
                j = FLIP (Hash [i]) ;
                ASSERT (Part [i] == EMPTY && j >= 0 && j < n && Cnw [i] == 0) ;
                Part [i] = Part [j] ;
            }
            ASSERT (Part [i] >= 0 && Part [i] <= 2) ;
        }

        #ifndef NDEBUG
        PRINT2 (("Part: ")) ;
        for (cnt = 0, j = 0 ; j < n ; j++)
        {
            ASSERT (Part [j] != EMPTY) ;
            PRINT2 (("" ID " ", Part [j])) ;
            if (Part [j] == 2) cnt += Cnw [j] ;
        }
        PRINT2 (("\n")) ;
        PRINT2 (("csep " ID " " ID "\n", cnt, csep)) ;
        ASSERT (cnt == csep) ;
        for (cnt = 0, j = 0 ; j < n ; j++) cnt += Cnw [j] ;
        ASSERT (cnt == total_weight) ;
        #endif

    }

    //--------------------------------------------------------------------------
    // return the separator (or -1 if error)
    //--------------------------------------------------------------------------

    PRINT2 (("Partition done, n " ID " csep " ID "\n", n, csep)) ;
    return (csep) ;
}

} // namespace MeshFEM::CholmodParallelNesdis

#endif /* MESHFEMSPARSE_CHOLMOD_INTERNAL_EXCERPTS_HH */
