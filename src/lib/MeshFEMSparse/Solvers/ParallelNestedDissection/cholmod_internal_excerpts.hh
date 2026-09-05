#ifndef MESHFEMSPARSE_CHOLMOD_INTERNAL_EXCERPTS_HH
#define MESHFEMSPARSE_CHOLMOD_INTERNAL_EXCERPTS_HH

//------------------------------------------------------------------------------
// CHOLMOD internal excerpts for parallel nested dissection
//------------------------------------------------------------------------------
//
// This private implementation header contains lightly adapted excerpts from
// CHOLMOD/Partition/cholmod_nesdis.c. CHOLMOD does not expose these helpers
// through its public API, but the parallel nested-dissection implementation
// needs the same graph compression, partition uncompression, flag clearing, and
// component-discovery logic as the upstream routine.

#include "cholmod_nesdis_parallel.hh"

#include <algorithm>
#include <cassert>
#include <chrono>
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
#define DEBUG(statement)
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
// range 0 to csize-1. The input graph (Cp, Ci) is destroyed.  Cew is all 1's
// on input and output.  Cnw [j] > 0 is the initial weight of node j.  On
// output, Cnw [i] = 0 if node i is absorbed into j and the original weight
// Cnw [i] is added to Cnw [j].  If compress is FALSE, the graph is not
// compressed and Cnw and Hash are unmodified.  The partition itself is held in
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

template<class Int>
static int64_t partition    // size of separator or -1 if failure
(
    // inputs, not modified on output
    #ifndef NDEBUG
    Int csize,          // upper bound on # of edges in the graph;
                        // csize >= MAX (n, nnz(C)) must hold.
    #endif
    int compress,       // if TRUE the compress the graph first
    Int nd_level,       // simulated nested-dissection recursion level

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
    Int Cew [ ],        // size csize, all 1's on input and output

    // more workspace, undefined on input and output
    Int Cmap [ ],       // size n

    // output
    Int Part [ ],       // size n, Part [j] = 0, 1, or 2.

    cholmod_common *Common
)
{
    Int n, hash, head, i, j, k, p, pend, ilen, ilast, pi, piend,
        jlen, ok, cn, csep, pdest, nodes_pruned, nz, total_weight, jscattered ;
    Int *Cp, *Ci, *Next, *Hhead ;
    double metis_time ;

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
    DEBUG (for (p = 0 ; p < csize ; p++) ASSERT (Cew [p] == 1)) ;
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
    DEBUG (for (p = 0 ; p < csize ; p++) ASSERT (Cew [p] == 1)) ;
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

        // FUTURE WORK: could call CHACO, SCOTCH, ... here too
        // auto metis_start = std::chrono::steady_clock::now () ;
        csep = CholmodApi<Int>::metis_bisector (C, Cnw, Cew, Part, Common) ;
        // metis_time = std::chrono::duration<double> (
        //         std::chrono::steady_clock::now () - metis_start).count () ;
        // if (getenv ("CHOLMOD_NESDIS_TRACE") != NULL)
        // {
        //     fprintf (stderr,
        //         "CHOLMOD_NESDIS_METIS level=%lld n=%lld nnz=%lld "
        //         "compressed=0 nodes_pruned=%lld separator=%lld time=%.9g\n",
        //         (long long) nd_level, (long long) n, (long long) nz,
        //         (long long) nodes_pruned, (long long) csep, metis_time) ;
        // }

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

        // FUTURE WORK: could call CHACO, SCOTCH, ... here too
        // auto metis_start = std::chrono::steady_clock::now () ;
        csep = CholmodApi<Int>::metis_bisector (C, Cnw, Cew, Part, Common) ;
        // metis_time = std::chrono::duration<double> (
        //         std::chrono::steady_clock::now () - metis_start).count () ;
        // if (getenv ("CHOLMOD_NESDIS_TRACE") != NULL)
        // {
        //     fprintf (stderr,
        //         "CHOLMOD_NESDIS_METIS level=%lld n=%lld nnz=%lld "
        //         "compressed=1 nodes_pruned=%lld separator=%lld time=%.9g\n",
        //         (long long) nd_level, (long long) cn, (long long) pdest,
        //         (long long) nodes_pruned, (long long) csep, metis_time) ;
        // }

        if (csep < 0)
        {
            // failed
            return (-1) ;
        }

        PRINT2 (("Part: ")) ;
        DEBUG (for (j = 0 ; j < cn ; j++) PRINT2 (("" ID " ", Part [j]))) ;
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

//------------------------------------------------------------------------------
// clear_flag
//------------------------------------------------------------------------------

// A node j has been removed from the graph if Flag [j] < EMPTY.
// If Flag [j] >= EMPTY && Flag [j] < mark, then node j is alive but unmarked.
// Flag [j] == mark means that node j is alive and marked.  Incrementing mark
// means that all nodes are either (still) dead, or live but unmarked.
//
// If Map is NULL, then on output, Common->mark < Common->Flag [i] for all i
// from 0 to Common->nrow.  This is the same output condition as
// cholmod_clear_flag, except that this routine maintains the Flag [i] < EMPTY
// condition as well, if that condition was true on input.
//
// If Map is non-NULL, then on output, Common->mark < Common->Flag [i] for all
// i in the set Map [0..cn-1].
//
// workspace: Flag (nrow)

template<class Int>
static int64_t clear_flag (Int *Map, Int cn, cholmod_common *Common)
{
    Int nrow, i ;
    Int *Flag ;
    PRINT2 (("old mark %ld\n", Common->mark)) ;
    Common->mark++ ;
    PRINT2 (("new mark %ld\n", Common->mark)) ;
    if (Common->mark <= 0)
    {
        nrow = Common->nrow ;
        Flag = (Int *) Common->Flag ;
        if (Map != NULL)
        {
            for (i = 0 ; i < cn ; i++)
            {
                // if Flag [Map [i]] < EMPTY, leave it alone
                if (Flag [Map [i]] >= EMPTY)
                {
                    Flag [Map [i]] = EMPTY ;
                }
            }
            // now Flag [Map [i]] <= EMPTY for all i
        }
        else
        {
            for (i = 0 ; i < nrow ; i++)
            {
                // if Flag [i] < EMPTY, leave it alone
                if (Flag [i] >= EMPTY)
                {
                    Flag [i] = EMPTY ;
                }
            }
            // now Flag [i] <= EMPTY for all i
        }
        Common->mark = 0 ;
    }
    return (Common->mark) ;
}

//------------------------------------------------------------------------------
// local_clear_mark
//------------------------------------------------------------------------------

template<class Int>
static Int local_clear_mark
(
    Int *Map,
    Int cn,
    Int *Mark,
    Int *pmark,
    Int n
)
{
    Int i ;
    (*pmark)++ ;
    if (*pmark <= 0)
    {
        if (Map != NULL)
        {
            for (i = 0 ; i < cn ; i++)
            {
                Mark [Map [i]] = EMPTY ;
            }
        }
        else
        {
            for (i = 0 ; i < n ; i++)
            {
                Mark [i] = EMPTY ;
            }
        }
        *pmark = 1 ;
    }
    return (*pmark) ;
}

//------------------------------------------------------------------------------
// find_components
//------------------------------------------------------------------------------

// Find all connected components of the current subgraph C.  The subgraph C
// consists of the nodes of B that appear in the set Map [0..cn-1].  If Map
// is NULL, then it is assumed to be the identity mapping
// (Map [0..cn-1] = 0..cn-1).
//
// A node j does not appear in B if it has been ordered (Flag [j] < EMPTY,
// which means that j has been ordered and is "deleted" from B).
//
// If the size of a component is large, it is placed on the component stack,
// Cstack.  Otherwise, its nodes are ordered and it is not placed on the Cstack.
//
// A component S is defined by a "representative node" (repnode for short)
// called the snode, which is one of the nodes in the subgraph.  Likewise, the
// subgraph C is defined by its repnode, called cnode.
//
// If Part is not NULL on input, then Part [i] determines how the components
// are placed on the stack.  Components containing nodes i with Part [i] == 0
// are placed first, followed by components with Part [i] == 1.
//
// The first node placed in each of the two parts is flipped when placed in
// the Cstack.  This allows the components of the two parts to be found simply
// by traversing the Cstack.
//
// workspace: Flag (nrow)

template<class Int>
static void find_components
(
    // inputs, not modified on output
    cholmod_sparse *B,
    Int Map [ ],            // size n, only Map [0..cn-1] used
    Int cn,                 // # of nodes in C
    Int cnode,              // root node of component C, or EMPTY if C is the
                            // entire graph B

    Int Part [ ],           // size cn, optional

    // input/output
    Int Bnz [ ],            // size n.  Bnz [j] = # nonzeros in column j of B.
                            // Reduce since B is pruned of dead nodes.

    Int CParent [ ],        // CParent [i] = j if component with repnode j is
                            // the parent of the component with repnode i.
                            // CParent [i] = EMPTY if the component with
                            // repnode i is a root of the separator tree.
                            // CParent [i] is -2 if i is not a repnode.
    Int Cstack [ ],         // component stack for nested dissection
    Int *top,               // Cstack [0..top] contains root nodes of the
                            // the components currently in the stack

    // workspace, undefined on input and output:
    Int State [ ],          // size n, persistent node state
    Int Mark [ ],           // size n, local traversal marks
    Int *pmark,             // local traversal mark counter
    Int Queue [ ],          // size n, for breadth-first search

    cholmod_common *Common
)
{

    Int n, mark, cj, j, sj, sn, p, i, snode, pstart, pdest, pend, nd_components,
        part, first ;
    Int *Bp, *Bi ;

    //--------------------------------------------------------------------------
    // get workspace
    //--------------------------------------------------------------------------

    PRINT2 (("find components: cn %d\n", cn)) ;
    mark = local_clear_mark<Int> (Map, cn, Mark, pmark, B->nrow) ;

    Bp = (Int *) B->p ;
    Bi = (Int *) B->i ;
    n = B->nrow ;
    ASSERT (cnode >= EMPTY && cnode < n) ;
    ASSERT (IMPLIES (cnode >= 0, State [cnode] < EMPTY)) ;

    // get ordering parameters
    nd_components = Common->method [Common->current].nd_components ;

    //--------------------------------------------------------------------------
    // find the connected components of C via a breadth-first search
    //--------------------------------------------------------------------------

    part = (Part == NULL) ? 0 : 1 ;

    // examine each part (part 1 and then part 0)
    for (part = (Part == NULL) ? 0 : 1 ; part >= 0 ; part--)
    {

        // first is TRUE for the first connected component in each part
        first = TRUE ;

        // find all connected components in the current part
        for (cj = 0 ; cj < cn ; cj++)
        {
            // get node snode, which is node cj of C.  It might already be in
            // the separator of C (and thus ordered, with Flag [snode] < EMPTY)
            snode = (Map == NULL) ? (cj) : (Map [cj]) ;
            ASSERT (snode >= 0 && snode < n) ;

            if (State [snode] >= EMPTY && Mark [snode] != mark
                    && ((Part == NULL) || Part [cj] == part))
            {

                //--------------------------------------------------------------
                // find new connected component S
                //--------------------------------------------------------------

                // node snode is the repnode of a connected component S, the
                // parent of which is cnode, the repnode of C.  If cnode is
                // EMPTY then C is the original graph B.
                PRINT2 (("----------:::snode " ID " cnode " ID "\n", snode, cnode));

                ASSERT (CParent [snode] == -2) ;
                if (first || nd_components)
                {
                    // If this is the first node in this part, then it becomes
                    // the repnode of all components in this part, and all
                    // components in this part form a single node in the
                    // separator tree.  If nd_components is TRUE, then all
                    // connected components form their own node in the
                    // separator tree.
                    CParent [snode] = cnode ;
                }

                // place j in the queue and mark it
                Queue [0] = snode ;
                Mark [snode] = mark ;
                sn = 1 ;

                // breadth-first traversal, starting at node j
                for (sj = 0 ; sj < sn ; sj++)
                {
                    // get node j from head of Queue and traverse its edges
                    j = Queue [sj] ;
                    PRINT2 (("    j: " ID "\n", j)) ;
                    ASSERT (j >= 0 && j < n) ;
                    ASSERT (Mark [j] == mark) ;
                    pstart = Bp [j] ;
                    pdest = pstart ;
                    pend = pstart + Bnz [j] ;
                    for (p = pstart ; p < pend ; p++)
                    {
                        i = Bi [p] ;
                        if (i != j && State [i] >= EMPTY)
                        {
                            // node is still in the graph
                            Bi [pdest++] = i ;
                            if (Mark [i] != mark)
                            {
                                // node i is in this component S, and unflagged
                                // (first time node i has been seen in this BFS)
                                // place node i in the queue and mark it
                                Queue [sn++] = i ;
                                Mark [i] = mark ;
                            }
                        }
                    }
                    // edges to dead nodes have been removed
                    Bnz [j] = pdest - pstart ;
                }

                //--------------------------------------------------------------
                // order S if it is small; place it on Cstack otherwise
                //--------------------------------------------------------------

                PRINT2 (("sn " ID "\n", sn)) ;

                // place the new component on the Cstack.  Flip the node if
                // is the first connected component of the current part,
                // or if all components are treated as their own node in
                // the separator tree.
                Cstack [++(*top)] =
                        (first || nd_components) ? FLIP (snode) : snode ;
                first = FALSE ;
            }
        }
    }

}


} // namespace MeshFEM::CholmodParallelNesdis

#endif /* MESHFEMSPARSE_CHOLMOD_INTERNAL_EXCERPTS_HH */
