# ParallelNestedDissection

This directory contains a modified version of cholmod_nesdis.c that introduces
TBB-based parallelism (i.e., a parallel traversal of the separator tree). The
original iterative algorithm was refactored to use explicit recursion to simplify
lifetime management of component stacks. The main tricky implementation point
was ensuring that state modified by the recursive calls is stored in
thread-local copies. Exploiting this parallelism has no impact on the final
ordering.

After parallelizing nested dissection, the time overhead of the `cholmod_nesdis`
algorithm's subsequent global CAMD ordering step becomes significant. This can
be avoided by setting the CHOLMOD options `nd_camd = 0` at the expense of
seriously degrading ordering quality; this degradation is less if `nd_small` is
reduced from the CHOLMOD default (200) to, e.g., `50` or `25`. In the standard
METIS ND implementation (e.g., `cholmod_metis`), the global CAMD is replaced
with CAMD ordering of the leaves; this is trivial to parallelize but tends to
produce slightly worse ordering since the separator nodes don't get reordered.
We have instead implemented a "hybrid" approach where separate CAMD
calls are made for each subtree at a configurable "cut depth"
(environment variable `CHOLMOD_NESDIS_CAMD_CUT_DEPTH`)
in parallel. Then a final CAMD is run at the upper part of the tree,
incorporating degree information from the quotient elements/cliques of each subtree.
This strategy allows us to interpolate between approximating the
`cholmod_metis` and `cholmod_nesdis` orderings:
the algorithm is similar to METIS_ND if we set a high cut depth
(and `CHOLMOD_NESDIS_CAMD_SKIP_UPPER=1` to avoid the final upper-level CAMD call),
while we can precisely reproduce `cholmod_nesdis` by setting a cut depth of -1.
Skipping upper CAMD keeps natural ordering inside each upper component.

Nested dissection time is dominated by separator computation, even more so
after several low-level optimizations described further down.
We default to using METIS with the same settings as the original
`cholmod_nesdis` code. In our experiments, this yields the highest-quality
separators, but it is a serial code and runs slower than other
partitioners. We have experimented with other multithreaded
partitioning codes (mt-KaHIP, mt-METIS, SCOTCH) and found SCOTCH to
provide the best trade-off between quality and speed among them.
We have therefore exposed an option to use SCOTCH to construct
the upper levels of the separator tree
(down to depth `CHOLMOD_NESDIS_SCOTCH_LEVELS`, with a setting of 0 disabling it).
Note that accelerating these top-level calls has the greatest impact
on ordering time since they dominate the critical path
at high core count (there is sufficient tree parallelism
in the lower levels that we want to run a serial partitioner like METIS anyway).
By adjusting `CHOLMOD_NESDIS_SCOTCH_LEVELS`
and `CHOLMOD_NESDIS_SCOTCH_STRATEGY`, different trade-offs can be selected between
ordering speed and quality; see [scotch_bisector.md](scotch_bisector.md)
for additional configuration details.

## Lower-level Accelerations

`CholmodOrdering` enables CHOLMOD's graph compression by default. Catamari,
Accelerate and Pardiso disable it only when their actual block size exceeds one,
after any scalar fallback. The per-instance `setNestedDissectionCompression`
setting updates both existing integer-width contexts and future ones. Direct
calls to the nested-dissection routines continue to respect `Common` settings.

Especially when using an inexpensive partitioner
(e.g., `CHOLMOD_NESDIS_SCOTCH_LEVELS=8 CHOLMOD_NESDIS_SCOTCH_STRATEGY=fast`),
other stages of the `cholmod_nesdis` algorithm begin to take nontrivial time.

Component discovery (`find_components`) now retains BFS-ordered vertex lists,
owned by the child tasks. Graph construction reuses those lists and the
already-pruned adjacency instead of rediscovering vertices, and small leaves
skip graph construction. When disconnected components are grouped together,
their saved BFS layers are interleaved to preserve the original multi-source BFS
numbering. This retains the serial traversal path. With compression disabled,
large components build their local adjacency in parallel: a serial vertex pass
fills the inverse map and column offsets, then TBB remaps neighbors into
disjoint column slices.
We also reduced the number of workspace arrays used by this algorithm (removing
the `Mark` array) by encoding both liveness and visitation in `Flag`: values
below -1 retain CHOLMOD's removed-node encoding, -1 means initially unvisited,
and nonnegative values are visit tags. A search uses its removed parent
separator's vertex ID plus one as its tag (zero for initial discovery without a
dense-node parent). This ID cannot recur in a descendant separator because that
vertex is removed; tags need not increase with depth.

We furthermore enable a bypass of CHOLMOD's serial upper-tri-to-full conversion
by supplying the full sparsity pattern (to the `nested_dissection_from_graph`
inteface). This inteface is now used by `CatamariFactorizer`, which already
needed to do this conversion, and did it with our faster parallel
implementation.

## Exploiting Temporal Coherence of Nested Dissection
The routine `nested_dissection_temporal` (used by, e.g., `CholeskyProvider::CatamariNesdisReuse` )
exploits temporal coherence in a sequence of sparsity patterns to avoid
bisector recomputation. This routine implements the core idea of the PARTH
algorithm ([Zarebavani et al. 2025: Adaptive Algebraic Reuse of Reordering in
Cholesky Factorizations with Dynamic Sparsity Patterns]; reference
implementation released [here](https://github.com/BehroozZare/Parth)), which is
to only recompute subtrees of the separator tree when they are violated by new
edges appearing in the matrix graph. In other words, recomputation is only
needed if an edge connects two vertices that belong to different components (and
one of those components is not an ancestor of the other); in this case, the
common ancestor of those two components must be rebuilt.

Our implementation differs from the one described in the paper in a few ways:
- It processes subtrees in parallel and a final parallel CAMD is run using the approaches described above.
  The global CAMD is still somewhat wasteful since we expect the orderings
  of components outside the rebuilt subtree to change very little if at all.
  A more targeted update would be made possible by retaining the full prior
  ordering (along with the separator tree) and slicing in updated CAMD orderings
  for just the rebuilt subtrees. However, the global CAMD time currently does not
  appear to be a major bottleneck.
- It does not implement certain features like aggressive reuse (the importance of which
  is deemphasized somewhat in the PARTH supplement) or support addition or removal of
  system variables. These features could be added in the future if we notice
  bottlenecks that they could resolve, but for now we hope that our faster parallel
  ND implementation will already larly mitigate such slowdowns.

Note that the PARTH approach can degrade ordering since it only rebuilds
separators when new edges violate them and does not revisit ordering when old
edges are removed (our current implementation *does* recompute CAMD ordering in
this case, but it does not update the separator tree).
Therefore, it can be beneficial to rebuild the full
ND partitioning occasionally. Ideally this rebuild would be driven by 
some principled estimate of separator tree quality/staleness, but for now
we simply rebuild it after a fixed number of analyiss calls
(after expiration of a "reuse period").
Selecting `CholeskyProvider::CatamariNesdisReuse` configures a default reuse period of 32
that can be overriden by a call to
`CatamariFactorizer::setTemporalReusePeriod(P)` (Python:
`factorizer.temporalReusePeriod = P`).
A full ND recomputation is also triggered if our `SparseLRU`
initiates a re-factorization due to entry cache expiration.

## Configuration and errors

Environment settings are parsed once per ordering call, before allocating graph
or traversal storage. Integer values must be complete decimal integers without
whitespace; malformed or out-of-range values fail with `CHOLMOD_INVALID`.
`CHOLMOD_NESDIS_NUM_THREADS` defaults to zero (inherit the existing TBB limit).
A positive value adds a limit for the entire call, including graph construction
and CAMD; it cannot raise another active TBB limit. `CHOLMOD_NESDIS_SERIAL_SUBTREE_SIZE`
must be positive and defaults to 2000. SCOTCH settings are documented in
[scotch_bisector.md](scotch_bisector.md).

Both C entry points and the full-graph C++ entry point return -1 on failure and
set `Common->status`. C++ allocation failures map to `CHOLMOD_OUT_OF_MEMORY`;
invalid settings and other exceptions map to `CHOLMOD_INVALID`.
We now use C++ exceptions internally to cancel parallel execution
and simplify cleanup upon failures, but no exceptions excape these functions.

## License

Since this implementation is derived from `CHOLMOD/Partition/cholmod_nesdis.c`,
whose upstream SPDX license identifier is `LGPL-2.1+`, we keep this same GNU
Lesser General Public License, version 2.1 or later. The SuiteSparse-bundled
copy of METIS used by CHOLMOD is separately licensed under the Apache License
2.0.
