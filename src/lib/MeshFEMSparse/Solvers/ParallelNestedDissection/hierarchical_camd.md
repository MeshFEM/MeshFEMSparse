# Hierarchical CAMD

`hierarchical_camd.hh/cc` implements a single coarse split of the existing ND
forest. The parallel nesdis path uses it by default at depth 4 for symmetric
input with `nd_camd == 1`; global CAMD remains available as an override.

## Selection

Select `CholmodNesdisParallel` as the Catamari ordering. To override its default depth:

```sh
CHOLMOD_NESDIS_CAMD_CUT_DEPTH=2 \
python benchmark_matrixmarket.py matrix.mtx 3
```

- `CHOLMOD_NESDIS_CAMD_CUT_DEPTH`: defaults to 4. Set -1 to force one global
  CAMD call, or a nonnegative value to select the hierarchical cut. Roots have depth zero;
  depths 1, 2, 3, 4 usually give 2, 4, 8, 16 tasks for a single binary tree.
  Forests, disconnected subdomains, and shallow branches can change that count.
  Depth zero orders independent connected components with no upper problem.
  The cut is clamped to the deepest level present in the forest. Branches ending
  above the cut remain in the upper problem. A forest with only one component
  uses global CAMD; independent roots without subdivision can use depth zero.
  The `CAMD_HIERARCHICAL.cut_depth` trace field reports the effective cut.
- `CHOLMOD_NESDIS_CAMD_TRACE=1`: JSON records prefixed by `CAMD_TASK`,
  `CAMD_HIERARCHICAL`, `CAMD_TOTAL`, and `NESDIS_TOTAL`, written to stderr.
- `CHOLMOD_NESDIS_CAMD_TRACE_TREE=1`: optional `CAMD_TREE` records with each
  component's size, depth, parent and subtree size; useful for baseline analysis.
- `CHOLMOD_NESDIS_CAMD_VERIFY=1`: checks the permutation and ancestor ordering;
  for graphs of at most 512 vertices, also independently simulates elimination
  and checks the exact remaining upper graph.
- `CHOLMOD_NESDIS_CAMD_SKIP_UPPER=1`: retain local CAMD but skip upper CAMD,
  boundary discovery and quotient-graph construction. Upper vertices are appended
  in separator postorder, with original vertex order within each component.
  This preserves contiguous groups for Catamari's preliminary assembly forest.
  The default is `0`. With this option, verification still checks the permutation
  and ancestry, but there is no upper graph to verify. Upper graph/CAMD times
  and boundary statistics are zero; `CAMD_HIERARCHICAL.skip_upper` reports the mode.

Skipping upper CAMD trades ordering quality for speed, particularly at deeper
cuts. It does not exactly reproduce `cholmod_metis`: the existing bisector,
separator tree and constrained local CAMD remain unchanged. Components ending
above the cut remain in the upper problem and are also left naturally ordered.
`CUT_DEPTH=-1` still explicitly selects global CAMD, regardless of SKIP_UPPER;
the single-component fallback is unchanged. For example:

```sh
CHOLMOD_NESDIS_CAMD_CUT_DEPTH=4 CHOLMOD_NESDIS_CAMD_SKIP_UPPER=1 \
python benchmark_matrixmarket.py matrix.mtx 3
```

The existing TBB limit controls local CAMD concurrency. As with partitioning,
`CHOLMOD_NESDIS_NUM_THREADS` optionally adds a `tbb::global_control` limit.
Hierarchical CAMD applies only to symmetric input with `nd_camd == 1`.
Unsymmetric CCOLAMD, `nd_camd == 2` (CSYMAMD), and `nd_camd == 0` retain their
existing behavior. Invalid settings fail the ordering rather than
silently choosing another configuration.

## Algorithm and existing CHOLMOD data flow

No new tree representation is needed. `CParent` is a postordered component
forest, with every parent numbered after its children. `Cmember[v]` is the
component/constraint label for the original vertex `v`. Reverse traversal of
`CParent` computes depths and coarse-task roots; a single scan of `Cmember`
assigns vertices to tasks or to the upper problem. This costs O(n + components).

Each task orders its original induced graph using stock `cholmod_[l_]camd`.
Constraint IDs are restricted and remapped in sorted order, preserving the
original hierarchy inside that subtree. Local problems currently omit ancestor
boundary vertices when estimating degrees. All task orderings are concatenated
in deterministic component-postorder order, followed by the upper ordering.
This preserves descendant-before-ancestor ordering, but deliberately permits
independent branches to commute relative to global numeric constraint labels.

Partition compression and dense-node removal have already been undone in
`Cmember`. The final CAMD stage uses a fresh copy of the original symmetric
matrix graph (or borrows the supplied full graph), not the compressed partition
graph. Dense nodes remain represented
in the separator forest. Each CAMD call has a freshly initialized CHOLMOD common
and independent workspace, copying only the selected method settings. CAMD's
normal dense-node and aggressive-absorption settings apply to local calls.

A partition/subtree is not assumed connected: the upstream partition helper
explicitly allows disconnected subgraphs. Each task therefore computes connected
components in its owned induced graph. For each such component it scans original
adjacency and returns the unique upper vertices touched. Boundaries can reach
multiple ancestors. Edges crossing distinct coarse tasks are rejected, since
they contradict the independence required by this decomposition.

Temporary graph arrays and searches are task-sized. The only global index maps
are shared read-only. There are no shared CAMD workspace arrays, and no task
allocates an O(global n) private map. CAMD returns `Perm[k] = original_vertex`;
local permutations are mapped through their vertex lists before concatenation.

## Quotient elements through stock CAMD

For each connected eliminated component with live boundary B, elimination
induces exactly a clique on B, regardless of the local elimination order.
Rather than materializing these O(sum |B|^2) clique edges, the upper graph
represents each boundary using a synthetic vertex. Explicit clique construction
is retained only as a small-graph correctness reference in the tests and verifier.

The trace reports boundary component counts, maximum/summed boundary sizes,
sum of squared boundary sizes, synthetic vertex count,
and the actual upper input edge count. Construction and upper CAMD are timed
separately; local-phase wall time includes graph extraction, CAMD and summaries.

Each boundary clique is replaced with a synthetic vertex
connected to exactly B. All synthetic vertices belong to the first constraint
group; real upper vertices keep their relative constraint order in later groups.
Eliminating a synthetic star creates the same clique structurally. CAMD stores
that contribution as one of its ordinary quotient-graph elements, so the clique
is never materialized in the input. Only the real upper suffix of the resulting
permutation is retained.

This uses stock CAMD throughout; there is no fork of `camd_2` and no manual
initialization or merging of internal quotient workspaces. It does perform a
small synthetic-pivot phase rather than entering CAMD with preexisting elements.
Its input storage is O(original upper edges + sum |B|).
Boundaries are deduplicated and each synthetic vertex is distinct, so no global
edge sort is needed. There is no longer an upper-representation mode switch.

Dense-node pruning must be disabled for this augmented upper call: otherwise
CAMD can defer/remove a synthetic vertex without building its boundary element.
Even a negative `prune_dense` still removes completely dense vertices upstream.
The augmented call instead sets `alpha = sqrt(augmented_n)`, making the threshold at
least the maximum possible degree. Local CAMD calls keep their original settings.
The augmented ordering can differ from CAMD on a materialized fill graph because its initial
degree history, absorption and upper dense-pruning policy differ. Identical
permutations are not required; exact structural summaries and valid tree order
are required.

`Common->lnz` and `Common->fl` are set to -1 on hierarchical success. Summing
local estimates would omit boundary interactions, and the element problem's
estimates include synthetic vertices. Use the subsequent factorizer's actual
symbolic estimates for comparisons.

## Validation and benchmarks

Catamari's serial and parallel nesdis paths now request a preliminary assembly
forest from `CholmodOrdering`. It groups the final permutation by `Cmember` and
remaps `CParent` into that group order. Catamari uses these groups to schedule
its existing parallel computation of exact scalar parents and column degrees.
The groups are scheduling regions, not fundamental supernodes, and the final
permutation is unchanged. This also works with global CAMD and requires no
additional option; Catamari's usual TBB thread limit controls the parallelism.
Other ordering users may omit the optional forest output.

The wrapper tests cover both integer widths, isolated vertices, global and
hierarchical CAMD, unchanged permutations, and exact agreement between Catamari's
serial and parallel scalar symbolic results with one and four threads.

The `[hierarchical_camd]` tests cover both index widths, multiple thread counts,
random graphs respecting an ND forest, disconnected components, empty graphs,
cuts beyond tree height, ancestor boundaries, a dense boundary, and actual
parallel-nesdis calls. For small graphs, the verifier compares symbolic
elimination in the original graph with elimination in the actual upper input
(including synthetic vertices).

MeshFEM's `python/benchmarking/benchmark_hierarchical_camd.py` runs serial nesdis,
parallel nesdis/global CAMD (`-1`), the default configuration, hierarchical CAMD
at depths 1-4, and METIS using
Catamari's block-matrix path. Example from the MeshFEM root:

```sh
python python/benchmarking/benchmark_hierarchical_camd.py matrix.mtx \
    --block-size 3 --threads 8 --repeats 3 --output build/camd_comparison
```

It discards one warmup per case and saves raw output/traces plus JSON summaries
with median timings, factor nonzeros, estimated flops and checked solve residuals.
`--cases global elements_cut2` restricts the sweep. Failed factorizations or bad
residuals are reported as failures instead of silently skipped.

`NESDIS_TOTAL.partition_seconds` includes partition workspace setup, connected
component discovery and the recursive partition traversal. Total ordering also
includes graph conversion, tree cleanup and final CAMD. Native CAMD timing
excludes any top-level symmetric-copy step; total ordering includes it when needed.
The serial upstream and METIS baselines have whole symbolic-factorization
timings but no new internal instrumentation.
