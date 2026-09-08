# Optional SCOTCH Bisection

Build with `MESHFEM_WITH_SCOTCH=ON` and SCOTCH 7 or newer. The parallel nested
dissection ordering can then use `SCOTCH_graphPartOvl` for its top few
bisections, followed by the usual CHOLMOD/METIS bisector below the cutoff.
This does not use SCOTCH's full nested dissection ordering: CHOLMOD's graph
compression, separator-tree traversal, and final CAMD processing remain in place.
Select `CholmodNesdisParallel` as usual in the Python factorizer.

## Configuration

Environment variables are read once per ordering invocation:

| Variable | Default | Meaning |
| --- | --- | --- |
| `CHOLMOD_NESDIS_SCOTCH_LEVELS` | `0` | Number of levels using SCOTCH. `0` disables it, `1` selects roots only, `2` includes their children. |
| `CHOLMOD_NESDIS_SCOTCH_THREADS` | Current TBB concurrency limit | SCOTCH threads at a root, capped by the TBB limit and any `CHOLMOD_NESDIS_NUM_THREADS` override. |
| `CHOLMOD_NESDIS_SCOTCH_STRATEGY` | `fast` | Named strategy or a raw SCOTCH overlap-partition strategy string. |
| `CHOLMOD_NESDIS_SCOTCH_SEED` | `0` | Nonnegative seed for the private RNG, reset before each bisection. Parallel matching can still vary between runs. |
| `CHOLMOD_NESDIS_SCOTCH_TRACE` | `0` | Set to `1` to print `NESDIS_BISECTOR` JSON records for the selected levels (or the METIS root when disabled). |
| `CHOLMOD_NESDIS_SCOTCH_VERIFY` | `0` | Set to `1` to validate that no edge crosses the two interiors. This adds a full adjacency scan for debugging. |

The root is depth zero. Disconnected input components each have a root at that
depth; `LEVELS=1` therefore does not necessarily mean exactly one SCOTCH call.
The bisector is not called at all for components that the existing traversal
decides are already small enough or cannot be split.

For example, from a Python benchmarking directory:

```sh
CHOLMOD_NESDIS_SCOTCH_LEVELS=1 \
CHOLMOD_NESDIS_SCOTCH_THREADS=8 \
CHOLMOD_NESDIS_SCOTCH_STRATEGY=fast2 \
python benchmark_matrixmarket.py /path/to/hessian.mtx 3
```

The script's TBB thread setting must also allow eight threads. All other ordering
options, including `CHOLMOD_NESDIS_CAMD_CUT_DEPTH`, retain their usual meaning.

## Strategies

All named strategies request two overlapping parts, hence one vertex separator.
Their balance tolerance is `0.2`, matching CHOLMOD/METIS's requested tolerance,
but their algorithms and balance objectives are not identical.

- `recursive`: SCOTCH's generated overlap strategy with `SCOTCH_STRATRECURSIVE`.
- `default`: SCOTCH's generated overlap strategy with `SCOTCH_STRATQUALITY` (an explicit choice, not the adapter's default).
- `fast` (default): `r{sep=m{vert=100,rat=0.7,low=h{pass=1},asc=f{bal=0.2}}}`.
- `fast2` and `fast4`: two or four alternatives of the `fast` inner separator
  strategy, joined with SCOTCH's `|` selection operator. Attempts run sequentially;
  each attempt can use parallel coarsening.

A raw strategy string makes further experiments possible without changing C++.
Smaller root separators alone do not guarantee better full-ordering fill or
numeric factorization time. Measure those quantities as well as bisection time.

## Implementation and Limits

`nesdis_bisector.cc` owns the optional integration. Each active TBB thread has
its own SCOTCH context, strategy, and RNG. There is no lock around bisection.
SCOTCH's initial global RNG initialization is performed once before private
contexts are cloned concurrently. Contexts are destroyed when the ordering ends.

SCOTCH uses its own workers, not TBB workers. The requested SCOTCH thread count
is halved at each deeper level, with a minimum of one. This limits oversubscription
for balanced binary traversal, but is not a strict combined thread budget:
disconnected roots and overlapping work at different depths can oversubscribe.
This option is intended for one or two expensive upper levels, not the whole tree.
Use a threaded SCOTCH build to get parallel coarsening.

Both 32-bit and 64-bit CHOLMOD interfaces are supported. Arrays are converted
when SCOTCH's integer type differs; graphs whose dimensions or aggregate node
weights exceed that type's range use METIS instead. Edge weights are ignored,
as in the current CHOLMOD/METIS bisector. The adapter checks labels and normalizes
empty separators/interiors to CHOLMOD's conventions. The additional full edge
validation is disabled unless `CHOLMOD_NESDIS_SCOTCH_VERIFY=1`.
SCOTCH failures propagate as CHOLMOD errors, rather
than silently switching algorithms. The non-exiting `scotcherr` library is used.

Trace times include adapter setup, conversions, and validation. Separator weights
refer to the graph passed to the bisector, potentially after compression. For
Catamari's block ordering they count block vertices, not scalar degrees of freedom.
