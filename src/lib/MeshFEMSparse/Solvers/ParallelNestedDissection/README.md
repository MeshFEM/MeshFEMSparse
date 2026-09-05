# ParallelNestedDissection

This directory contains a modified version of cholmod_nesdis.c that introduces
TBB-based parallelism (i.e., a parallel traversal of the separator tree). The
original iterative algorithm was refactored to use explicit recursion since I
thought this would simplify thread scheduling, but in retrospect this probably
was not actually necessary. The main tricky implementation point was ensuring
that state modified by the recursive calls is stored in thread-local copies.

This gets a solid speedup over the serial version (~2-3x on test matrices)
without changing the ordering. The speedup is limited by the critical
path of the separator tree, which is dominated by the global top-level
bisection done at the start. I have experimented with several
parallel partitioning codes (like mt-KaHIP and mt-Metis) for just the topmost
(or high-up) bisections but found they were either slower than the serial Metis or
produced substantially lower-quality separators; still, this seems to be the
main opportunity for further speedup.

There is also a significant serial postprocessing time introduced by the
global `camd` ordering step. This can be avoided by setting the CHOLMOD options
`nd_camd = 0` at the expense of degrading ordering quality; this degradation is
less if `nd_small` is reduced from the CHOLMOD default (200) to, e.g., `50` or `25`.
Another option would be to run AMD in parallel on the leaf nodes (essentially
a parallel version of `cholmod_metis`, which tends to be a slightly inferior
ordering). An even better option would be to implement a parallel approximation
to CAMD (parallelized using the separator tree), which is ongoing work.

## License

Since this implementation is derived from `CHOLMOD/Partition/cholmod_nesdis.c`,
whose upstream SPDX license identifier is `LGPL-2.1+`, we keep this same GNU
Lesser General Public License, version 2.1 or later. The SuiteSparse-bundled
copy of METIS used by CHOLMOD is separately licensed under the Apache License
2.0.
