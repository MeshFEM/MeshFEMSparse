////////////////////////////////////////////////////////////////////////////////
// compress_sparsity_pattern.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Compress a scalar sparsity pattern into a block sparsity pattern with
//  uniform block size (i.e., the sparsity pattern of a BCSC format).
//  A block nonzero is created if any input scalar nonzero falls within that
//  block; not all entries within a block need to be present in the scalar
//  pattern.
//
//  Author:  Julian Panetta (jpanetta), jpanetta@ucdavis.edu
//  Company:  University of California, Davis
//  Created:  08/05/2026 19:00:49
*///////////////////////////////////////////////////////////////////////////////
#ifndef COMPRESS_SPARSITY_PATTERN_HH
#define COMPRESS_SPARSITY_PATTERN_HH

#include <vector>
#include <set>

namespace MeshFEM {

template<typename Index>
void compress_sparsity_pattern(const Index *Ai, const Index *Ap, Index n, const int blockSize, std::vector<Index> &blockAi, std::vector<Index> &blockAp, bool keepUpperTriangleOnly) {
	if (n % blockSize != 0) throw std::runtime_error("compress_sparsity_pattern: n must be divisible by blockSize");
	Index n_block = n / blockSize;
	blockAp.resize(n_block + 1);
	blockAp[0] = 0;
	for (Index bj = 0; bj < n_block; ++bj) {
		// TODO: faster merge-based approach if this is ever a bottleneck.
		std::set<Index> uniqueBlocks;
		const Index end = Ap[(bj + 1) * blockSize];
		for (Index ii = Ap[bj * blockSize]; ii < end; ++ii) {
			Index bi = Ai[ii] / blockSize;
			if (keepUpperTriangleOnly && (bi > bj)) continue;
			uniqueBlocks.insert(bi);
		}
		blockAp[bj + 1] = blockAp[bj] + uniqueBlocks.size();
		blockAi.insert(blockAi.end(), uniqueBlocks.begin(), uniqueBlocks.end());
	}
}

} // namespace MeshFEM

#endif /* end of include guard: COMPRESS_SPARSITY_PATTERN_HH */
