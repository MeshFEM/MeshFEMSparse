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
#include <stdexcept>

#include <MeshFEMSparse/Utilities/ScalarCSCView.hh>

namespace MeshFEM {

template<typename SrcIndex, typename DstIndex>
void compress_sparsity_pattern(const SrcIndex *Ai, const SrcIndex *Ap, DstIndex n, const int blockSize, std::vector<DstIndex> &blockAi, std::vector<DstIndex> &blockAp, bool keepUpperTriangleOnly) {
	if (n % blockSize != 0) throw std::runtime_error("compress_sparsity_pattern: n must be divisible by blockSize");
	DstIndex n_block = n / blockSize;
	blockAp.resize(n_block + 1);
	blockAp[0] = 0;
	for (DstIndex bj = 0; bj < n_block; ++bj) {
		// TODO: faster merge-based approach if this is ever a bottleneck.
		std::set<DstIndex> uniqueBlocks;
		const SrcIndex end = Ap[(bj + 1) * blockSize];
		for (SrcIndex ii = Ap[bj * blockSize]; ii < end; ++ii) {
			DstIndex bi = Ai[ii] / blockSize;
			if (keepUpperTriangleOnly && (bi > bj)) continue;
			uniqueBlocks.insert(bi);
		}
		blockAp[bj + 1] = blockAp[bj] + uniqueBlocks.size();
		blockAi.insert(blockAi.end(), uniqueBlocks.begin(), uniqueBlocks.end());
	}
}

template<typename Index>
void compress_sparsity_pattern(const ScalarCSCView &src, const int blockSize, std::vector<Index> &blockAi, std::vector<Index> &blockAp, bool keepUpperTriangleOnly) {
	if (src.rows != src.cols) throw std::runtime_error("compress_sparsity_pattern: only square matrices are supported");
	src.visit([&](const auto *src_Ap, const auto *src_Ai, const double *) {
		compress_sparsity_pattern(src_Ai, src_Ap, Index(src.cols), blockSize, blockAi, blockAp, keepUpperTriangleOnly);
	});
}

} // namespace MeshFEM

#endif /* end of include guard: COMPRESS_SPARSITY_PATTERN_HH */
