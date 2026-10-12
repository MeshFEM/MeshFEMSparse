////////////////////////////////////////////////////////////////////////////////
// SparsityLRU.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Implements a strategy for reducing the number of symbolic factorizations
//  needed for problems with occasionally changing sparsity patterns, e.g.,
//  for elasticity simulations with contact.
//
//  The idea is to retain up to a configurable number of previously occupied
//  entries in case they reappear in upcoming iterations. When this budget is
//  exceeded, the least recently used (LRU) entries are evicted the next time
//  the sparsity pattern needs to be rebuilt.
//
//  Author:  Julian Panetta (jpanetta), jpanetta@ucdavis.edu
//  Company:  University of California, Davis
//  Created:  05/11/2025 12:05:06
*///////////////////////////////////////////////////////////////////////////////
#ifndef SPARSITYLRU_HH
#define SPARSITYLRU_HH

#include <MeshFEMSparse/SparseMatrices.hh>
#include <MeshFEMCore/GlobalBenchmark.hh>
#include <stdexcept>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>

namespace MeshFEM {

struct SparsityLRU {
    // The full sparsity pattern consists of a (typically large) static part
    // and (typically small) dynamic part. Entries in the static part are
    // assigned a special age to indicate they should never be removed.
    static constexpr int STATIC_ENTRY_AGE = -1;
    static constexpr int EXPIRED = std::numeric_limits<int>::max();
    using SpMat = SuiteSparseMatrix; // Base class of BlockCSCHessianBase...
    using Index = typename SpMat::index_type;

    SparsityLRU(const SpMat &S_static)
        : m_S(S_static, /* sparsityOnly = */ true), m_entryAge(S_static.Ai.size(), STATIC_ENTRY_AGE), m_numStaticEntries(S_static.Ai.size()) { }

    // Remove entries from the cache for which `shouldRemove` returns `true`.
    // Side effect: recomputes `m_maxAge` to account for the entry removal
    // (this operation typically lowers the maximum age).
    template<typename Predicate>
    Index pruneEntries(const Predicate &shouldRemove) {
        BENCHMARK_SCOPED_TIMER_SECTION timer("SparsityLRU.pruneEntries");
        if (m_S.Ax.size() != 0) throw std::runtime_error("SparsityLRU: pruneEntries called on non-sparsity-only matrix");
        Index removed = 0;
        size_t back = 0;
        Index colStart = m_S.Ap[0];
        m_maxAge = std::numeric_limits<int>::lowest();
        for (Index j = 0; j < m_S.n; ++j) {
            Index colEnd = m_S.Ap[j + 1];
            for (Index ii = colStart; ii < colEnd; ++ii) {
                if (shouldRemove(ii)) { ++removed; continue; }
                m_S.Ai[back] = m_S.Ai[ii];
                m_entryAge[back] = m_entryAge[ii];
                m_maxAge = std::max(m_maxAge, m_entryAge[back]);
                ++back;
            }
            m_S.Ap[j + 1] = back;
            colStart = colEnd;
        }

        m_S.nz = back;
        m_S.Ai.conservativeResize(back);
        m_entryAge.resize(back);

        return removed;
    }

    void resetAge(Index ii) {
        if (m_entryAge[ii] != STATIC_ENTRY_AGE)
            m_entryAge[ii] = 0;
    }

    // The routines that mutate the cached sparsity pattern communicate
    // more than whether the pattern has changed: they also indicate
    // separately the number of entries that have been added and removed,
    // and whether the changes were forced by a "hard expiration".
    struct UpdateInfo {
        Index newEntries = 0, removedEntries = 0;
        bool hardExpirationTriggered = false;
        // Allow the info record also to be cast to a boolean indicating whether
        // the cached pattern has changed or not.
        bool changed() const { return (newEntries != 0) || (removedEntries != 0); }
        explicit operator bool() const { return changed(); }
    };

    // When the Newton optimizer requests a sparsity pattern update
    // but nothing actually changes (i.e., the dynamic part is the
    // same as the last time), we still want to increment the ages of
    // all the old pattern entries in case they should expire.
    // This method does not check for budget excedance since
    // it assumes that any old entries already stored in the cache
    // fit within the budget.
    //
    // Specifically, only the entries that are older than `threshold` are aged.
    UpdateInfo increaseAgeOfOldEntries(int threshold = 0) {
        UpdateInfo info;
        if (m_maxAge > threshold) {
            // std::cout << "SparsityLRU: increaseAgeOfOldEntries called with over-threshold maxAge = " << m_maxAge << std::endl;
            int maxAge = STATIC_ENTRY_AGE;
            for (int &age : m_entryAge) {
                if (age > threshold) {
                    ++age;
                    maxAge = std::max(maxAge, age);
                }
            }
            assert(maxAge == m_maxAge + 1);
            info.hardExpirationTriggered = maxAge >= hardExpirationAge;
            if (info.hardExpirationTriggered) {
                info.removedEntries = pruneEntries([this](Index i) { return m_entryAge[i] >= expirationAge; });
                // std::cout << "increaseAgeOfOldEntries: hard expiration triggered, maxAge = " << m_maxAge << std::endl;
                return info;
            }
            m_maxAge = maxAge;
        }
        return info;
    }

    // Update the cached sparsity pattern based on the new contents of the dynamic part.
    // The return value indicates how many entries have been added, how many
    // have been removed, and whether a hard-expiration has been triggered
    // (a flush of especially old entries despite no new entries appearing).
    // The return value also can be cast to a `bool` indicating whether
    // the cached pattern has changed.
    UpdateInfo update(const SpMat &S_dynamic) {
        BENCHMARK_SCOPED_TIMER_SECTION timer("SparsityLRU.update");
        UpdateInfo info;

        if (m_entryAge.size() != size_t(m_S.Ai.size())) throw std::runtime_error("SparsityLRU: age size mismatch");
        if ((S_dynamic.m != m_S.m) || (S_dynamic.n != m_S.n)) throw std::runtime_error("SparsityLRU: S_dynamic size mismatch");
        if (S_dynamic.symmetry_mode != m_S.symmetry_mode) throw std::runtime_error("Symmetry mode mismatch");

        // Fast path: if the dynamic sparsity pattern is empty, **and if the budget is not exceeded**,
        // we can just increment the ages of all non-static
        // entries in the cache and prune upon hard expiration.
        if ((S_dynamic.nz == 0) && ((m_entryAge.size() - m_numStaticEntries) <= entryCacheBudgetRatio * m_numStaticEntries))
            return increaseAgeOfOldEntries(STATIC_ENTRY_AGE);

        int maxAge = STATIC_ENTRY_AGE;

        // Merge entries from the dynamic sparsity pattern into the cache.
        // First, update ages and count how many new entries are needed in each column.
        Index totalCurrentEntries = 0; // Entries actually present in the current "static + dynamic" sparsity pattern
        Index totalNewEntries = 0;     // Current entries that are not already in the cache.
        m_oldEntryLocations.clear();   // Non-static entries that are not present in the new dynamic sparsity pattern
        auto &newEntries = m_numNewEntriesPerColumnScratch;
        newEntries.resize(m_S.n);
        {
        // BENCHMARK_SCOPED_TIMER_SECTION t("parallel merge 1");
        struct ChunkResult {
            int maxAge = STATIC_ENTRY_AGE;
            Index totalCurrentEntries = 0, totalNewEntries = 0;
            std::vector<EntryLoc> oldEntryLocations;
        };
        constexpr Index chunkSize = 256;
        const Index numChunks = m_S.n / chunkSize + (m_S.n % chunkSize != 0);
        std::vector<ChunkResult> chunks(numChunks);
        tbb::parallel_for(Index(0), numChunks, [&](Index c) {
        auto &chunk = chunks[c];
        const Index end = std::min(m_S.n, (c + 1) * chunkSize);
        for (Index j = c * chunkSize; j < end; ++j) {
            newEntries[j] = 0;
            Index ii_S =       m_S.Ap[j], ii_S_end =       m_S.Ap[j + 1];
            Index ii_D = S_dynamic.Ap[j], ii_D_end = S_dynamic.Ap[j + 1];

            // Handle entries that don't appear in `S_dynamic`
            // (increase their age and mark them as old)
            auto entryInSOnly = [&ii_S, &chunk, j, this]() {
                if (m_entryAge[ii_S] != STATIC_ENTRY_AGE) {
                    chunk.maxAge = std::max(chunk.maxAge, ++m_entryAge[ii_S]);
                    chunk.oldEntryLocations.emplace_back(j, ii_S);
                }
                else ++chunk.totalCurrentEntries;
                ++ii_S;
            };

            // Handle entries that don't appear in `S` yet
            // (count new entries)
            auto entryInDOnly = [&ii_D, &chunk, &newEntries, j]() {
                ++newEntries[j]; ++chunk.totalNewEntries; ++chunk.totalCurrentEntries;
                ++ii_D;
            };

            while ((ii_S < ii_S_end) && (ii_D < ii_D_end)) {
                if (m_S.Ai[ii_S] == S_dynamic.Ai[ii_D]) {
                    resetAge(ii_S);
                    // Active dynamic entries outside the static pattern contribute age zero;
                    // we must update `maxAge` to account for these (lest it stay at STATIC_ENTRY_AGE).
                    chunk.maxAge = std::max(chunk.maxAge, m_entryAge[ii_S]);
                    ++chunk.totalCurrentEntries;
                    ++ii_S; ++ii_D;
                }
                else if (m_S.Ai[ii_S] < S_dynamic.Ai[ii_D]) {
                    entryInSOnly();
                }
                else entryInDOnly();
            }

            // Handle remaining entries in S and D
            while (ii_S < ii_S_end) entryInSOnly();
            while (ii_D < ii_D_end) entryInDOnly();
        }
        });
        size_t numOldEntries = 0;
        for (const auto &chunk : chunks) {
            maxAge = std::max(maxAge, chunk.maxAge);
            totalCurrentEntries += chunk.totalCurrentEntries;
            totalNewEntries += chunk.totalNewEntries;
            numOldEntries += chunk.oldEntryLocations.size();
        }
        m_oldEntryLocations.reserve(numOldEntries);
        for (const auto &chunk : chunks)
            m_oldEntryLocations.insert(m_oldEntryLocations.end(), chunk.oldEntryLocations.begin(), chunk.oldEntryLocations.end());
        }

        // if (maxAge > 0)
        //     std::cout << "Positive maxAge: " << maxAge << std::endl;
        if (verbose) std::cout << "totalNewEntries: " << totalNewEntries << std::endl;
        info.newEntries = totalNewEntries;

        // Semi-fast path: reuse (potentially pruned) cache if no new entries were added **and if the budget is not exceeded.**
        if (totalNewEntries == 0) {
            if (maxAge >= hardExpirationAge) {
                info.hardExpirationTriggered = true;
                info.removedEntries = pruneEntries([this](Index i) { return m_entryAge[i] >= expirationAge; });
                if (verbose) std::cout << "update: hard expiration triggered, maxAge = " << m_maxAge << std::endl;
                return info;
            }
            m_maxAge = maxAge;
            if (m_oldEntryLocations.size() <= totalCurrentEntries * entryCacheBudgetRatio)
                return info;
        }

        // Doing a full rebuild because new entries were added!
        // BENCHMARK_SCOPED_TIMER_SECTION t_rebuild("rebuild");
        SpMat S_new(m_S.m, m_S.n);
        S_new.symmetry_mode = m_S.symmetry_mode;
        SpMat::InOrderBuilder builder(S_new, [&](Index *colSizes) {
            // Count the number of nonzeros in each column of the new sparsity cache.
            // Start with old + new entries...
            for (Index j = 0; j < m_S.n; ++j)
                colSizes[j] = m_S.col_nnz(j) + newEntries[j];

            // ... then determine how many of the old entries must be dropped.
            Index budget = totalCurrentEntries * entryCacheBudgetRatio;
            Index numOldEntries = m_oldEntryLocations.size();
            // Subtract the entries to be removed.
            if (numOldEntries > budget) {
                // Keep only the newest `budget` entries, implemented by sorting
                // entries non-descending by age.
                // TODO: use a better algorithm based on order statistics?
                // First, sort the entries non-descending by age.
                std::sort(m_oldEntryLocations.begin(), m_oldEntryLocations.end(),
                          [this](const EntryLoc &a, const EntryLoc &b) { return m_entryAge[a.ii] < m_entryAge[b.ii]; });
            }

            if (verbose) std::cout << "Number of old entries: " << numOldEntries << ", budget: " << budget << std::endl;

            for (Index i = 0; i < numOldEntries; ++i) {
                const EntryLoc &loc = m_oldEntryLocations[i];
                if (i >= budget)
                    m_entryAge[loc.ii] = EXPIRED; // Mark the entries to be removed due to exceeding budget
                if (m_entryAge[loc.ii] >= expirationAge) {
                    --colSizes[loc.j];
                    ++info.removedEntries;
                }
            }
        }, /* sparsityOnly = */ true);

        // Fill in the new cache and associated entry ages by doing
        // another merge of the cached and dynamic sparsity patterns.
        std::vector<int> &new_ages = m_newEntryAgeScratch;
        {
        // BENCHMARK_SCOPED_TIMER_SECTION t("new_ages initialization");
        new_ages.resize(S_new.Ai.size());
        }

        {
        // BENCHMARK_SCOPED_TIMER_SECTION t("parallel merge 2");
        m_maxAge = tbb::parallel_reduce(tbb::blocked_range<Index>(0, m_S.n, 64),
            std::numeric_limits<int>::lowest(), [&](const auto &range, int localMaxAge) {
        for (Index j = range.begin(); j < range.end(); ++j) {
            // The builder stores this column's initial offset/cursor in Ap[j + 1].
            // Each column has one writer; Ap[j] may be changing on another thread.
            const Index columnBegin = S_new.Ap[j + 1];
            Index ii_out = columnBegin;
            auto recordAge = [&](int age) {
                new_ages[ii_out++] = age;
                localMaxAge = std::max(localMaxAge, age);
            };
            Index ii_S =       m_S.Ap[j], ii_S_end =       m_S.Ap[j + 1];
            Index ii_D = S_dynamic.Ap[j], ii_D_end = S_dynamic.Ap[j + 1];

            auto entryInSOnly = [&builder, &ii_S, &recordAge, j, this]() {
                if (m_entryAge[ii_S] < expirationAge) {
                    builder.insert(m_S.Ai[ii_S], j);
                    recordAge(m_entryAge[ii_S]);
                }
                ++ii_S;
            };

            auto entryInDOnly = [&builder, &ii_D, &recordAge, j, &S_dynamic]() {
                builder.insert(S_dynamic.Ai[ii_D], j);
                recordAge(0);
                ++ii_D;
            };

            while ((ii_S < ii_S_end) && (ii_D < ii_D_end)) {
                if (m_S.Ai[ii_S] == S_dynamic.Ai[ii_D]) { // Entry in both S and D
                    builder.insert(m_S.Ai[ii_S], j);
                    recordAge(std::min(0, m_entryAge[ii_S])); // Make sure negative ages (e.g., STATIC_ENTRY_AGE) are retained!
                    ++ii_S; ++ii_D;
                }
                else if (m_S.Ai[ii_S] < S_dynamic.Ai[ii_D])
                    entryInSOnly();
                else entryInDOnly();
            }

            // Handle remaining entries in S and D
            while (ii_S < ii_S_end) entryInSOnly();
            while (ii_D < ii_D_end) entryInDOnly();

            // Verify that S_new has all of its diagonal entries.
            if ((ii_out == columnBegin) || (S_new.Ai[ii_out - 1] != j))
                throw std::runtime_error("SparsityLRU: missing diagonal entry in new sparsity pattern");
        }
        return localMaxAge;
        }, [](int a, int b) { return std::max(a, b); });
        }

        m_S = std::move(S_new);
        m_entryAge.swap(new_ages);

#ifndef NDEBUG
        {
            // Validation
            int static_age_count = 0;
            for (int age : m_entryAge) static_age_count += (age == STATIC_ENTRY_AGE);
            assert(static_age_count == m_numStaticEntries);
        }

        if (*std::max_element(m_entryAge.begin(), m_entryAge.end()) != m_maxAge)
            throw std::runtime_error("SparsityLRU: max age mismatch after update");
#endif

        // if (totalNewEntries != 0)
        //     std::cout << "update: update triggered, maxAge = " << m_maxAge << std::endl;

        return info;
    }

    const std::vector<int> &entryAges() const { return m_entryAge; }

    const SpMat &operator*() const { return m_S; }
    const SpMat *operator->() const { return &m_S; }

    // Maximum number of old entries to retain in the cache
    double entryCacheBudgetRatio = 0.01;

    // Entries older than this are removed regardless of whether the budget is exceeded.
    int expirationAge = 25;
    int hardExpirationAge = 250; // Entries older than this trigger a rebuild.
    bool verbose = true;

private:
    SpMat m_S;
    Eigen::Array<Index, Eigen::Dynamic, 1> m_numNewEntriesPerColumnScratch;
    std::vector<int> m_entryAge, m_newEntryAgeScratch;
    struct EntryLoc {
        EntryLoc(Index j_, Index ii_) : j(j_), ii(ii_) { }
        Index j, ii; // Column index and location in `Ai`, respectively
    };

    int m_maxAge = STATIC_ENTRY_AGE;
    int m_numStaticEntries = 0;

    std::vector<EntryLoc> m_oldEntryLocations; // kept as a member variable to reduce memory allocations.
};

} // namespace MeshFEM

#endif /* end of include guard: SPARSITYLRU_HH */
