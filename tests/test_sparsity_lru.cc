#include <MeshFEMSparse/SparsityLRU.hh>
#include <catch2/catch.hpp>

using namespace MeshFEM;

namespace {
SuiteSparseMatrix pattern(std::initializer_list<std::pair<size_t, size_t>> edges) {
    SuiteSparseMatrix result(4, 4);
    result.symmetry_mode = SuiteSparseMatrix::SymmetryMode::UPPER_TRIANGLE;
    SuiteSparseMatrix::InOrderBuilder builder(result, [&](auto *sizes) {
        for (auto edge : edges) ++sizes[edge.second];
    }, true);
    for (auto edge : edges) builder.insert(edge.first, edge.second);
    return result;
}
}

TEST_CASE("SLRU ages active entries when the dynamic pattern becomes empty", "[slru]") {
    const auto stationary = pattern({{0, 0}, {1, 1}, {2, 2}, {3, 3}});
    const auto dynamic = pattern({{0, 2}, {1, 3}});
    const auto empty = pattern({});
    SparsityLRU cache(stationary);
    cache.verbose = false;
    cache.entryCacheBudgetRatio = 2; // Enable the empty-pattern fast path.
    cache.expirationAge = 2;
    cache.hardExpirationAge = 3;
    REQUIRE(cache.update(dynamic).newEntries == 2);
    SparsityLRU::Index remaining = 2;

    SECTION("after an unchanged update") {
        REQUIRE_FALSE(cache.update(dynamic).changed());
    }
    SECTION("after pruning leaves an active entry") {
        const bool ageOnly = GENERATE(false, true);
        const auto retained = pattern({{1, 3}});
        REQUIRE_FALSE(cache.update(retained).changed());
        auto advance = [&]() {
            return ageOnly ? cache.increaseAgeOfOldEntries() : cache.update(retained);
        };
        REQUIRE_FALSE(advance().changed());
        const auto expired = advance();
        REQUIRE(expired.hardExpirationTriggered);
        REQUIRE(expired.removedEntries == 1);
        remaining = 1;
    }

    for (int age = 1; age <= 3; ++age) {
        const auto info = cache.update(empty);
        REQUIRE(info.newEntries == 0);
        REQUIRE(info.hardExpirationTriggered == (age == 3));
        REQUIRE(info.removedEntries == (age == 3 ? remaining : 0));
        REQUIRE(info.changed() == (age == 3));
    }
    REQUIRE(cache->nz == stationary.nz);
}
