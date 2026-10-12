#include <MeshFEMSparse/Solvers/AdaptiveOrderingSelection.hh>
#include <catch2/catch.hpp>

using namespace MeshFEM;

namespace {
struct Choices {
    static constexpr int primary_method = 0, alternate_method = 1;
    inline static constexpr std::array<double, 2> optimistic_sym_time_multiplier_estimates{{1, 1}};
    inline static constexpr std::array<double, 2> optimistic_num_time_multiplier_estimates{{1, 1}};
};
using Selection = AdaptiveOrderingSelection<Choices>;

void record(Selection &s, double time, AnalysisReuseInfo reuse = {}) {
    s.recordSymbolic(time, 100, reuse);
    s.recordNumeric(2);
    s.recordSolve(1);
}
}

TEST_CASE("Adaptive analysis cost amortizes full ND over P incremental analyses", "[adaptive_ordering]") {
    Selection s;
    record(s, 100, {3, true});
    auto &times = s.factorization_times_for_method[0];
    REQUIRE(times.estimatedAnalysisTime() == Approx(100));
    record(s, 10, {3, false});
    REQUIRE(times.estimatedAnalysisTime() == Approx(32.5));
    REQUIRE(times.symbolic.average() == Approx(82));
    REQUIRE(s.currentMethodSymbolicCounts() == 2);
    REQUIRE(s.currentMethodNumericCounts() == 2);
    REQUIRE(s.totalSolves() == 2);
    REQUIRE(s.timeEstimateForMethod(0) == Approx(2 * 32.5 + 2 * 2 + 2));
    // Unknown-method estimates use the amortized cost too.
    REQUIRE(s.timeEstimateForMethod(1) == Approx(s.timeEstimateForMethod(0)));

    // A full rebuild updates its cost without discarding incremental samples.
    record(s, 200, {3, true});
    REQUIRE(times.fullAnalysis.count == 2);
    REQUIRE(times.incrementalAnalysis.count == 1);
    REQUIRE(times.estimatedAnalysisTime() == Approx((120 + 3 * 10) / 4.0));

    // The full sample remains available when the workload window rolls over.
    Selection rolling;
    rolling.window_size = 1;
    record(rolling, 100, {3, true});
    record(rolling, 10, {3, false});
    REQUIRE(rolling.symbolicFactorizationsInWindow() == 1);
    REQUIRE(rolling.timeEstimateForMethod(0) == Approx(32.5 + 2 + 1));
}

TEST_CASE("Adaptive timing falls back when either reuse sample type is missing", "[adaptive_ordering]") {
    Selection::FactorizationTimes times;
    const bool full = GENERATE(false, true);
    times.recordAnalysis(100, {32, full});
    times.recordAnalysis(50, {32, full});
    REQUIRE(times.estimatedAnalysisTime() == Approx(90));
    REQUIRE(times.symbolic.count == 2);

    // Disabling/changing reuse must not mix split samples from different settings.
    times.recordAnalysis(10, {});
    REQUIRE(times.incrementalReuseLimit == 0);
    REQUIRE_FALSE(times.fullAnalysis.known());
    REQUIRE_FALSE(times.incrementalAnalysis.known());
    REQUIRE(times.estimatedAnalysisTime() == Approx(times.symbolic.average()));
    times.recordAnalysis(100, {3, true});
    times.recordAnalysis(10, {3, false});
    REQUIRE(times.estimatedAnalysisTime() == Approx(32.5));
    times.recordAnalysis(20, {1, false});
    REQUIRE_FALSE(times.fullAnalysis.known());
    REQUIRE(times.estimatedAnalysisTime() == Approx(times.symbolic.average()));
    times.recordAnalysis(100, {1, true});
    REQUIRE(times.estimatedAnalysisTime() == Approx(60));
    times.reset();
    REQUIRE(times.incrementalReuseLimit == 0);
    REQUIRE_FALSE(times.fullAnalysis.known());
    REQUIRE_FALSE(times.incrementalAnalysis.known());
    REQUIRE(times.symbolic.count == 0);
    REQUIRE(times.estimatedAnalysisTime(-1) == -1);
}

TEST_CASE("Adaptive temporal warmup applies after each switch and excludes AMD", "[adaptive_ordering]") {
    Selection s;
    s.warmup_sym_count = 0; // Isolate per-switch protection from global startup warmup.
    s.method_switch_threshold_soft = s.method_switch_threshold_hard = 0;
    s.num_factorizations_before_permitting_hard_switch = 0;
    auto &alternate = s.factorization_times_for_method[1];
    alternate.recordAnalysis(1, {});
    alternate.numeric.add(2);
    alternate.solve.add(1);

    record(s, 100, {32, true});
    REQUIRE_FALSE(s.shouldTriggerSymbolicFactorizationRecompute());
    REQUIRE(s.updateSelection() == Choices::primary_method);

    // Even a second full rebuild releases the guard; it must not restart warmup.
    record(s, 100, {32, GENERATE(false, true)});
    REQUIRE(s.shouldTriggerSymbolicFactorizationRecompute());
    REQUIRE(s.updateSelection() == Choices::alternate_method);

    record(s, 1);
    REQUIRE(s.shouldTriggerSymbolicFactorizationRecompute());
    REQUIRE(s.updateSelection() == Choices::primary_method);

    // Previous lifetime samples do not bypass the new activation's warmup.
    // This also covers returning to a surviving ND cache (first sample incremental).
    record(s, 10, {32, false});
    REQUIRE_FALSE(s.shouldTriggerSymbolicFactorizationRecompute());
    REQUIRE(s.updateSelection() == Choices::primary_method);
    record(s, 10, {32, false});
    REQUIRE(s.updateSelection() == Choices::alternate_method);
}

TEST_CASE("Adaptive selection without temporal metadata preserves ordinary timing", "[adaptive_ordering]") {
    Selection s;
    s.warmup_sym_count = 0;
    s.method_switch_threshold_soft = 0;
    record(s, 100);
    REQUIRE(s.factorization_times_for_method[0].estimatedAnalysisTime() == Approx(100));
    REQUIRE(s.updateSelection() == Choices::alternate_method);
}
