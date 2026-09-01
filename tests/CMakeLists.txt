add_executable(MeshFEMSparse_unit_tests
    main.cc
    test_sparse_matrices.cc
    test_block_csc_hessian.cc
)

target_link_libraries(MeshFEMSparse_unit_tests PUBLIC
    Catch2::Catch2
    MeshFEMSparse
    warnings::all
)

if (MSVC)
else ()
target_compile_options(MeshFEMSparse_unit_tests PRIVATE -Wno-unused-parameter)
endif()

add_executable(solve_system solve_system.cc)
target_link_libraries(solve_system MeshFEMSparse)

add_executable(benchmark_linear_systems benchmark_linear_systems.cc)
target_link_libraries(benchmark_linear_systems MeshFEMSparse)

add_executable(benchmark_sparsity_lru benchmark_sparsity_lru.cc)
target_link_libraries(benchmark_sparsity_lru MeshFEMSparse)

add_executable(benchmark_solves benchmark_solves.cc)
target_link_libraries(benchmark_solves MeshFEMSparse)

add_executable(benchmark_sorts benchmark_sorts.cc)
target_link_libraries(benchmark_sorts MeshFEMSparse)

if (MESHFEM_WITH_CATAMARI AND NOT MESHFEM_USE_LEGACY_CATAMARI)
    add_executable(benchmark_dense_linalg benchmark_dense_linalg.cc)
    target_link_libraries(benchmark_dense_linalg MeshFEMSparse)

    if(APPLE)
        enable_language(OBJCXX)
    else()
        set_source_files_properties(
            MPSIntegration/MPSCholesky.mm
            MPSIntegration/MPSOuterProduct.mm
            MPSIntegration/MPSGemm.mm
            PROPERTIES LANGUAGE CXX
        )
    endif()

    add_library(MPSIntegration STATIC
        MPSIntegration/MPSCholesky.mm
        MPSIntegration/MPSOuterProduct.mm
        MPSIntegration/MPSGemm.mm
    )
    target_include_directories(MPSIntegration PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_features(MPSIntegration PUBLIC cxx_std_11)
    if(APPLE)
        target_link_libraries(MPSIntegration PUBLIC
            "-framework Foundation"
            "-framework Metal"
            "-framework MetalPerformanceShaders"
        )
    endif()

    add_executable(benchmark_dense_cholesky benchmark_dense_cholesky.cc)
    target_link_libraries(benchmark_dense_cholesky MeshFEMSparse MPSIntegration)

    add_executable(benchmark_dense_outer_product benchmark_dense_outer_product.cc)
    target_link_libraries(benchmark_dense_outer_product MeshFEMSparse MPSIntegration)

    add_executable(benchmark_dense_gemm benchmark_dense_gemm.cc)
    target_link_libraries(benchmark_dense_gemm MeshFEMSparse MPSIntegration)

endif()

if (MESHFEM_WITH_SCOTCH)
    add_executable(benchmark_scotch benchmark_scotch.cc)
    target_link_libraries(benchmark_scotch MeshFEMSparse)
endif()

include(Catch)
catch_discover_tests(MeshFEMSparse_unit_tests PROPERTIES LABELS MeshFEMSparse)
