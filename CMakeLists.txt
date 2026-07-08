cmake_minimum_required(VERSION 3.14...3.31)
project(MeshFEMSparse LANGUAGES CXX)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(MESHFEMSPARSE_TOPLEVEL_PROJECT ON)
else()
    set(MESHFEMSPARSE_TOPLEVEL_PROJECT OFF)
endif()

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
list(REMOVE_DUPLICATES CMAKE_MODULE_PATH)

option(MESHFEM_WITH_CATAMARI         "Compile MeshFEM with Catamari"   ON)
option(MESHFEM_USE_LEGACY_CATAMARI   "Use Jack Poulson's original Catamari code for comparisons instead of our accelerated BlockCatamari"  OFF)
option(MESHFEM_WITH_SCOTCH           "Compile MeshFEM with support for Scotch reordering, if possible" OFF)
option(MESHFEM_WITH_MKL_PARDISO      "Compile MeshFEM with MKL Pardiso"   OFF)
option(MESHFEM_WITH_PARDISO          "Compile MeshFEM with Pardiso"   OFF)
option(MESHFEM_WITH_CHOLMOD          "Compile MeshFEM with CHOLMOD for LL^T factorization" ON)
option(MESHFEM_WITH_UMFPACK          "Compile MeshFEM with UMFPACK for LU factorization (GPL licensed)" OFF)

option(MESHFEMSPARSE_BUILD_TESTS "Build MeshFEMSparse tests" ${MESHFEMSPARSE_TOPLEVEL_PROJECT})

if(NOT TARGET MeshFEMCore)
    include(FetchContent)
    # TODO: change these to https once public.
    set(MESHFEMCORE_GIT_REPOSITORY "git@github.com:MeshFEM/MeshFEMCore.git"
        CACHE STRING "MeshFEMCore git repository")
    set(MESHFEMCORE_GIT_TAG "2867a326f79269660da80846d273e4440817cbe8"
        CACHE STRING "MeshFEMCore git revision")
    set(MESHFEM_EXTERNAL "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty"
        CACHE PATH "External dependency source directory")

    FetchContent_Declare(MeshFEMCore
        GIT_REPOSITORY ${MESHFEMCORE_GIT_REPOSITORY}
        GIT_TAG        ${MESHFEMCORE_GIT_TAG}
        SOURCE_DIR     ${MESHFEM_EXTERNAL}/MeshFEMCore
    )
    FetchContent_MakeAvailable(MeshFEMCore)
    list(APPEND CMAKE_MODULE_PATH "${meshfemcore_SOURCE_DIR}/cmake")
endif()

include(MeshFEMSparseDependencies)

add_library(MeshFEMSparse STATIC
    src/lib/MeshFEMSparse/BlockCSCHessian.cc
    src/lib/MeshFEMSparse/BorderedSparseHessian.cc
    src/lib/MeshFEMSparse/Solvers/make_cholesky_factorizer.cc
    src/lib/MeshFEMSparse/Solvers/CatamariFactorizer.cc
    src/lib/MeshFEMSparse/Solvers/MatrixRecorder.cc
    src/lib/MeshFEMSparse/Solvers/PardisoFactorizer.cc
    src/lib/MeshFEMSparse/Solvers/pardiso_ordering.cc
    src/lib/MeshFEMSparse/Solvers/AccelerateFactorizer.cc
    src/lib/MeshFEMSparse/Solvers/accelerate_ordering.cc
)

if (MESHFEM_USE_LEGACY_CATAMARI)
    get_target_property(modified_sources MeshFEMSparse SOURCES)
    list(REMOVE_ITEM modified_sources src/lib/MeshFEMSparse/Solvers/CatamariFactorizer.cc)
    list(APPEND modified_sources src/lib/MeshFEMSparse/Solvers/CatamariFactorizerLegacy.cc)
    set_target_properties(MeshFEMSparse PROPERTIES SOURCES "${modified_sources}")
endif()

target_include_directories(MeshFEMSparse PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/lib>
    $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/exports>
)
target_include_directories(MeshFEMSparse PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/lib/MeshFEMSparse
    ${CMAKE_CURRENT_SOURCE_DIR}/src/lib/MeshFEMSparse/Solvers
)
target_compile_features(MeshFEMSparse PUBLIC cxx_std_17)

target_link_libraries(MeshFEMSparse
    PUBLIC
        MeshFEMCore
        Eigen3::Eigen
        TBB::tbb
        Threads::Threads
    PRIVATE
        warnings::all
)

target_compile_options(MeshFEMSparse PUBLIC -fvisibility=hidden)

if(MESHFEM_WITH_UMFPACK)
    target_compile_definitions(MeshFEMSparse PUBLIC -DMESHFEM_WITH_UMFPACK)
    target_link_libraries(MeshFEMSparse PUBLIC umfpack::umfpack)
endif()

if(MESHFEM_WITH_CHOLMOD)
    target_compile_definitions(MeshFEMSparse PUBLIC -DMESHFEM_WITH_CHOLMOD)
    target_link_libraries(MeshFEMSparse PUBLIC cholmod::cholmod)
endif()

if (MESHFEM_WITH_CATAMARI)
    target_compile_definitions(MeshFEMSparse PUBLIC -DMESHFEM_WITH_CATAMARI)
    target_link_libraries(MeshFEMSparse PUBLIC catamari)
    if (MESHFEM_USE_LEGACY_CATAMARI)
        target_compile_definitions(MeshFEMSparse PUBLIC -DMESHFEM_USE_LEGACY_CATAMARI)
    endif()
endif()

if (MESHFEM_WITH_SCOTCH)
    target_compile_definitions(MeshFEMSparse PUBLIC -DMESHFEM_WITH_SCOTCH)
    target_link_libraries(MeshFEMSparse PUBLIC SCOTCH::scotch SCOTCH::scotcherrexit)
endif()

if (MESHFEM_WITH_MKL_PARDISO)
    target_compile_definitions(MeshFEMSparse PUBLIC -DMESHFEM_WITH_MKL_PARDISO)
endif()

if (MESHFEM_WITH_PARDISO)
    set(MKLROOT /opt/intel/mkl)
    target_compile_definitions(MeshFEMSparse PUBLIC -DMESHFEM_WITH_PARDISO)
    target_link_libraries(MeshFEMSparse PUBLIC
        /home/jpanetta/software/pardiso/libpardiso600-GNU800-X86-64.so
         -L${MKLROOT}/lib/intel64 -lmkl_rt -Wl,--no-as-needed -lpthread -lm -ldl -lgomp
        -lgfortran)
endif()

if (MESHFEMSPARSE_BUILD_TESTS)
    include(CTest)
    if (BUILD_TESTING)
        add_subdirectory(tests)
    endif()
endif()
