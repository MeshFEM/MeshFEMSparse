# Prepare dependencies owned by MeshFEMSparse.

if(NOT DEFINED MESHFEMSPARSE_ROOT)
    get_filename_component(MESHFEMSPARSE_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
endif()

if(NOT DEFINED MESHFEM_EXTERNAL)
    set(MESHFEM_EXTERNAL "${MESHFEMSPARSE_ROOT}/3rdparty")
endif()

get_directory_property(hasParent PARENT_DIRECTORY)
if (hasParent)
    set(MESHFEMSPARSE_ROOT "${MESHFEMSPARSE_ROOT}" PARENT_SCOPE)
    set(MESHFEM_EXTERNAL "${MESHFEM_EXTERNAL}" PARENT_SCOPE)
endif()

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR})
list(REMOVE_DUPLICATES CMAKE_MODULE_PATH)

include(MeshFEMCoreDependencies)
include(MeshFEMSparseDownloadExternal)

function(meshfem_sparse_alias_existing_target alias_name)
    set(options)
    set(one_value_args)
    set(multi_value_args CANDIDATES DEPENDENCIES)
    cmake_parse_arguments(MESHFEM_ALIAS "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

    foreach(candidate IN LISTS MESHFEM_ALIAS_CANDIDATES)
        if(TARGET ${candidate})
            if(NOT TARGET ${alias_name})
                add_library(${alias_name} INTERFACE IMPORTED)
                target_link_libraries(${alias_name} INTERFACE ${candidate})
                foreach(dependency IN LISTS MESHFEM_ALIAS_DEPENDENCIES)
                    if(TARGET ${dependency})
                        target_link_libraries(${alias_name} INTERFACE ${dependency})
                    endif()
                endforeach()
            endif()
            return()
        endif()
    endforeach()
endfunction()

if(NOT TARGET Catch2::Catch2 AND MESHFEMSPARSE_BUILD_TESTS)
    meshfem_download_catch()
    add_subdirectory(${MESHFEM_EXTERNAL}/Catch2 ${CMAKE_BINARY_DIR}/3rdparty/Catch2)
    list(APPEND CMAKE_MODULE_PATH ${MESHFEM_EXTERNAL}/Catch2/contrib)
endif()

if (MESHFEM_WITH_CHOLMOD)
    meshfem_sparse_alias_existing_target(cholmod::cholmod
        CANDIDATES SuiteSparse::CHOLMOD cholmod
        DEPENDENCIES
            SuiteSparse::AMD
            SuiteSparse::CAMD
            SuiteSparse::CCOLAMD
            SuiteSparse::COLAMD
            SuiteSparse::Config
    )
    if(NOT TARGET cholmod::cholmod)
        find_package(CHOLMOD REQUIRED) # provides cholmod::cholmod
    endif()
endif()

if (MESHFEM_WITH_UMFPACK)
    meshfem_sparse_alias_existing_target(umfpack::umfpack
        CANDIDATES SuiteSparse::UMFPACK umfpack
        DEPENDENCIES
            SuiteSparse::AMD
            SuiteSparse::Config
    )
    if(NOT TARGET umfpack::umfpack)
        find_package(UMFPACK REQUIRED) # provides umfpack::umfpack
    endif()
endif()

if (MESHFEM_WITH_CATAMARI AND NOT TARGET catamari)
    if (MESHFEM_USE_LEGACY_CATAMARI)
        meshfem_download_catamari_legacy()
        add_subdirectory(${MESHFEM_EXTERNAL}/catamari_legacy ${CMAKE_BINARY_DIR}/3rdparty/catamari_legacy)
    else()
        meshfem_download_catamari()
    endif()
endif()

if (MESHFEM_WITH_SCOTCH)
    find_package(SCOTCH QUIET)
    if (NOT TARGET SCOTCH::scotch)
        message(STATUS "Scotch not found; support will be disabled")
        set(MESHFEM_WITH_SCOTCH OFF)
    endif()
endif()
