cmake_minimum_required(VERSION 3.27)

project(natten_mlx_nanobind LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(BUILD_SHARED_LIBS "Build extension as shared library" ON)

find_package(
  Python 3.8
  COMPONENTS Interpreter Development.Module
  REQUIRED)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m mlx --cmake-dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE MLX_ROOT)
find_package(MLX CONFIG REQUIRED PATHS "${MLX_ROOT}")

nanobind_add_module(
  _nanobind_ext
  NB_STATIC
  STABLE_ABI
  LTO
  NOMINSIZE
  NB_DOMAIN
  mlx
  ${CMAKE_CURRENT_LIST_DIR}/src/natten_mlx/_core/_nanobind_ext.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/natten_mlx/_core/nanobind/na_split_forward.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/natten_mlx/_core/nanobind/na_split_backward.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/natten_mlx/_core/nanobind/na_fused_forward.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/natten_mlx/_core/nanobind/na_fused_backward.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/natten_mlx/_core/nanobind/na_composed.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/natten_mlx/_core/nanobind/metal_runtime.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/natten_mlx/_core/nanobind/window_semantics.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/natten_mlx/_core/nanobind/na1d_primitive.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/natten_mlx/_core/nanobind/na2d_primitive.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/natten_mlx/_core/nanobind/na2d_split_primitive.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/natten_mlx/_core/nanobind/na3d_primitive.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/natten_mlx/_core/nanobind/na1d_bwd_primitive.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/natten_mlx/_core/nanobind/na2d_bwd_primitive.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/natten_mlx/_core/nanobind/na3d_bwd_primitive.cpp)

target_include_directories(
  _nanobind_ext PRIVATE ${CMAKE_CURRENT_LIST_DIR}/src/natten_mlx/_core)

target_link_libraries(_nanobind_ext PRIVATE mlx)

if(MLX_BUILD_METAL)
  mlx_build_metallib(
    TARGET natten_nb_metallib
    TITLE natten_nb
    SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/src/natten_mlx/_core/nanobind/natten_nb_split.metal
    ${CMAKE_CURRENT_LIST_DIR}/src/natten_mlx/_core/nanobind/natten_nb_fused.metal
    ${CMAKE_CURRENT_LIST_DIR}/src/natten_mlx/_core/nanobind/natten_nb_1d_v2.metal
    ${CMAKE_CURRENT_LIST_DIR}/src/natten_mlx/_core/nanobind/natten_nb_2d_v2.metal
    ${CMAKE_CURRENT_LIST_DIR}/src/natten_mlx/_core/nanobind/natten_nb_3d_v2.metal
    INCLUDE_DIRS
    ${CMAKE_CURRENT_LIST_DIR}/src/natten_mlx/_core/nanobind
    ${MLX_INCLUDE_DIRS}
    OUTPUT_DIRECTORY
    ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
  add_dependencies(_nanobind_ext natten_nb_metallib)
endif()

if(BUILD_SHARED_LIBS)
  target_link_options(_nanobind_ext PRIVATE -Wl,-rpath,@loader_path)
endif()
