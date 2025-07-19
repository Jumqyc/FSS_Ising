cmake_minimum_required(VERSION 3.14)
project(Ising LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose build type: Debug, Release, RelWithDebInfo" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(Ising_ENABLE_NATIVE_OPT "Enable native CPU optimization (-march=native)" ON)
option(Ising_ENABLE_LTO "Enable Link Time Optimization" ON)
option(Ising_ENABLE_AVX2 "Enable AVX2 vectorization" ON)
option(Ising_ENABLE_FAST_MATH "Enable fast math optimizations" ON)
option(Ising_ENABLE_OPENMP "Enable OpenMP parallelization" ON)

include(CheckCXXCompilerFlag)

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")


add_compile_options(
    "$<$<CONFIG:Release>:-fomit-frame-pointer>"
    "$<$<CONFIG:Release>:-funroll-loops>"
    "$<$<CONFIG:Release>:-floop-optimize>"
    "$<$<CONFIG:Release>:-fstrict-aliasing>"
    "$<$<CONFIG:Release>:-fno-semantic-interposition>"
)

# 启用向量化
if(Ising_ENABLE_AVX2)
    add_compile_options(
        "$<$<CONFIG:Release>:-mavx2>"
        "$<$<CONFIG:Release>:-mfma>"
        "$<$<CONFIG:Release>:-ftree-vectorize>"
    )
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DENABLE_VECTORIZATION=ON")
endif()

# 本地CPU优化
if(Ising_ENABLE_NATIVE_OPT)
    add_compile_options("$<$<CONFIG:Release>:-march=native>")
    message(STATUS "Enabled native CPU optimization (-march=native)")
else()
    # 保守的x86-64优化
    add_compile_options("$<$<CONFIG:Release>:-march=x86-64-v2>")
endif()

# 快速数学优化
if(Ising_ENABLE_FAST_MATH)
    add_compile_options(
        "$<$<CONFIG:Release>:-ffast-math>"
        "$<$<CONFIG:Release>:-fno-math-errno>"
    )
    message(STATUS "Enabled fast math optimizations")
endif()

# 链接时优化
if(Ising_ENABLE_LTO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT IPO_SUPPORTED OUTPUT IPO_OUTPUT)
    if(IPO_SUPPORTED)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
        message(STATUS "Enabled Link Time Optimization (LTO)")
    else()
        message(WARNING "LTO not supported: ${IPO_OUTPUT}")
    endif()
endif()

# OpenMP支持
if(Ising_ENABLE_OPENMP)
    find_package(OpenMP REQUIRED)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# 编译器特定优化
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    add_compile_options(
        "$<$<CONFIG:Release>:-fno-stack-protector>"
        "$<$<CONFIG:Release>:-fno-plt>"
    )
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} -Wl,-O3 -Wl,--as-needed -Wl,--strip-all")
    
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(
        "$<$<CONFIG:Release>:-fvectorize>"
        "$<$<CONFIG:Release>:-fslp-vectorize>"
    )
    if(Ising_ENABLE_AVX2)
        add_compile_options("$<$<CONFIG:Release>:-mllvm -vectorize-loops>")
    endif()
    
elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    add_compile_options(
        "$<$<CONFIG:Release>:/O2 /Ob2 /GL /fp:fast /arch:AVX2>"
    )
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} /LTCG")
endif()

# 添加pybind11子目录
add_subdirectory(extern/pybind11)

# 创建Ising模块
pybind11_add_module(Ising
    bind.cpp
    Ising.hpp
)

# 链接OpenMP（如果启用）
if(Ising_ENABLE_OPENMP)
    target_link_libraries(Ising PRIVATE OpenMP::OpenMP_CXX)
endif()

# 安装目标
install(TARGETS Ising DESTINATION .)

# 打印配置摘要
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Native optimization: ${Ising_ENABLE_NATIVE_OPT}")
message(STATUS "LTO: ${Ising_ENABLE_LTO}")
message(STATUS "AVX2: ${Ising_ENABLE_AVX2}")
message(STATUS "Fast math: ${Ising_ENABLE_FAST_MATH}")
message(STATUS "OpenMP: ${Ising_ENABLE_OPENMP}")