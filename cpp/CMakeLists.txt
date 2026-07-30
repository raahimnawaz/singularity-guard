cmake_minimum_required(VERSION 3.16)
project(armctl LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Header-only library: mat3 / kinematics / controllers / trajectory / sim.
# No dependencies -- that is the Phase 5 constraint, not an accident.
add_library(armctl INTERFACE)
target_include_directories(armctl INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(armctl_main src/main.cpp)
target_link_libraries(armctl_main PRIVATE armctl)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(armctl_main PRIVATE -Wall -Wextra -Wpedantic)
endif()

enable_testing()

find_package(Python3 COMPONENTS Interpreter)
if(Python3_Interpreter_FOUND)
  add_test(
    NAME parity_vs_golden
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/parity_check.py
            $<TARGET_FILE:armctl_main>
            ${CMAKE_CURRENT_SOURCE_DIR}/../results/golden_through.csv
  )
else()
  message(WARNING "Python3 interpreter not found -- skipping the parity_vs_golden test")
endif()
