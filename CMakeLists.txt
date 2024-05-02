cmake_minimum_required(VERSION 3.0...3.27)
project("vanilla_neural_network")

set(CMAKE_VERBOSE_MAKEFILE on)

if (CMAKE_GENERATOR MATCHES "Visual Studio")
    add_compile_options("$<$<C_COMPILER_ID:MSVC>:/utf-8>")
    add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")
endif()

list(APPEND HEADERS stack.h)

list(APPEND SOURCES main.cc)

add_executable(nn ${SOURCES} ${HEADERS})

set_target_properties(nn PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
