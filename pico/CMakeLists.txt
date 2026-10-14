cmake_minimum_required(VERSION 3.13)

# Pico 2 W by default; -DPICO_BOARD=pico for an RP2040 board, or
# -DPICO_PLATFORM=rp2350-riscv to run the same code on the RISC-V cores.
set(PICO_BOARD pico2_w CACHE STRING "Board type")
include(pico_sdk_import.cmake)

project(pico_bench C CXX ASM)
set(CMAKE_C_STANDARD 11)
pico_sdk_init()

add_executable(pico_bench pico_bench.c ../mcu_classifier.c)
target_include_directories(pico_bench PRIVATE ..)
target_compile_definitions(pico_bench PRIVATE MLP49_NO_MAIN)
target_compile_options(pico_bench PRIVATE -O2)
target_link_libraries(pico_bench pico_stdlib)
pico_enable_stdio_usb(pico_bench 1)
pico_enable_stdio_uart(pico_bench 0)
pico_add_extra_outputs(pico_bench)
