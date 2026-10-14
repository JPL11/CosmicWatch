# Minimal locator for the Raspberry Pi Pico SDK: set PICO_SDK_PATH in the
# environment (or -DPICO_SDK_PATH=...) to a pico-sdk checkout >= 2.0.
if (DEFINED ENV{PICO_SDK_PATH} AND (NOT PICO_SDK_PATH))
    set(PICO_SDK_PATH $ENV{PICO_SDK_PATH})
endif ()
if (NOT PICO_SDK_PATH)
    message(FATAL_ERROR "PICO_SDK_PATH not set (git clone https://github.com/raspberrypi/pico-sdk)")
endif ()
include(${PICO_SDK_PATH}/pico_sdk_init.cmake)
