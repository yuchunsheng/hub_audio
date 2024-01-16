
file(GLOB SRC
    *.h
    *.cpp
)

add_library(openwakeword ${SRC})

target_link_libraries(openwakeword
                      pthread
                      onnxruntime)

# # Example: openWakeWord-native/lib/x86_64
target_link_directories(openwakeword PUBLIC
                        ${PROJECT_SOURCE_DIR}/libs/onnxruntime/lib/)

target_include_directories(openwakeword PUBLIC
                           ${PROJECT_SOURCE_DIR}/libs/onnxruntime/include)

include(CMakePrintHelpers)
cmake_print_variables(CMAKE_CURRENT_LIST_DIR)