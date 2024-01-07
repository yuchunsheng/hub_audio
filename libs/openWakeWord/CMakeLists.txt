# target_link_libraries(audio_app
#                       pthread
#                       onnxruntime)

# # Example: openWakeWord-native/lib/x86_64
# target_link_directories(audio_app PUBLIC
#                         ${PROJECT_SOURCE_DIR}/libs/onnxruntime/lib/)

# target_include_directories(audio_app PUBLIC
#                            ${PROJECT_SOURCE_DIR}/libs/onnxruntime/include)

include(CMakePrintHelpers)
cmake_print_variables(CMAKE_CURRENT_LIST_DIR)