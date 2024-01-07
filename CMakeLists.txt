cmake_minimum_required(VERSION 3.0.0)
project(hub_audio VERSION 0.1.0 LANGUAGES C CXX)

# include(CTest)
# enable_testing()

add_subdirectory(libs/openWakeWord)
# add_subdirectory(libs/whisper.cpp)

add_subdirectory(src)

# set(CPACK_PROJECT_NAME ${PROJECT_NAME})
# set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
# include(CPack)
include(CMakePrintHelpers)
cmake_print_variables(PROJECT_SOURCE_DIR)
cmake_print_variables(CMAKE_CURRENT_LIST_DIR)
