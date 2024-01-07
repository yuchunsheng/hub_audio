
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

string(APPEND CMAKE_CXX_FLAGS " -Wall -Wextra -Wl,-rpath,'$ORIGIN'")
string(APPEND CMAKE_C_FLAGS " -Wall -Wextra")

file(GLOB SRC
    *.h
    *.cpp
)

ADD_EXECUTABLE(audio_app ${SRC})
# target_link_libraries(audio_app
#     common
#     # whisper # remember to copy dll or so to bin folder
# )

include(CMakePrintHelpers)

cmake_print_variables(CMAKE_CURRENT_LIST_DIR)