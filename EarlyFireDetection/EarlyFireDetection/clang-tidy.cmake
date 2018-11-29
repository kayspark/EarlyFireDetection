# Additional targets to perform clang-format/clang-tidy
# Get all project files
file(GLOB_RECURSE
        ALL_CXX_SOURCE_FILES
        ${PROJECT_SOURCE_DIR}/src/*.*
        ${PROJECT_SOURCE_DIR}/test/*.*
        ${PROJECT_SOURCE_DIR}/*.cpp
        ${PROJECT_SOURCE_DIR}/*.cc
        ${PROJECT_SOURCE_DIR}/*.c
        ${PROJECT_SOURCE_DIR}/*.h
        ${PROJECT_SOURCE_DIR}/*.hpp
        )
if (win32)
# Adding clang-format target if executable is found
find_program(CLANG_FORMAT "clang-format")
    add_custom_target(
            clang-format
            COMMAND "C:/Program Files/LLVM/bin/clang-format.exe" 
            -i
            -style=file
            ${ALL_CXX_SOURCE_FILES}
    )

# Adding clang-tidy target if executable is found
find_program(CLANG_TIDY "clang-tidy")
    add_custom_target(
            clang-tidy
            COMMAND "C:/Program Files/LLVM/bin/clang-tidy.exe" 
            ${ALL_CXX_SOURCE_FILES}
            -config=''
            --
            -std=c++17
            ${INCLUDE_DIRECTORIES}
    )
elseif(APPLE)
# Adding clang-format target if executable is found
find_program(CLANG_FORMAT "clang-format")
    add_custom_target(
            clang-format
            COMMAND /usr/local/opt/llvm/bin/clang-format
            -i
            -style=file
            ${ALL_CXX_SOURCE_FILES}
    )

# Adding clang-tidy target if executable is found
find_program(CLANG_TIDY "clang-tidy")
    add_custom_target(
            clang-tidy
            COMMAND /usr/local/opt/llvm/bin/clang-tidy
            ${ALL_CXX_SOURCE_FILES}
            -config=''
            --
            -std=c++17
            ${INCLUDE_DIRECTORIES}
    )
endif()
