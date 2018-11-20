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

# Adding clang-format target if executable is found
find_program(CLANG_FORMAT "clang-format")
if (CLANG_FORMAT)
    add_custom_target(
            clang-format
            COMMAND /usr/local/opt/llvm/bin/clang-format
            -i
            -style=file
            ${ALL_CXX_SOURCE_FILES}
    )
endif ()

# Adding clang-tidy target if executable is found
find_program(CLANG_TIDY "clang-tidy")
if (CLANG_TIDY)
    add_custom_target(
            clang-tidy
            COMMAND /usr/local/opt/llvm/bin/clang-tidy
            ${ALL_CXX_SOURCE_FILES}
            -config=''
            --
            -std=c++17
            ${INCLUDE_DIRECTORIES}
    )
endif ()
