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
else()
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
if (ENABLE_CLANG_TIDY)
    find_program (CLANG_TIDY_EXE NAMES "clang-tidy" PATHS /usr/local/opt/llvm/bin )
    if (CLANG_TIDY_EXE)
      message(STATUS "clang-tidy found: ${CLANG_TIDY_EXE}")
      set(CLANG_TIDY_CHECKS "bugprone-*, modernize-*")
      set(CMAKE_CXX_CLANG_TIDY "${CLANG_TIDY_EXE};-checks=${CLANG_TIDY_CHECKS};-header-filter='${ALL_CXX_SOURCE_FILES}'"
        CACHE STRING "" FORCE)
    else()
      message(AUTHOR_WARNING "clang-tidy not found!")
      set(CMAKE_CXX_CLANG_TIDY "" CACHE STRING "" FORCE) # delete it
    endif()
endif()
endif()
