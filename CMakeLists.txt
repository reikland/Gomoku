cmake_minimum_required(VERSION 3.16)
project(Gomoku LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(SFML 2.5 COMPONENTS graphics window system REQUIRED)

add_executable(gomoku
    src/main.cpp
)

target_link_libraries(gomoku PRIVATE sfml-graphics sfml-window sfml-system)

