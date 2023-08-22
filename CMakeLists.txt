cmake_minimum_required(VERSION 3.26.0)
project(ham)

add_executable(ham)

target_compile_options(ham PUBLIC -Wall -Werror -O3)
target_link_options(ham PUBLIC)
target_sources(ham PUBLIC ham2.c)
