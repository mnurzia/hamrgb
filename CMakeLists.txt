cmake_minimum_required(VERSION 3.26.0)
project(hamrgb)

add_executable(hamrgb)

target_compile_options(hamrgb PUBLIC -Wall -Werror --std=c99)
target_link_options(hamrgb PUBLIC)
target_sources(hamrgb PUBLIC hamrgb.c aparse/aparse.c)
