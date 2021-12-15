cmake_minimum_required(VERSION 3.0)
project( opencv_labs )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable( lab1 lab1.cpp )
add_executable( dop_1 lab1_dop.cpp )
add_executable( dop_2 lab1_dop2.cpp )
add_executable( lab2 lab1_dop2.cpp )

target_link_libraries( lab1 ${OpenCV_LIBS} )
target_link_libraries( dop_1 ${OpenCV_LIBS} )
target_link_libraries( dop_2 ${OpenCV_LIBS} )
target_link_libraries( lab2 ${OpenCV_LIBS} )
