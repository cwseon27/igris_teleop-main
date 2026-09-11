cmake_minimum_required(VERSION 3.8)
project(igris_lowcmd_relay)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(ament_cmake REQUIRED)
find_package(igris_c_sdk REQUIRED)
find_package(rclcpp REQUIRED)
find_package(Threads REQUIRED)

add_executable(lowcmd_relay_node src/lowcmd_relay_node.cpp)
ament_target_dependencies(lowcmd_relay_node igris_c_sdk rclcpp)
target_link_libraries(lowcmd_relay_node Threads::Threads)

install(TARGETS lowcmd_relay_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
