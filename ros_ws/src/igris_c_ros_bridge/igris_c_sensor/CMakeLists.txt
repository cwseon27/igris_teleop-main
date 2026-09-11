cmake_minimum_required(VERSION 3.8)
project(igris_c_sensor)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)

find_program(CLANG_FORMAT_BIN NAMES clang-format)
file(GLOB_RECURSE CLANG_FORMAT_FILES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/src/*.c"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cxx"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/*.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/*.hh"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/*.hxx"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/*.hh"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/*.hxx"
)
list(FILTER CLANG_FORMAT_FILES EXCLUDE REGEX ".*/robot_igris_c_msgs\\.hpp$")

if(CLANG_FORMAT_BIN AND CLANG_FORMAT_FILES)
  add_custom_target(${PROJECT_NAME}_clang_format_check ALL
    COMMAND ${CLANG_FORMAT_BIN} --dry-run --Werror --style=file ${CLANG_FORMAT_FILES}
    COMMENT "Checking ${PROJECT_NAME} source formatting"
    VERBATIM
  )
elseif(NOT CLANG_FORMAT_BIN)
  message(STATUS "clang-format not found; skipping ${PROJECT_NAME} source formatting check")
endif()

if(NOT igris_sdk_DIR)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty/igris_c_sdk_public/lib/cmake/igris_sdk/igris_sdk-config.cmake")
    set(igris_sdk_DIR
        "${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty/igris_c_sdk_public/lib/cmake/igris_sdk"
    )
  endif()
endif()

find_package(igris_sdk REQUIRED)

add_executable(igris_c_sensor_bridge_node src/igris_c_sensor_bridge_node.cpp)
target_include_directories(igris_c_sensor_bridge_node PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(igris_c_sensor_bridge_node igris_sdk::igris_sdk ${OpenCV_LIBS})
ament_target_dependencies(igris_c_sensor_bridge_node ament_index_cpp rclcpp sensor_msgs)

add_executable(igris_c_sensor_robot_bridge_node src/igris_c_sensor_robot_bridge_node.cpp)
target_include_directories(igris_c_sensor_robot_bridge_node
  PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
target_link_libraries(igris_c_sensor_robot_bridge_node igris_sdk::igris_sdk ${OpenCV_LIBS})
ament_target_dependencies(igris_c_sensor_robot_bridge_node ament_index_cpp rclcpp sensor_msgs)

install(TARGETS
  igris_c_sensor_bridge_node
  igris_c_sensor_robot_bridge_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_environment_hooks(
  "${CMAKE_CURRENT_SOURCE_DIR}/environment/ros_domain_id.dsv.in"
)

ament_package()
