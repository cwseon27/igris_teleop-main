cmake_minimum_required(VERSION 3.8)
project(igris_c_hand)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)

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

# The installed robot hand node uses a newer HandCmd/HandState wire schema
# than the public SDK used by the body bridge. Keep the matching snapshot
# isolated to this package so the working body LowCmd path is unaffected.
get_filename_component(IGRIS_REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../.." ABSOLUTE)
set(ROBOT_HAND_SDK_ROOT "${IGRIS_REPO_ROOT}/local_state/robot_hand_sdk")
set(ROBOT_HAND_SDK_HEADER "${ROBOT_HAND_SDK_ROOT}/include/igris_sdk/igris_c_msgs.hpp")
set(ROBOT_HAND_SDK_LIBRARY "${ROBOT_HAND_SDK_ROOT}/lib/libigris_sdk.a")
if(NOT EXISTS "${ROBOT_HAND_SDK_HEADER}" OR NOT EXISTS "${ROBOT_HAND_SDK_LIBRARY}")
  message(FATAL_ERROR
    "Robot-matched hand SDK snapshot is missing under ${ROBOT_HAND_SDK_ROOT}"
  )
endif()
set_property(TARGET igris_sdk::igris_sdk PROPERTY IMPORTED_LOCATION_RELEASE "${ROBOT_HAND_SDK_LIBRARY}")
set_property(TARGET igris_sdk::igris_sdk PROPERTY IMPORTED_LOCATION "${ROBOT_HAND_SDK_LIBRARY}")

add_executable(igris_c_hand_bridge_node src/igris_c_hand_bridge_node.cpp)
target_include_directories(igris_c_hand_bridge_node
  BEFORE PRIVATE
    "${ROBOT_HAND_SDK_ROOT}/include"
)
target_link_libraries(igris_c_hand_bridge_node igris_sdk::igris_sdk)
ament_target_dependencies(igris_c_hand_bridge_node rclcpp std_msgs std_srvs)

install(TARGETS igris_c_hand_bridge_node DESTINATION lib/${PROJECT_NAME})

ament_package()
