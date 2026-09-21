if(NOT TARGET GDS::gds)
  message(FATAL_ERROR "Configure the repository root with -DGDS_BUILD_HACKATHON=ON")
endif()
pybind11_add_module(_native native.cpp)
target_link_libraries(_native PRIVATE GDS::gds)
if(NOT MSVC)
  target_compile_options(_native PRIVATE -Wall -Wextra -Wpedantic)
endif()
set_target_properties(_native PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
foreach(config DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
  set_target_properties(_native PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY_${config} "${CMAKE_CURRENT_BINARY_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_${config} "${CMAKE_CURRENT_BINARY_DIR}")
endforeach()
