cmake_minimum_required( VERSION 3.6 )

# Require C++17 (or later)
set( CMAKE_CXX_STANDARD 17 )
set( CMAKE_CXX_STANDARD_REQUIRED ON )
set( CMAKE_CXX_EXTENSIONS OFF )

# Create Project
project( deeplab-v3 )
add_executable( deeplab-v3 deeplab-v3.cpp )

# Set StartUp Project
set_property( DIRECTORY PROPERTY VS_STARTUP_PROJECT "deeplab-v3" )

# Find Package
# OpenCV
find_package( OpenCV REQUIRED )
# OpenMP
find_package(OpenMP)

if( OpenCV_FOUND )
  # Additional Include Directories
  include_directories( ${OpenCV_INCLUDE_DIRS} )

  # Additional Dependencies
  target_link_libraries( deeplab-v3 ${OpenCV_LIBS} )
endif()

if( OpenMP_CXX_FOUND )
  # Additional Dependencies
  target_link_libraries( deeplab-v3 OpenMP::OpenMP_CXX)
endif()
