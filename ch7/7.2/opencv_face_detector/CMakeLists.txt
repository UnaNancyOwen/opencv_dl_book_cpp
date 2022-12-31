cmake_minimum_required( VERSION 3.6 )

# Require C++17 (or later)
set( CMAKE_CXX_STANDARD 17 )
set( CMAKE_CXX_STANDARD_REQUIRED ON )
set( CMAKE_CXX_EXTENSIONS OFF )

# Create Project
project( opencv_face_detector )
add_executable( opencv_face_detector opencv_face_detector.cpp )

# Set StartUp Project
set_property( DIRECTORY PROPERTY VS_STARTUP_PROJECT "opencv_face_detector" )

# Find Package
# OpenCV
find_package( OpenCV REQUIRED )

if( OpenCV_FOUND )
  # Additional Include Directories
  include_directories( ${OpenCV_INCLUDE_DIRS} )

  # Additional Dependencies
  target_link_libraries( opencv_face_detector ${OpenCV_LIBS} )
endif()
