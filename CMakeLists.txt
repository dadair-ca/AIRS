PROJECT(AIRS)

# CMake version requirements
CMAKE_MINIMUM_REQUIRED(VERSION 2.4)

# For CMake 2.6
IF(COMMAND CMAKE_POLICY)
  CMAKE_POLICY(SET CMP0003 NEW)
ENDIF(COMMAND CMAKE_POLICY)

#SET(CMAKE_SKIP_RPATH ON CACHE BOOL "rpath not included in .so files" FORCE)

# Look for a VTK build directory
FIND_PACKAGE(VTK)
IF (VTK_FOUND)
  INCLUDE (${VTK_USE_FILE})
ENDIF (VTK_FOUND)

# Look for an ITK build directory (optional)
OPTION(AIRS_USE_ITK "Use ITK in AIRS" OFF)
IF(AIRS_USE_ITK)
  FIND_PACKAGE(ITK QUIET)
  IF(ITK_FOUND)
    INCLUDE(${ITK_USE_FILE})
  ENDIF(ITK_FOUND)
ENDIF(AIRS_USE_ITK)

# Choose static or shared libraries.
OPTION(BUILD_SHARED_LIBS "Build shared libraries." OFF)
SET (AIRS_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS})
IF (BUILD_SHARED_LIBS)
  SET (BUILD_SHARED_LIBS_CONFIG 1)
ELSE (BUILD_SHARED_LIBS)
  SET (BUILD_SHARED_LIBS_CONFIG 0)
ENDIF (BUILD_SHARED_LIBS)

# Build documentation
OPTION(BUILD_DOCUMENTATION "Build Documentation" OFF)
IF(BUILD_DOCUMENTATION)
  FIND_PACKAGE(VTK)
  IF (USE_VTK_FILE)
    INCLUDE (${USE_VTK_FILE})
  ELSE (USE_VTK_FILE)
    MESSAGE(FATAL_ERROR "This application requires VTK.")
  ENDIF (USE_VTK_FILE)
  ADD_SUBDIRECTORY(Documentation/Doxygen)
ENDIF(BUILD_DOCUMENTATION)

# Output directories.
SET(LIBRARY_OUTPUT_PATH ${AIRS_BINARY_DIR}/bin CACHE INTERNAL "Single output directory for building all libraries.")
SET(EXECUTABLE_OUTPUT_PATH ${AIRS_BINARY_DIR}/bin CACHE INTERNAL "Single output directory for building all executables.")
SET(AIRS_LIBRARY_DIR ${LIBRARY_OUTPUT_PATH}/${CMAKE_CFG_INTDIR})
SET(AIRS_EXECUTABLE_DIR ${EXECUTABLE_OUTPUT_PATH}/${CMAKE_CFG_INTDIR})
SET(CXX_TEST_PATH ${EXECUTABLE_OUTPUT_PATH})

# Configure Dart testing support.
INCLUDE(${CMAKE_ROOT}/Modules/Dart.cmake)

# ---------------------------------------------------------------
# Initialize configuration

# Kits
SET(AIRS_KITS)

# Libraries
SET(AIRS_LIBRARIES)

# Include directories
SET(AIRS_INCLUDE_DIRS)

# ---------------------------------------------------------------
# Include the Tcl and Python wrappers
IF(VTK_WRAP_TCL AND ${VTK_MAJOR_VERSION} GREATER 4)
  IF(VTK_TCL_INCLUDE_DIR)
    INCLUDE_DIRECTORIES("${VTK_TCL_INCLUDE_DIR}")
  ENDIF(VTK_TCL_INCLUDE_DIR)
  INCLUDE(${VTK_CMAKE_DIR}/vtkWrapTcl.cmake)
ENDIF(VTK_WRAP_TCL AND ${VTK_MAJOR_VERSION} GREATER 4)

IF(VTK_WRAP_PYTHON AND ${VTK_MAJOR_VERSION} GREATER 4)
  IF(VTK_PYTHON_INCLUDE_DIR)
    INCLUDE_DIRECTORIES("${VTK_PYTHON_INCLUDE_DIR}")
  ENDIF(VTK_PYTHON_INCLUDE_DIR)
  INCLUDE(${VTK_CMAKE_DIR}/vtkWrapPython.cmake)
ENDIF(VTK_WRAP_PYTHON AND ${VTK_MAJOR_VERSION} GREATER 4)

IF(VTK_WRAP_JAVA AND ${VTK_MAJOR_VERSION} GREATER 4)
  SET(VTK_WRAP_JAVA 0)
ENDIF(VTK_WRAP_JAVA AND ${VTK_MAJOR_VERSION} GREATER 4)

# Include AIRSThirdParty
INCLUDE(${AIRS_SOURCE_DIR}/CMake/AIRSThirdParty.cmake)

# ---------------------------------------------------------------
# Start listing the AIRS kits

# ImageAnalysis
OPTION (AIRS_USE_IMAGEANALYSIS "Build the ImageAnalysis module" ON)
IF (AIRS_USE_IMAGEANALYSIS)
  SET(AIRS_INCLUDE_DIRS ${AIRS_INCLUDE_DIRS}
      "${AIRS_BINARY_DIR}/ImageAnalysis"
      "${AIRS_SOURCE_DIR}/ImageAnalysis")
  ADD_SUBDIRECTORY(ImageAnalysis)
  SET(AIRS_LIBRARIES ${AIRS_LIBRARIES} ImageAnalysis)
ENDIF (AIRS_USE_IMAGEANALYSIS)

# ImageRegistration
OPTION (AIRS_USE_IMAGEREGISTRATION "Build the ImageRegistration module" ON)
IF (AIRS_USE_IMAGEREGISTRATION)
  SET(AIRS_INCLUDE_DIRS ${AIRS_INCLUDE_DIRS}
      "${AIRS_BINARY_DIR}/ImageRegistration"
      "${AIRS_SOURCE_DIR}/ImageRegistration")
  ADD_SUBDIRECTORY(ImageRegistration)
  SET(AIRS_LIBRARIES ${AIRS_LIBRARIES} ImageRegistration)
ENDIF (AIRS_USE_IMAGEREGISTRATION)

# ImageSegmentation
OPTION (AIRS_USE_IMAGESEGMENTATION "Build the ImageSegmentation module" ON)
IF (AIRS_USE_IMAGESEGMENTATION)
  SET(AIRS_INCLUDE_DIRS ${AIRS_INCLUDE_DIRS}
      "${AIRS_BINARY_DIR}/ImageSegmentation"
      "${AIRS_SOURCE_DIR}/ImageSegmentation")
  ADD_SUBDIRECTORY(ImageSegmentation)
  SET(AIRS_LIBRARIES ${AIRS_LIBRARIES} ImageSegmentation)
ENDIF (AIRS_USE_IMAGESEGMENTATION)

# ITK classes
IF (AIRS_USE_ITK)

# ITKSegmentation
OPTION (AIRS_USE_ITKSEGMENTATION "Build the ITKSegmentation module" ON)
IF (AIRS_USE_ITKSEGMENTATION)
  SET(AIRS_INCLUDE_DIRS ${AIRS_INCLUDE_DIRS}
      "${AIRS_BINARY_DIR}/ITKSegmentation"
      "${AIRS_SOURCE_DIR}/ITKSegmentation")
  ADD_SUBDIRECTORY(ITKSegmentation)
  SET(AIRS_LIBRARIES ${AIRS_LIBRARIES} ITKSegmentation)
ENDIF (AIRS_USE_ITKSEGMENTATION)

ENDIF (AIRS_USE_ITK)

# Build Testing
OPTION(BUILD_TESTING "Build the examples" OFF)
IF(BUILD_TESTING)
   ADD_SUBDIRECTORY(Testing)
ENDIF(BUILD_TESTING)

# Build Examples
OPTION(BUILD_EXAMPLES "Build the examples" ON)
IF (BUILD_EXAMPLES)
   ADD_SUBDIRECTORY(Examples)
ENDIF (BUILD_EXAMPLES)

# Add airs.py.in file.  On Win32, this has to go into all config 
IF(VTK_WRAP_PYTHON)
  IF(WIN32 AND CMAKE_CONFIGURATION_TYPES)
    FOREACH(config ${CMAKE_CONFIGURATION_TYPES})
      CONFIGURE_FILE(
        ${AIRS_SOURCE_DIR}/airs.py.in
        ${LIBRARY_OUTPUT_PATH}/${config}/airs.py)
    ENDFOREACH(config)
  ELSE(WIN32 AND CMAKE_CONFIGURATION_TYPES)
    CONFIGURE_FILE(
      ${AIRS_SOURCE_DIR}/airs.py.in
      ${LIBRARY_OUTPUT_PATH}/airs.py)
  ENDIF(WIN32 AND CMAKE_CONFIGURATION_TYPES)
ENDIF(VTK_WRAP_PYTHON)

#-----------------------------------------------------------------------------
# The commands in this directory are intended to be executed as
# the end of the whole configuration process, as a "last step".
# This directory is typically the last SUBDIRS in the main CMakeLists.txt.
ADD_SUBDIRECTORY(CMake/LastConfigureStep)
