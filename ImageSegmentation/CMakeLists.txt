#PROJECT(ImageSegmentation)

SET(KIT ImageSegmentation)
SET(UKIT IMAGESEGMENTATION)
SET(KIT_LIBS vtkHybrid)

# Set your list of sources here.  Do not change the name of the
# Kit_SRCS variable.
SET(Kit_SRCS
  vtkImageFloodFill.cxx
  vtkImageMRIBrainExtractor.cxx
)

SET(Kit_EXTRA_SRCS)
SET(Kit_EXTRA_CMDS)
SET(Kit_TCL_EXTRA_SRCS)
SET(Kit_PYTHON_EXTRA_SRCS)
SET(Kit_JAVA_EXTRA_SRCS)

# -----------------------------------------------------------------
# Include the common setting file
INCLUDE(${AIRS_SOURCE_DIR}/CMake/AIRSCommonBlock.cmake)
