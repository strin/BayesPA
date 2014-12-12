CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT (BayesPA)

include_directories("inc")
include_directories("inc/utils")
add_subdirectory("src")

set(CMAKE_PREFIX_PATH /opt/local ${CMAKE_PREFIX_PATH}) 
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules) 

find_package(Boost COMPONENTS filesystem system date_time python thread REQUIRED)
message("Include dirs of boost: " ${Boost_INCLUDE_DIRS} )
message("Libs of boost: " ${Boost_LIBRARIES} )

find_package(PythonLibs REQUIRED)
message("Include dirs of Python: " ${PYTHON_INCLUDE_DIRS} )
message("Libs of Python: " ${PYTHON_LIBRARIES} )

include_directories(
  "inc"
  ${Boost_INCLUDE_DIRS}
  ${PYTHON_INCLUDE_DIRS}
)

message(STATUS ${LIBS})

IF(CMAKE_COMPILER_IS_GNUCXX)
  message("Found g++ version " ${CMAKE_CXX_COMPILER_VERSION})
  ADD_DEFINITIONS("-std=c++11" "-g" "-O3" "-fPIC" "-Wno-deprecated-register" "-Wno-overloaded-virtual" "-Wno-write-strings")
ELSE("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  message("Found clang++ version" ${CMAKE_CXX_COMPILER_VERSION})
  ADD_DEFINITIONS("-std=c++11" "-g" "-O3" "-fPIC" "-Wno-deprecated-register" "-Wno-overloaded-virtual" "-Wno-write-strings")
ELSE()
  MESSAGE(FATAL_ERROR "CMakeLists.txt has not been tested/written for your compiler.")
ENDIF()

ADD_LIBRARY(bayespagibbs SHARED
  src/utils/ap.cpp
  src/utils/Sample.cpp
  src/utils/cholesky.cpp
  src/utils/cokus.cpp
  src/utils/InverseGaussian.cpp
  src/utils/Mapper.cpp
  src/utils/MVGaussian.cpp
  src/utils/objcokus.cpp
  src/utils/spdinverse.cpp
  src/utils/Document.cpp
  src/OnlineGibbsMedLDA.cpp
  src/OnlineGibbsMedLDAWrapper.cpp
)

target_link_libraries(bayespagibbs
  ${LIBS}
  ${Boost_LIBRARIES}
  ${PYTHON_LIBRARIES}
)

ADD_EXECUTABLE(bayespaTest 
  test/test_pyutils.cpp
)

target_link_libraries(bayespaTest
  ${LIBS}
  ${Boost_LIBRARIES}
  ${PYTHON_LIBRARIES}
)

add_custom_target(dep ALL COMMAND mv libbayespagibbs.dylib libbayespagibbs.so )
