CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT (BayesPA)

include_directories("inc")
add_subdirectory("src")
add_subdirectory("src/utils")

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules) 

find_package(Boost COMPONENTS python REQUIRED)
message("Include dirs of boost: " ${Boost_INCLUDE_DIRS} )
message("Libs of boost: " ${Boost_LIBRARIES} )

find_package(PythonLibs REQUIRED)
message("Include dirs of Python: " ${PYTHON_INCLUDE_DIRS} )
message("Libs of Python: " ${PYTHON_LIBRARIES} )

include_directories(
  "inc"
  "inc/utils"
  ${Boost_INCLUDE_DIRS}
)

message(STATUS ${LIBS})

IF(CMAKE_COMPILER_IS_GNUCXX)
  message("Found g++ version " ${CMAKE_CXX_COMPILER_VERSION})
  ADD_DEFINITIONS("-std=c++11" "-Wall" "-g" "-O0" "-fPIC")
ELSE("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  message("Found clang++ version" ${CMAKE_CXX_COMPILER_VERSION})
  ADD_DEFINITIONS("-std=c++11" "-g" "-O3" "-fPIC" "-Wno-deprecated-register" "-Wno-overloaded-virtual" "-Wno-write-strings")
ELSE()
  MESSAGE(FATAL_ERROR "CMakeLists.txt has not been tested/written for your compiler.")
ENDIF()

ADD_LIBRARY(bayespa SHARED
    src/Corpus.cpp
    src/paMedLDAave.cpp
    src/utils/ap.cpp
    src/utils/cholesky.cpp
    src/utils/InverseGaussian.cpp
    src/utils/MVGaussian.cpp
    src/utils/objcokus.cpp
    src/utils/spdinverse.cpp
)

target_link_libraries(bayespa
  ${LIBS}
  ${Boost_LIBRARIES}
  ${PYTHON_LIBRARIES}
)

add_executable(test src/main.cpp)

target_link_libraries(test
  bayespa
  ${LIBS}
  ${Boost_LIBRARIES}
  ${PYTHON_LIBRARIES}
)
