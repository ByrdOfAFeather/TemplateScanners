cmake_minimum_required(VERSION 3.12)
project(TemplateScanners)
IF(MSVC)
    ADD_DEFINITIONS("/EHsc")
ENDIF(MSVC)
set(CMAKE_CXX_STANDARD 14)
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
find_package(PythonLibs 3 REQUIRED)
find_package (Python3 COMPONENTS Interpreter Development)
include_directories(${Python3_INCLUDE_DIRS})
find_package(Boost 1.66.0 COMPONENTS python3.6 system filesystem regex REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})
link_directories(PATH TO PYTHON LIBS)


#add_executable(PRIMEr main.cpp TemplateScanners.cpp)
#target_link_libraries( PRIMEr ${OpenCV_LIBS} ${Boost_LIBRARIES} -lboost_python3 ${Python_LIBRARIES} -lpthread)
add_library(templatescanners SHARED templatescanners.cpp)
target_link_libraries( templatescanners ${OpenCV_LIBS} ${Boost_LIBRARIES} ${Python_LIBRARIES} )


set_target_properties(
        templatescanners
        PROPERTIES
        PREFIX ""
		SUFFIX ".pyd"
        OUTPUT_NAME templatescanners
        LINKER_LANGUAGE C
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin  # only needed on windows
)