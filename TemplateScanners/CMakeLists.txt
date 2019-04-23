cmake_minimum_required(VERSION 3.12)
project(PRIMEr)

set(CMAKE_CXX_STANDARD 14)
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
find_package(Boost 1.67.0 COMPONENTS system filesystem REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})
find_package(PythonLibs 3 REQUIRED)
find_package (Python3 COMPONENTS Interpreter Development)
include_directories(${Python3_INCLUDE_DIRS})

#add_executable(PRIMEr main.cpp TemplateScanners.cpp)
#target_link_libraries( PRIMEr ${OpenCV_LIBS} ${Boost_LIBRARIES} -lboost_python3 ${Python_LIBRARIES} -lpthread)
add_library(templatescannersbeta SHARED TemplateScanners.cpp)
target_link_libraries( templatescannersbeta ${OpenCV_LIBS} ${Boost_LIBRARIES} -lboost_python3 ${Python_LIBRARIES} -lpthread) # link OpenCV libraries , hightgui.so not found by cmake so this hack


set_target_properties(
        templatescannersbeta
        PROPERTIES
        PREFIX ""
        OUTPUT_NAME templatescannersbeta
        LINKER_LANGUAGE C
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin  # only needed on windows
)

# -----------------------------------------------------------------------------
# installation

install(
        TARGETS templatescannersbeta
        LIBRARY
        DESTINATION
        "/usr/local/lib/python3.6/dist-packages"
)
