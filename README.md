# TemplateScanners
A C++ Library with Python Bindings to scan images with OpenCV

# Usage 

#### ThreadedVideoScanner

```py
import templatescanners 

scanner = templatescanners.ThreadedVideoScan() 

#################
# Threshold is an optional second parameter, if not passed, a adaptive threshold will be developed before scanning the entire video. 
# The threshold should be between 0 and 1, anything below 0 will, for the most part, result in tagging everything, anything above one will always return nothing
list_of_timestamps = scanner.run({"Name of Tempalte 1": ["path/to/image/describing/template/1", "path/to/image2/describing/template/1"]}, .8)

###### OR #######

# This is what the call should look like if a threshold is not provided.
list_of_timestamps = scanner.run_adaptive_threshold({"Name of Tempalte 1": ["path/to/image/describing/template/1", "path/to/image2/describing/template/1"]})
#################


# Timestamps are a simple class with two attributes, the original descriptor and the time it was found at
for timestamps in list_of_timestamps:
    print(f"FOUND {timestamps.marker}")
    print(f"AT {timestamps.time} SEC") 
```

#### TemplateScanner

```py
import templatescanners 

scanner = templatescanners.TemplateScanner({"Name of Template 1": ["path/to/image/describing/template/1", "path/to/image2/describing/template/1"]}, .8)
result = scanner.multithreaded_scan("path/to/image/to/scan")

# The scanner will either return an empty string or a string with the original descriptor for the found template. 
print(f"FOUND {result if result else 'NOTHING'}")
```

# Build From Source Instructions

First install the following pre-requisites <br>
[OpenCV-2 4.1](https://github.com/opencv/opencv/archive/4.0.1.zip) || [Install Instructions](https://docs.opencv.org/2.4/doc/tutorials/introduction/linux_install/linux_install.html)
<br>
[Boost/Boost::Python Linux](https://www.boost.org/doc/libs/1_61_0/more/getting_started/unix-variants.html) || [Install Instructions](https://www.boost.org/doc/libs/1_69_0/more/getting_started/windows.html)
<br>
[Boost/Boost::Python Windows](https://www.boost.org/users/history/version_1_67_0.html) || [Install Instructions](https://www.boost.org/doc/libs/1_69_0/more/getting_started/windows.html)
<br>

Note that you also will need python-dev libraries (The libraries for python-dev should be installed already with python on windows
but need to be acquired separately on linux with the command
```bash
$ sudo apt-get install python-dev
```
<br>

### Windows 

#### Install CMake

Go to [CMake's website](https://www.boost.org/doc/libs/1_69_0/more/getting_started/windows.html) and download the latest version for Windows.

open a command prompt after the install is complete and run the following to make sure it's installed correctly.  

```
cmake
```

#### Clone the repository and create build directory

```
git clone https://www.github.com/byrdofafeather/TemplateScanners
cd TemplateScanners/TemplateScanners
mkdir build 
cd build
```

#### Run Make 

Run the following: 
```
cmake -G "Visual Studio 15 2017 Win64" ..
```

It may be necessary to specify the location of library files if CMake can't find them in the path. 

Ex:
```
cmake -G "Visual Studio 15 2017 Win64" ..
-DOpenCV_DIR=C:\Users\username\Documents\opencv-4.0.1\build
```

Once the build is complete open the solution in Visual Studio (the version you built with). Right click on ALL_BUILD (recommended that release is selected at the top). Once the build is complete, in the file system under TemplateScanners/TemplateScanners/build/bin/Release, templatescanners.pyd should be found. 

To finally install the library, copy it into your python installations Lib folder (typically AppData/Local/Programs/Python36/libs). As well, locate the following files from your OpenCV installation and boost installation and move them there as well: 

(the ..... are used to indicate build information that is specific to the user) <br>
boost_python......dll <br>
opencv_core401.dll <br>
opencv_imgcodecs401.dll <br>
opencv_imgproc401.dll <br>
opencv_videoio401.dll <br>

#### Check Installation

Open a python terminal with whatever interpreter that you installed the .pyd into and run the following:
```py
import templatescanners 
test1 = templatescanners.TemplateScanner({"Name of Template 1": ["path/to/image/describing/template/1", "path/to/image2/describing/template/1"]}, .8)
test2 = templatescanners.ThreadedVideoScanner()
```
If no import errors occur everything should be working properly 

### Linux

#### Install CMake 

Run the following command: 
```bash
$ sudo apt-get install cmake
```

Then to ensure it installed properly run 
```
$ cmake --help
```

#### Clone the repository and create build directory

Run the following commands: 
```
$ git clone https://www.github.com/byrdofafeather/TemplateScanners
$ cd TemplateScanners/TemplateScanners
$ mkdir build 
$ cd build
```

#### Run Make 

Run the following commands: 
```
$ cmake (path)/TemplateScanners/TemplateScanners
$ make
```

As long as the compile is successful, the bin should contain a .so file. Copy the file into your desired interpreter directory.

#### Check Installation 

Open a python terminal with whatever interpreter that you installed the .pyd into and run the following:
```py
import templatescanners 
test1 = templatescanners.TemplateScanner({"Name of Template 1": ["path/to/image/describing/template/1", "path/to/image2/describing/template/1"]}, .8)
test2 = templatescanners.ThreadedVideoScanner()
```
If no import errors occur everything should be working properly 

# Projects

[Detecting Gameplay Moments From Video](https://www.github.com/byrdofafeather/PRIMEr)