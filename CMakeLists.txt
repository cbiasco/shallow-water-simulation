# CMake expects a few commands, like the version required. We'll only require an older one
# since we aren't doing anything fancy.
cmake_minimum_required(VERSION 2.6)

# We also have to assign a project name and set the language used. In this case, we're
# calling it "example" and will be using C and C++:
project(Shallow_Water C CXX)

# ExternalProject will be used to make clones of dependencies from github
include(ExternalProject)

# The directory that contains our "FindXXX.cmake" scripts
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

# ====================================================
#		Options
# ====================================================


# If you do not have the core arb headers, you need an extensions library like glew.
# Run cmake with "-DUSE_GLEW" (or just set this to on).
if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin") # Off with apple
	option(USE_GLEW "Compile with glew instead of core" OFF)
else() # ON with everything else
	option(USE_GLEW "Compile with glew instead of core" ON)
endif()

# Instead of linking installed libraries for glfw3 you
# can check out fresh copies of them from github.
option(FORCE_CLONE "Use fresh updated clones of dependencies" ON)

# This is supposed to help with the lack of debug symbols for release versions
set(CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /openmp")
set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "${CMAKE_SHARED_LINKER_FLAGS_RELEASE} /DEBUG /OPT:REF /OPT:ICF")


# ====================================================
#		Setup
# ====================================================


# We can set C++ flags (like optimizations) here. We'll compile with c++11 since it
# has a lot of classes that will be useful.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3 -fopenmp")

# This creates a #define for you to use in your source code. It's handy for
# file input/output as it provides a full path.
add_definitions( -DMY_SRC_DIR="${CMAKE_CURRENT_SOURCE_DIR}/src/" )
add_definitions( -DMY_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data/" )

# ====================================================
#		Dependencies
# ====================================================


# OpenGL does the rendering
# We'll use CMake variables to hold all of the includes and libraries
# then do the including/linking at the end.
find_package(OpenGL 3.2 REQUIRED)
set(MY_LIBRARIES ${OPENGL_LIBRARIES})
set(MY_INCLUDE_DIRS ${OPENGL_INCLUDE_DIR})

# Glew has some opengl extensions we'll need if we don't have updated ogl headers
if(USE_GLEW)
	add_definitions( -DUSE_GLEW ) # define to use in our code
	if(FORCE_CLONE)
		set(GLEW_DIR ${CMAKE_CURRENT_SOURCE_DIR}/lib/glew )
		IF(NOT EXISTS ${GLEW_DIR})
			execute_process(COMMAND git clone https://github.com/over0219/glew-cmake.git -b glew-cmake-release ${GLEW_DIR})
		endif()
		# Instead of worrying about linking, we'll just add the glew source to our build.
		# It's less of a hassle that way.
		set(GLEW_SRC ${GLEW_DIR}/src/glew.c ${GLEW_DIR}/include/GL/glew.h)
		set(MY_INCLUDE_DIRS ${MY_INCLUDE_DIRS} ${GLEW_DIR}/include)
	else()
		find_package(GLEW REQUIRED)
		set(MY_LIBRARIES ${MY_LIBRARIES} ${GLEW_LIBRARIES})
		set(MY_INCLUDE_DIRS ${MY_INCLUDE_DIRS} ${GLEW_INCLUDE_DIR})
		set(GLEW_SRC "")
	endif()
else()
	set(GLEW_SRC "")
	add_definitions( -DGLFW_INCLUDE_GLCOREARB ) # let glfw include the extensions
endif()

# Now we'll add the rendering libraries we need
# GLFW3 does the windowing and input
# There is a file called cmake/FindGLFW.cmake that will search
# for an install of glfw, and return the required libraries/includes.
# If GLFW isn't found, we'll have to download the source and compile that.
if(NOT FORCE_CLONE)
	# Only search if we aren't forcing a new clone
	find_package(GLFW 3 REQUIRED)
endif()
if( GLFW_FOUND )
	# GLFW install was found, append those libraries
	set(MY_LIBRARIES ${MY_LIBRARIES} ${GLFW_LIBRARIES})
	set(MY_INCLUDE_DIRS ${MY_INCLUDE_DIRS} ${GLFW_INCLUDE_DIR})
else()

	set(glfw_checkout_Dir ${CMAKE_SOURCE_DIR}/lib/glfw)
	make_directory(${glfw_checkout_Dir})

	set(glfw_PREFIX "${glfw_checkout_Dir}")
	set(glfw_INSTALL_DIR "${glfw_checkout_Dir}/install")

	set(glfw_CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${glfw_INSTALL_DIR} -DUSE_MSVC_RUNTIME_LIBRARY_DLL=ON -DGLFW_BUILD_DOCS=OFF -DGLFW_BUILD_EXAMPLES=OFF -DGLFW_BUILD_TESTS=OFF )
	set(glfw_DIR "${glfw_INSTALL_DIR}")

	if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
		set(glfw_CMAKE_ARGS ${glfw_CMAKE_ARGS} -DCMAKE_XCODE_ATTRIBUTE_CLANG_CXX_LANGUAGE_STANDARD="c++11" -DCMAKE_XCODE_ATTRIBUTE_CLANG_CXX_LIBRARY="libc++")
	endif()

	ExternalProject_add(glfw
		PREFIX ${glfw_PREFIX}
		GIT_REPOSITORY https://github.com/glfw/glfw
		INSTALL_DIR ${glfw_INSTALL_DIR}
		CMAKE_ARGS ${glfw_CMAKE_ARGS}
		UPDATE_COMMAND ""
	)
	 
	set_property(TARGET glfw PROPERTY FOLDER "Dependencies")
	set(GLFW_INCLUDE_DIR ${glfw_INSTALL_DIR}/include CACHE INTERNAL "Directory of GLFW header files")

	# Windows Section #
	if (MSVC)
		set(GLFW_LIBRARY ${glfw_INSTALL_DIR}/lib/glfw3.lib CACHE INTERNAL "GLFW lib file")
		add_definitions(-D_CRT_SECURE_NO_WARNINGS)
		# Tell MSVC to use main instead of WinMain for Windows subsystem executables
		set_target_properties(${WINDOWS_BINARIES} PROPERTIES LINK_FLAGS "/ENTRY:mainCRTStartup")
	endif()

	# Apple section
	if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
		set(GLFW_LIBRARY ${glfw_INSTALL_DIR}/lib/libglfw3.a CACHE INTERNAL "GLFW lib file")
		set(CMAKE_XCODE_ATTRIBUTE_CLANG_CXX_LANGUAGE_STANDARD "c++11")
		set(CMAKE_XCODE_ATTRIBUTE_CLANG_CXX_LIBRARY "libc++")
		find_library(COCOA_LIB Cocoa)
		find_library(CARBON_LIB Carbon)
		find_library(IOKIT_LIB IOKit)
		find_library(CORE_FOUNDATION_FRAMEWORK CoreFoundation)
		find_library(CORE_VIDEO_FRAMEWORK CoreVideo)
		set(LIBS_ALL ${LIBS_ALL} ${COCOA_LIB} ${CARBON_LIB} ${IOKIT_LIB} ${CORE_FOUNDATION_FRAMEWORK} ${CORE_VIDEO_FRAMEWORK})
		message(STATUS "${CORE_VIDEO_FRAMEWORK}")
	endif()

	# Linux section
	if (${CMAKE_SYSTEM_NAME} MATCHES "Linux")
		set(GLFW_LIBRARY ${glfw_INSTALL_DIR}/lib/libglfw3.a CACHE INTERNAL "GLFW lib file")
		find_package(Threads)
		find_package(X11)
		set(LIBS_ALL ${LIBS_ALL} ${CMAKE_THREAD_LIBS_INIT} rt Xrandr Xxf86vm Xi Xcursor Xinerama m dl ${X11_LIBRARIES})
	endif()

	set(MY_INCLUDE_DIRS ${MY_INCLUDE_DIRS} ${GLFW_INCLUDE_DIR})
	set(MY_LIBRARIES ${MY_LIBRARIES} ${GLFW_LIBRARY} ${LIBS_ALL})

endif()


# ====================================================
#		Build
# ====================================================


# We need to include the headers for our rendering library
include_directories(${MY_INCLUDE_DIRS})

# Here, we make a variable that is actually a big list of all our source files.
# Note that the file also contains the directory w/ respect to this CMakeLists.txt.
# Every time we add a new source file, remember to add it to this list before
# you compile! Doing so manually is better than recursively (i.e. with file(GLOB_RECURSE ...))
# because that can lead to frustraing build errors if you're not careful.
set(MY_SOURCES
	${GLEW_SRC} # empty unless USE_GLEW is set to ON
	src/main.cpp
	src/shader.hpp
	src/trimesh.hpp
)

# Now we can add an executable, link it with our rendering libraries, and we're done!
add_executable(${PROJECT_NAME}  ${MY_SOURCES})
target_link_libraries(${PROJECT_NAME}  ${MY_LIBRARIES})
if(FORCE_CLONE)
	add_dependencies(${PROJECT_NAME} glfw)
endif()

