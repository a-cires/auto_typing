# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/umdloop/auto_typing_ws/cpp_ext

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/umdloop/auto_typing_ws/_skbuild/linux-x86_64-3.10/cmake-build

# Include any dependencies generated for this target.
include CMakeFiles/motor_cpp.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/motor_cpp.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/motor_cpp.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/motor_cpp.dir/flags.make

CMakeFiles/motor_cpp.dir/src/bindings.cpp.o: CMakeFiles/motor_cpp.dir/flags.make
CMakeFiles/motor_cpp.dir/src/bindings.cpp.o: /home/umdloop/auto_typing_ws/cpp_ext/src/bindings.cpp
CMakeFiles/motor_cpp.dir/src/bindings.cpp.o: CMakeFiles/motor_cpp.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/umdloop/auto_typing_ws/_skbuild/linux-x86_64-3.10/cmake-build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/motor_cpp.dir/src/bindings.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/motor_cpp.dir/src/bindings.cpp.o -MF CMakeFiles/motor_cpp.dir/src/bindings.cpp.o.d -o CMakeFiles/motor_cpp.dir/src/bindings.cpp.o -c /home/umdloop/auto_typing_ws/cpp_ext/src/bindings.cpp

CMakeFiles/motor_cpp.dir/src/bindings.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/motor_cpp.dir/src/bindings.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/umdloop/auto_typing_ws/cpp_ext/src/bindings.cpp > CMakeFiles/motor_cpp.dir/src/bindings.cpp.i

CMakeFiles/motor_cpp.dir/src/bindings.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/motor_cpp.dir/src/bindings.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/umdloop/auto_typing_ws/cpp_ext/src/bindings.cpp -o CMakeFiles/motor_cpp.dir/src/bindings.cpp.s

CMakeFiles/motor_cpp.dir/src/motor_controller.cpp.o: CMakeFiles/motor_cpp.dir/flags.make
CMakeFiles/motor_cpp.dir/src/motor_controller.cpp.o: /home/umdloop/auto_typing_ws/cpp_ext/src/motor_controller.cpp
CMakeFiles/motor_cpp.dir/src/motor_controller.cpp.o: CMakeFiles/motor_cpp.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/umdloop/auto_typing_ws/_skbuild/linux-x86_64-3.10/cmake-build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/motor_cpp.dir/src/motor_controller.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/motor_cpp.dir/src/motor_controller.cpp.o -MF CMakeFiles/motor_cpp.dir/src/motor_controller.cpp.o.d -o CMakeFiles/motor_cpp.dir/src/motor_controller.cpp.o -c /home/umdloop/auto_typing_ws/cpp_ext/src/motor_controller.cpp

CMakeFiles/motor_cpp.dir/src/motor_controller.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/motor_cpp.dir/src/motor_controller.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/umdloop/auto_typing_ws/cpp_ext/src/motor_controller.cpp > CMakeFiles/motor_cpp.dir/src/motor_controller.cpp.i

CMakeFiles/motor_cpp.dir/src/motor_controller.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/motor_cpp.dir/src/motor_controller.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/umdloop/auto_typing_ws/cpp_ext/src/motor_controller.cpp -o CMakeFiles/motor_cpp.dir/src/motor_controller.cpp.s

# Object files for target motor_cpp
motor_cpp_OBJECTS = \
"CMakeFiles/motor_cpp.dir/src/bindings.cpp.o" \
"CMakeFiles/motor_cpp.dir/src/motor_controller.cpp.o"

# External object files for target motor_cpp
motor_cpp_EXTERNAL_OBJECTS =

motor_cpp.cpython-310-x86_64-linux-gnu.so: CMakeFiles/motor_cpp.dir/src/bindings.cpp.o
motor_cpp.cpython-310-x86_64-linux-gnu.so: CMakeFiles/motor_cpp.dir/src/motor_controller.cpp.o
motor_cpp.cpython-310-x86_64-linux-gnu.so: CMakeFiles/motor_cpp.dir/build.make
motor_cpp.cpython-310-x86_64-linux-gnu.so: /home/umdloop/auto_typing_ws/cpp_ext/lib/x86-64/libCTRE_Phoenix.so
motor_cpp.cpython-310-x86_64-linux-gnu.so: /home/umdloop/auto_typing_ws/cpp_ext/lib/x86-64/libCTRE_PhoenixCCI.so
motor_cpp.cpython-310-x86_64-linux-gnu.so: /home/umdloop/auto_typing_ws/cpp_ext/lib/x86-64/libCTRE_PhoenixTools.so
motor_cpp.cpython-310-x86_64-linux-gnu.so: CMakeFiles/motor_cpp.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/umdloop/auto_typing_ws/_skbuild/linux-x86_64-3.10/cmake-build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX shared module motor_cpp.cpython-310-x86_64-linux-gnu.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/motor_cpp.dir/link.txt --verbose=$(VERBOSE)
	/usr/bin/strip /home/umdloop/auto_typing_ws/_skbuild/linux-x86_64-3.10/cmake-build/motor_cpp.cpython-310-x86_64-linux-gnu.so

# Rule to build all files generated by this target.
CMakeFiles/motor_cpp.dir/build: motor_cpp.cpython-310-x86_64-linux-gnu.so
.PHONY : CMakeFiles/motor_cpp.dir/build

CMakeFiles/motor_cpp.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/motor_cpp.dir/cmake_clean.cmake
.PHONY : CMakeFiles/motor_cpp.dir/clean

CMakeFiles/motor_cpp.dir/depend:
	cd /home/umdloop/auto_typing_ws/_skbuild/linux-x86_64-3.10/cmake-build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/umdloop/auto_typing_ws/cpp_ext /home/umdloop/auto_typing_ws/cpp_ext /home/umdloop/auto_typing_ws/_skbuild/linux-x86_64-3.10/cmake-build /home/umdloop/auto_typing_ws/_skbuild/linux-x86_64-3.10/cmake-build /home/umdloop/auto_typing_ws/_skbuild/linux-x86_64-3.10/cmake-build/CMakeFiles/motor_cpp.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/motor_cpp.dir/depend

