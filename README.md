## Requirements
- CUDA v12.1 and NVIDIA GPU with [compute capability](https://developer.nvidia.com/cuda-gpus#compute) >= 8.6
- Visual Studio 22
- (optional, for plot recreation): Python and Jupyter Notebook

## Installation
#### IMPORTANT: some links that vcpkg references are unavailable in Russian Federation. You probably know how to deal with this. Travel to another country, do the installation steps, then go home.

Program was tested only on Windows 11 and instruction is written for x64-windows vcpkg triplet. If you have a different OS, you need to replace `x64-windows` in installation steps paths with your respective platform [triplet](https://learn.microsoft.com/en-us/vcpkg/concepts/triplets).

Installation steps:
1. Open `Developer Command Prompt for VS 22`. I used Visual Studio 2022 and it is not guaranteed that everything will work on other versions.
2. `git clone --recurse-submodules -j8 https://github.com/Fenrir7Asteron/NRC-HPM-Renderer.git`
3. `cd NRC-HPM-Renderer`
4. Open `NRC-HPM-Renderer` folder in Visual Studio 22. 
5. Configure root CMakeLists file with right click on Solution Explorer. \
In the process of configuring vcpkg will download all dependencies from vcpkg.json.
6. Return to Developer Command Prompt and execute \
`.\install-openvdb-<Target>.bat`
7. Go back to VS, set root CMakeLists as startup item and build the project

Project can be run with default arguments through the VS. \
If you want to run it with different arguments through command line you can look up \
default arguments at the bottom of `src/main.cu` file and meaning of arguments in the `include/engine/AppConfig.hpp`.

Startup arguments may vary for different modifications in different branches.
