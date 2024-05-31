## About
This repository is a fork of Jan Spindler's Neural Radiance Caching (NRC) implementation with two new modifications and 
an experimental implementation of two transmittance estimation methods from the paper [An unbiased ray-marching transmittance estimator](https://developer.nvidia.com/blog/nvidia-research-an-unbiased-ray-marching-transmittance-estimator/)

First modification of NRC is adaptive train batch filtering. It splits input image into batches with a multi level uniform grid. Train batches that contained no scattering events are useless for the training and are excluded from the training. Multi level structure allows to join multiple batches into a single bigger batch to reduce training time.

Second modification of NRC adds cloud density to the input of NRC neural network. This modification increases training speed of the neural network by the factor of 6.

Detailed description of modifications present in this fork are described in the text of my master's thesis stored in the `doc` folder of this repository.

## Requirements
- CUDA v12.1 and NVIDIA GPU with [compute capability](https://developer.nvidia.com/cuda-gpus#compute) >= 8.6
- Visual Studio 22. I used Visual Studio 2022 and it is not guaranteed that everything will work on other versions.
- (optional, for plot recreation): Python and Jupyter Notebook

## Installation
#### IMPORTANT: some links that vcpkg references are unavailable in Russian Federation. You probably know how to deal with this. Travel to another country, do the installation steps, then go home.

Program was tested only on Windows 11 and instruction is written for x64-windows vcpkg triplet. If you have a different OS, you need to replace `x64-windows` in installation steps paths with your respective platform [triplet](https://learn.microsoft.com/en-us/vcpkg/concepts/triplets).

Installation steps:
1. Install [requirements](#requirements)
2. Open `Developer Command Prompt for VS 22`.
3. `git clone --recurse-submodules -j8 https://github.com/Fenrir7Asteron/NRC-HPM-Renderer.git`
4. `cd NRC-HPM-Renderer`
5. Open `NRC-HPM-Renderer` folder in Visual Studio 22
6. Configure root CMakeLists file with right click on Solution Explorer. \
In the process of configuring vcpkg will download all dependencies from vcpkg.json.
7. Return to Developer Command Prompt and execute \
`.\install-openvdb-<Target>.bat`
8. Go back to VS, set root CMakeLists as startup item and build the project

Project can be run with default arguments with Visual Studio. \
If you want to run it with different arguments through command line you can look up \
default arguments at the bottom of `src/main.cu` file and meaning of arguments in the `include/engine/AppConfig.hpp`.

Built project files are in the `out/build/<build-target>` folder.

Startup arguments may vary for different modifications in different branches.

## Branches
Different branches contain different modifications of the base NRC implementation:
- `master` contains unmodified Jan Spindler's base NRC implementation
- `train_filter_*` contain several train filtering options
- `input_*` contains various options for additional input to neural network
- `new_transmittance_estimator` contains implementation of unbiased and biased raymarching transmittance estimators from [this paper](https://developer.nvidia.com/blog/nvidia-research-an-unbiased-ray-marching-transmittance-estimator/)
)
- `combined` contains combination of modifications from `train_filter_adaptive`, `input_density` and `new_transmittance_estimator` branches
- `approximate_mie` implements importance sampling by phase function from [this paper](https://research.nvidia.com/labs/rtr/approximate-mie/) but it is more noisy than Henyey-Greenstein so it is not used in the `combined` branch