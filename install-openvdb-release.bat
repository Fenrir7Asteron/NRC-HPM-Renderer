cd openvdb
if not exist build mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX:PATH="../../openvdb-install" ^
-DCMAKE_PREFIX_PATH="../../out/build/x64-Release/vcpkg_installed/x64-windows" ^
-DTBB_ROOT="../../out/build/x64-Release/vcpkg_installed/x64-windows" ^
-DBlosc_ROOT="../../out/build/x64-Release/vcpkg_installed/x64-windows" ^
-DZLIB_ROOT="../../out/build/x64-Release/vcpkg_installed/x64-windows" ^
-DBoost_ROOT="../../out/build/x64-Release/vcpkg_installed/x64-windows" ^
-A x64 ..
cmake --build . --parallel 16 --config Release --target install
cd ../..
