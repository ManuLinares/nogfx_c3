#!/bin/sh
# Runs the Vulkan backend against the stub driver in tests/fake_vulkan, on a machine without a capable GPU: builds the stub as libvulkan,
# puts it first in the library path, and runs the Vulkan tests. The shaders do not run on the stub; see the comment in fake_vulkan.c3.
set -e
cd "$(dirname "$0")/.."
C3C=${C3C:-c3c}
mkdir -p out/fake
$C3C build fake_vulkan
case "$(uname)" in
	Darwin) cp out/fake_vulkan.dylib out/fake/libvulkan.dylib; export DYLD_LIBRARY_PATH="$PWD/out/fake" ;;
	*)      cp $(ls out/libfake_vulkan.so out/fake_vulkan.so 2>/dev/null | head -1) out/fake/libvulkan.so.1; export LD_LIBRARY_PATH="$PWD/out/fake" ;;
esac
$C3C build vk_headless
$C3C build fake_features
# With the device address commands, and without them: the backend then resolves addresses to buffers.
./out/vk_headless
./out/fake_features
FAKE_NO_ADDRESS_COMMANDS=1 ./out/vk_headless
FAKE_NO_ADDRESS_COMMANDS=1 ./out/fake_features
