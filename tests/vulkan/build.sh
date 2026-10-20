#!/bin/sh
# Rebuilds the SPIR-V that tests/vk_headless.c3 embeds. Needs glslangValidator (glslang 16 or later) and spirv-val.
# Vulkan 1.4 with the scalar block layout, which is what the backend's root pointer (a uniform buffer at set 0 binding 0) uses.
set -e
cd "$(dirname "$0")"
for shader in scale.comp triangle.vert triangle.frag; do
	glslangValidator -V --target-env vulkan1.4 -o "$shader.spv" "$shader"
	spirv-val --target-env vulkan1.4 --scalar-block-layout "$shader.spv"
done
