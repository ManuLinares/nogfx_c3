# nogfx: NoGraphicsAPI for C3, on Metal 4 and Vulkan 1.4

A port of [NoGraphicsAPI](https://github.com/sebbbi/NoGraphicsAPI) (Sebastian Aaltonen, MIT) to C3, as a standalone project: it needs
`std::os::darwin` on Apple platforms and, for the example, `std::gfx` from a current c3c (built with 0.8.6 from master), and changes nothing in
the standard library. Both backends of upstream are here, with the same API: Metal 4 on Apple platforms, Vulkan 1.4 everywhere else.
It follows upstream's `main` as of ffd9f2a (2026-10-19, "Improve Metal 4 presentation and texture resource handling").

The model, as upstream has it: memory is raw GPU pointers into heaps the application owns; a draw or dispatch is a pipeline and one
16-byte aligned **root pointer** that the shader reads everything else from; textures and samplers are **indices** into descriptor heaps;
synchronization is **barriers by stage** and **timeline semaphores**. No bindings, no descriptor sets, no state tracking, no validation.
Resource destruction is immediate: wait for the GPU first.

```
src/nogfx.c3           the public types: formats, descriptors, stages, ...
src/common.c3          what both backends use: align_up, or_one, the normalized TextureDesc
src/address_map.c3     GPU address to buffer lookup (at most 64 live heaps, lock free reads), for Metal and for Vulkan without device address commands
src/metal.c3           the Metal 4 backend, ported function by function from src/NoGraphicsAPIMetal.mm
src/mtl.c3             the part of Metal 4 the backend talks to: constants generated from the SDK headers, value types, libdispatch
src/vulkan*.c3         the Vulkan backend, ported function by function from src/NoGraphicsAPI.cpp: device and instance, resources, swapchain, commands
src/vk.c3              the part of Vulkan the backend talks to, generated from vk.xml by tools/gen_vk.py. The loader is opened at run time
tools/                 gen_vk.py makes src/vk.c3; check_vk_layout.py compares every struct and constant in it with the C headers
examples/triangle.c3   Metal: a std::gfx window, compute-written texture sampled through the descriptor heaps, plus a blended triangle
tests/headless.c3      Metal, no window: heaps, root pointers, compute, descriptor heaps, a render pass, barriers, timeline, timestamps
tests/features.c3      Metal: indexed + indirect draws, depth, blending, descriptor copies, uploads, suspend/resume, aliasing, mesh shaders,
                       a compute and a copy queue, mip views, errors
tests/concurrency.c3   Metal: the lock free address lookup under concurrent heap creation and destruction
tests/timeline.c3      Both backends: a copy chain over three queues, submitted consumers first, waited for by polling and by waiting
tests/vk_headless.c3   Vulkan, no window: heaps, root pointers, compute (direct + indirect), texture copies, render passes (direct, indexed,
                       indirect), descriptor heaps, barriers, timeline, timestamps. Shaders are GLSL in tests/vulkan
tests/fake_vulkan/     a stub Vulkan driver that checks the calls; tests/fake_features.c3 and tests/run_fake.sh run the backend against it
```

```
c3c build triangle && ./out/triangle           # Metal. F toggles fullscreen, ESC quits
c3c build headless && ./out/headless           # Metal. Also: features, concurrency. Each exits 1 if a check fails
c3c build timeline && ./out/timeline           # Either backend: needs three queues
c3c build vk_headless && ./out/vk_headless     # Vulkan: needs a GPU with the extensions below, otherwise reports "skipped"
tests/run_fake.sh                              # Vulkan against the stub driver, no GPU needed (set C3C to a current c3c)
```

Which backend is built: `src/metal.c3` and `src/mtl.c3` on Apple platforms, `src/vulkan*.c3` and `src/vk.c3` elsewhere. Each file says so with a
`@feat` on its module, so a project lists all of `src` and gets one backend. To get Vulkan on an Apple platform, define `NOGFX_VULKAN`
(`c3c build -D NOGFX_VULKAN`, or `"features": ["NOGFX_VULKAN"]` in a target); the targets `vk_headless`, `fake_vulkan` and `fake_features` do.
The API is the same, only the shaders and the window differ.

Metal needs macOS 26 and an Apple7 GPU (M1 / A14 and later). Vulkan needs a recent driver, see below. Neither has a fallback.

## A frame

```c3
Device* device = nogfx::create_device({ .window = win.create_metal_layer()!!, .desired_swapchain_image_count = 3 })!!;   // an HWND on Vulkan
GpuHeap upload = device.create_gpu_heap(64 * 1024, CPU_VISIBLE)!!;           // .cpu and .gpu point at the same bytes
Pso* pso = device.create_graphics_pso({ .vertex = { .code = SOURCE, .entry_point = "vs" }, .fragment = { .code = SOURCE, .entry_point = "fs" },
                                        .color_targets = { { .format = BGRA8_UNORM } } })!!;

CommandBuffer* commands = pool.begin_commands()!!;
SwapchainFrame frame = commands.acquire();
commands.begin_render_pass({ .colors = { { .render_view = frame.render_view, .load = CLEAR } } });
commands.bind_pso(pso);
commands.draw(upload.gpu + root_offset, 3);        // the shader finds the vertices through this pointer
commands.end_render_pass();
commands.end();
device.submit_and_present({ .commands = { commands }, .completion = { timeline, ++value } });
```

On Metal the window and the device meet in one place: the `CAMetalLayer` that `GfxWindow.create_metal_layer()` returns. gfx keeps its drawable
size at the window's size in pixels and repaints it inside the resize transaction, so the device needs nothing else from the window.
On Vulkan the window is an `HWND` and presentation is Windows only, as upstream; a headless device (no window) works wherever there is a driver.

## Shaders

**Metal** takes a metallib (bytes that start with `MTLB`) or Metal Shading Language source, which it compiles when the pipeline is created.
Upstream only takes metallibs, made by Slang; source is here because the Metal toolchain (`xcrun metal`) is an optional download and runtime
compilation needs none. The shader ABI is the one upstream's `shader.slang` produces:

| slot | contents |
|---|---|
| `buffer(0)` | the root pointer passed to `draw` / `dispatch` |
| `buffer(1)` | the base resource id of the bound texture descriptor heap: add a texture index to it (`gpu_texture_from_id`) |
| `buffer(2)` | the bound sampler heap, an array of 8 byte `GPUSampler { sampler value; }` |

See `tests/headless.c3` and `examples/triangle.c3` for the 6 lines of MSL prelude.

**Vulkan** takes SPIR-V. The root is the uniform buffer at set 0, binding 0: the pipeline maps it to the address that `draw` / `dispatch` push
with `vkCmdPushDataEXT`, eight bytes, so nothing is allocated or copied. Textures and samplers are read through the bound descriptor heaps with
`SPV_EXT_descriptor_heap`, which today means Slang (upstream's `shader.slang`, see its `docs/slang.md`). Shaders that only use the root, such as
the ones in `tests/vulkan`, can be plain GLSL: `glslangValidator -V --target-env vulkan1.4`, with `GL_EXT_buffer_reference` and scalar block
layout for pointers in the root. SPIR-V does not have to be 4 byte aligned (a C3 `$embed` need not be): the backend copies it if it is not.

In both, a C3 struct has the layout of the shader's struct, a pointer in it is a GPU address, and `float[<4>]` is a `float4`.

## Vulkan requirements

Vulkan 1.4 alone is not enough. The device needs [`VK_EXT_descriptor_heap`][heap],
`VK_KHR_shader_untyped_pointers` and `VK_EXT_mesh_shader` with both its task and mesh features (and `VK_KHR_swapchain` with a window), and a long list of features that
`inspect_candidate` in `src/vulkan_device.c3` checks (they are upstream's). `VK_KHR_device_address_commands` is optional: without it the
index, indirect and copy commands take buffers, and addresses are resolved to them through `address_map.c3`, which holds 64 live GPU heaps and
needs every command range to lie in one heap. Of today's drivers that is recent NVIDIA and AMD on Windows and Mesa RADV on Linux; see upstream's README for the list.
**MoltenVK has none of it**, so Apple platforms use Metal. `create_device` returns `UNSUPPORTED` with a message that names every GPU and the
extensions it lacks.
`src/vk.c3` is made from Vulkan headers 1.4.357 or later, here 1.4.363 (`tools/gen_vk.py /opt/homebrew/share/vulkan/registry/vk.xml > src/vk.c3`); the
layouts of its structs and the values of its constants are checked against the C headers with
`tools/check_vk_layout.py registry/vk.xml include-dir c3c`.

[heap]: https://docs.vulkan.org/refpages/latest/refpages/source/VK_EXT_descriptor_heap.html

## Differences from the C++ API

- The functions are methods on the object they work on (`commands.draw(...)`, `device.create_gpu_heap(...)`, `heap.destroy()`).
- A creation that can fail returns an optional: `Device*?`, `GpuHeap?`, `RenderView*?`, ... with the faults `UNSUPPORTED`, `DEVICE_LOST`,
  `DRIVER_ERROR`, `OUT_OF_MEMORY` and `CREATION_FAILED`. Diagnostics still go to stderr and to the callback of `set_error_callback`.
- Flag types are bitstructs: `{ .compute = true }` for a `Stage`, `{ .sampled = true, .storage = true }` for a `TextureUsage`.
- A C3 struct has no field defaults, so everything is zero initialized, and zero is made to mean upstream's default wherever that can be said:
  the first value of an enum is the usual one (`TextureType.TWO_D`, `Filter.LINEAR`, `AddressMode.REPEAT`, `LoadOp.LOAD`, ...);
  `BlendFactor.DEFAULT` is ONE as a source and ZERO as a destination, so `{ .enabled = true }` passes the source color through;
  `CompareOp.DEFAULT` is LESS_EQUAL for the depth test and for comparison samplers and ALWAYS for a stencil face; stencil masks of zero are
  all bits; a texture without a format is RGBA8_UNORM and without a usage is `sampled`; zero extents and counts are one; a color target's
  `write_mask` of zero is all channels; a shader's `entry_point` of null is `"main"`; a swapchain without a format is BGRA8.
  What zero cannot say, and has to be set: a viewport's size and `max_depth` (1.0), a scissor's size, a depth attachment's `.clear = 1.0f`
  (zero otherwise) and a color attachment's clear alpha (zero otherwise, upstream's is 1.0).
- `DeviceDesc`: zero queues is one, zero swapchain images is two, and timestamps are off unless `timestamp_query_count` is set.
- C3's `Span<T>` is the slice `T[]`, `GpuCpuRange` is folded into `GpuHeap`, and `GpuRange`, `TimelinePoint` and `SubmitDesc` are as upstream.
- Textures are created from the device, `device.create_texture(desc, heap, offset)`, which records nothing; `commands.initialize_texture(texture)`
  makes it usable on the GPU (it does nothing on Metal). Textures that overlap in a heap are aliases: initialize the incoming one after the other
  was written. There is no `aliasable` flag and no `activate_texture_alias`.
- A render pass starts with no pipeline bound, also a resumed segment: `bind_pso` a graphics pipeline inside it, and a compute pipeline again
  before dispatching after it.
- Vulkan only: `ShaderStage.threadgroup_size` is ignored (SPIR-V carries it), and `DeviceCaps.max_push_data_size` is the extension's.
- Metal on macOS only: `DeviceDesc.steam_overlay_compatibility` lets Steam's overlay hook the presentation, from a second classic queue.

## Things to know

- **Metal: use three drawables with `create_metal_layer`.** Upstream's default of two (`desired_swapchain_image_count < 3`) makes `nextDrawable`
  block for a second at a time during a live resize, because gfx presents inside the Core Animation transaction, which holds a drawable
  until it commits. With three, the drag has no stalls beyond the one that every CAMetalLayer shows when the size first changes.
  `[drawable present]` after `signalDrawable:` is correct under `presentsWithTransaction`; no extra wait is needed.
- Metal pipelines are compiled synchronously and not cached, and draws are triangle lists only. Both are upstream's.
- The lookup from GPU address to `MTLBuffer` (copies need a buffer) is upstream's `BufferAddressMap`: a sorted array of 64 heaps behind a version
  that readers check. A copy, or a Vulkan index, indirect or copy command without device address commands, must lie in one heap.
- Asserts are C3's: on in debug and safe builds, off with `--safe=no`. As upstream, a bad argument is a failed assert, not an error. In debug
  and safe builds a Vulkan device also enables `VK_LAYER_KHRONOS_validation` and the debug messenger when they are installed.
- Metal: every function opens its own autorelease pool, as upstream's do, so the API can be called from any thread without one.
- Vulkan: queues and command pools are externally synchronized and there are no internal locks, as upstream. The loader is opened by
  `create_device` and closed by `Device.destroy`; nothing is linked.

## Tested

**Metal**, on an M3, macOS 26.6, debug and `-O2` / `-O3 --safe=no`: the four test programs pass; the example renders (checked in the window's
pixels), survives a live-resize drag with the content following the window exactly, and the fullscreen round trip (windowed, fullscreen,
windowed); it shuts down cleanly and does not leak per frame. Not tested: cube, array and 3D textures, compressed formats,
`MTL_SHADER_VALIDATION`, timestamps inside a render pass, a second window, `draw_meshlets_indirect`, and anything on a GPU other than Apple silicon.

**Vulkan has not run on a GPU.** The development machine is a Mac, and MoltenVK has none of the required extensions. What was done instead:

- `src/vk.c3` is generated, and `tools/check_vk_layout.py` compares its 148 structs (size, alignment, every member offset) and 998 constants
  with what a C compiler makes of the Vulkan 1.4.363 headers (Homebrew's) and of 1.4.365 (Khronos' repository): no differences.
- The backend compiles and links on macOS (`-D NOGFX_VULKAN`) and compiles for `windows-x64` (the Win32 presentation path) and `linux-x64`.
- On macOS `create_device` finds the system loader (1.4.363), enumerates the instance, and with portability enumeration finds MoltenVK's device
  (1.4.2, Vulkan 1.4.357), rejects it for the missing extensions and returns `UNSUPPORTED` with the message above. `vk_headless` reports that and skips.
- `tests/run_fake.sh` runs the backend against a stub driver (`tests/fake_vulkan`), with the C3 asserts and bounds checks on. The stub runs no
  shaders, but it checks the structure chains of pipeline creation (descriptor heap flag, root mapping, SPIR-V alignment), command buffer
  state, push data, timeline waits for values nobody signals and leaks, and executes copies between memory and images. Both tests pass on it,
  with `VK_KHR_device_address_commands` and (`FAKE_NO_ADDRESS_COMMANDS=1`) without it, where the buffer commands and the address map are used. A
  scratch copy of the backend with the Windows checks switched on also ran the swapchain against a stub surface: acquire / render / present
  handshakes, resize, a zero sized surface, format changes, no leaks.
- The SPIR-V of `tests/vulkan` is validated by `spirv-val`.

What this cannot show is that a real driver accepts the calls, that the validation layer is quiet, and that the shaders write what the test
expects. Run `vk_headless` on a machine with the extensions before relying on it; six of its checks are results that shaders write.

## License

MIT, as upstream: see `LICENSE`, which keeps Sebastian Aaltonen's notice for NoGraphicsAPI. `THIRD_PARTY_NOTICES.md` lists what else the sources
derive from (the Khronos Vulkan registry for the generated `src/vk.c3`).
