module vk_headless;
import std::io, nogfx;

// The Vulkan backend without a window: memory heaps, root pointers, compute (direct and indirect), texture copies, descriptor heaps, render
// passes (direct, indexed and indirect draws), barriers, timeline semaphores, timestamps and readback. Every step checks what the GPU wrote
// and the program exits with 1 if any check fails. A machine without the Vulkan 1.4 extensions the backend needs is reported and skipped.
//
// The shaders are GLSL in tests/vulkan, compiled to SPIR-V by tests/vulkan/build.sh. The root is the uniform buffer at set 0, binding 0.
// The shaders that sample a texture through the descriptor heaps need SPV_EXT_descriptor_heap, which needs Slang: see upstream's
// examples. What the descriptor heaps do on the CPU side, and binding them, is covered here.

const char[*] SCALE = $embed("vulkan/scale.comp.spv");
const char[*] TRIANGLE_VERTEX = $embed("vulkan/triangle.vert.spv");
const char[*] TRIANGLE_FRAGMENT = $embed("vulkan/triangle.frag.spv");

struct ScaleRoot
{
	float* source;
	float* destination;
	uint count;
	float factor;
}

struct Vertex
{
	float[<4>] position;
	float[<4>] color;
}

struct DrawRoot
{
	Vertex* vertices;
}

struct DrawIndirectCommand
{
	uint vertex_count;
	uint instance_count;
	uint first_vertex;
	uint first_instance;
}

int failures;
bool stub;

macro check(bool condition, String what)
{
	io::printfn("%s  %s", condition ? "pass" : "FAIL", what);
	if (!condition) failures++;
}

// A check of what a shader wrote: the stub driver runs no shaders, so it is skipped there.
macro check_shader(bool condition, String what)
{
	if (stub)
	{
		io::printfn("skip  %s", what);
		return;
	}
	check(condition, what);
}

fn ulong align_up(ulong value, ulong alignment) => (value + alignment - 1) / alignment * alignment;

fn int main()
{
	nogfx::set_error_callback(fn (message, fatal) { io::printfn("callback: %s", message); });
	Device*? created = nogfx::create_device({ .timestamp_query_count = 8 });
	if (catch err = created)
	{
		io::printfn("skipped: no device (%s). The backend needs Vulkan 1.4 with VK_EXT_descriptor_heap, VK_KHR_shader_untyped_pointers and "
			"VK_EXT_mesh_shader.", err);
		return err == nogfx::UNSUPPORTED ? 0 : 1;
	}
	Device* device = created;
	defer device.destroy();
	io::printfn("device %s, %d queue(s), %d bytes of push data, texture heap alignment %d", device.caps.device_name, device.caps.queue_count,
		device.caps.max_push_data_size, device.caps.texture_heap_alignment);
	// The stub driver of tests/fake_vulkan runs no shaders: it can only check what is copied and the order of the calls.
	stub = device.caps.device_name == "Fake Vulkan GPU";
	check(device.caps.queue_count >= 1 && device.caps.general_queue_count >= 1 && device.caps.indirect_mesh_draw, "caps: a general queue, indirect mesh draws");
	check(device.caps.max_push_data_size >= 8, "caps: eight bytes of push data for the root pointer");

	// Memory: a CPU visible heap for what the CPU writes, a GPU only heap, and a readback heap. Offsets inside them are the application's.
	GpuHeap upload = device.create_gpu_heap(1 << 20, CPU_VISIBLE)!!;
	GpuHeap gpu = device.create_gpu_heap(1 << 20, GPU_ONLY)!!;
	GpuHeap readback = device.create_gpu_heap(1 << 20, READBACK)!!;
	defer { upload.destroy(); gpu.destroy(); readback.destroy(); }
	check(upload.cpu && !gpu.cpu && readback.cpu, "heaps: cpu pointers for the visible and readback heaps only");
	check((((uptr)upload.gpu | (uptr)gpu.gpu | (uptr)readback.gpu) & 15) == 0, "heaps: gpu addresses are 16 byte aligned");

	const uint COUNT = 1000;
	const ulong ROOTS = 8192;
	float* source = (float*)upload.cpu;
	for (uint i = 0; i < COUNT; i++) source[i] = (float)i;
	ScaleRoot* scale_root = (ScaleRoot*)(upload.cpu + ROOTS);
	*scale_root = { .source = (float*)gpu.gpu, .destination = (float*)(gpu.gpu + 4096), .count = COUNT, .factor = 2.0f };
	ScaleRoot* indirect_root = (ScaleRoot*)(upload.cpu + ROOTS + 64);
	*indirect_root = { .source = (float*)gpu.gpu, .destination = (float*)(gpu.gpu + 8192), .count = COUNT, .factor = 3.0f };
	// A dispatch's arguments are three uints, read from the GPU address given to dispatch_indirect.
	uint* groups = (uint*)(upload.cpu + ROOTS + 128);
	*groups = (COUNT + 63) / 64;
	groups[1] = 1;
	groups[2] = 1;
	// Three triangles that cover the whole target, each a different color, and the arguments for drawing the third indirectly.
	Vertex* red = (Vertex*)(upload.cpu + ROOTS + 256);
	Vertex* green = (Vertex*)(upload.cpu + ROOTS + 512);
	Vertex* yellow = (Vertex*)(upload.cpu + ROOTS + 768);
	foreach (i, positions : (float[<4>][3]){ { -1, -1, 0, 1 }, { 3, -1, 0, 1 }, { -1, 3, 0, 1 } })
	{
		red[i] = { positions, { 1, 0, 0, 1 } };
		green[i] = { positions, { 0, 1, 0, 1 } };
		yellow[i] = { positions, { 1, 1, 0, 1 } };
	}
	DrawRoot* red_root = (DrawRoot*)(upload.cpu + ROOTS + 192);
	DrawRoot* green_root = (DrawRoot*)(upload.cpu + ROOTS + 208);
	DrawRoot* yellow_root = (DrawRoot*)(upload.cpu + ROOTS + 224);
	red_root.vertices = (Vertex*)(upload.gpu + ROOTS + 256);
	green_root.vertices = (Vertex*)(upload.gpu + ROOTS + 512);
	yellow_root.vertices = (Vertex*)(upload.gpu + ROOTS + 768);
	ushort* indices = (ushort*)(upload.cpu + ROOTS + 1024);
	indices[0] = 0;
	indices[1] = 1;
	indices[2] = 2;
	DrawIndirectCommand* indirect = (DrawIndirectCommand*)(upload.cpu + ROOTS + 1040);
	*indirect = { .vertex_count = 3, .instance_count = 1 };

	// Pipelines.
	Pso* scale_pso = device.create_compute_pso({ .code = SCALE[..] })!!;
	Pso* triangle_pso = device.create_graphics_pso({
		.vertex = { .code = TRIANGLE_VERTEX[..] },
		.fragment = { .code = TRIANGLE_FRAGMENT[..] },
		.color_targets = { { .format = RGBA8_UNORM } },
	})!!;
	// Nothing set but enabled: the default blend factors pass the source color through.
	Pso* blend_pso = device.create_graphics_pso({
		.vertex = { .code = TRIANGLE_VERTEX[..] },
		.fragment = { .code = TRIANGLE_FRAGMENT[..] },
		.color_targets = { { .format = RGBA8_UNORM, .blend = { .enabled = true } } },
	})!!;
	defer { scale_pso.destroy(); triangle_pso.destroy(); blend_pso.destroy(); }

	// Textures: placed in a texture heap, and made available to shaders as indices into descriptor heaps.
	TextureDesc storage_desc = { .extent = { 64, 64, 1 }, .format = RGBA8_UNORM, .usage = { .sampled = true, .storage = true, .transfer_source = true, .transfer_destination = true } };
	TextureDesc target_desc = { .extent = { 64, 64, 1 }, .format = RGBA8_UNORM, .usage = { .sampled = true, .color_attachment = true, .transfer_source = true } };
	SizeAlign storage_size = device.get_texture_size_align(storage_desc)!!;
	SizeAlign target_size = device.get_texture_size_align(target_desc)!!;
	check(storage_size.size >= 64 * 64 * 4 && device.caps.texture_heap_alignment % storage_size.align == 0, "textures: size and alignment of a 64x64 texture");
	ulong target_offset = align_up(storage_size.size, device.caps.texture_heap_alignment);
	TextureHeap texture_heap = device.create_texture_heap(target_offset + target_size.size)!!;
	defer texture_heap.destroy();
	TextureDescriptorHeap* textures = device.create_texture_descriptor_heap(16)!!;
	TextureDescriptorHeap* copies = device.create_texture_descriptor_heap(16)!!;
	SamplerDescriptorHeap* samplers = device.create_sampler_descriptor_heap(4)!!;
	defer { textures.destroy(); copies.destroy(); samplers.destroy(); }
	samplers.write(0, {});
	samplers.write(1, { .min_filter = NEAREST, .mag_filter = NEAREST, .address_u = CLAMP_TO_EDGE, .address_v = CLAMP_TO_EDGE });
	samplers.write(2, { .anisotropic = true, .compare_enabled = true, .compare = GREATER });
	samplers.copy_from(3, samplers, 1, 1);

	CommandPool* pool = device.create_command_pool()!!;
	defer pool.destroy();
	TimelineSemaphore* timeline = device.create_timeline_semaphore()!!;
	defer timeline.destroy();
	ulong[4] stamps;

	CommandBuffer* commands = pool.begin_commands()!!;
	Texture* storage = device.create_texture(storage_desc, texture_heap, 0)!!;
	Texture* target = device.create_texture(target_desc, texture_heap, target_offset)!!;
	commands.initialize_texture(storage);
	commands.initialize_texture(target);
	RenderView* target_view = target.create_render_view()!!;
	defer { storage.destroy(); target.destroy(); target_view.destroy(); }
	textures.write(3, storage, STORAGE);
	textures.write(7, storage, SAMPLED);
	textures.write(8, target, SAMPLED, { .base_mip = 0, .mip_count = 1 });
	copies.copy_from(0, textures, 3, 6);
	commands.set_texture_descriptor_heap(textures);
	commands.set_sampler_descriptor_heap(samplers);

	commands.write_timestamp(&stamps[0]);

	// 1. Upload to the GPU only heap, scale it with a compute shader, and read it back.
	commands.copy_memory({ upload.gpu, COUNT * 4 }, { gpu.gpu, COUNT * 4 });
	commands.barrier({ .transfer = true }, { .transfer_write = true }, { .compute = true }, { .shader_read = true });
	commands.bind_pso(scale_pso);
	commands.dispatch(upload.gpu + ROOTS, { (COUNT + 63) / 64, 1, 1 });
	// The same, with the group count read from memory by the GPU.
	commands.dispatch_indirect(upload.gpu + ROOTS + 64, { upload.gpu + ROOTS + 128, 12 });
	commands.barrier({ .compute = true }, { .shader_write = true }, { .transfer = true }, { .transfer_read = true });
	commands.copy_memory({ gpu.gpu + 4096, COUNT * 4 }, { readback.gpu, COUNT * 4 });
	commands.copy_memory({ gpu.gpu + 8192, COUNT * 4 }, { readback.gpu + 4096, COUNT * 4 });

	// 2. A texture round trip: a pattern written to it from memory, then a block of it and then all of it read back.
	char[4]* pattern = (char[4]*)(upload.cpu + 65536);
	for (uint y = 0; y < 64; y++) for (uint x = 0; x < 64; x++) pattern[y * 64 + x] = { (char)x, (char)y, (char)(x ^ y), 255 };
	commands.copy_memory_to_texture({ upload.gpu + 65536, 64 * 64 * 4 }, storage);
	commands.barrier({ .transfer = true }, { .transfer_write = true }, { .transfer = true }, { .transfer_read = true });
	commands.copy_texture_to_memory(storage, { readback.gpu + 16384, 64 * 64 * 4 });
	commands.copy_texture_to_memory(storage, { readback.gpu + 32768, 16 * 16 * 4 }, { .offset = { 8, 24, 0 }, .extent = { 16, 16, 1 } });

	// 3. Render passes that draw a triangle covering the target, with its vertices behind the root pointer: a direct draw, an indexed one
	//    and an indirect one. Each pass clears to blue first.
	for (uint pass = 0; pass < 3; pass++)
	{
		commands.barrier({ .transfer = true }, { .transfer_read = true }, { .color_output = true }, { .color_write = true });
		commands.begin_render_pass({ .colors = { { .render_view = target_view, .load = CLEAR, .clear = { 0, 0, 1, 1 } } } });
		commands.bind_pso(triangle_pso);
		switch (pass)
		{
			case 0: commands.draw(upload.gpu + ROOTS + 192, 3);
			case 1: commands.draw_indexed(upload.gpu + ROOTS + 208, { upload.gpu + ROOTS + 1024, 6 }, UINT16, 3);
			case 2: commands.draw_indirect(upload.gpu + ROOTS + 224, { upload.gpu + ROOTS + 1040, 16 });
		}
		commands.end_render_pass();
		commands.barrier({ .color_output = true }, { .color_write = true }, { .transfer = true }, { .transfer_read = true });
		commands.copy_texture_to_memory(target, { readback.gpu + 65536 + pass * 16384, 64 * 64 * 4 });
	}

	// 4. A render pass that loads what is there and draws nothing outside a small scissor.
	commands.barrier({ .transfer = true }, { .transfer_read = true }, { .color_output = true }, { .color_write = true });
	commands.begin_render_pass({ .colors = { { .render_view = target_view, .load = LOAD } } });
	commands.bind_pso(triangle_pso);
	commands.set_scissor({ 0, 0, 8, 8 });
	commands.draw(upload.gpu + ROOTS + 192, 3);
	commands.end_render_pass();
	commands.barrier({ .color_output = true }, { .color_write = true }, { .transfer = true }, { .transfer_read = true });
	commands.copy_texture_to_memory(target, { readback.gpu + 131072, 64 * 64 * 4 });

	// 5. The same with blending enabled and nothing else set: the red triangle must show, not nothing.
	commands.barrier({ .transfer = true }, { .transfer_read = true }, { .color_output = true }, { .color_write = true });
	commands.begin_render_pass({ .colors = { { .render_view = target_view, .load = CLEAR, .clear = { 0, 0, 1, 1 } } } });
	commands.bind_pso(blend_pso);
	commands.draw(upload.gpu + ROOTS + 192, 3);
	commands.end_render_pass();
	commands.barrier({ .color_output = true }, { .color_write = true }, { .transfer = true }, { .transfer_read = true });
	commands.copy_texture_to_memory(target, { readback.gpu + 196608, 64 * 64 * 4 });

	commands.write_timestamp(&stamps[1]);
	commands.end();

	TimelinePoint done = { timeline, 1 };
	device.submit({ .commands = { commands }, .completion = done });
	done.wait();
	check(timeline.completed_value() == 1, "timeline: the completion value was signaled");
	pool.read_timestamps();

	float* scaled = (float*)readback.cpu;
	float* scaled_indirect = (float*)(readback.cpu + 4096);
	bool scale_ok = true;
	bool indirect_ok = true;
	for (uint i = 0; i < COUNT; i++)
	{
		scale_ok &= scaled[i] == (float)i * 2.0f + 1.0f;
		indirect_ok &= scaled_indirect[i] == (float)i * 3.0f + 1.0f;
	}
	check_shader(scale_ok, "compute: 1000 floats scaled through root pointers and copied back");
	check_shader(indirect_ok, "compute: the same with the group count read by the GPU (dispatch_indirect)");

	char[4]* round_trip = (char[4]*)(readback.cpu + 16384);
	bool round_trip_ok = true;
	for (uint i = 0; i < 64 * 64; i++) round_trip_ok &= round_trip[i] == pattern[i];
	check(round_trip_ok, "texture: a 64x64 RGBA8 texture written from memory and read back is unchanged");
	char[4]* block = (char[4]*)(readback.cpu + 32768);
	bool block_ok = true;
	for (uint y = 0; y < 16; y++) for (uint x = 0; x < 16; x++) block_ok &= block[y * 16 + x] == pattern[(24 + y) * 64 + 8 + x];
	check(block_ok, "texture: a 16x16 block at (8,24) read back through a copy offset and extent");

	String[3] names = { "direct draw", "indexed draw", "indirect draw" };
	char[4][3] colors = { { 255, 0, 0, 255 }, { 0, 255, 0, 255 }, { 255, 255, 0, 255 } };
	for (uint pass = 0; pass < 3; pass++)
	{
		char[4]* drawn = (char[4]*)(readback.cpu + 65536 + pass * 16384);
		check_shader(drawn[32 * 64 + 32] == colors[pass] && drawn[0] == colors[pass] && drawn[63 * 64 + 63] == colors[pass], names[pass]);
	}
	char[4]* scissored = (char[4]*)(readback.cpu + 131072);
	check_shader(scissored[4 * 64 + 4] == (char[4]){ 255, 0, 0, 255 } && scissored[32 * 64 + 32] == (char[4]){ 255, 255, 0, 255 },
		"render pass: LOAD keeps the contents, the scissor limits the new draw to 8x8 pixels");

	char[4]* blended = (char[4]*)(readback.cpu + 196608);
	check_shader(blended[32 * 64 + 32] == (char[4]){ 255, 0, 0, 255 }, "blending: enabled with the default factors passes the source color");
	SizeAlign explicit_size = device.get_texture_size_align({ .extent = { 16, 16, 1 }, .format = RGBA8_UNORM, .usage = { .sampled = true } })!!;
	SizeAlign default_size = device.get_texture_size_align({ .extent = { 16, 16, 1 } })!!;
	check(explicit_size.size == default_size.size && explicit_size.align == default_size.align, "texture: a default description is a sampled RGBA8_UNORM");

	check(stamps[0] > 0 && stamps[1] > stamps[0], "timestamps: increasing");
	if (stamps[1] > stamps[0]) io::printfn("      the commands took %.3f ms on the GPU", (double)(stamps[1] - stamps[0]) * device.caps.timestamp_period_ns / 1e6);

	device.wait_idle();
	io::printfn(failures ? "%d check(s) FAILED" : "all checks passed", failures);
	return failures ? 1 : 0;
}
