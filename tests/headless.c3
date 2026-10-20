module headless;
import std::io, std::math, nogfx;

// Runs the backend without a window: memory heaps, root pointers, compute, descriptor heaps, an offscreen render pass, barriers, timeline
// semaphores, timestamps and readback. Every step checks what the GPU wrote and the program exits with 1 if any check fails.

const String SHADERS = `
#include <metal_stdlib>
using namespace metal;

// Shaders get the root pointer at buffer(0), the base of the texture descriptor heap at buffer(1) and the sampler heap at buffer(2).
template<typename T> T gpu_texture_from_id(ulong id) { return *reinterpret_cast<thread T*>(&id); }
struct GPUSampler { sampler value; };

struct ScaleRoot { device const float* input; device float* output; uint count; float factor; };
kernel void scale(device const ScaleRoot* root [[buffer(0)]], uint id [[thread_position_in_grid]])
{
	if (id < root->count) root->output[id] = root->input[id] * root->factor + 1.0;
}

struct FillRoot { uint texture_index; uint size; };
kernel void fill(device const FillRoot* root [[buffer(0)]], constant ulong& pool [[buffer(1)]], uint2 id [[thread_position_in_grid]])
{
	auto texture = gpu_texture_from_id<texture2d<float, access::write>>(pool + root->texture_index);
	if (id.x < root->size && id.y < root->size) texture.write(float4(float(id.x) / 255.0, float(id.y) / 255.0, 0.5, 1.0), id);
}

struct Vertex { float4 position; float4 color; };
struct DrawRoot { device const Vertex* vertices; };
struct VOut { float4 position [[position]]; float4 color; };
vertex VOut triangle_vertex(device const DrawRoot* root [[buffer(0)]], uint vid [[vertex_id]])
{
	return { root->vertices[vid].position, root->vertices[vid].color };
}
fragment float4 triangle_fragment(VOut in [[stage_in]]) { return in.color; }

struct SampleRoot { uint texture_index; uint sampler_index; };
constant float2 quad_corners[6] = { float2(-1, -1), float2(1, -1), float2(-1, 1), float2(-1, 1), float2(1, -1), float2(1, 1) };
vertex VOut quad_vertex(uint vid [[vertex_id]])
{
	float2 p = quad_corners[vid];
	return { float4(p, 0.0, 1.0), float4(p * 0.5 + 0.5, 0.0, 1.0) };
}
fragment float4 quad_fragment(VOut in [[stage_in]], device const SampleRoot* root [[buffer(0)]], constant ulong& pool [[buffer(1)]],
	device const GPUSampler* samplers [[buffer(2)]])
{
	auto texture = gpu_texture_from_id<texture2d<float>>(pool + root->texture_index);
	return texture.sample(samplers[root->sampler_index].value, in.color.xy);
}
`;

struct ScaleRoot
{
	float* input;
	float* output;
	uint count;
	float factor;
}

struct FillRoot
{
	uint texture_index;
	uint size;
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

struct SampleRoot
{
	uint texture_index;
	uint sampler_index;
}

int failures;

macro check(bool condition, String what)
{
	io::printfn("%s  %s", condition ? "pass" : "FAIL", what);
	if (!condition) failures++;
}

fn ShaderStage stage(String entry, Uint3 size = {})
{
	return { .code = SHADERS, .entry_point = entry.zstr_tcopy(), .threadgroup_size = size };
}

fn int main()
{
	nogfx::set_error_callback(fn (message, fatal) { io::printfn("callback: %s", message); });
	Device* device = nogfx::create_device({ .timestamp_query_count = 8 })!!;
	defer device.destroy();
	io::printfn("device %s, %d queue(s), Apple9 indirect mesh draws: %s", device.caps.device_name, device.caps.queue_count, device.caps.indirect_mesh_draw);

	// Memory: a CPU visible heap for what the CPU writes, a GPU only heap, and a readback heap. Offsets inside them are the application's.
	GpuHeap upload = device.create_gpu_heap(1 << 20, CPU_VISIBLE)!!;
	GpuHeap gpu = device.create_gpu_heap(1 << 20, GPU_ONLY)!!;
	GpuHeap readback = device.create_gpu_heap(1 << 20, READBACK)!!;
	defer { upload.destroy(); gpu.destroy(); readback.destroy(); }
	check(upload.cpu && !gpu.cpu && readback.cpu, "heaps: cpu pointers for the visible and readback heaps only");
	check((((uptr)upload.gpu | (uptr)gpu.gpu | (uptr)readback.gpu) & 15) == 0, "heaps: gpu addresses are 16 byte aligned");

	const uint COUNT = 1000;
	const ulong ROOTS = 8192;
	float* input = (float*)upload.cpu;
	for (uint i = 0; i < COUNT; i++) input[i] = (float)i;
	ScaleRoot* scale_root = (ScaleRoot*)(upload.cpu + ROOTS);
	*scale_root = { .input = (float*)gpu.gpu, .output = (float*)(gpu.gpu + 4096), .count = COUNT, .factor = 2.0f };
	FillRoot* fill_root = (FillRoot*)(upload.cpu + ROOTS + 64);
	*fill_root = { .texture_index = 3, .size = 64 };
	DrawRoot* draw_root = (DrawRoot*)(upload.cpu + ROOTS + 128);
	Vertex* vertices = (Vertex*)(upload.cpu + ROOTS + 256);
	vertices[0] = { { -1, -1, 0, 1 }, { 1, 0, 0, 1 } };
	vertices[1] = { { 3, -1, 0, 1 }, { 1, 0, 0, 1 } };
	vertices[2] = { { -1, 3, 0, 1 }, { 1, 0, 0, 1 } };
	draw_root.vertices = (Vertex*)(upload.gpu + ROOTS + 256);
	SampleRoot* sample_root = (SampleRoot*)(upload.cpu + ROOTS + 192);
	*sample_root = { .texture_index = 7, .sampler_index = 1 };

	// Pipelines.
	Pso* scale_pso = device.create_compute_pso(stage("scale", { 64, 1, 1 }))!!;
	Pso* fill_pso = device.create_compute_pso(stage("fill", { 8, 8, 1 }))!!;
	Pso* triangle_pso = device.create_graphics_pso({
		.vertex = stage("triangle_vertex"),
		.fragment = stage("triangle_fragment"),
		.color_targets = { { .format = RGBA8_UNORM } },
	})!!;
	Pso* quad_pso = device.create_graphics_pso({
		.vertex = stage("quad_vertex"),
		.fragment = stage("quad_fragment"),
		.color_targets = { { .format = RGBA8_UNORM } },
	})!!;
	defer { scale_pso.destroy(); fill_pso.destroy(); triangle_pso.destroy(); quad_pso.destroy(); }

	// Textures: placed in a texture heap, and made available to shaders as indices into descriptor heaps.
	TextureDesc storage_desc = { .extent = { 64, 64, 1 }, .format = RGBA8_UNORM, .usage = { .sampled = true, .storage = true, .transfer_source = true } };
	TextureDesc target_desc = { .extent = { 64, 64, 1 }, .format = RGBA8_UNORM, .usage = { .sampled = true, .color_attachment = true, .transfer_source = true } };
	SizeAlign storage_size = device.get_texture_size_align(storage_desc)!!;
	SizeAlign target_size = device.get_texture_size_align(target_desc)!!;
	ulong target_offset = align_up(storage_size.size, device.caps.texture_heap_alignment);
	TextureHeap texture_heap = device.create_texture_heap(target_offset + target_size.size)!!;
	defer texture_heap.destroy();
	TextureDescriptorHeap* textures = device.create_texture_descriptor_heap(16)!!;
	SamplerDescriptorHeap* samplers = device.create_sampler_descriptor_heap(4)!!;
	defer { textures.destroy(); samplers.destroy(); }
	samplers.write(0, {});
	samplers.write(1, { .min_filter = NEAREST, .mag_filter = NEAREST, .address_u = CLAMP_TO_EDGE, .address_v = CLAMP_TO_EDGE });

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

	commands.write_timestamp(&stamps[0]);

	// 1. Upload to the GPU only heap, scale it with a compute shader, and read it back.
	commands.copy_memory({ upload.gpu, COUNT * 4 }, { gpu.gpu, COUNT * 4 });
	commands.barrier({ .transfer = true }, { .transfer_write = true }, { .compute = true }, { .shader_read = true });
	commands.bind_pso(scale_pso);
	commands.dispatch(upload.gpu + ROOTS, { (COUNT + 63) / 64, 1, 1 });
	commands.barrier({ .compute = true }, { .shader_write = true }, { .transfer = true }, { .transfer_read = true });
	commands.copy_memory({ gpu.gpu + 4096, COUNT * 4 }, { readback.gpu, COUNT * 4 });

	// 2. A compute shader writes a texture through the descriptor heap.
	commands.set_texture_descriptor_heap(textures);
	commands.bind_pso(fill_pso);
	commands.dispatch(upload.gpu + ROOTS + 64, { 8, 8, 1 });
	commands.barrier({ .compute = true }, { .shader_write = true }, { .transfer = true }, { .transfer_read = true });
	commands.copy_texture_to_memory(storage, { readback.gpu + 8192, 64 * 64 * 4 });

	// 3. A render pass draws a triangle that covers the target, with its vertices behind the root pointer.
	commands.begin_render_pass({ .colors = { { .render_view = target_view, .load = CLEAR, .clear = { 0, 0, 1, 1 } } } });
	commands.bind_pso(triangle_pso);
	commands.draw(upload.gpu + ROOTS + 128, 3);
	commands.end_render_pass();
	commands.barrier({ .color_output = true }, { .color_write = true }, { .transfer = true }, { .transfer_read = true });
	commands.copy_texture_to_memory(target, { readback.gpu + 32768, 64 * 64 * 4 });

	// 4. A render pass samples the texture from 2 with a nearest sampler: a second target would do, the cleared one is reused.
	commands.barrier({ .transfer = true }, { .transfer_read = true }, { .color_output = true }, { .color_write = true });
	commands.begin_render_pass({ .colors = { { .render_view = target_view, .load = CLEAR } } });
	commands.bind_pso(quad_pso);
	commands.set_texture_descriptor_heap(textures);
	commands.set_sampler_descriptor_heap(samplers);
	commands.set_viewport({ 0, 0, 64, 64, 0, 1 });
	commands.draw(upload.gpu + ROOTS + 192, 6);
	commands.end_render_pass();
	commands.barrier({ .color_output = true }, { .color_write = true }, { .transfer = true }, { .transfer_read = true });
	commands.copy_texture_to_memory(target, { readback.gpu + 65536, 64 * 64 * 4 });

	commands.write_timestamp(&stamps[1]);
	commands.end();

	TimelinePoint done = { timeline, 1 };
	device.submit({ .commands = { commands }, .completion = done });
	done.wait();
	check(timeline.completed_value() == 1, "timeline: the completion value was signaled");
	pool.read_timestamps();

	float* scaled = (float*)readback.cpu;
	bool scale_ok = true;
	for (uint i = 0; i < COUNT; i++) scale_ok &= scaled[i] == (float)i * 2.0f + 1.0f;
	check(scale_ok, "compute: 1000 floats scaled through root pointers and copied back");

	char[4]* filled = (char[4]*)(readback.cpu + 8192);
	check(filled[0] == (char[4]){ 0, 0, 127, 255 } || filled[0] == (char[4]){ 0, 0, 128, 255 }, "storage texture: pixel (0,0) written by the shader");
	check(filled[10 * 64 + 20][0] == 20 && filled[10 * 64 + 20][1] == 10, "storage texture: pixel (20,10) is (20,10,.5)");

	char[4]* drawn = (char[4]*)(readback.cpu + 32768);
	check(drawn[32 * 64 + 32] == (char[4]){ 255, 0, 0, 255 }, "render pass: center pixel is the vertex color, red");
	check(drawn[0] == (char[4]){ 255, 0, 0, 255 } && drawn[63 * 64 + 63] == (char[4]){ 255, 0, 0, 255 }, "render pass: corners are covered");

	char[4]* sampled = (char[4]*)(readback.cpu + 65536);
	check(sampled[10 * 64 + 20][0] > 15 && sampled[10 * 64 + 20][0] < 25, "sampling: the quad shows the texture written by compute");
	check(sampled[5 * 64 + 40][1] > 0 && sampled[5 * 64 + 40][1] < 12, "sampling: green follows the texture y coordinate");

	check(stamps[0] > 0 && stamps[1] > stamps[0], "timestamps: increasing");
	if (stamps[1] > stamps[0]) io::printfn("      the commands took %.3f ms on the GPU", (double)(stamps[1] - stamps[0]) * device.caps.timestamp_period_ns / 1e6);

	device.wait_idle();
	io::printfn(failures ? "%d check(s) FAILED" : "all checks passed", failures);
	return failures ? 1 : 0;
}

fn ulong align_up(ulong value, ulong alignment) => (value + alignment - 1) / alignment * alignment;
