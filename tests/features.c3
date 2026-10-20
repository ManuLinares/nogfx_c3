module features;
import std::io, std::math, nogfx;

// The parts of the API that tests/headless.c3 does not touch: indexed and indirect draws and dispatches, depth testing, blending, descriptor
// copies, texture uploads with a row pitch, suspended and resumed render passes, a compute and a copy queue ordered with timeline
// semaphores, aliased textures, mesh shaders, and the error callback. Exits with 1 if a check fails.

const String SHADERS = `
#include <metal_stdlib>
using namespace metal;

template<typename T> T gpu_texture_from_id(ulong id) { return *reinterpret_cast<thread T*>(&id); }
struct GPUSampler { sampler value; };

struct Vertex { float4 position; float4 color; };
struct VRoot { device const Vertex* vertices; };
struct VOut { float4 position [[position]]; float4 color; };
vertex VOut vs(device const VRoot* root [[buffer(0)]], uint vid [[vertex_id]]) { return { root->vertices[vid].position, root->vertices[vid].color }; }
fragment float4 fs(VOut in [[stage_in]]) { return in.color; }

struct IotaRoot { device uint* out; };
kernel void iota(device const IotaRoot* root [[buffer(0)]], uint id [[thread_position_in_grid]]) { root->out[id] = id + 1; }

struct DoubleRoot { device float* data; uint count; };
kernel void twice(device const DoubleRoot* root [[buffer(0)]], uint id [[thread_position_in_grid]]) { if (id < root->count) root->data[id] *= 2.0; }

struct SampleRoot { uint texture_index; uint sampler_index; };
constant float2 corners[6] = { float2(-1, -1), float2(1, -1), float2(-1, 1), float2(-1, 1), float2(1, -1), float2(1, 1) };
vertex VOut quad_vertex(uint vid [[vertex_id]]) { float2 p = corners[vid]; return { float4(p, 0.0, 1.0), float4(p * 0.5 + 0.5, 0.0, 1.0) }; }
fragment float4 quad_fragment(VOut in [[stage_in]], device const SampleRoot* root [[buffer(0)]], constant ulong& pool [[buffer(1)]],
	device const GPUSampler* samplers [[buffer(2)]])
{
	auto texture = gpu_texture_from_id<texture2d<float>>(pool + root->texture_index);
	return texture.sample(samplers[root->sampler_index].value, in.color.xy);
}

using MeshT = metal::mesh<VOut, void, 3, 1, metal::topology::triangle>;
[[mesh]] void mesh_main(MeshT m, uint tid [[thread_index_in_threadgroup]])
{
	if (tid < 3)
	{
		VOut v;
		v.position = float4(tid == 1 ? 3.0 : -1.0, tid == 2 ? 3.0 : -1.0, 0.0, 1.0);
		v.color = float4(1.0, 0.0, 1.0, 1.0);
		m.set_vertex(tid, v);
		m.set_index(tid, tid);
	}
	if (tid == 0) m.set_primitive_count(1);
}
`;

struct Vertex
{
	float[<4>] position;
	float[<4>] color;
}

struct VRoot
{
	Vertex* vertices;
}

struct IotaRoot
{
	uint* output;
}

struct DoubleRoot
{
	float* data;
	uint count;
}

struct SampleRoot
{
	uint texture_index;
	uint sampler_index;
}

struct DrawIndexedArgs
{
	uint index_count;
	uint instance_count;
	uint index_start;
	int base_vertex;
	uint base_instance;
}

struct DrawArgs
{
	uint vertex_count;
	uint instance_count;
	uint vertex_start;
	uint base_instance;
}

int failures;

macro check(bool condition, String what)
{
	io::printfn("%s  %s", condition ? "pass" : "FAIL", what);
	if (!condition) failures++;
}

// Memory is carved up by hand: this is the "application owns the memory" part of the model. `cpu` and `gpu` point at the same bytes.
struct Memory
{
	char* cpu;
	char* gpu;
}

Device* device;
GpuHeap upload;
GpuHeap readback;
ulong upload_used;
ulong readback_used;
TimelineSemaphore* timeline;
ulong submitted;
CommandPool* pool;

fn Memory upload_alloc(ulong size)
{
	Memory result = { upload.cpu + upload_used, upload.gpu + upload_used };
	upload_used += (size + 255) & ~(ulong)255;
	return result;
}

fn Memory readback_alloc(ulong size)
{
	Memory result = { readback.cpu + readback_used, readback.gpu + readback_used };
	readback_used += (size + 255) & ~(ulong)255;
	return result;
}

fn ShaderStage stage(String entry, Uint3 size = {}) => { .code = SHADERS, .entry_point = entry.zstr_tcopy(), .threadgroup_size = size };

fn CommandBuffer* begin() => pool.begin_commands()!!;

// Ends, submits and waits for a buffer, or for a whole chain of them.
fn void finish(CommandBuffer*[] chain)
{
	foreach (c : chain) c.end();
	submitted++;
	TimelinePoint done = { timeline, submitted };
	device.submit({ .commands = chain, .completion = done });
	done.wait();
}

fn void finish1(CommandBuffer* commands) => finish({ commands });

alias Pixel = char[4];

// A pixel of a readback of a texture of `width` pixels per row.
fn Pixel pixel(Memory memory, uint width, uint x, uint y) => ((Pixel*)memory.cpu)[y * width + x];

fn bool near(Pixel p, int r, int g, int b, int tolerance = 3) => math::abs((int)p[0] - r) <= tolerance && math::abs((int)p[1] - g) <= tolerance && math::abs((int)p[2] - b) <= tolerance;

const uint W = 64;

// Textures are placed in one texture heap one after the other, 16 KB apart as the heap alignment says.
TextureHeap texture_heap;
ulong texture_used;
CommandBuffer* setup_commands;

fn Texture* new_texture(TextureDesc desc, ulong at = ulong::max)
{
	SizeAlign size = device.get_texture_size_align(desc)!!;
	ulong offset = at != ulong::max ? at : (texture_used + device.caps.texture_heap_alignment - 1) / device.caps.texture_heap_alignment * device.caps.texture_heap_alignment;
	if (at == ulong::max) texture_used = offset + size.size;
	Texture* texture = device.create_texture(desc, texture_heap, offset)!!;
	setup_commands.initialize_texture(texture);
	return texture;
}

fn void rt_pass(CommandBuffer* c, RenderView* view, bool clear_blue)
{
	c.begin_render_pass({ .colors = { { .render_view = view, .load = CLEAR, .clear = { 0, 0, clear_blue ? 1.0f : 0.0f, 1 } } } });
}

fn int main()
{
	nogfx::set_error_callback(fn (message, fatal) { io::printfn("      (error callback: %s)", message); });
	device = nogfx::create_device({ .timestamp_query_count = 0 })!!;
	defer device.destroy();
	upload = device.create_gpu_heap(1 << 20, CPU_VISIBLE)!!;
	readback = device.create_gpu_heap(8 << 20, READBACK)!!;
	GpuHeap gpu_only = device.create_gpu_heap(1 << 20, GPU_ONLY)!!;
	texture_heap = device.create_texture_heap(64 << 20)!!;
	defer { upload.destroy(); readback.destroy(); gpu_only.destroy(); texture_heap.destroy(); }
	timeline = device.create_timeline_semaphore()!!;
	pool = device.create_command_pool()!!;
	defer { pool.destroy(); timeline.destroy(); }

	Pso* color_pso = device.create_graphics_pso({ .vertex = stage("vs"), .fragment = stage("fs"), .color_targets = { { .format = RGBA8_UNORM } } })!!;
	Pso* blend_pso = device.create_graphics_pso({
		.vertex = stage("vs"),
		.fragment = stage("fs"),
		.color_targets = { { .format = RGBA8_UNORM, .blend = { .enabled = true, .color = { SOURCE_ALPHA, ONE_MINUS_SOURCE_ALPHA, ADD }, .alpha = { ONE, ZERO, ADD } } } },
	})!!;
	// Nothing set but enabled: the defaults pass the source color through.
	Pso* default_blend_pso = device.create_graphics_pso({ .vertex = stage("vs"), .fragment = stage("fs"), .color_targets = { { .format = RGBA8_UNORM, .blend = { .enabled = true } } } })!!;
	Pso* stencil_pso = device.create_graphics_pso({
		.vertex = stage("vs"), .fragment = stage("fs"), .color_targets = { { .format = RGBA8_UNORM } },
		.depth_format = D32_FLOAT_S8_UINT, .stencil_format = D32_FLOAT_S8_UINT,
	})!!;
	Pso* quad_pso = device.create_graphics_pso({ .vertex = stage("quad_vertex"), .fragment = stage("quad_fragment"), .color_targets = { { .format = RGBA8_UNORM } } })!!;
	Pso* iota_pso = device.create_compute_pso(stage("iota", { 64, 1, 1 }))!!;
	Pso* twice_pso = device.create_compute_pso(stage("twice", { 64, 1, 1 }))!!;
	defer { default_blend_pso.destroy(); stencil_pso.destroy(); color_pso.destroy(); blend_pso.destroy(); quad_pso.destroy(); iota_pso.destroy(); twice_pso.destroy(); }

	// All the textures of the test are created by one buffer.
	setup_commands = begin();
	TextureDesc target_desc = { .extent = { W, W, 1 }, .format = RGBA8_UNORM, .usage = { .sampled = true, .color_attachment = true, .transfer_source = true } };
	Texture* target = new_texture(target_desc);
	Texture* depth_texture = new_texture({ .extent = { W, W, 1 }, .format = D32_FLOAT, .usage = { .depth_stencil_attachment = true } });
	Texture* stencil_texture = new_texture({ .extent = { W, W, 1 }, .format = D32_FLOAT_S8_UINT, .usage = { .depth_stencil_attachment = true } });
	Texture* upload_texture = new_texture({ .extent = { 32, 32, 1 }, .format = RGBA8_UNORM, .usage = { .sampled = true, .transfer_source = true, .transfer_destination = true } });
	ulong alias_at = (texture_used + 16383) / 16384 * 16384;
	TextureDesc alias_desc = target_desc;
	Texture* alias_a = new_texture(alias_desc, alias_at);
	Texture* alias_b = new_texture(alias_desc, alias_at);
	Texture* mips = new_texture({ .extent = { W, W, 1 }, .mip_levels = 3, .format = RGBA8_UNORM, .usage = { .sampled = true, .color_attachment = true } });
	RenderView*[3] mip_views;
	foreach (i, &view : mip_views) *view = mips.create_render_view({ .mip_level = (uint)i })!!;
	RenderView* target_view = target.create_render_view()!!;
	RenderView* depth_view = depth_texture.create_render_view()!!;
	RenderView* stencil_view = stencil_texture.create_render_view()!!;
	RenderView* alias_a_view = alias_a.create_render_view()!!;
	RenderView* alias_b_view = alias_b.create_render_view()!!;
	finish1(setup_commands);
	defer { mips.destroy(); foreach (view : mip_views) view.destroy(); target.destroy(); depth_texture.destroy(); stencil_texture.destroy(); stencil_view.destroy(); upload_texture.destroy(); alias_a.destroy(); alias_b.destroy(); target_view.destroy(); depth_view.destroy(); alias_a_view.destroy(); alias_b_view.destroy(); }
	pool.reset();

	// Vertices and indices, in memory the GPU reads: a quad as two triangles (the first in the top left half, the second in the bottom right).
	Memory quad = upload_alloc(4 * 32);
	Vertex* qv = (Vertex*)quad.cpu;
	qv[0] = { { -1, -1, 0.5, 1 }, { 1, 0, 0, 1 } };
	qv[1] = { { 1, -1, 0.5, 1 }, { 1, 0, 0, 1 } };
	qv[2] = { { -1, 1, 0.5, 1 }, { 1, 0, 0, 1 } };
	qv[3] = { { 1, 1, 0.5, 1 }, { 1, 0, 0, 1 } };
	Memory indices = upload_alloc(12);
	ushort[6] index_data = { 0, 1, 2, 2, 1, 3 };
	mem::copy(indices.cpu, &index_data, 12);
	Memory quad_root = upload_alloc(16);
	((VRoot*)quad_root.cpu).vertices = (Vertex*)quad.gpu;

	// -- Indexed draws: first_index selects the second triangle only.
	Memory out = readback_alloc(W * W * 4);
	CommandBuffer* c = begin();
	rt_pass(c, target_view, true);
	c.bind_pso(color_pso);
	c.draw_indexed(quad_root.gpu, { indices.gpu, 12 }, UINT16, 3, 1, 3);
	c.end_render_pass();
	c.barrier({ .color_output = true }, { .color_write = true }, { .transfer = true }, { .transfer_read = true });
	c.copy_texture_to_memory(target, { out.gpu, W * W * 4 });
	finish1(c);
	check(near(pixel(out, W, W - 4, W - 4), 255, 0, 0) && near(pixel(out, W, 4, 4), 0, 0, 255), "draw_indexed: first_index 3 draws the bottom right triangle only");

	// -- Indirect draws. The arguments live in GPU memory, as if a shader had written them.
	Memory args = upload_alloc(64);
	*(DrawIndexedArgs*)args.cpu = { .index_count = 3, .instance_count = 1, .index_start = 0 };
	*(DrawArgs*)(args.cpu + 32) = { .vertex_count = 3, .instance_count = 1, .vertex_start = 1 };
	out = readback_alloc(W * W * 4);
	c = begin();
	rt_pass(c, target_view, true);
	c.bind_pso(color_pso);
	c.draw_indexed_indirect(quad_root.gpu, { indices.gpu, 12 }, UINT16, { args.gpu, 20 });
	c.end_render_pass();
	c.barrier({ .color_output = true }, { .color_write = true }, { .transfer = true }, { .transfer_read = true });
	c.copy_texture_to_memory(target, { out.gpu, W * W * 4 });
	finish1(c);
	check(near(pixel(out, W, 4, 4), 255, 0, 0) && near(pixel(out, W, W - 4, W - 4), 0, 0, 255), "draw_indexed_indirect: the first triangle from the arguments");
	// vertex_start 1 draws vertices 1, 2, 3: the bottom right triangle.
	out = readback_alloc(W * W * 4);
	c = begin();
	rt_pass(c, target_view, true);
	c.bind_pso(color_pso);
	c.draw_indirect(quad_root.gpu, { args.gpu + 32, 16 });
	c.end_render_pass();
	c.barrier({ .color_output = true }, { .color_write = true }, { .transfer = true }, { .transfer_read = true });
	c.copy_texture_to_memory(target, { out.gpu, W * W * 4 });
	finish1(c);
	check(near(pixel(out, W, W - 4, W - 4), 255, 0, 0) && near(pixel(out, W, 4, 4), 0, 0, 255), "draw_indirect: vertex_start from the arguments");

	// -- Indirect dispatch: two groups of 64 threads write 1..128, nothing else.
	Memory groups = upload_alloc(16);
	uint[3] group_data = { 2, 1, 1 };
	mem::copy(groups.cpu, &group_data, 12);
	Memory iota_root = upload_alloc(16);
	((IotaRoot*)iota_root.cpu).output = (uint*)gpu_only.gpu;
	Memory iota_out = readback_alloc(1024);
	c = begin();
	c.bind_pso(iota_pso);
	c.dispatch_indirect(iota_root.gpu, { groups.gpu, 12 });
	c.barrier({ .compute = true }, { .shader_write = true }, { .transfer = true }, { .transfer_read = true });
	c.copy_memory({ gpu_only.gpu, 132 * 4 }, { iota_out.gpu, 132 * 4 });
	finish1(c);
	uint* iota = (uint*)iota_out.cpu;
	bool iota_ok = true;
	for (uint i = 0; i < 128; i++) iota_ok &= iota[i] == i + 1;
	check(iota_ok && iota[128] == 0 && iota[131] == 0, "dispatch_indirect: 2 groups of 64 threads, and no thread outside them ran");

	// -- Depth test: a far triangle drawn after the near one must not cover it.
	Memory tri = upload_alloc(6 * 32);
	Vertex* tv = (Vertex*)tri.cpu;
	for (uint i = 0; i < 3; i++)
	{
		float px = i == 1 ? 3.0f : -1.0f;
		float py = i == 2 ? 3.0f : -1.0f;
		tv[i] = { { px, py, 0.3f, 1 }, { 0, 1, 0, 1 } };         // near, green
		tv[3 + i] = { { px, py, 0.8f, 1 }, { 1, 0, 0, 1 } };      // far, red
	}
	Memory near_root = upload_alloc(16);
	((VRoot*)near_root.cpu).vertices = (Vertex*)tri.gpu;
	Memory far_root = upload_alloc(16);
	((VRoot*)far_root.cpu).vertices = (Vertex*)(tri.gpu + 3 * 32);
	out = readback_alloc(W * W * 4);
	c = begin();
	c.begin_render_pass({
		.colors = { { .render_view = target_view, .load = CLEAR, .clear = { 0, 0, 1, 1 } } },
		.depth = { .render_view = depth_view, .load = CLEAR, .store = DISCARD, .clear = 1.0f },
	});
	c.bind_pso(color_pso);
	c.set_depth_stencil({ .depth_test = true, .depth_write = true });
	c.draw(near_root.gpu, 3);
	c.draw(far_root.gpu, 3);
	c.end_render_pass();
	c.barrier({ .color_output = true }, { .color_write = true }, { .transfer = true }, { .transfer_read = true });
	c.copy_texture_to_memory(target, { out.gpu, W * W * 4 });
	finish1(c);
	check(near(pixel(out, W, 32, 32), 0, 255, 0), "depth test: the near green triangle stays visible when the far red one is drawn later");

	// -- Blending: red at 50% alpha over blue.
	Memory half = upload_alloc(3 * 32);
	Vertex* hv = (Vertex*)half.cpu;
	for (uint i = 0; i < 3; i++) hv[i] = { { i == 1 ? 3.0f : -1.0f, i == 2 ? 3.0f : -1.0f, 0.5f, 1 }, { 1, 0, 0, 0.5f } };
	Memory half_root = upload_alloc(16);
	((VRoot*)half_root.cpu).vertices = (Vertex*)half.gpu;
	out = readback_alloc(W * W * 4);
	c = begin();
	rt_pass(c, target_view, true);
	c.bind_pso(blend_pso);
	c.draw(half_root.gpu, 3);
	c.end_render_pass();
	c.barrier({ .color_output = true }, { .color_write = true }, { .transfer = true }, { .transfer_read = true });
	c.copy_texture_to_memory(target, { out.gpu, W * W * 4 });
	finish1(c);
	check(near(pixel(out, W, 32, 32), 128, 0, 128, 2), "blending: 50% red over blue is purple");

	// -- Default blend factors: enabled and nothing else is ONE, ZERO and ADD, the source color as it is.
	out = readback_alloc(W * W * 4);
	c = begin();
	rt_pass(c, target_view, true);
	c.bind_pso(default_blend_pso);
	c.draw(half_root.gpu, 3);
	c.end_render_pass();
	c.barrier({ .color_output = true }, { .color_write = true }, { .transfer = true }, { .transfer_read = true });
	c.copy_texture_to_memory(target, { out.gpu, W * W * 4 });
	finish1(c);
	check(near(pixel(out, W, 32, 32), 255, 0, 0), "blending: the default factors pass the source color through, not black");

	// -- Default stencil state: the first draw replaces the stencil with 1 where it covers (compare ALWAYS, all mask bits), the second draws
	//    only where the stencil equals 1.
	Memory left = upload_alloc(6 * 32);
	Vertex* lv = (Vertex*)left.cpu;
	lv[0] = { { -1, -1, 0.5f, 1 }, { 1, 0, 0, 1 } };
	lv[1] = { { 0, -1, 0.5f, 1 }, { 1, 0, 0, 1 } };
	lv[2] = { { -1, 3, 0.5f, 1 }, { 1, 0, 0, 1 } };
	for (uint i = 0; i < 3; i++) lv[3 + i] = { { i == 1 ? 3.0f : -1.0f, i == 2 ? 3.0f : -1.0f, 0.5f, 1 }, { 0, 1, 0, 1 } };
	Memory left_root = upload_alloc(16);
	((VRoot*)left_root.cpu).vertices = (Vertex*)left.gpu;
	Memory all_root = upload_alloc(16);
	((VRoot*)all_root.cpu).vertices = (Vertex*)(left.gpu + 3 * 32);
	out = readback_alloc(W * W * 4);
	c = begin();
	c.begin_render_pass({
		.colors = { { .render_view = target_view, .load = CLEAR, .clear = { 0, 0, 1, 1 } } },
		.depth = { .render_view = stencil_view, .load = CLEAR, .store = DISCARD, .clear = 1.0f },
		.stencil = { .render_view = stencil_view, .load = CLEAR, .store = DISCARD, .clear = 0 },
	});
	c.bind_pso(stencil_pso);
	c.set_depth_stencil({ .stencil_test = true, .front = { .pass = REPLACE, .reference = 1 }, .back = { .pass = REPLACE, .reference = 1 } });
	c.draw(left_root.gpu, 3);
	c.set_depth_stencil({ .stencil_test = true, .front = { .compare = EQUAL, .reference = 1 }, .back = { .compare = EQUAL, .reference = 1 } });
	c.draw(all_root.gpu, 3);
	c.end_render_pass();
	c.barrier({ .color_output = true }, { .color_write = true }, { .transfer = true }, { .transfer_read = true });
	c.copy_texture_to_memory(target, { out.gpu, W * W * 4 });
	finish1(c);
	check(near(pixel(out, W, 8, 32), 0, 255, 0) && near(pixel(out, W, W - 8, 32), 0, 0, 255), "stencil: the default compare and masks write where covered, EQUAL then limits the second draw to it");

	// -- Default texture: no format and no usage is RGBA8_UNORM that can be sampled.
	SizeAlign explicit_size = device.get_texture_size_align({ .extent = { 16, 16, 1 }, .format = RGBA8_UNORM, .usage = { .sampled = true } })!!;
	SizeAlign default_size = device.get_texture_size_align({ .extent = { 16, 16, 1 } })!!;
	check(explicit_size.size == default_size.size && explicit_size.align == default_size.align, "texture: a default description is a sampled RGBA8_UNORM");
	// -- Texture upload with a row pitch, a partial copy, and reading it back tightly packed.
	Memory src = upload_alloc(32 * 256);
	for (uint y = 0; y < 32; y++)
	{
		for (uint x = 0; x < 32; x++) ((Pixel*)(src.cpu + y * 256))[x] = { (char)(x * 8), (char)(y * 8), 200, 255 };
	}
	Memory region = upload_alloc(8 * 8 * 4);
	for (uint i = 0; i < 64; i++) ((Pixel*)region.cpu)[i] = { 1, 2, 3, 4 };
	Memory back = readback_alloc(32 * 32 * 4);
	c = begin();
	c.copy_memory_to_texture({ src.gpu, 32 * 256 }, upload_texture, { .row_pitch_bytes = 256 });
	c.barrier({ .transfer = true }, { .transfer_write = true }, { .transfer = true }, { .transfer_read = true });
	c.copy_memory_to_texture({ region.gpu, 256 }, upload_texture, { .offset = { 4, 4, 0 }, .extent = { 8, 8, 1 } });
	c.barrier({ .transfer = true }, { .transfer_write = true }, { .transfer = true }, { .transfer_read = true });
	c.copy_texture_to_memory(upload_texture, { back.gpu, 32 * 32 * 4 });
	finish1(c);
	check(pixel(back, 32, 20, 10) == (Pixel){ 160, 80, 200, 255 } && pixel(back, 32, 31, 31) == (Pixel){ 248, 248, 200, 255 }, "copy_memory_to_texture: rows with a 256 byte pitch land on the right texels");
	check(pixel(back, 32, 4, 4) == (Pixel){ 1, 2, 3, 4 } && pixel(back, 32, 11, 11) == (Pixel){ 1, 2, 3, 4 } && pixel(back, 32, 12, 12) == (Pixel){ 96, 96, 200, 255 }, "copy_memory_to_texture: an 8 x 8 region at (4,4) overwrote only itself");

	// -- Descriptor heaps: descriptors copied between heaps (and inside one, overlapping) resolve to the same textures and samplers.
	TextureDescriptorHeap* heap_a = device.create_texture_descriptor_heap(16)!!;
	TextureDescriptorHeap* heap_b = device.create_texture_descriptor_heap(16)!!;
	SamplerDescriptorHeap* samplers_a = device.create_sampler_descriptor_heap(4)!!;
	SamplerDescriptorHeap* samplers_b = device.create_sampler_descriptor_heap(8)!!;
	defer { heap_a.destroy(); heap_b.destroy(); samplers_a.destroy(); samplers_b.destroy(); }
	heap_a.write(5, upload_texture, SAMPLED);
	heap_a.write(6, target, SAMPLED);
	heap_b.copy_from(2, heap_a, 5, 2);             // 5, 6 -> 2, 3
	heap_b.copy_from(3, heap_b, 2, 2);             // overlapping: 2, 3 -> 3, 4, so slot 3 is the texture from slot 5 and 4 the one from 6
	samplers_a.write(0, {});
	samplers_a.write(1, { .min_filter = NEAREST, .mag_filter = NEAREST, .address_u = CLAMP_TO_EDGE, .address_v = CLAMP_TO_EDGE });
	samplers_b.copy_from(3, samplers_a, 0, 2);
	samplers_b.write(0, {});
	Memory sroot = upload_alloc(16);
	*(SampleRoot*)sroot.cpu = { .texture_index = 3, .sampler_index = 4 };   // the upload texture, with the nearest sampler
	out = readback_alloc(W * W * 4);
	c = begin();
	c.set_texture_descriptor_heap(heap_b);
	c.set_sampler_descriptor_heap(samplers_b);
	rt_pass(c, target_view, true);
	c.bind_pso(quad_pso);
	c.draw(sroot.gpu, 6);
	c.end_render_pass();
	c.barrier({ .color_output = true }, { .color_write = true }, { .transfer = true }, { .transfer_read = true });
	c.copy_texture_to_memory(target, { out.gpu, W * W * 4 });
	finish1(c);
	check(near(pixel(out, W, 40, 20), 160, 80, 200, 4), "descriptor copies: the sampled texture is the copied one, through the copied sampler");

	// -- A view of one level of a mip chain: base_mip and mip_count select what the shader sees. Level 1 was cleared to green.
	out = readback_alloc(W * W * 4);
	Memory mip_root = upload_alloc(16);
	*(SampleRoot*)mip_root.cpu = { .texture_index = 8, .sampler_index = 0 };
	heap_b.write(8, mips, SAMPLED, { .base_mip = 1, .mip_count = 1 });
	c = begin();
	foreach (i, view : mip_views)
	{
		c.begin_render_pass({ .colors = { { .render_view = view, .load = CLEAR, .clear = { i == 0 ? 1.0f : 0.0f, i == 1 ? 1.0f : 0.0f, i == 2 ? 1.0f : 0.0f, 1 } } } });
		c.end_render_pass();
	}
	c.barrier({ .color_output = true }, { .color_write = true }, { .fragment = true }, { .shader_read = true });
	c.set_texture_descriptor_heap(heap_b);
	c.set_sampler_descriptor_heap(samplers_b);
	rt_pass(c, target_view, true);
	c.bind_pso(quad_pso);
	c.draw(mip_root.gpu, 6);
	c.end_render_pass();
	c.barrier({ .color_output = true }, { .color_write = true }, { .transfer = true }, { .transfer_read = true });
	c.copy_texture_to_memory(target, { out.gpu, W * W * 4 });
	finish1(c);
	check(near(pixel(out, W, 32, 32), 0, 255, 0), "descriptor view with base_mip 1, mip_count 1 shows level 1 of the chain");

	// -- Suspended and resumed render passes, one after the other: the second segment keeps what the first drew.
	out = readback_alloc(W * W * 4);
	Memory green = upload_alloc(4 * 32);
	Vertex* gv = (Vertex*)green.cpu;
	for (uint i = 0; i < 4; i++) { gv[i] = qv[i]; gv[i].color = { 0, 1, 0, 1 }; }
	Memory green_root = upload_alloc(16);
	((VRoot*)green_root.cpu).vertices = (Vertex*)green.gpu;
	c = begin();
	ColorAttachment attachment = { .render_view = target_view, .load = CLEAR, .clear = { 0, 0, 1, 1 } };
	c.begin_render_pass({ .colors = { attachment } }, { .suspending = true });
	c.bind_pso(color_pso);
	c.draw_indexed(quad_root.gpu, { indices.gpu, 12 }, UINT16, 3);
	c.end_render_pass();
	c.begin_render_pass({ .colors = { attachment } }, { .resuming = true });
	c.bind_pso(color_pso);
	c.draw_indexed(green_root.gpu, { indices.gpu, 12 }, UINT16, 3, 1, 3);
	c.end_render_pass();
	c.barrier({ .color_output = true }, { .color_write = true }, { .transfer = true }, { .transfer_read = true });
	c.copy_texture_to_memory(target, { out.gpu, W * W * 4 });
	finish1(c);
	check(near(pixel(out, W, 4, 4), 255, 0, 0) && near(pixel(out, W, W - 4, W - 4), 0, 255, 0), "suspend / resume: both segments of the pass are in the result");

	// -- Aliased textures share memory: each one is initialized before it is used, and contents are discarded.
	Memory alias_out_a = readback_alloc(W * W * 4);
	Memory alias_out_b = readback_alloc(W * W * 4);
	c = begin();
	c.initialize_texture(alias_a);
	c.begin_render_pass({ .colors = { { .render_view = alias_a_view, .load = CLEAR, .clear = { 0, 1, 0, 1 } } } });
	c.end_render_pass();
	c.barrier({ .color_output = true }, { .color_write = true }, { .transfer = true }, { .transfer_read = true });
	c.copy_texture_to_memory(alias_a, { alias_out_a.gpu, W * W * 4 });
	c.barrier({ .transfer = true }, { .transfer_read = true }, { .color_output = true }, { .color_write = true });
	c.initialize_texture(alias_b);
	c.begin_render_pass({ .colors = { { .render_view = alias_b_view, .load = CLEAR, .clear = { 1, 0, 1, 1 } } } });
	c.end_render_pass();
	c.barrier({ .color_output = true }, { .color_write = true }, { .transfer = true }, { .transfer_read = true });
	c.copy_texture_to_memory(alias_b, { alias_out_b.gpu, W * W * 4 });
	finish1(c);
	check(near(pixel(alias_out_a, W, 10, 10), 0, 255, 0) && near(pixel(alias_out_b, W, 10, 10), 255, 0, 255), "aliased textures: each shows what was drawn after its own initialization");

	// -- Mesh shaders: one meshlet group draws a triangle over the whole target.
	Pso* mesh_pso = device.create_mesh_pso({ .mesh = stage("mesh_main", { 32, 1, 1 }), .fragment = stage("fs"), .color_targets = { { .format = RGBA8_UNORM } } })!!;
	defer mesh_pso.destroy();
	out = readback_alloc(W * W * 4);
	c = begin();
	rt_pass(c, target_view, true);
	c.bind_pso(mesh_pso);
	c.draw_meshlets(null, { 1, 1, 1 });
	c.end_render_pass();
	c.barrier({ .color_output = true }, { .color_write = true }, { .transfer = true }, { .transfer_read = true });
	c.copy_texture_to_memory(target, { out.gpu, W * W * 4 });
	finish1(c);
	check(near(pixel(out, W, 32, 32), 255, 0, 255), "draw_meshlets: the mesh shader's triangle (the fragment shader returns its vertex color)");

	// -- A compute queue and a copy queue, ordered with timeline semaphores: compute doubles floats, then the copy queue copies them out.
	Device* multi = nogfx::create_device({ .desired_compute_queue_count = 1, .desired_copy_queue_count = 1 })!!;
	defer multi.destroy();
	check(multi.caps.queue_count == 3 && multi.caps.general_queue_count == 1 && multi.caps.compute_queue_count == 1 && multi.caps.copy_queue_count == 1, "queues: one general, one compute and one copy queue");
	GpuHeap shared = multi.create_gpu_heap(65536, CPU_VISIBLE)!!;
	GpuHeap results = multi.create_gpu_heap(65536, READBACK)!!;
	defer { shared.destroy(); results.destroy(); }
	float* data = (float*)shared.cpu;
	for (uint i = 0; i < 100; i++) data[i] = (float)i;
	((DoubleRoot*)(shared.cpu + 1024)).data = (float*)shared.gpu;
	((DoubleRoot*)(shared.cpu + 1024)).count = 100;
	Pso* multi_twice = multi.create_compute_pso(stage("twice", { 64, 1, 1 }))!!;
	CommandPool* compute_pool = multi.create_command_pool(1)!!;
	CommandPool* copy_pool = multi.create_command_pool(2)!!;
	TimelineSemaphore* compute_done = multi.create_timeline_semaphore()!!;
	TimelineSemaphore* copy_done = multi.create_timeline_semaphore()!!;
	defer { multi_twice.destroy(); compute_pool.destroy(); copy_pool.destroy(); compute_done.destroy(); copy_done.destroy(); }
	// Submit the copy first: it can only run once the compute queue has signaled, and must not run before.
	CommandBuffer* copy_commands = copy_pool.begin_commands()!!;
	copy_commands.copy_memory({ shared.gpu, 400 }, { results.gpu, 400 });
	copy_commands.end();
	multi.submit({ .commands = { copy_commands }, .waits = { { compute_done, 1 } }, .completion = { copy_done, 1 } }, 2);
	check(copy_done.completed_value() == 0, "queues: the copy queue waits for the compute queue's signal");
	CommandBuffer* compute_commands = compute_pool.begin_commands()!!;
	compute_commands.bind_pso(multi_twice);
	compute_commands.dispatch(shared.gpu + 1024, { 2, 1, 1 });
	compute_commands.end();
	multi.submit({ .commands = { compute_commands }, .completion = { compute_done, 1 } }, 1);
	TimelinePoint copied = { copy_done, 1 };
	copied.wait();
	float* doubled = (float*)results.cpu;
	bool doubled_ok = true;
	for (uint i = 0; i < 100; i++) doubled_ok &= doubled[i] == (float)i * 2.0f;
	check(doubled_ok, "queues: the copy shows the compute queue's result");
	multi.wait_idle();

	// -- Errors: a shader that does not compile is reported through the callback and a fault.
	ShaderStage broken = { .code = "this is not a shader", .entry_point = "main" };
	bool compile_failed;
	if (catch err = device.create_graphics_pso({ .vertex = broken, .color_targets = { { .format = RGBA8_UNORM } } })) compile_failed = err == nogfx::CREATION_FAILED;
	check(compile_failed, "errors: a shader that does not compile gives CREATION_FAILED and a message");

	device.wait_idle();
	io::printfn(failures ? "%d check(s) FAILED" : "all checks passed", failures);
	return failures ? 1 : 0;
}
