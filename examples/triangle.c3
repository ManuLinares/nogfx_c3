module triangle;
import std::gfx, std::io, std::time, std::math, nogfx;

// The NoGraphicsAPI model in a std::gfx window on macOS. Each frame:
//  1. a compute shader writes an animated plasma into a storage texture,
//  2. a render pass draws a full screen quad that samples it, through a texture index and a sampler index,
//  3. and a blended triangle on top of it, whose vertices the CPU wrote into GPU visible memory and the shader finds through the root pointer.
// There are no bindings, no vertex buffers and no descriptor sets: a draw is a pipeline and a pointer. The window is resizable, F toggles
// fullscreen and ESC quits. The window's CAMetalLayer comes from GfxWindow.create_metal_layer, which also repaints during live resize.

const String SHADERS = `
#include <metal_stdlib>
using namespace metal;

// Shaders get the root pointer at buffer(0), the base of the texture descriptor heap at buffer(1) and the sampler heap at buffer(2).
template<typename T> T gpu_texture_from_id(ulong id) { return *reinterpret_cast<thread T*>(&id); }
struct GPUSampler { sampler value; };

struct PlasmaRoot { uint texture_index; uint size; float time; float pad; };
kernel void plasma(device const PlasmaRoot* root [[buffer(0)]], constant ulong& pool [[buffer(1)]], uint2 id [[thread_position_in_grid]])
{
	auto texture = gpu_texture_from_id<texture2d<float, access::write>>(pool + root->texture_index);
	float2 p = float2(id) / float(root->size) * 6.2831853;
	float t = root->time;
	float v = sin(p.x * 2.0 + t) + sin(p.y * 3.0 - t * 1.3) + sin((p.x + p.y) * 2.0 + t * 0.7) + sin(length(p - 3.14) * 3.0 - t);
	texture.write(float4(0.5 + 0.5 * sin(v), 0.5 + 0.5 * sin(v + 2.1), 0.5 + 0.5 * sin(v + 4.2), 1.0), id);
}

struct VOut { float4 position [[position]]; float4 color; };

struct QuadRoot { uint texture_index; uint sampler_index; float time; float aspect; };
constant float2 quad_corners[6] = { float2(-1, -1), float2(1, -1), float2(-1, 1), float2(-1, 1), float2(1, -1), float2(1, 1) };
vertex VOut quad_vertex(uint vid [[vertex_id]])
{
	float2 p = quad_corners[vid];
	return { float4(p, 0.0, 1.0), float4(p * 0.5 + 0.5, 0.0, 1.0) };
}
fragment float4 quad_fragment(VOut in [[stage_in]], device const QuadRoot* root [[buffer(0)]], constant ulong& pool [[buffer(1)]],
	device const GPUSampler* samplers [[buffer(2)]])
{
	auto texture = gpu_texture_from_id<texture2d<float>>(pool + root->texture_index);
	float2 uv = in.color.xy * float2(root->aspect, 1.0) * 1.5 + float2(root->time * 0.02, 0.0);
	return texture.sample(samplers[root->sampler_index].value, uv) * 0.55;
}

struct Vertex { float4 position; float4 color; };
struct TriangleRoot { device const Vertex* vertices; };
vertex VOut triangle_vertex(device const TriangleRoot* root [[buffer(0)]], uint vid [[vertex_id]])
{
	return { root->vertices[vid].position, root->vertices[vid].color };
}
fragment float4 triangle_fragment(VOut in [[stage_in]]) { return in.color; }
`;

// The root structs are plain C structs. Their layout is the layout the shaders see, so a pointer in a root is just a pointer.
struct PlasmaRoot
{
	uint texture_index;
	uint size;
	float time;
	float pad;
}

struct QuadRoot
{
	uint texture_index;
	uint sampler_index;
	float time;
	float aspect;
}

struct Vertex
{
	float[<4>] position;
	float[<4>] color;
}

struct TriangleRoot
{
	Vertex* vertices;
}

struct Stamps
{
	ulong begin;
	ulong end;
}

// What one frame in flight needs written: the CPU fills it while the GPU may still be reading the other frame's.
struct FrameData
{
	PlasmaRoot plasma @align(16);
	QuadRoot quad @align(16);
	TriangleRoot triangle @align(16);
	Vertex[3] vertices @align(16);
}

const uint PLASMA_SIZE = 256;
const uint FRAMES = 2;
const ulong FRAME_STRIDE = 1024;   // At least FrameData::size, and a multiple of 16.

Device* device;
GpuHeap upload;
CommandPool*[FRAMES] pools;
TimelineSemaphore* timeline;
ulong frame_count;
Pso* plasma_pso;
Pso* quad_pso;
Pso* triangle_pso;
TextureDescriptorHeap* textures;
SamplerDescriptorHeap* samplers;
Texture* plasma_texture;
double start_time;
Stamps[FRAMES] stamps;
double gpu_ms_sum;
uint gpu_ms_count;
double last_report;

fn ShaderStage stage(String entry, Uint3 size = {}) => { .code = SHADERS, .entry_point = entry.zstr_tcopy(), .threadgroup_size = size };

// Draws a frame. gfx calls this while the window is resized, from inside the transaction that resizes the layer, and the loop calls it to animate.
fn void draw(GfxWindow* win, GfxRect rect)
{
	// Frame n uses pool and upload region n % 2, so frame n - 2 must be done with them.
	ulong frame = frame_count + 1;
	if (frame > FRAMES)
	{
		TimelinePoint previous = { timeline, frame - FRAMES };
		previous.wait();
	}
	CommandPool* pool = pools[frame % FRAMES];
	if (frame > FRAMES) read_gpu_time(pool, (uint)(frame % FRAMES));
	pool.reset();
	CommandBuffer* commands = pool.begin_commands()!!;
	SwapchainFrame target = commands.acquire();
	if (!target.render_view) return;   // The window has no size.

	FrameData* data = (FrameData*)(upload.cpu + (frame % FRAMES) * FRAME_STRIDE);
	float time = (float)(time::now().to_seconds() - start_time);
	float aspect = (float)target.extent.x / (float)target.extent.y;
	data.plasma = { .texture_index = 0, .size = PLASMA_SIZE, .time = time };
	data.quad = { .texture_index = 1, .sampler_index = 0, .time = time, .aspect = aspect };
	float angle = time;
	for (uint i = 0; i < 3; i++)
	{
		float a = angle + (float)i * 2.0944f;
		data.vertices[i].position = { math::cos(a) * 0.6f / aspect, math::sin(a) * 0.6f, 0, 1 };
		data.vertices[i].color = { i == 0 ? 1.0f : 0.0f, i == 1 ? 1.0f : 0.0f, i == 2 ? 1.0f : 0.0f, 0.85f };
	}
	data.triangle.vertices = (Vertex*)gpu_address(&data.vertices);

	commands.write_timestamp(&stamps[frame % FRAMES].begin);

	// The compute shader writes the storage view (index 0) of the texture and the quad samples the sampled view (index 1).
	commands.set_texture_descriptor_heap(textures);
	commands.set_sampler_descriptor_heap(samplers);
	commands.bind_pso(plasma_pso);
	commands.dispatch(gpu_address(&data.plasma), { PLASMA_SIZE / 8, PLASMA_SIZE / 8, 1 });
	commands.barrier({ .compute = true }, { .shader_write = true }, { .fragment = true }, { .shader_read = true });

	commands.begin_render_pass({ .colors = { { .render_view = target.render_view, .load = CLEAR, .clear = { 0.02f, 0.02f, 0.04f, 1 } } } });
	commands.bind_pso(quad_pso);
	commands.draw(gpu_address(&data.quad), 6);
	commands.bind_pso(triangle_pso);
	commands.draw(gpu_address(&data.triangle), 3);
	commands.end_render_pass();
	commands.write_timestamp(&stamps[frame % FRAMES].end);
	commands.end();

	frame_count = frame;
	device.submit_and_present({ .commands = { commands }, .completion = { timeline, frame } });
}

// The GPU address of something in the upload heap, which is what a root pointer or a pointer inside a root has to be.
fn void* gpu_address(void* cpu) => upload.gpu + ((char*)cpu - upload.cpu);

fn void read_gpu_time(CommandPool* pool, uint index)
{
	pool.read_timestamps();
	if (stamps[index].end > stamps[index].begin)
	{
		gpu_ms_sum += (double)(stamps[index].end - stamps[index].begin) * device.caps.timestamp_period_ns / 1e6;
		gpu_ms_count++;
	}
	double now = time::now().to_seconds();
	if (now - last_report >= 1.0 && gpu_ms_count)
	{
		io::printfn("frame %d: %.2f ms GPU per frame", frame_count, gpu_ms_sum / gpu_ms_count);
		gpu_ms_sum = 0;
		gpu_ms_count = 0;
		last_report = now;
	}
}

fn int main()
{
	gfx::init("NoGraphicsAPI triangle", {})!!;
	defer gfx::finalize();
	GfxWindow* win = gfx::create_window("NoGraphicsAPI triangle", 200, 200, 900, 600, {})!!;
	defer win.destroy();

	// The only coupling between the window and the device is the CAMetalLayer. Three drawables rather than the default two: while the window is
	// resized gfx presents inside the Core Animation transaction, which holds a drawable until it commits, and with two of them nextDrawable
	// then blocks for a second at a time.
	device = nogfx::create_device({
		.window = win.create_metal_layer()!!,
		.swapchain_format = BGRA8_UNORM,
		.desired_swapchain_image_count = 3,
		.timestamp_query_count = 4,
	})!!;
	defer device.destroy();
	io::printfn("Using %s", device.caps.device_name);

	upload = device.create_gpu_heap(64 * 1024, CPU_VISIBLE)!!;
	defer upload.destroy();
	plasma_pso = device.create_compute_pso(stage("plasma", { 8, 8, 1 }))!!;
	quad_pso = device.create_graphics_pso({
		.vertex = stage("quad_vertex"),
		.fragment = stage("quad_fragment"),
		.color_targets = { { .format = BGRA8_UNORM } },
	})!!;
	triangle_pso = device.create_graphics_pso({
		.vertex = stage("triangle_vertex"),
		.fragment = stage("triangle_fragment"),
		.color_targets = { {
			.format = BGRA8_UNORM,
			.blend = { .enabled = true, .color = { SOURCE_ALPHA, ONE_MINUS_SOURCE_ALPHA, ADD }, .alpha = { ONE, ONE_MINUS_SOURCE_ALPHA, ADD } },
		} },
	})!!;
	defer { plasma_pso.destroy(); quad_pso.destroy(); triangle_pso.destroy(); }

	TextureDesc desc = { .extent = { PLASMA_SIZE, PLASMA_SIZE, 1 }, .format = RGBA8_UNORM, .usage = { .sampled = true, .storage = true } };
	SizeAlign size = device.get_texture_size_align(desc)!!;
	TextureHeap texture_heap = device.create_texture_heap(size.size)!!;
	defer texture_heap.destroy();
	textures = device.create_texture_descriptor_heap(8)!!;
	samplers = device.create_sampler_descriptor_heap(2)!!;
	defer { textures.destroy(); samplers.destroy(); }
	samplers.write(0, {});   // linear, repeat

	foreach (&pool : pools) *pool = device.create_command_pool()!!;
	defer foreach (pool : pools) pool.destroy();
	timeline = device.create_timeline_semaphore()!!;
	defer timeline.destroy();

	// The placed texture is initialized by a command buffer, which is submitted before any frame uses the texture.
	CommandPool* setup_pool = device.create_command_pool()!!;
	defer setup_pool.destroy();
	CommandBuffer* setup = setup_pool.begin_commands()!!;
	plasma_texture = device.create_texture(desc, texture_heap, 0)!!;
	defer plasma_texture.destroy();
	setup.initialize_texture(plasma_texture);
	setup.end();
	TimelineSemaphore* setup_done = device.create_timeline_semaphore()!!;
	defer setup_done.destroy();
	device.submit({ .commands = { setup }, .completion = { setup_done, 1 } });
	TimelinePoint setup_point = { setup_done, 1 };
	setup_point.wait();
	textures.write(0, plasma_texture, STORAGE);
	textures.write(1, plasma_texture, SAMPLED);

	start_time = time::now().to_seconds();
	last_report = time::now().to_seconds();
	win.set_repaint_callback(&draw);
	bool quit = false;
	while (!quit)
	{
		if (try e = gfx::wait(2 * time::MS))
		{
			switch (e.type)
			{
				case WINDOW_CLOSE: quit = true;
				case KEY_PRESSED:
					if (e.key.value == ESCAPE) quit = true;
					if (e.key.value == F) win.set_fullscreen(!win.is_fullscreen());
				default: break;
			}
		}
		draw(win, {});
	}
	device.wait_idle();
	return 0;
}
