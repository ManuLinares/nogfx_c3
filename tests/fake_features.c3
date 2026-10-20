module fake_features;
import std::io, nogfx;

// Runs the Vulkan backend against the stub driver of tests/fake_vulkan: several queues, every kind of pipeline, aliasing, mip views, depth /
// stencil state, a render pass split over two command buffers, mesh and indirect draws, descriptor copies. The stub checks the calls and the
// order of them. The shaders are only stand-ins (a vertex shader serves as a mesh shader), so a real GPU would reject them: this refuses to run
// on anything else.

const char[*] VERTEX = $embed("vulkan/triangle.vert.spv");
const char[*] FRAGMENT = $embed("vulkan/triangle.frag.spv");
const char[*] SCALE = $embed("vulkan/scale.comp.spv");

int failures;
macro check(bool condition, String what)
{
	io::printfn("%s  %s", condition ? "pass" : "FAIL", what);
	if (!condition) failures++;
}

fn int main()
{
	nogfx::set_error_callback(fn (message, fatal) { io::printfn("callback: %s", message); });
	// Several queues of every kind; the stub has 2 general, 1 compute only and 1 copy only.
	Device* device = nogfx::create_device({ .desired_queue_count = 5, .desired_compute_queue_count = 1, .desired_copy_queue_count = 1, .timestamp_query_count = 4 })!!;
	defer device.destroy();
	if (device.caps.device_name != "Fake Vulkan GPU")
	{
		io::printn("skipped: this only runs against the stub driver, see tests/run_fake.sh");
		return 0;
	}
	check(device.caps.queue_count == 4 && device.caps.general_queue_count == 2 && device.caps.compute_queue_count == 1 && device.caps.copy_queue_count == 1, "queues: capped to what the families offer, general first");
	check(device.caps.copy_texture_granularity.x == 16, "caps: granularity is a copy queue property");

	// Unaligned SPIR-V (a copy at an odd address) must be accepted: the backend copies it.
	char[] odd = mem::new_array(char, SCALE.len + 1);
	mem::copy(&odd[1], &SCALE[0], SCALE.len);
	Pso* unaligned = device.create_compute_pso({ .code = odd[1..] })!!;
	unaligned.destroy();
	mem::free(odd.ptr);

	Pso* compute = device.create_compute_pso({ .code = SCALE[..], .entry_point = "main" })!!;
	Pso* graphics = device.create_graphics_pso({
		.vertex = { .code = VERTEX[..] }, .fragment = { .code = FRAGMENT[..] },
		.color_targets = { { .format = RGBA8_UNORM, .blend = { .enabled = true, .color = { .source = SOURCE_ALPHA, .destination = ONE_MINUS_SOURCE_ALPHA }, .alpha = { .source = ONE, .destination = ZERO } } }, { .format = RGBA16_FLOAT, .write_mask = 0b0111 } },
		.depth_format = D32_FLOAT,
		.rasterization = { .cull = COUNTER_CLOCKWISE, .depth_bias_constant = 1.0f, .depth_bias_slope = 2.0f },
	})!!;
	Pso* depth_only = device.create_graphics_pso({ .vertex = { .code = VERTEX[..] }, .depth_format = D24_UNORM_S8_UINT, .stencil_format = D24_UNORM_S8_UINT })!!;
	Pso* mesh = device.create_mesh_pso({ .task = { .code = VERTEX[..] }, .mesh = { .code = VERTEX[..] }, .fragment = { .code = FRAGMENT[..] }, .color_targets = { { .format = RGBA8_UNORM } } })!!;
	Pso* mesh_only = device.create_mesh_pso({ .mesh = { .code = VERTEX[..] }, .color_targets = { { .format = RGBA8_UNORM } } })!!;
	defer { compute.destroy(); graphics.destroy(); depth_only.destroy(); mesh.destroy(); mesh_only.destroy(); }
	check(true, "pipelines: compute, graphics with blend / two targets / depth / bias, depth only, mesh with and without a task stage");

	GpuHeap heap = device.create_gpu_heap(1 << 16)!!;
	defer heap.destroy();
	TimelineSemaphore* timeline = device.create_timeline_semaphore(5)!!;
	defer timeline.destroy();
	check(timeline.completed_value() == 5, "timeline: the initial value");

	// One command pool per queue family, work on each queue, ordered through the timeline.
	CommandPool*[4] pools;
	for (uint q = 0; q < 4; q++) pools[q] = device.create_command_pool(q)!!;
	defer for (uint q = 0; q < 4; q++) pools[q].destroy();
	ulong value = 5;
	ulong[4] stamps;
	for (uint q = 0; q < 4; q++)
	{
		CommandBuffer* commands = pools[q].begin_commands()!!;
		commands.copy_memory({ heap.gpu, 256 }, { heap.gpu + 1024, 256 });
		commands.barrier({ .transfer = true }, { .transfer_write = true }, { .all_commands = true }, { .shader_read = true, .descriptor_read = true });
		commands.write_timestamp(&stamps[q]);
		commands.end();
		device.submit({ .commands = { commands }, .waits = { { timeline, value } }, .completion = { timeline, value + 1 } }, q);
		value++;
	}
	((TimelinePoint){ timeline, value }).wait();
	check(timeline.completed_value() == 9, "queues: four submissions chained through one timeline");
	device.wait_idle();
	for (uint q = 0; q < 4; q++) { pools[q].read_timestamps(); pools[q].reset(); }

	// A second set of command buffers: aliasing, mip views, depth / stencil state, suspend / resume, mesh draws, descriptor copies.
	TextureDesc color_desc = { .extent = { 32, 32, 1 }, .mip_levels = 3, .format = RGBA8_UNORM, .mutable_format = true, .usage = { .sampled = true, .color_attachment = true, .storage = true } };
	TextureDesc depth_desc = { .extent = { 32, 32, 1 }, .format = D32_FLOAT, .usage = { .depth_stencil_attachment = true, .sampled = true } };
	TextureDesc cube_desc = { .type = CUBE, .extent = { 16, 16, 1 }, .layer_count = 6, .format = RGBA8_UNORM, .usage = { .sampled = true } };
	check(device.supports_texture_format(RGBA8_UNORM, { .sampled = true, .color_attachment = true }) && !device.supports_texture_format(D32_FLOAT, { .color_attachment = true }), "supports_texture_format: color and depth rules");
	check(!device.supports_texture_format(D32_FLOAT_S8_UINT, { .transfer_source = true }), "supports_texture_format: combined depth stencil cannot be a transfer source");
	ulong total;
	TextureDesc[3] all_descs = { color_desc, depth_desc, cube_desc };
	foreach (desc : all_descs) total += device.get_texture_size_align(desc)!!.size + 65536;
	TextureHeap texture_heap = device.create_texture_heap(total)!!;
	defer texture_heap.destroy();
	CommandBuffer* commands = pools[0].begin_commands()!!;
	Texture* color = device.create_texture(color_desc, texture_heap, 0)!!;
	Texture* aliased = device.create_texture(color_desc, texture_heap, 0)!!;
	Texture* depth = device.create_texture(depth_desc, texture_heap, 1 << 20)!!;
	Texture* cube = device.create_texture(cube_desc, texture_heap, 2 << 20)!!;
	commands.initialize_texture(color);
	commands.initialize_texture(aliased);
	commands.initialize_texture(depth);
	commands.initialize_texture(cube);
	RenderView* color_view = color.create_render_view({ .mip_level = 1 })!!;
	RenderView* depth_view = depth.create_render_view()!!;
	TextureDescriptorHeap* textures = device.create_texture_descriptor_heap(8)!!;
	SamplerDescriptorHeap* samplers = device.create_sampler_descriptor_heap(2)!!;
	textures.write(0, color, STORAGE, { .format = RGBA8_SRGB, .base_mip = 1, .mip_count = 1 });
	textures.write(1, cube, SAMPLED);
	textures.write(2, depth, SAMPLED, { .aspect = DEPTH });
	textures.copy_from(3, textures, 0, 2);
	textures.copy_from(1, textures, 0, 3);   // overlapping
	samplers.write(0, { .compare_enabled = true, .compare = ALWAYS });
	commands.set_texture_descriptor_heap(textures);
	commands.set_sampler_descriptor_heap(samplers);
	commands.begin_render_pass({ .colors = { { .render_view = color_view, .load = CLEAR, .store = DISCARD, .clear = { 1, 2, 3, 4 } } }, .depth = { .render_view = depth_view, .load = CLEAR, .clear = 1.0f } }, { .suspending = true });
	commands.end_render_pass();
	commands.end();
	device.submit({ .commands = { commands }, .completion = { timeline, 10 } });
	commands = pools[0].begin_commands()!!;
	commands.begin_render_pass({ .colors = { { .render_view = color_view } }, .depth = { .render_view = depth_view } }, { .resuming = true });
	commands.bind_pso(graphics);
	commands.set_depth_stencil({ .depth_test = true, .depth_write = true, .depth_compare = GREATER, .stencil_test = true, .front = { .compare = EQUAL, .pass = REPLACE, .reference = 3 } });
	commands.set_viewport({ 0, 0, 16, 16, 0, 1 });
	commands.draw(heap.gpu, 3, 2, 1, 1);
	commands.bind_pso(mesh);
	commands.draw_meshlets(heap.gpu, { 4, 1, 1 });
	commands.draw_meshlets_indirect(heap.gpu, { heap.gpu + 64, 24 }, 2);
	commands.draw_indexed_indirect(heap.gpu, { heap.gpu + 128, 64 }, UINT32, { heap.gpu + 256, 40 }, 2, 20);
	commands.end_render_pass();
	commands.bind_pso(compute);
	commands.dispatch(null, { 1, 1, 1 });
	commands.end();
	device.submit({ .commands = { commands }, .waits = { { timeline, 10 } }, .completion = { timeline, 11 } });
	((TimelinePoint){ timeline, 11 }).wait();
	device.wait_idle();
	check(timeline.completed_value() == 11, "render pass split over two command buffers, mesh draws, indirect draws, null root");

	textures.destroy();
	samplers.destroy();
	color_view.destroy();
	depth_view.destroy();
	cube.destroy();
	depth.destroy();
	aliased.destroy();
	color.destroy();
	check(@catch(device.set_swapchain_format(BGRA8_UNORM, SRGB)) == nogfx::UNSUPPORTED, "set_swapchain_format: unsupported without a window");
	check(device.drawable_extent().x == 0, "no window: no drawable extent");
	io::printfn(failures ? "%d check(s) FAILED" : "all checks passed", failures);
	return failures ? 1 : 0;
}
