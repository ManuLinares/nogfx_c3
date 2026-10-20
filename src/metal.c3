module nogfx @feat(DARWIN & !NOGFX_VULKAN) @link("Foundation.framework", "QuartzCore.framework");
import nogfx::mtl, std::io, std::thread, std::atomic, std::bits, std::collections, std::os::darwin, libc;

// The Metal 4 backend, a function by function port of src/NoGraphicsAPIMetal.mm in NoGraphicsAPI (MIT, Sebastian Aaltonen).
//
// Metal 4 needs an Apple7 GPU (M1 / A14) and macOS 26. The API is not thread safe except where upstream says so: queues and command pools
// are externally synchronized, and resource creation, lookup and timeline waits can run concurrently. The calls follow the C++ API with
// the C3 differences listed in nogfx.c3: the functions are methods on the object they work on, a creation that can fail returns an
// optional, and the Objective-C autorelease pools are the ones the C++ functions open themselves with @autoreleasepool.
//
// Shaders get their data through three argument table slots: buffer(0) is the root pointer passed to draw / dispatch, buffer(1) is the
// base resource id of the bound texture descriptor heap (a texture index is added to it) and buffer(2) is the bound sampler heap.

// ---- Objects --------------------------------------------------------------------------------------------------------------------

struct GpuHeapOwner
{
	Device* device;
	ObjId heap;
	ObjId buffer;
	BufferRecord* record;
}

struct TextureHeapOwner
{
	Device* device;
	ObjId heap;
}

struct Texture
{
	Device* device;
	ObjId texture;
	TextureDesc desc;
}

struct RenderView
{
	ObjId texture;
	uint mip;
	uint slice;
}

struct TextureDescriptorHeap
{
	Device* device;
	ObjId pool;
	ObjId base;
	uint capacity;
}

struct SamplerDescriptorHeap
{
	Device* device;
	ObjId buffer;
	ObjId* samplers;
	uint capacity;
}

struct TimelineSemaphore
{
	Device* device;
	ObjId event;
}

struct Pso
{
	Device* device;
	ObjId render;
	ObjId compute;
	RasterizationState rasterization;
	mtl::Size threads;
	mtl::Size object_threads;
	bool mesh;
}

struct CounterPage
{
	ObjId heap;
	CounterPage* next;
	ulong[64] occupied;
}

struct TimestampSlot
{
	ObjId heap;
	CounterPage* page;
	ulong* destination;
	uint index;
}

struct NativeCommandBuffer
{
	ObjId buffer;
	ObjId allocator;
	NativeCommandBuffer* next;
}

struct CommandBuffer
{
	Device* device;
	CommandPool* pool;
	CommandBuffer* next;
	ObjId commands;
	NativeCommandBuffer* native_buffers;
	NativeCommandBuffer* native;
	uint native_count;
	ObjId arguments;
	ObjId compute;
	ObjId render;
	ObjId pass;
	TimestampSlot* timestamps;
	uint timestamp_count;
	ObjId retirement;
	ulong retirement_value;
	bool recording;
	bool ended;
	bool render_continuation;
	bool acquired;
	Pso* pso;
}

struct DepthStateEntry
{
	DepthStencilState desc;
	ObjId state;
}

enum QueueKind : char { GENERAL, COMPUTE, COPY }

struct Queue
{
	ObjId queue;
	void* feedback;
	ObjId completion;
	ulong submitted_value;
	List{ObjId} submission;   // The native command buffers of a submit, kept between submissions.
	QueueKind kind;
}

struct CommandPool
{
	Device* device;
	QueueKind kind;
	CommandBuffer* first;
	CommandBuffer* last;
	CommandBuffer* next_buffer;
	List{DepthStateEntry} depth_states;   // The ones its commands asked for: depth stencil states are immutable Metal objects.
}

struct Device
{
	ObjId metal;
	Queue* queues;
	ObjId compiler;
	ObjId residency;
	ObjId layer;
	ObjId presentation_queue;   // Only with DeviceDesc.steam_overlay_compatibility: a classic queue that presents.
	ObjId drawable;
	CommandBuffer* acquired;
	RenderView drawable_view;
	DeviceCaps caps;
	char[256] name;
	uint residency_lock;                // An os_unfair_lock.
	BufferAddressMap buffer_map;
	CounterPage* counter_pages;
	uint counter_page_count;
	uint timestamp_query_count;
	bool shader_validation;
}

// ---- Helpers --------------------------------------------------------------------------------------------------------------------

const NSUInteger RENDER_STAGES = mtl::RenderStages.VERTEX | mtl::RenderStages.FRAGMENT | mtl::RenderStages.OBJECT | mtl::RenderStages.MESH;

const ObjId NIL = (ObjId)null;

fn mtl::Size metal_size(Uint3 size) @local => { or_one(size.x), or_one(size.y), or_one(size.z) };

// Zero means "all channels" for the C3 ColorTargetDesc, Metal's mask has red in the high bit.
fn NSUInteger color_write_mask(char mask) @local
{
	if (!mask) return mtl::ColorWriteMask.ALL;
	return (NSUInteger)(((mask & 1) << 3) | ((mask & 2) << 1) | ((mask & 4) >> 1) | ((mask & 8) >> 3));
}

fn void report_error(String operation, ObjId error) @local
{
	ZString text = error ? error.send(ObjId, "localizedDescription").send(ZString, "UTF8String") : (ZString)"creation failed";
	io::eprintfn("NoGraphicsAPI Metal: %s: %s", operation, text);
	if (error_callback) error_callback(string::tformat("Metal %s: %s", operation, text), false);
}

// Reports the failure of an operation and returns the fault, for `return fail(...)~`.
fn fault fail(fault reason, String operation, ObjId error = NIL) @local
{
	report_error(operation, error);
	return reason;
}

fn NSUInteger barrier_stages(Stage stages, bool source) @local
{
	if (stages.all_commands) return mtl::Stages.ALL;
	NSUInteger raster = mtl::Stages.FRAGMENT | mtl::Stages.TILE;
	NSUInteger geometry = mtl::Stages.VERTEX | mtl::Stages.OBJECT | mtl::Stages.MESH;
	NSUInteger result = 0;
	// Preserve Vulkan's logical execution scopes without joining its alternative graphics/compute branches.
	if (stages.indirect) result |= geometry | mtl::Stages.DISPATCH | (source ? 0 : raster);
	if (stages.index_input || stages.vertex) result |= mtl::Stages.VERTEX | (source ? 0 : raster);
	if (stages.task) result |= mtl::Stages.OBJECT | (source ? 0 : mtl::Stages.MESH | raster);
	if (stages.mesh) result |= mtl::Stages.MESH | (source ? mtl::Stages.OBJECT : raster);
	if (stages.depth_stencil_tests || stages.fragment || stages.color_output) result |= raster | (source ? geometry : 0);
	if (stages.compute) result |= mtl::Stages.DISPATCH;
	if (stages.transfer) result |= mtl::Stages.BLIT;
	return result;
}

fn NSUInteger pixel_format(Format format) @local
{
	switch (format)
	{
		case R8_SRGB: return MTLPixelFormat.R8_UNORM_SRGB;
		case RG8_SRGB: return MTLPixelFormat.RG8_UNORM_SRGB;
		case RGBA8_SRGB: return MTLPixelFormat.RGBA8_UNORM_SRGB;
		case BGRA8_SRGB: return MTLPixelFormat.BGRA8_UNORM_SRGB;
		case R8_UNORM: return MTLPixelFormat.R8_UNORM;
		case RG8_UNORM: return MTLPixelFormat.RG8_UNORM;
		case RGBA8_UNORM: return MTLPixelFormat.RGBA8_UNORM;
		case BGRA8_UNORM: return MTLPixelFormat.BGRA8_UNORM;
		case R16_UNORM: return MTLPixelFormat.R16_UNORM;
		case RG16_UNORM: return MTLPixelFormat.RG16_UNORM;
		case RGBA16_UNORM: return MTLPixelFormat.RGBA16_UNORM;
		case R8_UINT: return MTLPixelFormat.R8_UINT;
		case RG8_UINT: return MTLPixelFormat.RG8_UINT;
		case RGBA8_UINT: return MTLPixelFormat.RGBA8_UINT;
		case R16_UINT: return MTLPixelFormat.R16_UINT;
		case RG16_UINT: return MTLPixelFormat.RG16_UINT;
		case RGBA16_UINT: return MTLPixelFormat.RGBA16_UINT;
		case R32_UINT: return MTLPixelFormat.R32_UINT;
		case RG32_UINT: return MTLPixelFormat.RG32_UINT;
		case RGBA32_UINT: return MTLPixelFormat.RGBA32_UINT;
		case R16_FLOAT: return MTLPixelFormat.R16_FLOAT;
		case RG16_FLOAT: return MTLPixelFormat.RG16_FLOAT;
		case RGBA16_FLOAT: return MTLPixelFormat.RGBA16_FLOAT;
		case R32_FLOAT: return MTLPixelFormat.R32_FLOAT;
		case RG32_FLOAT: return MTLPixelFormat.RG32_FLOAT;
		case RGBA32_FLOAT: return MTLPixelFormat.RGBA32_FLOAT;
		case RGB10A2_UNORM: return MTLPixelFormat.RGB10A2_UNORM;
		case RG11B10_FLOAT: return MTLPixelFormat.RG11B10_FLOAT;
		case D16_UNORM: return MTLPixelFormat.DEPTH16_UNORM;
		case D32_FLOAT: return MTLPixelFormat.DEPTH32_FLOAT;
		case S8_UINT: return MTLPixelFormat.STENCIL8;
		case D32_FLOAT_S8_UINT: return MTLPixelFormat.DEPTH32_FLOAT_STENCIL8;
		case EAC_RG: return MTLPixelFormat.EAC_RG11_UNORM;
		case ASTC_4X4_SRGB: return MTLPixelFormat.ASTC_4X4_SRGB;
		case ASTC_4X4_UNORM: return MTLPixelFormat.ASTC_4X4_LDR;
		case BC3_SRGB: return MTLPixelFormat.BC3_RGBA_SRGB;
		case BC3_UNORM: return MTLPixelFormat.BC3_RGBA;
		case BC5_RG: return MTLPixelFormat.BC5_RG_UNORM;
		case BC6H_UFLOAT: return MTLPixelFormat.BC6H_RGB_UFLOAT;
		case BC6H_SFLOAT: return MTLPixelFormat.BC6H_RGB_FLOAT;
		case BC7_SRGB: return MTLPixelFormat.BC7_RGBA_UNORM_SRGB;
		case BC7_UNORM: return MTLPixelFormat.BC7_RGBA_UNORM;
		case RGBA4_UNORM: return MTLPixelFormat.ABGR4_UNORM;
		case R5G5B5A1_UNORM: return MTLPixelFormat.A1BGR5_UNORM;
		case R5G6B5_UNORM: return MTLPixelFormat.B5G6R5_UNORM;
		case BGRA8_UINT:
		case D24_UNORM_S8_UINT:
		case UNDEFINED: return MTLPixelFormat.INVALID;
	}
}

fn NSUInteger texture_type(TextureType type) @local
{
	switch (type)
	{
		case ONE_D: return mtl::TextureType.D1;
		case TWO_D: return mtl::TextureType.D2;
		case THREE_D: return mtl::TextureType.D3;
		case CUBE: return mtl::TextureType.CUBE;
		case TWO_D_ARRAY: return mtl::TextureType.D2_ARRAY;
		case CUBE_ARRAY: return mtl::TextureType.CUBE_ARRAY;
	}
}

// Metal's MTLCompareFunction order differs from CompareOp, whose zero is LESS_EQUAL.
fn NSUInteger compare_function(CompareOp op) @local
{
	switch (op)
	{
		case DEFAULT:
		case LESS_EQUAL: return mtl::CompareFunction.LESS_EQUAL;
		case NEVER: return mtl::CompareFunction.NEVER;
		case LESS: return mtl::CompareFunction.LESS;
		case EQUAL: return mtl::CompareFunction.EQUAL;
		case GREATER: return mtl::CompareFunction.GREATER;
		case NOT_EQUAL: return mtl::CompareFunction.NOT_EQUAL;
		case GREATER_EQUAL: return mtl::CompareFunction.GREATER_EQUAL;
		case ALWAYS: return mtl::CompareFunction.ALWAYS;
	}
}

// A new descriptor that the caller releases.
fn ObjId texture_descriptor(TextureDesc desc) @local
{
	ObjId result = objc::@class("MTLTextureDescriptor").send(ObjId, "new");
	result.send_void("setTextureType:", texture_type(desc.type));
	result.send_void("setPixelFormat:", pixel_format(desc.format));
	result.send_void("setWidth:", (NSUInteger)desc.extent.x);
	result.send_void("setHeight:", (NSUInteger)desc.extent.y);
	result.send_void("setDepth:", (NSUInteger)desc.extent.z);
	result.send_void("setMipmapLevelCount:", (NSUInteger)desc.mip_levels);
	result.send_void("setArrayLength:", (NSUInteger)(desc.type == CUBE_ARRAY ? desc.layer_count / 6 : desc.type == TWO_D_ARRAY ? desc.layer_count : 1));
	result.send_void("setStorageMode:", mtl::StorageMode.PRIVATE);
	result.send_void("setHazardTrackingMode:", mtl::HazardTrackingMode.UNTRACKED);
	NSUInteger usage = mtl::TextureUsage.UNKNOWN;
	if (desc.usage.sampled) usage |= mtl::TextureUsage.SHADER_READ;
	if (desc.usage.storage) usage |= mtl::TextureUsage.SHADER_READ | mtl::TextureUsage.SHADER_WRITE;
	if (desc.usage.color_attachment || desc.usage.depth_stencil_attachment) usage |= mtl::TextureUsage.RENDER_TARGET;
	if (desc.mutable_format || get_texture_format_info(desc.format).stencil) usage |= mtl::TextureUsage.PIXEL_FORMAT_VIEW;
	result.send_void("setUsage:", usage);
	return result;
}

// ---- Residency ------------------------------------------------------------------------------------------------------------------

fn void add_resident(Device* device, ObjId allocation) @local
{
	mtl::os_unfair_lock_lock(&device.residency_lock);
	device.residency.send_void("addAllocation:", allocation);
	device.residency.send_void("commit");
	mtl::os_unfair_lock_unlock(&device.residency_lock);
}

fn void remove_resident(Device* device, ObjId allocation) @local
{
	mtl::os_unfair_lock_lock(&device.residency_lock);
	device.residency.send_void("removeAllocation:", allocation);
	device.residency.send_void("commit");
	mtl::os_unfair_lock_unlock(&device.residency_lock);
}

// ---- Timestamp storage ----------------------------------------------------------------------------------------------------------

<*
 Claims `timestamp_query_count` slots from the lock free pages of counter heaps, adding a page when all of them are full.
*>
fn bool allocate_timestamps(Device* device, TimestampSlot* slots) @local
{
	uint allocated = 0;
	for (;;)
	{
		CounterPage* head = mem::@atomic_load(device.counter_pages, ACQUIRE);
		uint published = 0;
		for (CounterPage* page = head; page; page = page.next)
		{
			published++;
			for (uint word = 0; word < 64; word++)
			{
				ulong occupied = mem::@atomic_load(page.occupied[word], RELAXED);
				while (occupied != ulong::max)
				{
					ulong available = ~occupied;
					ulong claim = 0;
					for (uint i = allocated; i < device.timestamp_query_count && available; i++)
					{
						claim |= (ulong)1 << available.ctz();
						available &= available - 1;
					}
					ulong seen = mem::compare_exchange(&page.occupied[word], occupied, occupied | claim, ACQUIRE_RELEASE, RELAXED, false, true);
					if (seen != occupied)
					{
						occupied = seen;
						continue;
					}
					while (claim)
					{
						uint bit = (uint)claim.ctz();
						slots[allocated++] = { .heap = page.heap, .page = page, .index = word * 64 + bit };
						claim &= claim - 1;
					}
					if (allocated == device.timestamp_query_count) return true;
					break;
				}
			}
		}
		if (mem::@atomic_load(device.counter_pages, ACQUIRE) != head) continue;
		uint page_count = mem::@atomic_load(device.counter_page_count, RELAXED);
		if (page_count == 32)
		{
			if (published != 32) continue;
			report_error("timestamp storage capacity exhausted", NIL);
			return false;
		}
		if (mem::compare_exchange(&device.counter_page_count, page_count, page_count + 1, RELAXED, RELAXED, false, true) != page_count) continue;
		if (mem::@atomic_load(device.counter_pages, ACQUIRE) != head)
		{
			atomic::fetch_sub(&device.counter_page_count, (uint)1, RELAXED);
			continue;
		}
		ObjId error;
		ObjId counters = objc::@class("MTL4CounterHeapDescriptor").send(ObjId, "new");
		counters.send_void("setType:", mtl::CounterHeapType.TIMESTAMP);
		counters.send_void("setCount:", (NSUInteger)4096);
		CounterPage* page = mem::new(CounterPage);
		page.heap = device.metal.send(ObjId, "newCounterHeapWithDescriptor:error:", counters, &error);
		counters.release();
		if (!page.heap)
		{
			atomic::fetch_sub(&device.counter_page_count, (uint)1, RELAXED);
			report_error("timestamp storage", error);
			mem::free(page);
			return false;
		}
		// Capacity was reserved before creation, and every successfully created page is retained after publication races.
		for (;;)
		{
			page.next = head;
			ulong seen = mem::compare_exchange((ulong*)&device.counter_pages, (ulong)(uptr)head, (ulong)(uptr)page, ACQUIRE_RELEASE, ACQUIRE, false, true);
			if (seen == (ulong)(uptr)head) break;
			head = (CounterPage*)(uptr)seen;
		}
	}
}

// ---- Command buffers ------------------------------------------------------------------------------------------------------------

fn void destroy_context(Device* device, CommandBuffer* commands) @local
{
	for (NativeCommandBuffer* native = commands.native_buffers; native;)
	{
		NativeCommandBuffer* next = native.next;
		native.buffer.release();
		native.allocator.release();
		mem::free(native);
		native = next;
	}
	commands.arguments.release();
	commands.pass.release();
	for (uint i = 0; i < device.timestamp_query_count && commands.timestamps && commands.timestamps[i].page;)
	{
		CounterPage* page = commands.timestamps[i].page;
		uint word = commands.timestamps[i].index / 64;
		ulong release = 0;
		do
		{
			release |= (ulong)1 << (commands.timestamps[i++].index % 64);
		}
		while (i < device.timestamp_query_count && commands.timestamps[i].page == page && commands.timestamps[i].index / 64 == word);
		atomic::fetch_and(&page.occupied[word], ~release, RELEASE);
	}
	mem::free(commands.timestamps);
	mem::free(commands);
}

fn CommandBuffer*? create_context(CommandPool* pool) @local
{
	Device* device = pool.device;
	CommandBuffer* result = mem::new(CommandBuffer, { .device = device, .pool = pool });
	defer catch destroy_context(device, result);
	ObjId error;
	result.native_buffers = mem::new(NativeCommandBuffer, {
		.buffer = device.metal.send(ObjId, "newCommandBuffer"),
		.allocator = device.metal.send(ObjId, "newCommandAllocator"),
	});
	result.commands = result.native_buffers.buffer;
	ObjId arguments = objc::@class("MTL4ArgumentTableDescriptor").send(ObjId, "new");
	arguments.send_void("setMaxBufferBindCount:", (NSUInteger)3);
	arguments.send_void("setInitializeBindings:", true);
	result.arguments = device.metal.send(ObjId, "newArgumentTableWithDescriptor:error:", arguments, &error);
	arguments.release();
	if (!result.commands || !result.native_buffers.allocator || !result.arguments) return fail(CREATION_FAILED, "command context", error)~;
	result.pass = objc::@class("MTL4RenderPassDescriptor").send(ObjId, "new");
	if (device.timestamp_query_count)
	{
		result.timestamps = mem::new_array(TimestampSlot, device.timestamp_query_count);
		if (!allocate_timestamps(device, result.timestamps)) return CREATION_FAILED~;
	}
	return result;
}

fn void end_compute(CommandBuffer* commands) @local
{
	if (!commands.compute) return;
	commands.compute.send_void("endEncoding");
	commands.compute.release();
	commands.compute = NIL;
}

fn ObjId native_commands(CommandBuffer* commands) @local
{
	if (!commands.commands)
	{
		if (!commands.native.next)
		{
			commands.native.next = mem::new(NativeCommandBuffer, {
				.buffer = commands.device.metal.send(ObjId, "newCommandBuffer"),
				.allocator = commands.device.metal.send(ObjId, "newCommandAllocator"),
			});
		}
		commands.native = commands.native.next;
		commands.commands = commands.native.buffer;
		commands.native_count++;
		commands.commands.send_void("beginCommandBufferWithAllocator:", commands.native.allocator);
	}
	return commands.commands;
}

fn ObjId compute_encoder(CommandBuffer* commands) @local
{
	assert(commands.recording && !commands.render);
	if (!commands.compute)
	{
		commands.compute = native_commands(commands).send(ObjId, "computeCommandEncoder").retain();
		commands.compute.send_void("setArgumentTable:", commands.arguments);
		if (commands.pso && commands.pso.compute) commands.compute.send_void("setComputePipelineState:", commands.pso.compute);
	}
	return commands.compute;
}

// ---- Address to buffer lookup ---------------------------------------------------------------------------------------------------
//
// Metal's copy encoders take a buffer, the API only a GPU address: see address_map.c3.

fn ObjId resolve_buffer(Device* device, GpuRange range, ulong* offset) @local => (ObjId)(uptr)device.buffer_map.resolve(range, offset);

// ---- Device ---------------------------------------------------------------------------------------------------------------------

<*
 Creates the device. A headless device has no window; with a window, a CAMetalLayer*, the device presents to it.
 Needs an Apple7 GPU with Metal 4. Faults: UNSUPPORTED, DRIVER_ERROR.
*>
fn Device*? create_device(DeviceDesc desc = {}) => objc::@autoreleasepool()
{
	if (desc.swapchain_color_space != SRGB) return fail(UNSUPPORTED, "unsupported swapchain color space")~;
	Device* device = mem::new(Device);
	defer catch device.destroy();
	// MetalTools on macOS 26.6.2 cannot enumerate placed resources through heap residency.
	ZString shader_validation = libc::getenv("MTL_SHADER_VALIDATION");
	device.shader_validation = shader_validation && libc::atoi((char*)shader_validation) != 0;
	device.metal = darwin::mtlCreateSystemDefaultDevice();
	if (!device.metal || !device.metal.send(bool, "supportsFamily:", mtl::GPUFamily.APPLE7) || !device.metal.send(bool, "supportsFamily:", mtl::GPUFamily.METAL4))
	{
		return fail(UNSUPPORTED, "Apple7 and Metal4 GPU support is required")~;
	}
	ObjId error;
	uint general = or_one(desc.desired_queue_count);
	uint queue_count = general + desc.desired_compute_queue_count + desc.desired_copy_queue_count;
	device.caps.queue_count = queue_count;
	device.queues = mem::new_array(Queue, queue_count);
	for (uint i = 0; i < queue_count; i++)
	{
		Queue* queue = &device.queues[i];
		queue.submission.init(mem);
		queue.kind = i < general ? GENERAL : i < general + desc.desired_compute_queue_count ? COMPUTE : COPY;
		void* attr = mtl::dispatch_queue_attr_make_with_autorelease_frequency(null, mtl::DISPATCH_AUTORELEASE_FREQUENCY_WORK_ITEM);
		queue.feedback = mtl::dispatch_queue_create("NoGraphicsAPI Metal feedback", attr);
		ObjId descriptor = objc::@class("MTL4CommandQueueDescriptor").send(ObjId, "new");
		descriptor.send_void("setFeedbackQueue:", queue.feedback);
		queue.queue = device.metal.send(ObjId, "newMTL4CommandQueueWithDescriptor:error:", descriptor, &error);
		// The SDK's descriptor releases its assign-only feedbackQueue when it goes away; the queue keeps its own reference.
		descriptor.send_void("setFeedbackQueue:", (void*)null);
		descriptor.release();
		queue.completion = device.metal.send(ObjId, "newSharedEvent");
		if (!queue.queue || !queue.completion) return fail(DRIVER_ERROR, "command queue", error)~;
	}
	ObjId compiler = objc::@class("MTL4CompilerDescriptor").send(ObjId, "new");
	device.compiler = device.metal.send(ObjId, "newCompilerWithDescriptor:error:", compiler, &error);
	compiler.release();
	if (!device.compiler) return fail(DRIVER_ERROR, "pipeline compiler", error)~;
	ObjId residency = objc::@class("MTLResidencySetDescriptor").send(ObjId, "new");
	residency.send_void("setInitialCapacity:", (NSUInteger)256);
	device.residency = device.metal.send(ObjId, "newResidencySetWithDescriptor:error:", residency, &error);
	residency.release();
	if (!device.residency) return fail(DRIVER_ERROR, "residency set", error)~;
	for (uint i = 0; i < queue_count; i++) device.queues[i].queue.send_void("addResidencySet:", device.residency);
	device.timestamp_query_count = desc.timestamp_query_count;
	ZString name = device.metal.send(ObjId, "name").send(ZString, "UTF8String");
	sz name_len = min((sz)name.len(), (sz)(device.name.len - 1));
	mem::copy(&device.name[0], name, name_len);
	mtl::TimebaseInfo timebase;
	mtl::mach_timebase_info(&timebase);
	device.caps = {
		.device_name = (String)device.name[:name_len],
		.queue_count = queue_count,
		.general_queue_count = general,
		.compute_queue_count = desc.desired_compute_queue_count,
		.copy_queue_count = desc.desired_copy_queue_count,
		.copy_texture_granularity = { 1, 1, 1 },
		.max_push_data_size = 256,
		.texture_heap_alignment = 16384,
		.timestamp_period_ns = (float)timebase.numer / (float)timebase.denom,
		.sub_texel_precision_bits = 8,
		.texture_compression_bc = device.metal.send(bool, "supportsBCTextureCompression"),
		.texture_compression_astc = true,
		.storage_input_output16 = true,
		.indirect_mesh_draw = device.metal.send(bool, "supportsFamily:", mtl::GPUFamily.APPLE9),
	};
	if (desc.window)
	{
		device.layer = ((ObjId)desc.window).retain();
		device.layer.send_void("setDevice:", device.metal);
		device.layer.send_void("setPixelFormat:", pixel_format(desc.swapchain_format == UNDEFINED ? BGRA8_UNORM : desc.swapchain_format));
		device.layer.send_void("setMaximumDrawableCount:", (NSUInteger)(desc.desired_swapchain_image_count < 3 ? 2 : 3));
		device.queues[0].queue.send_void("addResidencySet:", device.layer.send(ObjId, "residencySet"));
		device.layer.send_void("setFramebufferOnly:", true);
		if (desc.steam_overlay_compatibility)
		{
			// Steam's overlay hooks the presentation of a classic command buffer, so a second queue presents. See submit_and_present.
			device.layer.send_void("setFramebufferOnly:", false);
			device.presentation_queue = device.metal.send(ObjId, "newCommandQueue");
			if (!device.presentation_queue) return fail(DRIVER_ERROR, "presentation queue")~;
		}
	}
	return device;
}

<*
 Waits for nothing: call `wait_idle` first, and destroy everything created from the device before it.
*>
fn void Device.destroy(&self) => objc::@autoreleasepool()
{
	assert(!self.drawable);
	if (self.layer) self.queues[0].queue.send_void("removeResidencySet:", self.layer.send(ObjId, "residencySet"));
	self.drawable.release();
	self.presentation_queue.release();
	self.layer.release();
	for (uint i = 0; i < self.caps.queue_count; i++)
	{
		Queue* queue = &self.queues[i];
		assert(queue.completion.send(ulong, "signaledValue") >= queue.submitted_value);
		if (self.residency) queue.queue.send_void("removeResidencySet:", self.residency);
		queue.completion.release();
		queue.queue.release();
		if (queue.feedback) mtl::dispatch_release(queue.feedback);
		queue.submission.free();
	}
	mem::free(self.queues);
	for (CounterPage* page = self.counter_pages; page;)
	{
		CounterPage* next = page.next;
		for (uint word = 0; word < 64; word++) assert(!mem::@atomic_load(page.occupied[word], RELAXED));
		page.heap.release();
		mem::free(page);
		page = next;
	}
	self.residency.release();
	self.compiler.release();
	self.metal.release();
	mem::free(self);
}

fn DeviceCaps* Device.get_caps(&self) => &self.caps;

fn bool Device.supports_texture_format(&self, Format format, TextureUsage usage)
{
	if (pixel_format(format) == MTLPixelFormat.INVALID || format == D24_UNORM_S8_UINT) return false;
	TextureFormatInfo info = get_texture_format_info(format);
	if (info.depth && info.stencil && (usage.transfer_source || usage.transfer_destination)) return false;
	bool compressed = info.block_extent.x > 1;
	bool bc = format.ordinal >= Format.BC3_SRGB.ordinal && format.ordinal <= Format.BC7_UNORM.ordinal;
	if (bc && !self.caps.texture_compression_bc) return false;
	if (usage.storage)
	{
		if (compressed || info.depth || info.stencil || format == R8_SRGB || format == RG8_SRGB || format == RGBA8_SRGB || format == BGRA8_SRGB
			|| format == RGBA4_UNORM || format == R5G5B5A1_UNORM || format == R5G6B5_UNORM) return false;
	}
	if (usage.color_attachment && (compressed || info.depth || info.stencil)) return false;
	if (usage.depth_stencil_attachment && !(info.depth || info.stencil)) return false;
	return true;
}

<*
 Call after `wait_idle`, outside an acquired frame. BGRA8_UNORM and BGRA8_SRGB only; an unsupported pair leaves the mode unchanged.
*>
fn void? Device.set_swapchain_format(&self, Format format, ColorSpace color_space) => objc::@autoreleasepool()
{
	assert(!self.drawable && !self.acquired);
	if (!self.layer || color_space != SRGB) return UNSUPPORTED~;
	if (self.layer.send(NSUInteger, "pixelFormat") == pixel_format(format)) return;
	if (format != BGRA8_UNORM && format != BGRA8_SRGB) return UNSUPPORTED~;
	self.layer.send_void("setPixelFormat:", pixel_format(format));
}

<*
 The size of the layer's drawable in pixels, zero when there is no window.
*>
fn Uint2 Device.drawable_extent(&self) => objc::@autoreleasepool()
{
	if (!self.layer) return {};
	mtl::CGSize size = self.layer.send(mtl::CGSize, "drawableSize");
	return { (uint)size.width, (uint)size.height };
}

// ---- Timeline semaphores --------------------------------------------------------------------------------------------------------

fn TimelineSemaphore*? Device.create_timeline_semaphore(&self, ulong initial_value = 0) => objc::@autoreleasepool()
{
	ObjId event = self.metal.send(ObjId, "newSharedEvent");
	if (!event) return fail(CREATION_FAILED, "timeline")~;
	event.send_void("setSignaledValue:", initial_value);
	return mem::new(TimelineSemaphore, { .device = self, .event = event });
}

fn void TimelineSemaphore.destroy(&self) => objc::@autoreleasepool()
{
	self.event.release();
	mem::free(self);
}

fn ulong TimelineSemaphore.completed_value(&self) => objc::@autoreleasepool()
{
	return self.event.send(ulong, "signaledValue");
}

fn void TimelinePoint.wait(self) => objc::@autoreleasepool()
{
	assert(self.semaphore);
	while (!self.semaphore.event.send(bool, "waitUntilSignaledValue:timeoutMS:", self.value, (ulong)1000)) {}
}

fn void Device.wait_idle(&self) => objc::@autoreleasepool()
{
	for (uint i = 0; i < self.caps.queue_count; i++)
	{
		Queue* queue = &self.queues[i];
		queue.queue.send_void("signalEvent:value:", queue.completion, ++queue.submitted_value);
	}
	for (uint i = 0; i < self.caps.queue_count; i++)
	{
		Queue* queue = &self.queues[i];
		while (!queue.completion.send(bool, "waitUntilSignaledValue:timeoutMS:", queue.submitted_value, (ulong)1000)) {}
	}
	if (self.presentation_queue)
	{
		ObjId drained = self.presentation_queue.send(ObjId, "commandBuffer");
		drained.send_void("commit");
		drained.send_void("waitUntilCompleted");
	}
}

// ---- GPU heaps ------------------------------------------------------------------------------------------------------------------

<*
 Allocates a heap of raw memory and returns its CPU and GPU addresses. Every non null pointer is 16 byte aligned. At most 64 heaps can be
 live per device, and the range of a copy must lie inside one heap.
*>
fn GpuHeap? Device.create_gpu_heap(&self, ulong byte_count, MemoryType memory = CPU_VISIBLE) => objc::@autoreleasepool()
{
	NSUInteger options = memory == GPU_ONLY ? mtl::ResourceOptions.STORAGE_MODE_PRIVATE : mtl::ResourceOptions.STORAGE_MODE_SHARED;
	if (memory == CPU_VISIBLE) options |= mtl::ResourceOptions.CPU_CACHE_MODE_WRITE_COMBINED;
	options |= mtl::ResourceOptions.HAZARD_TRACKING_MODE_UNTRACKED;
	mtl::SizeAndAlign requirements = self.metal.send(mtl::SizeAndAlign, "heapBufferSizeAndAlignWithLength:options:", (NSUInteger)byte_count, options);
	if (!requirements.size || !requirements.align) return fail(CREATION_FAILED, "GPU heap memory requirements")~;
	ObjId desc = objc::@class("MTLHeapDescriptor").send(ObjId, "new");
	defer desc.release();
	desc.send_void("setType:", mtl::HeapType.PLACEMENT);
	desc.send_void("setSize:", (NSUInteger)align_up(requirements.size, requirements.align));
	desc.send_void("setResourceOptions:", options);
	GpuHeapOwner* owner = mem::new(GpuHeapOwner, { .device = self, .heap = self.metal.send(ObjId, "newHeapWithDescriptor:", desc) });
	if (owner.heap) owner.buffer = owner.heap.send(ObjId, "newBufferWithLength:options:offset:", (NSUInteger)byte_count, options, (NSUInteger)0);
	if (!owner.buffer)
	{
		owner.heap.release();
		mem::free(owner);
		return fail(CREATION_FAILED, "GPU heap")~;
	}
	ulong address = owner.buffer.send(ulong, "gpuAddress");
	owner.record = self.buffer_map.insert(address, byte_count, (ulong)(uptr)owner.buffer);
	add_resident(self, self.shader_validation ? owner.buffer : owner.heap);
	return {
		.cpu = memory == GPU_ONLY ? null : owner.buffer.send(char*, "contents"),
		.gpu = (char*)(uptr)address,
		.size = byte_count,
		.owner = owner,
	};
}

fn void GpuHeap.destroy(&self) => objc::@autoreleasepool()
{
	GpuHeapOwner* owner = self.owner;
	if (!owner) return;
	owner.device.buffer_map.remove(owner.record);
	remove_resident(owner.device, owner.device.shader_validation ? owner.buffer : owner.heap);
	owner.buffer.release();
	owner.heap.release();
	mem::free(owner);
}

fn GpuRange GpuHeap.range(&self) => { self.gpu, self.size };

// ---- Textures -------------------------------------------------------------------------------------------------------------------

fn TextureHeap? Device.create_texture_heap(&self, ulong byte_count) => objc::@autoreleasepool()
{
	ObjId desc = objc::@class("MTLHeapDescriptor").send(ObjId, "new");
	defer desc.release();
	desc.send_void("setType:", mtl::HeapType.PLACEMENT);
	desc.send_void("setSize:", (NSUInteger)byte_count);
	desc.send_void("setStorageMode:", mtl::StorageMode.PRIVATE);
	desc.send_void("setHazardTrackingMode:", mtl::HazardTrackingMode.UNTRACKED);
	ObjId heap = self.metal.send(ObjId, "newHeapWithDescriptor:", desc);
	if (!heap) return fail(CREATION_FAILED, "texture heap")~;
	if (!self.shader_validation) add_resident(self, heap);
	return { .size = byte_count, .owner = mem::new(TextureHeapOwner, { .device = self, .heap = heap }) };
}

fn void TextureHeap.destroy(&self) => objc::@autoreleasepool()
{
	TextureHeapOwner* owner = self.owner;
	if (!owner) return;
	if (!owner.device.shader_validation) remove_resident(owner.device, owner.heap);
	owner.heap.release();
	mem::free(owner);
}

<*
 The size and alignment a texture needs in a TextureHeap. Placements must satisfy the alignment.
*>
fn SizeAlign? Device.get_texture_size_align(&self, TextureDesc desc) => objc::@autoreleasepool()
{
	desc = normalized(desc);
	if (!self.supports_texture_format(desc.format, desc.usage)) return fail(UNSUPPORTED, "unsupported texture format or usage")~;
	ObjId descriptor = texture_descriptor(desc);
	mtl::SizeAndAlign result = self.metal.send(mtl::SizeAndAlign, "heapTextureSizeAndAlignWithDescriptor:", descriptor);
	descriptor.release();
	if (!result.size || !result.align) return fail(CREATION_FAILED, "texture memory requirements")~;
	assert(self.caps.texture_heap_alignment % result.align == 0);
	return { result.size, result.align };
}

<*
 Places a texture in a texture heap. Records nothing: views and descriptors can be made at once, and `initialize_texture` is called
 before the first use on the GPU (it does nothing on Metal, portable code calls it).
*>
fn Texture*? Device.create_texture(&self, TextureDesc desc, TextureHeap heap, ulong offset) => objc::@autoreleasepool()
{
	Device* device = self;
	assert(heap.owner && heap.owner.device == device);
	desc = normalized(desc);
	ObjId descriptor = texture_descriptor(desc);
	ObjId texture = heap.owner.heap.send(ObjId, "newTextureWithDescriptor:offset:", descriptor, (NSUInteger)offset);
	descriptor.release();
	if (!texture) return fail(CREATION_FAILED, "texture")~;
	if (device.shader_validation) add_resident(device, texture);
	return mem::new(Texture, { .device = device, .texture = texture, .desc = desc });
}

fn void Texture.destroy(&self) => objc::@autoreleasepool()
{
	if (self.device.shader_validation) remove_resident(self.device, self.texture);
	self.texture.release();
	mem::free(self);
}

<*
 Initializes a texture's metadata outside a render pass, before its first use on the GPU and before reuse after writes through an overlapping
 texture. Discards the contents: clear or overwrite before reading. The caller synchronizes the earlier uses of the storage with barriers
 or timeline waits first. Metal needs none of it, so this only checks its arguments.
*>
fn void CommandBuffer.initialize_texture(&self, Texture* texture)
{
	assert(texture && self.device == texture.device && self.recording && !self.render);
}

fn RenderView*? Texture.create_render_view(&self, RenderViewDesc desc = {}) => objc::@autoreleasepool()
{
	return mem::new(RenderView, { .texture = self.texture.retain(), .mip = desc.mip_level, .slice = desc.slice });
}

fn void RenderView.destroy(&self) => objc::@autoreleasepool()
{
	self.texture.release();
	mem::free(self);
}

// ---- Descriptor heaps -----------------------------------------------------------------------------------------------------------

fn TextureDescriptorHeap*? Device.create_texture_descriptor_heap(&self, uint capacity) => objc::@autoreleasepool()
{
	ObjId error;
	ObjId desc = objc::@class("MTLResourceViewPoolDescriptor").send(ObjId, "new");
	defer desc.release();
	desc.send_void("setResourceViewCount:", (NSUInteger)capacity);
	ObjId pool = self.metal.send(ObjId, "newTextureViewPoolWithDescriptor:error:", desc, &error);
	if (!pool) return fail(CREATION_FAILED, "texture descriptor heap", error)~;
	// The shaders add a texture's index to this base, they read it from the buffer bound at slot 1.
	mtl::ResourceID base = pool.send(mtl::ResourceID, "baseResourceID");
	ObjId base_buffer = self.metal.send(ObjId, "newBufferWithBytes:length:options:", &base, (NSUInteger)mtl::ResourceID::size, mtl::ResourceOptions.STORAGE_MODE_SHARED);
	if (!base_buffer)
	{
		pool.release();
		return fail(CREATION_FAILED, "texture pool base")~;
	}
	add_resident(self, base_buffer);
	return mem::new(TextureDescriptorHeap, { .device = self, .pool = pool, .base = base_buffer, .capacity = capacity });
}

fn void TextureDescriptorHeap.destroy(&self) => objc::@autoreleasepool()
{
	remove_resident(self.device, self.base);
	self.base.release();
	self.pool.release();
	mem::free(self);
}

<*
 Writes a descriptor for `texture` into slot `index`. Do not modify a slot until all commands using its previous contents completed.
*>
fn void TextureDescriptorHeap.write(&self, uint index, Texture* texture, TextureDescriptorType type, TextureDescriptorDesc desc = {}) => objc::@autoreleasepool()
{
	assert(index < self.capacity && texture.device == self.device);
	ObjId view = objc::@class("MTLTextureViewDescriptor").send(ObjId, "new");
	view.send_void("setPixelFormat:", pixel_format(desc.format == UNDEFINED ? texture.desc.format : desc.format));
	if (desc.aspect == STENCIL && texture.desc.format == D32_FLOAT_S8_UINT) view.send_void("setPixelFormat:", (NSUInteger)MTLPixelFormat.X32_STENCIL8);
	NSUInteger view_type = texture.texture.send(NSUInteger, "textureType");
	if (type == STORAGE && (view_type == mtl::TextureType.CUBE || view_type == mtl::TextureType.CUBE_ARRAY)) view_type = mtl::TextureType.D2_ARRAY;
	view.send_void("setTextureType:", view_type);
	view.send_void("setLevelRange:", mtl::range(desc.base_mip, desc.mip_count ?: (texture.desc.mip_levels - desc.base_mip)));
	view.send_void("setSliceRange:", mtl::range(desc.base_layer, desc.layer_count ?: (texture.desc.layer_count - desc.base_layer)));
	mtl::ResourceID resource = self.pool.send(mtl::ResourceID, "setTextureView:descriptor:atIndex:", texture.texture, view, (NSUInteger)index);
	assert(resource.impl == self.pool.send(mtl::ResourceID, "baseResourceID").impl + index);
	view.release();
}

<*
 Copies `count` descriptors between heaps of the same device. The ranges may overlap.
*>
fn void TextureDescriptorHeap.copy_from(&self, uint destination_index, TextureDescriptorHeap* source, uint source_index, uint count) => objc::@autoreleasepool()
{
	assert(source.device == self.device && source_index + count <= source.capacity && destination_index + count <= self.capacity);
	if (!count || (source == self && source_index == destination_index)) return;
	if (source == self && source_index < destination_index + count && destination_index < source_index + count)
	{
		// Overlapping ranges are copied one descriptor at a time, from the end that is not overwritten first.
		bool backwards = destination_index > source_index;
		for (uint n = 0; n < count; n++)
		{
			uint i = backwards ? count - 1 - n : n;
			self.pool.send_void("copyResourceViewsFromPool:sourceRange:destinationIndex:", source.pool, mtl::range(source_index + i, 1), (NSUInteger)(destination_index + i));
		}
		return;
	}
	self.pool.send_void("copyResourceViewsFromPool:sourceRange:destinationIndex:", source.pool, mtl::range(source_index, count), (NSUInteger)destination_index);
}

fn SamplerDescriptorHeap*? Device.create_sampler_descriptor_heap(&self, uint capacity) => objc::@autoreleasepool()
{
	// A sampler descriptor is the 8 byte resource id of its state, in a buffer that the shader reads.
	ObjId buffer = self.metal.send(ObjId, "newBufferWithLength:options:", (NSUInteger)(capacity * mtl::ResourceID::size), mtl::ResourceOptions.STORAGE_MODE_SHARED);
	if (!buffer) return fail(CREATION_FAILED, "sampler heap")~;
	mem::clear(buffer.send(void*, "contents"), capacity * mtl::ResourceID::size);
	add_resident(self, buffer);
	return mem::new(SamplerDescriptorHeap, { .device = self, .buffer = buffer, .samplers = mem::new_array(ObjId, capacity), .capacity = capacity });
}

fn void SamplerDescriptorHeap.destroy(&self) => objc::@autoreleasepool()
{
	for (uint i = 0; i < self.capacity; i++) self.samplers[i].release();
	mem::free(self.samplers);
	remove_resident(self.device, self.buffer);
	self.buffer.release();
	mem::free(self);
}

fn void SamplerDescriptorHeap.write(&self, uint index, SamplerDesc desc = {}) => objc::@autoreleasepool()
{
	assert(index < self.capacity);
	NSUInteger[3] address_modes = { mtl::SamplerAddressMode.REPEAT, mtl::SamplerAddressMode.MIRROR_REPEAT, mtl::SamplerAddressMode.CLAMP_TO_EDGE };
	ObjId sampler = objc::@class("MTLSamplerDescriptor").send(ObjId, "new");
	sampler.send_void("setMinFilter:", desc.min_filter == LINEAR ? mtl::SamplerMinMagFilter.LINEAR : mtl::SamplerMinMagFilter.NEAREST);
	sampler.send_void("setMagFilter:", desc.mag_filter == LINEAR ? mtl::SamplerMinMagFilter.LINEAR : mtl::SamplerMinMagFilter.NEAREST);
	sampler.send_void("setMipFilter:", desc.mip_filter == LINEAR ? mtl::SamplerMipFilter.LINEAR : mtl::SamplerMipFilter.NEAREST);
	sampler.send_void("setSAddressMode:", address_modes[desc.address_u.ordinal]);
	sampler.send_void("setTAddressMode:", address_modes[desc.address_v.ordinal]);
	sampler.send_void("setRAddressMode:", address_modes[desc.address_w.ordinal]);
	sampler.send_void("setMaxAnisotropy:", (NSUInteger)(desc.anisotropic ? 4 : 1));
	sampler.send_void("setCompareFunction:", desc.compare_enabled ? compare_function(desc.compare) : mtl::CompareFunction.NEVER);
	sampler.send_void("setSupportArgumentBuffers:", true);
	ObjId state = self.device.metal.send(ObjId, "newSamplerStateWithDescriptor:", sampler);
	sampler.release();
	assert(state);
	self.samplers[index].release();
	self.samplers[index] = state;
	((mtl::ResourceID*)self.buffer.send(void*, "contents"))[index] = state.send(mtl::ResourceID, "gpuResourceID");
}

fn void SamplerDescriptorHeap.copy_from(&self, uint destination_index, SamplerDescriptorHeap* source, uint source_index, uint count) => objc::@autoreleasepool()
{
	assert(source.device == self.device && source_index + count <= source.capacity && destination_index + count <= self.capacity);
	if (!count || (source == self && source_index == destination_index)) return;
	bool backwards = source == self && destination_index > source_index;
	for (uint n = 0; n < count; n++)
	{
		uint i = backwards ? count - 1 - n : n;
		ObjId state = source.samplers[source_index + i].retain();
		self.samplers[destination_index + i].release();
		self.samplers[destination_index + i] = state;
		((mtl::ResourceID*)self.buffer.send(void*, "contents"))[destination_index + i] = state ? state.send(mtl::ResourceID, "gpuResourceID") : (mtl::ResourceID){};
	}
}

// ---- Pipelines ------------------------------------------------------------------------------------------------------------------

// The function descriptor of a shader stage, +1, or a null id for an empty stage.
fn ObjId? shader_function(Device* device, ShaderStage stage) @local
{
	if (!stage.code.len) return NIL;
	ObjId error;
	ObjId library;
	if (stage.code.len >= 4 && stage.code[:4] == "MTLB")
	{
		void* data = mtl::dispatch_data_create(stage.code.ptr, stage.code.len, null, null);
		library = device.metal.send(ObjId, "newLibraryWithData:error:", data, &error);
		mtl::dispatch_release(data);
	}
	else
	{
		// Metal Shading Language source. The Metal toolchain (xcrun metal) is not needed, the system compiles it.
		ZString source = ((String)stage.code).zstr_copy(mem);
		defer mem::free(source);
		library = device.metal.send(ObjId, "newLibraryWithSource:options:error:", darwin::string(source), NIL, &error);
	}
	if (!library) return fail(CREATION_FAILED, "shader library", error)~;
	defer library.release();
	ObjId function = objc::@class("MTL4LibraryFunctionDescriptor").send(ObjId, "new");
	function.send_void("setLibrary:", library);
	function.send_void("setName:", darwin::string(stage.entry_point ?: (ZString)"main"));
	return function;
}

fn NSUInteger blend_factor(BlendFactor factor) @local
{
	NSUInteger[11] factors = {
		mtl::BlendFactor.ZERO, mtl::BlendFactor.ONE, mtl::BlendFactor.SOURCE_COLOR, mtl::BlendFactor.ONE_MINUS_SOURCE_COLOR,
		mtl::BlendFactor.DESTINATION_COLOR, mtl::BlendFactor.ONE_MINUS_DESTINATION_COLOR, mtl::BlendFactor.SOURCE_ALPHA,
		mtl::BlendFactor.ONE_MINUS_SOURCE_ALPHA, mtl::BlendFactor.DESTINATION_ALPHA, mtl::BlendFactor.ONE_MINUS_DESTINATION_ALPHA,
		mtl::BlendFactor.SOURCE_ALPHA_SATURATED,
	};
	assert(factor != DEFAULT);   // Resolved by the caller: ONE as a source, ZERO as a destination.
	return factors[factor.ordinal - 1];
}

fn void color_targets(ObjId attachments, ColorTargetDesc[] targets) @local
{
	foreach (i, target : targets)
	{
		ObjId attachment = attachments.send(ObjId, "objectAtIndexedSubscript:", (NSUInteger)i);
		attachment.send_void("setPixelFormat:", pixel_format(target.format));
		attachment.send_void("setWriteMask:", color_write_mask(target.write_mask));
		attachment.send_void("setBlendingState:", target.blend.enabled ? mtl::BlendState.ENABLED : mtl::BlendState.DISABLED);
		attachment.send_void("setSourceRGBBlendFactor:", blend_factor(target.blend.color.source.or(ONE)));
		attachment.send_void("setDestinationRGBBlendFactor:", blend_factor(target.blend.color.destination.or(ZERO)));
		attachment.send_void("setRgbBlendOperation:", (NSUInteger)target.blend.color.operation.ordinal);
		attachment.send_void("setSourceAlphaBlendFactor:", blend_factor(target.blend.alpha.source.or(ONE)));
		attachment.send_void("setDestinationAlphaBlendFactor:", blend_factor(target.blend.alpha.destination.or(ZERO)));
		attachment.send_void("setAlphaBlendOperation:", (NSUInteger)target.blend.alpha.operation.ordinal);
	}
}

// Compiles the pipeline now and returns when it is done: there is no pipeline cache. The state is +1 and resident.
fn ObjId? Device.compile_render_pipeline(&self, ObjId pipeline, String what)
{
	ObjId error;
	ObjId state = self.compiler.send(ObjId, "newRenderPipelineStateWithDescriptor:compilerTaskOptions:error:", pipeline, NIL, &error);
	if (!state)
	{
		report_error(what, error);
		return CREATION_FAILED~;
	}
	add_resident(self, state);
	return state;
}

fn Pso*? Device.create_graphics_pso(&self, GraphicsPSODesc desc) => objc::@autoreleasepool()
{
	assert(desc.vertex.code.len, "Graphics PSOs require a vertex stage");
	ObjId pipeline = objc::@class("MTL4RenderPipelineDescriptor").send(ObjId, "new");
	defer pipeline.release();
	ObjId vertex = shader_function(self, desc.vertex)!;
	defer vertex.release();
	ObjId fragment = shader_function(self, desc.fragment)!;
	defer fragment.release();
	pipeline.send_void("setVertexFunctionDescriptor:", vertex);
	pipeline.send_void("setFragmentFunctionDescriptor:", fragment);
	pipeline.send_void("setInputPrimitiveTopology:", mtl::PrimitiveTopologyClass.TRIANGLE);
	color_targets(pipeline.send(ObjId, "colorAttachments"), desc.color_targets);
	ObjId state = self.compile_render_pipeline(pipeline, "graphics pipeline")!;
	return mem::new(Pso, { .device = self, .render = state, .rasterization = desc.rasterization, .threads = { 1, 1, 1 }, .object_threads = { 1, 1, 1 } });
}

fn Pso*? Device.create_mesh_pso(&self, MeshPSODesc desc) => objc::@autoreleasepool()
{
	assert(desc.mesh.code.len, "Mesh PSOs require a mesh stage");
	ObjId pipeline = objc::@class("MTL4MeshRenderPipelineDescriptor").send(ObjId, "new");
	defer pipeline.release();
	ObjId task = shader_function(self, desc.task)!;
	defer task.release();
	ObjId mesh = shader_function(self, desc.mesh)!;
	defer mesh.release();
	ObjId fragment = shader_function(self, desc.fragment)!;
	defer fragment.release();
	pipeline.send_void("setObjectFunctionDescriptor:", task);
	pipeline.send_void("setMeshFunctionDescriptor:", mesh);
	pipeline.send_void("setFragmentFunctionDescriptor:", fragment);
	pipeline.send_void("setRequiredThreadsPerMeshThreadgroup:", metal_size(desc.mesh.threadgroup_size));
	if (task) pipeline.send_void("setRequiredThreadsPerObjectThreadgroup:", metal_size(desc.task.threadgroup_size));
	color_targets(pipeline.send(ObjId, "colorAttachments"), desc.color_targets);
	ObjId state = self.compile_render_pipeline(pipeline, "mesh pipeline")!;
	return mem::new(Pso, {
		.device = self,
		.render = state,
		.rasterization = desc.rasterization,
		.threads = metal_size(desc.mesh.threadgroup_size),
		.object_threads = metal_size(desc.task.threadgroup_size),
		.mesh = true,
	});
}

fn Pso*? Device.create_compute_pso(&self, ShaderStage stage) => objc::@autoreleasepool()
{
	assert(stage.code.len, "Compute PSOs require a compute stage");
	ObjId function = shader_function(self, stage)!;
	defer function.release();
	ObjId desc = objc::@class("MTL4ComputePipelineDescriptor").send(ObjId, "new");
	defer desc.release();
	desc.send_void("setComputeFunctionDescriptor:", function);
	desc.send_void("setRequiredThreadsPerThreadgroup:", metal_size(stage.threadgroup_size));
	ObjId error;
	ObjId state = self.compiler.send(ObjId, "newComputePipelineStateWithDescriptor:compilerTaskOptions:error:", desc, NIL, &error);
	if (!state) return fail(CREATION_FAILED, "compute pipeline", error)~;
	add_resident(self, state);
	return mem::new(Pso, { .device = self, .compute = state, .threads = metal_size(stage.threadgroup_size), .object_threads = { 1, 1, 1 } });
}

fn void Pso.destroy(&self) => objc::@autoreleasepool()
{
	if (self.render) remove_resident(self.device, self.render);
	if (self.compute) remove_resident(self.device, self.compute);
	self.render.release();
	self.compute.release();
	mem::free(self);
}

// ---- Command pools and submission -----------------------------------------------------------------------------------------------

fn bool same_stencil(StencilFaceState a, StencilFaceState b) @local
{
	return a.compare == b.compare && a.fail == b.fail && a.pass == b.pass && a.depth_fail == b.depth_fail;
}

fn bool same_depth(DepthStencilState a, DepthStencilState b) @local
{
	return a.depth_test == b.depth_test && a.depth_write == b.depth_write && a.depth_compare == b.depth_compare && a.stencil_test == b.stencil_test
		&& a.stencil_read_mask == b.stencil_read_mask && a.stencil_write_mask == b.stencil_write_mask
		&& same_stencil(a.front, b.front) && same_stencil(a.back, b.back);
}

fn void stencil_descriptor(ObjId desc, StencilFaceState face, DepthStencilState state) @local
{
	desc.send_void("setStencilCompareFunction:", compare_function(face.compare.or(ALWAYS)));
	desc.send_void("setStencilFailureOperation:", (NSUInteger)face.fail.ordinal);
	desc.send_void("setDepthFailureOperation:", (NSUInteger)face.depth_fail.ordinal);
	desc.send_void("setDepthStencilPassOperation:", (NSUInteger)face.pass.ordinal);
	desc.send_void("setReadMask:", (uint)(state.stencil_read_mask ?: 0xff));
	desc.send_void("setWriteMask:", (uint)(state.stencil_write_mask ?: 0xff));
}

// The one form of a state that the cache keeps: defaults resolved, and the parts that are off or that Metal does not keep in the state object
// (the stencil reference) cleared, so equivalent states are one object.
fn DepthStencilState canonical_depth_state(DepthStencilState state) @local
{
	if (state.depth_test)
	{
		state.depth_compare = state.depth_compare.or(LESS_EQUAL);
	}
	else
	{
		state.depth_write = false;
		state.depth_compare = LESS_EQUAL;
	}
	if (state.stencil_test)
	{
		state.stencil_read_mask = state.stencil_read_mask ?: 0xff;
		state.stencil_write_mask = state.stencil_write_mask ?: 0xff;
		state.front.compare = state.front.compare.or(ALWAYS);
		state.back.compare = state.back.compare.or(ALWAYS);
		state.front.reference = 0;
		state.back.reference = 0;
	}
	else
	{
		state.stencil_read_mask = 0;
		state.stencil_write_mask = 0;
		state.front = {};
		state.back = {};
	}
	return state;
}

// Depth stencil states are immutable Metal objects. Each pool keeps the ones its commands asked for.
fn ObjId depth_state(CommandPool* pool, DepthStencilState state) @local
{
	state = canonical_depth_state(state);
	foreach (&entry : pool.depth_states)
	{
		if (same_depth(entry.desc, state)) return entry.state;
	}
	ObjId desc = objc::@class("MTLDepthStencilDescriptor").send(ObjId, "new");
	desc.send_void("setDepthCompareFunction:", state.depth_test ? compare_function(state.depth_compare) : mtl::CompareFunction.ALWAYS);
	desc.send_void("setDepthWriteEnabled:", state.depth_write && state.depth_test);
	if (state.stencil_test)
	{
		ObjId front = objc::@class("MTLStencilDescriptor").send(ObjId, "new");
		ObjId back = objc::@class("MTLStencilDescriptor").send(ObjId, "new");
		stencil_descriptor(front, state.front, state);
		stencil_descriptor(back, state.back, state);
		desc.send_void("setFrontFaceStencil:", front);
		desc.send_void("setBackFaceStencil:", back);
		front.release();
		back.release();
	}
	ObjId result = pool.device.metal.send(ObjId, "newDepthStencilStateWithDescriptor:", desc);
	desc.release();
	pool.depth_states.push({ .desc = state, .state = result });
	return result;
}

<*
 Pools retain command storage until destruction. Use one pool per worker and in flight frame. Buffers from a pool must be submitted
 to the queue family of `queue_index`.
*>
fn CommandPool*? Device.create_command_pool(&self, uint queue_index = 0) => objc::@autoreleasepool()
{
	assert(queue_index < self.caps.queue_count);
	CommandPool* pool = mem::new(CommandPool, { .device = self, .kind = self.queues[queue_index].kind });
	pool.depth_states.init(mem);
	defer catch pool.destroy();
	if (!depth_state(pool, {})) return fail(CREATION_FAILED, "command pool depth state")~;
	// Every depth test: each compare function, with and without writes. DEFAULT is LESS_EQUAL, which is among them.
	for (uint compare = 1; compare < CompareOp::values.len; compare++)
	{
		for (uint write = 0; write < 2; write++)
		{
			if (!depth_state(pool, { .depth_test = true, .depth_write = write != 0, .depth_compare = (CompareOp)compare })) return fail(CREATION_FAILED, "command pool depth state")~;
		}
	}
	assert(pool.depth_states.size == 1 + 2 * (CompareOp::values.len - 1), "equivalent depth states are one object");
	return pool;
}

<*
 Reset only after every submitted buffer from this pool has completed; unsubmitted buffers are discarded. Invalidates all the
 CommandBuffer pointers it returned.
*>
fn void CommandPool.reset(&self) => objc::@autoreleasepool()
{
	for (CommandBuffer* commands = self.first; commands; commands = commands.next)
	{
		assert(!commands.acquired && (!commands.retirement || commands.retirement.send(ulong, "signaledValue") >= commands.retirement_value));
		if (commands.render) commands.end_render_pass();
		end_compute(commands);
		if (commands.recording) commands.commands.send_void("endCommandBuffer");
		commands.recording = false;
		commands.ended = false;
		commands.retirement = NIL;
		commands.retirement_value = 0;
		for (NativeCommandBuffer* native = commands.native_buffers; native; native = native.next) native.allocator.send_void("reset");
	}
	self.next_buffer = self.first;
}

fn void CommandPool.destroy(&self) => objc::@autoreleasepool()
{
	self.reset();
	for (CommandBuffer* commands = self.first; commands;)
	{
		CommandBuffer* next = commands.next;
		destroy_context(self.device, commands);
		commands = next;
	}
	foreach (&entry : self.depth_states) entry.state.release();
	self.depth_states.free();
	mem::free(self);
}

fn CommandBuffer*? CommandPool.begin_commands(&self) => objc::@autoreleasepool()
{
	CommandBuffer* context = self.next_buffer;
	if (context)
	{
		self.next_buffer = context.next;
	}
	else
	{
		context = create_context(self)!;
		CommandBuffer** link = self.last ? &self.last.next : &self.first;
		*link = context;
		self.last = context;
	}
	assert(!context.recording && !context.ended);
	context.native = context.native_buffers;
	context.commands = context.native.buffer;
	context.native_count = 1;
	context.commands.send_void("beginCommandBufferWithAllocator:", context.native.allocator);
	context.arguments.send_void("setAddress:atIndex:", (ulong)0, (NSUInteger)0);
	context.arguments.send_void("setAddress:atIndex:", (ulong)0, (NSUInteger)1);
	context.arguments.send_void("setAddress:atIndex:", (ulong)0, (NSUInteger)2);
	context.timestamp_count = 0;
	context.pso = null;
	context.recording = true;
	context.render_continuation = false;
	return context;
}

fn void CommandBuffer.end(&self) => objc::@autoreleasepool()
{
	assert(self.recording && !self.render);
	end_compute(self);
	self.commands.send_void("endCommandBuffer");
	self.recording = false;
	self.ended = true;
}

<*
 Submits ended buffers (any subset, each exactly once before the pool reset) to the queue and signals `desc.completion`.
 Order completion semaphore signal values across queues.
*>
fn void Device.submit(&self, SubmitDesc desc, uint queue_index = 0) => objc::@autoreleasepool()
{
	assert(queue_index < self.caps.queue_count && desc.completion.semaphore && desc.completion.semaphore.device == self);
	Queue* queue = &self.queues[queue_index];
	queue.submission.clear();
	foreach (commands : desc.commands)
	{
		assert(commands.device == self && commands.pool.kind == queue.kind && commands.ended && !commands.retirement);
		commands.retirement = queue.completion;
		commands.retirement_value = queue.submitted_value + 1;
		NativeCommandBuffer* native = commands.native_buffers;
		for (uint j = 0; j < commands.native_count; j++, native = native.next) queue.submission.push(native.buffer);
	}
	foreach (wait : desc.waits)
	{
		assert(wait.semaphore && wait.semaphore.device == self);
		queue.queue.send_void("waitForEvent:value:", wait.semaphore.event, wait.value);
	}
	// MetalTools enumerates residency on the CPU during commit without synchronizing concurrent updates.
	if (self.shader_validation) mtl::os_unfair_lock_lock(&self.residency_lock);
	if (queue.submission.size) queue.queue.send_void("commit:count:", queue.submission.entries, (NSUInteger)queue.submission.size);
	if (self.shader_validation) mtl::os_unfair_lock_unlock(&self.residency_lock);
	queue.queue.send_void("signalEvent:value:", queue.completion, ++queue.submitted_value);
	queue.queue.send_void("signalEvent:value:", desc.completion.semaphore.event, desc.completion.value);
}

// ---- Swapchain ------------------------------------------------------------------------------------------------------------------

<*
 Acquires the next drawable outside a render pass. The result is empty while the drawable extent is zero. A non empty acquire
 must be submitted with `submit_and_present`, on queue zero, with this buffer among the commands.
*>
fn SwapchainFrame CommandBuffer.acquire(&self) => objc::@autoreleasepool()
{
	Device* device = self.device;
	assert(self.recording && !self.render && self.pool.kind == GENERAL && device.layer && !device.drawable);
	Uint2 size = device.drawable_extent();
	if (!size.x || !size.y) return {};
	device.drawable = device.layer.send(ObjId, "nextDrawable").retain();
	if (!device.drawable) return {};
	device.acquired = self;
	self.acquired = true;
	ObjId texture = device.drawable.send(ObjId, "texture");
	device.drawable_view = { .texture = texture };
	return { .render_view = &device.drawable_view, .extent = { (uint)texture.send(NSUInteger, "width"), (uint)texture.send(NSUInteger, "height") } };
}

fn void Device.submit_and_present(&self, SubmitDesc desc) => objc::@autoreleasepool()
{
	assert(self.drawable && self.acquired);
	bool found;
	foreach (commands : desc.commands) found |= commands == self.acquired;
	assert(found);
	self.queues[0].queue.send_void("waitForDrawable:", self.drawable);
	self.submit(desc, 0);
	if (self.presentation_queue)
	{
		// Steam's macOS overlay hooks command-buffer presentation, including frames rendered with Metal 4: a classic command buffer waits for
		// the render on the GPU and presents.
		ObjId presentation = self.presentation_queue.send(ObjId, "commandBuffer");
		presentation.send_void("encodeWaitForEvent:value:", self.queues[0].completion, self.queues[0].submitted_value);
		presentation.send_void("presentDrawable:", self.drawable);
		presentation.send_void("commit");
	}
	else
	{
		self.queues[0].queue.send_void("signalDrawable:", self.drawable);
		self.drawable.send_void("present");
	}
	self.drawable.release();
	self.acquired.acquired = false;
	self.acquired = null;
	self.drawable = NIL;
	self.drawable_view = {};
}

// ---- Recording ------------------------------------------------------------------------------------------------------------------

fn void CommandBuffer.set_texture_descriptor_heap(&self, TextureDescriptorHeap* heap) => objc::@autoreleasepool()
{
	assert(self.recording && heap.device == self.device);
	self.arguments.send_void("setAddress:atIndex:", heap.base.send(ulong, "gpuAddress"), (NSUInteger)1);
}

fn void CommandBuffer.set_sampler_descriptor_heap(&self, SamplerDescriptorHeap* heap) => objc::@autoreleasepool()
{
	assert(self.recording && heap.device == self.device);
	self.arguments.send_void("setAddress:atIndex:", heap.buffer.send(ulong, "gpuAddress"), (NSUInteger)2);
}

fn void CommandBuffer.copy_memory(&self, GpuRange source, GpuRange destination) => objc::@autoreleasepool()
{
	assert(source.size <= destination.size);
	ulong source_offset;
	ulong destination_offset;
	ObjId source_buffer = resolve_buffer(self.device, source, &source_offset);
	ObjId destination_buffer = resolve_buffer(self.device, { destination.gpu, source.size }, &destination_offset);
	compute_encoder(self).send_void("copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:",
		source_buffer, (NSUInteger)source_offset, destination_buffer, (NSUInteger)destination_offset, (NSUInteger)source.size);
}

struct TextureCopy @local
{
	mtl::Origin origin;
	mtl::Size extent;
	ulong row_pitch;
	ulong image_pitch;
	ulong slice_pitch;
	ulong size;
	uint slices;
}

fn TextureCopy texture_copy(Texture* texture, TextureCopyDesc desc) @local
{
	TextureFormatInfo format = get_texture_format_info(texture.desc.format);
	uint width = texture.desc.extent.x >> desc.mip_level;
	uint height = texture.desc.extent.y >> desc.mip_level;
	uint depth = texture.desc.extent.z >> desc.mip_level;
	TextureCopy copy = {
		.origin = { desc.offset.x, desc.offset.y, desc.offset.z },
		.extent = {
			(NSUInteger)(desc.extent.x ?: ((width ?: 1) - desc.offset.x)),
			(NSUInteger)(desc.extent.y ?: ((height ?: 1) - desc.offset.y)),
			(NSUInteger)(desc.extent.z ?: ((depth ?: 1) - desc.offset.z)),
		},
		.slices = desc.slice_count ?: (texture.desc.layer_count - desc.base_slice),
	};
	ulong row_bytes = ((copy.extent.width + format.block_extent.x - 1) / format.block_extent.x) * format.bytes_per_block;
	ulong rows = (copy.extent.height + format.block_extent.y - 1) / format.block_extent.y;
	copy.row_pitch = desc.row_pitch_bytes ?: row_bytes;
	copy.image_pitch = copy.row_pitch * rows;
	copy.slice_pitch = desc.slice_pitch_bytes ?: (copy.image_pitch * copy.extent.depth);
	if (texture.desc.type == THREE_D && desc.slice_pitch_bytes) copy.image_pitch = desc.slice_pitch_bytes;
	copy.size = copy.slice_pitch * (ulong)(copy.slices - 1) + copy.image_pitch * (ulong)(copy.extent.depth - 1) + copy.row_pitch * (rows - 1) + row_bytes;
	return copy;
}

fn void CommandBuffer.copy_memory_to_texture(&self, GpuRange source, Texture* destination, TextureCopyDesc desc = {}) => objc::@autoreleasepool()
{
	TextureCopy copy = texture_copy(destination, desc);
	assert(source.size >= copy.size);
	ulong offset;
	ObjId buffer = resolve_buffer(self.device, { source.gpu, copy.size }, &offset);
	for (uint slice = 0; slice < copy.slices; slice++)
	{
		compute_encoder(self).send_void("copyFromBuffer:sourceOffset:sourceBytesPerRow:sourceBytesPerImage:sourceSize:toTexture:destinationSlice:destinationLevel:destinationOrigin:",
			buffer, (NSUInteger)(offset + slice * copy.slice_pitch), (NSUInteger)copy.row_pitch, (NSUInteger)copy.image_pitch, copy.extent,
			destination.texture, (NSUInteger)(desc.base_slice + slice), (NSUInteger)desc.mip_level, copy.origin);
	}
}

fn void CommandBuffer.copy_texture_to_memory(&self, Texture* source, GpuRange destination, TextureCopyDesc desc = {}) => objc::@autoreleasepool()
{
	TextureCopy copy = texture_copy(source, desc);
	assert(destination.size >= copy.size);
	ulong offset;
	ObjId buffer = resolve_buffer(self.device, { destination.gpu, copy.size }, &offset);
	for (uint slice = 0; slice < copy.slices; slice++)
	{
		compute_encoder(self).send_void("copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toBuffer:destinationOffset:destinationBytesPerRow:destinationBytesPerImage:",
			source.texture, (NSUInteger)(desc.base_slice + slice), (NSUInteger)desc.mip_level, copy.origin, copy.extent,
			buffer, (NSUInteger)(offset + slice * copy.slice_pitch), (NSUInteger)copy.row_pitch, (NSUInteger)copy.image_pitch);
	}
}

<*
 A stage and access barrier, for work between render passes (not inside one). Only whether the earlier access writes matters on
 Metal, the destination access is not needed.
*>
fn void CommandBuffer.barrier(&self, Stage before, Access before_access, Stage after, Access after_access) => objc::@autoreleasepool()
{
	assert(!self.render);
	NSUInteger source = barrier_stages(before, true);
	NSUInteger destination = barrier_stages(after, false);
	// Shared event completion provides host visibility; a host only destination blocks no future GPU stages.
	if (!source || !destination) return;
	Access writes = { .transfer_write = true, .shader_write = true, .color_write = true, .depth_stencil_write = true };
	NSUInteger visibility = ((ulong)before_access & (ulong)writes) ? mtl::VisibilityOptions.DEVICE : mtl::VisibilityOptions.NONE;
	compute_encoder(self).send_void("barrierAfterStages:beforeQueueStages:visibilityOptions:", source, destination, visibility);
	end_compute(self);
}

<*
 Writes a GPU timestamp (in ticks of DeviceCaps.timestamp_period_ns) to `cpu_destination` once `read_timestamps` retrieves it.
 At most DeviceDesc.timestamp_query_count per command buffer. Ignored when timestamps are off. The destination must stay valid
 and distinct until then.
*>
fn void CommandBuffer.write_timestamp(&self, ulong* cpu_destination, Stage stage = { .all_commands = true }) => objc::@autoreleasepool()
{
	if (!self.device.timestamp_query_count) return;
	assert(self.recording && cpu_destination && self.timestamp_count < self.device.timestamp_query_count);
	uint index = self.timestamp_count++;
	self.timestamps[index].destination = cpu_destination;
	if (self.render)
	{
		NSUInteger after = mtl::RenderStages.FRAGMENT;
		if (stage.vertex || stage.index_input) after = mtl::RenderStages.VERTEX;
		if (stage.task) after = mtl::RenderStages.OBJECT;
		if (stage.mesh) after = mtl::RenderStages.MESH;
		self.render.send_void("writeTimestampWithGranularity:afterStage:intoHeap:atIndex:",
			mtl::TimestampGranularity.PRECISE, after, self.timestamps[index].heap, (NSUInteger)self.timestamps[index].index);
	}
	else
	{
		end_compute(self);
		native_commands(self).send_void("writeTimestampIntoHeap:atIndex:", self.timestamps[index].heap, (NSUInteger)self.timestamps[index].index);
	}
}

<*
 Retrieves the timestamps of submitted buffers once every submission from this pool completed. Does not wait. Call before the
 pool is reset, which discards unread results.
*>
fn void CommandPool.read_timestamps(&self) => objc::@autoreleasepool()
{
	for (CommandBuffer* commands = self.first; commands; commands = commands.next)
	{
		if (!commands.retirement || !commands.timestamp_count) continue;
		assert(commands.retirement.send(ulong, "signaledValue") >= commands.retirement_value);
		for (uint i = 0; i < commands.timestamp_count;)
		{
			TimestampSlot first = commands.timestamps[i];
			uint count = 1;
			while (i + count < commands.timestamp_count && commands.timestamps[i + count].heap == first.heap && commands.timestamps[i + count].index == first.index + count) count++;
			ObjId data = first.heap.send(ObjId, "resolveCounterRange:", mtl::range(first.index, count));
			assert(data.send(NSUInteger, "length") == count * mtl::TimestampHeapEntry::size);
			mtl::TimestampHeapEntry* values = data.send(mtl::TimestampHeapEntry*, "bytes");
			for (uint j = 0; j < count; j++) *commands.timestamps[i + j].destination = values[j].timestamp;
			i += count;
		}
		commands.timestamp_count = 0;
	}
}

// ---- Render passes --------------------------------------------------------------------------------------------------------------

fn NSUInteger load_action(LoadOp load) @local
{
	switch (load)
	{
		case LOAD: return mtl::LoadAction.LOAD;
		case CLEAR: return mtl::LoadAction.CLEAR;
		case DISCARD: return mtl::LoadAction.DONT_CARE;
	}
}

fn void render_attachment(ObjId attachment, RenderView* view, LoadOp load, StoreOp store) @local
{
	attachment.send_void("setTexture:", view ? view.texture : NIL);
	if (!view) return;
	bool volume = view.texture.send(NSUInteger, "textureType") == mtl::TextureType.D3;
	attachment.send_void("setLevel:", (NSUInteger)view.mip);
	attachment.send_void("setSlice:", (NSUInteger)(volume ? 0 : view.slice));
	attachment.send_void("setDepthPlane:", (NSUInteger)(volume ? view.slice : 0));
	attachment.send_void("setLoadAction:", load_action(load));
	attachment.send_void("setStoreAction:", store == STORE ? mtl::StoreAction.STORE : mtl::StoreAction.DONT_CARE);
}

<*
 Starts a render pass. Every pass needs at least one attachment to get its area from. It begins with a full area viewport and
 scissor, with depth and stencil off and with no pipeline: bind one inside the pass, also in a resumed segment. Suspended passes are resumed by the following segment (see RenderingFlags): submit the
 whole chain in order in one batch.
*>
fn void CommandBuffer.begin_render_pass(&self, RenderingDesc desc, RenderingFlags flags = {}) => objc::@autoreleasepool()
{
	assert(self.recording && !self.render && self.pool.kind == GENERAL && desc.colors.len <= 8);
	end_compute(self);
	RenderView* area = desc.colors.len ? desc.colors[0].render_view : (desc.depth.render_view ?: desc.stencil.render_view);
	ObjId color_attachments = self.pass.send(ObjId, "colorAttachments");
	for (uint i = 0; i < 8; i++)
	{
		ObjId attachment = color_attachments.send(ObjId, "objectAtIndexedSubscript:", (NSUInteger)i);
		if (i < desc.colors.len)
		{
			ColorAttachment color = desc.colors[i];
			render_attachment(attachment, color.render_view, color.load, color.store);
			attachment.send_void("setClearColor:", (mtl::ClearColor){ color.clear.x, color.clear.y, color.clear.z, color.clear.w });
			if (!area) area = color.render_view;
		}
		else
		{
			attachment.send_void("setTexture:", NIL);
		}
	}
	ObjId depth = self.pass.send(ObjId, "depthAttachment");
	render_attachment(depth, desc.depth.render_view, desc.depth.load, desc.depth.store);
	depth.send_void("setClearDepth:", (double)desc.depth.clear);
	ObjId stencil = self.pass.send(ObjId, "stencilAttachment");
	render_attachment(stencil, desc.stencil.render_view, desc.stencil.load, desc.stencil.store);
	stencil.send_void("setClearStencil:", (uint)desc.stencil.clear);
	assert(area);
	NSUInteger width = area.texture.send(NSUInteger, "width") >> area.mip;
	NSUInteger height = area.texture.send(NSUInteger, "height") >> area.mip;
	self.pass.send_void("setRenderTargetWidth:", width ?: 1);
	self.pass.send_void("setRenderTargetHeight:", height ?: 1);
	self.render_continuation = (uint)flags != 0;
	NSUInteger options = flags.suspending ? mtl::RenderEncoderOptions.SUSPENDING : mtl::RenderEncoderOptions.NONE;
	if (flags.resuming) options |= mtl::RenderEncoderOptions.RESUMING;
	self.render = native_commands(self).send(ObjId, "renderCommandEncoderWithDescriptor:options:", self.pass, options).retain();
	self.render.send_void("setArgumentTable:atStages:", self.arguments, RENDER_STAGES);
	self.render.send_void("setFrontFacingWinding:", mtl::Winding.COUNTER_CLOCKWISE);
	// Metal's clip space is y up with the origin of the framebuffer at the top, so the viewport is flipped to match Vulkan's convention.
	self.render.send_void("setViewport:", (mtl::Viewport){ 0, (double)(height ?: 1), (double)(width ?: 1), -(double)(height ?: 1), 0, 1 });
	self.render.send_void("setScissorRect:", (mtl::ScissorRect){ 0, 0, width ?: 1, height ?: 1 });
	self.render.send_void("setDepthStencilState:", depth_state(self.pool, {}));
	self.pso = null;   // A render pass starts with no pipeline: bind one inside it, also in a resumed segment.
}

fn void CommandBuffer.end_render_pass(&self) => objc::@autoreleasepool()
{
	assert(self.render);
	self.render.send_void("endEncoding");
	self.render.release();
	self.render = NIL;
	self.pso = null;   // Rebind a compute pipeline before dispatching after a render pass.
	if (self.render_continuation)
	{
		self.commands.send_void("endCommandBuffer");
		self.commands = NIL;
	}
	ObjId color_attachments = self.pass.send(ObjId, "colorAttachments");
	for (uint i = 0; i < 8; i++) color_attachments.send(ObjId, "objectAtIndexedSubscript:", (NSUInteger)i).send_void("setTexture:", NIL);
	self.pass.send(ObjId, "depthAttachment").send_void("setTexture:", NIL);
	self.pass.send(ObjId, "stencilAttachment").send_void("setTexture:", NIL);
}

fn void CommandBuffer.set_viewport(&self, Viewport viewport) => objc::@autoreleasepool()
{
	assert(self.recording);
	if (!self.render) return;
	self.render.send_void("setViewport:", (mtl::Viewport){ viewport.x, (double)viewport.y + viewport.height, viewport.width, -(double)viewport.height, viewport.min_depth, viewport.max_depth });
}

fn void CommandBuffer.set_scissor(&self, Scissor scissor) => objc::@autoreleasepool()
{
	assert(self.recording);
	if (!self.render) return;
	self.render.send_void("setScissorRect:", (mtl::ScissorRect){ (NSUInteger)scissor.x, (NSUInteger)scissor.y, scissor.width, scissor.height });
}

fn void CommandBuffer.set_depth_stencil(&self, DepthStencilState state) => objc::@autoreleasepool()
{
	assert(self.recording);
	ObjId depth = depth_state(self.pool, state);
	if (!self.render) return;
	self.render.send_void("setDepthStencilState:", depth);
	self.render.send_void("setStencilFrontReferenceValue:backReferenceValue:", (uint)state.front.reference, (uint)state.back.reference);
}

fn void CommandBuffer.bind_pso(&self, Pso* pso) => objc::@autoreleasepool()
{
	assert(self.recording && pso.device == self.device);
	self.pso = pso;
	if (pso.compute)
	{
		assert(!self.render);
		if (self.compute)
		{
			self.compute.send_void("setComputePipelineState:", pso.compute);
		}
		else
		{
			compute_encoder(self);
		}
	}
	else
	{
		assert(self.render, "graphics and mesh pipelines are bound inside a render pass");
		self.render.send_void("setRenderPipelineState:", pso.render);
		self.render.send_void("setCullMode:", pso.rasterization.cull == NONE ? mtl::CullMode.NONE : pso.rasterization.cull == CLOCKWISE ? mtl::CullMode.BACK : mtl::CullMode.FRONT);
		self.render.send_void("setDepthBias:slopeScale:clamp:", pso.rasterization.depth_bias_constant, pso.rasterization.depth_bias_slope, pso.rasterization.depth_bias_clamp);
	}
}

// ---- Draws and dispatches -------------------------------------------------------------------------------------------------------

// `root` is a 16 byte aligned GPU address, or null for shaders without root data. The application keeps its storage alive until the
// GPU is done and finishes writing it before submission. It is read by the shader at buffer(0).
fn void root_data(CommandBuffer* commands, void* root) @local
{
	if (!root) return;
	assert(((uptr)root & 15) == 0);
	commands.arguments.send_void("setAddress:atIndex:", (ulong)(uptr)root, (NSUInteger)0);
}

fn void CommandBuffer.draw(&self, void* root, uint vertex_count, uint instance_count = 1, uint first_vertex = 0, uint first_instance = 0) => objc::@autoreleasepool()
{
	assert(self.render && self.pso && !self.pso.mesh);
	root_data(self, root);
	self.render.send_void("drawPrimitives:vertexStart:vertexCount:instanceCount:baseInstance:",
		mtl::PrimitiveType.TRIANGLE, (NSUInteger)first_vertex, (NSUInteger)vertex_count, (NSUInteger)instance_count, (NSUInteger)first_instance);
}

fn void CommandBuffer.draw_indexed(&self, void* root, GpuRange indices, IndexType type, uint index_count, uint instance_count = 1,
	uint first_index = 0, int vertex_offset = 0, uint first_instance = 0) => objc::@autoreleasepool()
{
	assert(self.render && self.pso && !self.pso.mesh);
	root_data(self, root);
	ulong offset = (ulong)first_index * (type == UINT16 ? (ulong)2 : (ulong)4);
	self.render.send_void("drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferLength:instanceCount:baseVertex:baseInstance:",
		mtl::PrimitiveType.TRIANGLE, (NSUInteger)index_count, type == UINT16 ? mtl::IndexType.UINT16 : mtl::IndexType.UINT32,
		(ulong)(uptr)indices.gpu + offset, (NSUInteger)(indices.size - offset), (NSUInteger)instance_count, (NSInteger)vertex_offset, (NSUInteger)first_instance);
}

fn void CommandBuffer.draw_indirect(&self, void* root, GpuRange arguments, uint draw_count = 1, uint stride = 0) => objc::@autoreleasepool()
{
	assert(self.render && self.pso && !self.pso.mesh);
	root_data(self, root);
	if (!stride) stride = mtl::DrawPrimitivesIndirectArguments::size;
	for (uint i = 0; i < draw_count; i++)
	{
		self.render.send_void("drawPrimitives:indirectBuffer:", mtl::PrimitiveType.TRIANGLE, (ulong)(uptr)arguments.gpu + (ulong)i * stride);
	}
}

fn void CommandBuffer.draw_indexed_indirect(&self, void* root, GpuRange indices, IndexType type, GpuRange arguments, uint draw_count = 1, uint stride = 0) => objc::@autoreleasepool()
{
	assert(self.render && self.pso && !self.pso.mesh);
	root_data(self, root);
	if (!stride) stride = mtl::DrawIndexedPrimitivesIndirectArguments::size;
	for (uint i = 0; i < draw_count; i++)
	{
		self.render.send_void("drawIndexedPrimitives:indexType:indexBuffer:indexBufferLength:indirectBuffer:",
			mtl::PrimitiveType.TRIANGLE, type == UINT16 ? mtl::IndexType.UINT16 : mtl::IndexType.UINT32,
			(ulong)(uptr)indices.gpu, (NSUInteger)indices.size, (ulong)(uptr)arguments.gpu + (ulong)i * stride);
	}
}

fn void CommandBuffer.dispatch(&self, void* root, Uint3 group_count) => objc::@autoreleasepool()
{
	assert(self.pso && self.pso.compute);
	root_data(self, root);
	compute_encoder(self).send_void("dispatchThreadgroups:threadsPerThreadgroup:", metal_size(group_count), self.pso.threads);
}

fn void CommandBuffer.dispatch_indirect(&self, void* root, GpuRange arguments) => objc::@autoreleasepool()
{
	assert(self.pso && self.pso.compute);
	root_data(self, root);
	compute_encoder(self).send_void("dispatchThreadgroupsWithIndirectBuffer:threadsPerThreadgroup:", (ulong)(uptr)arguments.gpu, self.pso.threads);
}

fn void CommandBuffer.draw_meshlets(&self, void* root, Uint3 group_count) => objc::@autoreleasepool()
{
	assert(self.render && self.pso && self.pso.mesh);
	root_data(self, root);
	self.render.send_void("drawMeshThreadgroups:threadsPerObjectThreadgroup:threadsPerMeshThreadgroup:", metal_size(group_count), self.pso.object_threads, self.pso.threads);
}

<*
 Needs DeviceCaps.indirect_mesh_draw (Apple9 and later).
*>
fn void CommandBuffer.draw_meshlets_indirect(&self, void* root, GpuRange arguments, uint draw_count = 1, uint stride = 0) => objc::@autoreleasepool()
{
	assert(self.render && self.pso && self.pso.mesh);
	assert(self.device.caps.indirect_mesh_draw);
	root_data(self, root);
	if (!stride) stride = mtl::DispatchThreadgroupsIndirectArguments::size;
	for (uint i = 0; i < draw_count; i++)
	{
		self.render.send_void("drawMeshThreadgroupsWithIndirectBuffer:threadsPerObjectThreadgroup:threadsPerMeshThreadgroup:",
			(ulong)(uptr)arguments.gpu + (ulong)i * stride, self.pso.object_threads, self.pso.threads);
	}
}
