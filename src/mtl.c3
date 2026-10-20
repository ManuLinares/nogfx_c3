module nogfx::mtl @feat(DARWIN & !NOGFX_VULKAN) @link("Metal.framework");
import std::os::darwin;

// The parts of Metal and Metal 4 that the backend uses, as Objective-C is reached through objc::send. The constants were generated from the
// macOS SDK headers (MTLStages, MTLLoadAction, ...) and the names drop the "MTL" prefix and the type name: MTLLoadActionClear is LoadAction.CLEAR.

// MTLStages
constset Stages : inline NSUInteger
{
	VERTEX = 1,
	FRAGMENT = 2,
	TILE = 4,
	OBJECT = 8,
	MESH = 16,
	RESOURCE_STATE = 0x4000000,
	DISPATCH = 0x8000000,
	BLIT = 0x10000000,
	ACCELERATION_STRUCTURE = 0x20000000,
	MACHINE_LEARNING = 0x40000000,
	ALL = 0x7fffffffffffffff,
}

// MTLRenderStages
constset RenderStages : inline NSUInteger
{
	VERTEX = 1,
	FRAGMENT = 2,
	TILE = 4,
	OBJECT = 8,
	MESH = 16,
}

// MTLLoadAction
constset LoadAction : inline NSUInteger
{
	DONT_CARE = 0,
	LOAD = 1,
	CLEAR = 2,
}

// MTLStoreAction
constset StoreAction : inline NSUInteger
{
	DONT_CARE = 0,
	STORE = 1,
}

// MTLTextureType
constset TextureType : inline NSUInteger
{
	D1 = 0,
	D2 = 2,
	D2_ARRAY = 3,
	CUBE = 5,
	CUBE_ARRAY = 6,
	D3 = 7,
}

// MTLTextureUsage
constset TextureUsage : inline NSUInteger
{
	UNKNOWN = 0,
	SHADER_READ = 1,
	SHADER_WRITE = 2,
	RENDER_TARGET = 4,
	PIXEL_FORMAT_VIEW = 16,
}

// MTLResourceOptions
constset ResourceOptions : inline NSUInteger
{
	CPU_CACHE_MODE_WRITE_COMBINED = 1,
	STORAGE_MODE_SHARED = 0,
	STORAGE_MODE_PRIVATE = 0x20,
	HAZARD_TRACKING_MODE_UNTRACKED = 0x100,
}

// MTLStorageMode
constset StorageMode : inline NSUInteger
{
	SHARED = 0,
	PRIVATE = 2,
}

// MTLHazardTrackingMode
constset HazardTrackingMode : inline NSUInteger
{
	UNTRACKED = 1,
}

// MTLHeapType
constset HeapType : inline NSInteger
{
	PLACEMENT = 1,
}

// MTLSamplerAddressMode
constset SamplerAddressMode : inline NSUInteger
{
	CLAMP_TO_EDGE = 0,
	REPEAT = 2,
	MIRROR_REPEAT = 3,
}

// MTLSamplerMinMagFilter
constset SamplerMinMagFilter : inline NSUInteger
{
	NEAREST = 0,
	LINEAR = 1,
}

// MTLSamplerMipFilter
constset SamplerMipFilter : inline NSUInteger
{
	NEAREST = 1,
	LINEAR = 2,
}

// MTLCompareFunction
constset CompareFunction : inline NSUInteger
{
	NEVER = 0,
	LESS = 1,
	EQUAL = 2,
	LESS_EQUAL = 3,
	GREATER = 4,
	NOT_EQUAL = 5,
	GREATER_EQUAL = 6,
	ALWAYS = 7,
}

// MTLBlendFactor
constset BlendFactor : inline NSUInteger
{
	ZERO = 0,
	ONE = 1,
	SOURCE_COLOR = 2,
	ONE_MINUS_SOURCE_COLOR = 3,
	SOURCE_ALPHA = 4,
	ONE_MINUS_SOURCE_ALPHA = 5,
	DESTINATION_COLOR = 6,
	ONE_MINUS_DESTINATION_COLOR = 7,
	DESTINATION_ALPHA = 8,
	ONE_MINUS_DESTINATION_ALPHA = 9,
	SOURCE_ALPHA_SATURATED = 10,
}

// MTLBlendOperation
constset BlendOperation : inline NSUInteger
{
	ADD = 0,
	SUBTRACT = 1,
	REVERSE_SUBTRACT = 2,
	MIN = 3,
	MAX = 4,
}

// MTLColorWriteMask
constset ColorWriteMask : inline NSUInteger
{
	NONE = 0,
	RED = 8,
	GREEN = 4,
	BLUE = 2,
	ALPHA = 1,
	ALL = 15,
}

// MTLStencilOperation
constset StencilOperation : inline NSUInteger
{
	KEEP = 0,
	ZERO = 1,
	REPLACE = 2,
	INCREMENT_CLAMP = 3,
	DECREMENT_CLAMP = 4,
	INVERT = 5,
	INCREMENT_WRAP = 6,
	DECREMENT_WRAP = 7,
}

// MTLIndexType
constset IndexType : inline NSUInteger
{
	UINT16 = 0,
	UINT32 = 1,
}

// MTLPrimitiveType
constset PrimitiveType : inline NSUInteger
{
	TRIANGLE = 3,
}

// MTLPrimitiveTopologyClass
constset PrimitiveTopologyClass : inline NSUInteger
{
	TRIANGLE = 3,
}

// MTLCullMode
constset CullMode : inline NSUInteger
{
	NONE = 0,
	FRONT = 1,
	BACK = 2,
}

// MTLWinding
constset Winding : inline NSUInteger
{
	CLOCKWISE = 0,
	COUNTER_CLOCKWISE = 1,
}

// MTL4BlendState
constset BlendState : inline NSInteger
{
	DISABLED = 0,
	ENABLED = 1,
}

// MTL4VisibilityOptions
constset VisibilityOptions : inline NSUInteger
{
	NONE = 0,
	DEVICE = 1,
	RESOURCE_ALIAS = 2,
}

// MTL4RenderEncoderOptions
constset RenderEncoderOptions : inline NSUInteger
{
	NONE = 0,
	SUSPENDING = 1,
	RESUMING = 2,
}

// MTL4CounterHeapType
constset CounterHeapType : inline NSInteger
{
	TIMESTAMP = 1,
}

// MTL4TimestampGranularity
constset TimestampGranularity : inline NSInteger
{
	PRECISE = 1,
}

// MTLGPUFamily
constset GPUFamily : inline NSInteger
{
	APPLE7 = 0x3ef,
	APPLE9 = 0x3f1,
	METAL4 = 0x138a,
}

// Value types passed to and from Metal. The layouts match the SDK: sizes are 24, 24, 48, 32, 32, 16, 8 and 16 bytes.
struct CGSize { double width, height; }
struct Size { NSUInteger width, height, depth; }
struct Origin { NSUInteger x, y, z; }
struct Viewport { double x, y, width, height, znear, zfar; }
struct ScissorRect { NSUInteger x, y, width, height; }
struct ClearColor { double red, green, blue, alpha; }
struct Range { NSUInteger location, length; }
struct ResourceID { ulong impl; }
fn Range range(uint location, uint length) => { location, length };
struct SizeAndAlign { NSUInteger size, align; }
struct TimestampHeapEntry { ulong timestamp; }
struct DrawPrimitivesIndirectArguments { uint vertex_count, instance_count, vertex_start, base_instance; }
struct DrawIndexedPrimitivesIndirectArguments { uint index_count, instance_count, index_start; int base_vertex; uint base_instance; }
struct DispatchThreadgroupsIndirectArguments { uint[3] threadgroups_per_grid; }

// libdispatch, mach and os: what the backend needs besides Objective-C.
extern fn void* dispatch_data_create(void* buffer, usz size, void* queue, void* destructor);
extern fn void* dispatch_queue_attr_make_with_autorelease_frequency(void* attr, ulong frequency);
extern fn void* dispatch_queue_create(ZString label, void* attr);
extern fn void dispatch_release(void* object);
extern fn void os_unfair_lock_lock(uint* lock);
extern fn void os_unfair_lock_unlock(uint* lock);
struct TimebaseInfo { uint numer, denom; }
extern fn int mach_timebase_info(TimebaseInfo* info);

const ulong DISPATCH_AUTORELEASE_FREQUENCY_WORK_ITEM = 1;
