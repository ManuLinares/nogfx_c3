module nogfx @feat(!DARWIN, NOGFX_VULKAN);
import std::io, libc;

// A stub Vulkan driver, for running the backend where there is no capable GPU: build it as the dynamic library `libvulkan` and put it first in
// the library path, see tests/run_fake.sh. Nothing executes on a GPU: memory is host memory (a buffer's device address is its host address),
// copies between memory and images run when they are recorded, and timeline semaphores are signaled by the submission that names them.
// Shaders do not run, so what they would write is not there. What it checks, and aborts on: the structure chains of pipeline creation
// (descriptor heap flag, mappings, SPIR-V alignment and magic), command buffer state (recording, rendering, pipelines), the acquire /
// render / present handshake of binary semaphores and fences, timeline waits for values that no submission signaled, and leaks at the end.
// The surface of a window is 640x480 unless FAKE_EXTENT=<width>x<height> says otherwise, and can change while the program runs.

struct FakeHeader { VkStructureType sType; void* pNext; }

struct FakeRecord
{
	ulong memory;          // buffers and images: the bound memory's host pointer
	ulong offset;
	ulong size;
	ulong value;           // semaphores: the timeline value
	bool timeline;
	bool signaled;         // fences
	bool recording;        // command buffers
	bool ended;
	bool submitted;        // Command buffers are one time submit.
	bool rendering;
	bool pipeline_bound;
	uint bind_point;
	uint width, height, depth, layers, format;   // images
	uint query_count;
}

const ulong FAKE_PAGE = 65536;

int fake_dummy;
bool fake_address_commands;   // Whether the device was created with VK_KHR_device_address_commands.
enum FakeKind : char { BUFFER, MEMORY, IMAGE, VIEW, SEMAPHORE, FENCE, POOL, PIPELINE, QUERY, SWAPCHAIN, SURFACE, COMMAND_BUFFER }
long[12] fake_live;
String[12] fake_kind_names = { "buffers", "memory", "images", "image views", "semaphores", "fences", "command pools", "pipelines", "query pools", "swapchains", "surfaces", "command buffers" };
fn void fake_count(FakeKind kind, long delta) @private { fake_live[kind.ordinal] += delta; if (fake_live[kind.ordinal] < 0) fake_fail(string::tformat("%s destroyed more often than created", fake_kind_names[kind.ordinal])); }
ulong fake_clock;
ulong fake_calls;

fn void fake_fail(String message) @private
{
	io::eprintfn("FAKE VULKAN: %s", message);
	libc::abort();
}

fn FakeRecord* record_of(ulong handle) @private => (FakeRecord*)(uptr)handle;

fn ulong new_record() @private => (ulong)(uptr)mem::new(FakeRecord);

fn FakeRecord* cb_of(VkCommandBuffer cb, String call) @private
{
	FakeRecord* record = (FakeRecord*)cb;
	if (!record.recording) fake_fail(string::tformat("%s on a command buffer that is not recording", call));
	return record;
}

// ---- Instance and physical device -----------------------------------------------------------------------------------------------

fn VkResult fake_enumerate_instance_version(uint* version) { *version = VK_API_VERSION_1_4; return VK_SUCCESS; }

fn VkResult fake_enumerate_instance_extensions(ZString layer, uint* count, VkExtensionProperties* props)
{
	ZString[5] names = { VK_KHR_SURFACE_EXTENSION_NAME, VK_KHR_WIN32_SURFACE_EXTENSION_NAME, VK_KHR_GET_SURFACE_CAPABILITIES_2_EXTENSION_NAME, VK_KHR_SURFACE_MAINTENANCE_1_EXTENSION_NAME, VK_EXT_SWAPCHAIN_COLOR_SPACE_EXTENSION_NAME };
	if (props)
	{
		for (uint i = 0; i < 5 && i < *count; i++)
		{
			props[i] = {};
			mem::copy(&props[i].extensionName[0], names[i], (sz)((ZString)names[i]).len());
		}
	}
	*count = 5;
	return VK_SUCCESS;
}

fn VkResult fake_enumerate_instance_layers(uint* count, VkLayerProperties* props) { *count = 0; return VK_SUCCESS; }

fn VkResult fake_create_instance(VkInstanceCreateInfo* info, VkAllocationCallbacks* allocator, VkInstance* instance)
{
	if (info.sType != VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO) fake_fail("vkCreateInstance: sType");
	if (info.pApplicationInfo.apiVersion < VK_API_VERSION_1_4) fake_fail("vkCreateInstance: apiVersion");
	*instance = (VkInstance)&fake_dummy;
	return VK_SUCCESS;
}

fn void fake_destroy_instance(VkInstance instance, VkAllocationCallbacks* allocator)
{
	bool leaked;
	for (usz i = 0; i < 12; i++)
	{
		if (fake_live[i] != 0 && i != FakeKind.COMMAND_BUFFER.ordinal)
		{
			io::eprintfn("FAKE VULKAN LEAK: %d %s still alive at vkDestroyInstance", fake_live[i], fake_kind_names[i]);
			leaked = true;
		}
	}
	if (!leaked) io::eprintfn("FAKE VULKAN: no object leaked");
}

fn VkResult fake_enumerate_physical_devices(VkInstance instance, uint* count, VkPhysicalDevice* devices)
{
	if (devices && *count >= 1) devices[0] = (VkPhysicalDevice)&fake_dummy;
	*count = 1;
	return VK_SUCCESS;
}

fn VkResult fake_enumerate_device_extensions(VkPhysicalDevice device, ZString layer, uint* count, VkExtensionProperties* props)
{
	// FAKE_NO_ADDRESS_COMMANDS=1 leaves VK_KHR_device_address_commands out, which makes the backend resolve addresses to buffers.
	bool no_address_commands = libc::getenv("FAKE_NO_ADDRESS_COMMANDS") != null;
	ZString[7] names = {
		VK_EXT_DESCRIPTOR_HEAP_EXTENSION_NAME, VK_KHR_SHADER_UNTYPED_POINTERS_EXTENSION_NAME,
		VK_EXT_MESH_SHADER_EXTENSION_NAME, VK_KHR_UNIFIED_IMAGE_LAYOUTS_EXTENSION_NAME, VK_KHR_SWAPCHAIN_EXTENSION_NAME, VK_KHR_SWAPCHAIN_MAINTENANCE_1_EXTENSION_NAME,
		VK_KHR_DEVICE_ADDRESS_COMMANDS_EXTENSION_NAME,
	};
	uint total = no_address_commands ? 6 : 7;
	if (props)
	{
		for (uint i = 0; i < total && i < *count; i++)
		{
			props[i] = {};
			usz length = (usz)((ZString)names[i]).len();
			mem::copy(&props[i].extensionName[0], names[i], (sz)length);
		}
	}
	*count = total;
	return VK_SUCCESS;
}

fn void fake_fill_all_bools(void* chain, usz size) @private
{
	uint* bools = (uint*)((char*)chain + 16);
	for (usz i = 0; i < (size - 16) / 4; i++) bools[i] = 1;
}

fn void fake_get_features2(VkPhysicalDevice device, VkPhysicalDeviceFeatures2* features)
{
	uint* core = (uint*)&features.features;
	for (usz i = 0; i < VkPhysicalDeviceFeatures::size / 4; i++) core[i] = 1;
	for (FakeHeader* header = (FakeHeader*)features.pNext; header; header = (FakeHeader*)header.pNext)
	{
		switch (header.sType)
		{
			case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES: fake_fill_all_bools(header, VkPhysicalDeviceVulkan11Features::size);
			case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES: fake_fill_all_bools(header, VkPhysicalDeviceVulkan12Features::size);
			case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES: fake_fill_all_bools(header, VkPhysicalDeviceVulkan13Features::size);
			case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_4_FEATURES: fake_fill_all_bools(header, VkPhysicalDeviceVulkan14Features::size);
			case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_HEAP_FEATURES_EXT: fake_fill_all_bools(header, VkPhysicalDeviceDescriptorHeapFeaturesEXT::size);
			case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEVICE_ADDRESS_COMMANDS_FEATURES_KHR: fake_fill_all_bools(header, VkPhysicalDeviceDeviceAddressCommandsFeaturesKHR::size);
			case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_UNTYPED_POINTERS_FEATURES_KHR: fake_fill_all_bools(header, VkPhysicalDeviceShaderUntypedPointersFeaturesKHR::size);
			case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_UNIFIED_IMAGE_LAYOUTS_FEATURES_KHR: fake_fill_all_bools(header, VkPhysicalDeviceUnifiedImageLayoutsFeaturesKHR::size);
			case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT: fake_fill_all_bools(header, VkPhysicalDeviceMeshShaderFeaturesEXT::size);
			case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SWAPCHAIN_MAINTENANCE_1_FEATURES_KHR: fake_fill_all_bools(header, VkPhysicalDeviceSwapchainMaintenance1FeaturesKHR::size);
			default: fake_fail("vkGetPhysicalDeviceFeatures2: unknown structure in the chain");
		}
	}
}

fn void fake_get_properties2(VkPhysicalDevice device, VkPhysicalDeviceProperties2* props)
{
	props.properties.apiVersion = VK_API_VERSION_1_4;
	props.properties.deviceType = VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
	mem::copy(&props.properties.deviceName[0], (ZString)"Fake Vulkan GPU", 15);
	props.properties.limits.maxImageDimension2D = 16384;
	props.properties.limits.maxImageDimension3D = 2048;
	props.properties.limits.timestampPeriod = 1.0f;
	props.properties.limits.subTexelPrecisionBits = 8;
	for (FakeHeader* header = (FakeHeader*)props.pNext; header; header = (FakeHeader*)header.pNext)
	{
		switch (header.sType)
		{
			case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_HEAP_PROPERTIES_EXT:
				VkPhysicalDeviceDescriptorHeapPropertiesEXT* heap = (VkPhysicalDeviceDescriptorHeapPropertiesEXT*)header;
				heap.samplerHeapAlignment = 256;
				heap.resourceHeapAlignment = 256;
				heap.minSamplerHeapReservedRange = 1024;
				heap.minResourceHeapReservedRange = 2048;
				heap.samplerDescriptorSize = 16;
				heap.imageDescriptorSize = 64;
				heap.bufferDescriptorSize = 32;
				heap.samplerDescriptorAlignment = 16;
				heap.imageDescriptorAlignment = 32;
				heap.bufferDescriptorAlignment = 16;
				heap.maxPushDataSize = 256;
			case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_PROPERTIES:
				((VkPhysicalDeviceVulkan12Properties*)header).maxTimelineSemaphoreValueDifference = 1UL << 31;
			default: fake_fail("vkGetPhysicalDeviceProperties2: unknown structure in the chain");
		}
	}
}

fn void fake_get_properties(VkPhysicalDevice device, VkPhysicalDeviceProperties* props)
{
	props.apiVersion = VK_API_VERSION_1_4;
	props.deviceType = VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
	mem::copy(&props.deviceName[0], (ZString)"Fake Vulkan GPU", 15);
}

fn void fake_get_memory_properties(VkPhysicalDevice device, VkPhysicalDeviceMemoryProperties* props)
{
	props.memoryHeapCount = 2;
	props.memoryHeaps[0] = { .size = 4UL << 30, .flags = VK_MEMORY_HEAP_DEVICE_LOCAL_BIT };
	props.memoryHeaps[1] = { .size = 4UL << 30, .flags = VK_MEMORY_HEAP_DEVICE_LOCAL_BIT };
	props.memoryTypeCount = 2;
	props.memoryTypes[0] = { .propertyFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, .heapIndex = 0 };
	props.memoryTypes[1] = { .propertyFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, .heapIndex = 1 };
}

fn void fake_get_queue_families(VkPhysicalDevice device, uint* count, VkQueueFamilyProperties* props)
{
	if (props)
	{
		props[0] = { .queueFlags = VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT, .queueCount = 2, .timestampValidBits = 64, .minImageTransferGranularity = { 1, 1, 1 } };
		props[1] = { .queueFlags = VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT, .queueCount = 1, .timestampValidBits = 64, .minImageTransferGranularity = { 1, 1, 1 } };
		props[2] = { .queueFlags = VK_QUEUE_TRANSFER_BIT, .queueCount = 1, .timestampValidBits = 64, .minImageTransferGranularity = { 16, 16, 1 } };
	}
	*count = 3;
}

fn void fake_get_format_properties2(VkPhysicalDevice device, VkFormat format, VkFormatProperties2* props)
{
	for (FakeHeader* header = (FakeHeader*)props.pNext; header; header = (FakeHeader*)header.pNext)
	{
		if (header.sType != VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_3) fake_fail("vkGetPhysicalDeviceFormatProperties2: unknown structure");
		((VkFormatProperties3*)header).optimalTilingFeatures = ulong::max;
	}
}

fn VkResult fake_get_image_format_properties2(VkPhysicalDevice device, VkPhysicalDeviceImageFormatInfo2* info, VkImageFormatProperties2* props)
{
	props.imageFormatProperties = { .maxExtent = { 16384, 16384, 2048 }, .maxMipLevels = 15, .maxArrayLayers = 2048, .sampleCounts = 1, .maxResourceSize = 1UL << 32 };
	return VK_SUCCESS;
}

fn VkResult fake_create_device(VkPhysicalDevice device, VkDeviceCreateInfo* info, VkAllocationCallbacks* allocator, VkDevice* out)
{
	if (info.sType != VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO) fake_fail("vkCreateDevice: sType");
	if (info.queueCreateInfoCount == 0) fake_fail("vkCreateDevice: no queues");
	for (uint i = 0; i < info.queueCreateInfoCount; i++)
	{
		if (info.pQueueCreateInfos[i].queueCount == 0 || !info.pQueueCreateInfos[i].pQueuePriorities) fake_fail("vkCreateDevice: queue info");
	}
	bool has_features2 = info.pNext && ((FakeHeader*)info.pNext).sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
	if (!has_features2) fake_fail("vkCreateDevice: the chain starts with the features");
	fake_address_commands = false;
	for (uint i = 0; i < info.enabledExtensionCount; i++)
	{
		if (libc::strcmp(info.ppEnabledExtensionNames[i], VK_KHR_DEVICE_ADDRESS_COMMANDS_EXTENSION_NAME) == 0) fake_address_commands = true;
	}
	bool feature_in_chain;
	for (FakeHeader* header = (FakeHeader*)info.pNext; header; header = (FakeHeader*)header.pNext)
	{
		if (header.sType != VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEVICE_ADDRESS_COMMANDS_FEATURES_KHR) continue;
		feature_in_chain = true;
		if ((((VkPhysicalDeviceDeviceAddressCommandsFeaturesKHR*)header).deviceAddressCommands != 0) != fake_address_commands) fake_fail("vkCreateDevice: the address commands feature and extension disagree");
	}
	if (feature_in_chain && !fake_address_commands) fake_fail("vkCreateDevice: the address commands feature is chained without its extension");
	*out = (VkDevice)&fake_dummy;
	return VK_SUCCESS;
}

fn void fake_destroy_device(VkDevice device, VkAllocationCallbacks* allocator) {}

fn void fake_get_device_queue(VkDevice device, uint family, uint index, VkQueue* queue) { *queue = (VkQueue)mem::new(FakeRecord); }

// ---- Memory, buffers and images -------------------------------------------------------------------------------------------------

fn VkResult fake_create_buffer(VkDevice device, VkBufferCreateInfo* info, VkAllocationCallbacks* allocator, VkBuffer* buffer)
{
	if (info.sType != VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO || info.size == 0) fake_fail("vkCreateBuffer");
	if (info.sharingMode == VK_SHARING_MODE_CONCURRENT && info.queueFamilyIndexCount < 2) fake_fail("vkCreateBuffer: concurrent sharing needs two families");
	FakeRecord* record = mem::new(FakeRecord);
	record.size = info.size;
	fake_count(BUFFER, 1);
	*buffer = (VkBuffer)(uptr)record;
	return VK_SUCCESS;
}

fn void fake_destroy_buffer(VkDevice device, VkBuffer buffer, VkAllocationCallbacks* allocator) { fake_count(BUFFER, -1); mem::free(record_of((ulong)buffer)); }

fn void fake_get_buffer_requirements(VkDevice device, VkBuffer buffer, VkMemoryRequirements* req)
{
	req.size = (record_of((ulong)buffer).size + 255) / 256 * 256;
	req.alignment = 256;
	req.memoryTypeBits = 3;
}

fn void fake_get_device_buffer_requirements(VkDevice device, VkDeviceBufferMemoryRequirements* info, VkMemoryRequirements2* req)
{
	req.memoryRequirements.size = (info.pCreateInfo.size + 255) / 256 * 256;
	req.memoryRequirements.alignment = 256;
	req.memoryRequirements.memoryTypeBits = 3;
}

fn VkResult fake_allocate_memory(VkDevice device, VkMemoryAllocateInfo* info, VkAllocationCallbacks* allocator, VkDeviceMemory* memory)
{
	if (info.sType != VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO || info.allocationSize == 0 || info.memoryTypeIndex > 1) fake_fail("vkAllocateMemory");
	fake_count(MEMORY, 1);
	*memory = (VkDeviceMemory)(uptr)mem::new_array(char, (sz)(info.allocationSize + 4096)).ptr;
	return VK_SUCCESS;
}

fn void fake_free_memory(VkDevice device, VkDeviceMemory memory, VkAllocationCallbacks* allocator) { fake_count(MEMORY, -1); mem::free((void*)(uptr)(ulong)memory); }

fn VkResult fake_bind_buffer_memory(VkDevice device, VkBuffer buffer, VkDeviceMemory memory, VkDeviceSize offset)
{
	FakeRecord* record = record_of((ulong)buffer);
	record.memory = (ulong)memory;
	record.offset = offset;
	return VK_SUCCESS;
}

fn VkResult fake_map_memory(VkDevice device, VkDeviceMemory memory, VkDeviceSize offset, VkDeviceSize size, VkMemoryMapFlags flags, void** data)
{
	*data = (void*)(uptr)((ulong)memory + offset);
	return VK_SUCCESS;
}

fn void fake_unmap_memory(VkDevice device, VkDeviceMemory memory) {}

fn VkDeviceAddress fake_get_buffer_address(VkDevice device, VkBufferDeviceAddressInfo* info)
{
	FakeRecord* record = record_of((ulong)info.buffer);
	if (!record.memory) fake_fail("vkGetBufferDeviceAddress on a buffer without memory");
	return record.memory + record.offset;
}

fn VkResult fake_create_image(VkDevice device, VkImageCreateInfo* info, VkAllocationCallbacks* allocator, VkImage* image)
{
	if (info.sType != VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO || info.mipLevels == 0 || info.arrayLayers == 0 || info.extent.width == 0) fake_fail("vkCreateImage");
	if (info.pNext && ((FakeHeader*)info.pNext).sType != VK_STRUCTURE_TYPE_IMAGE_FORMAT_LIST_CREATE_INFO) fake_fail("vkCreateImage: pNext");
	FakeRecord* record = mem::new(FakeRecord);
	record.width = info.extent.width;
	record.height = info.extent.height;
	record.depth = info.extent.depth;
	record.layers = info.arrayLayers;
	record.format = (uint)info.format;
	fake_count(IMAGE, 1);
	*image = (VkImage)(uptr)record;
	return VK_SUCCESS;
}

fn void fake_destroy_image(VkDevice device, VkImage image, VkAllocationCallbacks* allocator) { fake_count(IMAGE, -1); mem::free(record_of((ulong)image)); }

fn VkResult fake_bind_image_memory(VkDevice device, VkImage image, VkDeviceMemory memory, VkDeviceSize offset)
{
	FakeRecord* record = record_of((ulong)image);
	record.memory = (ulong)memory;
	record.offset = offset;
	return VK_SUCCESS;
}

fn void fake_get_device_image_requirements(VkDevice device, VkDeviceImageMemoryRequirements* info, VkMemoryRequirements2* req)
{
	VkImageCreateInfo* image = info.pCreateInfo;
	ulong bytes = (ulong)image.extent.width * image.extent.height * image.extent.depth * image.arrayLayers * 16 * 2;
	req.memoryRequirements.size = (bytes + FAKE_PAGE - 1) / FAKE_PAGE * FAKE_PAGE;
	req.memoryRequirements.alignment = FAKE_PAGE;
	req.memoryRequirements.memoryTypeBits = 2;
}

fn VkResult fake_create_image_view(VkDevice device, VkImageViewCreateInfo* info, VkAllocationCallbacks* allocator, VkImageView* view)
{
	if (info.sType != VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO || info.image == 0) fake_fail("vkCreateImageView");
	fake_count(VIEW, 1);
	*view = (VkImageView)new_record();
	return VK_SUCCESS;
}

fn void fake_destroy_image_view(VkDevice device, VkImageView view, VkAllocationCallbacks* allocator) { fake_count(VIEW, -1); mem::free(record_of((ulong)view)); }

// ---- Synchronization ------------------------------------------------------------------------------------------------------------

fn VkResult fake_create_semaphore(VkDevice device, VkSemaphoreCreateInfo* info, VkAllocationCallbacks* allocator, VkSemaphore* semaphore)
{
	FakeRecord* record = mem::new(FakeRecord);
	if (info.pNext)
	{
		VkSemaphoreTypeCreateInfo* type_info = (VkSemaphoreTypeCreateInfo*)info.pNext;
		if (type_info.sType != VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO) fake_fail("vkCreateSemaphore: pNext");
		record.timeline = type_info.semaphoreType == VK_SEMAPHORE_TYPE_TIMELINE;
		record.value = type_info.initialValue;
	}
	fake_count(SEMAPHORE, 1);
	*semaphore = (VkSemaphore)(uptr)record;
	return VK_SUCCESS;
}

fn void fake_destroy_semaphore(VkDevice device, VkSemaphore semaphore, VkAllocationCallbacks* allocator) { fake_count(SEMAPHORE, -1); mem::free(record_of((ulong)semaphore)); }

fn VkResult fake_get_semaphore_value(VkDevice device, VkSemaphore semaphore, ulong* value)
{
	FakeRecord* record = record_of((ulong)semaphore);
	if (!record.timeline) fake_fail("vkGetSemaphoreCounterValue on a binary semaphore");
	*value = record.value;
	return VK_SUCCESS;
}

fn VkResult fake_wait_semaphores(VkDevice device, VkSemaphoreWaitInfo* info, ulong timeout)
{
	for (uint i = 0; i < info.semaphoreCount; i++)
	{
		FakeRecord* record = record_of((ulong)info.pSemaphores[i]);
		if (!record.timeline) fake_fail("vkWaitSemaphores on a binary semaphore");
		if (record.value < info.pValues[i]) fake_fail(string::tformat("vkWaitSemaphores waits for %d but only %d was ever signaled", info.pValues[i], record.value));
	}
	return VK_SUCCESS;
}

fn VkResult fake_create_fence(VkDevice device, VkFenceCreateInfo* info, VkAllocationCallbacks* allocator, VkFence* fence)
{
	fake_count(FENCE, 1);
	*fence = (VkFence)new_record();
	return VK_SUCCESS;
}

fn void fake_destroy_fence(VkDevice device, VkFence fence, VkAllocationCallbacks* allocator) { fake_count(FENCE, -1); mem::free(record_of((ulong)fence)); }

fn VkResult fake_ok() { return VK_SUCCESS; }

fn VkResult fake_wait_for_fences(VkDevice device, uint count, VkFence* fences, VkBool32 wait_all, ulong timeout)
{
	for (uint i = 0; i < count; i++)
	{
		if (!record_of((ulong)fences[i]).signaled) fake_fail("vkWaitForFences on a fence that nothing signaled: a deadlock");
	}
	return VK_SUCCESS;
}

fn VkResult fake_reset_fences(VkDevice device, uint count, VkFence* fences)
{
	for (uint i = 0; i < count; i++) record_of((ulong)fences[i]).signaled = false;
	return VK_SUCCESS;
}

fn VkResult fake_get_fence_status(VkDevice device, VkFence fence) => record_of((ulong)fence).signaled ? VK_SUCCESS : VK_NOT_READY;

// The extent of the surface, read from the environment so a test can resize it: FAKE_EXTENT=800x600.
fn VkExtent2D fake_surface_extent() @private
{
	ZString text = libc::getenv("FAKE_EXTENT");
	uint width = 640;
	uint height = 480;
	if (text)
	{
		char* chars = (char*)text;
		width = (uint)libc::atoi(chars);
		for (; *chars && *chars != 'x'; chars++) {}
		if (*chars) height = (uint)libc::atoi(chars + 1);
	}
	return { width, height };
}

fn VkResult fake_create_surface(VkInstance instance, VkWin32SurfaceCreateInfoKHR* info, VkAllocationCallbacks* allocator, VkSurfaceKHR* surface)
{
	if (info.sType != VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR || !info.hwnd) fake_fail("vkCreateWin32SurfaceKHR");
	fake_count(SURFACE, 1);
	*surface = (VkSurfaceKHR)new_record();
	return VK_SUCCESS;
}

fn void fake_destroy_surface(VkInstance instance, VkSurfaceKHR surface, VkAllocationCallbacks* allocator) { fake_count(SURFACE, -1); mem::free(record_of((ulong)surface)); }

fn VkResult fake_surface_support(VkPhysicalDevice device, uint family, VkSurfaceKHR surface, VkBool32* supported)
{
	*supported = family == 0 ? 1 : 0;
	return VK_SUCCESS;
}

fn void fake_fill_capabilities(VkSurfaceCapabilitiesKHR* caps) @private
{
	VkExtent2D extent = fake_surface_extent();
	*caps = {
		.minImageCount = 2,
		.maxImageCount = 4,
		.currentExtent = extent,
		.maxImageArrayLayers = 1,
		.supportedTransforms = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR,
		.currentTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR,
		.supportedCompositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
		.supportedUsageFlags = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
	};
}

fn VkResult fake_surface_capabilities(VkPhysicalDevice device, VkSurfaceKHR surface, VkSurfaceCapabilitiesKHR* caps)
{
	fake_fill_capabilities(caps);
	return VK_SUCCESS;
}

fn VkResult fake_surface_capabilities2(VkPhysicalDevice device, VkPhysicalDeviceSurfaceInfo2KHR* info, VkSurfaceCapabilities2KHR* caps)
{
	FakeHeader* header = (FakeHeader*)info.pNext;
	if (!header || header.sType != VK_STRUCTURE_TYPE_SURFACE_PRESENT_MODE_KHR) fake_fail("vkGetPhysicalDeviceSurfaceCapabilities2KHR: the present mode is not chained");
	fake_fill_capabilities(&caps.surfaceCapabilities);
	return VK_SUCCESS;
}

fn VkResult fake_surface_formats(VkPhysicalDevice device, VkSurfaceKHR surface, uint* count, VkSurfaceFormatKHR* formats)
{
	if (formats)
	{
		if (*count >= 1) formats[0] = { VK_FORMAT_B8G8R8A8_SRGB, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR };
		if (*count >= 2) formats[1] = { VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR };
		if (*count >= 3) formats[2] = { VK_FORMAT_R16G16B16A16_SFLOAT, VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT };
	}
	*count = 3;
	return VK_SUCCESS;
}

struct FakeSwapchain { VkExtent2D extent; uint image_count; ulong[8] images; uint next; bool destroyed; }

fn VkResult fake_create_swapchain(VkDevice device, VkSwapchainCreateInfoKHR* info, VkAllocationCallbacks* allocator, VkSwapchainKHR* swapchain)
{
	if (info.sType != VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR || info.imageArrayLayers != 1 || info.imageExtent.width == 0 || info.surface == 0) fake_fail("vkCreateSwapchainKHR");
	if (info.minImageCount < 2 || info.minImageCount > 4) fake_fail("vkCreateSwapchainKHR: image count outside the surface limits");
	FakeHeader* header = (FakeHeader*)info.pNext;
	if (!header || header.sType != VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_MODES_CREATE_INFO_KHR) fake_fail("vkCreateSwapchainKHR: the present modes are not chained");
	if (info.oldSwapchain != 0 && ((FakeSwapchain*)(uptr)(ulong)info.oldSwapchain).destroyed) fake_fail("vkCreateSwapchainKHR: oldSwapchain was destroyed");
	FakeSwapchain* result = mem::new(FakeSwapchain);
	result.extent = info.imageExtent;
	result.image_count = info.minImageCount;
	for (uint i = 0; i < result.image_count; i++)
	{
		FakeRecord* image = mem::new(FakeRecord);
		image.width = info.imageExtent.width;
		image.height = info.imageExtent.height;
		result.images[i] = (ulong)(uptr)image;
	}
	fake_count(SWAPCHAIN, 1);
	*swapchain = (VkSwapchainKHR)(uptr)result;
	return VK_SUCCESS;
}

fn void fake_destroy_swapchain(VkDevice device, VkSwapchainKHR swapchain, VkAllocationCallbacks* allocator)
{
	FakeSwapchain* chain = (FakeSwapchain*)(uptr)(ulong)swapchain;
	if (chain.destroyed) fake_fail("vkDestroySwapchainKHR twice");
	chain.destroyed = true;   // The struct stays so a later use of the handle is noticed.
	fake_count(SWAPCHAIN, -1);
}

fn VkResult fake_get_swapchain_images(VkDevice device, VkSwapchainKHR swapchain, uint* count, VkImage* images)
{
	FakeSwapchain* chain = (FakeSwapchain*)(uptr)(ulong)swapchain;
	if (images) for (uint i = 0; i < chain.image_count && i < *count; i++) images[i] = (VkImage)chain.images[i];
	*count = chain.image_count;
	return VK_SUCCESS;
}

fn VkResult fake_acquire_next_image(VkDevice device, VkSwapchainKHR swapchain, ulong timeout, VkSemaphore semaphore, VkFence fence, uint* index)
{
	FakeSwapchain* chain = (FakeSwapchain*)(uptr)(ulong)swapchain;
	if (chain.destroyed) fake_fail("vkAcquireNextImageKHR on a destroyed swapchain");
	VkExtent2D extent = fake_surface_extent();
	if (extent.width != chain.extent.width || extent.height != chain.extent.height) return VK_ERROR_OUT_OF_DATE_KHR;
	FakeRecord* wait = record_of((ulong)semaphore);
	if (wait.signaled) fake_fail("vkAcquireNextImageKHR: the semaphore is still signaled (not consumed by a submission)");
	wait.signaled = true;
	*index = chain.next;
	chain.next = (chain.next + 1) % chain.image_count;
	return VK_SUCCESS;
}

fn VkResult fake_queue_present(VkQueue queue, VkPresentInfoKHR* info)
{
	if (info.swapchainCount != 1 || info.waitSemaphoreCount != 1) fake_fail("vkQueuePresentKHR: counts");
	FakeRecord* wait = record_of((ulong)info.pWaitSemaphores[0]);
	if (!wait.signaled) fake_fail("vkQueuePresentKHR waits for a semaphore that no submission signaled");
	wait.signaled = false;
	FakeHeader* header = (FakeHeader*)info.pNext;
	if (!header || header.sType != VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_FENCE_INFO_KHR) fake_fail("vkQueuePresentKHR: the present fence info is not chained");
	VkSwapchainPresentFenceInfoKHR* fence_info = (VkSwapchainPresentFenceInfoKHR*)header;
	FakeRecord* fence = record_of((ulong)fence_info.pFences[0]);
	if (fence.signaled) fake_fail("vkQueuePresentKHR: the present fence was not reset");
	fence.signaled = true;
	FakeSwapchain* chain = (FakeSwapchain*)(uptr)(ulong)info.pSwapchains[0];
	if (chain.destroyed) fake_fail("vkQueuePresentKHR on a destroyed swapchain");
	VkExtent2D extent = fake_surface_extent();
	if (extent.width != chain.extent.width || extent.height != chain.extent.height) return VK_SUBOPTIMAL_KHR;
	return VK_SUCCESS;
}

fn VkResult fake_queue_submit2(VkQueue queue, uint count, VkSubmitInfo2* submits, VkFence fence)
{
	for (uint s = 0; s < count; s++)
	{
		VkSubmitInfo2* submit = &submits[s];
		if (submit.sType != VK_STRUCTURE_TYPE_SUBMIT_INFO_2) fake_fail("vkQueueSubmit2: sType");
		for (uint i = 0; i < submit.commandBufferInfoCount; i++)
		{
			FakeRecord* cb = (FakeRecord*)submit.pCommandBufferInfos[i].commandBuffer;
			if (cb.recording || !cb.ended) fake_fail("vkQueueSubmit2: a command buffer that was not ended");
			if (cb.submitted) fake_fail("vkQueueSubmit2: a one time submit command buffer submitted again");
			cb.submitted = true;
			if (submit.pCommandBufferInfos[i].sType != VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO) fake_fail("vkQueueSubmit2: command buffer info sType");
		}
		for (uint i = 0; i < submit.waitSemaphoreInfoCount; i++)
		{
			FakeRecord* wait = record_of((ulong)submit.pWaitSemaphoreInfos[i].semaphore);
			if (wait.timeline && wait.value < submit.pWaitSemaphoreInfos[i].value) fake_fail("vkQueueSubmit2: waits for a value that is not signaled");
			if (!wait.timeline)
			{
				if (!wait.signaled) fake_fail("vkQueueSubmit2: waits for a binary semaphore that nothing signaled");
				wait.signaled = false;
			}
		}
		for (uint i = 0; i < submit.signalSemaphoreInfoCount; i++)
		{
			FakeRecord* signal = record_of((ulong)submit.pSignalSemaphoreInfos[i].semaphore);
			if (signal.timeline)
			{
				if (submit.pSignalSemaphoreInfos[i].value <= signal.value) fake_fail("vkQueueSubmit2: a timeline signal that does not increase");
				signal.value = submit.pSignalSemaphoreInfos[i].value;
			}
			else
			{
				if (signal.signaled) fake_fail("vkQueueSubmit2: signals a binary semaphore that is already signaled");
				signal.signaled = true;
			}
		}
	}
	return VK_SUCCESS;
}

// ---- Command pools and buffers --------------------------------------------------------------------------------------------------

fn VkResult fake_create_record_handle(VkDevice device, void* info, VkAllocationCallbacks* allocator, ulong* out)
{
	fake_count(POOL, 1);
	*out = new_record();
	return VK_SUCCESS;
}

fn void fake_destroy_record_handle(VkDevice device, ulong handle, VkAllocationCallbacks* allocator) { mem::free(record_of(handle)); }

fn VkResult fake_allocate_command_buffers(VkDevice device, VkCommandBufferAllocateInfo* info, VkCommandBuffer* buffers)
{
	if (info.commandPool == 0 || info.level != VK_COMMAND_BUFFER_LEVEL_PRIMARY) fake_fail("vkAllocateCommandBuffers");
	for (uint i = 0; i < info.commandBufferCount; i++) buffers[i] = (VkCommandBuffer)mem::new(FakeRecord);
	return VK_SUCCESS;
}

fn void fake_free_command_buffers(VkDevice device, VkCommandPool pool, uint count, VkCommandBuffer* buffers) {}

fn VkResult fake_begin_command_buffer(VkCommandBuffer cb, VkCommandBufferBeginInfo* info)
{
	FakeRecord* record = (FakeRecord*)cb;
	if (record.recording) fake_fail("vkBeginCommandBuffer on a recording command buffer");
	record.recording = true;
	record.ended = false;
	record.submitted = false;
	record.rendering = false;
	record.pipeline_bound = false;
	return VK_SUCCESS;
}

fn VkResult fake_end_command_buffer(VkCommandBuffer cb)
{
	FakeRecord* record = cb_of(cb, "vkEndCommandBuffer");
	if (record.rendering) fake_fail("vkEndCommandBuffer inside a render pass");
	record.recording = false;
	record.ended = true;
	return VK_SUCCESS;
}

fn VkResult fake_reset_command_pool(VkDevice device, VkCommandPool pool, VkCommandPoolResetFlags flags) { return VK_SUCCESS; }

fn void fake_destroy_pool(VkDevice device, VkCommandPool pool, VkAllocationCallbacks* allocator) { fake_count(POOL, -1); }

fn VkResult fake_create_query_pool(VkDevice device, VkQueryPoolCreateInfo* info, VkAllocationCallbacks* allocator, VkQueryPool* pool)
{
	FakeRecord* record = mem::new(FakeRecord);
	record.query_count = info.queryCount;
	fake_count(QUERY, 1);
	*pool = (VkQueryPool)(uptr)record;
	return VK_SUCCESS;
}

fn void fake_destroy_query_pool(VkDevice device, VkQueryPool pool, VkAllocationCallbacks* allocator) { fake_count(QUERY, -1); mem::free(record_of((ulong)pool)); }

fn void fake_reset_query_pool(VkDevice device, VkQueryPool pool, uint first, uint count) {}

fn VkResult fake_get_query_results(VkDevice device, VkQueryPool pool, uint first, uint count, usz size, void* data, VkDeviceSize stride, VkQueryResultFlags flags)
{
	if (size < (usz)count * 8 || stride != 8 || !(flags & VK_QUERY_RESULT_64_BIT)) fake_fail("vkGetQueryPoolResults: arguments");
	ulong* results = (ulong*)data;
	for (uint i = 0; i < count; i++) results[i] = ++fake_clock * 1000;
	return VK_SUCCESS;
}

fn void fake_write_timestamp(VkCommandBuffer cb, VkPipelineStageFlags2 stage, VkQueryPool pool, uint query)
{
	cb_of(cb, "vkCmdWriteTimestamp2");
	if (query >= record_of((ulong)pool).query_count) fake_fail("vkCmdWriteTimestamp2: query out of range");
}

// ---- Pipelines ------------------------------------------------------------------------------------------------------------------

fn void fake_check_stage(VkPipelineShaderStageCreateInfo* stage) @private
{
	if (stage.sType != VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO || !stage.pName) fake_fail("pipeline stage: sType or name");
	if (stage.module_ != 0) fake_fail("pipeline stage: a module with the inline module chain");
	FakeHeader* mapping = (FakeHeader*)stage.pNext;
	if (!mapping || mapping.sType != VK_STRUCTURE_TYPE_SHADER_DESCRIPTOR_SET_AND_BINDING_MAPPING_INFO_EXT) fake_fail("pipeline stage: the mapping info is not first in the chain");
	VkShaderDescriptorSetAndBindingMappingInfoEXT* info = (VkShaderDescriptorSetAndBindingMappingInfoEXT*)mapping;
	if (info.mappingCount != 1) fake_fail("pipeline stage: one mapping");
	if (info.pMappings[0].source != VK_DESCRIPTOR_MAPPING_SOURCE_PUSH_ADDRESS_EXT || info.pMappings[0].bindingCount != 1 || info.pMappings[0].descriptorSet != 0
		|| info.pMappings[0].firstBinding != 0) fake_fail("pipeline stage: the root mapping");
	FakeHeader* module_info = (FakeHeader*)info.pNext;
	if (!module_info || module_info.sType != VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO) fake_fail("pipeline stage: the module info follows the mapping");
	VkShaderModuleCreateInfo* code = (VkShaderModuleCreateInfo*)module_info;
	if (code.codeSize < 20 || code.codeSize % 4 != 0 || (uptr)code.pCode % 4 != 0 || code.pCode[0] != 0x07230203) fake_fail("pipeline stage: SPIR-V size, alignment or magic");
}

fn void fake_check_flags(void* chain, bool graphics) @private
{
	FakeHeader* header = (FakeHeader*)chain;
	if (!header || header.sType != VK_STRUCTURE_TYPE_PIPELINE_CREATE_FLAGS_2_CREATE_INFO) fake_fail("pipeline: the create flags 2 info is not first in the chain");
	VkPipelineCreateFlags2CreateInfo* flags = (VkPipelineCreateFlags2CreateInfo*)header;
	if (!(flags.flags & VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT)) fake_fail("pipeline: the descriptor heap flag");
	if (graphics && (!flags.pNext || ((FakeHeader*)flags.pNext).sType != VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO)) fake_fail("pipeline: the rendering info follows the flags");
}

fn VkResult fake_create_graphics_pipelines(VkDevice device, VkPipelineCache cache, uint count, VkGraphicsPipelineCreateInfo* infos, VkAllocationCallbacks* allocator, VkPipeline* pipelines)
{
	for (uint i = 0; i < count; i++)
	{
		VkGraphicsPipelineCreateInfo* info = &infos[i];
		if (info.sType != VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO || info.layout != 0 || info.renderPass != 0) fake_fail("vkCreateGraphicsPipelines: layout / render pass");
		fake_check_flags(info.pNext, true);
		if (info.stageCount == 0 || info.stageCount > 3) fake_fail("vkCreateGraphicsPipelines: stage count");
		for (uint s = 0; s < info.stageCount; s++) fake_check_stage(&info.pStages[s]);
		VkShaderStageFlagBits first = info.pStages[0].stage;
		bool mesh = first == VK_SHADER_STAGE_TASK_BIT_EXT || first == VK_SHADER_STAGE_MESH_BIT_EXT;
		if (mesh != (info.pVertexInputState == null) || mesh != (info.pInputAssemblyState == null)) fake_fail("vkCreateGraphicsPipelines: vertex input state with mesh shaders");
		if (!info.pColorBlendState || !info.pDynamicState || !info.pRasterizationState || !info.pViewportState) fake_fail("vkCreateGraphicsPipelines: missing state");
		for (uint a = 0; a < info.pColorBlendState.attachmentCount; a++)
		{
			VkPipelineColorBlendAttachmentState* blend = &info.pColorBlendState.pAttachments[a];
			if (blend.blendEnable && blend.srcColorBlendFactor == VK_BLEND_FACTOR_ZERO && blend.dstColorBlendFactor == VK_BLEND_FACTOR_ZERO) fake_fail("vkCreateGraphicsPipelines: blending with ZERO and ZERO factors, which draws nothing");
			if (blend.colorWriteMask == 0) fake_fail("vkCreateGraphicsPipelines: a color write mask of no channels");
		}
		VkPipelineRenderingCreateInfo* rendering = (VkPipelineRenderingCreateInfo*)((VkPipelineCreateFlags2CreateInfo*)info.pNext).pNext;
		if (rendering.colorAttachmentCount != info.pColorBlendState.attachmentCount) fake_fail("vkCreateGraphicsPipelines: attachment counts");
		fake_count(PIPELINE, 1);
		pipelines[i] = (VkPipeline)new_record();
	}
	return VK_SUCCESS;
}

fn VkResult fake_create_compute_pipelines(VkDevice device, VkPipelineCache cache, uint count, VkComputePipelineCreateInfo* infos, VkAllocationCallbacks* allocator, VkPipeline* pipelines)
{
	for (uint i = 0; i < count; i++)
	{
		if (infos[i].sType != VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO || infos[i].layout != 0) fake_fail("vkCreateComputePipelines: layout");
		fake_check_flags(infos[i].pNext, false);
		if (infos[i].stage.stage != VK_SHADER_STAGE_COMPUTE_BIT) fake_fail("vkCreateComputePipelines: stage");
		fake_check_stage(&infos[i].stage);
		fake_count(PIPELINE, 1);
		pipelines[i] = (VkPipeline)new_record();
	}
	return VK_SUCCESS;
}

fn void fake_destroy_pipeline(VkDevice device, VkPipeline pipeline, VkAllocationCallbacks* allocator) { fake_count(PIPELINE, -1); mem::free(record_of((ulong)pipeline)); }

// ---- Commands -------------------------------------------------------------------------------------------------------------------

fn void fake_bind_pipeline(VkCommandBuffer cb, VkPipelineBindPoint point, VkPipeline pipeline)
{
	FakeRecord* record = cb_of(cb, "vkCmdBindPipeline");
	record.pipeline_bound = true;
	record.bind_point = (uint)point;
}

fn void fake_begin_rendering(VkCommandBuffer cb, VkRenderingInfo* info)
{
	FakeRecord* record = cb_of(cb, "vkCmdBeginRendering");
	if (record.rendering) fake_fail("vkCmdBeginRendering inside a render pass");
	if (info.sType != VK_STRUCTURE_TYPE_RENDERING_INFO || info.layerCount != 1 || info.renderArea.extent.width == 0) fake_fail("vkCmdBeginRendering: info");
	for (uint i = 0; i < info.colorAttachmentCount; i++)
	{
		if (info.pColorAttachments[i].imageView == 0 || info.pColorAttachments[i].imageLayout != VK_IMAGE_LAYOUT_GENERAL) fake_fail("vkCmdBeginRendering: color attachment");
	}
	record.rendering = true;
	record.pipeline_bound = false;
}

fn void fake_end_rendering(VkCommandBuffer cb)
{
	FakeRecord* record = cb_of(cb, "vkCmdEndRendering");
	if (!record.rendering) fake_fail("vkCmdEndRendering outside a render pass");
	record.rendering = false;
}

fn void fake_draw(VkCommandBuffer cb, uint a, uint b, uint c, uint d)
{
	FakeRecord* record = cb_of(cb, "vkCmdDraw");
	if (!record.rendering || !record.pipeline_bound || record.bind_point != (uint)VK_PIPELINE_BIND_POINT_GRAPHICS) fake_fail("vkCmdDraw: needs a render pass and a graphics pipeline");
}

fn void fake_draw_indexed(VkCommandBuffer cb, uint a, uint b, uint c, int d, uint e)
{
	FakeRecord* record = cb_of(cb, "vkCmdDrawIndexed");
	if (!record.rendering || !record.pipeline_bound) fake_fail("vkCmdDrawIndexed: needs a render pass and a pipeline");
}

fn void fake_draw_indirect2(VkCommandBuffer cb, VkDrawIndirect2InfoKHR* info)
{
	fake_require_address_commands("vkCmdDrawIndirect2KHR");
	FakeRecord* record = cb_of(cb, "vkCmdDrawIndirect2KHR");
	if (!record.rendering || !record.pipeline_bound) fake_fail("indirect draw: needs a render pass and a pipeline");
	if (info.sType != VK_STRUCTURE_TYPE_DRAW_INDIRECT_2_INFO_KHR || info.addressRange.address == 0 || info.addressRange.stride == 0 || info.drawCount == 0) fake_fail("indirect draw: info");
	if (!(info.addressFlags & VK_ADDRESS_COMMAND_FULLY_BOUND_BIT_KHR)) fake_fail("indirect draw: fully bound flag");
}

fn void fake_draw_mesh_tasks(VkCommandBuffer cb, uint x, uint y, uint z)
{
	FakeRecord* record = cb_of(cb, "vkCmdDrawMeshTasksEXT");
	if (!record.rendering || !record.pipeline_bound) fake_fail("vkCmdDrawMeshTasksEXT: needs a render pass and a pipeline");
}

fn void fake_dispatch(VkCommandBuffer cb, uint x, uint y, uint z)
{
	FakeRecord* record = cb_of(cb, "vkCmdDispatch");
	if (record.rendering || !record.pipeline_bound) fake_fail("vkCmdDispatch: inside a render pass or without a pipeline");
}

fn void fake_dispatch_indirect2(VkCommandBuffer cb, VkDispatchIndirect2InfoKHR* info)
{
	fake_require_address_commands("vkCmdDispatchIndirect2KHR");
	FakeRecord* record = cb_of(cb, "vkCmdDispatchIndirect2KHR");
	if (record.rendering || !record.pipeline_bound) fake_fail("vkCmdDispatchIndirect2KHR: inside a render pass or without a pipeline");
	if (info.sType != VK_STRUCTURE_TYPE_DISPATCH_INDIRECT_2_INFO_KHR || info.addressRange.address == 0 || info.addressRange.size < 12) fake_fail("vkCmdDispatchIndirect2KHR: info");
}

fn void fake_push_data(VkCommandBuffer cb, VkPushDataInfoEXT* info)
{
	cb_of(cb, "vkCmdPushDataEXT");
	if (info.sType != VK_STRUCTURE_TYPE_PUSH_DATA_INFO_EXT || info.offset != 0 || info.data.size != 8 || !info.data.address) fake_fail("vkCmdPushDataEXT: eight bytes at offset zero");
	ulong root = *(ulong*)info.data.address;
	if (root == 0 || root % 16 != 0) fake_fail("vkCmdPushDataEXT: the root is not a 16 byte aligned address");
}

fn void fake_bind_heap(VkCommandBuffer cb, VkBindHeapInfoEXT* info)
{
	cb_of(cb, "vkCmdBind*HeapEXT");
	if (info.sType != VK_STRUCTURE_TYPE_BIND_HEAP_INFO_EXT || info.heapRange.address == 0) fake_fail("vkCmdBind*HeapEXT: info");
	if (info.reservedRangeOffset + info.reservedRangeSize > info.heapRange.size) fake_fail("vkCmdBind*HeapEXT: the reserved range is outside the heap");
	if (info.reservedRangeOffset % 16 != 0) fake_fail("vkCmdBind*HeapEXT: reserved range alignment");
}

fn void fake_require_address_commands(String name) @private
{
	if (!fake_address_commands) fake_fail(string::tformat("%s was called, but the device was created without VK_KHR_device_address_commands", name));
}

fn void fake_bind_index_buffer(VkCommandBuffer cb, VkBindIndexBuffer3InfoKHR* info)
{
	fake_require_address_commands("vkCmdBindIndexBuffer3KHR");
	cb_of(cb, "vkCmdBindIndexBuffer3KHR");
	if (info.sType != VK_STRUCTURE_TYPE_BIND_INDEX_BUFFER_3_INFO_KHR || info.addressRange.address == 0 || info.addressRange.size == 0) fake_fail("vkCmdBindIndexBuffer3KHR: info");
}

fn void fake_copy_memory(VkCommandBuffer cb, VkCopyDeviceMemoryInfoKHR* info)
{
	fake_require_address_commands("vkCmdCopyMemoryKHR");
	FakeRecord* record = cb_of(cb, "vkCmdCopyMemoryKHR");
	if (record.rendering) fake_fail("vkCmdCopyMemoryKHR inside a render pass");
	for (uint i = 0; i < info.regionCount; i++)
	{
		VkDeviceMemoryCopyKHR* region = &info.pRegions[i];
		if (region.srcRange.address == 0 || region.dstRange.address == 0) fake_fail("vkCmdCopyMemoryKHR: null address");
		ulong size = min(region.srcRange.size, region.dstRange.size);
		mem::move((void*)(uptr)region.dstRange.address, (void*)(uptr)region.srcRange.address, (sz)size);
	}
}

// Images are kept linear, tightly packed, for layer 0 and mip 0: enough to run copies in and out.
fn void fake_copy_rows(FakeRecord* image, char* linear_base, uint row_length, VkOffset3D offset, VkExtent3D extent, bool to_image) @private
{
	for (uint y = 0; y < extent.height; y++)
	{
		char* linear = linear_base + (ulong)y * (row_length ?: extent.width) * 4;
		char* in_image = (char*)(uptr)(image.memory + image.offset) + ((ulong)(y + (uint)offset.y) * image.width + (uint)offset.x) * 4;
		if (to_image)
		{
			mem::copy(in_image, linear, (sz)extent.width * 4);
		}
		else
		{
			mem::copy(linear, in_image, (sz)extent.width * 4);
		}
	}
}

fn void fake_copy_image(VkCommandBuffer cb, VkCopyDeviceMemoryImageInfoKHR* info, bool to_image)
{
	fake_require_address_commands("vkCmdCopy*ImageKHR");
	FakeRecord* record = cb_of(cb, "vkCmdCopy*ImageKHR");
	if (record.rendering) fake_fail("image copy inside a render pass");
	FakeRecord* image = record_of((ulong)info.image);
	for (uint i = 0; i < info.regionCount; i++)
	{
		VkDeviceMemoryImageCopyKHR* region = &info.pRegions[i];
		if (region.sType != VK_STRUCTURE_TYPE_DEVICE_MEMORY_IMAGE_COPY_KHR || region.imageLayout != VK_IMAGE_LAYOUT_GENERAL || region.imageSubresource.aspectMask == 0) fake_fail("image copy: region");
		if (region.imageSubresource.mipLevel != 0 || region.imageSubresource.baseArrayLayer != 0) continue;
		fake_copy_rows(image, (char*)(uptr)region.addressRange.address, region.addressRowLength, region.imageOffset, region.imageExtent, to_image);
	}
}

fn void fake_copy_to_image(VkCommandBuffer cb, VkCopyDeviceMemoryImageInfoKHR* info) { fake_copy_image(cb, info, true); }
fn void fake_copy_from_image(VkCommandBuffer cb, VkCopyDeviceMemoryImageInfoKHR* info) { fake_copy_image(cb, info, false); }

// The same with buffers, which is what the backend uses without VK_KHR_device_address_commands.
fn char* fake_buffer_memory(VkBuffer buffer, ulong offset, ulong size) @private
{
	FakeRecord* record = record_of((ulong)buffer);
	if (!record.memory) fake_fail("a buffer command on a buffer without memory");
	if (offset + size > record.size) fake_fail(string::tformat("a buffer command reaches past the buffer: %d + %d > %d", offset, size, record.size));
	return (char*)(uptr)(record.memory + record.offset + offset);
}

fn void fake_copy_buffer_to_image(VkCommandBuffer cb, VkCopyBufferToImageInfo2* info)
{
	FakeRecord* record = cb_of(cb, "vkCmdCopyBufferToImage2");
	if (record.rendering || info.sType != VK_STRUCTURE_TYPE_COPY_BUFFER_TO_IMAGE_INFO_2 || info.dstImageLayout != VK_IMAGE_LAYOUT_GENERAL) fake_fail("vkCmdCopyBufferToImage2");
	for (uint i = 0; i < info.regionCount; i++)
	{
		VkBufferImageCopy2* region = &info.pRegions[i];
		if (region.sType != VK_STRUCTURE_TYPE_BUFFER_IMAGE_COPY_2 || region.imageSubresource.aspectMask == 0) fake_fail("vkCmdCopyBufferToImage2: region");
		if (region.imageSubresource.mipLevel != 0 || region.imageSubresource.baseArrayLayer != 0) continue;
		ulong rows = (ulong)region.imageExtent.height * ((region.bufferRowLength ?: region.imageExtent.width)) * 4;
		fake_copy_rows(record_of((ulong)info.dstImage), fake_buffer_memory(info.srcBuffer, region.bufferOffset, rows), region.bufferRowLength, region.imageOffset, region.imageExtent, true);
	}
}

fn void fake_copy_image_to_buffer(VkCommandBuffer cb, VkCopyImageToBufferInfo2* info)
{
	FakeRecord* record = cb_of(cb, "vkCmdCopyImageToBuffer2");
	if (record.rendering || info.sType != VK_STRUCTURE_TYPE_COPY_IMAGE_TO_BUFFER_INFO_2 || info.srcImageLayout != VK_IMAGE_LAYOUT_GENERAL) fake_fail("vkCmdCopyImageToBuffer2");
	for (uint i = 0; i < info.regionCount; i++)
	{
		VkBufferImageCopy2* region = &info.pRegions[i];
		if (region.sType != VK_STRUCTURE_TYPE_BUFFER_IMAGE_COPY_2 || region.imageSubresource.aspectMask == 0) fake_fail("vkCmdCopyImageToBuffer2: region");
		if (region.imageSubresource.mipLevel != 0 || region.imageSubresource.baseArrayLayer != 0) continue;
		ulong rows = (ulong)region.imageExtent.height * ((region.bufferRowLength ?: region.imageExtent.width)) * 4;
		fake_copy_rows(record_of((ulong)info.srcImage), fake_buffer_memory(info.dstBuffer, region.bufferOffset, rows), region.bufferRowLength, region.imageOffset, region.imageExtent, false);
	}
}

fn void fake_copy_buffer(VkCommandBuffer cb, VkCopyBufferInfo2* info)
{
	FakeRecord* record = cb_of(cb, "vkCmdCopyBuffer2");
	if (record.rendering || info.sType != VK_STRUCTURE_TYPE_COPY_BUFFER_INFO_2) fake_fail("vkCmdCopyBuffer2");
	for (uint i = 0; i < info.regionCount; i++)
	{
		VkBufferCopy2* region = &info.pRegions[i];
		mem::move(fake_buffer_memory(info.dstBuffer, region.dstOffset, region.size), fake_buffer_memory(info.srcBuffer, region.srcOffset, region.size), (sz)region.size);
	}
}

fn void fake_bind_index_buffer2(VkCommandBuffer cb, VkBuffer buffer, VkDeviceSize offset, VkDeviceSize size, VkIndexType type)
{
	cb_of(cb, "vkCmdBindIndexBuffer2");
	fake_buffer_memory(buffer, offset, size);
	if (type != VK_INDEX_TYPE_UINT16 && type != VK_INDEX_TYPE_UINT32) fake_fail("vkCmdBindIndexBuffer2: index type");
}

fn void fake_draw_indirect_buffer(VkCommandBuffer cb, VkBuffer buffer, VkDeviceSize offset, uint count, uint stride)
{
	FakeRecord* record = cb_of(cb, "vkCmdDraw*Indirect");
	if (!record.rendering || !record.pipeline_bound) fake_fail("an indirect draw needs a render pass and a pipeline");
	if (count == 0 || stride == 0) fake_fail("an indirect draw: count and stride");
	fake_buffer_memory(buffer, offset, (ulong)(count - 1) * stride + 16);
}

fn void fake_dispatch_indirect_buffer(VkCommandBuffer cb, VkBuffer buffer, VkDeviceSize offset)
{
	FakeRecord* record = cb_of(cb, "vkCmdDispatchIndirect");
	if (record.rendering || !record.pipeline_bound) fake_fail("vkCmdDispatchIndirect: inside a render pass or without a pipeline");
	fake_buffer_memory(buffer, offset, 12);
}

fn void fake_barrier(VkCommandBuffer cb, VkDependencyInfo* info)
{
	cb_of(cb, "vkCmdPipelineBarrier2");
	if (info.sType != VK_STRUCTURE_TYPE_DEPENDENCY_INFO) fake_fail("vkCmdPipelineBarrier2: sType");
}

fn void fake_set_viewport(VkCommandBuffer cb, uint count, VkViewport* viewports)
{
	cb_of(cb, "vkCmdSetViewportWithCount");
	if (count != 1 || viewports[0].width <= 0 || viewports[0].height <= 0) fake_fail("vkCmdSetViewportWithCount");
}

fn void fake_cmd_noop(VkCommandBuffer cb) { cb_of(cb, "a vkCmd"); }

fn VkResult fake_write_descriptors(VkDevice device, uint count, void* infos, VkHostAddressRangeEXT* destinations)
{
	for (uint i = 0; i < count; i++)
	{
		if (!destinations[i].address || destinations[i].size == 0) fake_fail("vkWrite*DescriptorsEXT: destination");
		mem::set(destinations[i].address, 0xAB, (sz)destinations[i].size);
	}
	return VK_SUCCESS;
}

// ---- The entry points -----------------------------------------------------------------------------------------------------------

struct FakeEntry { ZString name; PFN_vkVoidFunction function; }

FakeEntry[256] fake_table;
usz fake_table_count;

macro void fake_add(ZString name, f) @private
{
	fake_table[fake_table_count++] = { name, (PFN_vkVoidFunction)(void*)f };
}

fn void fake_init() @private
{
	if (fake_table_count) return;
	fake_add("vkCreateInstance", &fake_create_instance);
	fake_add("vkEnumerateInstanceVersion", &fake_enumerate_instance_version);
	fake_add("vkEnumerateInstanceExtensionProperties", &fake_enumerate_instance_extensions);
	fake_add("vkEnumerateInstanceLayerProperties", &fake_enumerate_instance_layers);
	fake_add("vkDestroyInstance", &fake_destroy_instance);
	fake_add("vkEnumeratePhysicalDevices", &fake_enumerate_physical_devices);
	fake_add("vkGetPhysicalDeviceProperties2", &fake_get_properties2);
	fake_add("vkGetPhysicalDeviceFeatures2", &fake_get_features2);
	fake_add("vkGetPhysicalDeviceMemoryProperties", &fake_get_memory_properties);
	fake_add("vkGetPhysicalDeviceQueueFamilyProperties", &fake_get_queue_families);
	fake_add("vkGetPhysicalDeviceFormatProperties2", &fake_get_format_properties2);
	fake_add("vkGetPhysicalDeviceImageFormatProperties2", &fake_get_image_format_properties2);
	fake_add("vkEnumerateDeviceExtensionProperties", &fake_enumerate_device_extensions);
	fake_add("vkCreateDevice", &fake_create_device);
	fake_add("vkGetDeviceProcAddr", &vkGetDeviceProcAddr);
	fake_add("vkDestroyDevice", &fake_destroy_device);
	fake_add("vkGetDeviceQueue", &fake_get_device_queue);
	fake_add("vkCreateBuffer", &fake_create_buffer);
	fake_add("vkDestroyBuffer", &fake_destroy_buffer);
	fake_add("vkGetBufferMemoryRequirements", &fake_get_buffer_requirements);
	fake_add("vkGetDeviceBufferMemoryRequirements", &fake_get_device_buffer_requirements);
	fake_add("vkAllocateMemory", &fake_allocate_memory);
	fake_add("vkFreeMemory", &fake_free_memory);
	fake_add("vkBindBufferMemory", &fake_bind_buffer_memory);
	fake_add("vkMapMemory", &fake_map_memory);
	fake_add("vkUnmapMemory", &fake_unmap_memory);
	fake_add("vkGetBufferDeviceAddress", &fake_get_buffer_address);
	fake_add("vkCreateImage", &fake_create_image);
	fake_add("vkDestroyImage", &fake_destroy_image);
	fake_add("vkBindImageMemory", &fake_bind_image_memory);
	fake_add("vkGetDeviceImageMemoryRequirements", &fake_get_device_image_requirements);
	fake_add("vkCreateImageView", &fake_create_image_view);
	fake_add("vkDestroyImageView", &fake_destroy_image_view);
	fake_add("vkCreateSemaphore", &fake_create_semaphore);
	fake_add("vkDestroySemaphore", &fake_destroy_semaphore);
	fake_add("vkGetSemaphoreCounterValue", &fake_get_semaphore_value);
	fake_add("vkWaitSemaphores", &fake_wait_semaphores);
	fake_add("vkCreateFence", &fake_create_fence);
	fake_add("vkDestroyFence", &fake_destroy_fence);
	fake_add("vkWaitForFences", &fake_wait_for_fences);
	fake_add("vkResetFences", &fake_reset_fences);
	fake_add("vkGetFenceStatus", &fake_get_fence_status);
	fake_add("vkCreateCommandPool", &fake_create_record_handle);
	fake_add("vkDestroyCommandPool", &fake_destroy_pool);
	fake_add("vkResetCommandPool", &fake_reset_command_pool);
	fake_add("vkAllocateCommandBuffers", &fake_allocate_command_buffers);
	fake_add("vkFreeCommandBuffers", &fake_free_command_buffers);
	fake_add("vkBeginCommandBuffer", &fake_begin_command_buffer);
	fake_add("vkEndCommandBuffer", &fake_end_command_buffer);
	fake_add("vkQueueSubmit2", &fake_queue_submit2);
	fake_add("vkDeviceWaitIdle", &fake_ok);
	fake_add("vkCreateQueryPool", &fake_create_query_pool);
	fake_add("vkDestroyQueryPool", &fake_destroy_query_pool);
	fake_add("vkResetQueryPool", &fake_reset_query_pool);
	fake_add("vkGetQueryPoolResults", &fake_get_query_results);
	fake_add("vkCmdWriteTimestamp2", &fake_write_timestamp);
	fake_add("vkCmdPipelineBarrier2", &fake_barrier);
	fake_add("vkCreateGraphicsPipelines", &fake_create_graphics_pipelines);
	fake_add("vkCreateComputePipelines", &fake_create_compute_pipelines);
	fake_add("vkDestroyPipeline", &fake_destroy_pipeline);
	fake_add("vkCmdBindPipeline", &fake_bind_pipeline);
	fake_add("vkCmdSetViewportWithCount", &fake_set_viewport);
	fake_add("vkCmdSetScissorWithCount", &fake_cmd_noop);
	fake_add("vkCmdSetDepthTestEnable", &fake_cmd_noop);
	fake_add("vkCmdSetDepthWriteEnable", &fake_cmd_noop);
	fake_add("vkCmdSetDepthCompareOp", &fake_cmd_noop);
	fake_add("vkCmdSetStencilTestEnable", &fake_cmd_noop);
	fake_add("vkCmdSetStencilOp", &fake_cmd_noop);
	fake_add("vkCmdSetStencilCompareMask", &fake_cmd_noop);
	fake_add("vkCmdSetStencilWriteMask", &fake_cmd_noop);
	fake_add("vkCmdSetStencilReference", &fake_cmd_noop);
	fake_add("vkCmdBeginRendering", &fake_begin_rendering);
	fake_add("vkCmdEndRendering", &fake_end_rendering);
	fake_add("vkCmdDraw", &fake_draw);
	fake_add("vkCmdDrawIndexed", &fake_draw_indexed);
	fake_add("vkCmdDispatch", &fake_dispatch);
	fake_add("vkWriteSamplerDescriptorsEXT", &fake_write_descriptors);
	fake_add("vkWriteResourceDescriptorsEXT", &fake_write_descriptors);
	fake_add("vkCmdBindSamplerHeapEXT", &fake_bind_heap);
	fake_add("vkCmdBindResourceHeapEXT", &fake_bind_heap);
	fake_add("vkCmdPushDataEXT", &fake_push_data);
	fake_add("vkCmdBindIndexBuffer3KHR", &fake_bind_index_buffer);
	fake_add("vkCmdDrawIndirect2KHR", &fake_draw_indirect2);
	fake_add("vkCmdDrawIndexedIndirect2KHR", &fake_draw_indirect2);
	fake_add("vkCmdDispatchIndirect2KHR", &fake_dispatch_indirect2);
	fake_add("vkCmdCopyMemoryKHR", &fake_copy_memory);
	fake_add("vkCmdCopyMemoryToImageKHR", &fake_copy_to_image);
	fake_add("vkCmdCopyImageToMemoryKHR", &fake_copy_from_image);
	fake_add("vkCmdDrawMeshTasksEXT", &fake_draw_mesh_tasks);
	fake_add("vkCmdDrawMeshTasksIndirect2EXT", &fake_draw_indirect2);
	fake_add("vkGetPhysicalDeviceProperties", &fake_get_properties);
	fake_add("vkCmdBindIndexBuffer2", &fake_bind_index_buffer2);
	fake_add("vkCmdDrawIndirect", &fake_draw_indirect_buffer);
	fake_add("vkCmdDrawIndexedIndirect", &fake_draw_indirect_buffer);
	fake_add("vkCmdDrawMeshTasksIndirectEXT", &fake_draw_indirect_buffer);
	fake_add("vkCmdDispatchIndirect", &fake_dispatch_indirect_buffer);
	fake_add("vkCmdCopyBuffer2", &fake_copy_buffer);
	fake_add("vkCmdCopyBufferToImage2", &fake_copy_buffer_to_image);
	fake_add("vkCmdCopyImageToBuffer2", &fake_copy_image_to_buffer);
	fake_add("vkCreateWin32SurfaceKHR", &fake_create_surface);
	fake_add("vkDestroySurfaceKHR", &fake_destroy_surface);
	fake_add("vkGetPhysicalDeviceSurfaceSupportKHR", &fake_surface_support);
	fake_add("vkGetPhysicalDeviceSurfaceCapabilitiesKHR", &fake_surface_capabilities);
	fake_add("vkGetPhysicalDeviceSurfaceCapabilities2KHR", &fake_surface_capabilities2);
	fake_add("vkGetPhysicalDeviceSurfaceFormatsKHR", &fake_surface_formats);
	fake_add("vkCreateSwapchainKHR", &fake_create_swapchain);
	fake_add("vkDestroySwapchainKHR", &fake_destroy_swapchain);
	fake_add("vkGetSwapchainImagesKHR", &fake_get_swapchain_images);
	fake_add("vkAcquireNextImageKHR", &fake_acquire_next_image);
	fake_add("vkQueuePresentKHR", &fake_queue_present);
}

fn PFN_vkVoidFunction fake_lookup(ZString name) @private
{
	fake_init();
	for (usz i = 0; i < fake_table_count; i++)
	{
		if (libc::strcmp(fake_table[i].name, name) == 0) return fake_table[i].function;
	}
	return null;
}

fn PFN_vkVoidFunction vkGetDeviceProcAddr(VkDevice device, ZString name) @export("vkGetDeviceProcAddr") => fake_lookup(name);

fn PFN_vkVoidFunction vkGetInstanceProcAddr(VkInstance instance, ZString name) @export("vkGetInstanceProcAddr") => fake_lookup(name);
