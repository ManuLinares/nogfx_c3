module nogfx @feat(!DARWIN, NOGFX_VULKAN);
import std::io, std::os, libc;

// ---- Textures -------------------------------------------------------------------------------------------------------------------

// Moves all of an image from one layout to another.
fn void record_layout_transition(VkFns* vk, VkCommandBuffer command_buffer, VkImage image, VkImageAspectFlags aspect, VkImageLayout from, VkImageLayout to,
	VkPipelineStageFlags2 source_stage, VkAccessFlags2 source_access, VkPipelineStageFlags2 destination_stage, VkAccessFlags2 destination_access) @private
{
	VkImageMemoryBarrier2 barrier = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
		.srcStageMask = source_stage,
		.srcAccessMask = source_access,
		.dstStageMask = destination_stage,
		.dstAccessMask = destination_access,
		.oldLayout = from,
		.newLayout = to,
		.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.image = image,
		.subresourceRange = { .aspectMask = aspect, .levelCount = VK_REMAINING_MIP_LEVELS, .layerCount = VK_REMAINING_ARRAY_LAYERS },
	};
	VkDependencyInfo dependency = { .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &barrier };
	vk.vkCmdPipelineBarrier2(command_buffer, &dependency);
}

// The image create info keeps pointers into its own storage: use it where it is made.
struct PreparedTexture
{
	VkFormat[FORMAT_COUNT] view_formats;
	VkImageFormatListCreateInfo format_list;
	VkImageCreateInfo image_info;
}

<*
 The image that a TextureDesc describes. With mutable_format it can be viewed as every compatible format the device can sample or store.
*>
fn void Device.prepare_texture(&self, TextureDesc desc, PreparedTexture* output)
{
	*output = {};
	output.format_list.sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_LIST_CREATE_INFO;
	VkImageUsageFlags usage;
	if (desc.usage.sampled) usage |= VK_IMAGE_USAGE_SAMPLED_BIT;
	if (desc.usage.storage) usage |= VK_IMAGE_USAGE_STORAGE_BIT;
	if (desc.usage.color_attachment) usage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
	if (desc.usage.depth_stencil_attachment) usage |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
	if (desc.usage.transfer_source) usage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
	if (desc.usage.transfer_destination) usage |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;

	VkFormat format = vk_format(desc.format);
	VkFormatFeatureFlags2 sampled_features = required_format_features({ .sampled = true });
	VkFormatFeatureFlags2 storage_features = required_format_features({ .storage = true });
	uint view_format_count = 1;
	output.view_formats[0] = format;
	for (uint value = 1; desc.mutable_format && value < FORMAT_COUNT; value++)
	{
		Format view_format = (Format)value;
		if (view_format == desc.format || !compatible_view_formats(desc.format, view_format)) continue;
		VkFormatFeatureFlags2 view_features = self.format_features[value];
		bool sampled = desc.usage.sampled && (view_features & sampled_features) == sampled_features;
		bool storage = desc.usage.storage && (view_features & storage_features) == storage_features;
		if (!sampled && !storage) continue;
		output.view_formats[view_format_count++] = vk_format(view_format);
	}

	VkImageCreateFlags image_flags;
	if (desc.type == CUBE || desc.type == CUBE_ARRAY) image_flags |= VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
	if (view_format_count > 1) image_flags |= VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT;

	output.format_list.viewFormatCount = view_format_count;
	output.format_list.pViewFormats = &output.view_formats[0];
	output.image_info = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
		.pNext = view_format_count > 1 ? (void*)&output.format_list : null,
		.flags = image_flags,
		.imageType = vk_image_type(desc.type),
		.format = format,
		.extent = { .width = desc.extent.x, .height = desc.extent.y, .depth = desc.extent.z },
		.mipLevels = desc.mip_levels,
		.arrayLayers = desc.layer_count,
		.samples = VK_SAMPLE_COUNT_1_BIT,
		.tiling = VK_IMAGE_TILING_OPTIMAL,
		.usage = usage,
		.sharingMode = self.sharing_mode(),
		.queueFamilyIndexCount = self.queue_family_count,
		.pQueueFamilyIndices = &self.queue_families[0],
		.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
	};
}

<*
 The size and alignment a texture needs in a TextureHeap. Placements must satisfy the alignment. UNSUPPORTED when the device cannot
 create such a texture.
*>
fn SizeAlign? Device.get_texture_size_align(&self, TextureDesc desc)
{
	desc = normalized(desc);
	PreparedTexture texture;
	self.prepare_texture(desc, &texture);
	if (!self.supports_image_create_info(&texture.image_info))
	{
		report("The GPU cannot create the requested texture format, usage, or extent.");
		return UNSUPPORTED~;
	}
	VkMemoryRequirements requirements = self.image_memory_requirements(&texture.image_info);
	return { requirements.size, requirements.alignment };
}

<*
 Places a texture in a texture heap. Records nothing: views and descriptors can be made at once, and `initialize_texture` moves the image
 out of its undefined layout before the first use on the GPU. Textures can share storage while their uses on the GPU are disjoint.
*>
fn Texture*? Device.create_texture(&self, TextureDesc desc, TextureHeap heap, ulong offset)
{
	Device* device = self;
	assert(heap.owner && heap.owner.device == device);
	desc = normalized(desc);
	PreparedTexture texture;
	device.prepare_texture(desc, &texture);
	Texture* result = mem::new(Texture, {
		.device = device,
		.width = desc.extent.x,
		.height = desc.extent.y,
		.depth = desc.extent.z,
		.layer_count = desc.layer_count,
		.type = desc.type,
		.format = desc.format,
	});
	defer catch mem::free(result);
	check_vk(device.vk.vkCreateImage(device.device, &texture.image_info, null, &result.image), "vkCreateImage")!;
	defer catch device.vk.vkDestroyImage(device.device, result.image, null);
	check_vk(device.vk.vkBindImageMemory(device.device, result.image, heap.owner.memory, offset), "vkBindImageMemory")!;
	return result;
}

fn void Texture.destroy(&self)
{
	self.device.vk.vkDestroyImage(self.device.device, self.image, null);
	mem::free(self);
}

<*
 Moves a texture's whole image from the undefined layout to GENERAL, outside a render pass, before its first use on the GPU and before reuse
 after writes through an overlapping texture. Discards the contents: clear or overwrite before reading. The caller synchronizes the
 earlier uses of the storage, and makes their writes available, with barriers or timeline waits first: this joins that dependency but does
 not make unrelated writes available again. Orders the transition before later accesses on this queue.
*>
fn void CommandBuffer.initialize_texture(&self, Texture* texture)
{
	assert(texture && self.device == texture.device);
	record_layout_transition(self.vk, self.command_buffer, texture.image, image_aspects(texture.format), VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
		VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, 0, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT);
}

<*
 A 2D view of one mip level and slice, for rendering to.
*>
fn RenderView*? Texture.create_render_view(&self, RenderViewDesc desc = {})
{
	RenderView* result = mem::new(RenderView, { .device = self.device, .width = max(self.width >> desc.mip_level, 1u), .height = max(self.height >> desc.mip_level, 1u) });
	VkImageViewCreateInfo view_info = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
		.image = self.image,
		.viewType = VK_IMAGE_VIEW_TYPE_2D,
		.format = vk_format(self.format),
		.subresourceRange = {
			.aspectMask = image_aspects(self.format),
			.baseMipLevel = desc.mip_level,
			.levelCount = 1,
			.baseArrayLayer = desc.slice,
			.layerCount = 1,
		},
	};
	defer catch mem::free(result);
	check_vk(self.device.vk.vkCreateImageView(self.device.device, &view_info, null, &result.view), "vkCreateImageView")!;
	return result;
}

fn void RenderView.destroy(&self)
{
	assert(!self.swapchain_view, "swapchain render views are owned by their swapchain");
	self.device.vk.vkDestroyImageView(self.device.device, self.view, null);
	mem::free(self);
}

// ---- Descriptor heaps -----------------------------------------------------------------------------------------------------------

fn DescriptorHeap? Device.create_descriptor_heap(&self, uint capacity, ulong stride, DescriptorHeapType type)
{
	assert(capacity != 0);
	GpuHeap storage = self.allocate_descriptor_heap(capacity * stride, type)!;
	return { .device = self, .storage = storage, .descriptors = mem::new_array(char, (sz)((ulong)capacity * stride)).ptr, .capacity = capacity, .stride = stride };
}

fn TextureDescriptorHeap*? Device.create_texture_descriptor_heap(&self, uint capacity)
{
	return mem::new(TextureDescriptorHeap, { .heap = self.create_descriptor_heap(capacity, self.heap_properties.imageDescriptorSize, TEXTURE)! });
}

fn SamplerDescriptorHeap*? Device.create_sampler_descriptor_heap(&self, uint capacity)
{
	return mem::new(SamplerDescriptorHeap, { .heap = self.create_descriptor_heap(capacity, self.heap_properties.samplerDescriptorSize, SAMPLER)! });
}

fn void DescriptorHeap.destroy(&self)
{
	self.storage.destroy();
	mem::free(self.descriptors);
	mem::free(self);
}

// Where a descriptor is written, in the CPU array.
fn VkHostAddressRangeEXT DescriptorHeap.slot(&self, uint index) => { .address = self.descriptors + (ulong)index * self.stride, .size = self.stride };

// Copies the descriptor that was written to the CPU array to the mapped heap.
fn void DescriptorHeap.publish(&self, uint index) => mem::copy(self.storage.cpu + (ulong)index * self.stride, self.descriptors + (ulong)index * self.stride, (sz)self.stride);

// The copy of both kinds of heap. The public `copy_from` methods take their own kind, so a texture heap cannot copy from a sampler heap.
fn void copy_descriptors(DescriptorHeap* destination, uint destination_index, DescriptorHeap* source, uint source_index, uint count) @private
{
	assert(source.device == destination.device && source.stride == destination.stride && source_index <= source.capacity && count <= source.capacity - source_index);
	assert(destination_index <= destination.capacity && count <= destination.capacity - destination_index);
	if (!count || (source == destination && source_index == destination_index)) return;
	ulong offset = (ulong)destination_index * destination.stride;
	mem::move(destination.descriptors + offset, source.descriptors + (ulong)source_index * source.stride, (sz)(count * source.stride));
	mem::copy(destination.storage.cpu + offset, destination.descriptors + offset, (sz)(count * destination.stride));
}

<*
 Copies `count` descriptors between heaps of the same device. The ranges may overlap.
*>
fn void TextureDescriptorHeap.copy_from(&self, uint destination_index, TextureDescriptorHeap* source, uint source_index, uint count)
{
	copy_descriptors(self, destination_index, source, source_index, count);
}

fn void SamplerDescriptorHeap.copy_from(&self, uint destination_index, SamplerDescriptorHeap* source, uint source_index, uint count)
{
	copy_descriptors(self, destination_index, source, source_index, count);
}

<*
 Writes a descriptor for `texture` into slot `index`. Do not modify a slot until all commands using its previous contents completed.
*>
fn void TextureDescriptorHeap.write(&self, uint index, Texture* texture, TextureDescriptorType type, TextureDescriptorDesc desc = {})
{
	assert(index < self.capacity && texture.device == self.device);
	VkImageAspectFlags descriptor_aspect;
	switch (desc.aspect)
	{
		case AUTOMATIC:
			TextureFormatInfo info = get_texture_format_info(texture.format);
			descriptor_aspect = info.depth ? VK_IMAGE_ASPECT_DEPTH_BIT : info.stencil ? VK_IMAGE_ASPECT_STENCIL_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
		case COLOR: descriptor_aspect = VK_IMAGE_ASPECT_COLOR_BIT;
		case DEPTH: descriptor_aspect = VK_IMAGE_ASPECT_DEPTH_BIT;
		case STENCIL: descriptor_aspect = VK_IMAGE_ASPECT_STENCIL_BIT;
	}
	VkImageViewUsageCreateInfo view_usage = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO,
		.usage = type == SAMPLED ? VK_IMAGE_USAGE_SAMPLED_BIT : VK_IMAGE_USAGE_STORAGE_BIT,
	};
	VkImageViewCreateInfo view_info = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
		.pNext = &view_usage,
		.image = texture.image,
		.viewType = vk_view_type(texture.type),
		.format = vk_format(desc.format == UNDEFINED ? texture.format : desc.format),
		.subresourceRange = {
			.aspectMask = descriptor_aspect,
			.baseMipLevel = desc.base_mip,
			.levelCount = desc.mip_count ?: VK_REMAINING_MIP_LEVELS,
			.baseArrayLayer = desc.base_layer,
			.layerCount = desc.layer_count ?: VK_REMAINING_ARRAY_LAYERS,
		},
	};
	VkImageDescriptorInfoEXT image_descriptor = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_DESCRIPTOR_INFO_EXT,
		.pView = &view_info,
		.layout = VK_IMAGE_LAYOUT_GENERAL,
	};
	VkResourceDescriptorInfoEXT descriptor_info = {
		.sType = VK_STRUCTURE_TYPE_RESOURCE_DESCRIPTOR_INFO_EXT,
		.type = type == SAMPLED ? VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
		.data = { .pImage = &image_descriptor },
	};
	VkHostAddressRangeEXT slot = self.slot(index);
	assert_vk(self.device.vk.vkWriteResourceDescriptorsEXT(self.device.device, 1, &descriptor_info, &slot));
	self.publish(index);
}

<*
 Writes a sampler into slot `index`. Do not modify a slot until all commands using its previous contents completed.
*>
fn void SamplerDescriptorHeap.write(&self, uint index, SamplerDesc desc = {})
{
	assert(index < self.capacity);
	VkSamplerCreateInfo sampler_info = {
		.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
		.magFilter = vk_filter(desc.mag_filter),
		.minFilter = vk_filter(desc.min_filter),
		.mipmapMode = vk_mipmap_mode(desc.mip_filter),
		.addressModeU = (VkSamplerAddressMode)desc.address_u.ordinal,
		.addressModeV = (VkSamplerAddressMode)desc.address_v.ordinal,
		.addressModeW = (VkSamplerAddressMode)desc.address_w.ordinal,
		.anisotropyEnable = desc.anisotropic ? VK_TRUE : VK_FALSE,
		.maxAnisotropy = desc.anisotropic ? 4.0f : 1.0f,
		.compareEnable = desc.compare_enabled ? VK_TRUE : VK_FALSE,
		.compareOp = vk_compare_op(desc.compare),
		.maxLod = VK_LOD_CLAMP_NONE,
	};
	VkHostAddressRangeEXT slot = self.slot(index);
	assert_vk(self.device.vk.vkWriteSamplerDescriptorsEXT(self.device.device, 1, &sampler_info, &slot));
	self.publish(index);
}
