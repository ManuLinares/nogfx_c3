module nogfx;

// What both backends use.

fn ulong align_up(ulong value, ulong alignment) @private
{
	assert(alignment != 0);
	return (value + alignment - 1) / alignment * alignment;
}

fn uint or_one(uint value) @private => value ?: 1;

// Zero extents and counts are one. Everything that reads a TextureDesc after creation sees the normalized one.
fn TextureDesc normalized(TextureDesc desc) @private
{
	if (desc.format == UNDEFINED) desc.format = RGBA8_UNORM;
	if ((uint)desc.usage == 0) desc.usage = { .sampled = true };
	desc.extent = { or_one(desc.extent.x), or_one(desc.extent.y), or_one(desc.extent.z) };
	desc.mip_levels = or_one(desc.mip_levels);
	desc.layer_count = or_one(desc.layer_count);
	return desc;
}
