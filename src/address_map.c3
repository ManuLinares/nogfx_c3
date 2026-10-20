module nogfx;
import std::thread, std::atomic;

// ---- Address to buffer lookup ---------------------------------------------------------------------------------------------------
//
// Metal's copy encoders take a buffer, and so do Vulkan's without VK_KHR_device_address_commands, while the API only has a GPU address.
// Every GPU heap is one buffer, so the live heaps are kept as a sorted array of at most 64 records. Readers never lock: they binary search
// and check the version afterwards. Updates are serialized by the version being odd while they run. Ported from BufferAddressMap.hpp.

struct BufferRecord
{
	ulong base;
	ulong size;
	ulong buffer;   // The backend's buffer handle.
}

struct BufferAddressMap
{
	BufferRecord[64] records;
	ulong[64] indices;   // The records sorted by base address.
	ulong sequence;
	ulong count;
	ulong occupied;
}

fn ulong BufferAddressMap.begin_update(&self)
{
	ulong previous = mem::@atomic_load(self.sequence, RELAXED);
	for (;;)
	{
		if (!(previous & 1))
		{
			ulong seen = mem::compare_exchange(&self.sequence, previous, previous + 1, ACQUIRE_RELEASE, RELAXED, false);
			if (seen == previous)
			{
				// Readers observing changed metadata must also observe this writer's odd sequence.
				thread::fence(RELEASE);
				return previous;
			}
			previous = seen;
		}
		else
		{
			previous = mem::@atomic_load(self.sequence, RELAXED);
		}
	}
}

<*
 Adds a heap. Asserts when 64 are live.
*>
fn BufferRecord* BufferAddressMap.insert(&self, ulong base, ulong size, ulong buffer)
{
	ulong previous = self.begin_update();
	assert(self.occupied != ulong::max, "at most 64 live GPU heaps per device");
	uint slot = (uint)(~self.occupied).ctz();
	self.occupied |= (ulong)1 << slot;
	mem::@atomic_store(self.records[slot].base, base, RELAXED);
	mem::@atomic_store(self.records[slot].size, size, RELAXED);
	mem::@atomic_store(self.records[slot].buffer, buffer, RELAXED);
	ulong index = mem::@atomic_load(self.count, RELAXED);
	mem::@atomic_store(self.count, index + 1, RELAXED);
	while (index && mem::@atomic_load(self.records[mem::@atomic_load(self.indices[index - 1], RELAXED)].base, RELAXED) > base)
	{
		mem::@atomic_store(self.indices[index], mem::@atomic_load(self.indices[index - 1], RELAXED), RELAXED);
		index--;
	}
	mem::@atomic_store(self.indices[index], slot, RELAXED);
	mem::@atomic_store(self.sequence, previous + 2, RELEASE);
	return &self.records[slot];
}

fn void BufferAddressMap.remove(&self, BufferRecord* record)
{
	ulong previous = self.begin_update();
	ulong size = mem::@atomic_load(self.count, RELAXED);
	ulong index = 0;
	while (index < size && &self.records[mem::@atomic_load(self.indices[index], RELAXED)] != record) index++;
	assert(index < size);
	for (; index + 1 < size; index++) mem::@atomic_store(self.indices[index], mem::@atomic_load(self.indices[index + 1], RELAXED), RELAXED);
	mem::@atomic_store(self.count, size - 1, RELAXED);
	self.occupied &= ~((ulong)1 << (record - &self.records[0]));
	mem::@atomic_store(self.sequence, previous + 2, RELEASE);
}

<*
 The buffer handle of the heap that `range` lies in, and the offset of the range in it. Asserts when the range is not inside one live heap.
*>
fn ulong BufferAddressMap.resolve(&self, GpuRange range, ulong* offset)
{
	ulong address = (ulong)(uptr)range.gpu;
	for (;;)
	{
		ulong previous = mem::@atomic_load(self.sequence, ACQUIRE);
		if (previous & 1) continue;
		uint first = 0;
		uint last = (uint)mem::@atomic_load(self.count, RELAXED);
		while (first < last)
		{
			uint middle = first + (last - first) / 2;
			BufferRecord* record = &self.records[mem::@atomic_load(self.indices[middle], RELAXED)];
			if (mem::@atomic_load(record.base, RELAXED) <= address)
			{
				first = middle + 1;
			}
			else
			{
				last = middle;
			}
		}
		ulong base;
		ulong size;
		ulong buffer;
		if (first)
		{
			BufferRecord* record = &self.records[mem::@atomic_load(self.indices[first - 1], RELAXED)];
			base = mem::@atomic_load(record.base, RELAXED);
			size = mem::@atomic_load(record.size, RELAXED);
			buffer = mem::@atomic_load(record.buffer, RELAXED);
		}
		thread::fence(ACQUIRE);
		if (mem::@atomic_load(self.sequence, RELAXED) != previous) continue;
		assert(first && address >= base && address - base <= size && range.size <= size - (address - base), "address range is not inside one live GPU heap");
		*offset = address - base;
		return buffer;
	}
}
