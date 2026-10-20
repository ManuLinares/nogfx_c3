module timeline;
import std::io, nogfx;

// A copy chain over three queues: the work of each queue waits for the one before it, and the submissions are made backwards, consumers
// first, so every wait is for a value that no submitted work has signaled yet. A successor buffer on the last queue is submitted after the
// consumer, and the CPU waits for that, by polling or by waiting: when it is done everything before it is done too (transitive retirement).
// With one timeline for all queues, or one each. Works on both backends; it needs three queues and skips otherwise.

int failures;

macro check(bool condition, String what)
{
	io::printfn("%s  %s", condition ? "pass" : "FAIL", what);
	if (!condition) failures++;
}

fn bool check_retirement(Device* device, bool shared_timeline, bool poll)
{
	bool valid = true;
	for (uint iteration = 0; iteration < 64; iteration++)
	{
		TimelineSemaphore*[3] timelines;
		TimelinePoint[3] points;
		CommandPool*[3] pools;
		CommandBuffer*[3] commands;
		for (uint queue = 0; queue < 3; queue++)
		{
			timelines[queue] = shared_timeline && queue ? timelines[0] : device.create_timeline_semaphore(7)!!;
			points[queue] = { timelines[queue], shared_timeline ? 8UL + queue : 8UL };
			pools[queue] = device.create_command_pool(queue)!!;
			commands[queue] = pools[queue].begin_commands()!!;
			valid &= timelines[queue].completed_value() == 7;
		}
		GpuHeap upload = device.create_gpu_heap(256)!!;
		GpuHeap first = device.create_gpu_heap(256, GPU_ONLY)!!;
		GpuHeap second = device.create_gpu_heap(256, GPU_ONLY)!!;
		GpuHeap readback = device.create_gpu_heap(256, READBACK)!!;
		uint[64] expected;
		for (uint i = 0; i < 64; i++) expected[i] = (iteration << 16) | i;
		mem::copy(upload.cpu, &expected, 256);
		commands[0].copy_memory(upload.range(), first.range());
		commands[1].copy_memory(first.range(), second.range());
		commands[2].copy_memory(second.range(), readback.range());
		commands[2].barrier({ .transfer = true }, { .transfer_write = true }, { .host = true }, { .host_read = true });
		foreach (queue_commands : commands) queue_commands.end();
		CommandBuffer* successor = pools[2].begin_commands()!!;
		successor.end();
		TimelinePoint completion = { timelines[2], points[2].value + 1 };
		// Consumers first.
		device.submit({ .commands = { commands[2] }, .waits = { points[1] }, .completion = points[2] }, 2);
		device.submit({ .commands = { successor }, .completion = completion }, 2);
		device.submit({ .commands = { commands[1] }, .waits = { points[0] }, .completion = points[1] }, 1);
		device.submit({ .commands = { commands[0] }, .completion = points[0] }, 0);
		if (poll)
		{
			while (completion.semaphore.completed_value() < completion.value) {}
		}
		else
		{
			completion.wait();
		}
		valid &= mem::equals(readback.cpu[:256], ((char*)&expected)[:256]);
		for (uint queue = 0; queue < 3; queue++)
		{
			valid &= points[queue].semaphore.completed_value() >= points[queue].value;
			pools[queue].destroy();
		}
		upload.destroy();
		first.destroy();
		second.destroy();
		readback.destroy();
		for (uint queue = 0; queue < (shared_timeline ? 1u : 3u); queue++) timelines[queue].destroy();
	}
	return valid;
}

fn int main()
{
	nogfx::set_error_callback(fn (message, fatal) { io::printfn("callback: %s", message); });
	Device*? created = nogfx::create_device({ .desired_queue_count = 3 });
	if (catch err = created)
	{
		io::printfn("skipped: no device (%s)", err);
		return err == nogfx::UNSUPPORTED ? 0 : 1;
	}
	Device* device = created;
	defer device.destroy();
	if (device.caps.queue_count < 3 || device.caps.device_name == "Fake Vulkan GPU")
	{
		io::printn("skipped: needs three queues and a driver that runs submissions in dependency order");
		return 0;
	}
	for (uint mode = 0; mode < 4; mode++)
	{
		bool shared_timeline = (mode & 1) != 0;
		bool poll = (mode & 2) != 0;
		check(check_retirement(device, shared_timeline, poll), string::tformat("transitive retirement, %s timeline, %s", shared_timeline ? "shared" : "separate", poll ? "poll" : "wait"));
	}
	device.wait_idle();
	io::printfn(failures ? "%d check(s) FAILED" : "all checks passed", failures);
	return failures ? 1 : 0;
}
