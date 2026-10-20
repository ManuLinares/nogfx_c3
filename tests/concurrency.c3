module concurrency;
import std::io, std::thread, nogfx;

// Heaps are created, used and destroyed from several threads at once, each thread with its own queue, command pool and semaphore, while one
// more thread only creates and destroys heaps. Every copy has to find the heap it is about to copy from through the lock free address
// index that the registrations and removals keep republishing. A wrong lookup is a crash, a failed assert or wrong data.

const uint WORKERS = 4;
const uint ROUNDS = 400;

Device* device;
int[WORKERS] bad_rounds;

struct Worker
{
	uint index;
}

Worker[WORKERS] workers;

fn int work(void* arg)
{
	Worker* worker = (Worker*)arg;
	uint queue = worker.index;
	CommandPool* pool = device.create_command_pool(queue)!!;
	TimelineSemaphore* timeline = device.create_timeline_semaphore()!!;
	defer { pool.destroy(); timeline.destroy(); }
	for (uint round = 0; round < ROUNDS; round++)
	{
		ulong size = 4096 * (ulong)(1 + (round + worker.index) % 8);
		GpuHeap source = device.create_gpu_heap(size, CPU_VISIBLE)!!;
		GpuHeap target = device.create_gpu_heap(size, READBACK)!!;
		uint* data = (uint*)source.cpu;
		for (uint i = 0; i < size / 4; i++) data[i] = i * 7 + round + worker.index * 1000;
		pool.reset();
		CommandBuffer* commands = pool.begin_commands()!!;
		// Copies from the middle of one heap into the middle of another make the lookup resolve an offset, not just a base.
		commands.copy_memory({ source.gpu + 1024, size - 2048 }, { target.gpu + 512, size - 1536 });
		commands.end();
		device.submit({ .commands = { commands }, .completion = { timeline, (ulong)round + 1 } }, queue);
		TimelinePoint done = { timeline, (ulong)round + 1 };
		done.wait();
		uint* result = (uint*)(target.cpu + 512);
		bool ok = true;
		for (uint i = 0; i < (size - 2048) / 4; i++) ok &= result[i] == (i + 256) * 7 + round + worker.index * 1000;
		if (!ok) bad_rounds[worker.index]++;
		source.destroy();
		target.destroy();
	}
	return 0;
}

fn int churn(void* arg)
{
	for (uint round = 0; round < ROUNDS * 20; round++)
	{
		GpuHeap a = device.create_gpu_heap((ulong)4096 + (ulong)(round % 5) * 4096, CPU_VISIBLE)!!;
		GpuHeap b = device.create_gpu_heap(8192, GPU_ONLY)!!;
		b.destroy();
		a.destroy();
	}
	return 0;
}

fn int main()
{
	device = nogfx::create_device({ .desired_queue_count = WORKERS })!!;
	defer device.destroy();
	Thread[WORKERS] threads;
	Thread churner;
	churner.create(&churn, null)!!;
	foreach (i, &thread : threads)
	{
		workers[i] = { (uint)i };
		thread.create(&work, &workers[i])!!;
	}
	foreach (&thread : threads) thread.join()!!;
	churner.join()!!;
	device.wait_idle();
	int total_bad;
	foreach (bad : bad_rounds) total_bad += bad;
	io::printfn("%d threads x %d rounds of create / copy / destroy, plus %d more heaps created and destroyed alongside: %d wrong results", WORKERS, ROUNDS, ROUNDS * 20, total_bad);
	io::printn(total_bad == 0 ? "pass  lock free address lookup under concurrent registration" : "FAIL  lock free address lookup");
	return total_bad == 0 ? 0 : 1;
}
