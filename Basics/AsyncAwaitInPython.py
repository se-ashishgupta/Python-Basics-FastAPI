# 1. Normal Function
def hello():
    return "Hello"

# 2. Async Function
import asyncio
async def hello():
    print("Hello async")

async def greet():
    print("Start")
    await asyncio.sleep(3) # wait 2 second (non blocking)
    print("End")

# asyncio.run(greet())

# Unlike time.sleep(2) (which blocks everything),
# asyncio.sleep(2) allows other tasks to run in those 2 seconds.

# 4. Running Multiple Async Tasks
import asyncio

async def task(name, seconds):
    print(f"task {name} started")
    await asyncio.sleep(seconds)
    print(f"task {name} finished")

async def main():
    await asyncio.gather(
        task("A", 3),
        task("B", 2),
        task("C", 1)
    )

asyncio.run(main())