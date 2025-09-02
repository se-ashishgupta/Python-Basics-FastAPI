count = 1

while count <= 5:
    print("hello", count)
    count += 1

for i in range(1, 6):
    print(i)

fruits = ["apple", "banana", "mango"]

for fruit in fruits:
    print(fruit)


for i in range(1, 10):
    if i == 5:
        break   # stop loop when i = 5
    print(i)

for i in range(1, 10):
    if i == 5:
        continue   # skip 5
    print(i)