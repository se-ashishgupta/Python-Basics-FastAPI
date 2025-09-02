# 🔹 Part 1: Modules in Python

# A module is just a file with Python code that you can import and use.
# Python already has many built-in modules.

# Example 1: Math Module
import math

print(math.sqrt(25))   # 5.0
print(math.pi)         # 3.14159...

# Example 2: Random Module
import random

print(random.randint(1,10))

# Example 3: Datetime Module
import datetime

today = datetime.date.today()
print("Today date is: ", today)


# 🔹 Part 2: File Handling

# Python can read/write files on your computer.

# Writing to a File, This will create test.txt.
file = open("test.txt", "w") # "w" = write mode
file.write("Hello, This is Ashish Gupta!\n")
file.write("Python is very easy.")
file.close()

# Reading from a File 
file = open("test.txt","r") # "r" = read mode
content = file.read()
print(content)
file.close()

# Appending (Adding) to a File
file = open("test.txt", "a") # "a" = append mode
file.write("\nhello once again.")
file.close()

file = open("test.txt","r") # "r" = read mode
content = file.read()
print(content)
file.close()