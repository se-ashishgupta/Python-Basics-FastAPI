# Write a program that:

# Asks the user for a number.

# Prints its square and square root.

import math

number = int(input("Enter a Number: "))

print("Square Root of Number is", math.sqrt(number))
print("Square of Number is", math.pow(number, 2))
print("Square of Number is", number ** 2)

# You can make your message nicer using f-strings:
print(f"Square of {number} is {math.sqrt(number)}")