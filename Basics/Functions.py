# 1. Basic Function
def greet():
    print("Hello! Welcome to Python")

greet()   # calling the function

# 2. Function with Parameters
def greeting(name):
    print(f"Hello {name}, Welcome to CaudateAi")

greeting("Ashish")
greeting("Rahul")

# 3. Function with Return Value
def sum(a, b):
    return a+b

result = sum(5,5)
print(result)


# 4. Default Parameters
def greet(name="Student"):
    print("Hello", name)

greet()          # Hello Student
greet("Ashish")  # Hello Ashish

# 5. Multiple Return Values
def math_Operations(a,b):
    return a+b, a-b, a*b

sum_, diff_, div_ = math_Operations(5,5)
print("Sum:", sum_)
print("Difference:", diff_)
print("Product:", div_)