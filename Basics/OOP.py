# 1. Creating a Class
class Student:
    def __init__(self, name, age): # constructor
        self.name = name
        self.age = age

    def show(self):
        print(f"Name: {self.name}, Age: {self.age}")

#Create Object
s1 = Student("Ashish", 25)
s1.show()


# 2. Inheritance
class Person:
    def __init__(self, name):
        self.name = name

    def greet(self):
        print("Hello, my name is", self.name)

class Student(Person): # student inherit Person
    def study(self):
        print(self.name, "is studying.")

s = Student("Ashish");
s.greet()
s.study()


# 3. Encapsulation (private variables)
# 👉 In Python, encapsulation is achieved using:

# Public members → accessible everywhere.
# Protected members (_variable) → meant for internal use (convention).
# Private members (__variable) → not accessible directly outside the class.


class BankAccount:
    def __init__(self, balance):
        self.__balance = balance  # private

    def deposite(self, amount):
        self.__balance += amount

    def getBalance(self):
        return self.__balance
    

account = BankAccount(1000)
account.deposite(10)
print(account.getBalance())