# 1. Lists (ordered, changeable)
# Think of a list like a shopping list 🛒.
fruits = ["apple", "banana", "mango"]
print(fruits)
print(fruits[0])
fruits.append("Graps")
print(fruits)


# 2. Tuples (ordered, unchangeable)
# Like a list, but you cannot modify it.
numbers = (1, 2, 3, 4)
print(numbers[1])   # 2


# 3. Sets (unordered, unique items)
# No duplicates allowed.
colors = {"red", "blue", "green", "red"}
print(colors)   # {'red', 'blue', 'green'}  (red appears once only)
colors.add("ashsih")
print(colors)

# 4. Dictionaries (key-value pairs)
# Like a mini database.
student = {
    "name": "Ashish",
    "age": 22,
    "marks": 95
}

print(student)
print(student["name"])   # Ashish
print(student["marks"])  # 95


# 🎯 Quick Recap

# List → [ ] → ordered, editable
# Tuple → ( ) → ordered, not editable
# Set → { } → unordered, unique
# Dictionary → {key: value} → mapping