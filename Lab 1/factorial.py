x = input("Please enter a number: ")
factorial = 1
for i in range(1,int(x)+1):
    factorial = factorial*i

print("factorial of " + x + " = " + str(factorial))