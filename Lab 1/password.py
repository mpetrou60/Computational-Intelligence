from codecs import BOM_BE


password = input("Please enter a password: ")
m = 1
while m == 1:
    check = input("What do you think the password is? ")
    if password == check:
        print("You got it right! The password is " + password)
        m = 0
    else:
        print("Not quite right, please try again")