import math
# display the maximum number that can be stored in a numerical variable
number = 10e300
# 32 bit number (2^31) is the biggest
while True: #math.isinf(number) == 0:
    if math.isinf(number) == 1:
        break
    else:
        number = number + 1
    #print(str(number))
print(str(number - 1)) # big number goes straight to infinity