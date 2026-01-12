
def fizz_buzz(number):
    if  number % 3 == 0 and number % 5 == 0:
        return "FizzBuzz"
    elif number % 5 == 0:
        return "Buzz"
    elif number % 3 == 0:
        return "Fizz"
    else:
        return ":("
    
numbers_to_iterate = range(1, 101)
    
for number in numbers_to_iterate:
    print(fizz_buzz(number))