from fragments import get_random_text

def easy_password(length: int = 1):
    password = ""
    while length > len(password):
        text = get_random_text()
        if len(password + text) <= length and length - len(password + text) != 1:
            password += text
        elif len(password) + 1 >= length:
            return password
    return password

print(easy_password(10))
print(easy_password(10))
print(easy_password(15))
print(easy_password(2))

# pw_length = int(input("Enter password length: "))
# print(easy_password(pw_length))
