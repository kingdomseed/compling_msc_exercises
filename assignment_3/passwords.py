from fragments import get_random_text

def easy_password(length: int = 1):
    password = ""
    while length > len(password):
        text = get_random_text()
        if len(password + text) <= length and length - len(password + text) != 1:
            password += text
    return password
