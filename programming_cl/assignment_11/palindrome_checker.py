def is_palindrome(s):
    """
    Returns True if the input string is a palindrome, ignoring case and non-alphanumeric characters.
    """
    
    s = ''.join(char.lower() for char in s if char.isalnum())
    return s == s[::-1]