def decipher(msg, perm):
    """Deciphers a message using a given permutation.
    Args:
        msg (str): The message to decipher.
        perm (str): A permutation of the alphabet used for deciphering.
    Returns:
        str: The deciphered message.
    """
    alphabet = "abcdefghijklmnopqrstuvwxyz"

    # Create a dictionary mapping each letter in the permutation
    # to the alphabet
    mapping = {perm[i]: alphabet[i] for i in range(len(alphabet))}

    # Use the mapping to decipher the message
    deciphered_msg = ""
    for char in msg:
        if char in mapping:
            deciphered_msg += mapping[char]
        else:
            deciphered_msg += char

    return deciphered_msg
