

def decipher(msg, perm):
    """Deciphers a message using a given permutation.
    Args:
        msg (str): The message to decipher.
        perm (str): A permutation of the alphabet used for deciphering.
    Returns:
        str: The deciphered message.
    """
    alphabet = "abcdefghijklmnopqrstuvwxyz"


perm = "wnoegbjpkyxlfiuastqhvmcrzd"
print(decipher("wnoeg", perm))
print(decipher("rzd", perm))
print(decipher("azhpui", perm))
print(decipher("hpg ntuci bur yvfaq umgt hpg euj", perm))