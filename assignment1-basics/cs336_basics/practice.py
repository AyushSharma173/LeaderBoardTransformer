



def decode_bytes(bytes_stuff: bytes):
    return bytes_stuff.decode("utf-8")


print(decode_bytes(bytes([60])))