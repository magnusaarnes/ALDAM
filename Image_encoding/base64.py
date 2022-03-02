import base64

def encode_image(path):
    """
    Input: 
        path (string): path of the image
    Return:
        b64struing (bytes): a byte string of the image
    """
    with open(path, "rb") as image:
        b64string = base64.b64encode(image.read())
        return b64string

def decode_image(b64string, path):
    """
    Input: 
        base64string (bytes): a byte string of an image
        path (string): the path where the image is saved
    """
    image = open(path, "wb")
    image.write(base64.b64decode(b64string))
    image.close()


def bytes_to_string(b64string):
    """
    Returns byte string as a string
    """
    return b64string.decode('utf-8')

def string_to_bytes(string):
    """
    Retruns the string as bytes
    """
    return bytes(string, 'utf-8')
