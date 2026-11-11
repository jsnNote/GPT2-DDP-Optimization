import sys


def parse_argument(keyword):
    if keyword:
        arguments = [arg for arg in sys.argv if keyword in arg]
        option, value = [arg for arg in arguments if keyword == arg.split("=")[0]][
            0
        ].split("=")
    else:
        option, value = (None, False)
    return option, value
