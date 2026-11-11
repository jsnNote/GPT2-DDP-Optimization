import sys


def strint_to_number(num):
    try:
        float(num)
        if len(num.split(".")) == 2:
            return float(num)
        else:
            return int(num)
    except ValueError as e:
        print(f"숫자를 입력하세요({e}).")


def is_number(num):
    try:
        float(num)
        return True
    except ValueError as e:
        return False


class KeyAsValueDict(dict):
    def __missing__(self, key):
        if is_number(key):
            self[key] = strint_to_number(key)
        else:
            self[key] = key
        return key


mapping = KeyAsValueDict()

mapping.update(
    {
        "True": True,
        "true": True,
        "False": False,
        "false": False,
    }
)


def parse_argument(keyword):
    if keyword:
        arguments = [arg for arg in sys.argv if keyword in arg]
        option, value = [arg for arg in arguments if keyword == arg.split("=")[0]][
            0
        ].split("=")
    else:
        option, value = (None, False)
    return option, value
