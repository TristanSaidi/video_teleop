""" Utils """


class SafeFormatMap(dict):
    """Dict returning "{key}" for mising keys"""

    def __missing__(self, key):
        return "{" + key + "}"


class SafeFString(str):
    """Template string that allows partial string substitution"""

    def format_map(self, _map: dict) -> str:
        return super().format_map(SafeFormatMap(_map))