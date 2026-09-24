import sys


class _ColorMeta(type):
    def __getattribute__(cls, name):
        value = super().__getattribute__(name)
        if isinstance(value, str) and value.startswith("\033["):
            if not sys.stdout.isatty():
                return ""
        return value


class bcolors(metaclass=_ColorMeta):
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    # Custom colors
    INTENTION = OKGREEN
    DESIRE = OKCYAN
    BELIEF = OKBLUE
    PERCEPTION = WARNING
    SYSTEM = HEADER
