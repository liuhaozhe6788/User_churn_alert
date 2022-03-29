def cast_str_to_int(src: str, **kwargs):
    return int(src)


def cast_int_to_str(src: str, **kwargs):
    return str(src)


def cast_str_to_list(src: str, sep: str = None, **kwargs):
    if sep is None: sep = ","
    return src.split(sep=sep)


def cast_list_to_str(src: list, sep: str = None, **kwargs):
    if sep is None: sep = ",";
    return sep.join(list(map(str, src)))


def cast_list_to_list(src: list, **kwargs):
    if "map" in kwargs:
        return cast_list_to_list(list(map(kwargs.pop("map"), src)), **kwargs)
    elif "filter" in kwargs:
        return cast_list_to_list(list(map(kwargs.pop("filter"), src)), **kwargs)
    else:
        return src.copy()


def cast(src, to, **kwargs) -> object:
    if isinstance(to, type):
        to = to.__name__
    return eval(f"cast_{type(src).__name__}_to_{to}")(src, **kwargs)


def caster(func):
    def wrapper(*args, **kwargs):
        types = dict([func.__name__.split("_")])
        return cast(args[0], types["to"], **kwargs)

    return wrapper


@caster
def to_int(src, **kwargs):
    pass


@caster
def to_str(src, **kwargs):
    pass


@caster
def to_list(src, **kwargs):
    pass


__all__ = [f"to_{to}" for to in ["int", "str", "list"]]

if __name__ == "__main__":
    print(to_str(1))
    print(cast_list_to_list(["1", "2"], map=int))
