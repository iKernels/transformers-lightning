from inspect import getmembers, isclass, isfunction


def get_classes_from_module(config_proto, parent=None, do_lower_case=False):
    res = {}
    for i in getmembers(config_proto):
        if not i[0].startswith('_') and isclass(i[1]) and (parent is None or issubclass(i[1], parent)):
            if not do_lower_case:
                res[i[0]] = i[1]
            else:
                res[i[0].lower()] = i[1]
    return res


def get_classes_from_module_list(configs_proto, parent=None):
    res = {}
    for config_proto in configs_proto:
        for i in getmembers(config_proto):
            if not i[0].startswith('_') and isclass(i[1]) and (parent is None or issubclass(i[1], parent)):
                assert not i[0] in res, f"Class '{i[0]}' is duplicated"
                res[i[0]] = i[1]
    return res


def get_functions_from_module(config_proto):
    res = {}
    for i in getmembers(config_proto):
        if not i[0].startswith('_') and isfunction(i[1]):
            res[i[0]] = i[1]
    return res


def get_types_from_module(config_proto, target_type=tuple):
    res = {}
    variables = [x for x in dir(config_proto) if not x.startswith("_")]
    for i in getmembers(config_proto):
        if i[0] in variables and isinstance(i[1], target_type):
            res[i[0]] = i[1]
    return res


def is_simple(thing):
    return not isfunction(thing) and not isclass(thing)
