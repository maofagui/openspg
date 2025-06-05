# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

def str_to_enum(src, enum_cls):
    if src is None:
        return None
    if isinstance(src, enum_cls):
        return src

    found = None
    for e in enum_cls:
        if e.name == src.lower():
            found = e
        elif e.value == src:
            found = e
    if not found:
        raise ValueError("Can't find {} in {}".format(src, enum_cls))
    return found
