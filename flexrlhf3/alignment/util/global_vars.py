# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import abc
import json
import os
# NOTE: DO NOT IMPORT tensorflow here!!! Otherwise the submit-side have to install tf

from enum import Enum
from fnmatch import fnmatch


class GlobalVars(object):
    def __init__(self):
        self.context = None    # type: BaseClusterContext
        self.args = None    # optional: save args when running from client. (use flags instead)

    @staticmethod
    def setup_context(context):
        _instance.context = context

    @classmethod
    def instance(cls):
        return _instance


_instance = GlobalVars()    # type: GlobalVar
global_vars = GlobalVars.instance

from threading import Lock
lock = Lock()

class ReducerType(Enum):
    HOROVOD = 1
    BYTEPS = 2
    NONE = 3

class InheritanceSingleton(object):
    _instance = {}

    @staticmethod
    def __get_base_class(clazz):
        if clazz == object:
            return None
        bases = clazz.__bases__
        for base in bases:
            if base == InheritanceSingleton:
                return clazz
            else:
                base_class = InheritanceSingleton.__get_base_class(base)
                if base_class:
                    return base_class
        return None

    def __new__(cls, *args, **kwargs):
        base = InheritanceSingleton.__get_base_class(cls)
        if base is None:
            raise ValueError("Singleton base not found")
        if base not in cls._instance:
            cls._instance[base] = super(InheritanceSingleton, cls).__new__(cls)
        else:
            got_instance = cls._instance[base]
            raise ValueError("Singleton error: %s has been created", got_instance)
        return cls._instance[base]

    @classmethod
    def clear_singleton(cls):
        base = InheritanceSingleton.__get_base_class(cls)
        cls._instance.pop(base, None)

class Context(InheritanceSingleton):
    __metaclass__ = abc.ABCMeta

    def __init__(self, config=None, engine=None, user=None, **kwargs):
        pass

    @staticmethod
    def setup():
        pass

def global_context():
    def _create_context():
        return Context.setup()

    if _instance.context:
        return _instance.context
    else:
        with lock:
            if not _instance.context:
                _instance.context = _create_context()
            return _instance.context


def has_global_context():
    return _instance.context is not None
