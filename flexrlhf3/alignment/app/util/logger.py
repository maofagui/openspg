# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import logging


class AppLogger(object):
    from alignment.app.util import base_logger
    log_instance = base_logger.get_custom_logger(base_logger.GLOBAL_LOGGER_NAME)
    log_instance.propagate = False


app_logger = AppLogger.log_instance


def flush():
    [h.flush() for h in app_logger.handlers]


def info(msg, *args, **kwargs):
    app_logger.info(msg, *args, **kwargs)


def error(msg, *args, **kwargs):
    app_logger.error(msg, *args, **kwargs)


def warn(msg, *args, **kwargs):
    """Some old source file is still using deprecated `warn'"""
    app_logger.warning(msg, *args, **kwargs)


def warning(msg, *args, **kwargs):
    app_logger.warning(msg, *args, **kwargs)


def debug(msg, *args, **kwargs):
    app_logger.debug(msg, *args, **kwargs)


def exception(msg, *args, **kwargs):
    # handlers = list(filter(lambda x: isinstance(x, logging.FileHandler), AppLogger.log_instance.handlers))
    # if len(handlers) != 0:
    #     AppLogger.log_instance.handlers = handlers
    app_logger.exception(msg, *args, **kwargs)


def set_log_level(level):
    return app_logger.handlers[0].setLevel(level)


def set_log_formatter():
    fmt = '[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d:%(funcName)s] %(message)s'
    def get_new_format(old_fmt):
        import torch
        if torch.distributed.is_initialized():
            return ("[rank-%s] " % torch.distributed.get_rank()) + old_fmt    
        return old_fmt
    formatter = logging.Formatter(get_new_format(fmt))
    app_logger.handlers[0].setFormatter(formatter)