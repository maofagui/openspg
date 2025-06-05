# coding: utf-8
# author: chengfu.wcy (chengfu.wcy@antfin.com)

import logging
import sys

GLOBAL_LOGGER_NAME = 'app-log'
JOB_LOGGER = 'job-log'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


def wrap_formatter(func):
    """
    protect
    """
    import re
    pats = [
        r'access.?key\'?\"?\s*[=:,]\s*((([\"\']).*\3)|(\S+?(?=[\s\n])))', '[\s\"\']+password *[=:] *\S+?(?=[\s\n])',
        '[\s\"\']+-k[\s=]*\S+?(?=[\s\n])', '[\s\"\']+-p[\s=]*\S+?(?=[\s\n])', r'\&key=[^&]*(?=&)'
    ]
    compiled_patterns = [re.compile(pat, re.IGNORECASE) for pat in pats]

    def inner_func(*args, **kwargs):
        msg = func(*args, **kwargs)
        for sub in compiled_patterns:
            msg = re.sub(sub, '--^_^--', msg)
        return msg

    return inner_func


_logger_created = {}

DEFAULT_FORMAT = '[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d:%(funcName)s] %(message)s'
DEFAULT_FORMATTER = logging.Formatter(DEFAULT_FORMAT)
DEFAULT_FORMATTER.format = wrap_formatter(DEFAULT_FORMATTER.format)
DIRECT_FORMATTER = logging.Formatter('%(message)s')
DIRECT_FORMATTER.format = wrap_formatter(DIRECT_FORMATTER.format)
ch = logging.StreamHandler(stream=sys.stderr)
ch.setLevel(logging.getLevelName('DEBUG'))
ch.setFormatter(DEFAULT_FORMATTER)

DEFAULT_HANDLERS = [ch]


def init_custom_logger(name, level='INFO', handlers=None, store=True):
    #FIXME chengfu.wcy comment is removed to solve problem of printing twice.
    if name in _logger_created:
        return
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if handlers is not None:
        logger.handlers = handlers
    logger.propagate = False
    if store:
        _logger_created[name] = logger
    return logger


def get_custom_logger(name):
    if name not in _logger_created:
        return init_custom_logger(name, 'INFO', DEFAULT_HANDLERS, store=True)
    return _logger_created[name]


class Logger(object):
    '''
    可以将一些日志打印到特定的文件path中,而不是全局的日志文件中.
    '''

    def __init__(self, path, console_level=logging.INFO, file_level=logging.INFO):
        self.logger = logging.getLogger(path)
        self.logger.setLevel(logging.DEBUG)
        fmt = logging.Formatter(DEFAULT_FORMAT)
        # fmt = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', '%Y-%m-%d %H:%M:%S')

        for handler in self.logger.handlers:
            handler.close()
        self.logger.handlers[:] = []
        if console_level is not None:
            sh = logging.StreamHandler()
            sh.setFormatter(fmt)
            sh.setLevel(console_level)
            self.logger.addHandler(sh)

        if path is not None:
            fh = logging.FileHandler(path)
            fh.setFormatter(fmt)
            fh.setLevel(file_level)
            self.logger.addHandler(fh)

    def debug(self, message):
        self.logger.debug(message)

    def info(self, message):
        self.logger.info(message)

    def warn(self, message):
        self.logger.warn(message)

    def error(self, message):
        self.logger.error(message)

    def critical(self, message):
        self.logger.critical(message)
