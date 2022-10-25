'''
Logger module to manage console outputs
with a custom coloured scheme

@date: Created on 20.10.2022
@author: Elena de la Fuente
'''

import logging

class Logger(logging.Formatter):

    grey = "\x1b[1;37m"
    blue = "\x1b[36m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: blue + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

def get_logger(Logger=Logger, level=2):
    log = logging.getLogger('Wakis')

    #set level
    if level == 1:
        log.setLevel(logging.DEBUG)
    elif level == 2:
        log.setLevel(logging.INFO)
    elif level == 3:
        log.setLevel(logging.WARNING)
    elif level == 4:
        log.setLevel(logging.ERROR)
    elif level == 5:
        log.setLevel(logging.CRITICAL)
    else:
        #default
        log.setLevel(logging.INFO)

    #handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(Logger())
    log.addHandler(ch)

    return log