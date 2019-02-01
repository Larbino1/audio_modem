import logging
import colorama

from logging import DEBUG, INFO, WARNING, ERROR, CRITICAL

colormap = {
    'DEFAULT'   : colorama.Fore.RESET,
    'INFO'      : colorama.Fore.WHITE,
    'DEBUG'     : colorama.Fore.LIGHTBLACK_EX,
    'WARNING': colorama.Fore.YELLOW,
    'ERROR'     : colorama.Fore.RED,
    'CRITICAL'  : colorama.Fore.RED,
    'SPECIAL'  : colorama.Fore.CYAN
}

LOGPATH = 'logs/audio_modem.log'

def debug(msg):
    logging.debug(f"{colormap.get('DEBUG', '')}{msg}")


def info(msg):
    logging.info(f"{colormap.get('INFO', '')}{msg}")


def special(msg):
    logging.info(f"{colormap.get('SPECIAL', '')}{msg}")


def error(msg):
    logging.error(f"{colormap.get('ERROR', '')}{msg}")


def warning(msg):
    logging.warning(f"{colormap.get('WARNING', '')}{msg}")


logFormatter = logging.Formatter(
    colormap['DEFAULT'] + "%(asctime)s [%(levelname)-s]  %(message)s", "%H:%M:%S")
rootLogger = logging.getLogger()
rootLogger.setLevel(DEBUG)

fileHandler = logging.FileHandler(LOGPATH)
fileHandler.setFormatter(logFormatter)
rootLogger.addHandler(fileHandler)

consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
rootLogger.addHandler(consoleHandler)
consoleHandler.setLevel(DEBUG)

info("#####################################################")