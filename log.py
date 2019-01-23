from logging import *

LOGPATH = 'logs/audio_modem.log'

logFormatter = Formatter("%(asctime)s [%(levelname)-s]  %(message)s", "%H:%M:%S")
rootLogger = getLogger()

rootLogger.setLevel(DEBUG)

fileHandler = FileHandler(LOGPATH)
fileHandler.setFormatter(logFormatter)
rootLogger.addHandler(fileHandler)

consoleHandler = StreamHandler()
consoleHandler.setFormatter(logFormatter)
rootLogger.addHandler(consoleHandler)
consoleHandler.setLevel(DEBUG)

info("#####################################################")