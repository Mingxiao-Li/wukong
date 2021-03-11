import logging


class XLogger(logging.Logger):

    def __init__(self, name: str, level: str,):
        super().__init__(name, level)
