# Copyright (c) 2022 Nariaki Tateiwa

# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:

# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


import logging

from colorlog import ColoredFormatter


class ColorLogger:
    def __init__(self):
        """Return a logger with a ColoredFormatter."""
        self.formatter = ColoredFormatter(
            "%(log_color)s%(levelname)s:%(name)s:%(funcName)s(%(lineno)s):%(message)s",
            datefmt=None,
            reset=True,
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "red,bg_white",
            },
        )

        logger = logging.getLogger()
        self.root_logger = logger

        self.setStreamHandler()

    def setStreamHandler(self):
        handler = logging.StreamHandler()
        handler.setFormatter(self.formatter)
        self.root_logger.addHandler(handler)

    def setFileHandler(self, filename="log", log_level=10):
        handler = logging.FileHandler(filename=filename)
        handler.setFormatter(self.formatter)
        handler.setLevel(log_level)
        self.root_logger.addHandler(handler)

    def setLevel(self, log_level):
        self.root_logger.setLevel(log_level)
        return self.root_logger

    def __call__(self, name):
        return self.root_logger.getChild(name)


getLogger = ColorLogger()
setLevel = getLogger.setLevel
setLogFile = getLogger.setFileHandler
