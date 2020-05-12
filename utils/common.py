# -*- coding: utf-8 -*-

import time
import logging
import datetime


class Logger(object):

    def __init__(self, file_path, name="logger"):
        """
        Initialize the log class
        :param file_path: Log file path
        """
        self.logger = logging.getLogger(name)
        handler = logging.FileHandler(filename=file_path)
        self.logger.setLevel(logging.INFO)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def out_print(self, line, with_time=False):
        """
        Print output to file
        :param with_time: Whether to print the date
        :param line: Input text line
        :return: None
        """
        if with_time:
            date_str = datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d %H:%M:%S.%f")
            self.logger.info("{} | {}".format(date_str, line))
        else:
            self.logger.info("{}".format(line))

