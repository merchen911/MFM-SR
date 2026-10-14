# -*- coding: utf-8 -*-
# @Time   : 2020/8/7
# @Author : Zihan Lin
# @Email  : linzihan.super@foxmail.com

# UPDATE
# @Time   : 2021/3/7
# @Author : Jiawei Guan
# @Email  : guanjw@ruc.edu.cn

"""
recbole.utils.logger
###############################
"""

import logging
import os
import colorlog
import re

from recbole.utils.utils import get_local_time, ensure_dir
from colorama import init

log_colors_config = {
    'DEBUG': 'cyan',
    'WARNING': 'yellow',
    'ERROR': 'red',
    'CRITICAL': 'red',
}


class RemoveColorFilter(logging.Filter):

    def filter(self, record):
        if record:
            ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
            record.msg = ansi_escape.sub('', str(record.msg))
        return True


def set_color(log, color, highlight=True):
    color_set = ['black', 'red', 'green', 'yellow', 'blue', 'pink', 'cyan', 'white']
    try:
        index = color_set.index(color)
    except:
        index = len(color_set) - 1
    prev_log = '\033['
    if highlight:
        prev_log += '1;3'
    else:
        prev_log += '0;3'
    prev_log += str(index) + 'm'
    return prev_log + log + '\033[0m'


def init_logger(config):
    """
    A logger that can show a message on standard output and write it into the
    file named `filename` simultaneously.
    All the message that you want to log MUST be str.

    Args:
        config (Config): An instance object of Config, used to record parameter information.

    Example:
        >>> logger = logging.getLogger(config)
        >>> logger.debug(train_state)
        >>> logger.info(train_result)
    """
    init(autoreset=True)
    # log_dir 미지정 시 기존 동작(./log/) 유지
    LOGROOT = config['log_dir'] or './log/'
    ensure_dir(LOGROOT)

    # log_file_suffix: 실험 매트릭스 등에서 dataset·seed 구분용 (없으면 기존과 동일)
    suffix = config['log_file_suffix']
    if suffix:
        safe = re.sub(r'[^0-9A-Za-z._-]+', '_', str(suffix))
        logfilename = '{}-{}-{}.log'.format(config['model'], safe, get_local_time())
    else:
        logfilename = '{}-{}.log'.format(config['model'], get_local_time())

    logfilepath = os.path.join(LOGROOT, logfilename)

    filefmt = "%(asctime)-15s %(levelname)s  %(message)s"
    filedatefmt = "%a %d %b %Y %H:%M:%S"
    fileformatter = logging.Formatter(filefmt, filedatefmt)

    sfmt = "%(log_color)s%(asctime)-15s %(levelname)s  %(message)s"
    sdatefmt = "%d %b %H:%M"
    sformatter = colorlog.ColoredFormatter(sfmt, sdatefmt, log_colors=log_colors_config)
    if config['state'] is None or config['state'].lower() == 'info':
        level = logging.INFO
    elif config['state'].lower() == 'debug':
        level = logging.DEBUG
    elif config['state'].lower() == 'error':
        level = logging.ERROR
    elif config['state'].lower() == 'warning':
        level = logging.WARNING
    elif config['state'].lower() == 'critical':
        level = logging.CRITICAL
    else:
        level = logging.INFO

    fh = logging.FileHandler(logfilepath)
    fh.setLevel(level)
    fh.setFormatter(fileformatter)
    remove_color_filter = RemoveColorFilter()
    fh.addFilter(remove_color_filter)

    sh = logging.StreamHandler()
    sh.setLevel(level)
    sh.setFormatter(sformatter)

    # run_matrix처럼 한 프로세스에서 여러 번 init_logger를 호출하면
    # basicConfig가 첫 호출 이후 무시되어 첫 파일에만 기록되는 문제가 생긴다.
    # 기존 root handler를 정리해 매 호출마다 대상 로그 파일을 교체한다.
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass

    logging.basicConfig(level=level, handlers=[sh, fh])
