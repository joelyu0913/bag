import errno
import logging
import os
import pty
import select
import subprocess
import sys
from typing import Optional


def run_with_tty(cmd: list[str]) -> subprocess.Popen:
    mo, so = pty.openpty()
    me, se = pty.openpty()

    p = subprocess.Popen(cmd, bufsize=1, stdout=so, stderr=se, close_fds=True)
    for fd in [so, se]:
        os.close(fd)

    timeout = 0.04  # seconds
    readable = [mo, me]
    try:
        while readable:
            ready, _, _ = select.select(readable, [], [], timeout)
            for fd in ready:
                try:
                    data = os.read(fd, 1024)
                    sys.stdout.buffer.write(data)
                    sys.stdout.buffer.flush()
                except OSError as e:
                    if e.errno != errno.EIO:
                        raise
                    # EIO means EOF on some systems
                    readable.remove(fd)
                else:
                    if not data:  # EOF
                        readable.remove(fd)

    finally:
        sys.stdout.buffer.flush()
        for fd in [mo, me]:
            os.close(fd)
        if p.poll() is None:
            p.kill()
        p.wait()
    return p


def run_cmd(cmd: list[str]) -> subprocess.Popen:
    p = subprocess.Popen(
        cmd, bufsize=1, stdout=sys.stdout, stderr=subprocess.STDOUT, close_fds=True
    )
    try:
        p.wait()
    finally:
        if p.poll() is None:
            p.kill()
            p.wait()
    return p


def configure_logging(level: str, log_file: Optional[str] = None) -> None:
    log_level = logging.getLevelName(level.upper())
    logging.basicConfig(level=log_level)

    # set level names to lower case
    logging.addLevelName(logging.DEBUG, "debug")
    logging.addLevelName(logging.INFO, "info")
    logging.addLevelName(logging.WARNING, "warning")
    logging.addLevelName(logging.ERROR, "error")
    logging.addLevelName(logging.CRITICAL, "critical")

    logger = logging.getLogger()
    logger.handlers = []
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(log_level)
    formatter = logging.Formatter(
        fmt="[%(asctime)s.%(msecs)03d] [%(process)d] [%(filename)s:%(lineno)s] [%(levelname)s] %(message)s",
        datefmt="%y-%m-%d %H:%M:%S",
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    if log_file is not None:
        fh = logging.FileHandler(log_file)
        fh.setLevel(log_level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    numba_logger = logging.getLogger("numba")
    numba_logger.setLevel(logging.WARNING)
