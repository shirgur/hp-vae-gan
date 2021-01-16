## Portions of Code from, copyright 2018 Jochen Gast

from __future__ import absolute_import, division, print_function

import logging
import colorama
import tqdm

TQDM_SMOOTHING = 0


def create_progressbar(iterable,
                       desc="",
                       train=False,
                       unit="it",
                       initial=0,
                       offset=0,
                       invert_iterations=False,
                       logging_on_update=False,
                       logging_on_close=True,
                       postfix=False):
    # ---------------------------------------------------------------
    # Pick colors
    # ---------------------------------------------------------------
    reset = colorama.Style.RESET_ALL
    bright = colorama.Style.BRIGHT
    cyan = colorama.Fore.CYAN
    dim = colorama.Style.DIM
    green = colorama.Fore.GREEN

    # ---------------------------------------------------------------
    # Specify progressbar layout:
    #   l_bar, bar, r_bar, n, n_fmt, total, total_fmt, percentage,
    #   rate, rate_fmt, rate_noinv, rate_noinv_fmt, rate_inv,
    #   rate_inv_fmt, elapsed, remaining, desc, postfix.
    # ---------------------------------------------------------------
    bar_format = ""
    bar_format += "%s==>%s%s {desc}:%s " % (cyan, reset, bright, reset)  # description
    bar_format += "{percentage:3.0f}%"  # percentage
    bar_format += "%s|{bar}|%s " % (dim, reset)  # bar
    bar_format += " {n_fmt}/{total_fmt}  "  # i/n counter
    bar_format += "{elapsed}<{remaining}"  # eta
    if invert_iterations:
        bar_format += " {rate_inv_fmt}  "  # iteration timings
    else:
        bar_format += " {rate_noinv_fmt}  "
    bar_format += "%s{postfix}%s" % (green, reset)  # postfix

    # ---------------------------------------------------------------
    # Specify TQDM arguments
    # ---------------------------------------------------------------
    tqdm_args = {
        "iterable": iterable,
        "desc": desc,  # Prefix for the progress bar
        "total": len(iterable),  # The number of expected iterations
        "leave": True,  # Leave progress bar when done
        "miniters": 1 if train else None,  # Minimum display update interval in iterations
        "unit": unit,  # String be used to define the unit of each iteration
        "initial": initial,  # The initial counter value.
        "dynamic_ncols": True,  # Allow window resizes
        "smoothing": TQDM_SMOOTHING,  # Moving average smoothing factor for speed estimates
        "bar_format": bar_format,  # Specify a custom bar string formatting
        "position": offset,  # Specify vertical line offset
        "ascii": True,
        "logging_on_update": logging_on_update,
        "logging_on_close": logging_on_close
    }

    return tqdm_with_logging(**tqdm_args)


# ----------------------------------------------------------------------------------------
# Comprehensively adds a new logging level to the `logging` module and the
# currently configured logging class.
# e.g. addLoggingLevel('TRACE', logging.DEBUG - 5)
# ----------------------------------------------------------------------------------------
def addLoggingLevel(level_name, level_num, method_name=None):
    if not method_name:
        method_name = level_name.lower()
    if hasattr(logging, level_name):
        raise AttributeError('{} already defined in logging module'.format(level_name))
    if hasattr(logging, method_name):
        raise AttributeError('{} already defined in logging module'.format(method_name))
    if hasattr(logging.getLoggerClass(), method_name):
        raise AttributeError('{} already defined in logger class'.format(method_name))

    # This method was inspired by the answers to Stack Overflow post
    # http://stackoverflow.com/q/2183233/2988730, especially
    # http://stackoverflow.com/a/13638084/2988730
    def logForLevel(self, message, *args, **kwargs):
        if self.isEnabledFor(level_num):
            self._log(level_num, message, args, **kwargs)

    def logToRoot(message, *args, **kwargs):
        logging.log(level_num, message, *args, **kwargs)

    logging.addLevelName(level_num, level_name)
    setattr(logging, level_name, level_num)
    setattr(logging.getLoggerClass(), method_name, logForLevel)
    setattr(logging, method_name, logToRoot)


# -----------------------------------------------------------------
# Subclass tqdm to achieve two things:
#   1) Output the progress bar into the logbook.
#   2) Remove the comma before {postfix} because it's annoying.
# -----------------------------------------------------------------
class TqdmToLogger(tqdm.tqdm):
    def __init__(self, iterable=None, desc=None, total=None, leave=True,
                 file=None, ncols=None, mininterval=0.1,
                 maxinterval=10.0, miniters=None, ascii=None, disable=False,
                 unit='it', unit_scale=False, dynamic_ncols=False,
                 smoothing=0.3, bar_format=None, initial=0, position=None,
                 postfix=None,
                 logging_on_close=True,
                 logging_on_update=False):

        self._logging_on_close = logging_on_close
        self._logging_on_update = logging_on_update
        self._closed = False

        super(TqdmToLogger, self).__init__(
            iterable=iterable, desc=desc, total=total, leave=leave,
            file=file, ncols=ncols, mininterval=mininterval,
            maxinterval=maxinterval, miniters=miniters, ascii=ascii, disable=disable,
            unit=unit, unit_scale=unit_scale, dynamic_ncols=dynamic_ncols,
            smoothing=smoothing, bar_format=bar_format, initial=initial, position=position,
            postfix=postfix)

    @staticmethod
    def format_meter(n, total, elapsed, ncols=None, prefix='', ascii=False,
                     unit='it', unit_scale=False, rate=None, bar_format=None,
                     postfix=None, unit_divisor=1000, **extra_kwargs):

        meter = tqdm.tqdm.format_meter(
            n=n, total=total, elapsed=elapsed, ncols=ncols, prefix=prefix, ascii=ascii,
            unit=unit, unit_scale=unit_scale, rate=rate, bar_format=bar_format,
            postfix=postfix, unit_divisor=unit_divisor, **extra_kwargs)

        # get rid of that stupid comma before the postfix
        if postfix is not None:
            postfix_with_comma = ", %s" % postfix
            meter = meter.replace(postfix_with_comma, postfix)

        return meter

    def update(self, n=1):
        if self._logging_on_update:
            msg = self.__repr__()
            logging.logbook(msg)
        return super(TqdmToLogger, self).update(n=n)

    def close(self):
        if self._logging_on_close and not self._closed:
            msg = self.__repr__()
            logging.logbook(msg)
            self._closed = True
        return super(TqdmToLogger, self).close()


def tqdm_with_logging(iterable=None, desc=None, total=None, leave=True,
                      ncols=None, mininterval=0.1,
                      maxinterval=10.0, miniters=None, ascii=None, disable=False,
                      unit="it", unit_scale=False, dynamic_ncols=False,
                      smoothing=0.3, bar_format=None, initial=0, position=None,
                      postfix=None,
                      logging_on_close=True,
                      logging_on_update=False):
    return TqdmToLogger(
        iterable=iterable, desc=desc, total=total, leave=leave,
        ncols=ncols, mininterval=mininterval,
        maxinterval=maxinterval, miniters=miniters, ascii=ascii, disable=disable,
        unit=unit, unit_scale=unit_scale, dynamic_ncols=dynamic_ncols,
        smoothing=smoothing, bar_format=bar_format, initial=initial, position=position,
        postfix=postfix,
        logging_on_close=logging_on_close,
        logging_on_update=logging_on_update)
