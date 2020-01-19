"""Execution Timeout

Two classes wich combined may be used as a wrapper to mark
a function with a maximum execution time.

If the timeout is met, the execution of the function will
be aborted and the wrapper will return the value set
in the configuration file to the variable:
    timeout_return

This file can be imported as a module and contains the following
wrapper:

    * timeout - aborts the execution of a function if the timeout
                is met.
"""

import threading
from datetime import datetime
from config_file import timeout_return


class InterruptableThread(threading.Thread):
    """Defines a thread class.

    Parameters
    ----------
    func : function
        The function to be called.
    *args : optional
        Used to allow multiple adtional input variables for the function.

    Returns
    -------
    self._results
        returns the call to the function with the input arguments.
    """
    def __init__(self, func, *args, **kwargs):
        threading.Thread.__init__(self)
        self._func = func
        self._args = args
        self._kwargs = kwargs
        self._result = None

    def run(self):
        self._result = self._func(*self._args, **self._kwargs)

    @property
    def result(self):
        return self._result


class timeout(object):
    """Class to be used as wrapper.

    Parameters
    ----------
    sec : int
        Number of seconds before aborting the marked function execution.

    Returns
    -------
    timeout_return
        Value defined in the config_file.
    """
    def __init__(self, sec):
        self._sec = sec

    def __call__(self, f):
        def wrapped_f(*args, **kwargs):
            it = InterruptableThread(f, *args, **kwargs)
            it.start()
            it.join(self._sec)
            if not it.is_alive():
                return it.result
            return timeout_return
        return wrapped_f
