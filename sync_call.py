#!/Library/Frameworks/Python.framework/Versions/3.4/bin/python3
"""
Synchronously call an asyncio task/coroutine from a non-asyncio thread.
usage:
    # block the current thread until the coroutine yields or returns
    value = sync_call(loop, coroutine, 1, 2, 3, four=4, five=5)
    # control returns here when the coroutine is done
"""
import asyncio
import threading


def _careful(loop):
    """Ensure that loop isn't the loop that belongs to the current thread.
    """
    # peek at asyncio.events._event_loop_policy to see if there's a loop on
    # this thread. Peek, because calling get_event_loop_policy() creates a new
    # one if it doesn't exist, which may be undesirable
    _policy = asyncio.events._event_loop_policy
    if _policy:
        try:
            _loop = _policy._local._loop
            if _loop:
                assert _loop != loop, "Can't synchronously call coroutine on it's own thread."
        except AttributeError:
            pass


def sync_call(loop, coro, *args, **kwargs):
    """Call a coroutine in the given event loop from some other thread,
    and block until it's done.
    """

    _careful(loop)
    event = threading.Event()

    @asyncio.coroutine
    def _wait_for_coro():
        try:
            return (yield from coro(*args, **kwargs))
        finally:
            event.set()

    t = asyncio.Task(_wait_for_coro(), loop=loop)
    event.wait()

    return t.result()