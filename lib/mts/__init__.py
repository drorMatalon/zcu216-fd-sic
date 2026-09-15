# The board driver. The .bit and .hwh sit beside mts.py because
# resolve_binary_path looks for them there.
#
# doaMtsOverlay is fetched only when it is asked for, so that importing the
# dummy overlay next to it does not drag pynq in on a laptop that has none.

def __getattr__(name):
    """Import the real driver on first use - it needs pynq, xrfclk and xrfdc."""
    if name == "doaMtsOverlay":
        from .mts import doaMtsOverlay
        return doaMtsOverlay
    raise AttributeError("module %r has no attribute %r" % (__name__, name))
