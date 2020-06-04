from ._json_log import JsonLog

from ..utils import tensorboard_available

if tensorboard_available:
    from ._tensorboard import TBLog


try:
    import sacred

    sacred_available = True
    from ._sacred import SacredLog
except ImportError:
    pass
