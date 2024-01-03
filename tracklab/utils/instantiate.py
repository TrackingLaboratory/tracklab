from hydra._internal.instantiate._instantiate2 import *
from hydra._internal.instantiate._instantiate2 import (
    _extract_pos_args,
    _convert_target_to_string,
)
from hydra._internal.instantiate import _instantiate2


def call_target(
    _target_: Callable[..., Any],
    _partial_: bool,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    full_key: str,
) -> Any:
    """Call target (type) with args and kwargs."""
    try:
        args, kwargs = _extract_pos_args(args, kwargs)
        # detaching configs from parent.
        # At this time, everything is resolved and the parent link can cause
        # issues when serializing objects in some scenarios.
        for arg in args:
            if OmegaConf.is_config(arg):
                arg._set_parent(None)
        for v in kwargs.values():
            if OmegaConf.is_config(v):
                v._set_parent(None)
    except Exception as e:
        msg = (
            f"Error in collecting args and kwargs for '{_convert_target_to_string(_target_)}':"
            + f"\n{repr(e)}"
        )
        if full_key:
            msg += f"\nfull_key: {full_key}"

        raise InstantiationException(msg) from e

    if _partial_:
        return functools.partial(_target_, *args, **kwargs)
    else:
        return _target_(*args, **kwargs)


_instantiate2._call_target = call_target
