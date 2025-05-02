import sys

from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin
from importlib import metadata
import logging

log = logging.getLogger(__name__)


class TracklabSearchPathPlugin(SearchPathPlugin):
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        # Appends the search path for tracklab plugins to the end of the search path
        tracklab_plugins = entry_points(group="tracklab_plugin")
        for tracklab_plugin in tracklab_plugins:
            m = tracklab_plugin.dist
            module = tracklab_plugin.load()
            if hasattr(module, "config_package"):
                search_path.append(provider="tracklab", path=module.config_package)
            else:
                log.warning(f"{module} doesn't provide a config path")


def entry_points(*, group: str) -> "metadata.EntryPoints":  # type: ignore[name-defined]
    """entry_points function that is compatible with Python 3.7+."""
    if sys.version_info >= (3, 10):
        return metadata.entry_points(group=group)

    epg = metadata.entry_points()

    if sys.version_info < (3, 8) and hasattr(epg, "select"):
        return epg.select(group=group)

    return epg.get(group, [])