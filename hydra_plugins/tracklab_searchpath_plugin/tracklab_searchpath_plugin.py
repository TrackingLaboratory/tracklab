from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin
from importlib.metadata import entry_points
import logging

log = logging.getLogger(__name__)


class PbTrackSearchPathPlugin(SearchPathPlugin):
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        # Appends the search path for tracklab plugins to the end of the search path
        groups = entry_points()
        pbtrack_plugins = groups.get("tracklab_plugin", [])
        for pbtrack_plugin in pbtrack_plugins:
            m = pbtrack_plugin.dist
            module = pbtrack_plugin.load()
            if hasattr(module, "config_package"):
                search_path.append(provider="tracklab", path=module.config_package)
            else:
                log.warning(f"{module} doesn't provide a config path")