"""Install only when the CSF module loads; leave launcher Python probes quiet."""
import importlib.abc
import importlib.machinery
import os
import sys


if os.environ.get("CSF_STREAMING") == "1":
    class StreamingLoader(importlib.abc.Loader):
        def __init__(self, original):
            self.original = original

        def create_module(self, spec):
            return self.original.create_module(spec)

        def exec_module(self, module):
            self.original.exec_module(module)
            try:
                from csf_streaming.integration import install
                install()
            except BaseException:
                import traceback
                traceback.print_exc()
                os._exit(78)

    class StreamingFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname != "vllm.models.deepseek_v4_1.mxfp4_csf":
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec is None or spec.loader is None:
                raise ImportError("Pinned CSF module is unavailable")
            spec.loader = StreamingLoader(spec.loader)
            return spec

    sys.meta_path.insert(0, StreamingFinder())
