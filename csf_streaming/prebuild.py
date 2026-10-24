"""Compile the CUDA extensions at image build time (no GPU present).

torch's loader compiles and then imports; the import needs the CUDA driver,
which a build container lacks, so an import failure after a successful
compile is expected here. At runtime the same build directory, sources and
flags are found up to date and only the import happens.
"""
from pathlib import Path
import os
import sys

from csf_streaming import fastcache

root = Path(os.environ["CSF_EXT_BUILD_DIR"])
for loader, name in ((fastcache.mailbox_ext, "csf-mailbox-build/csf_mailbox.so"),
                     (fastcache.v4_ext, "csf-v4-build/csf_v4.so")):
    try:
        loader()
    except Exception as error:  # noqa: BLE001 - driver absent at build time
        print(f"import skipped after build: {type(error).__name__}: {str(error)[:120]}")
    if not (root / name).exists():
        sys.exit(f"extension was not built: {root / name}")
    print("built", root / name)
