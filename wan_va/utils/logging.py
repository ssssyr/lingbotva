# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger()


def init_logger(log_file=None, rank=None, console=True, force=False):
    if force:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            try:
                handler.close()
            except Exception:
                pass
    elif logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    logger.propagate = False
    rank_prefix = f"rank={rank} | " if rank is not None else ""
    formatter = logging.Formatter(
        f"%(asctime)s | {rank_prefix}%(levelname)s | %(message)s"
    )

    if console:
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    # suppress verbose torch.profiler logging
    os.environ["KINETO_LOG_LEVEL"] = "5"
    return logger
