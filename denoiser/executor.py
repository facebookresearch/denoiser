# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# author: adefossez
"""
Start multiple process locally for DDP.
"""

import logging
import subprocess as sp
import sys

from hydra import utils

logger = logging.getLogger(__name__)


class ChildrenManager:
    def __init__(self):
        self.children = []
        self.failed = False

    def add(self, child):
        child.rank = len(self.children)
        self.children.append(child)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_value is not None:
            logger.error("An exception happened while starting workers %r", exc_value)
            self.failed = True
        try:
            while self.children and not self.failed:
                for child in list(self.children):
                    try:
                        exitcode = child.wait(0.1)
                    except sp.TimeoutExpired:
                        continue
                    else:
                        self.children.remove(child)
                        if exitcode:
                            logger.error(f"Worker {child.rank} died, killing all workers")
                            self.failed = True
        except KeyboardInterrupt:
            logger.error("Received keyboard interrupt, trying to kill all workers.")
            self.failed = True
        for child in self.children:
            child.terminate()
        if not self.failed:
            logger.info("All workers completed successfully")


def start_ddp_workers():
    import torch as th

    world_size = th.cuda.device_count()
    if not world_size:
        logger.error(
            "DDP is only available on GPU. Make sure GPUs are properly configured with cuda.")
        sys.exit(1)
    logger.info(f"Starting {world_size} worker processes for DDP.")
    with ChildrenManager() as manager:
        for rank in range(world_size):
            kwargs = {}
            argv = list(sys.argv)
            argv += [f"world_size={world_size}", f"rank={rank}"]
            if rank > 0:
                kwargs['stdin'] = sp.DEVNULL
                kwargs['stdout'] = sp.DEVNULL
                kwargs['stderr'] = sp.DEVNULL
                log = utils.HydraConfig().hydra.job_logging.handlers.file.filename
                log += f".{rank}"
                argv.append("hydra.job_logging.handlers.file.filename=" + log)
            manager.add(sp.Popen([sys.executable] + argv, cwd=utils.get_original_cwd(), **kwargs))
    sys.exit(int(manager.failed))
