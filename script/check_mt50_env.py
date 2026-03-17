#!/usr/bin/env python3

import os
import sys
from importlib import import_module

try:
    from importlib.metadata import PackageNotFoundError, version
except ImportError:
    from importlib_metadata import PackageNotFoundError, version


def package_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return "not-installed"


def import_or_fail(module_name):
    try:
        return import_module(module_name)
    except Exception as exc:
        print("[FAIL] import {}: {}".format(module_name, exc))
        raise


def main():
    os.environ.setdefault("MUJOCO_GL", "egl")

    print("Python: {}".format(sys.version.replace("\n", " ")))
    for package_name in [
        "torch",
        "diffusers",
        "transformers",
        "lerobot",
        "datasets",
        "jsonlines",
        "metaworld",
        "mujoco",
        "gymnasium",
    ]:
        print("{}: {}".format(package_name, package_version(package_name)))

    torch = import_or_fail("torch")
    import_or_fail("datasets")
    gym = import_or_fail("gymnasium")
    import_or_fail("lerobot.datasets.lerobot_dataset")
    metaworld = import_or_fail("metaworld")
    import_or_fail("mujoco")

    print("torch.cuda.is_available(): {}".format(torch.cuda.is_available()))
    print("torch.version.cuda: {}".format(torch.version.cuda))
    if torch.cuda.is_available():
        print("cuda.device_count: {}".format(torch.cuda.device_count()))
        print("cuda.device_name: {}".format(torch.cuda.get_device_name(0)))

    benchmark = metaworld.MT50()
    print("MT50 train classes: {}".format(len(benchmark.train_classes)))
    print("MT50 train tasks: {}".format(len(benchmark.train_tasks)))

    env = gym.make(
        "Meta-World/MT1",
        env_name="reach-v3",
        render_mode="rgb_array",
    )
    obs, info = env.reset(seed=42)
    frame = env.render()
    print("MT1 reset observation type: {}".format(type(obs).__name__))
    print("MT1 render frame shape: {}".format(getattr(frame, "shape", None)))
    env.close()

    print("[OK] MT50 environment smoke test passed.")


if __name__ == "__main__":
    main()
