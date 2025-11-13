from __future__ import annotations

import newton
from newton.examples import init, run

from pathlib import Path
from utils.config_loader import load_all_configs
from simulation.car_builder import build_car
from simulation.viewer_runner import RigidCarRunner


def main():
    base = Path(__file__).resolve().parent
    cfg = load_all_configs(base / "config")
    builder, model = build_car(cfg)

    viewer, args = init()
    viewer.set_model(model)

    runner = RigidCarRunner(viewer, model, cfg.sim_params)
    run(runner, args)


if __name__ == "__main__":
    main()
