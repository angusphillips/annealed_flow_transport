# Copyright 2020 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Main entry point for annealed flow transport and baselines.
Modified from https://github.com/google-deepmind/annealed_flow_transport to include hydra config and wandb logging"""

import os
import hydra

from annealed_flow_transport import train


@hydra.main(config_path="config", config_name="main")
def main(config):
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["WANDB_START_METHOD"] = "thread"
    train.run_experiment(config)


if __name__ == "__main__":
    main()
