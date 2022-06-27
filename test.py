import gym
import json
from pathlib import Path
# import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import os.path
import sys
from carla_gym.utils import config_utils


@hydra.main(config_path='config', config_name='benchmark')
def main(cfg: DictConfig):
    agents_dict = {}
    obs_configs = {}
    reward_configs = {}
    terminal_configs = {}
    agent_names = []

    # for k, v in cfg.items():
    #     print(f"{k}     {v}")

    for ev_id, ev_cfg in cfg.actors.items():
        # print(f"ev id: {ev_id} ev config: {ev_cfg}")
        agent_names.append(ev_cfg.agent)
        cfg_agent = cfg.agent[ev_cfg.agent]
        print(cfg.agent)
        print(ev_cfg.agent)
        print(cfg_agent)
    #     print(cfg_agent)
        OmegaConf.save(config=cfg_agent, f='./config_agent.yaml')
        AgentClass = config_utils.load_entry_point(cfg_agent.entry_point)
        agents_dict[ev_id] = AgentClass('./config_agent.yaml')
        obs_configs[ev_id] = agents_dict[ev_id].obs_configs
        print(obs_configs)

        # # get obs_configs from agent
        # reward_configs[ev_id] = OmegaConf.to_container(ev_cfg.reward)
        # terminal_configs[ev_id] = OmegaConf.to_container(ev_cfg.terminal)

if __name__=='__main__':
    main()