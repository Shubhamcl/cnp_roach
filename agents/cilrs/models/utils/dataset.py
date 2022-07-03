from pathlib import Path
import copy

from matplotlib.style import context
import h5py
import numpy as np
import logging

import torch
from torch.utils.data import Dataset, DataLoader
from . import augmenter

log = logging.getLogger(__name__)


class CilrsDataset(Dataset):
    def __init__(self, list_expert_h5, list_dagger_h5, env_wrapper, im_augmenter=None):

        self._env_wrapper = env_wrapper
        self._im_augmenter = im_augmenter
        self._batch_read_number = 0
        self._im_stack_idx = env_wrapper.im_stack_idx

        if env_wrapper.view_augmentation:
            self._obs_keys_to_load = ['speed', 'gnss', 'central_rgb', 'left_rgb', 'right_rgb']
        else:
            self._obs_keys_to_load = ['speed', 'gnss', 'central_rgb']

        self._obs_list = []
        self._supervision_list = []
        log.info(f'Loading Expert.')
        self.expert_frames = self._load_h5(list_expert_h5)
        log.info(f'Loading Dagger.')
        self.dagger_frames = self._load_h5(list_dagger_h5)

    def _load_h5(self, list_h5):
        n_frames = 0
        for h5_path in list_h5:
            log.info(f'Loading: {h5_path}')
            with h5py.File(h5_path, 'r', libver='latest', swmr=True) as hf:
                for step_str, group_step in hf.items():
                    if group_step.attrs['critical']: # Bullshit line of code
                        current_step = int(step_str.split('_')[-1]) # Always fucking step name
                        im_stack_idx_list = [max(0, current_step+i+1) for i in self._im_stack_idx]# [current_step]

                        obs_dict = {}
                        for obs_key in self._obs_keys_to_load:
                            # For image, attach the address and other attributes, rest attach numpy value
                            if 'rgb' in obs_key:
                                obs_dict[obs_key] = [
                                    self.read_group_to_dict(
                                        group_step['obs'][obs_key],
                                        [h5_path, f'step_{i}', 'obs', obs_key]) for i in im_stack_idx_list]

                                group_step['obs'][obs_key]
                            else:
                                obs_dict[obs_key] = self.read_group_to_dict(group_step['obs'][obs_key],
                                                                            [h5_path, step_str, 'obs', obs_key])

                        supervision_dict = self.read_group_to_dict(group_step['supervision'],
                                                                   [h5_path, step_str, 'supervision'])
                        self._obs_list.append(obs_dict)
                        self._supervision_list.append(self._env_wrapper.process_supervision(supervision_dict))
                        n_frames += 1
        return n_frames

    @staticmethod
    def read_group_to_dict(group, list_keys):
        data_dict = {}
        for k, v in group.items():
            if v.size > 5000:
                data_dict[k] = list_keys
            else:
                data_dict[k] = np.array(v)
        return data_dict

    def __len__(self):
        return len(self._obs_list)

    def __getitem__(self, idx):

        obs = copy.deepcopy(self._obs_list[idx])

        # load images/large dataset here
        for obs_key, obs_dict in obs.items():
            if 'rgb' in obs_key:
                for i in range(len(obs_dict)):
                    for k, v in obs_dict[i].items():
                        if type(v) is list:
                            with h5py.File(v[0], 'r', libver='latest', swmr=True) as hf:
                                group = hf
                                for key in v[1:]:
                                    group = group[key]
                                obs[obs_key][i][k] = np.array(group[k])
            else:
                for k, v in obs_dict.items():
                    if type(v) is list:
                        with h5py.File(v[0], 'r', libver='latest', swmr=True) as hf:
                            group = hf
                            for key in v[1:]:
                                group = group[key]
                            obs[obs_key][k] = np.array(group[k])

        supervision = copy.deepcopy(self._supervision_list[idx])

        if self._im_augmenter is not None:
            for obs_key, obs_dict in obs.items():
                if 'rgb' in obs_key:
                    for i in range(len(obs_dict)):
                        obs[obs_key][i]['data'] = self._im_augmenter(
                            self._batch_read_number).augment_image(obs[obs_key][i]['data'])

        policy_input, command = self._env_wrapper.process_obs(obs)

        self._batch_read_number += 1
        return command, policy_input, supervision


class CNPDataset(Dataset):
    def __init__(self, list_expert_h5, list_dagger_h5, env_wrapper, im_augmenter=None, context_size=0.15):

        self._env_wrapper = env_wrapper
        self._im_augmenter = im_augmenter
        self._batch_read_number = 0
        self._im_stack_idx = env_wrapper.im_stack_idx
        self.context_size = context_size

        if env_wrapper.view_augmentation:
            self._obs_keys_to_load = ['speed', 'gnss', 'central_rgb', 'left_rgb', 'right_rgb']
        else:
            self._obs_keys_to_load = ['speed', 'gnss', 'central_rgb']

        self._obs_list = []
        self._supervision_list = []
        self._cnxt_address = [] # context for the obs
        self._cnxt_set_count = 0 # changes for each h5 file
        self._cnxt_dict = {} # context at count for obs
        self._cnxt_sup_dict = {} # supervision for above equivalent

        log.info(f'Loading Expert.')
        self.expert_frames = self._load_h5(list_expert_h5)
        log.info(f'Loading Dagger.')
        self.dagger_frames = self._load_h5(list_dagger_h5)

    def _load_h5(self, list_h5):

        n_frames = 0

        # Load each h5 file
        for h5_path in list_h5:
            log.info(f'Loading: {h5_path}')
            # Open h5 file
            with h5py.File(h5_path, 'r', libver='latest', swmr=True) as hf:
                
                # Separate steps into context and observation
                all_steps = list(hf.items())
                sample_size = int(len(all_steps)*self.context_size)
                context_selection = np.random.randint(0, len(all_steps), sample_size)
                obs_selection = set(range(len(all_steps))) - set(context_selection)
                
                context_list = all_steps[context_selection]
                obs_list = all_steps[obs_selection]

                # For each observation step
                for step_str, group_step in obs_list:
                    if group_step.attrs['critical']: # Bullshit line of code
                        current_step = int(step_str.split('_')[-1]) # Always fucking step name
                        im_stack_idx_list = [max(0, current_step+i+1) for i in self._im_stack_idx]# [current_step]

                        obs_dict = {}
                        for obs_key in self._obs_keys_to_load:
                            # For image, attach the address and other attributes, rest attach numpy value
                            if 'rgb' in obs_key:
                                obs_dict[obs_key] = [
                                    self.read_group_to_dict(
                                        group_step['obs'][obs_key],
                                        [h5_path, f'step_{i}', 'obs', obs_key]) for i in im_stack_idx_list]

                                group_step['obs'][obs_key]
                            else:
                                obs_dict[obs_key] = self.read_group_to_dict(group_step['obs'][obs_key],
                                                                            [h5_path, step_str, 'obs', obs_key])

                        supervision_dict = self.read_group_to_dict(group_step['supervision'],
                                                                   [h5_path, step_str, 'supervision'])
                        self._obs_list.append(obs_dict)
                        self._supervision_list.append(self._env_wrapper.process_supervision(supervision_dict))
                        self._cnxt_address.append(self._cnxt_set_count)
                        n_frames += 1
                
                # For each context step
                temp_cnxt_list = []
                temp_cnxt_sup_list = []
                for step_str, group_step in context_list:
                    if group_step.attrs['critical']: # Bullshit line of code
                        current_step = int(step_str.split('_')[-1]) # Always fucking step name
                        im_stack_idx_list = [max(0, current_step+i+1) for i in self._im_stack_idx]# [current_step]

                        obs_dict = {}
                        for obs_key in self._obs_keys_to_load:
                            # For image, attach the address and other attributes, rest attach numpy value
                            if 'rgb' in obs_key:
                                obs_dict[obs_key] = [
                                    self.read_group_to_dict(
                                        group_step['obs'][obs_key],
                                        [h5_path, f'step_{i}', 'obs', obs_key]) for i in im_stack_idx_list]

                                group_step['obs'][obs_key]
                            else:
                                obs_dict[obs_key] = self.read_group_to_dict(group_step['obs'][obs_key],
                                                                            [h5_path, step_str, 'obs', obs_key])

                        supervision_dict = self.read_group_to_dict(group_step['supervision'],
                                                                   [h5_path, step_str, 'supervision'])
                        temp_cnxt_list.append(obs_dict)
                        temp_cnxt_sup_list.append(self._env_wrapper.process_supervision(supervision_dict))
                
                self._cnxt_dict[self._cnxt_set_count] = temp_cnxt_list
                self._cnxt_sup_dict[self._cnxt_set_count] = temp_cnxt_sup_list
                self._cnxt_set_count += 1

        return n_frames

    @staticmethod
    def read_group_to_dict(group, list_keys):
        data_dict = {}
        for k, v in group.items():
            if v.size > 5000:
                data_dict[k] = list_keys
            else:
                data_dict[k] = np.array(v)
        return data_dict

    def __len__(self):
        return len(self._obs_list)

    def __getitem__(self, idx):

        obs = copy.deepcopy(self._obs_list[idx])

        # load images/large dataset here
        for obs_key, obs_dict in obs.items():
            if 'rgb' in obs_key:
                for i in range(len(obs_dict)):
                    for k, v in obs_dict[i].items():
                        if type(v) is list:
                            with h5py.File(v[0], 'r', libver='latest', swmr=True) as hf:
                                group = hf
                                for key in v[1:]:
                                    group = group[key]
                                obs[obs_key][i][k] = np.array(group[k])
            else:
                for k, v in obs_dict.items():
                    if type(v) is list:
                        with h5py.File(v[0], 'r', libver='latest', swmr=True) as hf:
                            group = hf
                            for key in v[1:]:
                                group = group[key]
                            obs[obs_key][k] = np.array(group[k])

        supervision = copy.deepcopy(self._supervision_list[idx])

        if self._im_augmenter is not None:
            for obs_key, obs_dict in obs.items():
                if 'rgb' in obs_key:
                    for i in range(len(obs_dict)):
                        obs[obs_key][i]['data'] = self._im_augmenter(
                            self._batch_read_number).augment_image(obs[obs_key][i]['data'])

        context_index = np.random.randint(len(self._cnxt_dict[[self._cnxt_address[idx]]]))
        context = copy.deepcopy(self._cnxt_dict[self._cnxt_address[idx]][context_index])
        context_supervision = copy.deepcopy(self._cnxt_sup_dict[self._cnxt_address[idx]][context_index])
        
        # load images/large dataset here
        for obs_key, obs_dict in context.items():
            if 'rgb' in obs_key:
                for i in range(len(obs_dict)):
                    for k, v in obs_dict[i].items():
                        if type(v) is list:
                            with h5py.File(v[0], 'r', libver='latest', swmr=True) as hf:
                                group = hf
                                for key in v[1:]:
                                    group = group[key]
                                context[obs_key][i][k] = np.array(group[k])
            else:
                for k, v in obs_dict.items():
                    if type(v) is list:
                        with h5py.File(v[0], 'r', libver='latest', swmr=True) as hf:
                            group = hf
                            for key in v[1:]:
                                group = group[key]
                            context[obs_key][k] = np.array(group[k])


        policy_input, command = self._env_wrapper.process_obs(obs)
        context_input, context_command = self._env_wrapper.process_obs(context)
        context_input = self._expand_measures(context_input, context_command, context_supervision)
        self._batch_read_number += 1
        return command, policy_input, supervision, context_input

    def _expand_measures(self, context_input, context_command, context_supervision):
        context_input['state'] = torch.cat(
            context_input['state'],
            context_command,# len 1
            context_supervision['speed'],# len 1
            context_supervision['action'])# len 2
        return context_input

# [[ Thinking area ]] :
# Dataset functioning:
# an obs index gets called
# Single image gets called out
# that single image has a context index in a list, and then the context index also gets called
# returning same stuff and context data (not sure what format)

# Init logic:
# 

def get_dataloader(dataset_dir, env_wrapper, im_augmentation, batch_size=32, num_workers=8):

    def make_dataset(list_expert_h5, list_dagger_h5, is_train):

        if is_train and (im_augmentation is not None):
            im_augmenter = getattr(augmenter, im_augmentation)
        else:
            im_augmenter = None

        dataset = CilrsDataset(list_expert_h5, list_dagger_h5, env_wrapper, im_augmenter)
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                                shuffle=True, drop_last=True, pin_memory=False)
        return dataloader, dataset.expert_frames, dataset.dagger_frames

    dataset_path = Path(dataset_dir)

    list_expert_h5 = list(dataset_path.glob('expert/*.h5'))
    list_dagger_h5 = list(dataset_path.glob('*/*/*.h5'))

    list_expert_h5 = sorted(list_expert_h5, key=lambda x: int(x.name.split('.')[0]))
    list_dagger_h5 = sorted(list_dagger_h5, key=lambda x: int(x.name.split('.')[0]))

    list_expert_h5_train = [x for i, x in enumerate(list_expert_h5) if i % 10 != 0]
    list_dagger_h5_train = [x for i, x in enumerate(list_dagger_h5) if i % 10 != 0]

    list_expert_h5_val = [x for i, x in enumerate(list_expert_h5) if i % 10 == 0]
    list_dagger_h5_val = [x for i, x in enumerate(list_dagger_h5) if i % 10 == 0]

    # list_h5_train = list_h5[:2]
    # list_h5_val = list_h5[0:1]

    log.info(f'Loading training dataset')
    train, train_expert_frames, train_dagger_frames = make_dataset(list_expert_h5_train, list_dagger_h5_train, True)
    log.info(f'Loading validation dataset')
    val, val_expert_frames, val_dagger_frames = make_dataset(list_expert_h5_val, list_dagger_h5_val, False)

    log.info(f'TRAIN expert episodes: {len(list_expert_h5_train)}, DAGGER episodes: {len(list_dagger_h5_train)}, '
             f'expert hours: {train_expert_frames/10/3600:.2f}, DAGGER hours: {train_dagger_frames/10/3600:.2f}.')
    log.info(f'VAL expert episodes: {len(list_expert_h5_val)}, DAGGER episodes: {len(list_dagger_h5_val)}, '
             f'expert hours: {val_expert_frames/10/3600:.2f}, DAGGER hours: {val_dagger_frames/10/3600:.2f}.')

    return train, val

def get_cnp_dataloader(dataset_dir, env_wrapper, im_augmentation, batch_size=32, num_workers=8):

    def make_dataset(list_expert_h5, list_dagger_h5, is_train):

        if is_train and (im_augmentation is not None):
            im_augmenter = getattr(augmenter, im_augmentation)
        else:
            im_augmenter = None

        dataset = CNPDataset(list_expert_h5, list_dagger_h5, env_wrapper, im_augmenter)
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                                shuffle=True, drop_last=True, pin_memory=False)
        return dataloader, dataset.expert_frames, dataset.dagger_frames

    dataset_path = Path(dataset_dir)

    list_expert_h5 = list(dataset_path.glob('expert/*.h5'))
    list_dagger_h5 = list(dataset_path.glob('*/*/*.h5'))

    list_expert_h5 = sorted(list_expert_h5, key=lambda x: int(x.name.split('.')[0]))
    list_dagger_h5 = sorted(list_dagger_h5, key=lambda x: int(x.name.split('.')[0]))

    list_expert_h5_train = [x for i, x in enumerate(list_expert_h5) if i % 10 != 0]
    list_dagger_h5_train = [x for i, x in enumerate(list_dagger_h5) if i % 10 != 0]

    list_expert_h5_val = [x for i, x in enumerate(list_expert_h5) if i % 10 == 0]
    list_dagger_h5_val = [x for i, x in enumerate(list_dagger_h5) if i % 10 == 0]

    # list_h5_train = list_h5[:2]
    # list_h5_val = list_h5[0:1]

    log.info(f'Loading training dataset')
    train, train_expert_frames, train_dagger_frames = make_dataset(list_expert_h5_train, list_dagger_h5_train, True)
    log.info(f'Loading validation dataset')
    val, val_expert_frames, val_dagger_frames = make_dataset(list_expert_h5_val, list_dagger_h5_val, False)

    log.info(f'TRAIN expert episodes: {len(list_expert_h5_train)}, DAGGER episodes: {len(list_dagger_h5_train)}, '
             f'expert hours: {train_expert_frames/10/3600:.2f}, DAGGER hours: {train_dagger_frames/10/3600:.2f}.')
    log.info(f'VAL expert episodes: {len(list_expert_h5_val)}, DAGGER episodes: {len(list_dagger_h5_val)}, '
             f'expert hours: {val_expert_frames/10/3600:.2f}, DAGGER hours: {val_dagger_frames/10/3600:.2f}.')

    return train, val
