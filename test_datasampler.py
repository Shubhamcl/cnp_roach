from agents.cilrs.models.utils.dataset import get_cnp_dataloader
from agents.cilrs.cilrs_wrapper import CilrsWrapper

dataset_dir = "/scratch/lustre/home/shju7830/data/bc_test/"
env_wrapper = CilrsWrapper(
    acc_as_action=True,
    view_augmentation=False,
    value_as_supervision=False,
    value_factor=1.0,
    action_distribution=None,
    dim_features_supervision=0,
    input_states=["speed"],
    im_mean=[0.485, 0.456, 0.406],
    im_std=[0.229, 0.224, 0.225],
    im_stack_idx=[-1])
im_augmentation = 'hard'

train, val = get_cnp_dataloader(dataset_dir, env_wrapper, im_augmentation, \
    batch_size=32, num_workers=1)

for command, policy_input, supervision, context_input in train:
    print(context_input['im'].shape)
    #if idx==3:
    print(policy_input['im'].shape)
    break
#from agents.cilrs.models.utils.dataset import CNPDataset(list

#dataset = CNPDataset(list_expert_h5, list_dagger_h5, env_wrapper, im_augmenter)
#dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
#    shuffle=True, drop_last=True, pin_memory=False)



