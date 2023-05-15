from src.optimizer.radam import RAdam
import torch
import sys
import numpy as np
import os
from glob import glob


PARAM_CONFIG_FILE = "param_config.pk"

def one_hot(x, num_classes, dtype=torch.float32):
    if isinstance(x, np.ndarray):
        x_onehot = np.zeros(x.shape + (num_classes,), dtype=np.float32)
        x_onehot[np.arange(x.shape[0]), x] = 1.0
    elif isinstance(x, torch.Tensor):
        assert torch.max(
            x) < num_classes, "[!] ERROR: One-hot input has larger entries (%s) than classes (%i)" % (str(torch.max(x)), num_classes)
        x_onehot = x.new_zeros(x.shape + (num_classes,), dtype=dtype)
        x_onehot.scatter_(-1, x.unsqueeze(dim=-1), 1)
    else:
        print("[!] ERROR: Unknown object given for one-hot conversion:", x)
        sys.exit(1)
    return x_onehot

def _create_length_mask(length, max_len=None, dtype=torch.float32):
    if max_len is None:
        max_len = length.max()
    mask = (torch.arange(max_len, device=length.device).view(
        1, max_len) < length.unsqueeze(dim=-1)).to(dtype=dtype)
    return mask
    
def create_transformer_mask(length, max_len=None, dtype=torch.float32):
    mask = _create_length_mask(
        length=length, max_len=max_len, dtype=torch.bool)
    mask = ~mask  # Negating mask, as positions that should be masked, need a True, and others False
    # mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_channel_mask(length, max_len=None, dtype=torch.float32):
    mask = _create_length_mask(length=length, max_len=max_len, dtype=dtype)
    mask = mask.unsqueeze(dim=-1)  # Unsqueeze over channels
    return mask

class Tracker:

    def __init__(self, exp_decay=1.0):
        self.val_sum = 0.0
        self.counter = 0
        self.exp_decay = exp_decay

    def add(self, val):
        self.val_sum = self.val_sum * self.exp_decay + val
        self.counter = self.counter * self.exp_decay + 1

    def get_mean(self, reset=False):
        if self.counter <= 0:
            mean = 0
        else:
            mean = self.val_sum / self.counter
        if reset:
            self.reset()
        return mean

    def reset(self):
        self.val_sum = 0.0
        self.counter = 0


###################
## MODEL LOADING ##
###################

def load_model(checkpoint_path, model=None, optimizer=None, lr_scheduler=None, load_best_model=False, warn_unloaded_weights=True):
    # Determine the checkpoint file to load
    if os.path.isdir(checkpoint_path):
        checkpoint_files = sorted(glob(os.path.join(checkpoint_path, "*.tar")))
        if len(checkpoint_files) == 0:
            #print("No checkpoint files found at", checkpoint_path)
            return dict()
        checkpoint_file = checkpoint_files[-1]
    else:
        checkpoint_file = checkpoint_path

    # Loading checkpoint
    #print("Loading checkpoint \"" + str(checkpoint_file) + "\"")
    if torch.cuda.is_available():
        checkpoint = torch.load(checkpoint_file)
    else:
        checkpoint = torch.load(checkpoint_file, map_location='cpu')

    # If best model should be loaded, look for it if checkpoint_path is a directory
    if os.path.isdir(checkpoint_path) and load_best_model:
        if os.path.isfile(checkpoint["best_save_dict"]["file"]):
            print("Load best model!")
            return load_model(checkpoint["best_save_dict"]["file"], model=model, optimizer=optimizer, lr_scheduler=lr_scheduler, load_best_model=False)
        else:
            print("[!] WARNING: Best save dict file is listed as \"%s\", but file could not been found. Using default one..." %
                  checkpoint["best_save_dict"]["file"])

    # Load the model parameters
    if model is not None:
        pretrained_model_dict = {key: val for key,
                                 val in checkpoint['model_state_dict'].items()}
        model_dict = model.state_dict()
        unchanged_keys = [key for key in model_dict.keys(
        ) if key not in pretrained_model_dict.keys()]
        # Parameters in this list might have been forgotten to be saved
        if warn_unloaded_weights and len(unchanged_keys) != 0:
            print(
                "[#] WARNING: Some weights have been left unchanged by the loading of the model: " + str(unchanged_keys))
        model_dict.update(pretrained_model_dict)
        model.load_state_dict(model_dict)
    # Load the state and parameters of the optimizer
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # Load the state of the learning rate scheduler
    if lr_scheduler is not None and 'scheduler_state_dict' in checkpoint:
        lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    # Load the additional parameters that were saved in the dict
    add_param_dict = dict()
    for key, val in checkpoint.items():
        if "state_dict" not in key:
            add_param_dict[key] = val
    return add_param_dict


# def general_args_to_params(args, model_params=None):

#     optimizer_params = {
#         "optimizer": args.optimizer,
#         "learning_rate": args.learning_rate,
#         "weight_decay": args.weight_decay,
#         "lr_decay_factor": args.lr_decay_factor,
#         "lr_decay_step": args.lr_decay_step,
#         "lr_minimum": args.lr_minimum,
#         "momentum": args.momentum,
#         "beta1": args.beta1,
#         "beta2": args.beta2,
#         "warmup": args.warmup
#     }

#     # Set seed
#     np.random.seed(args.seed)
#     random.seed(args.seed)
#     torch.manual_seed(args.seed)
#     if torch.cuda.is_available:
#         torch.cuda.manual_seed(args.seed)
#         torch.cuda.manual_seed_all(args.seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

#     return model_params, optimizer_params


def create_optimizer_from_args(parameters_to_optimize, optimizer_params):
    optimizer = RAdam(parameters_to_optimize,
                      lr=optimizer_params.learning_rate,
                      betas=(optimizer_params.beta1,
                             optimizer_params.beta2),
                      weight_decay=optimizer_params.weight_decay)

    return optimizer


def get_param_val(param_dict, key, default_val=None, allow_default=True, error_location="", warning_if_default=True):
    if key in param_dict:
        return param_dict[key]
    elif allow_default:
        if warning_if_default:
            print("[#] WARNING: Using default value %s for key %s" %
                  (str(default_val), str(key)))
        return default_val
    else:
        assert False, "[!] ERROR (%s): could not find key \"%s\" in the dictionary although it is required." % (
            error_location, str(key))


def append_in_dict(val_dict, key, new_val):
    if key not in val_dict:
        val_dict[key] = list()
    val_dict[key].append(new_val)


####################################
## VISUALIZATION WITH TENSORBOARD ##
####################################

def write_dict_to_tensorboard(writer, val_dict, base_name, iteration):
    for name, val in val_dict.items():
        if isinstance(val, dict):
            write_dict_to_tensorboard(
                writer, val, base_name=base_name+"/"+name, iteration=iteration)
        elif isinstance(val, (list, np.ndarray)):
            continue
        elif isinstance(val, (int, float)):
            writer.add_scalar(base_name + "/" + name, val, iteration)


def get_device():

    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

