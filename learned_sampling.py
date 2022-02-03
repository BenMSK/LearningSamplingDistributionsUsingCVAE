import sys
sys.path.insert(0, "..")
from configparser import ConfigParser
from datetime import datetime
import argparse
import torch
from cvae_sampler import SamplerCVAE
from utils import PreprocessData, GetWorkspace
import os
import wandb

def argparser():
    """
    Command line argument parser
    """
    parser = argparse.ArgumentParser(description='CVAE adaptive sampler')
    parser.add_argument(
        '--globals', type=str, default='./configs/globals.ini', 
        help="Path to the configuration file containing the global variables "
    )
    parser.add_argument(
        "--data_gather", type=bool, default=False,
        help="Gather data. With executing RRTstar policy"
    )
    parser.add_argument(
        "--train", type=bool, default=False,
        help="Train a model"
    )
    parser.add_argument(
        "--test", type=bool, default=False,
        help="Test a model"
    )
    parser.add_argument(
        "--model_path", type=str, default='.',
        help="Model path"
    )
    return parser.parse_args()

def load_config(args):
    """
    Load .INI configuration files
    """
    config = ConfigParser()

    # Load global variable (e.g. paths)
    config.read(args.globals)

    # Load default model configuration
    default_model_config_filename = config['paths']['model_config_name']
    default_model_config_path = os.path.join(config['paths']['configs_directory'], default_model_config_filename)
    config.read(default_model_config_path)
    config.set('device', 'device', 'cuda' if torch.cuda.is_available() else 'cpu')
    if config.get('device', 'device') == 'cuda':
        print("the current GPU deivce: ", torch.cuda.current_device())
        print("the number of GPU devices: ", torch.cuda.device_count())
        print("GPU name: ", torch.cuda.get_device_name(0))
    config['mode'] = {}
    config['mode']['data_gather'] = str(1) if args.data_gather else str(0)
    config['mode']['train'] = str(1) if args.train else str(0)
    config['mode']['test'] = str(1) if args.test else str(0)

    return config

def main():
    args = argparser()
    config = load_config(args)
    if config.getint('mode', 'train') and config.getboolean("log", "wandb") is True:
        wandb.init(project="Learning_Sampling_Distribution_from_Demonstration_ItcherICRA18", tensorboard=False)
        wandb_config_dict = dict()
        for section in config.sections():
            for key, value in config[section].items():
                wandb_config_dict[key] = value
        wandb.config.update(wandb_config_dict)

    random_seed = config.getint("training", "seed")
    env_dir = config.get("paths", "env_directory") + config.get("paths", "env_name")
    env_workspaces = GetWorkspace(random_seed, env_dir)    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = SamplerCVAE(config, env_workspaces, device, data_gather= config.getint('mode', 'data_gather'), train=config.getint('mode', 'train'))
    model.to(config['device']['device'])

    if config.getint('mode', 'data_gather'):
        model.get_data()

    if config.getint('mode', 'train'):
        model.fit()
    
    if config.getint('mode', 'test'):
        model.test(args.model_path)
    
    print("Done.")
    
if __name__ == "__main__":
    main()