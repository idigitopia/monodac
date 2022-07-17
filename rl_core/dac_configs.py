import argparse
import torch
from munch import munchify
import itertools 
import hashlib
from datetime import datetime
from os import path
import os
import sys

class BaseConfig(object):
    def __init__(self, arg_str=None):

        # Override this function if you wish to add more param groups to the parser. 
        self.modify_parser = lambda parser: parser
                
        # Initialize Base Arguments
        self._initialize(arg_str)
        
    def _initialize(self, arg_str):
        arg_groups =  self._get_arg_groups(arg_str)
        self.arg_gnames = list(arg_groups)
        
        for group_name, args in arg_groups.items():
            setattr(self,group_name,args)

        self.seed_additional_arguments()
    

    def seed_additional_arguments(self):
        self.logArgs.wandb_id = self.pad_datetime(self.logArgs.exp_id) \
                                if self.logArgs.wandb_id == "default" else self.logArgs.wandb_id
        root_dir = os.getenv('DACMDP_ROOT_DIR', default='results')
        self.logArgs.results_dir = os.path.join(self.logArgs.results_dir,
                                                self.logArgs.wandb_entity,
                                                 self.logArgs.wandb_id) 

        self.mdpBuildArgs.save_folder = os.path.join(self.logArgs.results_dir, "mdp_dump")
        
        if self.logArgs.cache_mdp2wandb and not self.mdpBuildArgs.save_mdp2cache:
            print("Setting save 2 cache as True, cannot upload to Wandb without Saving")
            self.mdpBuildArgs.save_mdp2cache = True
        
        os.makedirs(self.logArgs.results_dir, exist_ok = True)
        os.makedirs(self.mdpBuildArgs.save_folder, exist_ok = True)

    def test_for_constraints(self):
        assert self.mdpBuildArgs.tran_type_count == self.actionModelArgs.n_actions
        
    def pad_datetime(self,s):
        return s + "-" + datetime.now().strftime('%b%d_%H-%M-%S')

    
    def _get_parser(self):
        parser = argparse.ArgumentParser()

        # Env Arguments
        envArgs = parser.add_argument_group(title="envArgs", description='Environment Specification')
        envArgs.add_argument("--env_name", help="environment name", type=str, default="CartPole-v1")
        envArgs.add_argument("--seed", help="choice of seed to use for single start state.", type=int, default=4444)

        # Log Arguments
        logArgs = parser.add_argument_group(title="logArgs", description='Logger / Save Specification')
        logArgs.add_argument("--exp_id", help="Id of the experiment", type=str, default="test_run")
        logArgs.add_argument("--exp_meta", help="meta data of the Experiment", type=str, default="test experiment")

        logArgs.add_argument("--no_wandb_logging", help="set to log a video of the evaluation run", action="store_true")
        logArgs.add_argument("--wandb_project", help="Wandb Project", type=str, default="DACMDP_Cont_V0")
        logArgs.add_argument("--wandb_entity", help="Wandb Entity", type=str, default="xanga")
        logArgs.add_argument("--wandb_id", help="Wandb Id", type=str, default="default")
        logArgs.add_argument("--cache_mdp2wandb", help="Set to upload the mdp Solution vectors to Wandb", action = "store_true")
        logArgs.add_argument("--log_mdp_attributes", help="Set to log different charactersitic distributiosn of the mdp.", action="store_true")
        logArgs.add_argument("--log_video", help="set to log a video of the evaluation run", action="store_true")
        logArgs.add_argument("--log_video_count", help="Number of episodes to evaluate and log the video of.", type=int, default=2)
        
        logArgs.add_argument("--results_dir", help="base folder for results", type=str, default = "default")
        
        # System Arguments 
        sysArgs = parser.add_argument_group(title="sysArgs", description='System Specification')
        sysArgs.add_argument("--no_cuda", help="environment name", action="store_true")
    #     sysArgs.add_argument("--device", help="environment name", type=str, default="cpu")

        # Buffer / Dataset Arguments
        dataArgs = parser.add_argument_group(title="dataArgs", description="dataset / buffer arguments")
        dataArgs.add_argument("--data_dir", help="Directory where the data is stored", type=str, default= "./")
        dataArgs.add_argument("--buffer_name", help="Name Identifier of the buffer", type=str, default= "default")
        dataArgs.add_argument("--buffer_size", help="Size of the buffer", type=int, default= 100000)
        dataArgs.add_argument("--load_buffer", help="Do a bellman backups every __k frames", action="store_true")
        dataArgs.add_argument("--buffer_device", help="Default device to use for the sampled tensors", type=str, default= "cpu")

        
        # State Action Representation Arguments
        reprModelArgs = parser.add_argument_group(title="reprModelArgs", description="Dataset transformation / repr arguments")
        reprModelArgs.add_argument("--repr_model_name", help="Name of the repr model which will be used to build the DAC Agent.", default= "identity")
        reprModelArgs.add_argument("--repr_model_save_dir", help="Directory where the representation will be saved / loaded from.", default= "repr_model")
        reprModelArgs.add_argument("--repr_model_seed", help="Seed for the representation model", type=int, default= 0)
        reprModelArgs.add_argument("--repr_model_checkpoint_iter", help="Specifies the iteration of the checkpoint for loading model", type=int, default= 0)
        reprModelArgs.add_argument("--s_multiplyer", help="multiplyer for s vector", type=int, default= 1)
        reprModelArgs.add_argument("--a_multiplyer", help="multiplyer for a vector", type=int, default= 1)
        

        # Action Model Arguments
        actionModelArgs = parser.add_argument_group(title="actionModelArgs", description="Dataset transformation / repr arguments")
        actionModelArgs.add_argument("--action_model_name", help="Name of the action model which will be used to build the DAC Agent.", default= "identity")
        # mdpBuildArgs.add_argument("--n_actions", help="Number of actions that the candidate action model will return for each query state", type=int, default= 10)
        actionModelArgs.add_argument("--action_model_save_dir", help="Directory where the action will be saved / loaded from.", default= "repr_model")
        actionModelArgs.add_argument("--action_model_seed", help="Seed for the action model", type=int, default= 0)
        actionModelArgs.add_argument("--action_model_checkpoint_iter", help="Specifies the iteration of the checkpoint for loading model", type=int, default= 0)
        actionModelArgs.add_argument("--action_model_checkpoint_iters", help="List of checkpoint iters for ensemble action model", nargs="+", type= int, default=[0,1000,10000]) 
        
        
        # MDP Build parameters
        mdpBuildArgs = parser.add_argument_group(title="mdpBuildArgs", description="MDP build arguments")
        mdpBuildArgs.add_argument("--dac_agent_name", help="Name of the dac build which will be used to build the DAC Agent.", default= "DACAgentBase")
        
        mdpBuildArgs.add_argument("--rmax_reward", help="Default reward for RMAX reward", type=int, default= 10000)
        # mdpBuildArgs.add_argument("--balanced_exploration", help="Try to go to all states equally often", type=int, default= 0)
        # mdpBuildArgs.add_argument("--rmax_threshold", help="Number of travesal before annealing rmax reward", type=int, default= 2)
        mdpBuildArgs.add_argument("--MAX_S_COUNT", help="maximum state count  for gpu rewource allocation", type=int, default= 250000)
        mdpBuildArgs.add_argument("--MAX_NS_COUNT", help="maximum nest state count  for gpu rewource allocation", type=int, default=20)
        # mdpBuildArgs.add_argument("--fill_with", help="Define how to fill missing state actions", type=str, default="0Q_src-KNN", choices=["0Q_src-KNN", "1Q_dst-KNN","kkQ_dst-1NN", "none"])
        mdpBuildArgs.add_argument("--mdp_build_k", help="Number of Nearest neighbor to consider k", type=int, default= 1)
        mdpBuildArgs.add_argument("--knn_delta", help="Define the bias parmeter for nearest neighbor distance", type=float, default=1e-8)
        mdpBuildArgs.add_argument("--penalty_type", help="penalized predicted rewards based on the distance to the state", type=str, default="linear", choices=["none", "linear", "exponential"])
        mdpBuildArgs.add_argument("--penalty_beta", help="beta multiplyer for penalizing rewards based on distance", type=float, default= 1)
        # mdpBuildArgs.add_argument("--filter_with_abstraction", help="Set to true, to filter the states to be added based on the radius.", type=int, default= 0)
        mdpBuildArgs.add_argument("--normalize_by_distance", help="set it on if the transition probabilities should be normalized by distance.", action = "store_true")
        mdpBuildArgs.add_argument("--tran_type_count", help="Number of Tran Types to consider", type=int, default= 10)
        mdpBuildArgs.add_argument("--ur", help="Reward for unknown transition, default = -1000.", type=float, default= -1000)
        mdpBuildArgs.add_argument("--exaggerate_ep_end", help="If set the end state will be exaggerated.", action="store_true")

        mdpBuildArgs.add_argument("--rebuild_mdpfcache", help="Set to rebuild the mdp from cache solution.", action="store_true")
        mdpBuildArgs.add_argument("--save_mdp2cache", help="Set to cache th esolution vectors", action="store_true")
        mdpBuildArgs.add_argument("--save_folder", help="Folder where the cached vectors will be saved.", type=str, default= "default")


        # MDP solve and lift up parameters
        mdpSolveArgs = parser.add_argument_group(title="mdpSolveArgs", description="MDP build arguments")
        mdpSolveArgs.add_argument("--default_mode", help="Default device to use for Solving the MDP", type=str, default= "GPU")
        mdpSolveArgs.add_argument("--gamma", help="Discount Factor for Value iteration", type=float, default= 0.99)
        mdpSolveArgs.add_argument("--slip_prob", help="Slip probability for safe policy", type=float, default= 0.1)
        mdpSolveArgs.add_argument("--target_vi_error", help="target belllman backup error for considering solved", type=float, default= 0.001)
        mdpSolveArgs.add_argument("--bellman_backup_every", help="Do a bellman backups every __k frames", type=int, default= 100)
        mdpSolveArgs.add_argument("--n_backups", help="The number of backups for every backup step", type=int, default= 10)

        # Evaluation Parameters
        evalArgs = parser.add_argument_group(title="evalArgs", description="Evaluation Arguments")
        evalArgs.add_argument("--eval_threshold_quantile", help="Quantile to cap the uncertainty metric in the dataset", type = float, default= 0.9)
        evalArgs.add_argument("--eval_episode_count", help="Number of episodes to evaluate the policy", type=int, default=50)
        evalArgs.add_argument("--soft_at_plcy", help="Sample according to Q values rather than max action", action="store_true")
        evalArgs.add_argument("--plcy_k", help="Set the lift up parameter policy_k you want to test with",  type= int, default=1)
        evalArgs.add_argument("--plcy_sweep", help="Set if evaluate the policy under different nearst neighbor numbers k", action="store_true")
        evalArgs.add_argument("--plcy_sweep_k", help="List the sweep lift up parameter policy_k you want to test with", nargs="+", type= int, default=[1,5,11]) 

        # parser.add_argument("--all_gammas", help="Name of the Environment to guild", type=int, default= "[0.1 ,0.9 ,0.99 ,0.999]")
        
        return self.modify_parser(parser)

    def _get_arg_groups(self,s = None):
        parser = self._get_parser()

        # Parse Arguments
        parsedArgs = parser.parse_args(s.split(" ") if s is not None else None)
        parsedArgs.device = 'cuda' if (not parsedArgs.no_cuda) and torch.cuda.is_available() else 'cpu'

        # Process Parsed arguments
        argGroups = {}
        for group in parser._action_groups:
            title = get_shorthand_group_title(group.title) 
            argGroups[title]={a.dest:getattr(parsedArgs,a.dest,None) for a in group._group_actions}

        return munchify(argGroups)
    
    def __str__(self):
        return BaseConfig.beautify_config(self)


    @property
    def flat_args(self):
        return get_flattened_attributes(self, self.arg_gnames)
        
def get_shorthand_group_title(t):
    title_map = {"positional arguments": "posArgs", "optional arguments":"optArgs"}
    title = title_map[t] if t in title_map else t
    return title 

def get_flattened_attributes(obj, attr_group_names):
    args = {}
    for attr_grp_name in attr_group_names:
        group = getattr(obj,attr_grp_name)
        for arg_name, arg_value in group.items():
            args[f"{attr_grp_name}:{arg_name}"] = arg_value
    return args


def beautify_config(config, header_width = 100, header_spacing = 3, col_spacing = 1):
    assert header_width%4 == 0, "width must be divisible by 4"
    
    # Header Formatting Logic
    
    # Define header format
    def _format_header(title):
        lpad_width = 5
        rpad_width = header_width - header_spacing * 2 - len(title) - lpad_width
        main_header = "#" * lpad_width + " " * header_spacing + title + " " * header_spacing + "#" * rpad_width + "\n" 
        header_border = "-" * header_width + "\n"
        header_format_str = header_border + main_header + header_border
        return header_format_str
    
    closing_str = lambda :"-"*header_width+ "\n\n\n" 

    # Define other formats
    col_width = int(header_width/4) 
    _pad_col_key = lambda elem: str(elem).ljust(col_width - 2 - col_spacing)
    _pad_col_val = lambda elem: str(elem).ljust(col_width)
    _format_col_left = lambda k, v : "|" + " "*col_spacing + _pad_col_key(k) + ":" + _pad_col_val(v) 
    _format_col_right = lambda k, v : _pad_col_key(k) + ":" + _pad_col_val(v) + " "*col_spacing + "|" + "\n"
    
    
    out_str = _format_header("All Arguments")  + "\n\n\n"
    
    for grp_name in config.arg_gnames: 
        group_items = getattr(config,grp_name).items()
        iter_by_two = itertools.zip_longest(*[iter(group_items)]*2)

        out_str += _format_header(grp_name)
        for arg1, arg2 in iter_by_two:
            dummy_col = (" ", " ")
            arg2 = arg2 or dummy_col
            left_col = _format_col_left(*arg1)
            right_col = _format_col_right(*arg2)
            if len(left_col + right_col) > header_width + 4:
                out_str+= left_col + _format_col_right(*dummy_col)
                out_str+= _format_col_right(*arg2) +  _format_col_right(*dummy_col) 
            else:
                out_str+= left_col + right_col

        out_str += closing_str()

    return out_str

