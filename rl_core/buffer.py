from collections import namedtuple
from typing import Tuple
import numpy as np
import torch
import time
import copy
from types import SimpleNamespace

# Generic replay buffer for standard gym tasks
# most commonly used.
class StandardBuffer(object):
    """
    Initializes an array for elements of transitions as per the maximum buffer size. 
    Keeps track of the crt_size. 
    Saves the buffer element-wise as numpy array. Fast save and retreival compared to pickle dumps. 
    """
    def __init__(self, state_shape, action_shape,  buffer_size, device, batch_size = 64):
        
        self.state_shape = state_shape
        self.action_shape = action_shape
        
        self.batch_size = batch_size
        self.max_size = int(buffer_size)
        self.device = device

        self.ptr = 0
        self.crt_size = 0

        self.state = np.zeros((self.max_size, *state_shape))
        self.action = np.zeros((self.max_size, *action_shape))
        self.next_state = np.array(self.state)
        self.reward = np.zeros((self.max_size, 1))
        self.not_done = np.zeros((self.max_size, 1))

        # Normalization parameters. 
        self.norm_params = None # Initialized in normalize method. 
    
    def __len__(self):
        return self.crt_size

    def __repr__(self):
        return f"Standard Buffer: \n \
                Total number of transitions: {len(self)}/{self.max_size} \n \
                State Store Shape: {self.state.shape} \n \
                Action Store Shape: {self.action.shape} \n"

    @property
    def all_states(self):
        return self.state[:self.crt_size]

    @property
    def all_next_states(self):
        return self.next_state[:self.crt_size]
    
    @property
    def all_actions(self):
        return self.action[:self.crt_size]

    @property
    def all_not_ep_ends(self):
        return self.not_done[:self.crt_size]

    @property
    def all_ep_ends(self):
        return 1- self.not_done[:self.crt_size]

    @property
    def all_rewards(self):
        return self.reward[:self.crt_size]


    def add(self, state, action, next_state, reward, done, episode_done=None, episode_start=None):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.crt_size = min(self.crt_size + 1, self.max_size)


    def sample(self, batch_size= None):
        inds = np.random.randint(0, self.crt_size, size=batch_size or self.batch_size)
        return self.sample_indices(inds)

    def sample_indices(self, inds):
        inds = np.array(inds)
        return (
            torch.FloatTensor(self.state[inds]).to(self.device),
            torch.FloatTensor(self.action[inds]).to(self.device),
            torch.FloatTensor(self.next_state[inds]).to(self.device),
            torch.FloatTensor(self.reward[inds]).to(self.device),
            torch.FloatTensor(self.not_done[inds]).to(self.device)
        )

    def save(self, save_folder):
        np.save(f"{save_folder}_state.npy", self.state[:self.crt_size])
        np.save(f"{save_folder}_action.npy", self.action[:self.crt_size])
        np.save(f"{save_folder}_next_state.npy", self.next_state[:self.crt_size])
        np.save(f"{save_folder}_reward.npy", self.reward[:self.crt_size])
        np.save(f"{save_folder}_not_done.npy", self.not_done[:self.crt_size])
        np.save(f"{save_folder}_ptr.npy", self.ptr)

    def load(self, save_folder, size=-1):
        reward_buffer = np.load(f"{save_folder}_reward.npy")

        # Adjust crt_size if we're using a custom size
        size = min(int(size), self.max_size) if size > 0 else self.max_size
        self.crt_size = min(reward_buffer.shape[0], size)

        self.state[:self.crt_size] = np.load(f"{save_folder}_state.npy")[:self.crt_size]
        self.action[:self.crt_size] = np.load(f"{save_folder}_action.npy")[:self.crt_size]
        self.next_state[:self.crt_size] = np.load(f"{save_folder}_next_state.npy")[:self.crt_size]
        self.reward[:self.crt_size] = reward_buffer[:self.crt_size]
        self.not_done[:self.crt_size] = np.load(f"{save_folder}_not_done.npy")[:self.crt_size]

        print(f"Replay Buffer loaded with {self.crt_size} elements.")



    def get_tran_tuples(self):
        batch = self.sample_indices(list(range(0, len(self))))
        batch_ob, batch_a, batch_ob_prime, batch_r, batch_nd = batch
        batch_d = 1 - batch_nd
        tran_tuples = [(tuple(s), tuple(a), tuple(ns), r, d) for s, a, ns, r, d in zip(batch_ob.numpy(),
                                                                                       batch_a.numpy(),
                                                                                       batch_ob_prime.numpy(),
                                                                                       batch_r.view((-1,)).numpy(),
                                                                                       batch_d.view((-1,)).numpy())]
        return tran_tuples

    # Normalization Logics

    @staticmethod
    def _normalize_vector(arr:np.array, max_vec = None, min_vec = None)->Tuple[np.array, np.array, np.array]:
        
        # use the given norm parameters if provided , else calculate max and min vectors. 
        max_vec = np.max(arr, axis = 0) if max_vec is None else max_vec
        min_vec = np.min(arr, axis = 0) if min_vec is None else min_vec

        # handles max == min cases. prevents overflow and normalizes everythign to 1
        max_min_same_indices = min_vec == max_vec
        min_vec[min_vec == max_vec] = np.ones_like(min_vec)[max_min_same_indices]

        # calculate normalized array
        normalized_array = (arr- min_vec) / (max_vec - min_vec)

        return normalized_array, max_vec, min_vec


    @staticmethod
    def _inverse_normalize_vector(arr:np.array, max_vec, min_vec)->np.array:
        # calculate normalized array
        denormalized_array = arr * (max_vec - min_vec) + min_vec

        return denormalized_array


    @staticmethod
    def normalize_buffer(buffer, normalize_state = True, normalize_action = True):
        """[summary]
        normalizes each channel of the feature representation, and each pixel for image representation. 

        Args:
            buffer ([type]): [description]
            normalize_state (bool, optional): [description]. Defaults to True.
            normalize_action (bool, optional): [description]. Defaults to True.
        """

        # Initialize norm params if not initalized already
        if buffer.norm_params is None: 
            buffer.norm_params = SimpleNamespace(is_state_normalized = False,  is_action_normalized = False, 
                                         state_max_vec = None, state_min_vec = None,
                                         action_max_vec = None, action_min_vec = None)

        if normalize_state:
            assert not buffer.norm_params.is_state_normalized , "Buffer has already been normalized Once, please inverse Normalize before calling."
            
            buffer.state, max_vec, min_vec = StandardBuffer._normalize_vector(buffer.state)
            buffer.next_state, _, _ = StandardBuffer._normalize_vector(buffer.next_state, max_vec, min_vec)
            buffer.norm_params.state_max_vec, buffer.norm_params.state_min_vec = max_vec, min_vec
            buffer.norm_params.is_state_normalized = True

        if normalize_action:
            assert not buffer.norm_params.is_action_normalized , "Buffer has already been normalized Once, please inverse Normalize before calling."

            buffer.action, max_vec, min_vec = StandardBuffer._normalize_vector(buffer.action)
            buffer.norm_params.action_max_vec, buffer.norm_params.action_min_vec = max_vec, min_vec
            buffer.norm_params.is_action_normalized = True

        return buffer

    @staticmethod
    def inverse_normalize_buffer(buffer, normalize_state = True, normalize_action = True):
        
        if buffer.norm_params is None: 
            buffer.norm_params = SimpleNamespace(is_state_normalized = False,  is_action_normalized = False, 
                                         state_max_vec = None, state_min_vec = None,
                                         action_max_vec = None, action_min_vec = None)

        if normalize_state:
            assert buffer.norm_params.is_state_normalized, "Buffer not Normalized, please normalize buffer before calling inverse normalization."
            assert buffer.norm_params.state_max_vec is not None , "Norm Params not initialized properly"
            assert buffer.norm_params.state_min_vec is not None , "Norm Params not initialized properly"

            buffer.state = StandardBuffer._inverse_normalize_vector(buffer.state, buffer.norm_params.state_max_vec, buffer.norm_params.state_min_vec)
            buffer.next_state = StandardBuffer._inverse_normalize_vector(buffer.next_state, buffer.norm_params.state_max_vec, buffer.norm_params.state_min_vec)
            
            buffer.norm_params.is_state_normalized = False
            buffer.norm_params.state_max_vec, buffer.norm_params.state_min_vec = None, None
        
        if normalize_action:
            assert buffer.norm_params.is_action_normalized, "Buffer not Normalized, please normalize buffer before calling inverse normalization."
            assert buffer.norm_params.action_max_vec is not None , "Norm Params not initialized properly"
            assert buffer.norm_params.action_min_vec is not None , "Norm Params not initialized properly"

            buffer.action = StandardBuffer._inverse_normalize_vector(buffer.action, buffer.norm_params.action_max_vec, buffer.norm_params.action_min_vec)
            
            buffer.norm_params.is_action_normalized = False
            buffer.norm_params.action_max_vec, buffer.norm_params.action_min_vec = None, None

        return buffer

    def query_denorm_action(self,a):
        max_vec, min_vec = self.norm_params.action_max_vec, self.norm_params.action_min_vec 
        return a * (max_vec - min_vec) + min_vec

    def query_denorm_state(self,s):
        max_vec, min_vec = self.norm_params.state_max_vec, self.norm_params.state_min_vec 
        return s * (max_vec - min_vec) + min_vec



    @staticmethod
    def populate_buffer(buffer, env, policy, episode_count=1,frame_count=None, render= False,  pad_attribute_fxn=None,
                                verbose=False):
        """

        :param exp_buffer:
        :param env:
        :param episodes:
        :param render:
        :param policy:
        :param frame_count:
        :param pad_attribute_fxn:
        :param verbose: Can be None,  True or 2 for maximum verboseness.
        :param policy_on_states:  if set to true , the policy provided is assumed to be on the state variable of unwrapped env
        :return:
        """
        # experience = obs, action, next_obs, reward, terminal_flag
        start_time = time.time()

        cum_rewards = 0
        frame_counter = 0
        eps_count = 0
        all_rewards = []
        for _ in range(episode_count):
            eps_count += 1
            episode_timesteps = 0
            done = False
            state = env.reset()
            ep_reward = 0
            episode_start = True
            while not done:
                episode_timesteps += 1
                if render:
                    env.render()

                action =policy(state)

                # Perform action and log results
                next_state, reward, done, info = env.step(action)
                ep_reward += reward

                # Only consider "done" if episode terminates due to failure condition
                done_float = float(done) if episode_timesteps < env._max_episode_steps else 0

                # Store data in replay buffer
                buffer.add(state, action, next_state, reward, done_float, done, episode_start)
                state = copy.copy(next_state)
                episode_start = False


            frame_counter += episode_timesteps
            all_rewards.append(ep_reward)

            if verbose:
                print(ep_reward, frame_count)

            if frame_count and frame_counter > frame_count:
                break

        print('Average Reward of collected trajectories:{}'.format(round(np.mean(all_rewards), 3)))

        info = {"all_rewards":all_rewards}
        if verbose:
            print("Data Collection Complete in {} Seconds".format(time.time()-start_time))
        return buffer, info


# Not Maintained
def ReplayBuffer(state_dim, is_atari, atari_preprocessing, batch_size, buffer_size, device):
    if is_atari:
        return AtariBuffer(atari_preprocessing, batch_size, buffer_size, device)
    else:
        return StandardBuffer(state_dim, batch_size, buffer_size, device)

# Not Maintianed
class AtariBuffer(object):
    def __init__(self, atari_preprocessing, batch_size, buffer_size, device):
        self.batch_size = batch_size
        self.max_size = int(buffer_size)
        self.device = device

        self.state_history = atari_preprocessing["state_history"]

        self.ptr = 0
        self.crt_size = 0

        self.state = np.zeros((
            self.max_size + 1,
            atari_preprocessing["frame_size"],
            atari_preprocessing["frame_size"]
        ), dtype=np.uint8)

        self.action = np.zeros((self.max_size, 1), dtype=np.int64)
        self.reward = np.zeros((self.max_size, 1))

        # not_done only consider "done" if episode terminates due to failure condition
        # if episode terminates due to timelimit, the transition is not added to the buffer
        self.not_done = np.zeros((self.max_size, 1))
        self.first_timestep = np.zeros(self.max_size, dtype=np.uint8)

    def add(self, state, action, next_state, reward, done, env_done, first_timestep):
        # If dones don't match, env has reset due to timelimit
        # and we don't add the transition to the buffer
        if done != env_done:
            return

        self.state[self.ptr] = state[0]
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done
        self.first_timestep[self.ptr] = first_timestep

        self.ptr = (self.ptr + 1) % self.max_size
        self.crt_size = min(self.crt_size + 1, self.max_size)

    def sample(self, batch_size= None):
        inds = np.random.randint(0, self.crt_size, size=batch_size or self.batch_size)
        return self.sample_indices(inds)

    def sample_indices(self, inds):
        inds = np.array(inds)
        batch_size = len(inds)
        # Note + is concatenate here
        state = np.zeros(((batch_size, self.state_history) + self.state.shape[1:]), dtype=np.uint8)
        next_state = np.array(state)

        state_not_done = 1.
        next_not_done = 1.
        for i in range(self.state_history):

            # Wrap around if the buffer is filled
            if self.crt_size == self.max_size:
                j = (inds - i) % self.max_size
                k = (inds - i + 1) % self.max_size
            else:
                j = inds - i
                k = (inds - i + 1).clip(min=0)
                # If j == -1, then we set state_not_done to 0.
                state_not_done *= (j + 1).clip(min=0, max=1).reshape(-1, 1,
                                                                     1)  # np.where(j < 0, state_not_done * 0, state_not_done)
                j = j.clip(min=0)

            # State should be all 0s if the episode terminated previously
            state[:, i] = self.state[j] * state_not_done
            next_state[:, i] = self.state[k] * next_not_done

            # If this was the first timestep, make everything previous = 0
            next_not_done *= state_not_done
            state_not_done *= (1. - self.first_timestep[j]).reshape(-1, 1, 1)

        return (
            torch.from_numpy(state).to(self.device).float(),
            torch.from_numpy(self.action[inds]).to(self.device).long(),
            torch.from_numpy(next_state).to(self.device).float(),
            torch.from_numpy(self.reward[inds]).to(self.device),
            torch.from_numpy(self.not_done[inds]).to(self.device)
        )

    def save(self, save_folder, chunk=int(1e5)):
        np.save(f"{save_folder}_action.npy", self.action[:self.crt_size])
        np.save(f"{save_folder}_reward.npy", self.reward[:self.crt_size])
        np.save(f"{save_folder}_not_done.npy", self.not_done[:self.crt_size])
        np.save(f"{save_folder}_first_timestep.npy", self.first_timestep[:self.crt_size])
        np.save(f"{save_folder}_replay_info.npy", [self.ptr, chunk])

        crt = 0
        end = min(chunk, self.crt_size + 1)
        while crt < self.crt_size + 1:
            np.save(f"{save_folder}_state_{end}.npy", self.state[crt:end])
            crt = end
            end = min(end + chunk, self.crt_size + 1)

    def load(self, save_folder, size=-1):
        reward_buffer = np.load(f"{save_folder}_reward.npy")
        size = min(int(size), self.max_size) if size > 0 else self.max_size
        self.crt_size = min(reward_buffer.shape[0], size)

        # Adjust crt_size if we're using a custom size
        size = min(int(size), self.max_size) if size > 0 else self.max_size
        self.crt_size = min(reward_buffer.shape[0], size)

        self.action[:self.crt_size] = np.load(f"{save_folder}_action.npy")[:self.crt_size]
        self.reward[:self.crt_size] = reward_buffer[:self.crt_size]
        self.not_done[:self.crt_size] = np.load(f"{save_folder}_not_done.npy")[:self.crt_size]
        self.first_timestep[:self.crt_size] = np.load(f"{save_folder}_first_timestep.npy")[:self.crt_size]

        self.ptr, chunk = np.load(f"{save_folder}_replay_info.npy")

        crt = 0
        end = min(chunk, self.crt_size + 1)
        while crt < self.crt_size + 1:
            self.state[crt:end] = np.load(f"{save_folder}_state_{end}.npy")
            crt = end
            end = min(end + chunk, self.crt_size + 1)

    def __len__(self):
        return self.crt_size
