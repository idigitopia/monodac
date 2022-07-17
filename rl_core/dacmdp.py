from collections import defaultdict
from re import M
import torch
from tqdm import tqdm
from functools import partial
class THelper():
    @staticmethod
    def lookup_index_by_nn(vector,matrix):
        matches = (torch.sum(matrix==vector, dim = 1) == len(vector)).nonzero()
        
        if len(matches)>1:
            assert False, matches
        elif len(matches) == 1: 
            return matches.item()
        else:
            assert False, "no index found" 
    
    @staticmethod
    def lookup_index_by_hash(query_row, torch_matrix):
        """
        Returns a function for quick lookup for an index of a tensor.
        Only Factors in for 5 Decimal point

        Returns:
            _type_: _description_
        """ 
        index_by_hash = defaultdict(None, {hash(r):i for i,r in enumerate(torch_matrix)} )
        return index_by_hash[hash(query_row)]


    # @staticmethod
    # def calc_knn_indices(data,test,k):
    #     n,d = data.shape
    #     D = torch.tensor()
    #     for t in test:
    #         dist = torch.norm(data - test, dim=1, p=None)
    #         knn = dist.topk(1, largest=False)

    #     return knn.indices[0]




# vanilla dac build logic.
# state representation = Identity
# candidate_action_selection = nn on state representation
# state action representation = cand action next_state



class DACMDP():

    ########################### Rabbit King ####################################
    @torch.jit.script
    def bellman_backup(Ti, Tp, R, Q, V, gamma, device = 'cuda'):
        """
        Computes 1 full Synchronous Bellman Backup.

        Args:
            Ti (_type_): _description_
            Tp (_type_): _description_
            R (_type_): _description_
            Q (_type_): _description_
            V (_type_): _description_
            gamma (_type_): _description_
            device (str, optional): _description_. Defaults to 'cuda'.

        Returns:
            _type_: _description_
        """
        gamma = torch.FloatTensor([gamma]).to(device)
        Q = torch.sum(torch.multiply(R, Tp),dim=2)  + \
                gamma[0]*torch.sum(torch.multiply(Tp, V[Ti]), dim=2)
        max_obj = torch.max(Q, dim=1)
        V_prime, Pi = max_obj.values, max_obj.indices
        epsilon = torch.max(V_prime-V)
        return epsilon, V_prime, Pi
    ############################################################################


    # Designed to be updatable on the fly. 

    # Entities to think about. 
    # S as a current Dataset State space for a compressed representation of the observation. 
    # A as a current Dataset Action space for a 1-1 representation of the actions available.
    # SA as a current Dataset State-Action space for holding the representations for all sa pairs in the datset.  
    # i.e. change the state vectors of certain indices. 
    # i.e. change the candidate_action and sa_repr of  
    def __init__(self, transitions, n_tran_types = 10, n_tran_targets = 5):
        """_summary_

        Args:
            transitions (_type_): eg 
                [(s,a,ns,r,d),(s,a,ns,r,d)]
            n_tran_types (int, optional): _description_. Defaults to 10.
            n_tran_targets (int, optional): _description_. Defaults to 5.

        Returns:
            _type_: _description_
        """

        # ToDo Some sanity checkes for transitions

        self.transitions = transitions
        all_states = torch.tensor(list(map(lambda t:t[0], transitions)))
        all_next_states = torch.tensor(list(map(lambda t:t[2], transitions)))
        all_actions = torch.tensor(list(map(lambda t:t[1], transitions)))
        

        ####################### Get the total state Space and Action Space for the MDP ##################
        self.end_state_vector = torch.full_like(all_states[0],999999)
        self.S = torch.FloatTensor(all_next_states)
        self.A = torch.unique(all_actions, dim =0)
        self.S.index = partial(THelper.lookup_index_by_hash, torch_matrix=self.S)
        self.A.index = partial(THelper.lookup_index_by_hash, torch_matrix=self.A)
        self.S.len , self.A.len = len(self.S), len(self.A)
        nn,aa,tt = S.len, n_tran_types, n_tran_targets
        ###############################################################################################################


        # Vectors Defined below are all all placeholders, all must be updated for sane values. 
        ###############################################################################################################
        # Fit Action Choice Model Here. 
        # For each state, the action index vector track the index of candidate action for that state. 
        # The index can be then tracked back to original action in the environment during policy lift.
        # If we are not interested in lifting the policy the tensor can be all zeros. 
        # This tensor is not used during value or policy calculation.
        Ai = torch.zeros((nn,aa)).to(device='cpu').type(torch.LongTensor) # ToDo find a way to fill this one up. 
        ###############################################################################################################

        # Transitions Representations. 
        # Ai can only be used if tran_repr are used to build the tran_reprs. 
        sa_repr_dim = 16
        tran_reprs = torch.zeros((nn,aa,sa_repr_dim)).to(device='cpu').type(torch.float32)

        ###############################################################################################################
        Tp = torch.zeros((nn,aa,tt)).to(device='cpu').type(torch.float32)
        Ti = torch.zeros((nn,aa,tt)).to(device='cpu').type(torch.LongTensor)
        R = torch.zeros((nn,aa,tt)).to(device='cpu').type(torch.float32)
        Q = torch.zeros((nn,aa)).to(device='cpu').type(torch.float32)
        V = torch.zeros((nn,)).to(device='cpu').type(torch.float32)
        Pi = torch.zeros((nn)).to(device='cpu').type(torch.LongTensor)
        ###############################################################################################################
        

        ###################### Start with calculating all necessary state action representation ###############################
        # for each idx of dataset , find nn_state
        tran_count = len(transitions) 
        sa_repr_dim = # ToDo Set it here 
        SA = torch.zeros((nn,)).to(device='cpu').type(torch.float32)# data store for state_action pairs of transitions. 
        SA.index(repr) # should return a index if exists. 
        SA.knn_indices(repr) # should return knn 
        #######################################################################################################################


        # Start filling in the vectors now one state at a time.
        for s_i in range(S.len):
            for slot_i in range(n_actions):
                # get action for the slot. 
                # get sa_repr for the current state and action. 
                # calculate target prob and indices based on teh sa_repr of the sa_pair.
                # Calculate reward for the respective targets
                for 
        for s,a,ns,r,d in tqdm(transitions): 
            s_i, a_i, ns_i = S.index(torch.tensor(s)),\
                            A.index(torch.tensor(a)),\
                            S.index(torch.tensor(ns))
                        
            if d:
                ns_i = S.index(S[0])
            
            matching_slots = (Ti[s_i, a_i, :]  == ns_i).nonzero().numel()
            available_slot_idxs = (Ti[s_i, a_i, :]==0).nonzero().reshape(-1)
            assert matching_slots <=1 or d

            is_already_slotted = matching_slots>0
            if is_already_slotted:
                slot_idx = (Ti[s_i, a_i, :]  == ns_i).nonzero().reshape(-1)[0] 
            elif len(available_slot_idxs)>0:
                slot_idx = available_slot_idxs[0]
            else:
                continue

            # print(type(avail_slots[0]), avail_slots, Ti[s_i, a_i, :], s_i, a_i)
            Ti[s_i, a_i, slot_idx] = ns_i
            Tp_raw[s_i, a_i, slot_idx] += 1
            R_raw[s_i, a_i, slot_idx] += r

        idxs = R_raw.nonzero().t().chunk(chunks=3,dim=0)
        R[idxs] =R_raw[idxs]/ Tp_raw[idxs]
        Tp = torch.nn.Softmax(dim = 2)(torch.log(Tp_raw+0.00001))

        return (S,A), (Ai, Ti, Tp, R, Q, V, Pi), (R_raw,Tp_raw)


    def compute_bellman_backups(self, gamma = 0.99, n_backups=2000, verbose = False)->None:
        v_iter = tqdm(range(n_backups)) if verbose else range(n_backups)
        n_backups = 0
        for i in v_iter:
            curr_error, self.V, self.Pi = DACMDP.bellman_backup(self.Ti, self.Tp, self.R, self.Q, self.V, gamma)



    def solve(self, gamma = 0.99, epsilon = 0.001, n_backups=2000, verbose = False):
        v_iter = tqdm(range(n_backups)) if verbose else range(n_backups)
        n_backups = 0
        for i in v_iter:
            self.curr_error, self.V, self.Pi = DACMDP.bellman_backup(self.Ti, self.Tp, self.R, self.Q, self.V, gamma)
            if self.curr_error < epsilon:
                n_backups = i
                break
            if i%250 == 0:
                print(i,self.curr_error)
        print(f"Solved MDP in {n_backups} Backups")
