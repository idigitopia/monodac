from collections import defaultdict
from re import M
from pyrsistent import T
import torch
from tqdm import tqdm
from functools import partial


class THelper():
    @staticmethod
    def lookup_index_by_nn(vector, matrix):
        matches = (torch.sum(matrix == vector, dim=1) == len(vector)).nonzero()

        if len(matches) > 1:
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
        index_by_hash = defaultdict(None, {hash(r): i for i, r in enumerate(torch_matrix)})
        return index_by_hash[hash(query_row)]

    @staticmethod
    def calc_knn_indices(query:torch.Tensor,data:torch.Tensor,k):
        dist = torch.norm(data - query, dim=1, p=None)
        knn = dist.topk(k, largest=False)
        return knn.indices

    @staticmethod
    def calc_knn(query:torch.Tensor,data:torch.Tensor,k):
        dist = torch.norm(data - query, dim=1, p=None)
        knn = dist.topk(k, largest=False)
        return knn

    @staticmethod
    def batch_calc_knn_ret_flat(query_batch:torch.Tensor,data:torch.Tensor, k):
        knn_batch = [THelper.calc_knn(q, data) for q in query_batch]
        knn_indices_flat = torch.concat([knn.indices for knn in knn_batch])
        knn_values_flat = torch.concat([knn.values for knn in knn_batch])
        return knn_indices_flat, knn_values_flat
# vanilla dac build logic.
# state representation = Identity
# candidate_action_selection = nn on state representation
# state action representation = cand action next_state


class DACMDP():

    ########################### Rabbit King ####################################
    @torch.jit.script
    def bellman_backup(Ti, Tp, R, Q, V, gamma, device='cuda'):
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
        Q = torch.sum(torch.multiply(R, Tp), dim=2) + \
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

    def __init__(self, transitions, n_tran_types=10, n_tran_targets=5, sa_repr_dim = None, sa_repr_fxn = None, device='cpu'):
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
        all_states = torch.tensor(list(map(lambda t: t[0], transitions)))
        all_next_states = torch.tensor(list(map(lambda t: t[2], transitions)))
        all_actions = torch.tensor(list(map(lambda t: t[1], transitions)))

        ####################### Get the total state Space and Action Space for the MDP ##################
        self.end_state_vector = torch.full_like(all_states[0], 999999)
        self.S = torch.FloatTensor(all_next_states)
        self.A = torch.unique(all_actions, dim=0)
        self.S.index = partial(THelper.lookup_index_by_hash, torch_matrix=self.S)
        self.A.index = partial(THelper.lookup_index_by_hash, torch_matrix=self.A)
        self.S.len, self.A.len = len(self.S), len(self.A)
        nn, aa, tt = self.S.len, n_tran_types, n_tran_targets
        self.n_tran_types, self.n_tran_targets = n_tran_types, n_tran_targets
        ###############################################################################################################

        # Vectors Defined below are all all placeholders, all must be updated for sane values.
        ###############################################################################################################
        # Fit Action Choice Model Here.
        # For each state, the action index vector track the index of candidate action for that state.
        # The index can be then tracked back to original action in the environment during policy lift.
        # If we are not interested in lifting the policy the tensor can be all zeros.
        # This tensor is not used during value or policy calculation.
        Ai = torch.zeros((nn, aa)).to(device=device).type(torch.LongTensor)  # ToDo find a way to fill this one up.
        ###############################################################################################################

        # Transitions Representations.
        # Ai can only be used if tran_repr are built independent of tran_repr. 
        self.sa_repr_dim = 16
        self.D_repr = torch.zeros((nn,sa_repr_dim)).to(device=device).type(torch.float32) 
        self.T_repr = torch.zeros((nn, aa, sa_repr_dim)).to(device=device).type(torch.float32)
        self.T_repr_nn_indices = torch.zeros((nn, aa, tt)).to(device=device).type(torch.LongTensor)
        ###############################################################################################################
        self.Tp = torch.zeros((nn, aa, tt)).to(device=device).type(torch.float32)
        self.Ti = torch.zeros((nn, aa, tt)).to(device=device).type(torch.LongTensor)
        self.R = torch.zeros((nn, aa, tt)).to(device=device).type(torch.float32)
        self.Q = torch.zeros((nn, aa)).to(device=device).type(torch.float32)
        self.V = torch.zeros((nn,)).to(device=device).type(torch.float32)
        self.Pi = torch.zeros((nn)).to(device=device).type(torch.LongTensor)
        ###############################################################################################################

        ###################### Start with calculating all necessary state action representation ###############################
        # for each idx of dataset , find nn_state
        self.sa_repr_dim =  sa_repr_dim# ToDo Set it here
        # data store for state_action pairs of transitions.
        SA = torch.zeros((nn,)).to(device=device).type(torch.float32)
        SA.index(repr)  # should return a index if exists.
        SA.knn_indices(repr)  # should return knn
        #######################################################################################################################

    def set_candidate_action_indices(self, cand_action_idx_sets: torch.tensor, state_indices: torch.tensor) -> bool:
        n, t = cand_action_idx_sets.shape
        assert t == self.n_tran_types and  n == len(state_indices) 
        self.Ai[state_indices.to(self.deviece)] = cand_action_idx_sets.to(self.device)
        return True

    def set_transition_reprsentations(self,tran_repr_sets: torch.tensor, state_indices:torch.tensor )->bool:
        """ each transtition representation represents a transtion type and is used to 
        infer the target states. 
        Returns:
            _type_: _description_
        """
        n, a, t = tran_repr_sets.shape
        assert a == self.n_tran_types and t == self.sa_repr_dim and n == len(state_indices)
        self.T_repr[state_indices.to(self.device)] = tran_repr_sets.to(self.device)
        return True

    def set_dataset_representations(self, sa_reprs: torch.tensor, state_indices:torch.tensor)->bool:
        n, r_dim = sa_reprs.shape
        assert len(state_indices) == n and r_dim == self.sa_repr_dim
        self.D_repr[state_indices.to(self.device)] = sa_reprs.to(self.device) 
        return True

    def update_tran_vectors(self, state_indices:torch.tensor)->bool:
        # Get KNN for each 
        T_repr_slice = self.T_repr[state_indices]
        nn,aa,s_dim = T_repr_slice.shape
        
        all_sa_reprs = T_repr_slice.view((-1, self.s_repr_dim))
        nn_indices_flat, nn_values_flat = THelper.batch_calc_knn_ret_flat(all_sa_reprs, k= self.n_tran_targets) 
        knn_idx_tensor = nn_indices_flat.view((nn,aa,self.n_tran_targets))
        knn_values_tensor = nn_indices_flat.view((nn,aa,self.n_tran_targets))

        assert self.Ti[state_indices].shape == knn_idx_tensor.shape
        assert self.Tp[state_indices].shape == knn_values_tensor.shape

        self.Ti[state_indices] = knn_idx_tensor
        self.Tp[state_indices] = torch.nn.Softmax(dim = 2)(knn_values_tensor)

        return True

    def compute_bellman_backups(self, gamma=0.99, n_backups=2000, verbose=False) -> None:
        v_iter = tqdm(range(n_backups)) if verbose else range(n_backups)
        n_backups = 0
        for i in v_iter:
            curr_error, self.V, self.Pi = DACMDP.bellman_backup(self.Ti, self.Tp, self.R, self.Q, self.V, gamma)

    def solve(self, gamma=0.99, epsilon=0.001, n_backups=2000, verbose=False):
        v_iter = tqdm(range(n_backups)) if verbose else range(n_backups)
        n_backups = 0
        for i in v_iter:
            self.curr_error, self.V, self.Pi = DACMDP.bellman_backup(self.Ti, self.Tp, self.R, self.Q, self.V, gamma)
            if self.curr_error < epsilon:
                n_backups = i
                break
            if i % 250 == 0:
                print(i, self.curr_error)
        print(f"Solved MDP in {n_backups} Backups")
