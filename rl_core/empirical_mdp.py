from logging import warnings
from timeit import Timer
from .utils_misc import timer_at_func
import torch 
from tqdm import tqdm


class MDPBuild:

    @staticmethod
    def fetch_row_index(matrix,vector):
        matches = (torch.sum(matrix==vector, dim = 1) == len(vector)).nonzero()
        
        if len(matches)>1:
            assert False, matches
        elif len(matches) == 1: 
            return matches.item()

        else:
            assert False, "no index found"

    @staticmethod
    def get_empirical_mdp_params_from_transitions(transitions, n_targets = 5):

        all_states = torch.tensor(list(map(lambda t:t[0], transitions)))
        all_next_states = torch.tensor(list(map(lambda t:t[2], transitions)))
        all_actions = torch.tensor(list(map(lambda t:t[1], transitions)))
        
        end_state_vector = torch.full_like(all_states[0],999999)
        S_raw = torch.unique(torch.cat((all_states,all_next_states)), dim =0)
        S =  torch.cat((end_state_vector.unsqueeze(0), S_raw))
        A = torch.unique(all_actions, dim =0)
        S.index = lambda s:MDPBuild.fetch_row_index(S,s)
        A.index = lambda s:MDPBuild.fetch_row_index(A,s)
        n_core_states = len(S)
        n_actions = len(A)
        # print(S)
        # print(S.index(torch.FloatTensor((2))))


        nn,aa,tt = n_core_states,n_actions, n_targets
        Ti = torch.zeros((nn,aa,tt)).to(device='cpu').type(torch.LongTensor)
        Tp_raw = torch.zeros((nn,aa,tt)).to(device='cpu').type(torch.LongTensor)
        Tp = torch.zeros((nn,aa,tt)).to(device='cpu').type(torch.float32)
        R_raw = torch.zeros((nn,aa,tt)).to(device='cpu').type(torch.float32)
        R = torch.zeros((nn,aa,tt)).to(device='cpu').type(torch.float32)
        Q = torch.zeros((nn,aa)).to(device='cpu').type(torch.float32)
        V = torch.zeros((nn,)).to(device='cpu').type(torch.float32)
        Pi = torch.zeros((nn)).to(device='cpu').type(torch.LongTensor)

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

        return (S,A), (Ti, Tp, R, Q, V, Pi), (R_raw,Tp_raw)




class MDP:
    """
    Simple MDP with a small number of Discrete Actions.
    """

    def __init__(self,space_vectors,core_vectors, gamma = 0.99):

        S, A = space_vectors
        self.S = torch.Tensor(S).to(device='cuda').type(torch.float32)
        self.A = torch.Tensor(A).to(device='cuda').type(torch.float32)
        self.S.index = lambda s:MDPBuild.fetch_row_index(S,s)
        self.A.index = lambda s:MDPBuild.fetch_row_index(A,s)

        # Action index track the index of the action for which we are tracking the target states.
        # Tran index tracks the index of each target state. 
        # Tran prob tracks the prob of landing on each target state. 
        # R tracks the reward for each target state
        # Q tracks the Q value for each action slots.
        # V tracks the value for each state
        # Pi tracks the best action slot for each state 
        Ti, Tp, R, Q, V, Pi = core_vectors
        # Sanity check 
        n,a,t = Ti.shape # Number of States, Number of Actions, Number of Target States
        # assert Ai.shape == (n,a) , (Ti.shape , Ai.shape)
        assert Tp.shape == (n,a,t) , (Ti.shape , Tp.shape)
        assert R.shape == (n,a,t), (Ti.shape , R.shape)
        assert Q.shape == (n,a), (Ti.shape , Q.shape)
        assert V.shape == (n,) , (Ti.shape , V.shape, n)
        assert Pi.shape == (n,), (Ti.shape , Pi.shape, n)

        # self.Ai = torch.LongTensor(Ai).to(device='cuda')
        self.Ti = torch.LongTensor(Ti).to(device='cuda')
        self.Tp = torch.FloatTensor(Tp).to(device='cuda')
        self.R = torch.FloatTensor(R).to(device='cuda')
        self.Q = torch.FloatTensor(Q).to(device='cuda')
        self.V = torch.FloatTensor(V).to(device='cuda')
        self.Pi = torch.LongTensor(Pi).to(device='cuda')

        self.gamma = torch.FloatTensor([gamma]).to(device='cuda')

    @torch.jit.script
    def bellman_backup(Ti, Tp, R, Q, V, gamma):
        Q = torch.sum(torch.multiply(R, Tp),dim=2)  + \
        gamma[0]*torch.sum(torch.multiply(Tp, V[Ti]), dim=2)
        max_obj = torch.max(Q, dim=1)
        V_prime, Pi = max_obj.values, max_obj.indices
        epsilon = torch.max(V_prime-V)
        return epsilon, V_prime, Pi

    @timer_at_func
    def solve(self, epsilon = 0.001, n_backups=2000, verbose = False):
        v_iter = tqdm(range(n_backups)) if verbose else range(n_backups)
        n_backups = 0
        for i in v_iter:
            self.curr_error, self.V, self.Pi = MDP.bellman_backup(self.Ti, self.Tp, self.R, self.Q, self.V, self.gamma)
            if self.curr_error < epsilon:
                n_backups = i
                break
            if i%250 == 0:
                print(i,self.curr_error)
        print(f"Solved MDP in {n_backups} Backups")

    def nn_index_for_state(self,s):
        dist = torch.norm(self.S - s, dim=1, p=None)
        knn = dist.topk(1, largest=False)
        return knn.indices[0]

    def policy(self,s):
        s_i = self.nn_index_for_state(torch.tensor(s).to(device = "cuda"))
        a_i = self.Pi[s_i]
        return self.A[a_i]

    def batch_policy(s_batch):
        pass

            

# Sanity Check
def sanity_check():
    import torch
    import time
    from tqdm.notebook import tqdm

    N,A,T = 1000000,10,5 # Number of states, actions, targets
    # N,A,T = 2,2,2 # Number of states, actions, targets

    s = time.time()
    Tp_raw = torch.rand((N,A,T)).to(device='cuda').type(torch.float32)
    Tp = torch.nn.Softmax(dim = 2)(Tp_raw)
    R = torch.rand((N,A,T)).to(device='cuda').type(torch.float32)
    Ti = torch.randint(0, N, (N,A,T)).to(device='cuda').type(torch.LongTensor)
    Q = torch.zeros((N,A)).to(device='cuda').type(torch.float32)
    V = torch.zeros((N,)).to(device='cuda').type(torch.float32)
    Pi = torch.zeros((N,A)).to(device='cuda').type(torch.LongTensor)
    t = time.time()
    print("Time for Instantiating vectors",t-s)

    # bellman backup 
    curr_error = 10
    gamma = 0.99
    for i in tqdm(range(2000)):
        epsilon, V, Pi = MDP.bellman_backup(Ti, Tp, R, Q, V)
        if i%250 == 0:
            print(i,epsilon)