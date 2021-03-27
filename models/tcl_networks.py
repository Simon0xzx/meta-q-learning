from __future__ import  print_function, division
import copy
import torch
import torch.nn as nn
from misc.utils import get_action_info

##################################
# Actor and Critic Newtork for TD3
##################################
class TCLActor(nn.Module):
    """
      This arch is standard based on https://github.com/sfujim/TD3/blob/master/TD3.py
    """
    def __init__(self, action_space,
                 hidden_sizes = [400, 300],
                 input_dim = None,
                 max_action= None,
                 latent_dim=7):

        super(TCLActor, self).__init__()
        self.hsize_1 = hidden_sizes[0]
        self.hsize_2 = hidden_sizes[1]
        action_dim, action_space_type = get_action_info(action_space)
        self.latent_dim = latent_dim
        self.actor = nn.Sequential(
                        nn.Linear(input_dim[0], self.hsize_1),
                        nn.ReLU(),
                        nn.Linear(self.hsize_1, self.hsize_2),
                        nn.ReLU())
        self.out = nn.Linear(self.hsize_2,  action_dim)
        self.max_action = max_action


    def forward(self, x, latent):
        '''
            input (x  : B * D where B is batch size and D is input_dim
            pre_act_rew: B * (A + 1) where B is batch size and A + 1 is input_dim
        '''
        xz = torch.cat([x, latent], dim = -1)

        output = self.actor(xz)
        output = self.max_action * torch.tanh(self.out(output))

        return output

class TCLCritic(nn.Module):
    """
      This arch is standard based on https://github.com/sfujim/TD3/blob/master/TD3.py
    """
    def __init__(self,
                action_space,
                hidden_sizes = [400, 300],
                input_dim = None,
                latent_dim = 7
                ):

        super(TCLCritic, self).__init__()
        self.hsize_1 = hidden_sizes[0]
        self.hsize_2 = hidden_sizes[1]
        action_dim, action_space_type = get_action_info(action_space)
        # handling extra dim
        self.latent_dim = latent_dim # right now, we add reward + previous action

        # It uses two different Q networks
        # Q1 architecture
        self.q1 = nn.Sequential(nn.Linear(input_dim[0] + action_dim + self.latent_dim, self.hsize_1),
                                nn.ReLU(),
                                nn.Linear(self.hsize_1, self.hsize_2),
                                nn.ReLU(),
                                nn.Linear(self.hsize_2, 1))


        # Q2 architecture
        self.q2 = nn.Sequential(nn.Linear(input_dim[0] + action_dim + self.latent_dim, self.hsize_1),
                                nn.ReLU(),
                                nn.Linear(self.hsize_1, self.hsize_2),
                                nn.ReLU(),
                                nn.Linear(self.hsize_2, 1))

    def forward(self, x, u, latent):
        '''
            input (x): B * D where B is batch size and D is input_dim
            input (u): B * A where B is batch size and A is action_dim
            pre_act_rew: B * (A + 1) where B is batch size and A + 1 is input_dim
        '''
        xu = torch.cat([x, u], 1)
        xuz = torch.cat([xu, latent], dim = -1)
        x1 = self.q1(xuz) # Q1
        x2 = self.q2(xuz) # Q2
        return x1, x2

    def Q1(self, x, u, latent):
        '''
            input (x): B * D where B is batch size and D is input_dim
            input (u): B * A where B is batch size and A is action_dim
            pre_act_rew: B * (A + 1) where B is batch size and A + 1 is input_dim
        '''

        xu = torch.cat([x, u], 1)
        xuz = torch.cat([xu, latent], dim = -1)
        x1 = self.q1(xuz) # Q1
        return x1

class TCLContext(nn.Module):
    """
      This layer just does non-linear transformation(s)
    """
    def __init__(self, hidden_sizes = [50],
                 output_dim = None,
                 input_dim = None,
                 action_dim = None,
                 obsr_dim = None,
                 enable_masking = False,
                 device = 'cpu'):

        super(TCLContext, self).__init__()
        self.hidden_sizes = hidden_sizes
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        self.action_dim = action_dim
        self.obsr_dim = obsr_dim
        self.enable_masking = enable_masking

        #### build LSTM or GRU for both query and key
        self.query_recur = nn.GRU(self.input_dim,
                                self.hidden_sizes[0],
                                bidirectional = False,
                                batch_first = True,
                                num_layers = 1)
        self.key_recur = copy.deepcopy(self.query_recur)

        if self.enable_masking:
            self.query_mlp = nn.Sequential(
                nn.Linear(self.hidden_sizes[0], 100),
                nn.ReLU(),
                nn.Linear(100, self.output_dim),
                nn.ReLU())
            self.key_mlp = copy.deepcopy(self.query_mlp)


    def init_recurrent(self, bsize = None):
        '''
            init hidden states
            Batch size can't be none
        '''
        # The order is (num_layers, minibatch_size, hidden_dim)
        # LSTM ==> return (torch.zeros(1, bsize, self.hidden_sizes[0]),
        #        torch.zeros(1, bsize, self.hidden_sizes[0]))
        return torch.zeros(1, bsize, self.hidden_sizes[0]).to(self.device)

    def get_key_latent(self, data):
        previous_action, previous_reward, pre_x = data[0], data[1], data[2]

        # first prepare data for LSTM
        bsize, dim = previous_action.shape  # previous_action is B* (history_len * D)
        pacts = previous_action.view(bsize, -1, self.action_dim)  # view(bsize, self.hist_length, -1)
        prews = previous_reward.view(bsize, -1, 1)  # reward dim is 1, view(bsize, self.hist_length, 1)
        pxs = pre_x.view(bsize, -1, self.obsr_dim)  # view(bsize, self.hist_length, -1)
        pre_act_rew = torch.cat([pacts, prews, pxs], dim=-1)  # input to LSTM is [action, reward]

        # init lstm/gru
        hidden = self.init_recurrent(bsize=bsize)

        # lstm/gru
        _, hidden = self.key_recur(pre_act_rew, hidden)  # hidden is (1, B, hidden_size)
        if self.enable_masking:
            hidden = self.key_mlp(hidden)

        return hidden.squeeze(0)  # (1, B, hidden_size) ==> (B, hidden_size)

    def get_query_latent(self, data):
        previous_action, previous_reward, pre_x = data[0], data[1], data[2]

        # first prepare data for LSTM
        bsize, dim = previous_action.shape  # previous_action is B* (history_len * D)
        pacts = previous_action.view(bsize, -1, self.action_dim)  # view(bsize, self.hist_length, -1)
        prews = previous_reward.view(bsize, -1, 1)  # reward dim is 1, view(bsize, self.hist_length, 1)
        pxs = pre_x.view(bsize, -1, self.obsr_dim)  # view(bsize, self.hist_length, -1)
        pre_act_rew = torch.cat([pacts, prews, pxs], dim=-1)  # input to LSTM is [action, reward]

        # init lstm/gru
        hidden = self.init_recurrent(bsize=bsize)

        # lstm/gru
        _, hidden = self.query_recur(pre_act_rew, hidden)  # hidden is (1, B, hidden_size)
        if self.enable_masking:
            hidden = self.query_mlp(hidden)

        return hidden.squeeze(0)  # (1, B, hidden_size) ==> (B, hidden_size)

    def forward(self, data):
        '''
            pre_x : B * D where B is batch size and D is input_dim
            pre_a : B * A where B is batch size and A is input_dim
            previous_reward: B * 1 where B is batch size and 1 is input_dim
        '''
        previous_action, previous_reward, pre_x = data[0], data[1], data[2]

        # first prepare data for LSTM
        bsize, dim = previous_action.shape # previous_action is B* (history_len * D)
        pacts = previous_action.view(bsize, -1, self.action_dim) # view(bsize, self.hist_length, -1)
        prews = previous_reward.view(bsize, -1, 1) # reward dim is 1, view(bsize, self.hist_length, 1)
        pxs   = pre_x.view(bsize, -1, self.obsr_dim) # view(bsize, self.hist_length, -1)
        pre_act_rew = torch.cat([pacts, prews, pxs], dim = -1) # input to LSTM is [action, reward]

        # init lstm/gru
        hidden = self.init_recurrent(bsize=bsize)

        # lstm/gru
        _, hidden = self.query_recur(pre_act_rew, hidden) # hidden is (1, B, hidden_size)
        out = hidden.squeeze(0) # (1, B, hidden_size) ==> (B, hidden_size)

        return out

    @property
    def network(self):
        nets = [self.query_recur, self.key_recur]
        if self.enable_masking:
            nets.extend([self.query_mlp, self.key_mlp])
        return nets
