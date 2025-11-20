import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

#for intalling the tourch pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# note it uses cuda 11.8 ,so i have nvidia hardware that supports cuda 


class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        '''
        
        DQN Network initialization
        
        
        '''
        pass

    def forward(self, x):
        '''
        Forward pass of the network.
        returns--> Q-values for each action
        
        '''
        pass


######################################################################
class DQN_Agent:
    def __init__(self,input_shape,num_actions,lr=1e-4,gamma=0.99,device="cuda"):
        '''
        DQN_Agent initialization
        '''
        pass
######################################################################
    def select_action(self, state, epsilon):
        '''
        Select an action using epsilon-greedy policy 
        returns--> action 
        
        '''
        pass
    def remember(self, state, action, reward, next_state, done):
        '''
        Store experience in replay buffer
        
        '''
        pass
    def learn_from_replay(self):
        '''
        uses replay buffer and perform a training step
        
        '''
        pass
    def update_epsilon(self):
        '''
        Update epsilon for epsilon-greedy policy
        
        
        '''
        pass
    

   
######################################################################
    def train_step(self, batch):
        states = torch.tensor(batch["states"], dtype=torch.float32, device=self.device)
        actions = torch.tensor(batch["actions"], dtype=torch.int64, device=self.device)
        rewards = torch.tensor(batch["rewards"], dtype=torch.float32, device=self.device)
        next_states = torch.tensor(batch["next_states"], dtype=torch.float32, device=self.device)
        dones = torch.tensor(batch["dones"], dtype=torch.float32, device=self.device)

       
    '''
    it will perform a single training step using a batch of experiences from the replay buffer
    
    then it will calculate the predicted Q-value for the current states q_pred = Q(s,a)
    and the target Q-values using the Bellman equation q_target = r + y * max Q'
    
    returns--> the loss 
    
    '''
 ######################################################################   
    def update_target(self):
        ''' 
        Update target network by copying weights of policy network. 
        
        '''""
        pass
########################################################################
def save_model(self, filepath):
        """
        Saves the main model's weights to a file.
        """
        pass

def load_model(self, filepath):
        """
        Loads model weights from a file into the main model.
        """
        pass
        