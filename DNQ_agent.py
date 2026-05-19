import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as opt
import numpy as np



class DNQ(nn.Module):
    def __init__(self, output_size=9):
        super(DNQ, self).__init__()
        #conv 4,84,84-> 16, 20,20
        self.conv1 = nn.Conv2d(4,16,kernel_size=8,stride=4)
        #conv 16,20,20-> 32,9,9

        self.conv2 = nn.Conv2d(16,32,kernel_size=4,stride=2)

        self.fc1 = nn.Linear(32 * 9 * 9 ,256)
        self.fc2 = nn.Linear(256 ,output_size)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        #flatten to fit the fc
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        
        return (self.fc2(x))



class DNQAgent:
    def __init__(self,  device, actions=9, gamma=0.9):
        
        #setup DNQ
        self.device = device
        self.DNQ = DNQ(actions).to(self.device)
        self.loss_fn = torch.nn.MSELoss()
        self.optimizer= opt.RMSprop(self.DNQ.parameters(), lr=0.0001) #well play with the lr later
        self.gamma = gamma


    # infrence time, finds best action 
    def select_action(self, frame):
        state_tensor = torch.from_numpy(np.array(frame)).float().unsqueeze(0).to(self.device)
        self.DNQ.eval()
        with torch.no_grad():
            # need to add 'unsqueeze' for cnn to work, adds a dim in satart for batch size
            q_values = self.DNQ(state_tensor)
        self.DNQ.train()

        #pick the acton with highst score
        best_action = q_values.argmax().detach().cpu().item()
        return best_action


    '''
        here is the entire learning process. recives a batch of state from the 
        buffer {('state','action','reward','next_state', 'ended'),()...}. ff to find *current* reward for action Si
        then plugges into bellmon eq then fowrd on Si+1 and MSE on the dif
    '''
    def learn_samples(self, batch):
        #extract state, conter to tensor. size: (batch_size,4,84,84). ff-> (batch_size,9) (every action per state)
        
        states, actions, rewards, next_states, dones = zip(*batch)
        #convert to tensors
        states_t = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        actions_t = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(self.device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32).to(self.device)

        # bounding transformation [-1.0, 1.0] 
        rewards_t = torch.clamp(rewards_t, min=-1.0, max=1.0).to(self.device)

        next_states_t = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32).to(self.device)

       # set up the Qs+1 values for bellmon, meaning how much we'd make from next_state
        with torch.no_grad():
            
            next_q_values = self.DNQ(next_states_t)
            #pick the acton with highst score
            max_next_q_values = next_q_values.max(1)[0]
            #trick from gemini,if finished it will be only the reward, like the paper
            expected_q_values = rewards_t + (self.gamma * max_next_q_values * (1 - dones_t)) 
        

        #pick the matching action to what was done, set as 
        q_values = self.DNQ(states_t)

        current_q_values = q_values.gather(1, actions_t).squeeze(1)

        loss = self.loss_fn(current_q_values, expected_q_values)

        self.optimizer.zero_grad()
        loss.backward()

        # ==========================================
        # DIAGNOSTIC: Calculate the L2 Gradient Norm
        # ==========================================
        # total_norm = 0.0
        # for p in self.DNQ.parameters():
        #     if p.grad is not None:
        #         # Calculate the L2 norm of the gradients for this specific layer
        #         param_norm = p.grad.data.norm(2)
        #         # Square it and add to the total sum
        #         total_norm += param_norm.item() ** 2
        # 
        # # Take the square root of the total sum
        # total_norm = total_norm ** 0.5
        # 
        # # Print the scalar loss and the magnitude of the update
        # print(f"Loss: {loss.item():.4f} | Update Magnitude (L2 Norm): {total_norm:.4f}")
        # ==========================================

        self.optimizer.step()





