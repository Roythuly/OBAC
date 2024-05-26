import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
import copy
from .utils import soft_update, hard_update
from .model import GaussianPolicy, QNetwork, DeterministicPolicy, ValueNetwork

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20


class OBAC(object):
    def __init__(self, num_inputs, action_space, args):

        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha
        self.quantile = args.quantile
        self.bc_weight = args.bc_weight

        self.policy_type = args.policy
        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning

        self.device = torch.device("cuda:{}".format(str(args.device)) if args.cuda else "cpu")

        self.critic = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)

        self.critic_target = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
        hard_update(self.critic_target, self.critic)

        if self.policy_type == "Gaussian":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=args.lr)

            self.policy = GaussianPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)
        
        self.critic_buffer = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(device=self.device)
        self.critic_buffer_optim = Adam(self.critic_buffer.parameters(), lr=args.lr)
        hard_update(self.critic_buffer, self.critic)

        self.critic_target_buffer = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
        hard_update(self.critic_target_buffer, self.critic_buffer)

        self.V_critic_buffer = ValueNetwork(num_inputs, args.hidden_size).to(device=self.device)
        self.V_critic_buffer_optim = Adam(self.V_critic_buffer.parameters(), lr=args.lr)

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if evaluate is False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]

    def update_parameters(self, memory, batch_size, updates):
        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)
        
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)
        
        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            # Compute the target Q value for current policy
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)

        # compute the Q loss for current policy
        qf1, qf2 = self.critic(state_batch, action_batch)
        qf1_loss = F.mse_loss(qf1, next_q_value) 
        qf2_loss = F.mse_loss(qf2, next_q_value)
        qf_loss = qf1_loss + qf2_loss
        
        # Compute the target Q value for behavior policy
        vf_pred = self.V_critic_buffer(state_batch)
        target_Vf_pred = self.V_critic_buffer(next_state_batch)
        next_q_value_buffer = reward_batch + mask_batch * self.gamma * target_Vf_pred
        
        # compute the Q loss for behavior policy
        qf1_buffer, qf2_buffer = self.critic_buffer(state_batch, action_batch)
        qf_buffer = torch.min(qf1_buffer, qf2_buffer).mean()   # compute the Q value for (s,a) pair under the behavior policy
        qf1_buffer_loss = F.mse_loss(qf1_buffer, next_q_value_buffer)  
        qf2_buffer_loss = F.mse_loss(qf2_buffer, next_q_value_buffer)
        qf_buffer_loss = qf1_buffer_loss + qf2_buffer_loss
        
        # compute the V loss for behavior policy
        q_pred_1, q_pred_2 = self.critic_target_buffer(state_batch, action_batch)
        q_pred = torch.min(q_pred_1, q_pred_2)
        vf_err = q_pred - vf_pred
        vf_sign = (vf_err < 0).float()
        vf_weight = (1 - vf_sign) * self.quantile + vf_sign * (1 - self.quantile)
        vf_loss = (vf_weight * (vf_err ** 2)).mean()
        
        # compute action by current policy
        pi, log_pi, _ = self.policy.sample(state_batch)
        # estimate the Q value 
        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi) # compute the Q value for (s,a) pair under the current policy
        qf_pi = min_qf_pi.mean()
        
        if updates == 0:
            self.policy_loss = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_loss = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_tlogs = torch.zeros(1, requires_grad=True, device=self.device)
            
        # update Q value of current policy
        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()
        
        # update Q value of behavior policy
        self.critic_buffer_optim.zero_grad()
        qf_buffer_loss.backward()
        self.critic_buffer_optim.step()
        
        # update V value of behavior policy
        self.V_critic_buffer_optim.zero_grad()
        vf_loss.backward()
        self.V_critic_buffer_optim.step()
        
        if updates % self.target_update_interval == 0:
            if qf_pi >= qf_buffer:  # means current policy can surpass behavior policy; or current policy can get exploration bonus
                policy_loss = (self.alpha * log_pi - min_qf_pi).mean()
            else:
                log_density = self.policy.get_log_density(state_batch, action_batch)
                log_density = torch.clamp(log_density, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
                policy_loss = (self.alpha * log_pi - self.bc_weight * log_density - min_qf_pi).mean()
            
            # update policy
            self.policy_optim.zero_grad()
            policy_loss.backward()
            self.policy_optim.step()

            if self.automatic_entropy_tuning:
                alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

                self.alpha_optim.zero_grad()
                alpha_loss.backward()
                self.alpha_optim.step()

                self.alpha = self.log_alpha.exp()
                alpha_tlogs = self.alpha.clone() # For TensorboardX logs
            else:
                alpha_loss = torch.tensor(0.).to(self.device)
                alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs
            
            soft_update(self.critic_target, self.critic, self.tau)
            soft_update(self.critic_target_buffer, self.critic_buffer, self.tau)
            self.policy_loss = copy.copy(policy_loss)
            self.alpha_loss = copy.copy(alpha_loss)
            self.alpha_tlogs = copy.copy(alpha_tlogs)
            
        return qf1_loss.item(), qf2_loss.item(), vf_loss.item(), self.policy_loss.item(), self.alpha_loss.item(), self.alpha_tlogs.item(), qf_pi.item(), qf_buffer.item()
    
    # Save model parameters
    def save_checkpoint(self, path, i_episode):
        ckpt_path = path + '/' + '{}.torch'.format(i_episode)
        print('Saving models to {}'.format(ckpt_path))
        torch.save({'policy_state_dict': self.policy.state_dict(),
                    'critic_state_dict': self.critic.state_dict(),
                    'critic_target_state_dict': self.critic_target.state_dict(),
                    'critic_optimizer_state_dict': self.critic_optim.state_dict(),
                    'policy_optimizer_state_dict': self.policy_optim.state_dict(),
                    'critic_buffer_state_dict': self.critic_buffer.state_dict(),
                    'critic_target_buffer_state_dict': self.critic_target_buffer.state_dict(),
                    'critic_buffer_optimizer_state_dict': self.critic_buffer_optim.state_dict(),
                    'V_critic_buffer_state_dict': self.V_critic_buffer.state_dict(),
                    'V_critic_buffer_optimizer_state_dict': self.V_critic_buffer_optim.state_dict()
                    },
                    ckpt_path)
    
    # Load model parameters
    def load_checkpoint(self, path, i_episode, evaluate=False):
        ckpt_path = path + '/' + '{}.torch'.format(i_episode)
        print('Loading models from {}'.format(ckpt_path))
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path)
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
            self.critic_optim.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self.policy_optim.load_state_dict(checkpoint['policy_optimizer_state_dict'])
            self.critic_buffer.load_state_dict(checkpoint['critic_buffer_state_dict'])
            self.critic_target_buffer.load_state_dict(checkpoint['critic_target_buffer_state_dict'])
            self.critic_buffer_optim.load_state_dict(checkpoint['critic_buffer_optimizer_state_dict'])
            self.V_critic_buffer.load_state_dict(checkpoint['V_critic_buffer_state_dict'])
            self.V_critic_buffer_optim.load_state_dict(checkpoint['V_critic_buffer_optimizer_state_dict'])

            if evaluate:
                self.policy.eval()
                self.critic.eval()
                self.critic_target.eval()
                self.critic_buffer.eval()
                self.critic_target_buffer.eval()
                self.V_critic_buffer.eval()
            else:
                self.policy.train()
                self.critic.train()
                self.critic_target.train()
                self.critic_buffer.train()
                self.critic_target_buffer.train()
                self.V_critic_buffer.train()
