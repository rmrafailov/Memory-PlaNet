import torch
from torch import jit
from torch import nn
import copy

# Model-predictive control planner with cross-entropy method and learned transition model
class MPCPlanner(nn.Module):
  __constants__ = ['action_size', 'planning_horizon', 'optimisation_iters', 'candidates', 'top_candidates']

  def __init__(self, action_size, planning_horizon, optimisation_iters, candidates, top_candidates, transition_model, reward_model, mode='continuous', num_actions=2):
    super().__init__()
    #self.transition_model, self.reward_model = transition_model, reward_model
    self.reward_model = reward_model
    # mode: str, 'continuous' for Gaussian action distributions, 'discrete' for categorical action distributions
    self.mode = mode
    self.action_size = action_size
    self.num_actions = num_actions
    self.planning_horizon = planning_horizon
    self.optimisation_iters = optimisation_iters
    self.candidates, self.top_candidates = candidates, top_candidates

  #@jit.script_method
  def forward(self, belief, state, transition_model):
    B, H, Z = belief.size(0), belief.size(1), state.size(1)
    belief, state = belief.unsqueeze(dim=1).expand(B, self.candidates, H).reshape(-1, H), state.unsqueeze(dim=1).expand(B, self.candidates, Z).reshape(-1, Z)
    if self.mode is 'continuous':
      # Initialize factorized belief over action sequences q(a_t:t+H) ~ N(0, I)
      action_mean, action_std_dev = torch.zeros(self.planning_horizon, B, 1, self.action_size, device=belief.device), torch.ones(self.planning_horizon, B, 1, self.action_size, device=belief.device)
    elif self.mode is 'discrete':
      # Initialize factorized belief over action sequences q(a_t:t+H) ~ categorical(v_i,k=1/num_actions)
      # i.e. initialize uniformly
      action_probs = torch.ones(self.planning_horizon, B, self.num_actions, device=belief.device)/self.num_actions  
      #action_probs = torch.new_full((self.planning_horizon, B, self.num_actions), fill_value=1/self.num_actions, device=belief.device)
    for _ in range(self.optimisation_iters):
      running_transition_model = copy.deepcopy(transition_model)
      running_transition_model.rnn.allow_writting = False
      # Evaluate J action sequences from the current belief (over entire sequence at once, batched over particles)
      if self.mode is 'continuous':
        actions = (action_mean + action_std_dev * torch.randn(self.planning_horizon, B, self.candidates, self.action_size, device=action_mean.device)).view(self.planning_horizon, B * self.candidates, self.action_size)  # Sample actions (time x (batch x candidates) x actions)
      elif self.mode is 'discrete':
        categorical=torch.distributions.categorical.Categorical(probs=action_probs)
        #actions = categorical.sample([self.candidates]).permute(1, 0, 2).type(torch.FloatTensor).to(belief.device)
        actions = categorical.sample((self.candidates,)).permute(1, 2, 0).contiguous().view(self.planning_horizon, B * self.candidates, 1).type('torch.FloatTensor').to(belief.device)
      # Sample next states
      #beliefs, states, _, _ = self.transition_model(state, actions, belief)
      #print(state.shape)
      #print(actions.shape)
      #print(belief.shape)
      beliefs, states, _, _ = running_transition_model(state, actions, belief)
      # Calculate expected returns (technically sum of rewards over planning horizon)
      returns = self.reward_model(beliefs.view(-1, H), states.view(-1, Z)).view(self.planning_horizon, -1).sum(dim=0)
      # Re-fit belief to the K best action sequences
      _, topk = returns.reshape(B, self.candidates).topk(self.top_candidates, dim=1, largest=True, sorted=False)
      topk += self.candidates * torch.arange(0, B, dtype=torch.int64, device=topk.device).unsqueeze(dim=1)  # Fix indices for unrolled actions
      if self.mode is 'continuous':
        best_actions = actions[:, topk.view(-1)].reshape(self.planning_horizon, B, self.top_candidates, self.action_size)
        # Update belief with new means and standard deviations
        action_mean, action_std_dev = best_actions.mean(dim=2, keepdim=True), best_actions.std(dim=2, unbiased=False, keepdim=True)
      elif self.mode is 'discrete':
        best_actions = actions[:, topk.view(-1)].reshape(self.planning_horizon, B, self.top_candidates)
        # Update belief with new class probabilities
        for a in range(self.num_actions):
          action_probs[:, :, a] = torch.eq(best_actions, a).type('torch.DoubleTensor').mean(dim=2)

    if self.mode is 'continuous':
      # Return first action mean µ_t
      #print(len(action_mean[0].squeeze(dim=1)))
      #print(action_mean[0].squeeze(dim=1).shape)
      #print(type(action_mean[0].squeeze(dim=1)))
      return action_mean[0].squeeze(dim=1)
    elif self.mode is 'discrete':
      out = action_probs.max(dim=2)
      return out[1].type('torch.FloatTensor')
