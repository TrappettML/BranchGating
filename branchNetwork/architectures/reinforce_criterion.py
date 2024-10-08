import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules import CrossEntropyLoss


class RLCrit(CrossEntropyLoss):
    def __init__(self, ignore_index=-100):
        super(RLCrit, self).__init__(ignore_index=ignore_index)
        # print('Using RLCrit')
        self.baseline = 0
        self.count = 0

    def forward(self, input, target):
        # using the script from simple_pg.py from spinningup at https://github.com/openai/spinningup/blob/master/spinup/examples/pytorch/pg_math/1_simple_pg.py#L44
        # based on policy gradient/Reinforce with baseline using Christian's yŷ defitintion of reward in https://arxiv.org/pdf/2409.03749
        input_detach = input.clone().detach()
        input_norm = torch.nn.functional.normalize(input_detach, p=2, dim=-1)
        input_hat = input_norm + torch.randn_like(input_norm)
        input_predictions = torch.argmax(F.softmax(input_hat, dim=1), dim=-1)
        # the distribution is over the true outputs, the input_predicitons are the noisy actions/predictions/samples from the distribution
        log_probs = torch.distributions.Categorical(logits=input).log_prob(input_predictions) 
        # reward = noisy predictions x true labels // I got this from Christians paper for the RL learning rule.
        y_y_hat = torch.sum(torch.nn.functional.one_hot(target, num_classes=10) * input_hat, dim=1)  
        self.baseline = (self.baseline * self.count + y_y_hat) / (self.count + 1) 
        self.count += 1 # count for running average
        td_error = y_y_hat - self.baseline # TD error
        loss = -(log_probs * td_error).mean()
        return loss

        # with torch.no_grad(): # is this necessary?
        #     input_hat = input + torch.randn_like(input) # add noise to output of network
        #     input_hat_probs = torch.softmax(input_hat, dim=-1) # convert to probabilities for 
        #     y_y_hat = torch.nn.functional.one_hot(target) * input_hat_probs
        #     self.baseline = (self.baseline * self.count + y_y_hat) / (self.count + 1)
        #     td_error = y_y_hat - self.baseline
        # loss = super().forward(input_hat, td_error)
        # self.count += 1
        # return loss
