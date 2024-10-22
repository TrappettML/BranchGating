import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules import CrossEntropyLoss
from ipdb import set_trace


class RLCrit(CrossEntropyLoss):
    def __init__(self, ignore_index=-100):
        super(RLCrit, self).__init__(ignore_index=ignore_index)
        # print('Using RLCrit')
        self.baseline = torch.tensor([0]).requires_grad_(False).detach()
        self.count = 0
        self.temperature = 1.0

    def forward(self, input, target):
        # using the script from simple_pg.py from spinningup at https://github.com/openai/spinningup/blob/master/spinup/examples/pytorch/pg_math/1_simple_pg.py#L44
        # based on policy gradient/Reinforce with baseline using Christian's yŷ defitintion of reward in https://arxiv.org/pdf/2409.03749
        # input_detach = input.clone().detach()
        input_probs = self.gumbel_softmax_sample(input)
        # input_predictions = torch.argmax(F.softmax(input_hat, dim=1), dim=-1)
        # y_hat = torch.distributions.Categorical(logits=input).sample() # noisy sample from the output
        # the distribution is over the true outputs, the input_predicitons are the noisy actions/predictions/samples from the distribution
        # log_probs = torch.distributions.Categorical(logits=input).log_prob(input_predictions) # verify shape
        # reward = noisy predictions x true labels // I got this from Christians paper for the RL learning rule.
        yyhat = torch.sum(torch.nn.functional.one_hot(target, num_classes=10) * input_probs, dim=1)  # batchx10 x batchx10 -> batch
        # yyhat_detached = yyhat.clone().detach()
        if yyhat.shape[0] != self.baseline.shape[0] and self.count > 0:
            pad_size = self.baseline.shape[0] - yyhat.shape[0]
            yyhat= torch.nn.functional.pad(yyhat, (0, pad_size), value=0) # add padding to match baseline
        td_error = 0.5 * (yyhat - self.baseline) ** 2 # TD error
        yyhat_detached = yyhat.clone().detach()
        self.baseline = (self.baseline * self.count + yyhat_detached) / (self.count + 1) # add padding
        self.baseline.detach()
        self.count += 1 # count for running average
        # loss = -(log_probs * td_error).mean() # check if convert to vector
        loss = td_error.mean()
        # set_trace()
        return loss

    def gumbel_softmax_sample(self, logits):
        # Generate Gumbel noise
        gumbel_noise = -torch.empty_like(logits).exponential_().log()
        # Add Gumbel noise to the logits
        y = logits + gumbel_noise
        # Apply softmax with temperature
        return F.softmax(y / self.temperature, dim=-1)
