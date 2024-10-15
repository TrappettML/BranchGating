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

    def forward(self, input, target):
        # using the script from simple_pg.py from spinningup at https://github.com/openai/spinningup/blob/master/spinup/examples/pytorch/pg_math/1_simple_pg.py#L44
        # based on policy gradient/Reinforce with baseline using Christian's yŷ defitintion of reward in https://arxiv.org/pdf/2409.03749
        # input_detach = input.clone().detach()
        input_norm = torch.nn.functional.normalize(input, p=2, dim=-1)
        input_hat = input_norm + torch.randn_like(input_norm)
        input_probs = F.softmax(input_hat, dim=1) # Batch x 10 
        # input_predictions = torch.argmax(F.softmax(input_hat, dim=1), dim=-1)
        # y_hat = torch.distributions.Categorical(logits=input).sample() # noisy sample from the output
        # the distribution is over the true outputs, the input_predicitons are the noisy actions/predictions/samples from the distribution
        # log_probs = torch.distributions.Categorical(logits=input).log_prob(input_predictions) # verify shape
        # reward = noisy predictions x true labels // I got this from Christians paper for the RL learning rule.
        yyhat = torch.sum(torch.nn.functional.one_hot(target, num_classes=10) * input_probs, dim=1)  # batchx10 x batchx10 -> batch
        # yyhat_detached = yyhat.clone().detach()
        if yyhat.shape[0] != self.baseline.shape[0] and self.count > 0:
            pad_size = self.baseline.shape[0] - yyhat.shape[0]
            yyhat= torch.nn.functional.pad(yyhat, (0, 0, 0, pad_size), value=0) # add padding to match baseline
        td_error = yyhat - self.baseline # TD error
        yyhat_detached = yyhat.clone().detach()
        self.baseline = (self.baseline * self.count + yyhat_detached) / (self.count + 1) # add padding
        self.baseline.detach()
        self.count += 1 # count for running average
        # loss = -(log_probs * td_error).mean() # check if convert to vector
        loss = td_error.mean()
        # set_trace()
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
