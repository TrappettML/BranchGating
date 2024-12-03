import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules import CrossEntropyLoss
from ipdb import set_trace

def sample_gumbel(shape, eps=1e-10):
    U = torch.rand(shape)
    U = torch.clamp(U, min=eps, max=1.0 - eps)  # Avoid log(0)
    return -torch.log(-torch.log(U))


class RLCrit(CrossEntropyLoss):
    def __init__(self, ignore_index=-100):
        super(RLCrit, self).__init__(ignore_index=ignore_index)
        # print('Using RLCrit')
        self.baseline = torch.tensor([0]).requires_grad_(False).detach()
        self.count = 0
        self.temperature = 1.0
        self.loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, pred_logits, y_true):
        ''''pred_logits is the output of the model (batch x categories), 
            y_true: is the true labels (batch x 1)'''
        # using the script from simple_pg.py from spinningup at https://github.com/openai/spinningup/blob/master/spinup/examples/pytorch/pg_math/1_simple_pg.py#L44
        # based on policy gradient/Reinforce with baseline using Christian's yŷ defitintion of reward in https://arxiv.org/pdf/2409.03749
        # y_pred_detach = y_pred.clone().detach()
        # pred_probs = self.gumbel_softmax_sample(pred_logits)
        ########## this finally worked ############
        # try this since the next two lines methods didn't work: https://gist.github.com/EderSantana/1ad56b7720af8d706e7f22cbcb8c6d70
        gumbel_noise = sample_gumbel(pred_logits.shape) * 0.01
        noised_logits = pred_logits + gumbel_noise
        # act_probs = F.softmax(pred_logits, dim=1)
        act_probs = F.softmax(noised_logits, dim=1)
        # y_pred = torch.argmax(act_probs, dim=-1)
        logs = torch.log(torch.max(act_probs, dim=1)[0])
        # r = torch.eq(y_pred, y_true).float()
        r = -self.loss(pred_logits, y_true).float()
        baseline = torch.mean(r)
        adv = r - baseline
        loss = -1 * torch.mean(logs * adv)
        ###########################################
        # y_pred_predictions = torch.argmax(F.softmax(y_pred_hat, dim=1), dim=-1)
        # y_hat = torch.distributions.Categorical(logits=y_pred).sample() # noisy sample from the output
        # the distribution is over the true outputs, the y_pred_predicitons are the noisy actions/predictions/samples from the distribution
        # log_probs = torch.distributions.Categorical(logits=y_pred).log_prob(y_pred_predictions) # verify shape
        # reward = noisy predictions x true labels // I got this from Christians paper for the RL learning rule.
        # yyhat = torch.sum(torch.nn.functional.one_hot(y_true, num_classes=10) * y_pred_probs, dim=1)  # batchx10 x batchx10 -> batch
        # # yyhat_detached = yyhat.clone().detach()
        # if yyhat.shape[0] != self.baseline.shape[0] and self.count > 0:
        #     pad_size = self.baseline.shape[0] - yyhat.shape[0]
        #     yyhat= torch.nn.functional.pad(yyhat, (0, pad_size), value=0) # add padding to match baseline
        # td_error = 0.5 * (yyhat - self.baseline) ** 2 # TD error
        # yyhat_detached = yyhat.clone().detach()
        # self.baseline = (self.baseline * self.count + yyhat_detached) / (self.count + 1) # add padding
        # self.baseline.detach()
        # self.count += 1 # count for running average
        # loss = -(log_probs * td_error).mean() # check if convert to vector
        # loss = td_error.mean()
        # set_trace()
        return loss


