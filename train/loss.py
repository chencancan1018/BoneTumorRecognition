import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class SoftFocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.


    """

    def __init__(self, class_num, alpha=None, gamma=2, size_average=True, smoothing=0.1, w_list=[1,1,1], is_mixup=False):
        super(SoftFocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.w_list = w_list
        self.is_mixup = is_mixup
    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs, dim=1)
        inputs = inputs.float()
        ids = targets.view(-1, 1)
        logprobs = nn.functional.log_softmax(inputs, dim=-1)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        class_mask.scatter_(1, ids.data.long(), 1.)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1).long()]

        probs = (P * class_mask).sum(1).view(-1, 1)
        mask_tmp = class_mask
        if self.is_mixup:
            mask_tmp = targets
        nll_loss = self.w_list[0] * (torch.eq(ids.data.long(), 0)) * (logprobs * mask_tmp).sum(1).view(-1, 1)
        for i in range(1, len(self.w_list)):
            w_tmp = self.w_list[i]
            nll_loss = nll_loss + w_tmp * (torch.eq(ids.data.long(), i)) * (logprobs * mask_tmp).sum(1).view(-1, 1)
        smooth_loss = logprobs.mean(dim=-1).view(-1, 1)

        batch_loss = -alpha * (torch.pow((1-probs), self.gamma))* (self.confidence * nll_loss + self.smoothing * smooth_loss)

        if self.size_average:
            loss = torch.mean(batch_loss)
        else:
            loss = batch_loss.sum()
        # print('softfocalloss')
        return loss