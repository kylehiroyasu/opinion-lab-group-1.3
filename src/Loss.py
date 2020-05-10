# Taken and adapted from https://gitbub.com/GT-RIPL/L2C
import torch.nn as nn

class KLDivergence(nn.Module):
    # Calculate KL-Divergence

    eps = 1e-7
        
    def forward(self, predict, target):
       assert predict.ndimension()==2,'Input dimension must be 2'
       target = target.detach()

       # KL(T||I) = \sum T(logT-logI)
       predict += KLDivergence.eps
       target += KLDivergence.eps
       logI = predict.log()
       logT = target.log()
       TlogTdI = target * (logT - logI)
       kld = TlogTdI.sum(1)
       return kld

class KCL(nn.Module):
    # KLD-based Clustering Loss (KCL)

    def __init__(self, margin=2.0):
        super(KCL,self).__init__()
        self.kld = KLDivergence()
        self.hingeloss = nn.HingeEmbeddingLoss(margin)

    def forward(self, prob1, prob2, simi):
        # simi: 1->similar; -1->dissimilar; 0->unknown(ignore)
        assert len(prob1)==len(prob2)==len(simi), 'Wrong input size:{0},{1},{2}'.format(str(len(prob1)),str(len(prob2)),str(len(simi)))

        kld = self.kld(prob1,prob2)
        output = self.hingeloss(kld,simi)
        return output


class MCL(nn.Module):
    # Meta Classification Likelihood (MCL)

    eps = 1e-7 # Avoid calculating log(0). Use the small value of float16.
        
    def forward(self, prob1, prob2, simi):
        # simi: 1->similar; -1->dissimilar; 0->unknown(ignore)
        assert len(prob1)==len(prob2)==len(simi), 'Wrong input size:{0},{1},{2}'.format(str(len(prob1)),str(len(prob2)),str(len(simi)))

        P = prob1.mul_(prob2)
        P = P.sum(1)
        P.mul_(simi).add_(simi.eq(-1).type_as(P))
        neglogP = -P.add_(MCL.eps).log_()
        return neglogP.mean()


def PairEnum(x, mask=None):
    # Enumerate all pairs of feature in x
    # This method is used by the learner to iterate through all possible pairs
    # of similar/disimilar samples
    assert x.ndimension() == 2, 'Input dimension must be 2'
    x1 = x.repeat(x.size(0),1)
    x2 = x.repeat(1,x.size(0)).view(-1,x.size(1))
    if mask is not None:
        xmask = mask.view(-1,1).repeat(1,x.size(1))
        #dim 0: #sample, dim 1:#feature 
        x1 = x1[xmask].view(-1,x.size(1))
        x2 = x2[xmask].view(-1,x.size(1))
    return x1,x2


def Class2Simi(target, mode='hinge', mask=None):
    # Convert class label to pairwise similarity
    n = target.nelement()
    assert (n-target.ndimension()+1) == n,'Dimension of Label is not right'
    expand1 = target.view(-1,1).expand(n,n)
    expand2 = target.view(1,-1).expand(n,n)
    out = expand1 - expand2    
    out[out!=0] = -1 #dissimilar pair: label=-1
    out[out==0] = 1 #Similar pair: label=1
    if mode=='cls':
        out[out==-1] = 0 #dissimilar pair: label=0
    if mode=='hinge':
        out = out.float() #hingeloss require float type
    if mask is None:
        out = out.view(-1)
    else:
        mask = mask.detach()
        out = out[mask]
    return out