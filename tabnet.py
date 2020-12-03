import torch
import torch.nn as nn

class TabNet_Decoder(nn.Module):
    class DecoderStep(nn.Module):
        def __init__(inp_dim, out_dim, shared, n_ind, vbs, device):
            super.__init__()

            self.fea_tran = FeatureTransformer(inp_dim, out_dim, shared, n_ind, vbs, device)
            self.fc = nn.Linear(out_dim, out_dim).to(device)
        
        def forward(self, x):
            x = self.fea_tran(x)
            return self.fc(x)

    def __init__(inp_dim, out_dim, n_shared, n_ind, vbs, n_steps, device):
        """
        TabNet decoder that is used in pre-training
        """
        super().__init__()
        if n_shared > 0:
            self.shared = nn.ModuleList()
            self.shared.append(nn.Linear(inp_dim, 2 * (n_d + n_a)))
            for x in range(n_shared - 1):
                self.shared.append(nn.Linear(n_d + n_a, 2 * (n_d + n_a))) # preset the linear function we will use
        else:
            self.shared=None
        self.n_steps = n_steps
        self.steps = nn.ModuleList()
        for x in range(n_steps):
            self.steps.append(DecoderStep(inp_dim, out_dim, self.shared, n_ind, vbs, device))

    def forward(self, x):
        out = torch.zeros(x.size(0), out_dim).to(x.device)
        for step in self.steps:
            out += step(x)
        return out



class TabNet(nn.Module):
    def __init__(self, inp_dim, final_out_dim, n_d=64, n_a=64, n_shared=2, n_ind=2, n_steps=5, relax=1.2, vbs=128, device):
    	"""
		TabNet AKA the original encoder

		Args:
			n_d: dimension of the features used to calculate the final results
			n_a: dimension of the features input to the attention transformer of the next step
			n_shared: numbr of shared steps in feature transfomer(optional)
			n_ind: number of independent steps in feature transformer
			n_steps: number of steps of pass through tabbet
			relax coefficient:
			virtual batch size:
    	"""
        super().__init__()

        # set the number of shared step in feature transformer
        if n_shared > 0:
            self.shared = nn.ModuleList()
            self.shared.append(nn.Linear(inp_dim, 2 * (n_d + n_a)))
            for x in range(n_shared - 1):
                self.shared.append(nn.Linear(n_d + n_a, 2 * (n_d + n_a))) # preset the linear function we will use
        else:
            self.shared=None

        self.first_step = FeatureTransformer(inp_dim, n_d + n_a, self.shared, n_ind, device) 
        self.steps = nn.ModuleList()
        for x in range(n_steps-1):
            self.steps.append(DecisionStep(inp_dim, n_d, n_a, self.shared, n_ind, relax, vbs, device))
        self.fc = nn.Linear(n_d, final_out_dim)
        self.bn = nn.BatchNorm1d(inp_dim)
        self.n_d = n_d
    
    def forward(self, x):
        x = self.bn(x)
        x_a = self.first_step(x)[:, self.n_d:]
        sparse_loss = torch.zeros(1).to(x.device)
        out = torch.zeros(x.size(0), self.n_d).to(x.device)
        priors = torch.ones(x.shape).to(x.device) # all priors were set to be one intialially

        for step in self.steps:
            x_te, l = step(x, x_a, priors)
            out += F.relu(x_te[:, :self.n_d]) #split the feautre from feat_transformer
            x_a = x_te[:, self.n_d:]
            sparse_loss += l
        return self.fc(out), sparse_loss


class GBN(nn.Module):
	"""
	Ghost Batch Normalization
	an efficient way of doing batch normalization

	Args:
		vbs: virtual batch size
	"""
    def __init__(self, inp, vbs=128, momentum=0.01):
        super().__init__()
        self.bn = nn.BatchNorm1d(inp, momentum=momentum)
        self.vbs = vbs
    
    def forward(self, x):
        chunk = torch.chunk(x, x.size(0)//self.vbs,0)
        res = [self.bn(y) for y in chunk]
        return torch.cat(res, 0)


class GLU(nn.Module):
	"""
	GLU block that extracts only the most essential information

	Args:
		vbs: virtual batch size
	"""
    def __init__(self, inp_dim, out_dim, fc=None, vbs=128):
        super().__init__()
        if fc:
            self.fc = fc
        else:
            self.fc = nn.Linear(inp_dim, out_dim*2)
        self.bn = GBN(out_dim * 2, vbs=vbs) 
        self.od = out_dim
   
    def forward(self, x):
        x = self.bn(self.fc(x))
        return torch.mul(x[:, :self.od], torch.sigmoid(x[:, self.od:]))


class AttentionTransformer(nn.Module):
	"""
	Args:
		relax: relax coefficient. The greater it is, we can
		use the same features more. When it is set to 1
		we can use every feature only once
	"""
    def __init__(self, d_a, inp_dim, relax, vbs=128):
        super().__init__()
        self.fc = nn.Linear(d_a, inp_dim)
        self.bn = GBN(out_dim,vbs=vbs)
        self.r = relax

    #a:feature from previous decision step
    def forward(self, a, priors): 
        a = self.bn(self.fc(a)) 
        mask = SparseMax(a * priors) 
        priors = priors * (self.r - mask)  #updating the prior
        return mask


class FeatureTransformer(nn.Module):
    def __init__(self, inp_dim, out_dim, shared, n_ind, vbs=128, device):
        super().__init__()
        first = True
        self.shared = nn.ModuleList()
        if shared:
            self.shared.append(GLU(inp_dim, out_dim, shared[0], vbs=vbs))
            first= False    
            for fc in shared[1:]:
                self.shared.append(GLU(out_dim, out_dim, fc, vbs=vbs))
        else:
            self.shared = None
        self.independ = nn.ModuleList()
        if first:
            self.independ.append(GLU(inp,out_dim,vbs=vbs))
        for x in range(first, n_ind):
            self.independ.append(GLU(out_dim,out_dim,vbs=vbs))
        self.scale = torch.sqrt(torch.tensor([.5], device=device))
    
    def forward(self, x):
        if self.shared:
            x = self.shared[0](x)
            for glu in self.shared[1:]:
                x = torch.add(x, glu(x))
                x = x * self.scale

        for glu in self.independ:
            x = torch.add(x, glu(x))
            x = x * self.scale
        return x



class DecisionStep(nn.Module):
	"""
	One step for the TabNet 
	"""
    def __init__(self, inp_dim, n_d, n_a, shared, n_ind, relax, vbs=128, device):
        super().__init__()
        self.atten_tran =  AttentionTransformer(n_a, inp_dim, relax,vbs)
        self.fea_tran = FeatureTransformer(inp_dim, n_d + n_a, shared, n_ind, vbs, device)

    def forward(self, x, a, priors):
        mask = self.atten_tran(a, priors)
        sparse_loss = SparseMax(mask)
        x = self.fea_tran(x * mask)
        return x ,sparse_loss

def SparseMax(mask):
    return ((-1)*mask*torch.log(mask+1e-10)).mean()