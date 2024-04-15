import torch
import torch.nn as nn
import numpy as np

def LinearBlock(n_in, n_out, activation='relu'):
    # do not work with ModuleList here either.
    linear = nn.Linear(n_in, n_out)
    if activation == 'relu':
        activ = nn.ReLU()
    elif activation == 'tanh':
        activ = nn.Tanh()
    else:
        raise ValueError('Define activation function')
    block = nn.Sequential(linear, activ)
    return block
    

class MLP(nn.Module):
    
    def __init__(self, in_dim, out_dim, hidden_num, hidden_dim, inner_act_fun, do_skip):
        super(MLP, self).__init__()

        dim_layers = [in_dim] + hidden_num * [hidden_dim] + [out_dim]
        assert hidden_dim > hidden_num
        self.do_skip = do_skip
        layers = []
        num_layers = len(dim_layers)
        
        blocks = []
        for l in range(num_layers-2):
            blocks.append(LinearBlock(dim_layers[l], dim_layers[l+1], activation=inner_act_fun))
        blocks.append(nn.Linear(dim_layers[-2], dim_layers[-1]))
        self.network = nn.Sequential(*blocks)
    
    def forward(self, xin):
        if self.do_skip:
            for l, layer in enumerate(self.network):
                if l == 0:
                    x = layer(xin)
                elif l == len(self.network) - 1:
                   return layer(x)
                else:
                    x = layer(x) + x if l % 3 == 2 else layer(x)
        else:            
            return self.network(xin)


class RFF(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_num, hidden_dim, hidden_act_fun,
                    feature_dim, feature_scale, do_skip):
        super(RFF, self).__init__()

        self.net = MLP(2*feature_dim, out_dim, hidden_num, hidden_dim, inner_act_fun=hidden_act_fun, do_skip=do_skip)
        self.B = torch.nn.Parameter(torch.randn((in_dim, feature_dim)) * 2 * np.pi * feature_scale[0], requires_grad=False)

    def positional_encoding(self, x):
        return torch.cat((torch.cos(x @ self.B), torch.sin(x @ self.B)), dim=-1)

    def forward(self, xin):
        emb_x = self.positional_encoding(xin)
        return self.net(emb_x)
    

class SSNSirenLayer(nn.Module):
    def __init__(self, in_f, out_f, w=[1.], lam=[1.], is_first=False, is_last=False):
        super().__init__()
    
        if is_first:
            assert len(w) == 1
            assert len(lam) == in_f

        self.in_f = in_f
        self.lam = nn.Parameter(torch.Tensor(lam), requires_grad=False)
        self.w = nn.Parameter(torch.Tensor(w), requires_grad=False)
        self.linear = nn.Linear(in_f, out_f)
        self.is_first = is_first
        self.is_last = is_last

    def forward(self, x):
        if self.is_first:
            x = self.linear(x*self.lam) * self.w
        else:
            x = self.linear(x)
        return x if self.is_last else torch.sin(x)

class SirenLayer(nn.Module):
    def __init__(self, in_f, out_f, w0=[1], is_first=False, is_last=False):
        super().__init__()
        
        if is_first:
            assert in_f == len(w0)

        self.in_f = in_f
        self.w0 = w0
        self.linear = nn.Linear(in_f, out_f)
        # self.s = nn.Parameter(torch.randn(out_f,1), requires_grad=True)
        self.is_first = is_first
        self.is_last = is_last
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            b = 1 / self.in_f if self.is_first else np.sqrt(6 / self.in_f)
            if self.is_first:
                for k in range(self.in_f):
                    self.linear.weight.data[:,k] *= self.w0[k]
            else:
                # self.s.uniform_(-b,b)
                self.linear.weight.uniform_(-b, b)
                # self.linear.weight.uniform_(-1, 1)
            self.linear.bias.uniform_(-3,3)

    def forward(self, x, cond_freq=None, cond_phase=None):
        x = self.linear(x)
        if not cond_freq is None:
            freq = cond_freq #.unsqueeze(1).expand_as(x)
            x = freq * x
        if not cond_phase is None:
            phase_shift = cond_phase #unsqueeze(1).expand_as(x)
            x = x + phase_shift
        return x if self.is_last else torch.sin(x)
    

class RealGaborLayer(nn.Module):
    '''
        Implicit representation with Gabor nonlinearity
        
        Inputs;
            in_features: Input features
            out_features; Output features
            bias: if True, enable bias for the linear operation
            is_first: Legacy SIREN parameter
            omega_0: Legacy SIREN parameter
            omega: Frequency of Gabor sinusoid term
            scale: Scaling of Gabor Gaussian term
    '''
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega0=10.0, sigma0=10.0):
        
        super().__init__()
        self.omega_0 = omega0
        self.scale_0 = sigma0
        self.is_first = is_first
        self.is_last = False

        self.in_features = in_features
        
        self.freqs = nn.Linear(in_features, out_features, bias=bias)
        self.scale = nn.Linear(in_features, out_features, bias=bias)
        
    def forward(self, input):
        omega = self.omega_0 * self.freqs(input)
        scale = self.scale(input) * self.scale_0
        
        return torch.cos(omega)*torch.exp(-(scale**2))


class WIRE(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_num, hidden_dim, feature_scales, do_skip):
        super(WIRE, self).__init__()
        self.hidden_dim = hidden_dim
        model_dim = [in_dim] + hidden_num * [hidden_dim] + [out_dim]
        assert len(feature_scales) == in_dim
        first_layer = RealGaborLayer(model_dim[0], model_dim[1], 
                                     omega0=feature_scales[0], sigma0=feature_scales[0], is_first=True)
        other_layers = []
        for dim0, dim1 in zip(model_dim[1:-2], model_dim[2:-1]):
            other_layers.append(RealGaborLayer(dim0, dim1, omega0=1., sigma0=1))
        final_layer = nn.Linear(model_dim[-2], model_dim[-1])
        final_layer.is_last = True
        final_layer.is_first = False
        self.siren = nn.Sequential(first_layer, *other_layers, final_layer)
        self.do_skip = do_skip

    def forward(self, xin):
        if self.do_skip:
            for l, layer in enumerate(self.siren):
                if layer.is_first:
                    x = layer(xin)
                elif layer.is_last:
                    out = layer(x)
                else:
                    x = layer(x) + x if l % 3 == 2 else layer(x)
        else:
            out = self.siren(xin)
        return out

class SSN(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_num, hidden_dim, w, feature_scales, do_skip):
        super(SSN, self).__init__()
        self.hidden_dim = hidden_dim
        model_dim = [in_dim] + hidden_num * [hidden_dim] + [out_dim]
        assert len(feature_scales) == in_dim
        first_layer = SSNSirenLayer(model_dim[0], model_dim[1], w=w, lam=feature_scales, is_first=True)
        other_layers = []
        for dim0, dim1 in zip(model_dim[1:-2], model_dim[2:-1]):
            other_layers.append(SSNSirenLayer(dim0, dim1, w=[1.], lam=[1.]*dim0))
            # other_layers.append(nn.LayerNorm(dim1))
        final_layer = nn.Linear(model_dim[-2], model_dim[-1])
        final_layer.is_last = True
        final_layer.is_first = False
        # final_layer = SirenLayer(model_dim[-2], model_dim[-1], is_last=True, w0=[1.])
        # final_layer = SirenLayer(model_dim[-2], model_dim[-1], is_last=True, w0=[1.])
        self.siren = nn.Sequential(first_layer, *other_layers, final_layer)
        self.do_skip = do_skip

    def forward(self, xin):
        if self.do_skip:
            for l, layer in enumerate(self.siren):
                if layer.is_first:
                    x = layer(xin)
                elif layer.is_last:
                    out = layer(x)
                else:
                    x = layer(x) + x if l % 2 == 1 else layer(x)
        else:
            out = self.siren(xin)
        return out

class SIREN(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_num, hidden_dim, feature_scales, do_skip):
        super(SIREN, self).__init__()
        self.hidden_dim = hidden_dim
        model_dim = [in_dim] + hidden_num * [hidden_dim] + [out_dim]
        assert len(feature_scales) == in_dim
        first_layer = SirenLayer(model_dim[0], model_dim[1], w0=feature_scales, is_first=True)
        other_layers = []
        for dim0, dim1 in zip(model_dim[1:-2], model_dim[2:-1]):
            other_layers.append(SirenLayer(dim0, dim1, w0=[1.]))
            # other_layers.append(nn.LayerNorm(dim1))
        final_layer = nn.Linear(model_dim[-2], model_dim[-1])
        final_layer.is_last = True
        final_layer.is_first = False
        # final_layer = SirenLayer(model_dim[-2], model_dim[-1], is_last=True, w0=[1.])
        # final_layer = SirenLayer(model_dim[-2], model_dim[-1], is_last=True, w0=[1.])
        self.siren = nn.Sequential(first_layer, *other_layers, final_layer)
        self.do_skip = do_skip

    def forward(self, xin):
        if self.do_skip:
            for l, layer in enumerate(self.siren):
                if layer.is_first:
                    x = layer(xin)
                elif layer.is_last:
                    out = layer(x)
                else:
                    x = layer(x) + x if l % 2 == 1 else layer(x)
        else:
            out = self.siren(xin)
        return out

class GaborFilter(nn.Module):
    def __init__(self, in_dim, out_dim, weight_scales, alpha=2, beta=1.0):
        super(GaborFilter, self).__init__()

        self.mu = nn.Parameter(torch.rand((out_dim, in_dim)) * 2 - 1)
        self.gamma = nn.Parameter(torch.distributions.gamma.Gamma(alpha, beta).sample((out_dim, )))
        self.linear = torch.nn.Linear(in_dim, out_dim)

        # Init weights
        for k in range(in_dim):
            self.linear.weight.data[:,k] *=  weight_scales[k] * torch.sqrt(self.gamma)
        self.linear.bias.data.uniform_(-np.pi, np.pi)

    def forward(self, x):
        norm = (x ** 2).sum(dim=1).unsqueeze(-1) + (self.mu ** 2).sum(dim=1).unsqueeze(0) - 2 * x @ self.mu.T
        return torch.exp(- self.gamma.unsqueeze(0) / 2. * norm) * torch.sin(self.linear(x))


class FourierFilter(nn.Module):
    def __init__(self, in_dim, out_dim, weight_scales, bias=True):

        super(FourierFilter, self).__init__()

        assert len(weight_scales) == in_dim
        
        self.linear = torch.nn.Linear(in_dim, out_dim, bias=bias)
        for k in range(in_dim): 
            self.linear.weight.data[:,k] *= weight_scales[k]

        self.linear.bias.data.uniform_(-np.pi, np.pi)
        
    def forward(self, x):
        return torch.sin(self.linear(x))


def mfn_weights_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-np.sqrt(6/num_input), np.sqrt(6/num_input))


class MFN(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_num, hidden_dim, feature_scales=[128.]):
        super(MFN, self).__init__()

        filter_fun = FourierFilter
        # if filter == 'Fourier':
        #     filter_fun = FourierFilter
        # elif filter == 'Gabor':
        #     filter_fun = GaborFilter
            
        quantization_interval = 2 * np.pi
        assert len(feature_scales) == in_dim
        input_scales = [round((np.pi * freq / (hidden_num + 1))
                / quantization_interval) * quantization_interval for freq in feature_scales]

        self.K = hidden_num
        self.filters = nn.ModuleList(
            [filter_fun(in_dim, hidden_dim, input_scales) for _ in range(self.K)])
        self.linear = nn.ModuleList(
            [torch.nn.Linear(hidden_dim, hidden_dim) for _ in range(self.K - 1)] + [torch.nn.Linear(hidden_dim, out_dim)])
        self.linear.apply(mfn_weights_init)

    def forward(self, x):
        # Recursion - Equation 3
        zi = self.filters[0](x)  # Eq 3.a
        for i in range(self.K - 1):
            zi = self.linear[i](zi) * self.filters[i + 1](x)  # Eq 3.b

        x = self.linear[-1](zi)  # Eq 3.c
        return x
