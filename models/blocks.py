import torch
import torch.nn as nn
import brevitas.nn as qnn

#NOTE: BraggNN does not divide by sqrt(d) like in traditional trasnformers
class ConvAttn(torch.nn.Module):
    def __init__(self, in_channels = 16, hidden_channels = 8, norm = None, act = None):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.Wq = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, stride=1)
        self.Wk = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, stride=1)
        self.Wv = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, stride=1)
        self.softmax = nn.Softmax(dim=-1)
        self.proj = nn.Conv2d(hidden_channels, in_channels, kernel_size=1, stride=1)
        self.act = act

    def forward(self, x):
        b, c, h, w = x.size()
        #q shape (b, seq, embed_dim) -> permute -> (b, embed_dim, seq)
        query = self.Wq(x).view(b, self.hidden_channels, -1).permute(0, 2, 1)
        key = self.Wk(x).view(b, self.hidden_channels, -1)
        value = self.Wv(x).view(b, self.hidden_channels, -1).permute(0, 2, 1)

        z = self.softmax(torch.matmul(query,key)) 
        z = torch.matmul(z, value).permute(0, 2, 1).view(b, self.hidden_channels, h, w)
        
        x = x + self.proj(z)
        if self.act is not None:
            x = self.act(x)
        return x

class ConvBlock(torch.nn.Module):
    def __init__(self, channels, kernels, acts, norms, img_size):
        super().__init__()
        self.layers = []
        for i in range(len(kernels)):
            self.layers.append( nn.Conv2d(channels[i], channels[i+1], 
                                          kernel_size=kernels[i], stride=1, 
                                          padding = 0 )) #padding = (kernels[i] - 1) // 2)
            if kernels[i] == 3: img_size -= 2
            if norms[i] == 'batch':
                self.layers.append( nn.BatchNorm2d(channels[i+1]) )
            elif norms[i] == 'layer':
                self.layers.append( nn.LayerNorm([channels[i+1], img_size, img_size]) ) #DEPRECATED
            if acts[i] != None:
                self.layers.append(acts[i])
        self.layers = nn.Sequential(*self.layers)
      
    def forward(self, x):
        return self.layers(x)

#TODO: Add variable length with pass through layers ie lambda x: x
class MLP(torch.nn.Module):
    def __init__(self, widths, acts, norms):
        super().__init__()

        self.layers = []
        for i in range(len(acts)): 
            self.layers.append( nn.Linear(widths[i], widths[i+1]) )
            if norms[i] == 'batch':
                self.layers.append( nn.BatchNorm1d(widths[i+1]) )
            elif norms[i] == 'layer':
                self.layers.append( nn.LayerNorm(widths[i+1]) ) #DEPRECATED
            #elif None, skip
            if acts[i] != None:
                self.layers.append( acts[i] )
        self.layers = nn.Sequential(*self.layers)
        

    def forward(self, x):
        return self.layers(x)

def sample_ConvAttn(trial, prefix):
    channel_space = (1,2,4,8,16,32)
    act_space = (nn.ReLU(), nn.LeakyReLU(negative_slope=0.01), None)
    hidden_channels = channel_space[trial.suggest_int(prefix + '_hiddenchannel', 0, len(channel_space) - 1)]
    act = act_space[trial.suggest_categorical(prefix + '_act', [k for k in range(len(act_space))])]
    return hidden_channels, act


    widths = [in_dim] + [width_space[trial.suggest_int(prefix + '_width_' + str(i), 0, len(width_space) - 1)] for i in range(num_layers-1)] + [out_dim]
    acts = [act_space[trial.suggest_categorical(prefix + '_acts_' + str(i), (0,1,2))] for i in range(num_layers)]
    norms = [trial.suggest_categorical(prefix + '_norms_' + str(i), norm_space) for i in range(num_layers)]

    return widths, acts, norms

def sample_MLP(trial, in_dim, out_dim, prefix, num_layers=3):
    width_space = (8, 16, 32, 64, 128)
    act_space = (nn.ReLU(), nn.LeakyReLU(negative_slope=2**-7))
    norm_space = (None, 'batch', 'layer')

    widths = [in_dim] + [width_space[trial.suggest_int(prefix + '_width_' + str(i), 0, len(width_space) - 1)] for i in range(num_layers-1)] + [out_dim]
    acts = [act_space[trial.suggest_categorical(prefix + '_acts_' + str(i), (0,1))] for i in range(num_layers-1)] + [None]
    norms = [trial.suggest_categorical(prefix + '_norms_' + str(i), norm_space) for i in range(num_layers)]

    return widths, acts, norms

def sample_ConvBlock(trial, in_channels, prefix, num_layers = 2):
    #Search space to sample from
    channel_space = []
    kernel_space = [1,3,5]
    act_space = [nn.ReLU(), nn.LeakyReLU(), nn.GeLU(), lambda x: x]
    norm_space = [None, 'layer', 'batch']

    channels = [in_channels] + [channel_space[ trial.suggest_int(prefix + '_channels_' + str(i), 0, len(channel_space) - 1) ]
                                    for i in range(num_layers)] #Picks an integer an index of channel_space for easier sampling
    kernels = [trial.suggest_categorical(prefix + '_kernels_' + str(i), kernel_space) for i in range(num_layers)]
    norms = [trial.suggest_categorical(prefix + '_norms_' + str(i), norm_space) for i in range(num_layers)]
    acts = [trial.suggest_categorical(prefix + '_acts_' + str(i), act_space) for i in range(num_layers)]

    return channels, kernels, acts, norms




class CandidateArchitecture(torch.nn.Module):
    def __init__(self, Blocks, MLP, hidden_channels, input_channels = 1):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, hidden_channels, kernel_size=(3, 3), stride=(1, 1)) #Initial Projection Layer
        self.Blocks = Blocks
        self.MLP = MLP

    def forward(self, x):
        x = self.conv(x)
        x = self.Blocks(x)
        x = torch.flatten(x, 1)
        x = self.MLP(x)
        return x
    
class DeepSetsArchitecture(torch.nn.Module):
    def __init__(self, phi, rho, aggregator):
        super().__init__()
        self.phi = phi
        self.rho = rho
        self.aggregator = aggregator

    def forward(self, x):
        x = self.phi(x)
        x = self.aggregator(x) #torch.mean(x, dim=1)
        #x = torch.squeeze(x, dim=1)
        x = self.rho(x)
        return x
    
class SequenceNorm1D(torch.nn.Module):
    def __init__(self, seq_len): 
        super().__init__()
        self.norm = nn.BatchNorm1d(seq_len)

    def forward(self, x):
        if len(x.size()) == 3:
            indices = (0,2,1)
        elif len(x.size()) == 2:
            indices = (0,1) #dont permute.
        else:
            print(f'Input x of size {x.size()} is not supported!')

        x = torch.permute(x, indices)
        x = self.norm(x)
        x = torch.permute(x, indices)
        return x

class Phi(torch.nn.Module):
    def __init__(self, widths, acts, norms):
        super().__init__()

        self.layers = []
        for i in range(len(acts)): 
            self.layers.append( nn.Linear(widths[i], widths[i+1]) )
            if norms[i] == 'batch':
                self.layers.append( nn.BatchNorm1d(8) )
            elif norms[i] == 'sequence':
                self.layers.append( SequenceNorm1D(widths[i+1]) )
            if acts[i] != None:
                self.layers.append( acts[i] )
        self.layers = nn.Sequential(*self.layers)
        

    def forward(self, x):
        return self.layers(x)
    
class Rho(torch.nn.Module):
    def __init__(self, widths, acts, norms):
        super().__init__()

        self.layers = []
        for i in range(len(acts)): 
            self.layers.append( nn.Linear(widths[i], widths[i+1]) )
            if norms[i] == 'batch':
                self.layers.append( nn.BatchNorm1d(widths[i+1]) )
            elif norms[i] == 'sequence':
                self.layers.append( SequenceNorm1D(widths[i+1]) )

            if acts[i] != None:
                self.layers.append( acts[i] )
        self.layers = nn.Sequential(*self.layers)
        

    def forward(self, x):
        return self.layers(x)
    
class ConvPhi(torch.nn.Module):
    def __init__(self, widths, acts, norms):
        super().__init__()

        self.layers = []
        for i in range(len(acts)): 
            self.layers.append( nn.Conv1d(widths[i], widths[i+1], kernel_size=1,stride=1) )
            if norms[i] == 'batch':
                self.layers.append( nn.BatchNorm1d(widths[i+1]) )
            elif norms[i] == 'sequence':
                self.layers.append( SequenceNorm1D(widths[i+1]) )
            if acts[i] != None:
                self.layers.append( acts[i] )
        self.layers = nn.Sequential(*self.layers)
        

    def forward(self, x):
        return self.layers(x)
    
class QAT_Phi(torch.nn.Module):
    def __init__(self, widths, acts, norms, bit_width=8):
        super().__init__()
        self.layers = nn.Sequential()
        self.quant_inp = qnn.QuantIdentity(bit_width=bit_width, return_quant_tensor=True)
        for i in range(len(acts)):
            linear_layer = qnn.QuantLinear(widths[i], widths[i+1], bias=True, weight_bit_width=bit_width)
            self.layers.add_module(f'linear_{i}', linear_layer)
            if norms[i] == 'batch':
                self.layers.add_module(f'norm_{i}', nn.BatchNorm1d(8))
            if acts[i] is not None:
                act_layer = acts[i] if isinstance(acts[i], nn.Module) else qnn.QuantReLU(bit_width=bit_width, return_quant_tensor=True)
                self.layers.add_module(f'act_{i}', act_layer)
                
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.quant_inp(x)
            x = layer(x)
        return x
    
class QAT_ConvPhi(torch.nn.Module):
    def __init__(self, widths, acts, norms, bit_width=8):
        super().__init__()
        self.layers = nn.Sequential()
        self.quant_inp = qnn.QuantIdentity(bit_width=bit_width, return_quant_tensor=True)
        for i in range(len(acts)):
            linear_layer = qnn.QuantConv1d(widths[i], widths[i+1], kernel_size=1, stride=1, weight_bit_width=bit_width)
            self.layers.add_module(f'linear_{i}', linear_layer)
            if norms[i] == 'batch':
                self.layers.add_module(f'norm_{i}', nn.BatchNorm1d(widths[i+1]))
            if acts[i] is not None:
                act_layer = acts[i] if isinstance(acts[i], nn.Module) else qnn.QuantReLU(bit_width=bit_width, return_quant_tensor=True)
                self.layers.add_module(f'act_{i}', act_layer)
                
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.quant_inp(x)
            x = layer(x)
        return x
    
class QAT_Rho(torch.nn.Module):
    def __init__(self, widths, acts, norms, bit_width=8):
        super().__init__()
        self.layers = nn.Sequential()
        self.quant_inp = qnn.QuantIdentity(bit_width=bit_width, return_quant_tensor=True)
        for i in range(len(acts)):
            linear_layer = qnn.QuantLinear(widths[i], widths[i+1], bias=True, weight_bit_width=bit_width)
            self.layers.add_module(f'linear_{i}', linear_layer)
            if norms[i] == 'batch':
                self.layers.add_module(f'norm_{i}', nn.BatchNorm1d(widths[i+1]))
            elif norms[i] == 'layer':
                self.layers.add_module(f'norm_{i}', nn.LayerNorm(widths[i+1]))
            if acts[i] is not None:
                act_layer = acts[i] if isinstance(acts[i], nn.Module) else qnn.QuantReLU(bit_width=bit_width, return_quant_tensor=True)
                self.layers.add_module(f'act_{i}', act_layer)
                
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.quant_inp(x)
            x = layer(x)
        return x

class Identity(torch.nn.Module):
    def __init__(self):
        super(self).__init__()
    
    def forward(self, x):
        return x

class QAT_ConvAttn(torch.nn.Module):
    def __init__(self, in_channels = 16, hidden_channels = 8, norm = None, act = None, bit_width=8):
        super().__init__()
        self.quant_inp = qnn.QuantIdentity(bit_width=bit_width, return_quant_tensor=True)
        self.hidden_channels = hidden_channels
        self.Wq = qnn.QuantConv2d(in_channels, hidden_channels, kernel_size=1, stride=1, weight_bit_width=bit_width)
        self.Wk = qnn.QuantConv2d(in_channels, hidden_channels, kernel_size=1, stride=1, weight_bit_width=bit_width)
        self.Wv = qnn.QuantConv2d(in_channels, hidden_channels, kernel_size=1, stride=1, weight_bit_width=bit_width)
        self.softmax = nn.Softmax(dim=-1) # kept in floating point
        self.proj = qnn.QuantConv2d(hidden_channels, in_channels, kernel_size=1, stride=1, weight_bit_width=bit_width)
        self.act = act if act is not None else qnn.QuantIdentity(bit_width=bit_width, return_quant_tensor=True)
        # Initialize the QuantIdentity layer for softmax output
        #self.quant_identity = qnn.QuantIdentity(bit_width=bit_width, return_quant_tensor=True)
    def forward(self, x):
        x = self.quant_inp(x)
        #print("Entering ConvAttn - Input shape:", x.shape)
        b, c, h, w = x.size()
        query = self.Wq(x).view(b, self.hidden_channels, -1).permute(0, 2, 1)
        key = self.Wk(x).view(b, self.hidden_channels, -1)
        value = self.Wv(x).view(b, self.hidden_channels, -1).permute(0, 2, 1)
        z = self.softmax(torch.matmul(query,key))
        z = torch.matmul(z, value).permute(0, 2, 1).view(b, self.hidden_channels, h, w)
        z = self.quant_inp(z) #z = self.quant_identity(z)
        x = x + self.proj(z)
        if self.act is not None:
            x = self.act(x)
        return x
   
class QAT_ConvBlock(nn.Module):
    def __init__(self, channels, kernels, acts, norms, img_size, bit_width=8):
        super().__init__()
        self.layers = nn.Sequential()
        self.quant_inp = qnn.QuantIdentity(bit_width=bit_width, return_quant_tensor=True)
        for i in range(len(kernels)):
            conv = qnn.QuantConv2d(channels[i], channels[i+1], kernel_size=kernels[i], stride=1, padding=0, weight_bit_width=bit_width)
            self.layers.append(conv)
            if kernels[i] == 3: img_size -= 2
            if norms[i] == 'batch':
                norm_layer = nn.BatchNorm2d(channels[i+1])
                self.layers.append(norm_layer)
            elif norms[i] == 'layer': #DEPRECATED
                norm_layer = nn.LayerNorm([channels[i+1], img_size, img_size])
                self.layers.append(norm_layer)
            if norms[i] is not None:
                self.layers.append(qnn.QuantIdentity(bit_width=bit_width, return_quant_tensor=True))
            if acts[i] is not None:
                act_layer = acts[i] if isinstance(acts[i], nn.Module) else qnn.QuantIdentity(bit_width=bit_width, return_quant_tensor=True)
                self.layers.append(act_layer)

    def forward(self, x):
        #print("entering block")
        for layer in self.layers:
            x = self.quant_inp(x)
            x = layer(x)
        #print("exiting block")
        return x
    
class QAT_MLP(torch.nn.Module):
    def __init__(self, widths, acts, norms, bit_width=8):
        super().__init__()
        self.layers = nn.Sequential()
        self.quant_inp = qnn.QuantIdentity(bit_width=bit_width, return_quant_tensor=True)
        for i in range(len(acts)):
            linear_layer = qnn.QuantLinear(widths[i], widths[i+1], bias=True, weight_bit_width=bit_width)
            self.layers.add_module(f'linear_{i}', linear_layer)
            if norms[i] == 'batch':
                self.layers.add_module(f'norm_{i}', nn.BatchNorm1d(widths[i+1]))
            elif norms[i] == 'layer':
                self.layers.add_module(f'norm_{i}', nn.LayerNorm(widths[i+1]))
            if acts[i] is not None:
                act_layer = acts[i] if isinstance(acts[i], nn.Module) else qnn.QuantReLU(bit_width=bit_width, return_quant_tensor=True)
                self.layers.add_module(f'act_{i}', act_layer)
                
    def forward(self, x):
        x = self.quant_inp(x)
        for i, layer in enumerate(self.layers):
            x = self.quant_inp(x)
            x = layer(x)
        return x
    
class QAT_CandidateArchitecture(torch.nn.Module):
    def __init__(self, Blocks, MLP, hidden_channels, input_channels=1, bit_width=8):
        super().__init__()
        self.quant_inp = qnn.QuantIdentity(bit_width=bit_width, return_quant_tensor=True) # initial layer to initialize quantization
        self.conv = qnn.QuantConv2d(input_channels, hidden_channels, kernel_size=(3, 3), # initial projection layer quantized
                                    stride=(1, 1), weight_bit_width=bit_width)
        self.Blocks = Blocks
        self.MLP = MLP
    def forward(self, x):
        x = self.quant_inp(x)
        x = self.conv(x)
        x = self.Blocks(x)
        x = torch.flatten(x, 1)
        x = self.MLP(x)
        return x
    

    

#Leaving this in for later, currently not used/working.
class TransformerBlock(torch.nn.Module):
    def __init__(self, input_size = (4, 16, 9, 9)):
        super().__init__()
        embed_dim = input_size[-2] * input_size[-1]
               
        #Sample parameters for LinearAttention, rewrite with optuna samplers.
        trial = {
            'num_heads' : 1, #pick from [1,2,4,6,8]
            'norm' : nn.BatchNorm1d(input_size[1]), #pick from [nn.BatchNorm1d(input_size[1]), nn.LayerNorm((4,16,81))]
            'hidden_dim_scale' : 2, #pick from [1,2,4]
            'dropout' : .1, #float
            'bias' : True, #[True, False]
            'num_layers': 1, #[1,2,3]
        }

        self.layers = [ nn.TransformerEncoderLayer(d_model=embed_dim,
                                  nhead=trial['num_heads'],
                                  dim_feedforward=embed_dim * trial['hidden_dim_scale'],
                                  dropout=trial['dropout'],
                                  bias=trial['bias']) for i in range(trial['num_layers'])]

    def forward(self, x):
        x = torch.flatten(x, 2) #now (4, 16, 81)
        for l in self.layers:
          x = l(x)

#Leaving this in for later, currently not used/working.
class SkipBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        #Sample hyperparameters
        self.input_size = (4,16,11,11)
        self.channels = [16,4,4,16]
        self.kernels = [1,3,1] #[1,3,5]
        self.act = [nn.ReLU(), nn.ReLU(), lambda x: x, nn.ReLU()] #pick from [nn.ReLU(), nn.LeakyReLU(), nn.GeLU(), lambda x: x]
        self.norm = ['batch', 'batch', 'batch'] #pick from ['identity', 'layer', 'batch']

        self.layers = []
        for i in range(len(self.kernels)):
            self.layers.append( nn.Conv2d(self.channels[i], self.channels[i+1], 
                                          kernel_size=self.kernels[i], stride=1, 
                                          padding = (self.kernels[i] - 1) // 2) )
            if self.norm[i] == 'batch':
                self.layers.append( nn.BatchNorm2d(self.channels[i+1]) )
            elif self.norm[i] == 'layer':
                self.layers.append( nn.LayerNorm(self.input_size) )

            self.layers.append( self.act[i] )
      

    def forward(self, x):
        z = x 
        for l in self.layers:
            z = l(z)
        x += z
        x = self.act[-1](x) #Activation after skip
        return x