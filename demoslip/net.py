
import torch
from torch import nn 



class Net(nn.Module):
    def __init__(self, dropout_temporal_proba, rnn_dropout_proba, batch_norm, mid_feature_dim, mid_depth, cfg):
        
        super().__init__()        

        self.cfg = cfg
        bidirectional = False
        self.D = 2 if bidirectional else 1

        input_feature_dim = cfg.net_extras.maxfreq-cfg.net_extras.minfreq

        #  haptic fusion : proprioceptive signals of 2 joints torques *2 (torque and diff torque)
        if self.cfg.net_extras.conscious_torques == "True": 
            input_feature_dim += 2*2*1

        # conditionning on sensor id, for different pze sensors
        if self.cfg.net_extras.conscious_sensor == "True" :
            input_feature_dim += 12 


        self.rnn = nn.GRU(input_size = input_feature_dim, hidden_size = mid_feature_dim, num_layers= mid_depth, batch_first =False, dropout = rnn_dropout_proba, bidirectional = bidirectional)
        self.bn1 = nn.BatchNorm1d(self.D *mid_feature_dim) if batch_norm else nn.Identity()
        
        self.norm = nn.LayerNorm(self.D * mid_feature_dim)



        self.out_classif_dim = 1
        self.classif = nn.Linear(self.D *mid_feature_dim, self.out_classif_dim )

        self.post_rnn_dropout = nn.Dropout(0.1)

        self.dropout_input= nn.Dropout(p=dropout_temporal_proba)
 
        self.mid_feature_dim = mid_feature_dim
        self.mid_depth = mid_depth






    def forward(self, x):
       
        # BCL
        B,_,L = x.size()

        x = self.dropout_input(x)
        

        x = x.permute((2,0,1)) # BCL to LBC

        ####### NO RNN
        # x = x.reshape(1, L*B, -1)



        if self.training : 
            # h0 = torch.randn([self.D*self.mid_depth, B, self.mid_feature_dim], dtype=torch.float32, device=x.device)
            h0 = torch.zeros([self.D*self.mid_depth, B, self.mid_feature_dim], dtype=torch.float32, device=x.device)
            # h0 = torch.randn([self.mid_depth, L*B, self.mid_feature_dim], dtype=torch.float32, device=x.device)
            # no RNN
        else : 
            h0 = torch.zeros([self.D*self.mid_depth, B, self.mid_feature_dim], dtype=torch.float32, device=x.device)
            # h0 = torch.zeros([self.mid_depth, L*B, self.mid_feature_dim], dtype=torch.float32, device=x.device)
            # no RNN



        x, hn = self.rnn(x, h0)



        # x = x.reshape(L*B, -1) # NO RNN 
        
        # LBC 
        L,B,C = x.size()
        x = x.view(L*B, C) # C is self.D * mid_feature_dim 

        

        x = self.post_rnn_dropout(x)

        x = self.norm(x)       

        x = self.bn1 (x)
        x = self.classif(x) 
        
        x = x.view(L,B, self.out_classif_dim )
        x = x.permute((1,2,0)) # LBC to BCL 

        return x



 