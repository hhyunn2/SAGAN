import torch
import torch.nn as nn

class Attn_Module(nn.Module):
    def __init__(self, input_dim): 
        super(Attn_Module, self).__init__()
        self.conv_f = nn.Conv2d(in_channels=int(input_dim), out_channels=int(input_dim)//8, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_g = nn.Conv2d(in_channels=int(input_dim), out_channels=int(input_dim)//8, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_h = nn.Conv2d(in_channels=int(input_dim), out_channels=int(input_dim)//8, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_v = nn.Conv2d(in_channels=int(input_dim)//8, out_channels=int(input_dim), kernel_size=1, stride=1, padding=0, bias=False)
        

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        bs, c, w, h = x.shape
        
        out_f = self.conv_f(x).view(bs, -1, w*h)  # (bs, c/8, w*h)
        out_g = self.conv_g(x).view(bs, -1, w*h)  # (bs, c/8, w*h)
        out_h = self.conv_h(x).view(bs, -1, w*h)  # (bs, c/8, w*h)
        out_f = out_f.permute(0, 2, 1)  # (bs, w*h, c/8)
       
        attn_map = torch.bmm(out_f, out_g) # (bs, w*h, w*h)
        attn_map = F.softmax(attn_map, 1)
        
        self_attn_map = torch.bmm(out_h, attn_map) # (bs, c/8, w*h)
      
        self_attn_map = self_attn_map.view(bs,-1, w, h) # (bs, c/8, w, h)   
        
        self_attn_map = self.conv_v(self_attn_map) # (bs, c, w, h)
        
        out= self_attn_map * self.gamma + x

        return out, self_attn_map
