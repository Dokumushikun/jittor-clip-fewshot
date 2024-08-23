import torch
import jittor as jt

# 加载 torch 的权重
#torch_weights = torch.load('vit-b-300ep.pth.tar')
torch_weights = torch.load('checkpoint_0844.pth.tar')
# 创建一个字典来存储 jittor 的权重
jittor_weights = {}

# 将 torch 的权重转换为 jittor 的权重
for k, v in torch_weights.items():
    if isinstance(v, torch.Tensor):
        jittor_weights[k] = jt.array(v.float().cpu().numpy())
    else:
        jittor_weights[k] = v

# 保存 jittor 的权重
jt.save(jittor_weights, 'r-50-1000ep.pkl')

print("转换完成，并保存为 'r-50-1000ep.pkl'")
