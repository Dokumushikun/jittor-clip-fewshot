from torchvision.models import resnet50
import torch
import torch.nn as nn
import os
import vits
import timm

def load_moco(pretrain_path):
    print("=> creating model")
    model = resnet50()
    linear_keyword = 'fc'
    if os.path.isfile(pretrain_path):
        print("=> loading checkpoint '{}'".format(pretrain_path))
        checkpoint = torch.load(pretrain_path, map_location="cpu")
        state_dict = checkpoint["state_dict"]
        for k in list(state_dict.keys()):
            if k.startswith('module.base_encoder') and not k.startswith('module.base_encoder.%s' % linear_keyword):
                # remove prefix
                state_dict[k[len("module.base_encoder."):]] = state_dict[k]
            # delete renamed or unused k

            del state_dict[k]

        msg = model.load_state_dict(state_dict, strict=False)
        print(msg)
        assert set(msg.missing_keys) == {f"{linear_keyword}.weight", f"{linear_keyword}.bias"}

        print("=> loaded pre-trained model '{}'".format(pretrain_path))
    else:
        print("=> no checkpoint found at '{}'".format(pretrain_path))
        raise FileNotFoundError
    model.fc = nn.Identity()
    return model, 2048


def load_moco_vit(pretrain_path):
    print("=> 创建模型")
    model = timm.create_model('vit_base_patch16_224', pretrained=False)
    linear_keyword = 'head'
    if os.path.isfile(pretrain_path):
        print("=> 加载检查点 '{}'".format(pretrain_path))
        checkpoint = torch.load(pretrain_path, map_location="cpu")
        state_dict = checkpoint["state_dict"]
        for k in list(state_dict.keys()):
            if k.startswith('module.base_encoder') and not k.startswith('module.base_encoder.%s' % linear_keyword):
                # 移除前缀
                state_dict[k[len("module.base_encoder."):]] = state_dict[k]
            # 删除重命名或未使用的键
            del state_dict[k]

        msg = model.load_state_dict(state_dict, strict=False)
        print(msg)
        assert set(msg.missing_keys) == {f"{linear_keyword}.weight", f"{linear_keyword}.bias"}

        print("=> 已加载预训练模型 '{}'".format(pretrain_path))
    else:
        print("=> 在 '{}' 找不到检查点".format(pretrain_path))
        raise FileNotFoundError
    model.head = nn.Identity()
    return model, model.embed_dim

if __name__ == "__main__":
    '''    
    aux_model, feat_dim = load_moco("./r-50-1000ep.pth.tar")
    #model = load_moco("r-50-1000ep.pth.tar", ).cuda()
    aux_model = aux_model.cuda()
    #print(model)
    print(aux_model(torch.rand(32,3,224,224).cuda()).shape)
    '''
    from tqdm import tqdm
    aux_model, feat_dim = load_moco_vit("./vit-b-300ep.pth.tar")
    aux_model = aux_model.cuda()
    for i in tqdm(range(1000)):
        a = aux_model(torch.rand(32, 3, 224, 224).cuda()).shape


