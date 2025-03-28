import torch
from collections import OrderedDict

def rename_state_dict_keys(old_checkpoint_path, new_checkpoint_path):
    # 加载原始参数文件
    state_dict = torch.load(old_checkpoint_path)
    new_state_dict = OrderedDict()
    keys_to_delete = [k for k in state_dict if 'mlp' in k.lower()]  # 支持大小写不敏感匹配
    keys_to_delete_conv = [k for k in state_dict if 'conv_5' in k.lower()]
    target_keys=["Gmask.se.fc.0.weight", "Gmask.se.fc.2.weight"]
    # 删除目标参数
    for key in keys_to_delete:
        del state_dict[key]
    for key in keys_to_delete_conv:
        del state_dict[key]
    for key in target_keys:
       if key in state_dict:
           del state_dict[key]
           print(f"已删除参数：{key}")
       else:
           print(f"⚠️ 参数 {key} 不存在")
    # 遍历所有参数键值对
    for key, value in state_dict.items():
        # 执行字符串替换
        new_key = key.replace("vis", "hsi").replace("lwir", "lidar").replace("sru","sr").replace("fbc","fhfl").replace("Gate","Route")
        new_state_dict[new_key] = value

    # 保存修改后的参数文件
    torch.save(new_state_dict, new_checkpoint_path)
    print(f"参数名替换完成，新文件已保存至：{new_checkpoint_path}")


# 使用示例
rename_state_dict_keys(
    old_checkpoint_path="cls_model\Yancheng.pth",
    new_checkpoint_path="cls_model\Yancheng_new.pth"
)