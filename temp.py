# temp.py
import torch

from config import detcfg


def inspect_weights(weight_path):
    try:
        state_dict = torch.load(weight_path, map_location='cpu')
        print("Model keys in checkpoint:")
        for key in state_dict.keys():
            print(f"- {key}")
            if isinstance(state_dict[key], dict):
                print("  Subkeys:")
                for subkey in state_dict[key].keys():
                    print(f"  - {subkey}")

        # 检查MTCNN专用键
        mtcnn_keys = ['pnet', 'rnet', 'onet']
        missing = [k for k in mtcnn_keys if k not in state_dict]
        if missing:
            print(f"\n警告：缺少MTCNN专用键 {missing} [1][4][7]")

    except Exception as e:
        print(f"加载失败：{str(e)} [4][7]")


if __name__ == "__main__":
    inspect_weights(detcfg.rnet_weight)
