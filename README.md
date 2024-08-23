# jittor挑战赛-赛道一B榜 
##shi山警告。比赛时间有点赶，代码没整理好，审核能跑通就行，其他同学暂时不要参考了，等过几天再整理一下。
## 一、当前基于 CLIP 的小样本学习概况

本队伍旨在最大化模型微调性能。对于基于 CLIP 的小样本学习，目前主流方法主要采用以下几种策略：

1. **头部微调**：主要包括但不限于 `Prompt Tuning`（文本提示调优和视觉提示调优）。
2. **模型内部微调**：选择部分参数或层进行微调，如低秩近似（LoRA）和全量微调。
3. **尾部微调**：主要是视觉分类头的微调。

## 二、本组方案

1. **微调策略**：为了最大化利用上述微调手段，我们采用了 `Prompt Tuning` + `LoRA` + 分类头的方式进行优化。具体而言：
   - **头部**：仅调整 `Text Prompt`（由于视觉提示调优与 LoRA 存在冲突，因此不展开）。
   - **模型内部**：采用 `LoRA` 方式进行微调。
   - **分类头**：引入通道重加权和 `LP++` 的方式进行优化。

2. **学习节奏控制**：由于各个部分的学习节奏不一致，具体情况如下：
   - `Prompt Tuning` 会在 10 个 `epoch` 内拟合完成。
   - `VPT` 和 `LoRA` 等方法需要 25-30 个 `epoch`。
   - 分类头（如线性探测）则需要数百个 `epoch`。

   因此，我们需要控制各个部件的学习进度。我们隔离了分类头，让其利用上游特征单独训练。对于 `Prompt Tuning`，我们维护了一个队列，每 N 个 `epoch`，将学习到的 `Prompt` 加入队列，同时弹出队尾的 `Prompt`。队列中的 `Prompt` 与 LLM 生成的手工 `Prompt` 进行加权平均。

3. **测试策略**：测试时采用 `TTA`（Test-Time Augmentation）。具体而言，我们采用 `MTA`方法对图像进行随机裁剪（比例范围为 0.5-1）N 次，根据重要性评估求解符合分布的特征。

4. **ood**：对于base类，我们尽可能提高精度，但是对于new类，我们考虑，不采用promptSRC的方式，因为网络学习到的上限即zeroshot clip的信息，所以利用正则化去约束clip的学习反而会造成模型对于base类的精度损失，故采用ood检测的方式，检测分布外的类别。我们发现，clip对于类别描述和类别数量是较为敏感的，类别数量越多，描述词越细粒度，则ood检测能力越强，而实验发现，仅采用文本增强和扩大类别数的方式，即可具备非常好的ood性能。我们在测试时直接利用zeroshot clip（文本做增强、类别数扩大）将新类别剔除。其他观点暂不阐述。

## 三、训练和测试

### 下载模型权重

官方vit-b-32版本clip的“ViT-B-32.pkl”放在主目录。
请下载模型权重（链接：https://pan.baidu.com/s/1vyJHgwidVQWuG-U869awfQ 密码：3a36），并按照以下要求放置：

- `res1000epoch.pkl` 放置在主目录。
- 其他 `.pkl` 权重文件放置在 `test_pkl` 文件夹中。

### 数据集放置

- 测试集 `TestSetB` 放在 `Dataset` 文件夹下。
- 训练集 `TrainSet` 放置在 `Dataset/TrainSet/TrainSet` 目录下。

**注意**：训练集路径为 `Dataset/TrainSet/TrainSet`，即 `TrainSet` 文件夹嵌套于 `TrainSet` 目录下。

### 1. 测试

运行以下命令进行测试：

```bash
pip install -r requirements.txt
python test.py
```
### 2. 训练
```bash
bash train.sh
```
qq 917846323
## 引用
```
@inproceedings{zanella2024low,
  title={Low-Rank Few-Shot Adaptation of Vision-Language Models},
  author={Zanella, Maxime and Ben Ayed, Ismail},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={1593--1603},
  year={2024}
}

@inproceedings{zanella2024test,
  title={On the test-time zero-shot generalization of vision-language models: Do we really need prompt learning?},
  author={Zanella, Maxime and Ben Ayed, Ismail},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={23783--23793},
  year={2024}
}

@inproceedings{zhang2024dept,
  title={Dept: Decoupled prompt tuning},
  author={Zhang, Ji and Wu, Shihan and Gao, Lianli and Shen, Heng Tao and Song, Jingkuan},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={12924--12933},
  year={2024}
}

@inproceedings{lp24,
    title={LP++: A Surprisingly Strong Linear Probe for Few-Shot CLIP},
    author={Yunshi Huang and Fereshteh Shakeri and Jose Dolz and Malik Boudiaf and Houda Bahig and Ismail Ben Ayed},
    booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2024}
    }
```
