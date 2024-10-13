## PVIC OOD 分支

- 未更改 PVIC 结构
- 直接使用原始的 PVIC 解决 OOD 任务

checkpoints/pvic-detr-r50-hicodet.pth 是 PVIC 作者公开的模型参数

评估指令：

```shell
DETR=base python main.py --world-size 1 \
                         --batch-size 1 \
                         --eval \
                         --resume checkpoints/pvic-detr-r50-hicodet.pth
```


### release v1.0.0

- 直接将 PVIC 输出的 logit 分数作为判断 ID/OOD 的依据，使用如下三种方法得到 OOD 评估结果
    - MSP
    - MaxLogit
    - Energy

评估结果如下：

```txt
eval on HICO-DET testset...
The mAP is 0.3486, rare: 0.3263, none-rare: 0.3553

eval on SWIG-HOI oodset...
ID/OOD: 19498/2194159
MSP: auroc=95.11, fpr=24.75
MaxLogit: auroc=95.13, fpr=24.62
Energy: auroc=94.95, fpr=22.37
```

结果分析：

- 这里的结果要比 ADA-CM+CLIPN 得到的结果好很多，估计是 PVIC 产生了大量的 OOD 人物对，这些人物对具有非常小的 logit


### release v1.1.0

该版本为解决 release v1.0.0 中由于 PVIC 产生了大量低 logit 的人物对，导致 OOD 评测结果不准确的问题。

- 仅考虑与 ground-truth 人物边界框匹配的预测，其余预测结果不参与最终的评测。

评测模型时，模型预测结果保存为二进制文件 `all_ood_results.pkl`，格式如下：

```python
outptu = {
    "label": 形状为 [N] 的 numpy 数组, 表示 ground-truth, 1 表示 ID, 0 表示 OOD,
    "logit": 形状为 [N, 117] 的 numpy 数组, 表示模型输出的动作类别置信度分数
}
```

使用 [eval_ood.py](./eval_ood.py) 脚本读取 `all_ood_results.pkl` 并进行 OOD 检测任务的性能评估

```shell
python eval_ood.py --file all_ood_results.pkl
```

