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
MSP: auroc=95.11, fpr=24.75
MaxLogit: auroc=95.13, fpr=24.62
Energy: auroc=94.95, fpr=22.37
```

结果分析：

- 这里的结果要比 ADA-CM+CLIPN 得到的结果好很多，估计是 PVIC 产生了大量的 OOD 人物对，这些人物对具有非常小的 logit

