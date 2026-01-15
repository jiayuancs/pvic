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



## 说明

对  [hicodet/hicodet.py](https://github.com/fredzzhang/hicodet/tree/047bbdc0dcd7d5caf0d90634b465001d92a4df9d) 的修改如下：

```git
diff --git a/hicodet.py b/hicodet.py
index 014374f..941f692 100644
--- a/hicodet.py
+++ b/hicodet.py
@@ -62,6 +62,7 @@ class HICODet(ImageDataset):
             and its target as entry and returns a transformed version.
     """
     def __init__(self, root: str, anno_file: str,
+            object_cls_num: int = 80, hoi_cls_num: int = 600, verb_cls_num: int = 117,
             transform: Optional[Callable] = None,
             target_transform: Optional[Callable] = None,
             transforms: Optional[Callable] = None) -> None:
@@ -69,9 +70,9 @@ class HICODet(ImageDataset):
         with open(anno_file, 'r') as f:
             anno = json.load(f)
 
-        self.num_object_cls = 80
-        self.num_interation_cls = 600
-        self.num_action_cls = 117
+        self.num_object_cls = object_cls_num
+        self.num_interation_cls = hoi_cls_num
+        self.num_action_cls = verb_cls_num
         self._anno_file = anno_file
 
         # Load annotations
@@ -311,5 +312,5 @@ class HICODet(ImageDataset):
         self._empty_idx = f['empty']
         self._objects = f['objects']
         self._verbs = f['verbs']
-        self._rare = f['rare']
-        self._non_rare = f['non_rare']
+        if 'rare' in f.keys(): self._rare = f['rare']
+        if 'non_rare' in f.keys(): self._non_rare = f['non_rare']
```

