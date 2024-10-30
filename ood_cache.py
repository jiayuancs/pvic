import pickle
import torch
from torch import Tensor
from copy import deepcopy
from pocket.utils import BoxPairAssociation
from tqdm import tqdm


class OODDetResultTemplate:
    """保存OOD检测模型的预测结果"""

    def __init__(self):
        self.results = []

    def append(self,
               image_id,
               boxes: Tensor,
               human_box_idx: Tensor,
               object_box_idx: Tensor,
               hoi_logits: Tensor,
               gt_is_id: bool,
               gt_human_boxes: Tensor,
               gt_object_boxes: Tensor,
               ood_scores: dict=dict(),
            ):
        """
        添加预测结果
        Args:
            boxes: 该图片中所有边界框的坐标，格式为[(x1,y1,x2,y2), ...]
            human_box_idx: human_box_idx[i]表示第i个 human-object pair 中，人的边界框在 boxes 中的索引
            object_box_idx: object_box_idx[i]表示第i个 human-object pair 中，物的边界框在 boxes 中的索引
            hoi_logits: 表示第 i 个 human-object pair 属于各个类别的置信度分数
            gt_is_id: ground-truth标注的是ID还是OOD人物对, true表示ID, false表示OOD
            gt_human_boxes: 真实边界框坐标格式为[(x1,y1,x2,y2), ...]
            gt_object_boxes: 真实边界框坐标格式为[(x1,y1,x2,y2), ...]
            ood_scores: 表示第 i 个 human-object pair 属于 ID 类别的置信度分数，可添加多种，例如：
                - ctw: list[float]
                - atd: list[float]
                - MSP: list[float]
                - ...
        """
        self.results.append({
            'boxes': deepcopy(boxes.numpy()),
            'human_idx': deepcopy(human_box_idx.numpy()),
            'object_idx': deepcopy(object_box_idx.numpy()),
            'hoi_logits': deepcopy(hoi_logits.numpy()),
            'ood_scores': deepcopy(ood_scores),
            'ground_truth': {
                "image_id": image_id,
                "is_in_distribution": gt_is_id,
                "human_boxes": deepcopy(gt_human_boxes.numpy()),
                "object_boxes": deepcopy(gt_object_boxes.numpy())
            },
        })

    def save(self, file_path):
        """将结果保存到file_path文件中"""
        with open(file_path, mode='wb') as fd:
            pickle.dump(self.results, fd)


class OODDetResultEvaluator:

    def __init__(self, result_path):
        with open(result_path, "rb") as f:
            origin_data = pickle.load(f)
        assert origin_data
        self.print_metainfo(origin_data)

        self.results = origin_data["id_part"] + origin_data["ood_part"]
        self.hoi_performance = origin_data["hoi_performance"]
        self.metadata = origin_data["metadata"]

        ##### 临时变量 #####
        # _id_score[method][i]表示使用method方法得到的第i个人物对属于ID类别的概率
        self._id_score = None
        # ho_match_label[i]表示第i个人物对是否能够与ground-truth人物对匹配
        # - 0 表示不匹配
        # - 1 表示匹配且ground-truth是ID
        # - 2 表示匹配且ground-truth是OOD
        self._ho_match_label = None
        # 混淆矩阵dict
        self._confusion_matrix = None
        # ground-truth 中的人物对总数
        self._gt_ho_cnt = None
        ##### 参数 #####
        self._ood_method = None
        self._iou_threshold = None
        self._ood_threshold = None

    @staticmethod
    def print_metainfo(origin_data):
        """输出该OOD检测结果的基本信息"""
        metadata = origin_data["metadata"]
        for key, value in metadata.items():
            print(f"{key}: {value}")
        id_part_len = len(origin_data["id_part"])
        ood_part_len = len(origin_data["ood_part"])
        print(f"image num(ID/OOD): {id_part_len}/{ood_part_len}")

    def match_ho_box(self, threshold: float):
        """
        将模型输出的人物对边界框与ground-truth进行匹配。
        
        Args:
            - pred_data: list, 在 ID 数据集或 OOD 数据集上的预测结果
            - threshold: float, 指定IoU阈值, 小于等于该阈值则被认为是不匹配的
        
        Return:
            - match_list: list[Tensor], match_list[i][j]表示第i张图片的第j个人物对是否能够与ground-truth人物对匹配,
                        - 0 表示不匹配
                        - 1 表示匹配且ground-truth是ID
                        - 2 表示匹配且ground-truth是OOD
            - fn_id_cnt: int, 实际为ID人物对, 但没有被检测出的数量
            - fn_ood_cnt: int, 实际为OOD人物对, 但没有被检测出的数量
            
        Note:
            匹配流程(针对每一张图片), 与HOI Detection中的匹配流程相同, 唯一不同之处在于不考虑人物对的HOI类别:
            首先针对每一个预测人物对：
                1. 计算检测到的人物对边界框与ground-truth的IoU
                2. 过滤掉IoU小于等于threshold的检测结果
                3. 如果存在IoU大于threshold的人物对, 则将这个检测人物对分配给具有最大IoU的那个真实人物对
            然后针对每个真实人物对：
                1. 如果存在与之匹配的预测人物对, 则仅保留置信度最大的那个
            
            保留下的的预测人物对被认为是匹配的, 后续将参与OOD Detection的评测
            被过滤掉的预测人物对和没有匹配的真实人物对将用于计算FN_ID和FP_ID
        """
        print(f"matching human-object boxes with ground-truth")
        self._iou_threshold = threshold
        pred_data = self.results
        associate = BoxPairAssociation(min_iou=threshold)

        # match_list[i][j]表示第i张图片的第j个人物对是否能够与ground-truth人物对匹配
        match_list = []
        # id_score_dict[method][i]表示使用method方法得到的第i个人物对属于ID类别的概率
        id_score_dict = {key:[] for key in pred_data[0]["ood_scores"].keys()}
        fn_id_cnt = 0    # 实际为ID，没有检测出
        fn_ood_cnt = 0   # 实际为OOD，没有被检测出
        gt_ho_cnt = 0    # ground-truth 中人物对总数
        for image_pred in tqdm(pred_data):
            # prediction
            boxes = image_pred["boxes"]
            human_boxes = torch.from_numpy(boxes[image_pred["human_idx"]])
            object_boxes = torch.from_numpy(boxes[image_pred["object_idx"]])
            hoi_logits = torch.from_numpy(image_pred["hoi_logits"])
            score = torch.max(hoi_logits, dim=-1)[0]

            # ID score
            for method, id_score in image_pred["ood_scores"].items():
                id_score_dict[method].append(torch.from_numpy(id_score))

            # ground-truth
            target = image_pred["ground_truth"]
            gt_human_boxes = torch.from_numpy(target["human_boxes"])
            gt_object_boxes = torch.from_numpy(target["object_boxes"])
            gt_is_id = target["is_in_distribution"]

            # match
            label = associate(
                        (gt_human_boxes, gt_object_boxes),
                        (human_boxes, object_boxes),
                        score.view(-1)
                    ).int()
            
            gt_ho_cnt += gt_human_boxes.shape[0]
            # 实际存在，但没有被检测出来的人物对数量
            gt_missing_cnt = gt_human_boxes.shape[0] - label.sum()
            if gt_is_id:
                fn_id_cnt += gt_missing_cnt
            else:
                fn_ood_cnt += gt_missing_cnt
                label = label * 2
            match_list.append(label)

        # 保存临界供其他方法使用
        self._gt_ho_cnt = gt_ho_cnt
        self._ho_match_label = torch.cat(match_list)
        self._confusion_matrix = {
            "fn_id": fn_id_cnt,
            "fn_ood": fn_ood_cnt
        }
        self._id_score = {}
        for key, value in id_score_dict.items():
            self._id_score[key] = torch.cat(value)

    def eval_with_threshold(self, threshold: float, ood_method: str):
        """
        指定OOD阈值, 小于等于threshold被认为是OOD类别, 大于threshold则被认为是ID类别

        Args:
            - threshold: float
            - ood_method: str, 指定要使用的OOD分数计算方法, 例如ctw、atd、msp
        Return:
            - 3 x 3 的混淆矩阵
        """
        assert self._ho_match_label is not None, "run match_ho_box() first"
        self._ood_threshold = threshold
        self._ood_method = ood_method
        
        id_score = self._id_score[ood_method]
        label = id_score > threshold
        
        mat = self._calc_confusion_matrix(
            id_label=label,
            gt_id_label=self._ho_match_label == 1,
            gt_ood_label=self._ho_match_label == 2,
        )
        self._confusion_matrix.update(mat)

    @property
    def confusion_matrix(self):
        """获取混淆矩阵"""
        return self._confusion_matrix

    def print_confusion_matrix(self):
        print("---- Confusion Matrix ----")
        print(
            "Args:\n"
            f" - IoU Threshold={self._iou_threshold}\n"
            f" - OOD Threshold={self._ood_threshold}\n"
            f" - OOD Mathod={self._ood_method}"
        )
        print("Performance:")
        for key, value in self._confusion_matrix.items():
            print(f" - {key:6} : {value:6}")
        print("--------------------------")
        self._check_confusion_matrix()

    @staticmethod
    def _calc_confusion_matrix(id_label, gt_id_label, gt_ood_label):
        """
        计算OOD Detection的混淆矩阵
        Args:
            - id_label: id_label[i]表示模型是否将第i个人物对分类为ID类别
            - gt_id_label: gt_id_label[i]=True表示第i个人物对实际上是ID类别, False表示不匹配或OOD类别
            - gt_ood_label: gt_ood_label[i]=True表示第i个人物对实际上是OOD类别, False表示不匹配或ID类别
        Return: dict(tp, fp, fn, tn, fp_id, fp_ood)
        """
        assert id_label.dtype == torch.bool
        assert gt_id_label.dtype == torch.bool
        assert gt_ood_label.dtype == torch.bool
        assert torch.all(torch.logical_and(gt_id_label, gt_ood_label) == False)

        ood_label = torch.logical_not(id_label)
        tp = torch.logical_and(id_label, gt_id_label).sum()
        fp = torch.logical_and(id_label, gt_ood_label).sum()
        fn = torch.logical_and(ood_label, gt_id_label).sum()
        tn = torch.logical_and(ood_label, gt_ood_label).sum()

        # 没有与ground-truth匹配的人物对
        gt_mismatch_label = torch.logical_not(torch.logical_or(gt_id_label, gt_ood_label))
        fp_id = torch.logical_and(gt_mismatch_label, id_label).sum()
        fp_ood = torch.logical_and(gt_mismatch_label, ood_label).sum()

        mat = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "fp_id": fp_id,
            "fp_ood": fp_ood
        }
        return mat

    def _check_confusion_matrix(self):
        """检查混淆矩阵中的数值是否正确"""
        pred_cnt = len(self._ho_match_label)
        mat = self._confusion_matrix
        match_cnt = mat["tp"] + mat["fp"] + mat["fn"] + mat["tn"]
        assert match_cnt + mat["fp_id"] + mat["fp_ood"] == pred_cnt
        assert match_cnt + mat["fn_id"] + mat["fn_ood"] == self._gt_ho_cnt


if __name__ == "__main__":
    path = "checkpoints/hico-ood-v2.2.4-1e3-1e4-10/ood_pred_56448_12.pkl"
    evaluator = OODDetResultEvaluator(result_path=path)
    evaluator.match_ho_box(threshold=0.5)

    evaluator.eval_with_threshold(threshold=0.1, ood_method="atd")
    evaluator.print_confusion_matrix()

    evaluator.eval_with_threshold(threshold=0.2, ood_method="atd")
    evaluator.print_confusion_matrix()

    evaluator.eval_with_threshold(threshold=0.3, ood_method="atd")
    evaluator.print_confusion_matrix()

    evaluator.eval_with_threshold(threshold=0.4, ood_method="atd")
    evaluator.print_confusion_matrix()

    evaluator.eval_with_threshold(threshold=0.5, ood_method="atd")
    evaluator.print_confusion_matrix()
    
    print("Done")
