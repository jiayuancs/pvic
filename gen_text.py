"""
使用 pattern 获取动词的现在进行时形式，构造句子
pattern 的官方文档：https://github.com/clips/pattern/wiki/pattern-en

注意：pattern 仅支持 python2.7 或 python3.6
"""
from pattern.en import conjugate, referenced


def get_verbing(verb):
    """获取动词的ing形式"""
    return conjugate(verb, "part")

def get_a_or_an_noun(noun):
    """获取名词的带冠词形式"""
    return referenced(noun)


def get_hoi_text(action_text, object_text):
    action_text = action_text.replace("_", " ")
    object_text = object_text.replace("_", " ")

    # 获取动词和名词对应的形式
    verbing_text = get_verbing(action_text)
    a_or_an_object_text = get_a_or_an_noun(object_text)

    # 当前
    no_interaction_flag = "interaction" in action_text and "no" in action_text
    if no_interaction_flag:
        normal_text = f"A photo of a person is not interacting with {a_or_an_object_text}."
        not_text = f"A photo of a person is interacting with {a_or_an_object_text}."
    else:
        normal_text = f"A photo of a person is {verbing_text} {a_or_an_object_text}."
        not_text = f"A photo of a person is not {verbing_text} {a_or_an_object_text}."

    return normal_text, not_text


if __name__ == "__main__":
    import os
    import json
    from tqdm import tqdm

    from hicodet_text import HICO_DET_HOI_LABEL_DICT

    interactions = list(HICO_DET_HOI_LABEL_DICT.values())

    all_hoi_text_dict = {}
    for hoi_name in tqdm(interactions):
        assert len(hoi_name.split(" ")) == 2
        action_text, object_text = hoi_name.split(" ")
        normal_text, not_text = get_hoi_text(action_text, object_text)

        key = f"{action_text}+{object_text}"
        value = {
            "normal": normal_text,
            "not": not_text
        }
        all_hoi_text_dict[key] = value


    write_text = "\nHOI_TEXT = "
    write_text += json.dumps(all_hoi_text_dict, indent=4)
    write_text += "\n"
    with open("hicodet_text.py", "a") as f:
        f.write(write_text)
    print("Done")
