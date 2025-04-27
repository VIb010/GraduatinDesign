import os
import json
import glob
import argparse
from openai import OpenAI

# —————— 配置 ——————
BASE_URL    = "https://api.deepseek.com"
API_KEY     = os.getenv("DEEPSEEK_API_KEY")
MODEL       = "deepseek-chat"
# ——————————————————

if not API_KEY:
    raise RuntimeError("请先通过 `export DEEPSEEK_API_KEY=\"your_deepseek_api_key\"` 设置环境变量")

# 初始化 DeepSeek 客户端（与 OpenAI 调用方式完全一致）
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

def get_advice_for_label(label: str) -> str:
    """
    针对单一 pred_label（"0"–"5"）调用 DeepSeek API，
    返回一段简洁明了的皮肤保养建议，并缓存结果。
    """
    label_map = {
        "0": "黑头 (Blackheads)",
        "1": "色斑 (Dark spots)",
        "2": "结节 (Nodules)",
        "3": "丘疹 (Papules)",
        "4": "脓疱 (Pustules)",
        "5": "白头 (Whiteheads)"
    }
    desc = label_map.get(label, label)
    system_prompt =("你是资深皮肤科医生，"
        "回答要包含两部分："
        "A) 专业建议——使用临床术语、成分浓度、频次等；"
        "B) 简单来说——一句话通俗概括，普通用户一下就能明白。")
    user_prompt   = (
        f"用户面部检测到：{desc}。"
        "请分别从“1. 清洁”“2. 保湿”“3. 局部治疗”三方面：\n"
        "- 在“专业建议”中给出具体产品成分、使用频率和操作方法；\n"
        "- 然后在“简单来说”中一句话归纳每个要点。"
    )

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0.7,
        max_tokens=500,
        stream=False
    )

    return resp.choices[0].message.content.strip()

def process_files(input_dir: str):
    """
    遍历 output/*/annotated/*.json，原地为每个 detection 添加 advice 字段。
    """
    advice_cache: dict[str, str] = {}
    pattern = os.path.join(input_dir, "*", "annotated", "*.json")

    for json_path in glob.glob(pattern):
        with open(json_path, 'r', encoding='utf-8') as f:
            detections = json.load(f)

        # 为每条检测添加 advice
        for det in detections:
            lbl = det.get("pred_label")
            if lbl not in advice_cache:
                advice_cache[lbl] = get_advice_for_label(lbl)
            det["advice"] = advice_cache[lbl]

        # 原地覆盖写回 JSON
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(detections, f, ensure_ascii=False, indent=2)

        print(f"Updated with advice: {json_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="为 CNN 检测结果添加皮肤保养建议 (使用 DeepSeek API)"
    )
    parser.add_argument(
        "--input_dir", default="output",
        help="根目录，包含 output/<stem>/annotated/*.json"
    )
    args = parser.parse_args()
    process_files(args.input_dir)