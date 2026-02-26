"""
generate_prompt.py - 从 semantic_dict.json 自动生成系统提示词
"""

import json
import argparse


def dict_to_prompt(sem_dict):
    comps = []
    for c in sem_dict["components"]:
        comps.append(
            {
                "id": c["semantic_id"],
                "name": c["name"],
                "aliases": c.get("aliases", []),
                "description": c.get("description", ""),
            }
        )

    groups = sem_dict.get("groups", [])
    spatial = sem_dict.get("spatial_relations", [])
    tours = sem_dict.get("guided_tours", [])
    actions = sem_dict["action_schema"]["supported_actions"]

    groups_text = (
        "分组："
        + json.dumps(
            [
                {
                    "name": g["name"],
                    "aliases": g.get("aliases", []),
                    "children": g.get("children_names", g.get("children", [])),
                }
                for g in groups
            ],
            ensure_ascii=False,
            indent=2,
        )
        if groups
        else ""
    )
    spatial_text = "空间关系：" + json.dumps(spatial[:10], ensure_ascii=False, indent=2) if spatial else ""
    tours_text = (
        "预置导览："
        + json.dumps(
            [{"name": t["name"], "description": t.get("description", "")} for t in tours],
            ensure_ascii=False,
            indent=2,
        )
        if tours
        else ""
    )

    prompt = f"""你是沉浸式数据可视化助手。用户在VR环境中探索三维体数据。

当前数据集：{sem_dict['dataset']}
{sem_dict.get('description', '')}

语义组件：
{json.dumps(comps, ensure_ascii=False, indent=2)}

{groups_text}

{spatial_text}

{tours_text}

输出规则：
1. 仅操作已知对象，输出严格 JSON
2. aliases 匹配口语化表达并映射到正确 semantic_id
3. 分组名可直接作为 target（如 "fins"）
4. 支持的 action_type: {actions}
5. 用户说“导览/介绍”时，优先调用 guided_tour

JSON 输出格式：
{{"actions": [{{"action_type":"...", "target":"name或id或all", "parameters":{{}}, "narration":"可选中文解说"}}]}}
"""
    return prompt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dict", required=True)
    parser.add_argument("--output", default="system_prompt.txt")
    args = parser.parse_args()

    with open(args.dict, "r", encoding="utf-8") as f:
        sem_dict = json.load(f)
    prompt = dict_to_prompt(sem_dict)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(prompt)
    print(f"Prompt saved: {args.output} ({len(prompt)} chars)")


if __name__ == "__main__":
    main()
