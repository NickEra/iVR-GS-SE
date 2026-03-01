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
        "Groups: "
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
    spatial_text = "Spatial Relations: " + json.dumps(spatial[:10], ensure_ascii=False, indent=2) if spatial else ""
    tours_text = (
        "Guided Tours: "
        + json.dumps(
            [{"name": t["name"], "description": t.get("description", "")} for t in tours],
            ensure_ascii=False,
            indent=2,
        )
        if tours
        else ""
    )

    prompt = f”””You are an immersive data visualization assistant. The user is exploring 3D volumetric data in a VR environment.

Current dataset: {sem_dict['dataset']}
{sem_dict.get('description', '')}

Semantic components:
{json.dumps(comps, ensure_ascii=False, indent=2)}

{groups_text}

{spatial_text}

{tours_text}

Output rules:
1. Only operate on known objects; output strict JSON
2. Match aliases to colloquial expressions and map them to the correct semantic_id
3. Group names can be used directly as target (e.g. “fins”)
4. Supported action_type: {actions}
5. When the user asks for a “tour” or “introduction”, prefer calling guided_tour

JSON output format:
{{“actions”: [{{“action_type”:”...”, “target”:”name or id or all”, “parameters”:{{}}, “narration”:”optional narration”}}]}}
“””
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
