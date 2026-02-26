"""
test_llm_interaction.py
本地规则匹配 + 云端 LLM 回退的交互测试脚本。
"""

import json
import argparse
import re


class LocalMatcher:
    PATTERNS = [
        (r"(?:显示|展示|show)\s*(.+)", "show"),
        (r"(?:隐藏|关闭|hide)\s*(.+)", "hide"),
        (r"(?:高亮|强调|highlight)\s*(.+)", "highlight"),
        (r"(?:只看|只显示|isolate)\s*(.+)", "isolate"),
        (r"(?:重置|reset|恢复)", "reset"),
    ]

    def __init__(self, sem_dict):
        self.components = sem_dict["components"]
        self.groups = sem_dict.get("groups", [])
        self._alias_map = {}

        for c in self.components:
            for alias in [c["name"]] + c.get("aliases", []):
                self._alias_map[alias.lower()] = c["name"]
        for g in self.groups:
            for alias in [g["name"]] + g.get("aliases", []):
                self._alias_map[alias.lower()] = g["name"]

    def resolve(self, text):
        text_lower = text.lower().strip()
        for alias, name in self._alias_map.items():
            if alias in text_lower:
                return name
        return None

    def try_match(self, user_input):
        for pattern, action_type in self.PATTERNS:
            m = re.search(pattern, user_input)
            if not m:
                continue
            if action_type == "reset":
                return {"actions": [{"action_type": "reset", "target": "all", "parameters": {}}]}
            target = self.resolve(m.group(1).strip())
            if target:
                return {"actions": [{"action_type": action_type, "target": target, "parameters": {}}]}
        return None


class SceneState:
    def __init__(self, sem_dict):
        self.visible = {c["name"]: True for c in sem_dict["components"]}
        self.highlighted = set()

    def apply(self, actions):
        for action in actions.get("actions", []):
            target = action.get("target", "all")
            action_type = action.get("action_type")

            if action_type == "show":
                if target == "all":
                    for k in self.visible:
                        self.visible[k] = True
                elif target in self.visible:
                    self.visible[target] = True
            elif action_type == "hide":
                if target == "all":
                    for k in self.visible:
                        self.visible[k] = False
                elif target in self.visible:
                    self.visible[target] = False
            elif action_type == "reset":
                for k in self.visible:
                    self.visible[k] = True
                self.highlighted.clear()

    def summary(self):
        return {
            "visible": [k for k, v in self.visible.items() if v],
            "hidden": [k for k, v in self.visible.items() if not v],
            "highlighted": list(self.highlighted),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dict", required=True)
    parser.add_argument("--api_key", default="")
    parser.add_argument("--base_url", default="")
    args = parser.parse_args()

    with open(args.dict, "r", encoding="utf-8") as f:
        sem_dict = json.load(f)

    matcher = LocalMatcher(sem_dict)
    state = SceneState(sem_dict)

    llm_client = None
    system_prompt = ""
    if args.api_key or args.base_url:
        try:
            from openai import OpenAI

            kwargs = {}
            if args.api_key:
                kwargs["api_key"] = args.api_key
            if args.base_url:
                kwargs["base_url"] = args.base_url
            llm_client = OpenAI(**kwargs)

            prompt_path = args.dict.replace("semantic_dict.json", "system_prompt.txt")
            with open(prompt_path, "r", encoding="utf-8") as f:
                system_prompt = f.read()
            print("[LLM mode enabled: local match -> cloud fallback]\n")
        except Exception as e:
            print(f"[LLM not available: {e}. Local-only mode.]\n")

    print(f"Dataset: {sem_dict['dataset']} ({len(sem_dict['components'])} components)")
    print(f"Groups: {[g['name'] for g in sem_dict.get('groups', [])]}")
    print("Type commands, input 'quit' to exit.\n")

    history = []
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["quit", "exit", "q"]:
            break

        result = matcher.try_match(user_input)
        if result:
            print(f"[LOCAL MATCH] {json.dumps(result, ensure_ascii=False, indent=2)}")
            state.apply(result)
            print(f"[STATE] {json.dumps(state.summary(), ensure_ascii=False)}\n")
            continue

        if llm_client:
            messages = [{"role": "system", "content": system_prompt}]
            messages.append({"role": "system", "content": f"当前场景状态: {json.dumps(state.summary(), ensure_ascii=False)}"})
            messages.extend(history)
            messages.append({"role": "user", "content": user_input})

            try:
                resp = llm_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    temperature=0.3,
                    response_format={"type": "json_object"},
                )
                result_text = resp.choices[0].message.content
                result = json.loads(result_text)
                print(f"[LLM] {json.dumps(result, ensure_ascii=False, indent=2)}")
                state.apply(result)
                print(f"[STATE] {json.dumps(state.summary(), ensure_ascii=False)}\n")
                history.append({"role": "user", "content": user_input})
                history.append({"role": "assistant", "content": result_text})
            except Exception as e:
                print(f"[LLM ERROR] {e}\n")
        else:
            print("[No local match, LLM not available. Try simpler command.]\n")


if __name__ == "__main__":
    main()
