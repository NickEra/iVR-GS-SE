"""
semantic_config.py - 语义配置解析器（支持分组与导览）
"""

import yaml


class SemanticConfig:
    def __init__(self, config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            self.raw = yaml.safe_load(f)

        self.dataset = self.raw["dataset"]
        self.description = self.raw.get("description", "")
        self.components = self.raw["components"]
        self.groups = self.raw.get("groups", [])
        self.guided_tours = self.raw.get("guided_tours", [])

        self._by_id = {c["semantic_id"]: c for c in self.components}
        self._by_tf = {c["tf_id"]: c for c in self.components}
        self._by_name = {c["name"]: c for c in self.components}

    def get_by_id(self, semantic_id):
        return self._by_id.get(semantic_id)

    def get_by_tf(self, tf_id):
        return self._by_tf.get(tf_id)

    def get_by_name(self, name):
        return self._by_name.get(name)

    def get_all_tf_ids(self):
        return [c["tf_id"] for c in self.components]

    def get_all_names(self):
        return [c["name"] for c in self.components]

    def get_all_ids(self):
        return [c["semantic_id"] for c in self.components]

    def num_components(self):
        return len(self.components)

    def resolve_group(self, group_name):
        for g in self.groups:
            if g["name"] == group_name or group_name in g.get("aliases", []):
                return [
                    self._by_name[n]["semantic_id"]
                    for n in g.get("children", [])
                    if n in self._by_name
                ]
        return []

    def __repr__(self):
        return (
            f"SemanticConfig({self.dataset}, {self.num_components()} components, "
            f"{len(self.groups)} groups)"
        )


def load_config(config_path):
    return SemanticConfig(config_path)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        cfg = load_config(sys.argv[1])
        print(cfg)
        for c in cfg.components:
            print(f"  {c['tf_id']} -> id={c['semantic_id']}, name={c['name']}")
        for g in cfg.groups:
            print(f"  [group] {g['name']}: {g.get('children', [])}")
