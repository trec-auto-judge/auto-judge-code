from pathlib import Path
from glob import glob
import json
from collections import defaultdict

from typing import List
from .report import load_report, Report


# # TODO: Incorporate the pydantic file that Laura shared.
# _REQUIRED_FIELDS = ["metadata", "responses", "answer"]

# def load_run_failsave(path: Path):
#     if not path or not path.exists() or not path.is_file():
#         return []

#     ret = []
#     error_lines = 0
#     with open(path, 'r') as f:
#         for l in f:
#             if error_lines > 4:
#                 return []

#             try:
#                 l = json.loads(l)
#             except:
#                 error_lines += 1
#                 continue

#             if not isinstance(l, dict):
#                 error_lines += 1
#                 continue

#             if "responses" in l and "answer" not in l:
#                 l["answer"] = l["responses"]
#             if "answer" in l and "responses" not in l:
#                 l["responses"] = l["answer"]
#             for required_field in _REQUIRED_FIELDS:
#                 if required_field not in l:
#                     error_lines += 1
#                     continue

#             if "qa_id" in l["metadata"] and "topic_id" not in l["metadata"]:
#                 l["metadata"]["topic_id"] = l["metadata"]["qa_id"]
#                 del l["metadata"]["qa_id"]

#             if "narrative_id" in l["metadata"] and "topic_id" not in l["metadata"]:
#                 l["metadata"]["topic_id"] = l["metadata"]["narrative_id"]

#             if "topic_id" in l["metadata"] and "narrative_id" not in l["metadata"]:
#                 l["metadata"]["narrative_id"] = l["metadata"]["topic_id"]

#             if "topic_id" in l["metadata"] and "narrative_id" in l["metadata"] and l["metadata"]["topic_id"] != l["metadata"]["narrative_id"]:
#                 raise ValueError(f"Inconsistent metadata: {l['metadata']}")

#             if "narrative_id" in l["metadata"]:
#                 l["metadata"]["narrative_id"] = str(l["metadata"]["narrative_id"])

#             if "topic_id" in l["metadata"]:
#                 l["metadata"]["topic_id"] = str(l["metadata"]["topic_id"])

#             l["path"] = str(path.absolute()).replace(".jsonl", "")
#             ret.append(l)

#     return ret
                

def load_runs_failsave(path: Path)->List[Report]:
    ret = []
    path = path.absolute()
    print(f"globs: {glob(f"{path}/*") + glob(f"{path}/*/*") + glob(f"{path}/*/*/*")}")

    for f in sorted(glob(f"{path}/*") + glob(f"{path}/*/*") + glob(f"{path}/*/*/*")):
        # ret.extend(load_run_failsave(Path(f)))
        print("Report path", f)
        ret.extend(load_report(Path(f)))
    return ret


def irds_from_dir(directory):
    from tira.ir_datasets_util import load_ir_dataset_from_local_file
    from ir_datasets import registry

    ds = load_ir_dataset_from_local_file(Path(directory), str(directory))
    if str(directory) not in registry:
        registry.register(str(directory), ds)
    return ds


def load_hf_dataset_config_or_none(c, required_fields):
    from yaml import safe_load

    if c.is_file():
        txt = c.read_text()
        for cfg in txt.split("---"):
            try:
                ret = safe_load(cfg)
                if all(f in ret for f in required_fields):
                    return ret
            except:
                pass
