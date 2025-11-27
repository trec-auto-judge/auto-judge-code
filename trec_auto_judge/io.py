from pathlib import Path
from glob import glob
import json
from collections import defaultdict

# TODO: Incorporate the pydantic file that Laura shared.
_REQUIRED_FIELDS = ["metadata", "responses"]


def load_run_failsave(path: Path):
    if not path or not path.exists() or not path.is_file():
        return []

    ret = []
    with open(path, 'r') as f:
        for l in f:
            try:
                l = json.loads(l)
            except:
                continue
            for required_field in _REQUIRED_FIELDS:
                if required_field not in l:
                    continue

            if "narrative_id" in l["metadata"] and "topic_id" not in l["metadata"]:
                l["metadata"]["topic_id"] = l["metadata"]["narrative_id"]

            if "topic_id" in l["metadata"] and "narrative_id" not in l["metadata"]:
                l["metadata"]["narrative_id"] = l["metadata"]["topic_id"]

            if "topic_id" in l["metadata"] and "narrative_id" in l["metadata"] and l["metadata"]["topic_id"] != l["metadata"]["narrative_id"]:
                raise ValueError(f"Inconsistent metadata: {l["metadata"]}")


            l["path"] = str(path.absolute())
            ret.append(l)

    return ret
                

def load_runs_failsave(path: Path):
    ret = []
    path = path.absolute()

    for f in sorted(glob(f"{path}/*") + glob(f"{path}/*/*") + glob(f"{path}/*/*/*")):
        ret.extend(load_run_failsave(Path(f)))
    
    run_id_to_topics = defaultdict(set)
    for l in [i["metadata"] for i in ret]:
        if "topic_id" in l and "run_id" in l:
            if l["topic_id"] in run_id_to_topics[l["run_id"]]:
                raise ValueError(f"There are duplicate entries for the metadata: {l}")
            run_id_to_topics[l["run_id"]].add(l["topic_id"])
    
    return ret
