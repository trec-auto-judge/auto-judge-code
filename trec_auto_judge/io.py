from pathlib import Path
from glob import glob
import json

# TODO: Incorporate the pydantic file that Laura shared.
_REQUIRED_FIELDS = ["metadata", "responses"]


def load_run_failsave(path: Path):
    if not path or not path.exists() or not path.is_file():
        return []

    ret = []
    with open(path, 'r') as f:
        print(path)
        for l in f:
            try:
                l = json.loads(l)
            except:
                continue
            for required_field in _REQUIRED_FIELDS:
                if required_field not in l:
                    continue
            l["path"] = path
            ret.append(l)

    return ret
                

def load_runs_failsave(path: Path):
    ret = []
    path = path.absolute()

    for f in sorted(glob(f"{path}/*") + glob(f"{path}/*/*") + glob(f"{path}/*/*/*")):
        ret.extend(load_run_failsave(Path(f)))
    
    return ret
