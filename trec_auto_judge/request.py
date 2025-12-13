from enum import Enum
import gzip
from typing import Any, Dict, Iterable, List, Optional, Set, TextIO, Union, TypeAlias
from io import StringIO
from pathlib import Path
import json
from pydantic import BaseModel

class Request(BaseModel):
    request_id:str
    collection_ids:Optional[List[str]]= None
    background:Optional[str] = None
    original_background:Optional[str] = None
    problem_statement:Optional[str] = None
    limit:Optional[int] = None
    word_limit:Optional[int] = None
    title:str
    
     
def load_requests(reports_path:Path)->List[Request]:
    requests = list()
    with open(file=reports_path) as f:
        for line in f.readlines():
            data = json.load(fp=StringIO(line))
            request = Request.validate(data)
            requests.append(request)
    return requests

