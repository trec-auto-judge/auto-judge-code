
from typing import List, Dict, Optional, Set
from typing import TypeAlias, List
from collections import defaultdict
from pathlib import Path
from dataclasses import dataclass
import hashlib


def doc_id_md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class QrelEntry:
    query_id:str
    doc_id:str
    grade:int
    
    def format_qrels(self) -> str:
        return f"{self.query_id} 0 {self.doc_id} {self.grade}\n"

Qrels: TypeAlias = List[QrelEntry]

def write_qrel_file(qrel_out_file:Path, qrel_entries:List[QrelEntry]):
    '''Use to write qrels file'''
    with open(qrel_out_file, 'wt', encoding='utf-8') as file:
        file.writelines(entry.format_qrels() for entry in qrel_entries)
        file.close()

def read_qrel_file(qrel_in_file:Path) ->List[QrelEntry]:
    '''Use to read qrel file'''
    with open(qrel_in_file, 'rt') as file:
        qrel_entries:List[QrelEntry] = list()
        for line in file.readlines():
            splits = line.split(" ")
            if len(splits)>=4:
                qrel_entries.append(QrelEntry(query_id=splits[0], doc_id=splits[2], grade=int(splits[3])))
            else:
                raise RuntimeError(f"All lines in qrels file needs to contain four columns. Offending line: \"{line}\"")
    return qrel_entries
