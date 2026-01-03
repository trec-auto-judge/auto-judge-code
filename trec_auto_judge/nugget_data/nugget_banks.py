"""
Multi-topic NuggetBanks container with generic I/O.

Supports both NuggetBanks (autoargue) and NuggetizerNuggetBanks (autonuggetizer).
"""

import gzip
import json
from pathlib import Path
from typing import Dict, List, Type, TypeVar, Union, TextIO

from pydantic import BaseModel

from .nugget_data import NuggetBank

class NuggetBanks(BaseModel):
    """
    Container for multiple NuggetBanks, keyed by query_id.

    Access banks directly via the `banks` field:
        banks.banks["topic-1"]
        banks.banks.get("topic-1")
        for qid, bank in banks.banks.items(): ...
    """

    _bank_model = NuggetBank  # For protocol-based I/O

    format_version: str = "v3"
    banks: Dict[str, NuggetBank] = {}

    @classmethod
    def from_banks_list(
        cls, banks: List[NuggetBank], overwrite: bool = False
    ) -> "NuggetBanks":
        """
        Create from list of banks.

        Args:
            banks: List of NuggetBank instances
            overwrite: If False (default), raise error on duplicate query_id

        Raises:
            ValueError: If bank has no query_id, or duplicate query_id without overwrite
        """
        result: Dict[str, NuggetBank] = {}
        for bank in banks:
            qid = bank.query_id
            if qid is None:
                raise ValueError("NuggetBank must have a query_id")
            if qid in result and not overwrite:
                raise ValueError(f"Duplicate query_id: {qid}")
            result[qid] = bank
        return cls(banks=result)



# Curried I/O functions for NuggetBank loading
from .io import make_io_functions

load_nugget_banks_from_file, load_nugget_banks_from_directory, write_nugget_banks = \
    make_io_functions(NuggetBank, NuggetBanks)
