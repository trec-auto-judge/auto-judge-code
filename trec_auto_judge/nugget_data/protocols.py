"""Protocols for generic nugget bank types."""

from typing import ClassVar, Dict, List, Protocol, Type, runtime_checkable


@runtime_checkable
class NuggetBankProtocol(Protocol):
    """Protocol for single-topic nugget banks.

    Implementations: NuggetBank, NuggetizerNuggetBank
    """

    @property
    def query_id(self) -> str:
        """Canonical identifier for the topic/query."""
        ...


@runtime_checkable
class NuggetBanksProtocol(Protocol):
    """Protocol for multi-topic nugget bank containers.

    Implementations: NuggetBanks, NuggetizerNuggetBanks
    """

    _bank_model: ClassVar[Type[NuggetBankProtocol]]
    """The bank model class for this container type."""

    banks: Dict[str, NuggetBankProtocol]

    @classmethod
    def from_banks_list(
        cls, banks: List[NuggetBankProtocol], overwrite: bool = False
    ) -> "NuggetBanksProtocol":
        """Create container from list of banks."""
        ...