"""
Data schemas and manifest structures for RegimeFlow.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


REQUIRED_COLUMNS = ["open", "high", "low", "close", "adj_close", "volume"]


@dataclass
class DatasetManifest:
    manifest_version: str = "1.0.0"
    symbol: str = "SPY"
    source: str = "yfinance"
    retrieval_timestamp_utc: str = field(
        default_factory=lambda: datetime.utcnow().isoformat() + "Z"
    )
    start_date: str = ""
    end_date: str = ""
    row_count: int = 0
    sha256_hash: str = ""
    adjusted: bool = True
    columns: List[str] = field(default_factory=lambda: list(REQUIRED_COLUMNS))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetManifest":
        return cls(**data)
