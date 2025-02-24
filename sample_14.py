from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, ContextManager
from contextlib import contextmanager
import datetime
import logging
from pathlib import Path
import json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

@dataclass
class DataProcessor:
    name: str
    data: List[float] = field(default_factory=list)
    created_at: datetime.datetime = field(default_factory=datetime.datetime.now)
    logger: logging.Logger = field(init=False)

    def __post_init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)

    @contextmanager
    def error_handling(self) -> ContextManager[Any]:
        try:
            yield
        except Exception as e:
            self.logger.error(f"Error processing data: {str(e)}")
            raise

    def process_data(self) -> Dict[str, float]:
        with self.error_handling():
            if not self.data:
                self.logger.warning("No data available for processing")
                return {}
            stats = {
                "mean": sum(self.data) / len(self.data),
                "max": max(self.data),
                "min": min(self.data),
                "variance": sum((x - (sum(self.data) / len(self.data))) ** 2 for x in self.data) / len(self.data)
            }
            self.logger.info(f"Processed {len(self.data)} data points")
            return stats

    def save_to_file(self, filepath: Path) -> None:
        with self.error_handling():
            data = {
                "name": self.name,
                "data": self.data,
                "created_at": self.created_at.isoformat()
            }
            filepath.write_text(json.dumps(data, indent=2))
            self.logger.info(f"Saved data to {filepath}")