from fastapi import FastAPI, HTTPException, Depends, status
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime
import asyncio
from abc import ABC, abstractmethod

class StorageBackend(ABC):
    @abstractmethod
    async def get_items(self) -> List[Dict]:
        pass

class InMemoryStorage(StorageBackend):
    def __init__(self):
        self.items = []

    async def get_items(self) -> List[Dict]:
        return self.items

class Item(BaseModel):
    name: str = Field(..., min_length=1)
    description: Optional[str] = Field(None, max_length=1000)
    price: float = Field(..., gt=0)
    created_at: datetime = Field(default_factory=datetime.now)

app = FastAPI(title="Modern API", version="1.0.0")
storage = InMemoryStorage()

async def get_storage() -> StorageBackend:
    return storage

@app.get("/items/", response_model=List[Item])
async def read_items(storage: StorageBackend = Depends(get_storage)):
    return await storage.get_items()

@app.post("/items/", response_model=Item, status_code=status.HTTP_201_CREATED)
async def create_item(item: Item, storage: StorageBackend = Depends(get_storage)):
    await asyncio.sleep(0.1)  # Simulate IO operation
    storage.items.append(item.dict())
    return item