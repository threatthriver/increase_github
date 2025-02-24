import os
import random
import datetime
import subprocess
from typing import List, Tuple
import json
from pathlib import Path

# Smart limits for daily commits
MIN_DAILY_COMMITS = 3
MAX_DAILY_COMMITS = 30
MIN_INTERVAL_MINUTES = 30
MAX_INTERVAL_MINUTES = 120

# File extensions for different languages
LANGUAGE_EXTENSIONS = {
    'Python': '.py',
    'JavaScript': '.js',
    'TypeScript': '.ts',
    'Java': '.java',
    'C++': '.cpp',
    'Ruby': '.rb',
    'Go': '.go',
    'Rust': '.rs',
    'Swift': '.swift',
    'Kotlin': '.kt'
}

def load_content_data() -> Tuple[List[str], List[str], List[str], List[str]]:
    quotes = [
        # Language-agnostic quotes
        "Stay hungry, stay foolish.",
        "Innovation distinguishes between a leader and a follower.",
        "Think different.",
        "Code is poetry.",
        "Keep pushing forward.",
        "The best way to predict the future is to invent it.",
        "Make it work, make it right, make it fast.",
        "Simplicity is the ultimate sophistication.",
        "Good code is its own best documentation.",
        "The only way to do great work is to love what you do.",
        "Every commit tells a story.",
        "Clean code always looks like it was written by someone who cares.",
        "First, solve the problem. Then, write the code.",
        "Programming isn't about what you know; it's about what you can figure out.",
        "The best error message is the one that never shows up.",
        # Language-specific quotes
        "Python: Readability counts.",
        "JavaScript: Everything is an object.",
        "Java: Write once, run anywhere.",
        "C++: Zero-cost abstractions.",
        "Ruby: Convention over configuration.",
        "Go: Simplicity is complicated.",
        "Rust: Memory safety without garbage collection.",
        "TypeScript: JavaScript that scales.",
        "Swift: Modern, safe, fast.",
        "Kotlin: Concise but expressive."
    ]
    
    activities = [
        # General activities
        "Refactoring core modules",
        "Optimizing performance",
        "Implementing new features",
        "Fixing edge cases",
        "Improving documentation",
        "Enhancing user experience",
        "Updating dependencies",
        "Adding test coverage",
        "Code cleanup and maintenance",
        "Security improvements",
        # Language-specific activities
        "Implementing async/await patterns",
        "Optimizing memory management",
        "Adding type annotations",
        "Improving error handling",
        "Implementing design patterns",
        "Enhancing code modularity",
        "Optimizing database queries",
        "Adding middleware components",
        "Implementing caching strategies",
        "Building microservices"
    ]
    
    categories = [
        # General categories
        "productivity",
        "innovation",
        "maintenance",
        "testing",
        "improvement",
        "optimization",
        "development",
        # Language-specific categories
        "python",
        "javascript",
        "java",
        "cpp",
        "ruby",
        "go",
        "rust",
        "typescript",
        "swift",
        "kotlin"
    ]
    
    languages = [
        "Python",
        "JavaScript",
        "Java",
        "C++",
        "Ruby",
        "Go",
        "Rust",
        "TypeScript",
        "Swift",
        "Kotlin"
    ]
    
    return quotes, activities, categories, languages

def generate_activity_message() -> str:
    _, activities, categories, _ = load_content_data()
    activity = random.choice(activities)
    category = random.choice(categories)
    return f"{category}: {activity}"

def generate_daily_inspiration() -> str:
    quotes, activities, categories, _ = load_content_data()
    now = datetime.datetime.now()
    
    # Ensure commits only happen during working hours (9 AM to 6 PM)
    current_hour = now.hour
    if current_hour < 9 or current_hour > 18:
        return ''
    
    # Add some randomness to the timestamp within working hours
    hour = random.randint(9, 18)
    minute = random.randint(0, 59)
    timestamp = now.replace(hour=hour, minute=minute)
    
    content = [
        f"Update: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Activity: {generate_activity_message()}",
        f"Quote: {random.choice(quotes)}",
        f"Status: Completed",
        "---"
    ]
    
    return '\n'.join(content) + '\n'

def get_smart_commit_count() -> int:
    # Determine number of commits based on day of week
    now = datetime.datetime.now()
    if now.weekday() >= 5:  # Weekend
        return random.randint(MIN_DAILY_COMMITS - 1, MIN_DAILY_COMMITS + 1)
    return random.randint(MIN_DAILY_COMMITS, MAX_DAILY_COMMITS)

def generate_code_changes():
    quotes, activities, categories, languages = load_content_data()
    selected_lang = random.choice(['Python', 'Python', 'Python', 'JavaScript'] + languages)  # Higher weight for Python
    ext = LANGUAGE_EXTENSIONS[selected_lang]
    
    # Create a new file or update existing one
    filename = f"sample_{random.randint(1, 100)}{ext}"
    
    if not os.path.exists(filename):
        with open(filename, 'w') as f:
            if ext == '.py':
                # Generate diverse Python code templates showcasing modern features and best practices
                templates = [
                    # Advanced data processing with context manager and error handling
                    [
                        'from dataclasses import dataclass, field',
                        'from typing import List, Optional, Dict, Any, ContextManager',
                        'from contextlib import contextmanager',
                        'import datetime',
                        'import logging',
                        'from pathlib import Path',
                        'import json',
                        '',
                        'logging.basicConfig(',
                        '    level=logging.INFO,',
                        '    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"',
                        ')',
                        '',
                        '@dataclass',
                        'class DataProcessor:',
                        '    name: str',
                        '    data: List[float] = field(default_factory=list)',
                        '    created_at: datetime.datetime = field(default_factory=datetime.datetime.now)',
                        '    logger: logging.Logger = field(init=False)',
                        '',
                        '    def __post_init__(self) -> None:',
                        '        self.logger = logging.getLogger(self.__class__.__name__)',
                        '',
                        '    @contextmanager',
                        '    def error_handling(self) -> ContextManager[Any]:',
                        '        try:',
                        '            yield',
                        '        except Exception as e:',
                        '            self.logger.error(f"Error processing data: {str(e)}")',
                        '            raise',
                        '',
                        '    def process_data(self) -> Dict[str, float]:',
                        '        with self.error_handling():',
                        '            if not self.data:',
                        '                self.logger.warning("No data available for processing")',
                        '                return {}',
                        '            stats = {',
                        '                "mean": sum(self.data) / len(self.data),',
                        '                "max": max(self.data),',
                        '                "min": min(self.data),',
                        '                "variance": sum((x - (sum(self.data) / len(self.data))) ** 2 for x in self.data) / len(self.data)',
                        '            }',
                        '            self.logger.info(f"Processed {len(self.data)} data points")',
                        '            return stats',
                        '',
                        '    def save_to_file(self, filepath: Path) -> None:',
                        '        with self.error_handling():',
                        '            data = {',
                        '                "name": self.name,',
                        '                "data": self.data,',
                        '                "created_at": self.created_at.isoformat()',
                        '            }',
                        '            filepath.write_text(json.dumps(data, indent=2))',
                        '            self.logger.info(f"Saved data to {filepath}")',
                    ],
                    # Modern async API with dependency injection
                    [
                        'from fastapi import FastAPI, HTTPException, Depends, status',
                        'from pydantic import BaseModel, Field',
                        'from typing import List, Optional, Dict',
                        'from datetime import datetime',
                        'import asyncio',
                        'from abc import ABC, abstractmethod',
                        '',
                        'class StorageBackend(ABC):',
                        '    @abstractmethod',
                        '    async def get_items(self) -> List[Dict]:',
                        '        pass',
                        '',
                        'class InMemoryStorage(StorageBackend):',
                        '    def __init__(self):',
                        '        self.items = []',
                        '',
                        '    async def get_items(self) -> List[Dict]:',
                        '        return self.items',
                        '',
                        'class Item(BaseModel):',
                        '    name: str = Field(..., min_length=1)',
                        '    description: Optional[str] = Field(None, max_length=1000)',
                        '    price: float = Field(..., gt=0)',
                        '    created_at: datetime = Field(default_factory=datetime.now)',
                        '',
                        'app = FastAPI(title="Modern API", version="1.0.0")',
                        'storage = InMemoryStorage()',
                        '',
                        'async def get_storage() -> StorageBackend:',
                        '    return storage',
                        '',
                        '@app.get("/items/", response_model=List[Item])',
                        'async def read_items(storage: StorageBackend = Depends(get_storage)):',
                        '    return await storage.get_items()',
                        '',
                        '@app.post("/items/", response_model=Item, status_code=status.HTTP_201_CREATED)',
                        'async def create_item(item: Item, storage: StorageBackend = Depends(get_storage)):',
                        '    await asyncio.sleep(0.1)  # Simulate IO operation',
                        '    storage.items.append(item.dict())',
                        '    return item',
                    ],
                    # Advanced ML with type hints and custom exceptions
                    [
                        'import numpy as np',
                        'import numpy.typing as npt',
                        'from dataclasses import dataclass',
                        'from typing import Optional, Tuple',
                        'import logging',
                        '',
                        'class ModelError(Exception):',
                        '    """Base exception for model errors"""',
                        '',
                        'class NotFittedError(ModelError):',
                        '    """Raised when prediction is attempted on an unfitted model"""',
                        '',
                        '@dataclass',
                        'class BinaryClassifier:',
                        '    learning_rate: float = 0.01',
                        '    max_iterations: int = 1000',
                        '    weights: Optional[npt.NDArray] = None',
                        '    logger: logging.Logger = logging.getLogger("BinaryClassifier")',
                        '',
                        '    def fit(self, X: npt.NDArray, y: npt.NDArray) -> Tuple[float, float]:',
                        '        """Train the model using gradient descent."""',
                        '        if len(X.shape) != 2:',
                        '            raise ValueError(f"Expected 2D array, got {len(X.shape)}D")',
                        '        if len(y.shape) != 1:',
                        '            raise ValueError(f"Expected 1D array for labels, got {len(y.shape)}D")',
                        '',
                        '        n_samples, n_features = X.shape',
                        '        self.weights = np.zeros(n_features)',
                        '        best_accuracy = 0.0',
                        '        best_loss = float("inf")',
                        '',
                        '        for i in range(self.max_iterations):',
                        '            predictions = self._sigmoid(np.dot(X, self.weights))',
                        '            loss = self._compute_loss(y, predictions)',
                        '            accuracy = np.mean((predictions >= 0.5) == y)',
                        '',
                        '            if accuracy > best_accuracy:',
                        '                best_accuracy = accuracy',
                        '                best_loss = loss',
                        '',
                        '            gradient = np.dot(X.T, (predictions - y)) / n_samples',
                        '            self.weights -= self.learning_rate * gradient',
                        '',
                        '            if i % 100 == 0:',
                        '                self.logger.info(',
                        '                    f"Iteration {i}: loss={loss:.4f}, accuracy={accuracy:.4f}"',
                        '                )',
                        '',
                        '        return best_loss, best_accuracy',
                        '',
                        '    def predict(self, X: npt.NDArray) -> npt.NDArray:',
                        '        """Predict binary classes for X."""',
                        '        if self.weights is None:',
                        '            raise NotFittedError("Model must be fitted before prediction")',
                        '        return self._sigmoid(np.dot(X, self.weights)) >= 0.5',
                        '',
                        '    @staticmethod',
                        '    def _sigmoid(x: npt.NDArray) -> npt.NDArray:',
                        '        return 1 / (1 + np.exp(-np.clip(x, -250, 250)))',
                        '',
                        '    @staticmethod',
                        '    def _compute_loss(y: npt.NDArray, y_pred: npt.NDArray) -> float:',
                        '        epsilon = 1e-15',
                        '        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)',
                        '        return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))',
                    ]
                ]
                code = random.choice(templates)
                f.write('\n'.join(code))
            elif ext == '.js':
                code = [
                    'class DataProcessor {',
                    '    constructor() {',
                    '        this.data = [];',
                    '    }',
                    '',
                    '    processData(items) {',
                    '        return items.map(item => item * 2);',
                    '    }',
                    '',
                    '    analyzeResults(data) {',
                    '        return data.length ? data.reduce((a, b) => a + b) / data.length : 0;',
                    '    }',
                    '}',
                    '',
                    'const processor = new DataProcessor();',
                    'const testData = Array.from({length: 5}, () => Math.floor(Math.random() * 100));',
                    'const results = processor.processData(testData);',
                    'const average = processor.analyzeResults(results);',
                    'console.log(`Processed data: ${results}`);',
                    'console.log(`Average: ${average}`);'
                ]
                f.write('\n'.join(code))
            elif ext == '.java':
                code = [
                    'public class DataProcessor {',
                    '    private List<Integer> data = new ArrayList<>();',
                    '',
                    '    public List<Integer> processData(List<Integer> items) {',
                    '        return items.stream().map(item -> item * 2).collect(Collectors.toList());',
                    '    }',
                    '',
                    '    public double analyzeResults(List<Integer> data) {',
                    '        return data.stream().mapToInt(Integer::intValue).average().orElse(0.0);',
                    '    }',
                    '}'
                ]
                f.write('\n'.join(code))
    
    return filename

def update_daily_log():
    try:
        commit_count = get_smart_commit_count()
        print(f'Planning to generate {commit_count} updates today')
        
        for i in range(commit_count):
            inspiration = generate_daily_inspiration()
            if not inspiration:  # Skip if outside working hours
                continue
                
            with open('daily_update.txt', 'a') as f:
                f.write(inspiration)
                print(f'Generated inspiration #{i + 1}')
            
            # Generate code changes
            code_file = generate_code_changes()
            subprocess.run(['git', 'add', code_file], check=True)
                
            # Add random delay between commits
            if i < commit_count - 1:  # Don't wait after the last commit
                delay_minutes = random.randint(MIN_INTERVAL_MINUTES, MAX_INTERVAL_MINUTES)
                print(f'Waiting {delay_minutes} minutes before next update...')
                
    except IOError as e:
        print(f'Error writing to files: {str(e)}')
        return False
    return True

def save_and_backup():
    try:
        # Check if there are changes to commit
        status = subprocess.run(['git', 'status', '--porcelain'], capture_output=True, text=True, check=True)
        if not status.stdout.strip():
            print('No changes to commit')
            return True

        # Add all changes
        subprocess.run(['git', 'add', '.'], check=True)
        
        message = generate_activity_message()
        try:
            # Configure git to handle line endings
            subprocess.run(['git', 'config', 'core.autocrlf', 'input'], check=True)
            
            # Commit and push changes
            subprocess.run(['git', 'commit', '-m', message], check=True)
            
            # Pull before push to avoid conflicts
            subprocess.run(['git', 'pull', '--rebase'], check=True)
            subprocess.run(['git', 'push'], check=True)
            print('Successfully committed and pushed changes')
        except subprocess.CalledProcessError as e:
            print(f'Error in git operations: {str(e)}')
            # Try to recover from common git errors
            try:
                subprocess.run(['git', 'reset', '--mixed'], check=True)
                print('Successfully reset git state after error')
            except subprocess.CalledProcessError:
                print('Failed to reset git state')
            return False
    except Exception as e:
        print(f'Unexpected error: {str(e)}')
        return False
    return True

if __name__ == '__main__':
    if update_daily_log():
        save_and_backup()

def main():
    try:
        # Create daily_update.txt if it doesn't exist
        if not os.path.exists('daily_update.txt'):
            open('daily_update.txt', 'w').close()
            print('Created new daily_update.txt file')
        
        if not update_daily_log():
            print('Failed to update daily log')
            return
        
        if not save_and_backup():
            print('Failed to save and backup changes')
            return
        
        print('Successfully completed daily inspiration generation and backup')
    except Exception as e:
        print(f'Critical error in main execution: {str(e)}')

if __name__ == '__main__':
    main()