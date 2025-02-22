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
    quotes, activities, categories = load_content_data()
    activity = random.choice(activities)
    category = random.choice(categories)
    return f"{category}: {activity}"

def generate_daily_inspiration() -> str:
    quotes, _, _ = load_content_data()
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
                # Generate different types of Python code templates
                templates = [
                    # Modern class with type hints and dataclass
                    [
                        'from dataclasses import dataclass',
                        'from typing import List, Optional, Dict',
                        'import datetime',
                        'import random',
                        '',
                        '@dataclass',
                        'class SmartDataProcessor:',
                        '    name: str',
                        '    data: List[float]',
                        '    created_at: datetime.datetime = datetime.datetime.now()',
                        '',
                        '    def process_data(self) -> Dict[str, float]:',
                        '        "
                ]
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
        # Single git add for all changes
        subprocess.run(['git', 'add', 'daily_update.txt'], check=True)
        
        message = generate_activity_message()
        try:
            # Commit and push changes
            subprocess.run(['git', 'commit', '-m', message], check=True)
            subprocess.run(['git', 'push'], check=True)
            print('Successfully committed and pushed changes')
        except subprocess.CalledProcessError as e:
            print(f'Error in git operations: {str(e)}')
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