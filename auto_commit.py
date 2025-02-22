import os
import random
import datetime
import subprocess
from typing import List, Tuple
import json
from pathlib import Path

def load_content_data() -> Tuple[List[str], List[str], List[str]]:
    quotes = [
        "Stay hungry, stay foolish.",
        "Innovation distinguishes between a leader and a follower.",
        "Think different.",
        "Code is poetry.",
        "Keep pushing forward.",
        "The best way to predict the future is to invent it.",
        "Make it work, make it right, make it fast.",
        "Simplicity is the ultimate sophistication.",
        "Good code is its own best documentation.",
        "The only way to do great work is to love what you do."
    ]
    
    activities = [
        "Refactoring core modules",
        "Optimizing performance",
        "Implementing new features",
        "Fixing edge cases",
        "Improving documentation",
        "Enhancing user experience",
        "Updating dependencies",
        "Adding test coverage",
        "Code cleanup and maintenance",
        "Security improvements"
    ]
    
    file_types = [
        "documentation",
        "feature",
        "bugfix",
        "test",
        "refactor",
        "style",
        "chore"
    ]
    
    return quotes, activities, file_types

def generate_commit_message() -> str:
    quotes, activities, file_types = load_content_data()
    activity = random.choice(activities)
    file_type = random.choice(file_types)
    return f"{file_type}: {activity}"

def generate_random_content() -> str:
    quotes, _, _ = load_content_data()
    now = datetime.datetime.now()
    
    # Add some randomness to the timestamp within working hours (9 AM to 6 PM)
    hour = random.randint(9, 18)
    minute = random.randint(0, 59)
    timestamp = now.replace(hour=hour, minute=minute)
    
    content = [
        f"Update: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Activity: {generate_commit_message()}",
        f"Quote: {random.choice(quotes)}",
        f"Status: Completed",
        "---"
    ]
    
    return '\n'.join(content) + '\n'

def update_file():
    with open('daily_update.txt', 'a') as f:
        f.write(generate_random_content())

def git_commit_and_push():
    commit_message = generate_commit_message()
    commands = [
        ['git', 'add', 'daily_update.txt'],
        ['git', 'commit', '-m', commit_message],
        ['git', 'push']
    ]
    
    for cmd in commands:
        try:
            subprocess.run(cmd, check=True)
            print(f'Successfully executed: {" ".join(cmd)}')
        except subprocess.CalledProcessError as e:
            print(f'Error executing {" ".join(cmd)}: {str(e)}')
            return False
    return True

def main():
    # Create daily_update.txt if it doesn't exist
    if not os.path.exists('daily_update.txt'):
        open('daily_update.txt', 'w').close()
    
    update_file()
    if git_commit_and_push():
        print('Successfully updated repository')
    else:
        print('Failed to update repository')

if __name__ == '__main__':
    main()