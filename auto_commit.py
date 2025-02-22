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
        "The only way to do great work is to love what you do.",
        "Every commit tells a story.",
        "Clean code always looks like it was written by someone who cares.",
        "First, solve the problem. Then, write the code.",
        "Programming isn't about what you know; it's about what you can figure out.",
        "The best error message is the one that never shows up."
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
    
    categories = [
        "productivity",
        "innovation",
        "maintenance",
        "testing",
        "improvement",
        "optimization",
        "development"
    ]
    
    return quotes, activities, categories

def generate_activity_message() -> str:
    quotes, activities, categories = load_content_data()
    activity = random.choice(activities)
    category = random.choice(categories)
    return f"{category}: {activity}"

def generate_daily_inspiration() -> str:
    quotes, _, _ = load_content_data()
    now = datetime.datetime.now()
    
    # Add some randomness to the timestamp within working hours (9 AM to 6 PM)
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

def update_daily_log(count=30):
    try:
        for i in range(count):
            with open('daily_update.txt', 'a') as f:
                inspiration = generate_daily_inspiration()
                f.write(inspiration)
                print(f'Generated inspiration #{i + 1}')
    except IOError as e:
        print(f'Error writing to daily_update.txt: {str(e)}')
        return False
    return True

def save_and_backup(count=30):
    try:
        # Single git add for all changes
        subprocess.run(['git', 'add', 'daily_update.txt'], check=True)
        
        for i in range(count):
            message = generate_activity_message()
            try:
                # Commit and push for each inspiration
                subprocess.run(['git', 'commit', '-m', message], check=True)
                subprocess.run(['git', 'push'], check=True)
                print(f'Successfully saved and backed up inspiration #{i + 1}')
            except subprocess.CalledProcessError as e:
                print(f'Error in git operations for inspiration #{i + 1}: {str(e)}')
                return False
        return True
    except Exception as e:
        print(f'Unexpected error in save_and_backup: {str(e)}')
        return False

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