# GitHub Profile Activity Generator

This tool automatically generates daily commits to help maintain an active GitHub profile. It creates a daily update with an inspirational quote and commits it to your repository.

## Setup

1. Clone this repository to your local machine
2. Make sure you have Python 3.x installed
3. Set up your GitHub credentials:
   - Configure your Git username and email
   - Set up SSH keys or store your GitHub credentials

## How it Works

The `auto_commit.py` script:
- Creates/updates a `daily_update.txt` file with a timestamp and random quote
- Commits the changes to your repository
- Pushes the changes to GitHub

## Automated Daily Runs

To run the script automatically every day:

### On macOS/Linux:
1. Open terminal and type: `crontab -e`
2. Add this line to run it daily at 12:00 PM:
   ```
   0 12 * * * cd /path/to/repository && /usr/bin/python3 auto_commit.py
   ```
3. Save and exit

### On Windows:
1. Open Task Scheduler
2. Create a new task to run `python auto_commit.py` daily
3. Set the working directory to your repository location

## Manual Run

To run the script manually:
```bash
python3 auto_commit.py
```

## Note

Make sure to:
- Keep your local repository updated
- Maintain proper GitHub credentials
- Check the logs periodically to ensure the script is running correctly