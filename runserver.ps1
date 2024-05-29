# Set execution policy for the current session
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process

# Activate the virtual environment
.\.venv\Scripts\Activate.ps1

# Run the Django server
python manage.py runserver

# Keep the window open after the server is stopped
pause
