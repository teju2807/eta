import subprocess
import os

def deploy_app():
    print("Starting deployment...")
    # Install dependencies
    subprocess.run(['pip', 'install', '-r', 'requirements.txt'])

    # Set environment variables (if any)
    os.environ['FLASK_APP'] = 'app.py'
    os.environ['FLASK_ENV'] = 'production'

    # Run the Flask app using Gunicorn
    subprocess.run(['gunicorn', '--bind', '0.0.0.0:5000', 'app:app'])

if __name__ == "__main__":
    deploy_app()
