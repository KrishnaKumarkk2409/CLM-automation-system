#!/usr/bin/env python3
"""
Streamlit chatbot runner for CLM automation system.
Handles path configuration and launches the Streamlit interface.
"""

import os
import sys
import subprocess

def main():
    """Run the Streamlit chatbot with proper path configuration"""
    
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Add current directory to Python path
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    # Set PYTHONPATH environment variable
    env = os.environ.copy()
    pythonpath = env.get('PYTHONPATH', '')
    if pythonpath:
        env['PYTHONPATH'] = f"{current_dir}:{pythonpath}"
    else:
        env['PYTHONPATH'] = current_dir
    
    # Run streamlit with the chatbot interface
    chatbot_file = os.path.join(current_dir, 'src', 'chatbot_interface.py')
    
    try:
        subprocess.run([
            sys.executable, 
            '-m', 'streamlit', 
            'run', 
            chatbot_file
        ], env=env, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running Streamlit: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nStreamlit stopped by user")
        sys.exit(0)

if __name__ == "__main__":
    main()