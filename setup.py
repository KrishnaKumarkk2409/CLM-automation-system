"""
Setup script for CLM automation system.
Handles installation, configuration, and initial setup.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def print_banner():
    """Print installation banner"""
    print("=" * 60)
    print("🚀 CLM Automation System Setup")
    print("=" * 60)
    print()

def check_python_version():
    """Check Python version requirements"""
    print("🔍 Checking Python version...")
    
    if sys.version_info < (3, 9):
        print("❌ Error: Python 3.9+ is required")
        print(f"   Current version: {sys.version}")
        return False
    
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True

def install_dependencies():
    """Install Python dependencies"""
    print("\n📦 Installing dependencies...")
    
    try:
        # Install requirements
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    print("\n📁 Creating directories...")
    
    directories = [
        "./logs",
        "./documents", 
        "./tests",
        "./backups"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created: {directory}")

def setup_environment():
    """Set up environment configuration"""
    print("\n⚙️ Setting up environment configuration...")
    
    env_template = ".env.template"
    env_file = ".env"
    
    if not os.path.exists(env_template):
        print(f"❌ Error: {env_template} not found")
        return False
    
    if os.path.exists(env_file):
        response = input(f"📝 {env_file} already exists. Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("⏭️  Skipping environment setup")
            return True
    
    # Copy template to .env
    shutil.copy2(env_template, env_file)
    print(f"✅ Created {env_file} from template")
    
    print("\n🔧 Please edit .env with your configuration:")
    print("   - Supabase URL and key")
    print("   - OpenAI API key") 
    print("   - Email settings (optional)")
    
    return True

def check_system_dependencies():
    """Check for system-level dependencies"""
    print("\n🔍 Checking system dependencies...")
    
    # Check for Tesseract OCR
    try:
        result = subprocess.run(
            ["tesseract", "--version"], 
            capture_output=True, 
            text=True
        )
        if result.returncode == 0:
            print("✅ Tesseract OCR found")
        else:
            print("⚠️  Tesseract OCR not found (required for scanned PDFs)")
            print_tesseract_install_instructions()
    except FileNotFoundError:
        print("⚠️  Tesseract OCR not found (required for scanned PDFs)")
        print_tesseract_install_instructions()
    
    return True

def print_tesseract_install_instructions():
    """Print Tesseract installation instructions"""
    print("\n📋 To install Tesseract OCR:")
    
    if sys.platform == "darwin":  # macOS
        print("   brew install tesseract")
    elif sys.platform.startswith("linux"):  # Linux
        print("   sudo apt-get install tesseract-ocr  # Ubuntu/Debian")
        print("   sudo yum install tesseract         # CentOS/RHEL")
    elif sys.platform == "win32":  # Windows
        print("   Download from: https://github.com/UB-Mannheim/tesseract/wiki")

def generate_sample_data():
    """Generate sample contract data"""
    print("\n📄 Generating sample contract data...")
    
    try:
        result = subprocess.run([
            sys.executable, "src/generate_synthetic_data.py"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Sample contract documents generated")
            return True
        else:
            print(f"❌ Failed to generate sample data: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error generating sample data: {e}")
        return False

def verify_installation():
    """Verify installation by running basic tests"""
    print("\n🧪 Verifying installation...")
    
    try:
        # Test imports
        print("   Testing Python imports...")
        test_imports = [
            "langchain",
            "supabase", 
            "openai",
            "streamlit",
            "reportlab",
            "docx"
        ]
        
        for module in test_imports:
            try:
                __import__(module)
                print(f"   ✅ {module}")
            except ImportError:
                print(f"   ❌ {module} - not found")
                return False
        
        # Test configuration loading
        print("   Testing configuration...")
        try:
            import sys
            sys.path.append('src')
            from config import Config
            print("   ✅ Configuration loaded")
        except Exception as e:
            print(f"   ⚠️  Configuration warning: {e}")
        
        print("✅ Installation verification completed")
        return True
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False

def print_next_steps():
    """Print next steps for the user"""
    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next Steps:")
    print("1. Edit .env file with your API keys and database credentials")
    print("2. Set up your Supabase database using the SQL schema in README.md")
    print("3. Run: python main.py --process (to process sample documents)")
    print("4. Run: python main.py --chatbot (to start the web interface)")
    print("5. Or run: python main.py --interactive (for CLI interface)")
    
    print("\n📖 Documentation:")
    print("   - Full documentation in README.md")
    print("   - Check logs/ folder for system logs")
    print("   - Use --help flag with main.py for options")
    
    print("\n🆘 Need help?")
    print("   - Check troubleshooting section in README.md")
    print("   - Review log files for errors")

def main():
    """Main setup function"""
    print_banner()
    
    # Check requirements
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Setup failed: Could not install dependencies")
        sys.exit(1)
    
    # Set up environment
    if not setup_environment():
        print("❌ Setup failed: Could not set up environment")
        sys.exit(1)
    
    # Check system dependencies
    check_system_dependencies()
    
    # Ask about sample data
    response = input("\n📄 Generate sample contract documents for testing? (y/n): ")
    if response.lower() == 'y':
        generate_sample_data()
    
    # Verify installation
    if not verify_installation():
        print("⚠️  Setup completed with warnings")
    
    # Print next steps
    print_next_steps()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Setup cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Setup failed with error: {e}")
        sys.exit(1)