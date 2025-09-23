#!/usr/bin/env python3
"""
Test script for CLM automation system.
Verifies basic functionality and configuration.
"""

import sys
import os
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that all required modules can be imported"""
    print("🧪 Testing imports...")
    
    try:
        from src.config import Config
        print("   ✅ Config module imported successfully")
        
        # Test configuration validation
        is_valid, missing = Config.validate_config()
        if is_valid:
            print("   ✅ Configuration is valid")
        else:
            print(f"   ⚠️  Configuration missing: {missing}")
            print("   📝 Please check your .env file")
        
    except Exception as e:
        print(f"   ❌ Config import failed: {e}")
        return False
    
    try:
        from src.database import DatabaseManager
        print("   ✅ Database module imported successfully")
    except Exception as e:
        print(f"   ❌ Database import failed: {e}")
        return False
    
    try:
        from src.embeddings import EmbeddingManager
        print("   ✅ Embeddings module imported successfully")
    except Exception as e:
        print(f"   ❌ Embeddings import failed: {e}")
        return False
    
    try:
        from src.document_processor import DocumentProcessor
        print("   ✅ Document processor imported successfully")
    except Exception as e:
        print(f"   ❌ Document processor import failed: {e}")
        return False
    
    try:
        from src.rag_pipeline import RAGPipeline
        print("   ✅ RAG pipeline imported successfully")
    except Exception as e:
        print(f"   ❌ RAG pipeline import failed: {e}")
        return False
    
    try:
        from src.contract_agent import ContractAgent
        print("   ✅ Contract agent imported successfully")
    except Exception as e:
        print(f"   ❌ Contract agent import failed: {e}")
        return False
    
    return True

def test_dependencies():
    """Test that required dependencies are available"""
    print("\n📦 Testing dependencies...")
    
    required_packages = [
        'supabase',
        'openai', 
        'streamlit',
        'langchain',
        'langchain_openai',
        'langchain_core',
        'PyPDF2',
        'docx',
        'pytesseract',
        'PIL',
        'pdf2image',
        'pandas',
        'plotly',
        'numpy'
    ]
    
    failed_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} - not available")
            failed_packages.append(package)
    
    if failed_packages:
        print(f"\n⚠️  Missing packages: {', '.join(failed_packages)}")
        print("   Run: pip install -r requirements.txt")
        return False
    
    return True

def test_directories():
    """Test that required directories exist"""
    print("\n📁 Testing directories...")
    
    required_dirs = ['./logs', './documents', './src']
    
    for directory in required_dirs:
        if os.path.exists(directory):
            print(f"   ✅ {directory}")
        else:
            print(f"   ❌ {directory} - missing")
            try:
                os.makedirs(directory, exist_ok=True)
                print(f"   🔧 Created {directory}")
            except Exception as e:
                print(f"   ❌ Failed to create {directory}: {e}")
                return False
    
    return True

def test_configuration():
    """Test configuration validation"""
    print("\n⚙️  Testing configuration...")
    
    try:
        from src.config import Config
        
        # Check if .env file exists
        if os.path.exists('.env'):
            print("   ✅ .env file found")
        else:
            print("   ⚠️  .env file not found")
            if os.path.exists('.env.template'):
                print("   📝 Copy .env.template to .env and fill in your values")
            else:
                print("   ❌ .env.template not found")
            return False
        
        # Test configuration validation
        is_valid, missing = Config.validate_config()
        
        if is_valid:
            print("   ✅ All required configuration present")
        else:
            print(f"   ⚠️  Missing configuration: {', '.join(missing)}")
            print("   📝 Please update your .env file with the missing values")
        
        return is_valid
        
    except Exception as e:
        print(f"   ❌ Configuration test failed: {e}")
        return False

def test_database_connection():
    """Test database connection (if configured)"""
    print("\n🗄️  Testing database connection...")
    
    try:
        from src.config import Config
        from src.database import DatabaseManager
        
        is_valid, missing = Config.validate_config()
        if not is_valid:
            print("   ⏭️  Skipping database test - configuration incomplete")
            return True
        
        db = DatabaseManager()
        
        # Try a simple query
        result = db.client.table('documents').select('id').limit(1).execute()
        print("   ✅ Database connection successful")
        return True
        
    except Exception as e:
        print(f"   ❌ Database connection failed: {e}")
        print("   📝 Check your Supabase URL and key in .env")
        return False

def test_synthetic_data():
    """Test synthetic data generation"""
    print("\n📄 Testing synthetic data...")
    
    try:
        from src.generate_synthetic_data import SyntheticDataGenerator
        
        # Check if documents already exist
        if os.path.exists('./documents') and os.listdir('./documents'):
            print("   ✅ Sample documents found in ./documents")
            print(f"   📊 {len(os.listdir('./documents'))} files present")
            return True
        else:
            print("   ⚠️  No documents found")
            print("   🔧 Generating sample data...")
            
            generator = SyntheticDataGenerator('./documents')
            generator.generate_all_documents()
            
            if os.path.exists('./documents') and os.listdir('./documents'):
                print("   ✅ Sample documents generated successfully")
                return True
            else:
                print("   ❌ Failed to generate sample documents")
                return False
                
    except Exception as e:
        print(f"   ❌ Synthetic data test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 CLM System Test Suite")
    print("=" * 40)
    
    tests = [
        ("Import Test", test_imports),
        ("Dependencies Test", test_dependencies), 
        ("Directories Test", test_directories),
        ("Configuration Test", test_configuration),
        ("Database Test", test_database_connection),
        ("Synthetic Data Test", test_synthetic_data),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}")
        print("-" * len(test_name))
        
        try:
            if test_func():
                print(f"   🎉 {test_name} PASSED")
                passed += 1
            else:
                print(f"   ❌ {test_name} FAILED")
        except Exception as e:
            print(f"   💥 {test_name} ERROR: {e}")
    
    print("\n" + "=" * 40)
    print(f"📊 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready to use.")
        print("\n🚀 Next steps:")
        print("   1. python main.py --process  (process documents)")
        print("   2. python main.py --chatbot  (start web interface)")
        print("   3. python main.py --interactive  (CLI interface)")
    else:
        print("⚠️  Some tests failed. Please address the issues above.")
        if passed < total // 2:
            print("   Run: python setup.py  for automated setup")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)