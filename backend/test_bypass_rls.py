#!/usr/bin/env python3
"""
Temporary test script to bypass RLS by using a direct client connection
This is for testing purposes only.
"""

import os
import sys
from pathlib import Path

# Add the src directory to Python path
sys.path.append(str(Path(__file__).parent / "src"))

from supabase import create_client
from config import Config

def test_document_insert():
    """Test document insertion with direct client setup"""
    
    print("Testing document insertion with different approaches...")
    
    # Method 1: Try with current setup
    print("\n1. Testing with current SUPABASE_KEY...")
    try:
        client = create_client(Config.SUPABASE_URL, Config.SUPABASE_KEY)
        result = client.table('documents').insert({
            'filename': 'test_direct.txt',
            'file_type': 'text/plain',
            'content': 'This is a direct test document',
            'metadata': {'source': 'direct_test'}
        }).execute()
        print(f"✅ Success with regular key: {result.data[0]['id']}")
        return result.data[0]['id']
    except Exception as e:
        print(f"❌ Failed with regular key: {e}")
    
    # Method 2: Try to get service key from user input
    print("\n2. If you have the Supabase Service Role Key, enter it now:")
    print("   (Go to Supabase Dashboard → Settings → API → Service Role Key)")
    service_key = input("Service Role Key (or press Enter to skip): ").strip()
    
    if service_key:
        try:
            client = create_client(Config.SUPABASE_URL, service_key)
            result = client.table('documents').insert({
                'filename': 'test_service.txt',
                'file_type': 'text/plain',
                'content': 'This is a service role test document',
                'metadata': {'source': 'service_test'}
            }).execute()
            print(f"✅ Success with service key: {result.data[0]['id']}")
            
            # Save the service key for future use
            env_file = Path(__file__).parent / ".env"
            with open(env_file, "a") as f:
                f.write(f"\nSUPABASE_SERVICE_ROLE_KEY={service_key}\n")
            print(f"✅ Service key saved to {env_file}")
            
            return result.data[0]['id']
        except Exception as e:
            print(f"❌ Failed with service key: {e}")
    
    print("\n3. Next steps:")
    print("   - Option 1: Get your Service Role Key from Supabase Dashboard")
    print("   - Option 2: Disable RLS temporarily using temp_disable_rls.sql")
    print("   - Option 3: Set up proper RLS policies using setup_rls_policies.sql")
    
    return None

if __name__ == "__main__":
    test_document_insert()