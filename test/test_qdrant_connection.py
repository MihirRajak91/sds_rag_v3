"""
Test Qdrant connection directly
"""
from qdrant_client import QdrantClient
import time

def test_connection():
    print("Testing Qdrant connection...")
    
    try:
        # Try with timeout
        client = QdrantClient(host="localhost", port=6333, timeout=10)
        
        # Test basic operations
        print("1. Getting collections...")
        collections = client.get_collections()
        print(f"   Found {len(collections.collections)} collections")
        
        # Test creating a simple collection
        print("2. Creating test collection...")
        from qdrant_client.http.models import Distance, VectorParams
        
        try:
            client.create_collection(
                collection_name="connection_test",
                vectors_config=VectorParams(size=384, distance=Distance.COSINE)
            )
            print("   Collection created successfully")
        except Exception as e:
            if "already exists" in str(e):
                print("   Collection already exists")
            else:
                print(f"   Error creating collection: {e}")
        
        # Get info about the test collection
        print("3. Getting collection info...")
        try:
            info = client.get_collection("connection_test")
            print(f"   Collection status: {info.status}")
            print(f"   Points count: {info.points_count}")
        except Exception as e:
            print(f"   Error getting collection info: {e}")
        
        print("SUCCESS: Qdrant connection working!")
        return client
        
    except Exception as e:
        print(f"FAILED: {e}")
        return None

if __name__ == "__main__":
    client = test_connection()