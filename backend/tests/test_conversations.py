"""
Quick test script to verify conversation endpoints are working.

Tests:
1. List conversations (should be empty or show existing)
2. Send a chat message (creates new conversation)
3. List conversations again (should show the new one)
4. Get specific conversation details
5. Update conversation title (PATCH endpoint)
6. Delete conversation
"""

import sys
from pathlib import Path

# Add parent directory to path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

import requests
from jwt_generation import generate_jwt_token

# Configuration
BASE_URL = "http://localhost:5001"
token = generate_jwt_token()
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

# Test data
USER_TEST_MESSAGE = "Hello, tell me the weather forecast for kungälv tomorrow."


def test_conversations():
    print("=" * 60)
    print("CONVERSATION ENDPOINTS TEST")
    print("=" * 60)

    # Step 1: List conversations (before)
    print("\n1. Listing conversations (before)...")
    response = requests.get(f"{BASE_URL}/conversations", headers=headers)
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}")
    initial_count = len(response.json().get("conversations", []))

    # Step 2: Send a chat message
    print("\n2. Sending chat message to create conversation...")
    chat_data = {
        "messages": [
            {
                "role": "user",
                "content": USER_TEST_MESSAGE,
            }
        ]
    }
    response = requests.post(f"{BASE_URL}/chat/agent", headers=headers, json=chat_data)
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        print(f"   Message sent successfully!")
    else:
        print(f"   Error: {response.text}")
        return

    # Step 3: List conversations (after)
    print("\n3. Listing conversations (after)...")
    response = requests.get(f"{BASE_URL}/conversations", headers=headers)
    print(f"   Status: {response.status_code}")
    conversations = response.json().get("conversations", [])
    print(f"   Found {len(conversations)} conversation(s)")

    if conversations:
        print(f"\n   Latest conversation:")
        latest = conversations[0]
        print(f"   - ID: {latest['id']}")
        print(f"   - Title: {latest['title']}")
        print(f"   - Preview: {latest.get('preview', 'N/A')}")
        print(f"   - Updated: {latest['updated_at']}")

        # Step 4: Get specific conversation
        print(f"\n4. Getting conversation details...")
        conv_id = latest["id"]
        response = requests.get(f"{BASE_URL}/conversations/{conv_id}", headers=headers)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            conv_data = response.json()
            print(f"   Title: {conv_data['title']}")
            print(f"   Messages: {len(conv_data.get('messages', []))}")
            print(f"\n   First message:")
            if conv_data.get("messages"):
                msg = conv_data["messages"][0]
                print(f"   - Role: {msg['role']}")
                print(f"   - Content: {msg['content'][:100]}...")
    else:
        print("   No conversations found!")

    # Step 5: Update conversation title
    if conversations:
        print(f"\n5. Updating conversation title...")
        conv_id = conversations[0]["id"]
        old_title = conversations[0]["title"]
        new_title = "Updated Title - Test Conversation"

        update_data = {"title": new_title}
        response = requests.patch(
            f"{BASE_URL}/conversations/{conv_id}",
            headers=headers,
            json=update_data
        )
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            updated_conv = response.json()
            print(f"   Old title: {old_title}")
            print(f"   New title: {updated_conv['title']}")
            print(f"   Updated at: {updated_conv['updated_at']}")

            # Verify the update
            print(f"\n   Verifying update...")
            response = requests.get(f"{BASE_URL}/conversations/{conv_id}", headers=headers)
            if response.status_code == 200:
                conv_data = response.json()
                if conv_data['title'] == new_title:
                    print(f"   ✓ Title successfully updated!")
                else:
                    print(f"   ✗ Title mismatch: {conv_data['title']}")
        else:
            print(f"   Error updating conversation: {response.text}")

    # Step 6: Delete the latest conversation if exists
    if conversations:
        print(f"\n6. Deleting the latest conversation...")
        conv_id = conversations[0]["id"]
        response = requests.delete(f"{BASE_URL}/conversations/{conv_id}", headers=headers)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            print("   Conversation deleted successfully.")
        else:
            print(f"   Error deleting conversation: {response.text}")

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    try:
        test_conversations()
    except requests.exceptions.ConnectionError:
        print("ERROR: Could not connect to Flask backend.")
        print("Make sure Flask is running on http://localhost:5000")
    except Exception as e:
        print(f"ERROR: {str(e)}")
