from backend.database import SessionLocal, User, ChatHistory
import sys

# Ensure we can import from backend
sys.path.append('.')

def view_data():
    db = SessionLocal()
    try:
        users = db.query(User).all()
        print(f"\n===== 📊 DATABASE REPORT =====")
        print(f"Total Users: {len(users)}")
        
        for user in users:
            print(f"\n👤 [ID: {user.id}] {user.username} ({user.email})")
            
            chats = db.query(ChatHistory).filter(ChatHistory.user_id == user.id).all()
            print(f"   💬 History ({len(chats)} messages):")
            for chat in chats:
                print(f"      - [{chat.timestamp.strftime('%Y-%m-%d %H:%M')}] {chat.query}")
                print(f"        Response: {chat.response[:50]}...")
        
        if not users:
            print("\n❌ No users found in the database yet.")
            
        print(f"\n==============================")
    finally:
        db.close()

if __name__ == "__main__":
    view_data()
