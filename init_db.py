"""
Database initialization script for Grocera
Run this script to create the MySQL database and tables
"""
import pymysql
from app import app, db

def create_database():
    """Create the grocera_db database if it doesn't exist"""
    try:
        # Connect to MySQL server without database
        connection = pymysql.connect(
            host='localhost',
            user='root',
            password='Sharvan@05',  # Add your MySQL password if you have one
            charset='utf8mb4'
        )
        
        with connection.cursor() as cursor:
            # Create database if not exists
            cursor.execute("CREATE DATABASE IF NOT EXISTS grocera_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
            print("✅ Database 'grocera_db' created or already exists")
        
        connection.close()
        return True
    except Exception as e:
        print(f"❌ Error creating database: {e}")
        print("Make sure MySQL is running and credentials are correct")
        return False

def init_tables():
    """Create all tables defined in models"""
    try:
        with app.app_context():
            db.create_all()
            print("✅ All database tables created successfully!")
            
            # Print created tables
            from sqlalchemy import inspect
            inspector = inspect(db.engine)
            tables = inspector.get_table_names()
            print(f"\n📋 Created tables: {', '.join(tables)}")
            
        return True
    except Exception as e:
        print(f"❌ Error creating tables: {e}")
        return False

def create_admin_user():
    """Create a default admin user"""
    try:
        from src.models.models import User
        
        with app.app_context():
            # Check if admin exists
            admin = User.query.filter_by(username='admin').first()
            if not admin:
                admin = User(
                    username='admin',
                    email='admin@grocera.com',
                    role='admin',
                    is_active=True
                )
                admin.set_password('admin123')  # Change this password!
                db.session.add(admin)
                db.session.commit()
                print("✅ Admin user created successfully!")
                print("   Username: admin")
                print("   Password: admin123")
                print("   ⚠️  Please change this password after first login!")
            else:
                print("ℹ️  Admin user already exists")
                
        return True
    except Exception as e:
        print(f"❌ Error creating admin user: {e}")
        return False

if __name__ == '__main__':
    print("🚀 Initializing Grocera Database...")
    print("=" * 50)
    
    # Step 1: Create database
    if create_database():
        # Step 2: Create tables
        if init_tables():
            # Step 3: Create admin user
            create_admin_user()
            print("\n" + "=" * 50)
            print("✅ Database initialization complete!")
            print("You can now run the application with: python app.py")
        else:
            print("\n❌ Failed to create tables")
    else:
        print("\n❌ Failed to create database")
        print("Please check:")
        print("1. MySQL server is running")
        print("2. Root password is correct in this script")
        print("3. You have necessary permissions")
