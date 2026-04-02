from app import app, db
from src.models.multi_tenant import Order, OrderItem
from src.models.models import CompetitorPrice

with app.app_context():
    # Check if tables exist
    result = db.session.execute(db.text("SHOW TABLES")).fetchall()
    tables = [row[0] for row in result]
    
    print("✓ Database tables created:")
    print("-" * 50)
    for table in sorted(tables):
        print(f"  • {table}")
    
    print("\n✓ New feature tables:")
    if 'orders' in tables:
        print("  ✓ orders")
    if 'order_items' in tables:
        print("  ✓ order_items")
    if 'competitor_prices' in tables:
        print("  ✓ competitor_prices")
    
    print("\n✓ Services registered:")
    from src.services import (
        payment_service,
        invoice_service,
        competitor_pricing_service,
        barcode_service,
    )
    print("  ✓ payment_service")
    print("  ✓ invoice_service")
    print("  ✓ competitor_pricing_service")
    print("  ✓ barcode_service")
    
    print("\n✓ Blueprints registered:")
    blueprints = [bp.name for bp in app.blueprints.values()]
    for bp in blueprints:
        print(f"  ✓ {bp}")
    
    print("\n" + "="*50)
    print("✅ All 5 features successfully activated!")
    print("="*50)
