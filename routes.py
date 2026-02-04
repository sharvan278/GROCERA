"""
Main application routes
Separated from app.py for better organization
"""
from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
from services import inventory_service, analytics_service, grok_service
from models import db
import pandas as pd
import os
from config import Config

# Main blueprint for web pages
main_bp = Blueprint('main', __name__)

# API blueprint for REST endpoints
api_bp = Blueprint('api', __name__)


# ============= WEB ROUTES =============

@main_bp.route('/')
def index():
    """Home page"""
    return render_template('index.html')


@main_bp.route('/dashboard')
@login_required
def dashboard():
    """Main dashboard"""
    summary = inventory_service.get_inventory_summary(current_user.id)
    alerts = analytics_service.get_stock_alerts(current_user.id)
    trending = analytics_service.get_trending_items(current_user.id, limit=5)
    
    return render_template('dashboard.html',
                         summary=summary,
                         alerts=alerts[:5],
                         trending=trending)


@main_bp.route('/inventory')
@login_required
def inventory():
    """Inventory listing page"""
    page = request.args.get('page', 1, type=int)
    search = request.args.get('search', '')
    
    if search:
        items = inventory_service.search_items(current_user.id, search)
        total = len(items)
    else:
        items, total = inventory_service.get_user_inventory(
            current_user.id, page, Config.ITEMS_PER_PAGE
        )
    
    inventory_status = analytics_service.get_inventory_status(current_user.id)
    
    return render_template('inventory.html',
                         items=items,
                         inventory_status=inventory_status,
                         page=page,
                         total=total,
                         per_page=Config.ITEMS_PER_PAGE,
                         search=search)


@main_bp.route('/upload', methods=['GET', 'POST'])
@login_required
def upload_file():
    """Handle CSV/Excel file uploads"""
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file selected', 'error')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(Config.UPLOAD_FOLDER, filename)
            file.save(filepath)
            
            try:
                # Process file
                df = process_csv(filepath)
                if df is not None and not df.empty:
                    # Import to database
                    results = inventory_service.bulk_import_from_dataframe(
                        current_user.id, df
                    )
                    
                    if results['success'] > 0:
                        flash(f'✅ Successfully imported {results["success"]} items!', 'success')
                    if results['errors'] > 0:
                        flash(f'⚠️ {results["errors"]} items failed to import', 'warning')
                    
                    if results['details']:
                        for detail in results['details'][:5]:
                            flash(detail, 'warning')
                    
                    return redirect(url_for('main.inventory'))
                else:
                    flash('Failed to process file or file is empty', 'error')
            except Exception as e:
                flash(f'Error processing file: {str(e)}', 'error')
        else:
            flash('Invalid file format. Please upload CSV or Excel', 'error')
    
    return render_template('upload.html')


@main_bp.route('/analytics')
@login_required
def analytics():
    """Analytics and insights page"""
    alerts = analytics_service.get_stock_alerts(current_user.id)
    trending = analytics_service.get_trending_items(current_user.id)
    restock = analytics_service.get_restock_recommendations(current_user.id)
    expiry = analytics_service.get_expiry_recommendations(current_user.id)
    price_recs = analytics_service.calculate_price_recommendations(current_user.id)
    
    return render_template('analytics.html',
                         alerts=alerts,
                         trending=trending,
                         restock_recommendations=restock,
                         expiry_recommendations=expiry,
                         price_recommendations=price_recs)


@main_bp.route('/chat', methods=['GET', 'POST'])
@login_required
def chat():
    """AI Chat assistant"""
    if request.method == 'POST':
        user_message = request.form.get('message', '').strip()
        
        if not user_message:
            return jsonify({'error': 'Empty message'}), 400
        
        try:
            # Get user's inventory as context
            items, _ = inventory_service.get_user_inventory(current_user.id, page=1, per_page=100)
            context = "\n".join([f"{item.item_name}: ${item.price}, Qty: {item.quantity}" 
                               for item in items[:20]])
            
            response = grok_service.chat_completion(user_message, context)
            return jsonify({'response': response})
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return render_template('chat.html')


@main_bp.route('/cart')
@login_required
def view_cart():
    """View shopping cart"""
    from models import Cart
    cart_items = Cart.query.filter_by(user_id=current_user.id).all()
    
    total = sum(item.subtotal for item in cart_items)
    
    return render_template('cart.html', cart_items=cart_items, total=total)


@main_bp.route('/cart/add/<int:item_id>', methods=['POST'])
@login_required
def add_to_cart(item_id):
    """Add item to cart"""
    from models import Cart, InventoryItem
    
    quantity = request.form.get('quantity', 1, type=int)
    
    # Check if item exists and belongs to user's inventory
    item = InventoryItem.query.filter_by(id=item_id, user_id=current_user.id).first()
    if not item:
        flash('Item not found', 'error')
        return redirect(url_for('main.inventory'))
    
    # Check if already in cart
    cart_item = Cart.query.filter_by(user_id=current_user.id, item_id=item_id).first()
    
    if cart_item:
        cart_item.quantity += quantity
    else:
        cart_item = Cart(user_id=current_user.id, item_id=item_id, quantity=quantity)
        db.session.add(cart_item)
    
    db.session.commit()
    flash(f'Added {item.item_name} to cart', 'success')
    
    return redirect(url_for('main.view_cart'))


@main_bp.route('/cart/update/<int:cart_id>', methods=['POST'])
@login_required
def update_cart(cart_id):
    """Update cart item quantity"""
    from models import Cart
    
    quantity = request.form.get('quantity', 1, type=int)
    cart_item = Cart.query.filter_by(id=cart_id, user_id=current_user.id).first()
    
    if cart_item:
        if quantity > 0:
            cart_item.quantity = quantity
            db.session.commit()
            flash('Cart updated', 'success')
        else:
            db.session.delete(cart_item)
            db.session.commit()
            flash('Item removed from cart', 'success')
    
    return redirect(url_for('main.view_cart'))


@main_bp.route('/cart/remove/<int:cart_id>', methods=['POST'])
@login_required
def remove_from_cart(cart_id):
    """Remove item from cart"""
    from models import Cart
    
    cart_item = Cart.query.filter_by(id=cart_id, user_id=current_user.id).first()
    if cart_item:
        db.session.delete(cart_item)
        db.session.commit()
        flash('Item removed from cart', 'success')
    
    return redirect(url_for('main.view_cart'))


@main_bp.route('/cart/clear', methods=['POST'])
@login_required
def clear_cart():
    """Clear entire cart"""
    from models import Cart
    
    Cart.query.filter_by(user_id=current_user.id).delete()
    db.session.commit()
    flash('Cart cleared', 'success')
    
    return redirect(url_for('main.view_cart'))


# ============= API ROUTES =============

@api_bp.route('/inventory', methods=['GET'])
@login_required
def api_get_inventory():
    """API: Get user inventory"""
    items, total = inventory_service.get_user_inventory(current_user.id)
    
    return jsonify({
        'success': True,
        'total': total,
        'items': [serialize_item(item) for item in items]
    })


@api_bp.route('/inventory', methods=['POST'])
@login_required
def api_create_item():
    """API: Create inventory item"""
    data = request.get_json()
    
    try:
        item = inventory_service.create_item(current_user.id, data)
        return jsonify({
            'success': True,
            'item': serialize_item(item)
        }), 201
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@api_bp.route('/inventory/<int:item_id>', methods=['PUT'])
@login_required
def api_update_item(item_id):
    """API: Update inventory item"""
    data = request.get_json()
    
    item = inventory_service.update_item(item_id, current_user.id, data)
    if item:
        return jsonify({'success': True, 'item': serialize_item(item)})
    else:
        return jsonify({'success': False, 'error': 'Item not found'}), 404


@api_bp.route('/inventory/<int:item_id>', methods=['DELETE'])
@login_required
def api_delete_item(item_id):
    """API: Delete inventory item"""
    success = inventory_service.delete_item(item_id, current_user.id)
    
    if success:
        return jsonify({'success': True, 'message': 'Item deleted'})
    else:
        return jsonify({'success': False, 'error': 'Item not found'}), 404


@api_bp.route('/analytics/alerts', methods=['GET'])
@login_required
def api_get_alerts():
    """API: Get stock alerts"""
    alerts = analytics_service.get_stock_alerts(current_user.id)
    return jsonify({'success': True, 'alerts': alerts})


@api_bp.route('/analytics/trending', methods=['GET'])
@login_required
def api_get_trending():
    """API: Get trending items"""
    trending = analytics_service.get_trending_items(current_user.id)
    return jsonify({'success': True, 'trending': trending})


@main_bp.route('/export/report')
@login_required
def download_report():
    """Download inventory report"""
    import json
    from flask import send_file
    from datetime import datetime
    
    items, _ = inventory_service.get_user_inventory(current_user.id, page=1, per_page=10000)
    
    report_data = {
        'generated_at': datetime.utcnow().isoformat(),
        'user': current_user.username,
        'total_items': len(items),
        'items': [serialize_item(item) for item in items]
    }
    
    filename = f'grocery_report_{datetime.utcnow().strftime("%Y%m%d_%H%M%S")}.json'
    filepath = os.path.join('processed', filename)
    
    with open(filepath, 'w') as f:
        json.dump(report_data, f, indent=2)
    
    return send_file(filepath, as_attachment=True)


@main_bp.route('/export/stock')
@login_required
def export_stock_report():
    """Export stock report as CSV"""
    from flask import Response
    import io
    
    items, _ = inventory_service.get_user_inventory(current_user.id, page=1, per_page=10000)
    
    # Create CSV
    output = io.StringIO()
    output.write('Item Name,Category,Price,Quantity,Unit,Store,Stock Status,Expiry Date,Total Value\n')
    
    for item in items:
        expiry = item.expiry_date.isoformat() if item.expiry_date else 'N/A'
        output.write(f'{item.item_name},{item.category},{item.price},{item.quantity},{item.unit},'
                    f'{item.store_name},{item.stock_status},{expiry},{item.total_value}\n')
    
    output.seek(0)
    return Response(
        output.getvalue(),
        mimetype='text/csv',
        headers={'Content-Disposition': f'attachment;filename=stock_report_{datetime.now().strftime("%Y%m%d")}.csv'}
    )


@main_bp.route('/clear-data', methods=['POST'])
@login_required
def clear_data():
    """Clear all user inventory data"""
    if current_user.role == 'admin':
        from models import InventoryItem, Alert, Cart
        
        InventoryItem.query.filter_by(user_id=current_user.id).delete()
        Alert.query.filter_by(user_id=current_user.id).delete()
        Cart.query.filter_by(user_id=current_user.id).delete()
        
        db.session.commit()
        flash('All data cleared successfully', 'success')
    else:
        flash('Unauthorized: Admin access required', 'error')
    
    return redirect(url_for('main.dashboard'))


# ============= UTILITY FUNCTIONS =============

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS


def process_csv(filepath):
    """Process CSV or Excel file"""
    try:
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
        elif filepath.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(filepath)
        else:
            return None
        
        if df.empty:
            return None
        
        # Clean column names
        df.columns = [str(col).strip().lower().replace(" ", "_").replace("-", "_") 
                     for col in df.columns]
        
        # Remove unnamed columns
        df = df.loc[:, ~df.columns.str.contains('^unnamed')]
        
        return df
    except Exception as e:
        print(f"Error processing file: {e}")
        return None


def serialize_item(item):
    """Serialize inventory item to JSON"""
    return {
        'id': item.id,
        'item_name': item.item_name,
        'category': item.category,
        'price': item.price,
        'quantity': item.quantity,
        'unit': item.unit,
        'store_name': item.store_name,
        'stock_status': item.stock_status,
        'expiry_date': item.expiry_date.isoformat() if item.expiry_date else None,
        'total_value': item.total_value,
        'days_to_expiry': item.days_to_expiry
    }
