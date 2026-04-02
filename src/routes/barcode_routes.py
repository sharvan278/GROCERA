"""
Barcode Routes Blueprint

Handles barcode generation, scanning, and product lookup.
"""
from flask import Blueprint, render_template, request, jsonify, flash, redirect, url_for, send_file
from flask_login import login_required, current_user
from src.models.models import InventoryItem
from src.services.barcode_service import barcode_service
import os
from werkzeug.utils import secure_filename

barcode_bp = Blueprint('barcode', __name__, url_prefix='/barcode')


@barcode_bp.route('/generate/<int:item_id>')
@login_required
def generate_barcode(item_id):
    """Generate barcode for an item"""
    item = InventoryItem.query.filter_by(
        id=item_id,
        user_id=current_user.id
    ).first_or_404()
    
    try:
        filepath = barcode_service.generate_barcode(item_id)
        
        return jsonify({
            'success': True,
            'barcode_path': filepath,
            'barcode_value': item.barcode
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@barcode_bp.route('/generate-qr/<int:item_id>')
@login_required
def generate_qr(item_id):
    """Generate QR code for an item"""
    item = InventoryItem.query.filter_by(
        id=item_id,
        user_id=current_user.id
    ).first_or_404()
    
    include_details = request.args.get('details', 'true').lower() == 'true'
    
    try:
        filepath = barcode_service.generate_qr_code(item_id, include_details)
        
        return jsonify({
            'success': True,
            'qr_path': filepath
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@barcode_bp.route('/scan', methods=['GET', 'POST'])
@login_required
def scan():
    """Barcode scanning page"""
    if request.method == 'POST':
        # Handle barcode from camera/input
        barcode_value = request.form.get('barcode_value')
        
        if barcode_value:
            # Look up in inventory
            item = barcode_service.lookup_barcode(barcode_value)
            
            if item and item.user_id == current_user.id:
                return jsonify({
                    'success': True,
                    'found_in_inventory': True,
                    'item': {
                        'id': item.id,
                        'name': item.item_name,
                        'price': item.price,
                        'quantity': item.quantity
                    }
                })
            
            # Try external lookup
            external_data = barcode_service.lookup_barcode_external(barcode_value)
            
            if external_data:
                return jsonify({
                    'success': True,
                    'found_in_inventory': False,
                    'external_data': external_data
                })
            
            return jsonify({'success': False, 'error': 'Product not found'}), 404
    
    # Return JSON for now (frontend to be implemented)
    return jsonify({
        'message': 'Barcode scanner - frontend to be implemented',
        'instructions': 'POST barcode_value to scan'
    })


@barcode_bp.route('/scan-image', methods=['POST'])
@login_required
def scan_image():
    """Scan barcode from uploaded image"""
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'No image provided'}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'}), 400
    
    try:
        # Save uploaded file temporarily
        filename = secure_filename(file.filename)
        temp_path = os.path.join('uploads', 'temp', filename)
        os.makedirs(os.path.dirname(temp_path), exist_ok=True)
        file.save(temp_path)
        
        # Scan barcode from image
        barcode_value = barcode_service.scan_barcode_from_image(temp_path)
        
        # Clean up temp file
        os.remove(temp_path)
        
        if barcode_value:
            # Look up item
            item = barcode_service.lookup_barcode(barcode_value)
            
            if item and item.user_id == current_user.id:
                return jsonify({
                    'success': True,
                    'barcode': barcode_value,
                    'item': {
                        'id': item.id,
                        'name': item.item_name,
                        'price': item.price,
                        'quantity': item.quantity
                    }
                })
            
            # Try external lookup
            external_data = barcode_service.lookup_barcode_external(barcode_value)
            
            return jsonify({
                'success': True,
                'barcode': barcode_value,
                'external_data': external_data
            })
        
        return jsonify({
            'success': False,
            'error': 'No barcode found in image'
        }), 404
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@barcode_bp.route('/lookup/<barcode_value>')
@login_required
def lookup(barcode_value):
    """Look up product by barcode"""
    # Check inventory first
    item = barcode_service.lookup_barcode(barcode_value)
    
    if item and item.user_id == current_user.id:
        return jsonify({
            'success': True,
            'found_in_inventory': True,
            'item': {
                'id': item.id,
                'name': item.item_name,
                'category': item.category,
                'price': item.price,
                'quantity': item.quantity
            }
        })
    
    # Try external lookup
    external_data = barcode_service.lookup_barcode_external(barcode_value)
    
    if external_data:
        return jsonify({
            'success': True,
            'found_in_inventory': False,
            'external_data': external_data
        })
    
    return jsonify({
        'success': False,
        'error': 'Product not found'
    }), 404


@barcode_bp.route('/add-from-scan', methods=['POST'])
@login_required
def add_from_scan():
    """Add item to inventory from barcode scan"""
    data = request.get_json()
    barcode_value = data.get('barcode_value')
    
    if not barcode_value:
        return jsonify({'success': False, 'error': 'Barcode required'}), 400
    
    try:
        item = barcode_service.create_item_from_barcode(
            user_id=current_user.id,
            barcode_value=barcode_value,
            external_lookup=True
        )
        
        return jsonify({
            'success': True,
            'message': 'Item added to inventory',
            'item': {
                'id': item.id,
                'name': item.item_name,
                'category': item.category
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@barcode_bp.route('/bulk-generate', methods=['POST'])
@login_required
def bulk_generate():
    """Generate barcodes for all items"""
    try:
        results = barcode_service.bulk_generate_barcodes(current_user.id)
        
        flash(f"Generated {results['success']} barcodes successfully", 'success')
        
        return jsonify({
            'success': True,
            'results': results
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@barcode_bp.route('/labels')
@login_required
def print_labels():
    """Generate printable labels"""
    item_ids = request.args.getlist('item_ids', type=int)
    
    if not item_ids:
        # Get all items
        items = InventoryItem.query.filter_by(user_id=current_user.id).all()
        item_ids = [item.id for item in items]
    
    labels = barcode_service.get_printable_labels(item_ids)
    
    # Return JSON for now (frontend to be implemented)
    return jsonify({
        'labels': labels,
        'message': 'Printable labels - frontend to be implemented'
    })
