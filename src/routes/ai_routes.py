"""
Advanced AI Routes Blueprint

Handles demand forecasting, dynamic pricing, expiry detection, and recommendations.
"""
from flask import Blueprint, request, jsonify
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
from src.services import (
    demand_forecasting_service,
    dynamic_pricing_engine,
    expiry_detection_service,
    persona_recommendation_service
)
import os

ai_bp = Blueprint('ai', __name__, url_prefix='/ai')


# ============= DEMAND FORECASTING =============

@ai_bp.route('/demand/train', methods=['POST'])
@login_required
def train_demand_model():
    """Train demand forecasting model"""
    result = demand_forecasting_service.train_model(current_user.id)
    return jsonify(result)


@ai_bp.route('/demand/predict/<int:item_id>')
@login_required
def predict_demand(item_id):
    """Predict demand for an item"""
    days = request.args.get('days', 7, type=int)
    result = demand_forecasting_service.predict_demand(item_id, days)
    return jsonify(result)


@ai_bp.route('/demand/alerts')
@login_required
def demand_alerts():
    """Get low stock alerts based on predictions"""
    alerts = demand_forecasting_service.get_low_stock_alerts(current_user.id)
    return jsonify({
        'success': True,
        'alerts': alerts,
        'count': len(alerts)
    })


# ============= DYNAMIC PRICING =============

@ai_bp.route('/pricing/train', methods=['POST'])
@login_required
def train_pricing_model():
    """Train Q-learning pricing model"""
    result = dynamic_pricing_engine.train_from_history(current_user.id)
    return jsonify(result)


@ai_bp.route('/pricing/suggest/<int:item_id>')
@login_required
def suggest_price(item_id):
    """Get pricing suggestion for an item"""
    result = dynamic_pricing_engine.suggest_price(item_id)
    return jsonify(result)


@ai_bp.route('/pricing/apply', methods=['POST'])
@login_required
def apply_dynamic_pricing():
    """Apply dynamic pricing to all items"""
    auto_apply = request.args.get('auto', 'false').lower() == 'true'
    result = dynamic_pricing_engine.apply_dynamic_pricing(current_user.id, auto_apply)
    return jsonify(result)


# ============= EXPIRY DETECTION =============

@ai_bp.route('/expiry/detect', methods=['POST'])
@login_required
def detect_expiry():
    """Detect expiry date from uploaded image"""
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'No image provided'}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'}), 400
    
    # Save uploaded file
    filename = secure_filename(file.filename)
    temp_path = os.path.join('uploads', 'temp', filename)
    os.makedirs(os.path.dirname(temp_path), exist_ok=True)
    file.save(temp_path)
    
    # Detect expiry
    result = expiry_detection_service.detect_expiry_from_image(temp_path)
    
    # Clean up temp file
    try:
        os.remove(temp_path)
    except:
        pass
    
    return jsonify(result)


@ai_bp.route('/expiry/freshness', methods=['POST'])
@login_required
def analyze_freshness():
    """Analyze product freshness from image"""
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'No image provided'}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'}), 400
    
    # Save uploaded file
    filename = secure_filename(file.filename)
    temp_path = os.path.join('uploads', 'temp', filename)
    os.makedirs(os.path.dirname(temp_path), exist_ok=True)
    file.save(temp_path)
    
    # Analyze freshness
    result = expiry_detection_service.analyze_product_freshness(temp_path)
    
    # Clean up
    try:
        os.remove(temp_path)
    except:
        pass
    
    return jsonify(result)


@ai_bp.route('/expiry/batch', methods=['POST'])
@login_required
def batch_detect_expiry():
    """Batch detect expiry dates from multiple images"""
    image_dir = request.json.get('image_dir', 'uploads/products')
    results = expiry_detection_service.batch_detect_expiry(image_dir)
    
    return jsonify({
        'success': True,
        'results': results,
        'total': len(results)
    })


# ============= PERSONA RECOMMENDATIONS =============

@ai_bp.route('/persona/identify')
@login_required
def identify_persona():
    """Identify user persona"""
    result = persona_recommendation_service.identify_persona(current_user.id)
    return jsonify(result)


@ai_bp.route('/recommendations')
@login_required
def recommendations_page():
    """Render recommendations page"""
    from flask import render_template
    return render_template('recommendations_page.html')


@ai_bp.route('/recommendations/data')
@login_required
def get_recommendations():
    """Get personalized recommendations"""
    limit = request.args.get('limit', 10, type=int)
    result = persona_recommendation_service.recommend_products(current_user.id, limit)
    return jsonify(result)


@ai_bp.route('/recommendations/cross-sell/<int:item_id>')
@login_required
def cross_sell(item_id):
    """Get cross-sell recommendations"""
    limit = request.args.get('limit', 5, type=int)
    result = persona_recommendation_service.get_cross_sell_recommendations(item_id, limit)
    return jsonify(result)


@ai_bp.route('/recommendations/trending/<int:persona_id>')
@login_required
def trending_for_persona(persona_id):
    """Get trending items for persona"""
    limit = request.args.get('limit', 10, type=int)
    result = persona_recommendation_service.get_trending_for_persona(persona_id, limit)
    return jsonify(result)


# ============= COMBINED INSIGHTS =============

@ai_bp.route('/insights/<int:item_id>')
@login_required
def get_item_insights(item_id):
    """Get comprehensive AI insights for an item"""
    from src.models.models import InventoryItem
    
    item = InventoryItem.query.filter_by(id=item_id, user_id=current_user.id).first_or_404()
    
    # Get all insights
    demand = demand_forecasting_service.predict_demand(item_id, 7)
    pricing = dynamic_pricing_engine.suggest_price(item_id)
    cross_sell = persona_recommendation_service.get_cross_sell_recommendations(item_id, 3)
    
    return jsonify({
        'success': True,
        'item': {
            'id': item.id,
            'name': item.item_name,
            'category': item.category,
            'price': item.price,
            'quantity': item.quantity
        },
        'demand_forecast': demand if demand['success'] else None,
        'pricing_suggestion': pricing if pricing['success'] else None,
        'cross_sell': cross_sell['recommendations'] if cross_sell['success'] else []
    })


@ai_bp.route('/dashboard')
@login_required
def ai_dashboard():
    """Render AI dashboard page"""
    from flask import render_template
    return render_template('ai_dashboard.html')


@ai_bp.route('/dashboard/data')
@login_required
def ai_dashboard_data():
    """Get AI dashboard with all insights"""
    # Get persona
    persona = persona_recommendation_service.identify_persona(current_user.id)
    
    # Get demand alerts
    demand_alerts = demand_forecasting_service.get_low_stock_alerts(current_user.id)
    
    # Get recommendations
    recommendations = persona_recommendation_service.recommend_products(current_user.id, 5)
    
    return jsonify({
        'success': True,
        'persona': persona,
        'demand_alerts': demand_alerts[:5],
        'recommendations': recommendations['recommendations'] if recommendations['success'] else [],
        'message': 'AI Dashboard - all advanced features active'
    })


@ai_bp.route('/trending')
@login_required
def trending_items():
    """Get trending items for user's persona"""
    trending = persona_recommendation_service.get_trending_for_persona(current_user.id, limit=10)
    return jsonify(trending)
