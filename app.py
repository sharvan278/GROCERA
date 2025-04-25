from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory, jsonify, session
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from typing import List, Dict, Union
from config import Config
import os
import pandas as pd
import numpy as np
import joblib
import re
from datetime import datetime, date
import json
from json import JSONEncoder
import google.generativeai as genai

load_dotenv()

app = Flask(__name__)
app.config.from_object(Config)
app.secret_key = app.config['SECRET_KEY']

# Initialize Gemini AI
try:
    genai.configure(api_key=Config.GEMINI_API_KEY)
    print("Gemini configured successfully")
except Exception as e:
    print(f"Gemini configuration failed: {str(e)}")

class CustomJSONEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

app.json_encoder = CustomJSONEncoder

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

# Load ML models at startup
try:
    price_model = joblib.load('models/price_prediction_model.pkl')
    expiry_model = joblib.load('models/stock_expiry_alert_model.pkl')
    print("Models loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")
    price_model, expiry_model = None, None

# ======================
# UTILITY FUNCTIONS
# ======================

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def clean_price(price):
    try:
        return float(re.sub(r'[^\d.]', '', str(price))) if isinstance(price, str) else float(price)
    except ValueError:
        return np.nan

def parse_date(date_str):
    if pd.isna(date_str):
        return None
    for fmt in ('%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y'):
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    return None

def process_csv(filepath):
    try:
        # Support for CSV and Excel files
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
        elif filepath.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(filepath)
        else:
            print("❌ Unsupported file type.")
            return None

        # Empty file check
        if df.empty:
            print("❌ Uploaded file is empty.")
            return None

        # Clean column names
        print("Original columns:", df.columns.tolist())
        df.columns = [str(col).strip().lower().replace(" ", "_").replace("-", "_") for col in df.columns]
        print("Cleaned columns:", df.columns.tolist())

        # Remove unnamed/index-like columns
        df = df.loc[:, ~df.columns.str.contains('^unnamed')]
        
        return df

    except pd.errors.ParserError:
        print("❌ File format issue. Could not parse the file correctly.")
    except FileNotFoundError:
        print("❌ File not found. Please check the path.")
    except Exception as e:
        print("❌ Unexpected error:", e)
    return None

# ======================
# DATA ANALYSIS FUNCTIONS
# ======================

def get_market_comparison_prices(item_name):
    """Mock function to simulate market price comparison"""
    market_prices = {
        'potatoes': [{'store': 'SuperMart', 'price': 28.00}, {'store': 'FreshGrocer', 'price': 23.50}],
        'salt': [{'store': 'SuperMart', 'price': 22.00}, {'store': 'FreshGrocer', 'price': 19.80}],
        'bananas': [{'store': 'SuperMart', 'price': 42.00}, {'store': 'FreshGrocer', 'price': 38.50}],
        'milk': [{'store': 'SuperMart', 'price': 48.00}, {'store': 'FreshGrocer', 'price': 45.50}],
        'bread': [{'store': 'SuperMart', 'price': 28.00}, {'store': 'FreshGrocer', 'price': 25.50}],
        'eggs': [{'store': 'SuperMart', 'price': 68.00}, {'store': 'FreshGrocer', 'price': 65.50}]
    }
    return market_prices.get(item_name.lower(), [])

def calculate_price_recommendations(df):
    """Calculate optimal pricing recommendations with sales strategy"""
    if df is None or 'item_name' not in df.columns or 'price' not in df.columns:
        return None
    
    recommendations = []
    
    # Price elasticity data
    ELASTICITY_DATA = {
        'potatoes': -1.8,  # Highly elastic (price sensitive)
        'salt': -1.2,      # Less elastic
        'bananas': -1.5,
        'milk': -1.4,
        'bread': -1.7,
        'eggs': -1.3,
        'default': -1.5
    }
    
    for _, row in df.iterrows():
        item_name = row['item_name'].lower()
        
        # Ensure prices are numeric
        try:
            current_price = pd.to_numeric(row['price'], errors='coerce')
            quantity = pd.to_numeric(row.get('quantity', 0), errors='coerce')
        except ValueError:
            current_price = np.nan
            quantity = 0
        
        if np.isnan(current_price):
            continue  # Skip items with invalid prices
        
        # Get market comparison data
        market_prices = get_market_comparison_prices(item_name)
        if not market_prices:
            continue
            
        # Calculate market metrics
        competitor_prices = [p['price'] for p in market_prices]
        avg_competitor_price = np.mean(competitor_prices)
        min_competitor_price = min(competitor_prices)
        
        # Get elasticity for this product
        elasticity = ELASTICITY_DATA.get(item_name, ELASTICITY_DATA['default'])
        
        # Calculate base recommended price
        if elasticity < -1.5:  # Highly elastic goods
            recommended_price = avg_competitor_price * 0.95  # 5% below market
        elif elasticity < -1:  # Moderately elastic
            recommended_price = avg_competitor_price * 0.98  # 2% below market
        else:  # Inelastic goods
            recommended_price = avg_competitor_price * 1.05  # 5% above market
        
        # Adjust for inventory levels
        if quantity > 50:  # Overstock - discount to clear
            recommended_price = min(recommended_price, current_price * 0.85)
        elif quantity < 5:  # Low stock - premium pricing
            recommended_price = max(recommended_price, current_price * 1.15)
        
        # Ensure price stays within reasonable bounds
        recommended_price = max(
            min_competitor_price * 0.8,  # Don't go below 80% of cheapest competitor
            min(recommended_price, min_competitor_price * 1.5)  # Don't exceed 150% of cheapest
        )
        
        # Calculate potential sales impact
        if current_price != 0:
            price_change_pct = (recommended_price - current_price) / current_price * 100
        else:
            price_change_pct = 0
            
        if price_change_pct < 0:
            estimated_sales_change = abs(elasticity * price_change_pct)
            sales_impact = f"Estimated {estimated_sales_change:.1f}% sales increase"
        else:
            estimated_sales_change = elasticity * price_change_pct
            sales_impact = f"Estimated {abs(estimated_sales_change):.1f}% sales decrease"
        
        recommendations.append({
            'item': row['item_name'],
            'current_price': current_price,
            'recommended_price': round(recommended_price, 2),
            'competitor_avg': round(avg_competitor_price, 2),
            'min_competitor': min_competitor_price,
            'price_change_pct': round(price_change_pct, 1),
            'sales_impact': sales_impact,
            'reason': get_recommendation_reason(current_price, recommended_price, quantity, elasticity)
        })
    
    return sorted(recommendations, key=lambda x: abs(x['price_change_pct']), reverse=True)

def get_recommendation_reason(current_price, recommended_price, quantity, elasticity):
    """Generate detailed pricing recommendation reason"""
    price_diff = recommended_price - current_price
    reasons = []
    
    # Price direction reason
    if price_diff < 0:
        reasons.append("price reduction recommended")
    else:
        reasons.append("price increase recommended")
    
    # Elasticity reason
    if elasticity < -1.5:
        reasons.append("highly price-sensitive product")
    elif elasticity < -1:
        reasons.append("moderately price-sensitive product")
    else:
        reasons.append("price-stable product")
    
    # Inventory reason
    if quantity > 50:
        reasons.append("high inventory level")
    elif quantity < 5:
        reasons.append("low inventory level")
    
    # Competitive positioning
    if recommended_price < current_price:
        reasons.append("more competitive pricing")
    else:
        reasons.append("premium pricing justified")
    
    return ". ".join(reasons).capitalize()

def get_price_comparisons(df):
    """Generate price comparison data"""
    if df is None or 'item_name' not in df.columns or 'price' not in df.columns:
        return []

    comparisons = []
    for item in df['item_name'].unique():
        try:
            your_price = float(df.loc[df['item_name'] == item, 'price'].values[0])
        except (ValueError, TypeError, IndexError):
            your_price = 0.0

        market_prices = get_market_comparison_prices(item)
        item_comparisons = []
        
        for price in market_prices:
            try:
                competitor_price = float(price.get('price', 0))
                difference = round((competitor_price - your_price) / your_price * 100, 1) if your_price != 0 else 0
                absolute_diff = round(competitor_price - your_price, 2)
            except (ValueError, TypeError, ZeroDivisionError):
                difference = 0
                absolute_diff = 0

            item_comparisons.append({
                'store': price.get('store', 'Unknown'),
                'price': competitor_price,
                'difference': difference,
                'absolute_diff': absolute_diff
            })

        comparisons.append({
            'product_name': item,
            'your_price': your_price,
            'comparisons': item_comparisons
        })

    return comparisons

def get_inventory_status(df):
    """Generate inventory status"""
    if df is None or 'quantity' not in df.columns:
        return []
    
    inventory = []
    for _, row in df.iterrows():
        quantity = row['quantity']
        if quantity < 3:
            status = 'critical'
        elif quantity < 10:
            status = 'low'
        else:
            status = 'adequate'
        
        inventory.append({
            'name': row.get('item_name', 'Unknown'),
            'quantity': quantity,
            'unit': row.get('unit', 'units'),
            'unit_price': row.get('price', 0),
            'total_value': row.get('total_value', 0),
            'stock_level': status,
            'progress': min(100, quantity)
        })
    
    return inventory

def get_stock_alerts(df):
    """Generate stock alerts"""
    if df is None:
        return []
    
    alerts = []
    today = datetime.now()
    
    for _, row in df.iterrows():
        if 'quantity' in df.columns and row['quantity'] <= app.config['STOCK_ALERT_THRESHOLD']:
            alerts.append({
                'type': 'low_stock',
                'item': row.get('item_name', 'Unknown'),
                'current_stock': row['quantity'],
                'reorder_point': app.config['STOCK_ALERT_THRESHOLD'],
                'unit': row.get('unit', 'units'),
                'severity': 'high' if row['quantity'] < 2 else 'medium'
            })
        
        if 'expiry_date' in df.columns and pd.notna(row['expiry_date']):
            try:
                expiry_date = datetime.strptime(row['expiry_date'], '%Y-%m-%d')
                days_until_expiry = (expiry_date - today).days
                if days_until_expiry <= 7:
                    alerts.append({
                        'type': 'expiry',
                        'item': row.get('item_name', 'Unknown'),
                        'expiry_date': row['expiry_date'],
                        'days_until_expiry': days_until_expiry,
                        'quantity': row.get('quantity', 0),
                        'unit': row.get('unit', 'units'),
                        'severity': 'high' if days_until_expiry <= 3 else 'medium'
                    })
            except ValueError:
                continue
    
    return alerts

def get_restock_recommendations(df):
    """Generate inventory restocking recommendations"""
    if df is None or 'quantity' not in df.columns:
        return []
    
    recommendations = []
    for _, row in df.iterrows():
        quantity = row['quantity']
        if quantity < 5:
            recommendations.append({
                'type': 'restock',
                'item': row.get('item_name', 'Unknown'),
                'current_stock': quantity,
                'suggested_order': max(10 - quantity, 5),
                'unit': row.get('unit', 'units'),
                'priority': 'high' if quantity < 2 else 'medium'
            })
    
    return recommendations

def get_expiry_recommendations(df):
    """Generate expiry recommendations"""
    if df is None or 'expiry_date' not in df.columns:
        return []
    
    recommendations = []
    today = datetime.now()
    
    for _, row in df.iterrows():
        if pd.notna(row['expiry_date']):
            try:
                expiry_date = datetime.strptime(row['expiry_date'], '%Y-%m-%d')
                days_left = (expiry_date - today).days
                if days_left <= 7:
                    action = "discount" if days_left <= 3 else "promote"
                    recommendations.append({
                        'type': 'expiry',
                        'item': row.get('item_name', 'Unknown'),
                        'expiry_date': row['expiry_date'],
                        'days_left': days_left,
                        'quantity': row.get('quantity', 0),
                        'suggested_action': action,
                        'discount_pct': min(30, (8 - days_left) * 5)
                    })
            except ValueError:
                continue
    
    return recommendations

def get_trending_items(df):
    """Calculate trending items"""
    if df is None or df.empty:
        return []

    # Ensure required columns exist
    for col in ['item_name', 'price', 'quantity']:
        if col not in df.columns:
            df[col] = 0

    df['price'] = pd.to_numeric(df['price'], errors='coerce').fillna(0)
    df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce').fillna(0)
    df['velocity'] = df['quantity'] / (df['price'] + 0.001)
    df['total_value'] = df['price'] * df['quantity']

    trending = df.sort_values(by=['velocity', 'total_value'], ascending=[False, False])
    return trending[['item_name', 'price', 'quantity', 'total_value']].to_dict('records')

# ======================
# AI FUNCTIONS
# ======================

FALLBACK_BUNDLES = [
    {"items": {"bread", "eggs", "milk"}, "name": "Breakfast Essentials", "desc": "Complete breakfast package"},
    {"items": {"bread", "peanut butter"}, "name": "Protein Snack", "desc": "Energy-boosting combination"},
    {"items": {"bread", "nutella", "bananas"}, "name": "Sweet Breakfast", "desc": "Delicious morning treat"},
    {"items": {"bananas", "milk", "peanut butter"}, "name": "Smoothie Pack", "desc": "Perfect for protein shakes"},
]

def generate_ai_bundles(inventory_items: List[str]) -> List[Dict[str, Union[str, set]]]:
    """Generate product bundles using Gemini API"""
    prompt = f"""
    As a retail product bundling expert, suggest 5-7 product bundles that would appeal to customers 
    based on these available items: {', '.join(inventory_items)}.
    
    For each bundle:
    1. Combine 2-4 complementary products that are commonly purchased together
    2. Give the bundle a short, creative name (max 3 words)
    3. Provide a one-sentence description (max 15 words)
    
    Format each bundle exactly as: "name|description|item1,item2,item3"
    """
    
    try:
        model = genai.GenerativeModel('gemini-1.5-pro-latest')
        response = model.generate_content(prompt)
        
        bundles = []
        for line in response.text.split('\n'):
            if '|' in line and line.count('|') == 2:
                try:
                    name, desc, items = line.split('|')
                    bundles.append({
                        "name": name.strip(),
                        "desc": desc.strip(),
                        "items": set(item.strip().lower() for item in items.split(','))
                    })
                except:
                    continue
        return bundles
    
    except Exception as e:
        print(f"AI generation failed: {e}")
        return []

def generate_smart_recommendations(df: pd.DataFrame) -> List[str]:
    """Generate smart recommendations with AI-powered bundling"""
    if df is None or df.empty:
        return ["No inventory data available"]
    
    # Standardize column names
    df.columns = [str(col).lower().strip().replace(" ", "_") for col in df.columns]
    item_col = next((col for col in df.columns if col in ['item_name', 'item', 'product', 'product_name', 'name']), None)
    
    if not item_col:
        return ["No valid item name column found"]
    
    # Get inventory items
    inventory_items = list(df[item_col].astype(str).str.lower().unique())
    
    # Generate AI-powered bundles
    ai_bundles = generate_ai_bundles(inventory_items)
    all_bundles = ai_bundles + FALLBACK_BUNDLES
    
    # Find matching bundles
    output = []
    complete_bundles = []
    partial_bundles = []
    
    for bundle in all_bundles:
        matches = bundle["items"].intersection(inventory_items)
        match_count = len(matches)
        
        if match_count >= 2:
            if matches == bundle["items"]:
                complete_bundles.append(f"{bundle['name']}|{bundle['desc']}")
            else:
                missing = bundle["items"] - matches
                partial_bundles.append(
                    f"{bundle['name']} (Partial)|"
                    f"Contains: {', '.join(matches)}. "
                    f"Add {', '.join(missing)} to complete!"
                )
    
    # Organize output
    if complete_bundles:
        output.append("--- Complete Bundles ---")
        output.extend(complete_bundles)
    
    if partial_bundles:
        output.append("--- Suggested Bundles (Add Items to Complete) ---")
        output.extend(partial_bundles)
    
    if not output:
        output.append("No bundle matches found|Try adding more inventory variety")
    
    return output

def generate_ai_response(user_message, data):
    """Generate response using either predefined patterns or Gemini API"""
    # Try predefined patterns first
    response = generate_predefined_response(user_message, data)
    if response:
        return response
    
    # Fall back to Gemini
    try:
        model = genai.GenerativeModel('gemini-1.5-pro-latest')
        response = model.generate_content(
            f"You're a grocery inventory assistant. Current inventory:\n"
            f"{data.to_string()}\n\n"
            f"User question: {user_message}\n"
            f"Answer concisely and professionally:"
        )
        return response.text
    except Exception as e:
        print(f"Gemini Error: {str(e)}")
        return "I can't answer that right now. Please try asking about inventory levels or product prices."

def generate_predefined_response(message, data):
    """Handle common questions without API calls"""
    message = message.lower().strip()
    
    # Inventory summary
    if any(q in message for q in ['total inventory', 'how many products']):
        return f"We have {len(data)} products in inventory worth ${data['price'].sum():.2f}"
    
    # Stock level check
    stock_match = re.search(r'stock (?:level|amount) of (.+)|how much (.+) do we have', message)
    if stock_match:
        product = (stock_match.group(1) or stock_match.group(2)).strip()
        return get_product_info(product, data, 'quantity', 'units')
    
    # Price check
    price_match = re.search(r'price of (.+)|how much is (.+)', message)
    if price_match:
        product = (price_match.group(1) or price_match.group(2)).strip()
        return get_product_info(product, data, 'price', '$')
    
    return None

def get_product_info(product_name, data, field, unit):
    """Generic function to get product information"""
    matches = data[data['item_name'].str.contains(product_name, case=False)]
    if matches.empty:
        return f"Couldn't find '{product_name}' in inventory"
    
    return "\n".join(
        f"{row['item_name']}: {unit}{row[field]}" if unit == '$' else f"{row['item_name']}: {row[field]} {unit}"
        for _, row in matches.iterrows()
    )

# ======================
# FLASK ROUTES
# ======================

@app.route('/')
def index():
    """Home page"""
    has_data = session.get('has_data', False)
    return render_template('index.html', has_data=has_data)

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    """Handle file uploads and preview"""
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('⚠️ No file part in request.', 'error')
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            flash('⚠️ No file selected.', 'error')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            processed_data = process_csv(filepath)
            if processed_data is not None and not processed_data.empty:
                session['processed_data'] = processed_data.to_dict('records')
                session['has_data'] = True
                session['filename'] = filename
                flash(f'✅ {filename} uploaded and processed successfully!', 'success')
                return redirect(url_for('upload_file'))
            else:
                flash('❌ Failed to process CSV or file is empty.', 'error')
                return redirect(url_for('upload_file'))
        else:
            flash('❌ Invalid file format. Please upload a CSV.', 'error')
            return redirect(request.url)

    # GET Request
    has_data = session.get('has_data', False)
    filename = session.get('filename', None)
    processed_data = pd.DataFrame(session.get('processed_data', [])) if has_data else pd.DataFrame()

    return render_template("upload.html",
        has_data=has_data,
        filename=filename,
        price_data=get_price_comparisons(processed_data),
        inventory=get_inventory_status(processed_data),
        stock_alerts=get_stock_alerts(processed_data),
        expiry_recommendations=get_expiry_recommendations(processed_data),
        trending_items=get_trending_items(processed_data),
        smart_recommendations=generate_smart_recommendations(processed_data)
    )

@app.route('/dashboard')
def dashboard():
    """Main dashboard view"""
    has_data = session.get('has_data', False)
    processed_data = pd.DataFrame(session.get('processed_data', [])) if has_data else pd.DataFrame()
    
    return render_template('dashboard.html',
                         has_data=has_data,
                         price_data=get_price_comparisons(processed_data) if has_data else [],
                         inventory=get_inventory_status(processed_data) if has_data else [],
                         alerts=get_stock_alerts(processed_data) if has_data else [],
                         recommendations=get_restock_recommendations(processed_data) if has_data else [])

@app.route('/price_comparisons')
def price_comparisons():
    """Detailed price comparisons"""
    if not session.get('has_data', False):
        flash('Please upload a CSV file first', 'error')
        return redirect(url_for('upload_file'))
    
    processed_data = pd.DataFrame(session.get('processed_data', []))
    return render_template('price_comparison.html', 
                         price_data=get_price_comparisons(processed_data),
                         price_recommendations=calculate_price_recommendations(processed_data))

@app.route('/stock_tracker')
def stock_tracker():
    """Inventory tracking view"""
    if not session.get('has_data', False):
        flash('Please upload a CSV file first', 'error')
        return redirect(url_for('upload_file'))
    
    processed_data = pd.DataFrame(session.get('processed_data', []))
    return render_template('stock_tracker.html', 
                         inventory=get_inventory_status(processed_data),
                         restock_recommendations=get_restock_recommendations(processed_data))

@app.route('/recommendations')
def recommendations():
    """Combined recommendations view"""
    if not session.get('has_data', False):
        flash('Please upload a CSV file first', 'error')
        return redirect(url_for('upload_file'))
    
    processed_data = pd.DataFrame(session.get('processed_data', []))
    
    return render_template('recommendations.html',
        price_recommendations=calculate_price_recommendations(processed_data),
        restock_recommendations=get_restock_recommendations(processed_data),
        expiry_recommendations=get_expiry_recommendations(processed_data),
        trending_items=get_trending_items(processed_data)
    )

@app.route('/stock_alerts')
def stock_alerts():
    """Stock alerts view"""
    if not session.get('has_data', False):
        flash('Please upload a CSV file first', 'error')
        return redirect(url_for('upload_file'))
    
    processed_data = pd.DataFrame(session.get('processed_data', []))
    alerts = get_stock_alerts(processed_data)
    return render_template('stock_alerts.html',
                         alerts=alerts,
                         expiry_recommendations=get_expiry_recommendations(processed_data))

@app.route('/pricing_recommendations')
def pricing_recommendations():
    """Pricing strategy recommendations"""
    if not session.get('has_data', False):
        flash('Please upload a CSV file first', 'error')
        return redirect(url_for('upload_file'))
    
    processed_data = pd.DataFrame(session.get('processed_data', []))
    return render_template('pricing_recommendations.html',
                         recommendations=calculate_price_recommendations(processed_data),
                         price_data=get_price_comparisons(processed_data))

@app.route('/trending')
def trending_items():
    """Trending products view"""
    if not session.get('has_data', False):
        flash('Please upload a CSV file first', 'error')
        return redirect(url_for('upload_file'))
    
    processed_data = pd.DataFrame(session.get('processed_data', []))
    return render_template('trending_items.html',
                         trending_items=get_trending_items(processed_data))

@app.route('/shopping_assistant', methods=['GET', 'POST'])
def shopping_assistant():
    """AI Shopping Assistant"""
    if not session.get('has_data', False):
        flash('Please upload inventory data first', 'error')
        return redirect(url_for('upload_file'))
    
    processed_data = pd.DataFrame(session.get('processed_data', []))
    
    if request.method == 'POST':
        user_message = request.form.get('message', '').strip()
        if not user_message:
            return jsonify({'error': 'Empty message'}), 400
        
        try:
            response = generate_ai_response(user_message, processed_data)
            
            # Store in chat history
            session.setdefault('chat_history', []).append({
                'user': user_message,
                'ai': response
            })
            session.modified = True
            
            return jsonify({'response': response})
            
        except Exception as e:
            error_msg = f"⚠️ System error: {str(e)}" if app.debug else "Sorry, I'm having trouble responding"
            return jsonify({'error': error_msg}), 500
    
    return render_template('shopping_assistant.html',
                         chat_history=session.get('chat_history', []))

@app.route('/download-report')
def download_report():
    """Generate downloadable report"""
    if not session.get('has_data', False):
        flash('No data available to download', 'error')
        return redirect(url_for('index'))
    
    processed_data = pd.DataFrame(session.get('processed_data', []))
    
    report_data = {
        'price_comparisons': get_price_comparisons(processed_data),
        'inventory_status': get_inventory_status(processed_data),
        'stock_alerts': get_stock_alerts(processed_data),
        'pricing_recommendations': calculate_price_recommendations(processed_data),
        'restock_recommendations': get_restock_recommendations(processed_data),
        'expiry_recommendations': get_expiry_recommendations(processed_data),
        'trending_items': get_trending_items(processed_data),
        'generated_at': datetime.now().isoformat()
    }
    
    report_filename = f"grocery_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report_path = os.path.join(app.config['PROCESSED_FOLDER'], report_filename)
    
    with open(report_path, 'w') as f:
        json.dump(report_data, f, indent=2, cls=CustomJSONEncoder)
    
    return send_from_directory(app.config['PROCESSED_FOLDER'], report_filename, as_attachment=True)

@app.route('/download/<filename>')
def download_file(filename):
    """Download processed files"""
    return send_from_directory(
        app.config['PROCESSED_FOLDER'],
        filename,
        as_attachment=True
    )

@app.route("/clear_data", methods=["POST"])
def clear_data():
    """Clear session data"""
    session.clear()
    flash("Data has been cleared. Please upload a new CSV.")
    return redirect(url_for("upload_file"))

if __name__ == '__main__':
    app.run(debug=True)