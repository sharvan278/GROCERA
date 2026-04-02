"""
GROCERA Flask Application
Multi-tenant marketplace connecting households with kirana stores
Clean, modular architecture following SOLID principles
"""
from flask import Flask
from src.config import Config
from src.models.models import db
from src.models.multi_tenant import Store, StoreCustomer, Order, OrderItem, KhataLedger
from flask_login import LoginManager
from flask_migrate import Migrate

# Initialize Flask app
app = Flask(__name__)
app.config.from_object(Config)
Config.ensure_directories()

# Initialize extensions
db.init_app(app)
migrate = Migrate(app, db)

# Initialize authentication
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'auth.login'
login_manager.login_message = 'Please log in to access this page.'

@login_manager.user_loader
def load_user(user_id):
    from src.models.models import User
    return db.session.get(User, int(user_id))

# Register blueprints
from src.routes.auth_routes import auth_bp
from src.routes.main_routes import main_bp, api_bp
from src.routes.payment_routes import payment_bp
from src.routes.competitor_routes import competitor_bp
from src.routes.barcode_routes import barcode_bp
from src.routes.ai_routes import ai_bp

app.register_blueprint(auth_bp)
app.register_blueprint(main_bp)
app.register_blueprint(api_bp, url_prefix='/api/v1')
app.register_blueprint(payment_bp)
app.register_blueprint(competitor_bp)
app.register_blueprint(barcode_bp)
app.register_blueprint(ai_bp)


@app.get('/healthz')
def healthz():
    """Health endpoint for deployment checks."""
    from flask import jsonify
    try:
        db.session.execute(db.text('SELECT 1'))
        return jsonify({'status': 'ok', 'database': 'connected'}), 200
    except Exception as exc:
        return jsonify({'status': 'degraded', 'database': 'unavailable', 'error': str(exc)}), 503

# Error handlers
@app.errorhandler(404)
def not_found(error):
    from flask import render_template
    return render_template('errors/404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    from flask import render_template
    db.session.rollback()
    return render_template('errors/500.html'), 500

if __name__ == '__main__':
    app.run(debug=Config.DEBUG)
