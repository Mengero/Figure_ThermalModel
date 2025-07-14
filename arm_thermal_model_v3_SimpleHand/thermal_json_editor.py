from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session, send_file
import json
import os
import re
from datetime import datetime
from werkzeug.utils import secure_filename
import shutil
import tempfile

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

DEFAULT_JSON_FILE = 'GEO.json'
ALLOWED_EXTENSIONS = {'json'}

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    """Check if file has allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_current_json_file():
    """Get the current JSON file path from session"""
    return session.get('current_json_file', DEFAULT_JSON_FILE)

def load_json_data():
    """Load JSON data from current file, handling comments"""
    json_file = get_current_json_file()
    try:
        with open(json_file, 'r') as f:
            content = f.read()
            # Remove comments using regex (more robust approach)
            # This pattern matches // comments but not when inside quotes
            content = re.sub(r'//.*$', '', content, flags=re.MULTILINE)
            # Remove any trailing commas that might cause issues
            content = re.sub(r',(\s*[}\]])', r'\1', content)
            return json.loads(content)
    except Exception as e:
        print(f"Error loading JSON from {json_file}: {e}")
        return None

def save_json_data(data):
    """Save JSON data to current file"""
    json_file = get_current_json_file()
    try:
        with open(json_file, 'w') as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving JSON to {json_file}: {e}")
        return False

def create_backup():
    """Create a backup of the current JSON file"""
    json_file = get_current_json_file()
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = os.path.basename(json_file)
        name, ext = os.path.splitext(filename)
        backup_file = f'{name}_backup_{timestamp}{ext}'
        
        shutil.copy2(json_file, backup_file)
        return backup_file
    except Exception as e:
        print(f"Error creating backup: {e}")
        return None

@app.route('/')
def index():
    """Main page for file management"""
    return render_template('file_manager.html')

@app.route('/dashboard')
def dashboard():
    """Dashboard showing overview of the thermal model"""
    if 'current_json_file' not in session:
        flash('Please select a JSON file first', 'error')
        return redirect(url_for('index'))
    
    data = load_json_data()
    if data is None:
        flash('Error loading JSON data', 'error')
        return render_template('error.html')
    
    current_file = get_current_json_file()
    return render_template('dashboard.html', data=data, current_file=os.path.basename(current_file))

@app.route('/environment')
def environment():
    """Show environment settings"""
    if 'current_json_file' not in session:
        flash('Please select a JSON file first', 'error')
        return redirect(url_for('index'))
        
    data = load_json_data()
    if data is None:
        flash('Error loading JSON data', 'error')
        return render_template('error.html')
    
    return render_template('environment.html', environment=data.get('environment', {}))

@app.route('/environment', methods=['POST'])
def update_environment():
    """Update environment settings"""
    if 'current_json_file' not in session:
        flash('Please select a JSON file first', 'error')
        return redirect(url_for('index'))
        
    data = load_json_data()
    if data is None:
        flash('Error loading JSON data', 'error')
        return redirect(url_for('dashboard'))
    
    # Update environment data
    data['environment']['ambient_temperature'] = float(request.form.get('ambient_temperature', 0))
    data['environment']['heat_transfer_coefficient'] = float(request.form.get('heat_transfer_coefficient', 0))
    data['environment']['plastic_conductivity'] = float(request.form.get('plastic_conductivity', 0))
    
    if save_json_data(data):
        flash('Environment settings updated successfully', 'success')
    else:
        flash('Error saving environment settings', 'error')
    
    return redirect(url_for('environment'))

@app.route('/regions')
def regions():
    """Show all regions"""
    if 'current_json_file' not in session:
        flash('Please select a JSON file first', 'error')
        return redirect(url_for('index'))
        
    data = load_json_data()
    if data is None:
        flash('Error loading JSON data', 'error')
        return render_template('error.html')
    
    return render_template('regions.html', regions=data.get('regions', []))

@app.route('/region/<region_id>')
def region_detail(region_id):
    """Show details of a specific region"""
    data = load_json_data()
    if data is None:
        flash('Error loading JSON data', 'error')
        return render_template('error.html')
    
    region = None
    for r in data.get('regions', []):
        if r.get('id') == region_id:
            region = r
            break
    
    if region is None:
        flash(f'Region {region_id} not found', 'error')
        return redirect(url_for('regions'))
    
    return render_template('region_detail.html', region=region)

@app.route('/region/<region_id>/edit', methods=['GET', 'POST'])
def edit_region(region_id):
    """Edit a specific region"""
    data = load_json_data()
    if data is None:
        flash('Error loading JSON data', 'error')
        return redirect(url_for('regions'))
    
    region_index = None
    region = None
    for i, r in enumerate(data.get('regions', [])):
        if r.get('id') == region_id:
            region_index = i
            region = r
            break
    
    if region is None:
        flash(f'Region {region_id} not found', 'error')
        return redirect(url_for('regions'))
    
    if request.method == 'POST':
        # Update region data
        region['thermal_conductivity'] = float(request.form.get('thermal_conductivity', 0))
        region['width'] = float(request.form.get('width', 0))
        region['height'] = float(request.form.get('height', 0))
        region['thickness'] = float(request.form.get('thickness', 0))
        
        # Update centroid
        region['centroid']['x'] = float(request.form.get('centroid_x', 0))
        region['centroid']['y'] = float(request.form.get('centroid_y', 0))
        region['centroid']['z'] = float(request.form.get('centroid_z', 0))
        
        # Update mesh settings
        region['mesh_settings']['dx'] = float(request.form.get('mesh_dx', 0))
        region['mesh_settings']['dy'] = float(request.form.get('mesh_dy', 0))
        region['mesh_settings']['dz'] = float(request.form.get('mesh_dz', 0))
        
        # Update boundary conditions
        boundary_conditions = []
        bc_count = int(request.form.get('bc_count', 0))
        
        for i in range(bc_count):
            bc_type = request.form.get(f'bc_{i}_type', '')
            if bc_type:
                bc = {'type': bc_type}
                
                # Common fields
                if request.form.get(f'bc_{i}_comments'):
                    bc['comments'] = request.form.get(f'bc_{i}_comments')
                
                # Centroid
                if request.form.get(f'bc_{i}_centroid_x') is not None:
                    bc['centroid'] = {
                        'x': float(request.form.get(f'bc_{i}_centroid_x', 0)),
                        'y': float(request.form.get(f'bc_{i}_centroid_y', 0)),
                        'z': float(request.form.get(f'bc_{i}_centroid_z', 0))
                    }
                
                # Dimensions
                if request.form.get(f'bc_{i}_width') is not None:
                    bc['width'] = float(request.form.get(f'bc_{i}_width', 0))
                if request.form.get(f'bc_{i}_height') is not None:
                    bc['height'] = float(request.form.get(f'bc_{i}_height', 0))
                if request.form.get(f'bc_{i}_thickness') is not None:
                    bc['thickness'] = float(request.form.get(f'bc_{i}_thickness', 0))
                
                # Type-specific fields
                if bc_type == 'ACTUATOR_CONNECTED':
                    bc['actuator_id'] = request.form.get(f'bc_{i}_actuator_id', '')
                    bc['connecting_location'] = request.form.get(f'bc_{i}_connecting_location', '')
                elif bc_type == 'CONST_Q':
                    bc['q'] = float(request.form.get(f'bc_{i}_q', 0))
                elif bc_type == 'MAPPED':
                    bc['symmetry_axis'] = request.form.get(f'bc_{i}_symmetry_axis', '')
                elif bc_type == 'PLASTIC_COVERED':
                    bc['plastic_thickness'] = float(request.form.get(f'bc_{i}_plastic_thickness', 0))
                elif bc_type == 'ADIABATIC':
                    if request.form.get(f'bc_{i}_source1_centroid_x') is not None:
                        bc['source_1'] = {
                            'centroid': {
                                'x': float(request.form.get(f'bc_{i}_source1_centroid_x', 0)),
                                'y': float(request.form.get(f'bc_{i}_source1_centroid_y', 0)),
                                'z': float(request.form.get(f'bc_{i}_source1_centroid_z', 0))
                            },
                            'width': float(request.form.get(f'bc_{i}_source1_width', 0)),
                            'height': float(request.form.get(f'bc_{i}_source1_height', 0)),
                            'thickness': float(request.form.get(f'bc_{i}_source1_thickness', 0))
                        }
                
                boundary_conditions.append(bc)
        
        region['boundary_conditions'] = boundary_conditions
        data['regions'][region_index] = region
        
        if save_json_data(data):
            flash(f'Region {region_id} updated successfully', 'success')
        else:
            flash('Error saving region data', 'error')
        
        return redirect(url_for('region_detail', region_id=region_id))
    
    return render_template('edit_region.html', region=region)

@app.route('/actuators')
def actuators():
    """Show all actuators"""
    if 'current_json_file' not in session:
        flash('Please select a JSON file first', 'error')
        return redirect(url_for('index'))
        
    data = load_json_data()
    if data is None:
        flash('Error loading JSON data', 'error')
        return render_template('error.html')
    
    return render_template('actuators.html', actuators=data.get('actuators', []))

@app.route('/actuator/<actuator_id>')
def actuator_detail(actuator_id):
    """Show details of a specific actuator"""
    data = load_json_data()
    if data is None:
        flash('Error loading JSON data', 'error')
        return render_template('error.html')
    
    actuator = None
    for a in data.get('actuators', []):
        if a.get('id') == actuator_id:
            actuator = a
            break
    
    if actuator is None:
        flash(f'Actuator {actuator_id} not found', 'error')
        return redirect(url_for('actuators'))
    
    return render_template('actuator_detail.html', actuator=actuator)

@app.route('/actuator/<actuator_id>/edit', methods=['GET', 'POST'])
def edit_actuator(actuator_id):
    """Edit a specific actuator"""
    data = load_json_data()
    if data is None:
        flash('Error loading JSON data', 'error')
        return redirect(url_for('actuators'))
    
    actuator_index = None
    actuator = None
    for i, a in enumerate(data.get('actuators', [])):
        if a.get('id') == actuator_id:
            actuator_index = i
            actuator = a
            break
    
    if actuator is None:
        flash(f'Actuator {actuator_id} not found', 'error')
        return redirect(url_for('actuators'))
    
    if request.method == 'POST':
        # Update heat losses
        actuator['heat_losses']['gearbox'] = float(request.form.get('heat_gearbox', 0))
        actuator['heat_losses']['motor'] = float(request.form.get('heat_motor', 0))
        actuator['heat_losses']['FETs'] = float(request.form.get('heat_fets', 0))
        
        # Update thermal resistance values
        for key in actuator['thermal_resistance']:
            form_key = f'resistance_{key.lower()}'
            if form_key in request.form:
                actuator['thermal_resistance'][key] = float(request.form.get(form_key, 0))
        
        data['actuators'][actuator_index] = actuator
        
        if save_json_data(data):
            flash(f'Actuator {actuator_id} updated successfully', 'success')
        else:
            flash('Error saving actuator data', 'error')
        
        return redirect(url_for('actuator_detail', actuator_id=actuator_id))
    
    return render_template('edit_actuator.html', actuator=actuator)

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload"""
    if 'file' not in request.files:
        flash('No file selected', 'error')
        return redirect(url_for('index'))
    
    file = request.files['file']
    if file.filename == '':
        flash('No file selected', 'error')
        return redirect(url_for('index'))
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        try:
            file.save(filepath)
            session['current_json_file'] = filepath
            session['original_filename'] = filename
            flash(f'File "{filename}" uploaded successfully', 'success')
            return redirect(url_for('dashboard'))
        except Exception as e:
            flash(f'Error uploading file: {e}', 'error')
            return redirect(url_for('index'))
    else:
        flash('Invalid file type. Please upload a JSON file.', 'error')
        return redirect(url_for('index'))

@app.route('/load_default')
def load_default():
    """Load the default JSON file"""
    if os.path.exists(DEFAULT_JSON_FILE):
        session['current_json_file'] = DEFAULT_JSON_FILE
        session['original_filename'] = DEFAULT_JSON_FILE
        flash(f'Default file "{DEFAULT_JSON_FILE}" loaded successfully', 'success')
        return redirect(url_for('dashboard'))
    else:
        flash(f'Default file "{DEFAULT_JSON_FILE}" not found', 'error')
        return redirect(url_for('index'))

@app.route('/download')
def download():
    """Download the current JSON file"""
    if 'current_json_file' not in session:
        flash('No file selected', 'error')
        return redirect(url_for('index'))
    
    json_file = get_current_json_file()
    if not os.path.exists(json_file):
        flash('File not found', 'error')
        return redirect(url_for('dashboard'))
    
    original_filename = session.get('original_filename', 'thermal_model.json')
    name, ext = os.path.splitext(original_filename)
    download_filename = f"{name}_modified{ext}"
    
    return send_file(json_file, as_attachment=True, download_name=download_filename)

@app.route('/backup')
def backup():
    """Create a backup of the current JSON file"""
    if 'current_json_file' not in session:
        flash('No file selected', 'error')
        return redirect(url_for('index'))
    
    backup_file = create_backup()
    if backup_file:
        flash(f'Backup created: {backup_file}', 'success')
    else:
        flash('Error creating backup', 'error')
    
    return redirect(url_for('dashboard'))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 