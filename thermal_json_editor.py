from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session, send_file
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from werkzeug.utils import secure_filename
import shutil
import tempfile
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for web
import matplotlib.pyplot as plt
import numpy as np
import threading
import time

# Performance optimizations for matplotlib
plt.rcParams['path.simplify'] = True
plt.rcParams['path.simplify_threshold'] = 0.1
plt.rcParams['agg.path.chunksize'] = 10000

# Add imports for thermal model visualization
try:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), 'header_files'))
    from thermal_parameters import get_parameters, ThermalParameters
    from thermal_analysis import calculate_system_size, ElementCoordinates
    from boundary_conditions import ElementBoundary
    from visualization import plot_boundary_conditions
    VISUALIZATION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Visualization modules not available: {e}")
    VISUALIZATION_AVAILABLE = False

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

DEFAULT_JSON_FILE = 'GEO.json'
ALLOWED_EXTENSIONS = {'json'}

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global simulation status tracking
simulation_status = {
    'running': False,
    'completed': False,
    'error': None,
    'start_time': None,
    'end_time': None,
    'output': '',
    'results_file': None,
    'plot_file': None
}

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
            data = json.loads(content)
            # Enrich node networks with NODE_CONNECTED boundary conditions as connections
            enrich_node_networks_with_boundary_connections(data)
            return data
    except Exception as e:
        print(f"Error loading JSON from {json_file}: {e}")
        return None

def enrich_node_networks_with_boundary_connections(data):
    """Add NODE_CONNECTED boundary conditions as connection entries in node networks"""
    if 'node_networks' not in data or 'regions' not in data:
        return
    
    # For each node network, find NODE_CONNECTED boundaries that connect to its nodes
    for network in data['node_networks']:
        network_node_ids = [node['id'] for node in network.get('nodes', [])]
        
        # Initialize connections if not present
        if 'connections' not in network:
            network['connections'] = []
        
        # Find existing boundary connection IDs to avoid duplicates
        existing_boundary_connection_ids = set()
        for conn in network['connections']:
            if conn.get('type') == 'boundary_connection':
                existing_boundary_connection_ids.add(conn['id'])
        
        # Scan all regions for NODE_CONNECTED boundaries
        for region in data['regions']:
            for bc_index, bc in enumerate(region.get('boundary_conditions', [])):
                if bc.get('type') == 'NODE_CONNECTED':
                    connected_node_id = bc.get('node_id')
                    if connected_node_id and connected_node_id in network_node_ids:
                        # Create a unique connection ID for this boundary connection
                        connection_id = f"region_{region['id']}_bc_{bc_index}_to_{connected_node_id}"
                        
                        # Skip if this connection is already added
                        if connection_id in existing_boundary_connection_ids:
                            continue
                        
                        # Add this boundary as a connection entry
                        boundary_connection = {
                            'id': connection_id,
                            'from_node': f"region_{region['id']}_boundary",
                            'to_node': connected_node_id,
                            'thermal_resistance': bc.get('thermal_resistance', 1.0),
                            'description': f"Region {region['id']} boundary connection",
                            'type': 'boundary_connection',  # Special type to distinguish from node-to-node connections
                            'region_id': region['id'],
                            'boundary_condition_index': bc_index,
                            'boundary_condition': bc.copy()  # Store the full BC data for reference
                        }
                        
                        network['connections'].append(boundary_connection)

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

def validate_boundary_condition_bounds(bc, region):
    """
    Validate that boundary condition bounds are within region limits.
    Returns error message if invalid, None if valid.
    """
    try:
        bc_type = bc.get('type', '')
        
        # Handle MAPPED boundary conditions
        if bc_type == 'MAPPED':
            if 'face_selection' not in bc or 'start_location' not in bc or 'end_location' not in bc:
                return None  # Skip validation if required fields are missing
                
            region_width = region['width']
            region_height = region['height']
            face_selection = bc['face_selection']
            start_location = bc['start_location']
            end_location = bc['end_location']
            
            if face_selection == 'y':
                # Y face selected: check X limitations [-width/2, width/2]
                region_min_x = -region_width / 2
                region_max_x = region_width / 2
                
                if start_location < region_min_x or start_location > region_max_x:
                    return f"Start location {start_location:.1f} outside region X limits [{region_min_x:.1f}, {region_max_x:.1f}] for Y face"
                if end_location < region_min_x or end_location > region_max_x:
                    return f"End location {end_location:.1f} outside region X limits [{region_min_x:.1f}, {region_max_x:.1f}] for Y face"
                    
            elif face_selection == 'x':
                # X face selected: check Y limitations [0, height]
                region_min_y = 0
                region_max_y = region_height
                
                if start_location < region_min_y or start_location > region_max_y:
                    return f"Start location {start_location:.1f} outside region Y limits [{region_min_y:.1f}, {region_max_y:.1f}] for X face"
                if end_location < region_min_y or end_location > region_max_y:
                    return f"End location {end_location:.1f} outside region Y limits [{region_min_y:.1f}, {region_max_y:.1f}] for X face"
            
            return None  # Valid
        
        # Handle ADIABATIC boundary conditions
        if bc_type == 'ADIABATIC':
            if 'point1' not in bc or 'point2' not in bc:
                return None  # Skip validation if required fields are missing
                
            region_width = region['width']
            region_height = region['height']
            point1 = bc['point1']
            point2 = bc['point2']
            
            # Region bounds: X from -width/2 to +width/2, Y from 0 to height
            region_min_x = -region_width / 2
            region_max_x = region_width / 2
            region_min_y = 0
            region_max_y = region_height
            
            # Validate point1
            if point1['x'] < region_min_x or point1['x'] > region_max_x:
                return f"Point 1 X coordinate {point1['x']:.1f} outside region X limits [{region_min_x:.1f}, {region_max_x:.1f}]"
            if point1['y'] < region_min_y or point1['y'] > region_max_y:
                return f"Point 1 Y coordinate {point1['y']:.1f} outside region Y limits [{region_min_y:.1f}, {region_max_y:.1f}]"
            
            # Validate point2
            if point2['x'] < region_min_x or point2['x'] > region_max_x:
                return f"Point 2 X coordinate {point2['x']:.1f} outside region X limits [{region_min_x:.1f}, {region_max_x:.1f}]"
            if point2['y'] < region_min_y or point2['y'] > region_max_y:
                return f"Point 2 Y coordinate {point2['y']:.1f} outside region Y limits [{region_min_y:.1f}, {region_max_y:.1f}]"
            
            return None  # Valid
        
        # Handle standard boundary conditions with centroid and dimensions
        if 'centroid' not in bc or 'width' not in bc or 'height' not in bc:
            return None  # Skip validation if required fields are missing
        
        # Get boundary condition properties
        centroid = bc['centroid']
        bc_x = centroid['x']
        bc_y = centroid['y']
        bc_width = bc['width']
        bc_height = bc['height']
        
        # Get region properties
        region_width = region['width']
        region_height = region['height']
        
        # Calculate boundary condition bounds
        bc_min_x = bc_x - bc_width/2
        bc_max_x = bc_x + bc_width/2
        bc_min_y = bc_y - bc_height/2
        bc_max_y = bc_y + bc_height/2
        
        # Region bounds: X from -regionWidth/2 to +regionWidth/2, Y from 0 to regionHeight
        region_min_x = -region_width/2
        region_max_x = region_width/2
        region_min_y = 0
        region_max_y = region_height
        
        # Validate X bounds
        if bc_min_x < region_min_x or bc_max_x > region_max_x:
            return f"X bounds [{bc_min_x:.1f}, {bc_max_x:.1f}] exceed region X limits [{region_min_x:.1f}, {region_max_x:.1f}]"
        
        # Validate Y bounds
        if bc_min_y < region_min_y or bc_max_y > region_max_y:
            return f"Y bounds [{bc_min_y:.1f}, {bc_max_y:.1f}] exceed region Y limits [{region_min_y:.1f}, {region_max_y:.1f}]"
        
        return None  # Valid
        
    except (KeyError, TypeError, ValueError) as e:
        return f"Invalid boundary condition data: {e}"

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

@app.route('/structures')
def structures():
    """Show all structures (both structures and actuators)"""
    if 'current_json_file' not in session:
        flash('Please select a JSON file first', 'error')
        return redirect(url_for('index'))
        
    data = load_json_data()
    if data is None:
        flash('Error loading JSON data', 'error')
        return render_template('error.html')
    
    # Prepare structures data
    mechanical_structures = []
    for region in data.get('regions', []):
        mechanical_structures.append({
            'id': region['id'],
            'width': region['width'],
            'height': region['height'],
            'thickness': region['thickness'],
            'thermal_conductivity': region['thermal_conductivity'],
            'centroid': region['centroid'],
            'mesh_settings': region['mesh_settings'],
            'boundary_conditions': region['boundary_conditions']
        })
    
    # Prepare thermal structures data
    thermal_structures = []
    total_heat_generation = 0
    for actuator in data.get('actuators', []):
        heat_total = actuator['heat_losses']['gearbox'] + actuator['heat_losses']['motor'] + actuator['heat_losses']['FETs']
        total_heat_generation += heat_total
        thermal_structures.append({
            'id': actuator['id'],
            'type': 'actuator',  # Default type for actuators
            'heat_losses': actuator['heat_losses'],
            'thermal_resistance': actuator['thermal_resistance']
        })
    
    # Prepare node networks data (including converted actuators)
    node_networks = []
    total_nodes = 0
    total_network_connections = 0
    total_network_heat = 0
    
    # Regular node networks
    for network in data.get('node_networks', []):
        network_heat = sum(node.get('heat_source', 0) for node in network.get('nodes', []))
        total_network_heat += network_heat
        total_nodes += len(network.get('nodes', []))
        
        # Count all connections (now includes boundary connections automatically)
        total_connections_for_network = len(network.get('connections', []))
        total_network_connections += total_connections_for_network
        
        node_networks.append({
            'id': network['id'],
            'name': network.get('name', network['id']),
            'description': network.get('description', ''),
            'node_count': len(network.get('nodes', [])),
            'connection_count': total_connections_for_network,
            'total_heat': network_heat,
            'type': 'network'
        })
    
    # Convert actuators to node networks
    for actuator in data.get('actuators', []):
        actuator_network = convert_actuator_to_node_network(actuator)
        network_heat = sum(node.get('heat_source', 0) for node in actuator_network.get('nodes', []))
        total_network_heat += network_heat
        total_nodes += len(actuator_network.get('nodes', []))
        total_network_connections += len(actuator_network.get('connections', []))
        
        node_networks.append({
            'id': actuator_network['id'],
            'name': actuator_network.get('name', actuator_network['id']),
            'description': actuator_network.get('description', ''),
            'node_count': len(actuator_network.get('nodes', [])),
            'connection_count': len(actuator_network.get('connections', [])),
            'total_heat': network_heat,
            'type': 'actuator',
            'source_actuator_id': actuator_network.get('source_actuator_id')
        })

    # Count total connections from regions
    total_connections = 0
    for region in data.get('regions', []):
        total_connections += len(region.get('boundary_conditions', []))
    
    return render_template('structures.html', 
                         mechanical_structures=mechanical_structures,
                         thermal_structures=thermal_structures,
                         node_networks=node_networks,
                         total_heat_generation=total_heat_generation,
                         total_network_heat=total_network_heat,
                         total_nodes=total_nodes,
                         total_network_connections=total_network_connections,
                         total_connections=total_connections)



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
    
    # Get available actuators and node networks for dropdowns (needed for validation error returns)
    actuators = data.get('actuators', [])
    node_networks = data.get('node_networks', [])
    
    if request.method == 'POST':
        # Update region data
        region['type'] = request.form.get('region_type', 'standard')
        region['thermal_conductivity'] = float(request.form.get('thermal_conductivity', 0))
        region['width'] = float(request.form.get('width', 0))
        region['height'] = float(request.form.get('height', 0))
        region['thickness'] = float(request.form.get('thickness', 0))
        
        # Auto-calculate centroid based on geometry
        height = float(request.form.get('height', 0))
        thickness = float(request.form.get('thickness', 0))
        region['centroid']['x'] = 0.0  # Always centered
        region['centroid']['y'] = height / 2.0  # Half of height
        region['centroid']['z'] = thickness / 2.0  # Half of thickness
        
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
                
                # Centroid and dimensions (not for ADIABATIC or MAPPED)
                if bc_type != 'ADIABATIC' and bc_type != 'MAPPED':
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
                elif bc_type == 'CONST_T':
                    bc['temperature'] = float(request.form.get(f'bc_{i}_temperature', 20.0))
                elif bc_type == 'ADIABATIC':
                    bc['point1'] = {
                        'x': float(request.form.get(f'bc_{i}_point1_x', 0.0)),
                        'y': float(request.form.get(f'bc_{i}_point1_y', 0.0))
                    }
                    bc['point2'] = {
                        'x': float(request.form.get(f'bc_{i}_point2_x', 0.0)),
                        'y': float(request.form.get(f'bc_{i}_point2_y', 0.0))
                    }
                elif bc_type == 'MAPPED':
                    bc['face_selection'] = request.form.get(f'bc_{i}_face_selection', 'x')
                    bc['start_location'] = float(request.form.get(f'bc_{i}_start_location', 0.0))
                    bc['end_location'] = float(request.form.get(f'bc_{i}_end_location', 100.0))
                elif bc_type == 'NODE_CONNECTED':
                    bc['node_id'] = request.form.get(f'bc_{i}_node_id', '')
                    bc['thermal_resistance'] = float(request.form.get(f'bc_{i}_thermal_resistance', 1.0))
                elif bc_type == 'PLASTIC_COVERED':
                    bc['plastic_thickness'] = float(request.form.get(f'bc_{i}_plastic_thickness', 0))
                    bc['plastic_conductivity'] = float(request.form.get(f'bc_{i}_plastic_conductivity', 0.25))
                elif bc_type == 'USERDEF_CONDUCTION':
                    bc['conductivity'] = float(request.form.get(f'bc_{i}_conductivity', 1.0))
                elif bc_type == 'USERDEF_CONVECTION':
                    bc['heat_transfer_coefficient'] = float(request.form.get(f'bc_{i}_heat_transfer_coefficient', 15.0))
                    bc['ambient_temperature'] = float(request.form.get(f'bc_{i}_ambient_temperature', 25.0))

                
                # Validate boundary condition bounds
                if bc_type not in []:  # Validate all boundary conditions
                    validation_error = validate_boundary_condition_bounds(bc, region)
                    if validation_error:
                        flash(f'Boundary condition {i+1} validation error: {validation_error}', 'error')
                        return render_template('edit_region.html', region=region, actuators=actuators, node_networks=node_networks)
                
                boundary_conditions.append(bc)
        
        region['boundary_conditions'] = boundary_conditions
        data['regions'][region_index] = region
        
        if save_json_data(data):
            flash(f'Region {region_id} updated successfully', 'success')
        else:
            flash('Error saving region data', 'error')
        
        return redirect(url_for('region_detail', region_id=region_id))
    
    return render_template('edit_region.html', region=region, actuators=actuators, node_networks=node_networks)

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
        
        # Update thermal resistance values (R1-R5)
        actuator['thermal_resistance'] = {
            'R1': float(request.form.get('resistance_r1', 0)),
            'R2': float(request.form.get('resistance_r2', 0)),
            'R3': float(request.form.get('resistance_r3', 0)),
            'R4': float(request.form.get('resistance_r4', 0)),
            'R5': float(request.form.get('resistance_r5', 0))
        }
        
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

def visualize_regions_web(regions, region_info, params):
    """
    Create visualization plots for all regions (web version).
    Saves the plot as an image file instead of displaying it.
    Optimized for speed.
    """
    if not regions:
        return create_simple_error_plot('No regions found to visualize')
    
    # Use a simplified approach to avoid complex subplot issues
    try:
        return create_simple_region_plot(regions)
    except Exception as e:
        print(f"Error in simplified plot: {e}")
        return create_simple_error_plot(f'Error creating visualization: {str(e)}')

def create_simple_error_plot(message):
    """Create a simple error plot"""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.text(0.5, 0.5, message, 
            ha='center', va='center', fontsize=16, transform=ax.transAxes)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    # Save the error plot
    static_dir = os.path.join(app.root_path, 'static')
    os.makedirs(static_dir, exist_ok=True)
    image_path = os.path.join(static_dir, 'boundary_conditions.png')
    plt.savefig(image_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    return 'static/boundary_conditions.png'

def create_simple_region_plot(regions):
    """Create a simple region visualization without complex subplot logic"""
    from matplotlib.patches import Rectangle
    
    # Create a single large plot
    fig, ax = plt.subplots(figsize=(15, 10))
    
    colors = {
        'ACTUATOR_CONNECTED': 'red',
        'CONST_Q': 'orange',
        'CONST_T': 'darkred',
        'MAPPED': 'blue', 
        'NODE_CONNECTED': 'magenta',
        'PLASTIC_COVERED': 'purple',
        'USERDEF_CONDUCTION': 'brown',
        'USERDEF_CONVECTION': 'lightblue',
        'ADIABATIC': 'gray'
    }
    
    # Plot all regions on the same axes with offsets
    y_offset = 0
    max_width = 0
    min_x = 0
    max_x = 0
    min_y = 0
    max_y = 0
    
    for region in regions:
        region_id = region["id"]
        width = region["width"]
        height = region["height"]
        
        # Draw region outline (centered around x=0)
        rect = Rectangle((-width/2, y_offset), width, height, 
                        fill=False, edgecolor='blue', linewidth=2,
                        label=f'{region_id}' if region == regions[0] else "")
        ax.add_patch(rect)
        
        # Add region label at top-left corner inside the region
        ax.text(-width/2 + 5, y_offset + height - 5, region_id,
               ha='left', va='top', fontsize=12, weight='bold',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.8))
        
        # Track region boundaries
        region_min_x = -width/2
        region_max_x = width/2
        region_min_y = y_offset
        region_max_y = y_offset + height
        min_x = min(min_x, region_min_x)
        max_x = max(max_x, region_max_x)
        min_y = min(min_y, region_min_y)
        max_y = max(max_y, region_max_y)
        
        # Add boundary conditions
        for bc_data in region.get('boundary_conditions', []):
            bc_type = bc_data.get('type', 'ADIABATIC')
            comments = bc_data.get('comments', '')
            color = colors.get(bc_type, 'gray')
            
            if bc_type == 'MAPPED':
                # Handle MAPPED boundary conditions as edge highlighting
                face_selection = bc_data.get('face_selection', '')
                start_location = bc_data.get('start_location', 0.0)
                end_location = bc_data.get('end_location', 0.0)
                
                if face_selection == 'x':
                    # X face mapping: highlight left and right edges
                    # Left edge (x = -width/2)
                    ax.plot([-width/2, -width/2], [y_offset + start_location, y_offset + end_location], 
                           color=color, linewidth=6, alpha=0.8, label=f'{bc_type} (X faces)' if bc_data == region.get('boundary_conditions', [])[0] else "")
                    # Right edge (x = +width/2)
                    ax.plot([width/2, width/2], [y_offset + start_location, y_offset + end_location], 
                           color=color, linewidth=6, alpha=0.8)
                    
                    # Add connecting lines to show mapping
                    ax.plot([-width/2, width/2], [y_offset + start_location, y_offset + start_location], 
                           color=color, linewidth=2, alpha=0.4, linestyle='--')
                    ax.plot([-width/2, width/2], [y_offset + end_location, y_offset + end_location], 
                           color=color, linewidth=2, alpha=0.4, linestyle='--')
                    
                    # Add label at region's left side, vertically centered within the mapped span
                    label_text = f'MAPPED X\n[{start_location:.1f}, {end_location:.1f}]'
                    if comments:
                        label_text += f'\n{comments}'
                    ax.text(region_min_x + 5, y_offset + (start_location + end_location)/2, label_text,
                           ha='left', va='center', fontsize=8,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.25))
                    
                elif face_selection == 'y':
                    # Y face mapping: highlight bottom and top edges
                    # Bottom edge (y = y_offset)
                    ax.plot([start_location, end_location], [y_offset, y_offset], 
                           color=color, linewidth=6, alpha=0.8, label=f'{bc_type} (Y faces)' if bc_data == region.get('boundary_conditions', [])[0] else "")
                    # Top edge (y = y_offset + height)
                    ax.plot([start_location, end_location], [y_offset + height, y_offset + height], 
                           color=color, linewidth=6, alpha=0.8)
                    
                    # Add connecting lines to show mapping
                    ax.plot([start_location, start_location], [y_offset, y_offset + height], 
                           color=color, linewidth=2, alpha=0.4, linestyle='--')
                    ax.plot([end_location, end_location], [y_offset, y_offset + height], 
                           color=color, linewidth=2, alpha=0.4, linestyle='--')
                    
                    # Add label at region's left-top corner area
                    label_text = f'MAPPED Y\n[{start_location:.1f}, {end_location:.1f}]'
                    if comments:
                        label_text += f'\n{comments}'
                    ax.text(region_min_x + 5, y_offset + height - 5, label_text,
                           ha='left', va='top', fontsize=8,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.25))
                
            elif bc_type == 'ADIABATIC':
                # Handle ADIABATIC boundary conditions as a line between two points
                point1 = bc_data.get('point1', {'x': 0, 'y': 0})
                point2 = bc_data.get('point2', {'x': 0, 'y': 0})
                
                # Draw the adiabatic line with dashed style (offset by region position)
                ax.plot([point1['x'], point2['x']], [y_offset + point1['y'], y_offset + point2['y']], 
                       color=color, linewidth=5, alpha=0.9, linestyle='--', 
                       label=f'{bc_type}' if bc_data == region.get('boundary_conditions', [])[0] else "")
                
                # Add square markers at the endpoints to distinguish from other BCs
                ax.plot(point1['x'], y_offset + point1['y'], 's', color=color, markersize=8, alpha=0.9, markeredgecolor='black', markeredgewidth=1)
                ax.plot(point2['x'], y_offset + point2['y'], 's', color=color, markersize=8, alpha=0.9, markeredgecolor='black', markeredgewidth=1)
                
                # Add perpendicular marks to show insulation (no heat flow)
                import numpy as np
                # Calculate perpendicular direction
                dx = point2['x'] - point1['x']
                dy = point2['y'] - point1['y']
                length = np.sqrt(dx**2 + dy**2)
                if length > 0:
                    # Normalized perpendicular vector
                    perp_x = -dy / length * 5  # 5mm perpendicular marks
                    perp_y = dx / length * 5
                    
                    # Add perpendicular marks at 1/4, 1/2, and 3/4 along the line
                    for frac in [0.25, 0.5, 0.75]:
                        mark_x = point1['x'] + frac * dx
                        mark_y = y_offset + point1['y'] + frac * dy
                        ax.plot([mark_x - perp_x, mark_x + perp_x], [mark_y - perp_y, mark_y + perp_y], 
                               color=color, linewidth=2, alpha=0.8)
                
                # Add label at the midpoint
                mid_x = (point1['x'] + point2['x']) / 2
                mid_y = y_offset + (point1['y'] + point2['y']) / 2
                label_text = f'ADIABATIC\n({point1["x"]:.0f},{point1["y"]:.0f}) → ({point2["x"]:.0f},{point2["y"]:.0f})'
                if comments:
                    label_text += f'\n{comments}'
                ax.text(mid_x, mid_y, label_text, 
                       ha='center', va='center', fontsize=8,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.3, edgecolor='black'))
                
            else:
                # Handle standard boundary conditions with centroid and dimensions
                centroid = bc_data.get('centroid', {'x': 0, 'y': 0, 'z': 0})
                bc_width = bc_data.get('width', 10)
                bc_height = bc_data.get('height', 10)
                
                # Track boundary condition boundaries
                bc_min_x = centroid['x'] - bc_width/2
                bc_max_x = centroid['x'] + bc_width/2
                bc_min_y = y_offset + centroid['y'] - bc_height/2
                bc_max_y = y_offset + centroid['y'] + bc_height/2
                min_x = min(min_x, bc_min_x)
                max_x = max(max_x, bc_max_x)
                min_y = min(min_y, bc_min_y)
                max_y = max(max_y, bc_max_y)
                
                # Add BC rectangle (offset by region position)
                bc_rect = Rectangle((centroid['x'] - bc_width/2, y_offset + centroid['y'] - bc_height/2), 
                                  bc_width, bc_height, 
                                  facecolor=color, alpha=0.6, edgecolor=color)
                ax.add_patch(bc_rect)
                
                # Add BC label
                if comments:
                    ax.text(centroid['x'], y_offset + centroid['y'], comments, 
                           ha='center', va='center', fontsize=8,
                           bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
        
        max_width = max(max_width, width)
        y_offset += height + 50  # Add spacing between regions
    
    # Set axis properties to show exact region boundaries with no padding
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_title('Boundary Conditions for All Regions', fontsize=16)
    
    # Save the plot
    static_dir = os.path.join(app.root_path, 'static')
    os.makedirs(static_dir, exist_ok=True)
    image_path = os.path.join(static_dir, 'boundary_conditions.png')
    plt.savefig(image_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    return 'static/boundary_conditions.png'

def visualize_regions_web_complex(regions, region_info, params):
    """
    The original complex visualization (keeping as backup)
    """
    
    num_regions = len(regions)
    num_cols = 3
    num_rows = (num_regions + num_cols - 1) // num_cols
    
    # Use non-interactive backend for faster rendering
    plt.ioff()  # Turn off interactive mode
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 6*num_rows))
    # Ensure axes is always a flat list of axis objects
    if num_rows == 1 and num_cols == 1:
        axes = [axes]  # Single subplot case
    elif num_rows == 1 or num_cols == 1:
        axes = axes.flatten() if hasattr(axes, 'flatten') else list(axes)  # 1D array case
    else:
        axes = axes.flatten()  # 2D array case
    
    try:
        print(f"DEBUG: Processing {num_regions} regions")
        print(f"DEBUG: axes type: {type(axes)}, length: {len(axes)}")
        print(f"DEBUG: axes[0] type: {type(axes[0])}")
        
        for idx, region in enumerate(regions):
            region_id = region["id"]
            print(f"DEBUG: Processing region {idx}: {region_id}")
            
            if region_id not in region_info:
                print(f"DEBUG: Skipping {region_id} - not in region_info")
                continue
                
            info = region_info[region_id]
            coords = info['coords']
            adiabatic_pairs = info.get('adiabatic_pairs', [])
            
            print(f"DEBUG: Getting axis {idx}, type: {type(axes[idx])}")
            try:
                # Create mock boundary conditions in the expected format
                # The plot_boundary_conditions function expects a dictionary mapping element indices to BC types
                boundary_conditions = {}
                
                # Create a simple plot just showing the region outline
                ax = axes[idx]
                print(f"DEBUG: Got axis for region {region_id}, type: {type(ax)}")
                
                # Clear the axis first
                ax.clear()
                
                # Just draw the region boundary without detailed boundary conditions
                width = region["width"]
                height = region["height"]
                
                # Draw region outline
                from matplotlib.patches import Rectangle
                ax.add_patch(Rectangle((-width/2, 0), width, height, 
                                     fill=False, edgecolor='blue', linewidth=2))
                
                # Add boundary condition visualization from region data
                for bc_data in region.get('boundary_conditions', []):
                    bc_type = bc_data.get('type', 'ADIABATIC')
                    comments = bc_data.get('comments', '')
                    
                    # Color mapping for BC types
                    colors = {
                        'ACTUATOR_CONNECTED': 'red',
                        'CONST_Q': 'orange',
                        'CONST_T': 'darkred',
                        'MAPPED': 'blue',
                        'NODE_CONNECTED': 'magenta',
                        'PLASTIC_COVERED': 'purple',
                        'USERDEF_CONDUCTION': 'brown',
                        'USERDEF_CONVECTION': 'lightblue',
                        'ADIABATIC': 'gray'
                    }
                    color = colors.get(bc_type, 'gray')
                    
                    if bc_type == 'MAPPED':
                        # Handle MAPPED boundary conditions as edge highlighting
                        face_selection = bc_data.get('face_selection', '')
                        start_location = bc_data.get('start_location', 0.0)
                        end_location = bc_data.get('end_location', 0.0)
                        
                        if face_selection == 'x':
                            # X face mapping: highlight left and right edges
                            ax.plot([-width/2, -width/2], [start_location, end_location], 
                                   color=color, linewidth=4, alpha=0.8)
                            ax.plot([width/2, width/2], [start_location, end_location], 
                                   color=color, linewidth=4, alpha=0.8)
                            
                            # Add connecting lines
                            ax.plot([-width/2, width/2], [start_location, start_location], 
                                   color=color, linewidth=1, alpha=0.4, linestyle='--')
                            ax.plot([-width/2, width/2], [end_location, end_location], 
                                   color=color, linewidth=1, alpha=0.4, linestyle='--')
                            
                            # Add label
                            label_text = f'MAPPED X [{start_location:.1f}, {end_location:.1f}]'
                            if comments:
                                label_text += f'\n{comments}'
                            ax.text(0, (start_location + end_location)/2, label_text, 
                                   ha='center', va='center', fontsize=6,
                                   bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.3))
                            
                        elif face_selection == 'y':
                            # Y face mapping: highlight bottom and top edges
                            ax.plot([start_location, end_location], [0, 0], 
                                   color=color, linewidth=4, alpha=0.8)
                            ax.plot([start_location, end_location], [height, height], 
                                   color=color, linewidth=4, alpha=0.8)
                            
                            # Add connecting lines
                            ax.plot([start_location, start_location], [0, height], 
                                   color=color, linewidth=1, alpha=0.4, linestyle='--')
                            ax.plot([end_location, end_location], [0, height], 
                                   color=color, linewidth=1, alpha=0.4, linestyle='--')
                            
                            # Add label
                            label_text = f'MAPPED Y [{start_location:.1f}, {end_location:.1f}]'
                            if comments:
                                label_text += f'\n{comments}'
                            ax.text((start_location + end_location)/2, height/2, label_text, 
                                   ha='center', va='center', fontsize=6,
                                   bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.3))
                    
                    elif bc_type == 'ADIABATIC':
                        # Handle ADIABATIC boundary conditions as a line between two points
                        point1 = bc_data.get('point1', {'x': 0, 'y': 0})
                        point2 = bc_data.get('point2', {'x': 0, 'y': 0})
                        
                        # Draw the adiabatic line with dashed style
                        ax.plot([point1['x'], point2['x']], [point1['y'], point2['y']], 
                               color=color, linewidth=4, alpha=0.9, linestyle='--')
                        
                        # Add square markers at the endpoints
                        ax.plot(point1['x'], point1['y'], 's', color=color, markersize=6, alpha=0.9, markeredgecolor='black', markeredgewidth=0.5)
                        ax.plot(point2['x'], point2['y'], 's', color=color, markersize=6, alpha=0.9, markeredgecolor='black', markeredgewidth=0.5)
                        
                        # Add perpendicular marks to show insulation
                        import numpy as np
                        dx = point2['x'] - point1['x']
                        dy = point2['y'] - point1['y']
                        length = np.sqrt(dx**2 + dy**2)
                        if length > 0:
                            # Smaller perpendicular marks for complex view
                            perp_x = -dy / length * 3  # 3mm perpendicular marks
                            perp_y = dx / length * 3
                            
                            # Add perpendicular marks at 1/3 and 2/3 along the line
                            for frac in [0.33, 0.67]:
                                mark_x = point1['x'] + frac * dx
                                mark_y = point1['y'] + frac * dy
                                ax.plot([mark_x - perp_x, mark_x + perp_x], [mark_y - perp_y, mark_y + perp_y], 
                                       color=color, linewidth=1.5, alpha=0.8)
                        
                        # Add label at the midpoint
                        mid_x = (point1['x'] + point2['x']) / 2
                        mid_y = (point1['y'] + point2['y']) / 2
                        label_text = f'ADIABATIC\n({point1["x"]:.0f},{point1["y"]:.0f}) → ({point2["x"]:.0f},{point2["y"]:.0f})'
                        if comments:
                            label_text += f'\n{comments}'
                        ax.text(mid_x, mid_y, label_text, 
                               ha='center', va='center', fontsize=6,
                               bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.3, edgecolor='black'))
                    
                    else:
                        # Handle standard boundary conditions with centroid and dimensions
                        centroid = bc_data.get('centroid', {'x': 0, 'y': 0, 'z': 0})
                        bc_width = bc_data.get('width', 10)
                        bc_height = bc_data.get('height', 10)
                        
                        # Add BC rectangle
                        bc_rect = Rectangle((centroid['x'] - bc_width/2, centroid['y'] - bc_height/2), 
                                          bc_width, bc_height, 
                                          facecolor=color, alpha=0.6, edgecolor=color)
                        ax.add_patch(bc_rect)
                        
                        # Add label
                        if comments:
                            ax.text(centroid['x'], centroid['y'], comments, 
                                   ha='center', va='center', fontsize=8,
                                   bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
                
                # Set axis properties
                ax.set_xlim(-width/2 - 20, width/2 + 20)
                ax.set_ylim(-20, height + 20)
                ax.set_aspect('equal')
                ax.grid(True, alpha=0.3)
                ax.set_xlabel('X (mm)')
                ax.set_ylabel('Y (mm)')
                
                thickness = region["thickness"]
                ax.set_title(f"{region_id}\nSize: {width}x{height}x{thickness} mm", fontsize=12, pad=10)
                
            except Exception as e:
                # Create a simple error subplot
                print(f"Error processing region {region_id}: {e}")  # Debug print
                ax_error = axes[idx]
                ax_error.text(0.5, 0.5, f'Error processing\n{region_id}', 
                             ha='center', va='center', fontsize=12, transform=ax_error.transAxes)
                ax_error.set_title(f"{region_id} (Error)", fontsize=12, pad=10)
                ax_error.set_xlim(0, 1)
                ax_error.set_ylim(0, 1)
        
        # Remove any unused subplots
        for idx in range(num_regions, len(axes)):
            try:
                fig.delaxes(axes[idx])
            except:
                pass  # Skip if there's an issue with deleting axes
        
        fig.suptitle("Boundary Conditions for All Regions (z=0 surface)", fontsize=16, y=0.95)
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        
        # Save the plot with optimized settings for speed
        static_dir = os.path.join(app.root_path, 'static')
        os.makedirs(static_dir, exist_ok=True)
        image_path = os.path.join(static_dir, 'boundary_conditions.png')
        plt.savefig(image_path, dpi=150, bbox_inches='tight', facecolor='white')  # Optimized for speed
        plt.close()
        
        return 'static/boundary_conditions.png'
        
    except Exception as e:
        plt.close()
        raise

@app.route('/visualize')
def visualize():
    """Generate boundary condition visualization"""
    if not VISUALIZATION_AVAILABLE:
        flash('Visualization modules not available. Please ensure thermal analysis dependencies are installed.', 'error')
        return redirect(url_for('regions'))
    
    if 'current_json_file' not in session:
        flash('Please select a JSON file first', 'error')
        return redirect(url_for('index'))
        
    data = load_json_data()
    if data is None:
        flash('Error loading JSON data', 'error')
        return redirect(url_for('regions'))
    
    try:
        # Get parameters and regions
        params = get_parameters(data)
        regions = data.get('regions', [])
        actuators = data.get('actuators', [])
        
        # Calculate system size and get region info
        region_info, actuator_info, total_elements, actuator_start_idx = calculate_system_size(
            regions, actuators, params
        )
        
        # Generate visualization
        image_path = visualize_regions_web(regions, region_info, params)
        
        flash('Visualization generated successfully!', 'success')
        return render_template('visualization.html', 
                             image_path=image_path, 
                             regions=regions,
                             total_elements=total_elements)
        
    except Exception as e:
        flash(f'Error generating visualization: {str(e)}', 'error')
        return redirect(url_for('regions'))

@app.route('/download_visualization')
def download_visualization():
    """Download the generated visualization image"""
    static_dir = os.path.join(app.root_path, 'static')
    image_path = os.path.join(static_dir, 'boundary_conditions.png')
    
    if os.path.exists(image_path):
        return send_file(image_path, as_attachment=True, download_name='boundary_conditions.png')
    else:
        flash('No visualization available. Please generate one first.', 'error')
        return redirect(url_for('regions'))

@app.route('/region/add', methods=['GET', 'POST'])
def add_region():
    """Add a new region"""
    if 'current_json_file' not in session:
        flash('Please select a JSON file first', 'error')
        return redirect(url_for('index'))
        
    data = load_json_data()
    if data is None:
        flash('Error loading JSON data', 'error')
        return redirect(url_for('regions'))
    
    if request.method == 'POST':
        # Get geometry values for auto-calculating centroid
        width = float(request.form.get('width', 100))
        height = float(request.form.get('height', 100))
        thickness = float(request.form.get('thickness', 2))
        
        # Create new region
        new_region = {
            'id': request.form.get('region_id', ''),
            'type': request.form.get('region_type', 'standard'),
            'centroid': {
                'x': 0.0,  # Always centered
                'y': height / 2.0,  # Half of height
                'z': thickness / 2.0  # Half of thickness
            },
            'width': width,
            'height': height,
            'thickness': thickness,
            'thermal_conductivity': float(request.form.get('thermal_conductivity', 120.0)),
            'mesh_settings': {
                'dx': float(request.form.get('mesh_dx', 5.0)),
                'dy': float(request.form.get('mesh_dy', 5.0)),
                'dz': float(request.form.get('mesh_dz', 0.5))
            },
            'boundary_conditions': []
        }
        
        # Check if region ID already exists
        existing_ids = [r['id'] for r in data.get('regions', [])]
        if new_region['id'] in existing_ids:
            flash(f'Region ID "{new_region["id"]}" already exists', 'error')
            return render_template('add_region.html', form_data=request.form)
        
        # Add new region to data
        if 'regions' not in data:
            data['regions'] = []
        data['regions'].append(new_region)
        
        if save_json_data(data):
            flash(f'Region "{new_region["id"]}" added successfully', 'success')
            return redirect(url_for('regions'))
        else:
            flash('Error saving region data', 'error')
    
    return render_template('add_region.html')

@app.route('/region/<region_id>/delete', methods=['POST'])
def delete_region(region_id):
    """Delete a specific region"""
    if 'current_json_file' not in session:
        flash('Please select a JSON file first', 'error')
        return redirect(url_for('index'))
        
    data = load_json_data()
    if data is None:
        flash('Error loading JSON data', 'error')
        return redirect(url_for('regions'))
    
    # Find and remove the region
    region_index = None
    for i, r in enumerate(data.get('regions', [])):
        if r.get('id') == region_id:
            region_index = i
            break
    
    if region_index is None:
        flash(f'Region {region_id} not found', 'error')
        return redirect(url_for('regions'))
    
    # Remove the region
    removed_region = data['regions'].pop(region_index)
    
    if save_json_data(data):
        flash(f'Region "{region_id}" deleted successfully', 'success')
    else:
        flash('Error saving changes', 'error')
    
    return redirect(url_for('regions'))

@app.route('/actuator/add', methods=['GET', 'POST'])
def add_actuator():
    """Add a new actuator"""
    if 'current_json_file' not in session:
        flash('Please select a JSON file first', 'error')
        return redirect(url_for('index'))
        
    data = load_json_data()
    if data is None:
        flash('Error loading JSON data', 'error')
        return redirect(url_for('actuators'))
    
    if request.method == 'POST':
        # Create new actuator
        new_actuator = {
            'id': request.form.get('actuator_id', ''),
            'heat_losses': {
                'gearbox': float(request.form.get('heat_gearbox', 0)),
                'motor': float(request.form.get('heat_motor', 0)),
                'FETs': float(request.form.get('heat_fets', 0))
            },
            'thermal_resistance': {
                'R1': float(request.form.get('resistance_r1', 2.0)),
                'R2': float(request.form.get('resistance_r2', 6.3)),
                'R3': float(request.form.get('resistance_r3', 2.2)),
                'R4': float(request.form.get('resistance_r4', 0.7)),
                'R5': float(request.form.get('resistance_r5', 0.3))
            }
        }
        
        # Check if actuator ID already exists
        existing_ids = [a['id'] for a in data.get('actuators', [])]
        if new_actuator['id'] in existing_ids:
            flash(f'Actuator ID "{new_actuator["id"]}" already exists', 'error')
            return render_template('add_actuator.html', form_data=request.form)
        
        # Add new actuator to data
        if 'actuators' not in data:
            data['actuators'] = []
        data['actuators'].append(new_actuator)
        
        if save_json_data(data):
            flash(f'Actuator "{new_actuator["id"]}" added successfully', 'success')
            return redirect(url_for('actuators'))
        else:
            flash('Error saving actuator data', 'error')
    
    return render_template('add_actuator.html')

@app.route('/actuator/<actuator_id>/delete', methods=['POST'])
def delete_actuator(actuator_id):
    """Delete a specific actuator"""
    if 'current_json_file' not in session:
        flash('Please select a JSON file first', 'error')
        return redirect(url_for('index'))
        
    data = load_json_data()
    if data is None:
        flash('Error loading JSON data', 'error')
        return redirect(url_for('actuators'))
    
    # Find and remove the actuator
    actuator_index = None
    for i, a in enumerate(data.get('actuators', [])):
        if a.get('id') == actuator_id:
            actuator_index = i
            break
    
    if actuator_index is None:
        flash(f'Actuator {actuator_id} not found', 'error')
        return redirect(url_for('actuators'))
    
    # Check if actuator is referenced in any boundary conditions
    references = []
    for region in data.get('regions', []):
        for bc in region.get('boundary_conditions', []):
            if bc.get('type') == 'ACTUATOR_CONNECTED' and bc.get('actuator_id') == actuator_id:
                references.append(f"Region '{region['id']}' boundary condition")
    
    if references:
        flash(f'Cannot delete actuator "{actuator_id}". It is referenced in: {", ".join(references)}', 'error')
        return redirect(url_for('actuators'))
    
    # Remove the actuator
    removed_actuator = data['actuators'].pop(actuator_index)
    
    if save_json_data(data):
        flash(f'Actuator "{actuator_id}" deleted successfully', 'success')
    else:
        flash('Error saving changes', 'error')
    
    return redirect(url_for('actuators'))

def convert_actuator_to_node_network(actuator):
    """Convert an actuator to a node network representation"""
    actuator_id = actuator['id']
    
    # Create thermal nodes for the actuator components
    nodes = []
    connections = []
    
    # Housing node (connects to regions via ACTUATOR_CONNECTED)
    housing_node = {
        'id': f"{actuator_id}_housing",
        'name': f"{actuator_id} Housing",
        'thermal_capacity': 0.0,  # Actuators typically don't store thermal capacity
        'heat_source': actuator['heat_losses']['gearbox']  # Gearbox heat goes to housing
    }
    nodes.append(housing_node)
    
    # Motor node (internal heat generation)
    motor_node = {
        'id': f"{actuator_id}_motor",
        'name': f"{actuator_id} Motor",
        'thermal_capacity': 0.0,
        'heat_source': actuator['heat_losses']['motor']
    }
    nodes.append(motor_node)
    
    # FETs node (electronic heat generation)
    fets_node = {
        'id': f"{actuator_id}_fets",
        'name': f"{actuator_id} FETs",
        'thermal_capacity': 0.0,
        'heat_source': actuator['heat_losses']['FETs']
    }
    nodes.append(fets_node)
    
    # Air node for ambient heat transfer
    # Note: Air is handled as a special case, so we connect to "air" directly
    
    # Create thermal resistance connections based on actuator thermal network
    resistance = actuator['thermal_resistance']
    
    # R1: Motor to housing (junction to case)
    if resistance.get('R1', 0) > 0:
        connections.append({
            'id': f"{actuator_id}_R1",
            'from_node': f"{actuator_id}_motor",
            'to_node': f"{actuator_id}_housing",
            'thermal_resistance': resistance['R1'],
            'description': f"Motor to housing thermal resistance (R1)"
        })
    
    # R2: Housing to air (case to ambient) 
    if resistance.get('R2', 0) > 0:
        connections.append({
            'id': f"{actuator_id}_R2",
            'from_node': f"{actuator_id}_housing",
            'to_node': "air",
            'thermal_resistance': resistance['R2'],
            'description': f"Housing to ambient thermal resistance (R2)"
        })
    
    # R3: Motor to air (direct motor cooling)
    if resistance.get('R3', 0) > 0:
        connections.append({
            'id': f"{actuator_id}_R3",
            'from_node': f"{actuator_id}_motor",
            'to_node': "air",
            'thermal_resistance': resistance['R3'],
            'description': f"Motor to ambient thermal resistance (R3)"
        })
    
    # R4: FETs to housing (electronics to case)
    if resistance.get('R4', 0) > 0:
        connections.append({
            'id': f"{actuator_id}_R4",
            'from_node': f"{actuator_id}_fets",
            'to_node': f"{actuator_id}_housing",
            'thermal_resistance': resistance['R4'],
            'description': f"FETs to housing thermal resistance (R4)"
        })
    
    # R5: Additional thermal path (if present)
    if resistance.get('R5', 0) > 0:
        connections.append({
            'id': f"{actuator_id}_R5",
            'from_node': f"{actuator_id}_fets",
            'to_node': "air",
            'thermal_resistance': resistance['R5'],
            'description': f"FETs to ambient thermal resistance (R5)"
        })
    
    # Create the node network
    network = {
        'id': f"actuator_{actuator_id}",
        'name': f"Actuator {actuator_id}",
        'description': f"Thermal network for actuator {actuator_id}. Heat sources: Gearbox {actuator['heat_losses']['gearbox']}W, Motor {actuator['heat_losses']['motor']}W, FETs {actuator['heat_losses']['FETs']}W",
        'type': 'actuator',  # Special type to identify actuator networks
        'source_actuator_id': actuator_id,  # Reference to original actuator
        'nodes': nodes,
        'connections': connections
    }
    
    return network

@app.route('/node_networks')
def node_networks():
    """Show all node networks including converted actuators"""
    if 'current_json_file' not in session:
        flash('Please select a JSON file first', 'error')
        return redirect(url_for('index'))
        
    data = load_json_data()
    if data is None:
        flash('Error loading JSON data', 'error')
        return render_template('error.html')
    
    # Get regular node networks - connections now include boundary connections automatically
    networks = []
    for network in data.get('node_networks', []):
        # Count all connections (now includes boundary connections automatically)
        total_connections = len(network.get('connections', []))
        
        # Create network data with connection count
        network_data = network.copy()
        network_data['connection_count'] = total_connections
        networks.append(network_data)
    
    # Convert actuators to node networks
    actuator_networks = []
    for actuator in data.get('actuators', []):
        actuator_network = convert_actuator_to_node_network(actuator)
        actuator_networks.append(actuator_network)
    
    # Combine both types
    all_networks = networks + actuator_networks
    
    return render_template('node_networks.html', node_networks=all_networks)

@app.route('/node_network/<network_id>')
def node_network_detail(network_id):
    """Show details of a specific node network (including converted actuators)"""
    data = load_json_data()
    if data is None:
        flash('Error loading JSON data', 'error')
        return render_template('error.html')
    
    network = None
    
    # First check regular node networks
    for n in data.get('node_networks', []):
        if n.get('id') == network_id:
            network = n
            break
    
    # If not found, check if it's a converted actuator network
    if network is None and network_id.startswith('actuator_'):
        actuator_id = network_id.replace('actuator_', '')
        for actuator in data.get('actuators', []):
            if actuator.get('id') == actuator_id:
                network = convert_actuator_to_node_network(actuator)
                break
    
    if network is None:
        flash(f'Node network {network_id} not found', 'error')
        return redirect(url_for('node_networks'))
    
    # Find regions with NODE_CONNECTED boundary conditions that are NOT yet connected to this network
    network_node_ids = [node['id'] for node in network.get('nodes', [])]
    node_connected_regions = []
    already_connected_regions = set()
    
    # Identify regions already connected through boundary connections
    for connection in network.get('connections', []):
        if connection.get('type') == 'boundary_connection':
            already_connected_regions.add(connection.get('region_id'))
    
    # Find available regions with NODE_CONNECTED boundaries not yet connected to this network
    for region in data.get('regions', []):
        for bc in region.get('boundary_conditions', []):
            if bc.get('type') == 'NODE_CONNECTED' and region['id'] not in already_connected_regions:
                node_connected_regions.append({
                    'id': f"region_{region['id']}_bc_{bc.get('type', 'boundary')}",
                    'name': f"Region {region['id']} - Node Connected Boundary",
                    'region_id': region['id'],
                    'boundary_type': bc.get('type')
                })
                break  # Only add region once even if multiple NODE_CONNECTED boundaries
    
    return render_template('node_network_detail.html', 
                         network=network, 
                         node_connected_regions=node_connected_regions)

@app.route('/node_network/add', methods=['GET', 'POST'])
def add_node_network():
    """Add a new node network"""
    if 'current_json_file' not in session:
        flash('Please select a JSON file first', 'error')
        return redirect(url_for('index'))
        
    data = load_json_data()
    if data is None:
        flash('Error loading JSON data', 'error')
        return redirect(url_for('node_networks'))
    
    if request.method == 'POST':
        # Create new node network
        new_network = {
            'id': request.form.get('network_id', ''),
            'name': request.form.get('network_name', ''),
            'description': request.form.get('description', ''),
            'nodes': [],
            'connections': []
        }
        
        # Check if network ID already exists
        existing_ids = [n['id'] for n in data.get('node_networks', [])]
        if new_network['id'] in existing_ids:
            flash(f'Node network ID "{new_network["id"]}" already exists', 'error')
            return render_template('add_node_network.html', form_data=request.form)
        
        # Add new network to data
        if 'node_networks' not in data:
            data['node_networks'] = []
        data['node_networks'].append(new_network)
        
        if save_json_data(data):
            flash(f'Node network "{new_network["id"]}" added successfully', 'success')
            return redirect(url_for('node_networks'))
        else:
            flash('Error saving node network data', 'error')
    
    return render_template('add_node_network.html')

@app.route('/node_network/<network_id>/delete', methods=['POST'])
def delete_node_network(network_id):
    """Delete a specific node network"""
    if 'current_json_file' not in session:
        flash('Please select a JSON file first', 'error')
        return redirect(url_for('index'))
        
    data = load_json_data()
    if data is None:
        flash('Error loading JSON data', 'error')
        return redirect(url_for('node_networks'))
    
    # Find and remove the network
    network_index = None
    for i, n in enumerate(data.get('node_networks', [])):
        if n.get('id') == network_id:
            network_index = i
            break
    
    if network_index is None:
        flash(f'Node network {network_id} not found', 'error')
        return redirect(url_for('node_networks'))
    
    # Remove the network
    removed_network = data['node_networks'].pop(network_index)
    
    if save_json_data(data):
        flash(f'Node network "{network_id}" deleted successfully', 'success')
    else:
        flash('Error saving changes', 'error')
    
    return redirect(url_for('node_networks'))

@app.route('/node_network/<network_id>/edit', methods=['GET', 'POST'])
def edit_node_network(network_id):
    """Edit a specific node network"""
    if 'current_json_file' not in session:
        flash('Please select a JSON file first', 'error')
        return redirect(url_for('index'))
        
    data = load_json_data()
    if data is None:
        flash('Error loading JSON data', 'error')
        return redirect(url_for('node_networks'))
    
    # Find the network
    network_index = None
    network = None
    for i, n in enumerate(data.get('node_networks', [])):
        if n.get('id') == network_id:
            network_index = i
            network = n
            break
    
    if network is None:
        flash(f'Node network {network_id} not found', 'error')
        return redirect(url_for('node_networks'))
    
    if request.method == 'POST':
        # Update network properties
        network['name'] = request.form.get('network_name', '')
        network['description'] = request.form.get('description', '')
        
        data['node_networks'][network_index] = network
        
        if save_json_data(data):
            flash(f'Node network "{network_id}" updated successfully', 'success')
            return redirect(url_for('node_network_detail', network_id=network_id))
        else:
            flash('Error saving node network data', 'error')
    
    return render_template('edit_node_network.html', network=network)

@app.route('/node_network/<network_id>/node/add', methods=['POST'])
def add_node_to_network(network_id):
    """Add a new node to a network"""
    if 'current_json_file' not in session:
        flash('Please select a JSON file first', 'error')
        return redirect(url_for('index'))
        
    data = load_json_data()
    if data is None:
        flash('Error loading JSON data', 'error')
        return redirect(url_for('node_networks'))
    
    # Find the network
    network = None
    network_index = None
    for i, n in enumerate(data.get('node_networks', [])):
        if n.get('id') == network_id:
            network = n
            network_index = i
            break
    
    if network is None:
        flash(f'Node network {network_id} not found', 'error')
        return redirect(url_for('node_networks'))
    
    # Create new node
    new_node = {
        'id': request.form.get('node_id', ''),
        'name': request.form.get('node_name', ''),
        'thermal_capacity': float(request.form.get('thermal_capacity', 0)),
        'heat_source': float(request.form.get('heat_source', 0))
    }
    
    # Check if node ID already exists in this network
    existing_node_ids = [node['id'] for node in network.get('nodes', [])]
    if new_node['id'] in existing_node_ids:
        flash(f'Node ID "{new_node["id"]}" already exists in this network', 'error')
        return redirect(url_for('node_network_detail', network_id=network_id))
    
    # Add new node to network
    network['nodes'].append(new_node)
    data['node_networks'][network_index] = network
    
    if save_json_data(data):
        flash(f'Node "{new_node["id"]}" added successfully', 'success')
    else:
        flash('Error saving node data', 'error')
    
    return redirect(url_for('node_network_detail', network_id=network_id))

@app.route('/node_network/<network_id>/connection/add', methods=['POST'])
def add_connection_to_network(network_id):
    """Add a new thermal resistance connection to a network"""
    if 'current_json_file' not in session:
        flash('Please select a JSON file first', 'error')
        return redirect(url_for('index'))
        
    data = load_json_data()
    if data is None:
        flash('Error loading JSON data', 'error')
        return redirect(url_for('node_networks'))
    
    # Find the network
    network = None
    network_index = None
    for i, n in enumerate(data.get('node_networks', [])):
        if n.get('id') == network_id:
            network = n
            network_index = i
            break
    
    if network is None:
        flash(f'Node network {network_id} not found', 'error')
        return redirect(url_for('node_networks'))
    
    # Create new connection
    new_connection = {
        'id': request.form.get('connection_id', ''),
        'from_node': request.form.get('from_node', ''),
        'to_node': request.form.get('to_node', ''),
        'thermal_resistance': float(request.form.get('thermal_resistance', 1.0)),
        'description': request.form.get('description', '')
    }
    
    # Check if connection ID already exists in this network
    existing_connection_ids = [conn['id'] for conn in network.get('connections', [])]
    if new_connection['id'] in existing_connection_ids:
        flash(f'Connection ID "{new_connection["id"]}" already exists in this network', 'error')
        return redirect(url_for('node_network_detail', network_id=network_id))
    
    # Validate that both nodes exist in the network (or are "air" or region boundaries)
    node_ids = [node['id'] for node in network.get('nodes', [])]
    node_ids.append('air')  # Add air as a valid option
    
    # Add region boundaries with NODE_CONNECTED boundary conditions
    for region in data.get('regions', []):
        for bc in region.get('boundary_conditions', []):
            if bc.get('type') == 'NODE_CONNECTED':
                region_boundary_id = f"region_{region['id']}_bc_{bc.get('type', 'boundary')}"
                node_ids.append(region_boundary_id)
                break  # Only add region once even if multiple NODE_CONNECTED boundaries
    
    if new_connection['from_node'] not in node_ids:
        flash(f'Node 1 "{new_connection["from_node"]}" does not exist in this network', 'error')
        return redirect(url_for('node_network_detail', network_id=network_id))
    
    if new_connection['to_node'] not in node_ids:
        flash(f'Node 2 "{new_connection["to_node"]}" does not exist in this network', 'error')
        return redirect(url_for('node_network_detail', network_id=network_id))
    
    # Prevent connecting a node to itself
    if new_connection['from_node'] == new_connection['to_node']:
        flash('Cannot connect a node to itself', 'error')
        return redirect(url_for('node_network_detail', network_id=network_id))
    
    # Add new connection to network
    network['connections'].append(new_connection)
    data['node_networks'][network_index] = network
    
    if save_json_data(data):
        flash(f'Connection "{new_connection["id"]}" added successfully', 'success')
    else:
        flash('Error saving connection data', 'error')
    
    return redirect(url_for('node_network_detail', network_id=network_id))

@app.route('/node_network/<network_id>/node/<node_id>/edit', methods=['GET', 'POST'])
def edit_node_in_network(network_id, node_id):
    """Edit a specific node in a network"""
    if 'current_json_file' not in session:
        flash('Please select a JSON file first', 'error')
        return redirect(url_for('index'))
        
    data = load_json_data()
    if data is None:
        flash('Error loading JSON data', 'error')
        return redirect(url_for('node_networks'))
    
    # Find the network
    network = None
    network_index = None
    for i, n in enumerate(data.get('node_networks', [])):
        if n.get('id') == network_id:
            network = n
            network_index = i
            break
    
    if network is None:
        flash(f'Node network {network_id} not found', 'error')
        return redirect(url_for('node_networks'))
    
    # Find the node
    node = None
    node_index = None
    for i, n in enumerate(network.get('nodes', [])):
        if n.get('id') == node_id:
            node = n
            node_index = i
            break
    
    if node is None:
        flash(f'Node {node_id} not found in network {network_id}', 'error')
        return redirect(url_for('node_network_detail', network_id=network_id))
    
    if request.method == 'POST':
        # Update node properties
        node['name'] = request.form.get('node_name', '')
        node['thermal_capacity'] = float(request.form.get('thermal_capacity', 0))
        node['heat_source'] = float(request.form.get('heat_source', 0))
        
        network['nodes'][node_index] = node
        data['node_networks'][network_index] = network
        
        if save_json_data(data):
            flash(f'Node "{node_id}" updated successfully', 'success')
            return redirect(url_for('node_network_detail', network_id=network_id))
        else:
            flash('Error saving node data', 'error')
    
    return render_template('edit_node_in_network.html', network=network, node=node)

@app.route('/node_network/<network_id>/node/<node_id>/delete', methods=['POST'])
def delete_node_from_network(network_id, node_id):
    """Delete a specific node from a network"""
    if 'current_json_file' not in session:
        flash('Please select a JSON file first', 'error')
        return redirect(url_for('index'))
        
    data = load_json_data()
    if data is None:
        flash('Error loading JSON data', 'error')
        return redirect(url_for('node_networks'))
    
    # Find the network
    network = None
    network_index = None
    for i, n in enumerate(data.get('node_networks', [])):
        if n.get('id') == network_id:
            network = n
            network_index = i
            break
    
    if network is None:
        flash(f'Node network {network_id} not found', 'error')
        return redirect(url_for('node_networks'))
    
    # Find the node
    node_index = None
    for i, n in enumerate(network.get('nodes', [])):
        if n.get('id') == node_id:
            node_index = i
            break
    
    if node_index is None:
        flash(f'Node {node_id} not found in network {network_id}', 'error')
        return redirect(url_for('node_network_detail', network_id=network_id))
    
    # Check if node is referenced in any connections
    references = []
    for conn in network.get('connections', []):
        if conn.get('from_node') == node_id or conn.get('to_node') == node_id:
            references.append(f"Connection '{conn['id']}'")
    
    if references:
        flash(f'Cannot delete node "{node_id}". It is referenced in: {", ".join(references)}', 'error')
        return redirect(url_for('node_network_detail', network_id=network_id))
    
    # Remove the node
    removed_node = network['nodes'].pop(node_index)
    data['node_networks'][network_index] = network
    
    if save_json_data(data):
        flash(f'Node "{node_id}" deleted successfully', 'success')
    else:
        flash('Error saving changes', 'error')
    
    return redirect(url_for('node_network_detail', network_id=network_id))

@app.route('/node_network/<network_id>/connection/<connection_id>/edit', methods=['GET', 'POST'])
def edit_connection_in_network(network_id, connection_id):
    """Edit a specific connection in a network"""
    if 'current_json_file' not in session:
        flash('Please select a JSON file first', 'error')
        return redirect(url_for('index'))
        
    data = load_json_data()
    if data is None:
        flash('Error loading JSON data', 'error')
        return redirect(url_for('node_networks'))
    
    # Find the network
    network = None
    network_index = None
    for i, n in enumerate(data.get('node_networks', [])):
        if n.get('id') == network_id:
            network = n
            network_index = i
            break
    
    if network is None:
        flash(f'Node network {network_id} not found', 'error')
        return redirect(url_for('node_networks'))
    
    # Find the connection
    connection = None
    connection_index = None
    for i, c in enumerate(network.get('connections', [])):
        if c.get('id') == connection_id:
            connection = c
            connection_index = i
            break
    
    if connection is None:
        flash(f'Connection {connection_id} not found in network {network_id}', 'error')
        return redirect(url_for('node_network_detail', network_id=network_id))
    
    if request.method == 'POST':
        # Update connection properties
        connection['from_node'] = request.form.get('from_node', '')
        connection['to_node'] = request.form.get('to_node', '')
        connection['thermal_resistance'] = float(request.form.get('thermal_resistance', 1.0))
        connection['description'] = request.form.get('description', '')
        
        # Validate that both nodes exist in the network (or are "air" or region boundaries)
        node_ids = [node['id'] for node in network.get('nodes', [])]
        node_ids.append('air')  # Add air as a valid option
        
        # Add region boundaries with NODE_CONNECTED boundary conditions
        for region in data.get('regions', []):
            for bc in region.get('boundary_conditions', []):
                if bc.get('type') == 'NODE_CONNECTED':
                    region_boundary_id = f"region_{region['id']}_bc_{bc.get('type', 'boundary')}"
                    node_ids.append(region_boundary_id)
                    break  # Only add region once even if multiple NODE_CONNECTED boundaries
        
        if connection['from_node'] not in node_ids:
            flash(f'Node 1 "{connection["from_node"]}" does not exist in this network', 'error')
            return render_template('edit_connection_in_network.html', network=network, connection=connection)
        
        if connection['to_node'] not in node_ids:
            flash(f'Node 2 "{connection["to_node"]}" does not exist in this network', 'error')
            return render_template('edit_connection_in_network.html', network=network, connection=connection)
        
        # Prevent connecting a node to itself
        if connection['from_node'] == connection['to_node']:
            flash('Cannot connect a node to itself', 'error')
            return render_template('edit_connection_in_network.html', network=network, connection=connection)
        
        network['connections'][connection_index] = connection
        data['node_networks'][network_index] = network
        
        if save_json_data(data):
            flash(f'Connection "{connection_id}" updated successfully', 'success')
            return redirect(url_for('node_network_detail', network_id=network_id))
        else:
            flash('Error saving connection data', 'error')
    
    # Find regions with NODE_CONNECTED boundary conditions for template
    node_connected_regions = []
    for region in data.get('regions', []):
        for bc in region.get('boundary_conditions', []):
            if bc.get('type') == 'NODE_CONNECTED':
                node_connected_regions.append({
                    'id': f"region_{region['id']}_bc_{bc.get('type', 'boundary')}",
                    'name': f"Region {region['id']} - Node Connected Boundary",
                    'region_id': region['id'],
                    'boundary_type': bc.get('type')
                })
                break  # Only add region once even if multiple NODE_CONNECTED boundaries
    
    return render_template('edit_connection_in_network.html', network=network, connection=connection, node_connected_regions=node_connected_regions)

@app.route('/node_network/<network_id>/connection/<connection_id>/delete', methods=['POST'])
def delete_connection_from_network(network_id, connection_id):
    """Delete a specific connection from a network"""
    if 'current_json_file' not in session:
        flash('Please select a JSON file first', 'error')
        return redirect(url_for('index'))
        
    data = load_json_data()
    if data is None:
        flash('Error loading JSON data', 'error')
        return redirect(url_for('node_networks'))
    
    # Find the network
    network = None
    network_index = None
    for i, n in enumerate(data.get('node_networks', [])):
        if n.get('id') == network_id:
            network = n
            network_index = i
            break
    
    if network is None:
        flash(f'Node network {network_id} not found', 'error')
        return redirect(url_for('node_networks'))
    
    # Find the connection
    connection_index = None
    for i, c in enumerate(network.get('connections', [])):
        if c.get('id') == connection_id:
            connection_index = i
            break
    
    if connection_index is None:
        flash(f'Connection {connection_id} not found in network {network_id}', 'error')
        return redirect(url_for('node_network_detail', network_id=network_id))
    
    # Remove the connection
    removed_connection = network['connections'].pop(connection_index)
    data['node_networks'][network_index] = network
    
    if save_json_data(data):
        flash(f'Connection "{connection_id}" deleted successfully', 'success')
    else:
        flash('Error saving changes', 'error')
    
    return redirect(url_for('node_network_detail', network_id=network_id))

@app.route('/node_network/<network_id>/nodes/bulk_delete', methods=['POST'])
def bulk_delete_nodes_from_network(network_id):
    """Delete multiple nodes from a network"""
    if 'current_json_file' not in session:
        flash('Please select a JSON file first', 'error')
        return redirect(url_for('index'))
        
    data = load_json_data()
    if data is None:
        flash('Error loading JSON data', 'error')
        return redirect(url_for('node_networks'))
    
    # Find the network
    network = None
    network_index = None
    for i, n in enumerate(data.get('node_networks', [])):
        if n.get('id') == network_id:
            network = n
            network_index = i
            break
    
    if network is None:
        flash(f'Node network {network_id} not found', 'error')
        return redirect(url_for('node_networks'))
    
    # Get the node IDs to delete
    node_ids_to_delete = request.form.getlist('node_ids')
    if not node_ids_to_delete:
        flash('No nodes selected for deletion', 'error')
        return redirect(url_for('node_network_detail', network_id=network_id))
    
    # Check if any nodes are referenced in connections
    references = []
    for conn in network.get('connections', []):
        if conn.get('from_node') in node_ids_to_delete or conn.get('to_node') in node_ids_to_delete:
            references.append(f"Connection '{conn['id']}'")
    
    # Remove nodes and their associated connections
    nodes_to_keep = []
    connections_to_keep = []
    deleted_nodes = []
    deleted_connections = []
    
    # Keep nodes that are not in the deletion list
    for node in network.get('nodes', []):
        if node.get('id') not in node_ids_to_delete:
            nodes_to_keep.append(node)
        else:
            deleted_nodes.append(node['id'])
    
    # Keep connections that don't reference deleted nodes
    for conn in network.get('connections', []):
        if conn.get('from_node') not in node_ids_to_delete and conn.get('to_node') not in node_ids_to_delete:
            connections_to_keep.append(conn)
        else:
            deleted_connections.append(conn['id'])
    
    # Update the network
    network['nodes'] = nodes_to_keep
    network['connections'] = connections_to_keep
    data['node_networks'][network_index] = network
    
    if save_json_data(data):
        success_msg = f'Successfully deleted {len(deleted_nodes)} nodes'
        if deleted_connections:
            success_msg += f' and {len(deleted_connections)} associated connections'
        flash(success_msg, 'success')
    else:
        flash('Error saving changes', 'error')
    
    return redirect(url_for('node_network_detail', network_id=network_id))

@app.route('/node_network/<network_id>/connections/bulk_delete', methods=['POST'])
def bulk_delete_connections_from_network(network_id):
    """Delete multiple connections from a network"""
    if 'current_json_file' not in session:
        flash('Please select a JSON file first', 'error')
        return redirect(url_for('index'))
        
    data = load_json_data()
    if data is None:
        flash('Error loading JSON data', 'error')
        return redirect(url_for('node_networks'))
    
    # Find the network
    network = None
    network_index = None
    for i, n in enumerate(data.get('node_networks', [])):
        if n.get('id') == network_id:
            network = n
            network_index = i
            break
    
    if network is None:
        flash(f'Node network {network_id} not found', 'error')
        return redirect(url_for('node_networks'))
    
    # Get the connection IDs to delete
    connection_ids_to_delete = request.form.getlist('connection_ids')
    if not connection_ids_to_delete:
        flash('No connections selected for deletion', 'error')
        return redirect(url_for('node_network_detail', network_id=network_id))
    
    # Remove connections
    connections_to_keep = []
    deleted_connections = []
    
    for conn in network.get('connections', []):
        if conn.get('id') not in connection_ids_to_delete:
            connections_to_keep.append(conn)
        else:
            deleted_connections.append(conn['id'])
    
    # Update the network
    network['connections'] = connections_to_keep
    data['node_networks'][network_index] = network
    
    if save_json_data(data):
        flash(f'Successfully deleted {len(deleted_connections)} connections', 'success')
    else:
        flash('Error saving changes', 'error')
    
    return redirect(url_for('node_network_detail', network_id=network_id))

def run_simulation_background():
    """Run the thermal simulation in a background thread"""
    global simulation_status
    
    try:
        simulation_status['running'] = True
        simulation_status['completed'] = False
        simulation_status['error'] = None
        simulation_status['start_time'] = datetime.now()
        simulation_status['output'] = ''
        
        # Change to the directory containing the solver
        original_dir = os.getcwd()
        solver_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'header_files')
        os.chdir(solver_dir)
        
        # Run the solver
        process = subprocess.Popen(
            [sys.executable, 'arm_main_solver.py'],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            cwd=solver_dir
        )
        
        # Capture output in real-time
        output_lines = []
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                output_lines.append(output.strip())
                simulation_status['output'] = '\n'.join(output_lines)
        
        # Wait for process to complete
        return_code = process.poll()
        
        # Change back to original directory
        os.chdir(original_dir)
        
        if return_code == 0:
            simulation_status['completed'] = True
            simulation_status['error'] = None
            
            # Check for output files
            results_file = os.path.join(solver_dir, 'arm_thermal_results.txt')
            plot_file = os.path.join(solver_dir, 'temperature_distribution.png')
            
            if os.path.exists(results_file):
                simulation_status['results_file'] = results_file
            if os.path.exists(plot_file):
                simulation_status['plot_file'] = plot_file
        else:
            simulation_status['error'] = f"Simulation failed with return code {return_code}"
            
    except Exception as e:
        simulation_status['error'] = str(e)
    
    finally:
        simulation_status['running'] = False
        simulation_status['end_time'] = datetime.now()

@app.route('/run_simulation', methods=['POST'])
def run_simulation():
    """Start the thermal simulation"""
    if 'current_json_file' not in session:
        flash('Please select a JSON file first', 'error')
        return redirect(url_for('index'))
    
    global simulation_status
    
    if simulation_status['running']:
        flash('Simulation is already running', 'warning')
        return redirect(url_for('simulation_status'))
    
    # Copy current JSON file to the solver directory
    try:
        current_file = get_current_json_file()
        solver_dir = os.path.join(os.path.dirname(__file__), 'header_files')
        target_file = os.path.join(solver_dir, 'GEO.json')
        
        shutil.copy2(current_file, target_file)
        
        # Start simulation in background thread
        thread = threading.Thread(target=run_simulation_background)
        thread.daemon = True
        thread.start()
        
        flash('Simulation started successfully', 'success')
        return redirect(url_for('simulation_status'))
        
    except Exception as e:
        flash(f'Error starting simulation: {str(e)}', 'error')
        return redirect(url_for('dashboard'))

@app.route('/simulation_status')
def simulation_status_page():
    """Show simulation status and results"""
    if 'current_json_file' not in session:
        flash('Please select a JSON file first', 'error')
        return redirect(url_for('index'))
    
    return render_template('simulation_status.html', 
                         status=simulation_status,
                         current_file=os.path.basename(get_current_json_file()))

@app.route('/simulation_status_api')
def simulation_status_api():
    """API endpoint for simulation status (for AJAX polling)"""
    global simulation_status
    
    # Calculate duration if simulation is running or completed
    duration = None
    if simulation_status['start_time']:
        end_time = simulation_status['end_time'] or datetime.now()
        duration = (end_time - simulation_status['start_time']).total_seconds()
    
    return jsonify({
        'running': simulation_status['running'],
        'completed': simulation_status['completed'],
        'error': simulation_status['error'],
        'duration': duration,
        'has_results': bool(simulation_status['results_file']),
        'has_plot': bool(simulation_status['plot_file']),
        'output_lines': len(simulation_status['output'].split('\n')) if simulation_status['output'] else 0
    })

@app.route('/simulation_output')
def simulation_output():
    """Get simulation output for real-time display"""
    return jsonify({'output': simulation_status['output']})

@app.route('/download_simulation_results')
def download_simulation_results():
    """Download the simulation results file"""
    if simulation_status['results_file'] and os.path.exists(simulation_status['results_file']):
        return send_file(simulation_status['results_file'], 
                        as_attachment=True, 
                        download_name='arm_thermal_results.txt')
    else:
        flash('No results file available', 'error')
        return redirect(url_for('simulation_status'))

@app.route('/view_simulation_plot')
def view_simulation_plot():
    """View the simulation temperature distribution plot"""
    if simulation_status['plot_file'] and os.path.exists(simulation_status['plot_file']):
        return send_file(simulation_status['plot_file'])
    else:
        flash('No plot file available', 'error')
        return redirect(url_for('simulation_status'))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 