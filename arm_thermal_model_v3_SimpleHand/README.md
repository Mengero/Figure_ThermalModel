# Thermal Model Editor

A Flask web application for editing thermal model configuration files with an easy-to-use interface.

## Features

- **Environment Settings**: Edit ambient temperature, heat transfer coefficient, and plastic conductivity
- **Regions Management**: View, edit, and manage thermal regions including:
  - Geometry (dimensions, centroid position)
  - Thermal properties (thermal conductivity)
  - Mesh settings (dx, dy, dz)
  - **Boundary Conditions**: Full editing support for all boundary condition types
- **Actuators Management**: View and edit actuator properties including:
  - Heat losses (gearbox, motor, FETs)
  - Thermal resistance values
- **Backup System**: Create backups of your configuration files
- **Modern UI**: Bootstrap-based responsive interface with intuitive navigation

## Installation

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Ensure your JSON file is present**:
   - The application looks for `GEO.json` in the same directory
   - The JSON file can contain comments (they will be handled automatically)

3. **Run the Application**:
   ```bash
   python app.py
   ```

4. **Access the Application**:
   - Open your browser and go to `http://localhost:5000`
   - Or `http://127.0.0.1:5000`

## Usage

### Main Dashboard
- Overview of environment settings, regions, and actuators
- Quick access to all major sections
- Summary statistics and quick actions

### Environment Settings
- Edit ambient temperature (°C)
- Modify heat transfer coefficient (W/m²K)
- Adjust plastic conductivity (W/mK)
- Real-time validation and unit display

### Regions Management

#### Viewing Regions
- List all thermal regions with key properties
- View detailed information for each region
- Access boundary conditions in expandable accordions

#### Editing Regions
- **Basic Properties**: Edit dimensions, centroid position, thermal conductivity, and mesh settings
- **Boundary Conditions**: Full editing support including:
  - **ACTUATOR_CONNECTED**: Link to actuators with specific connection locations
  - **CONST_Q**: Constant heat flux boundary conditions
  - **MAPPED**: Symmetry boundary conditions
  - **ADIABATIC**: Adiabatic boundary conditions with source definitions
  - **PLASTIC_COVERED**: Plastic covering with thickness specifications
  - **GEARBOX_HAND**: Special gearbox-hand boundary conditions

#### Boundary Condition Editing Features
- **Add New**: Use the "Add Boundary Condition" button to create new boundary conditions
- **Remove Existing**: Use the "Remove" button to delete boundary conditions
- **Type-Specific Fields**: Form fields automatically update based on selected boundary condition type
- **Dynamic Validation**: Real-time form validation with appropriate input types

### Actuators Management

#### Viewing Actuators
- List all actuators with type and heat loss information
- Detailed view with heat distribution visualization
- Thermal resistance summary

#### Editing Actuators
- **Heat Losses**: Edit gearbox, motor, and FETs heat losses (Watts)
- **Thermal Resistance**: Modify all thermal resistance values (K/W)
- **Real-time Totals**: Automatic calculation of total heat loss

### Backup System
- Create timestamped backups of your configuration
- Backup files are saved with format: `GEO_backup_YYYYMMDD_HHMMSS.json`
- Access via the "Backup" button in the navigation bar

## File Structure

```
.
├── app.py                      # Main Flask application
├── requirements.txt            # Python dependencies
├── GEO.json                   # Your thermal model configuration
├── templates/                 # HTML templates
│   ├── base.html             # Base template with navigation
│   ├── index.html            # Main dashboard
│   ├── environment.html      # Environment settings
│   ├── regions.html          # Regions list
│   ├── region_detail.html    # Region details view
│   ├── edit_region.html      # Region editing form
│   ├── actuators.html        # Actuators list
│   ├── actuator_detail.html  # Actuator details view
│   ├── edit_actuator.html    # Actuator editing form
│   └── error.html            # Error page
└── README.md                 # This file
```

## JSON File Format

The application handles JSON files with the following structure:

```json
{
  "environment": {
    "ambient_temperature": 40.0,
    "heat_transfer_coefficient": 15,
    "plastic_conductivity": 0.25
  },
  "regions": [
    {
      "id": "region_name",
      "centroid": {"x": 0, "y": 0, "z": 0},
      "width": 100,
      "height": 100,
      "thickness": 2,
      "thermal_conductivity": 120.0,
      "mesh_settings": {"dx": 5, "dy": 5, "dz": 0.5},
      "boundary_conditions": [...]
    }
  ],
  "actuators": [
    {
      "id": "actuator_name",
      "type": "PITCHYAW",
      "heat_losses": {"gearbox": 0, "motor": 0, "FETs": 0},
      "thermal_resistance": {"R1": 4.0, "R2": 3.8, ...}
    }
  ]
}
```

## Features in Detail

### Comment Handling
- The application automatically handles JSON comments (// style)
- Comments are preserved in the original file structure
- Safe parsing prevents JSON syntax errors

### Form Validation
- Real-time validation for all numeric inputs
- Required field validation
- Appropriate input types (number, text, select)
- Step values for precise decimal input

### Responsive Design
- Bootstrap-based UI works on desktop and mobile
- Intuitive navigation with breadcrumbs
- Modal dialogs and collapsible sections
- Progress bars for data visualization

### Error Handling
- Graceful error handling for file operations
- User-friendly error messages
- Automatic backup suggestions on save errors

## Troubleshooting

### Common Issues

1. **JSON Parsing Error**: 
   - Check that your JSON file doesn't have syntax errors
   - Ensure comments are properly formatted (// style)
   - Verify all brackets and commas are correctly placed

2. **File Not Found**:
   - Ensure `GEO.json` exists in the same directory as `app.py`
   - Check file permissions

3. **Port Already in Use**:
   - Change the port in `app.py`: `app.run(port=5001)`
   - Or kill the process using port 5000

### Development

To run in development mode:
```bash
export FLASK_ENV=development
python app.py
```

## Safety Features

- **Backup Creation**: Automatic timestamped backups
- **Input Validation**: Comprehensive form validation
- **Error Recovery**: Graceful error handling with user feedback
- **Data Integrity**: Careful JSON parsing and writing

## Browser Support

- Chrome (recommended)
- Firefox
- Safari
- Edge

## License

This project is for thermal model configuration management. Please ensure you have appropriate permissions for your thermal model files. 