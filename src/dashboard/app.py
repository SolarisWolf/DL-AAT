"""
Web-based dashboard for real-time grid monitoring.
"""
import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import numpy as np
from collections import deque
import time
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils import get_config, setup_logger
from src.iot.sensor_simulator import SensorNetwork


class GridMonitoringDashboard:
    """Interactive dashboard for smart grid monitoring."""
    
    def __init__(self, config=None):
        """
        Initialize dashboard.
        
        Args:
            config: Configuration object
        """
        self.config = config or get_config()
        self.logger = setup_logger("dashboard")
        
        # Initialize Dash app
        self.app = dash.Dash(__name__)
        self.app.title = "Smart Grid Fault Detection"
        
        # Initialize sensor network
        self.sensor_network = SensorNetwork(self.config)
        
        # Data buffers for real-time plotting
        self.max_points = self.config.get('visualization.plot_history_length', 100)
        self.voltage_data = {phase: deque(maxlen=self.max_points) for phase in ['A', 'B', 'C']}
        self.current_data = {phase: deque(maxlen=self.max_points) for phase in ['A', 'B', 'C']}
        self.time_data = deque(maxlen=self.max_points)
        
        # Fault detection results
        self.latest_detection = {
            'fault_type': 'Normal',
            'confidence': 1.0,
            'alert_level': 'INFO',
            'timestamp': time.time()
        }
        
        # Setup layout and callbacks
        self._setup_layout()
        self._setup_callbacks()
        
        self.logger.info("Dashboard initialized")
    
    def _setup_layout(self):
        """Setup dashboard layout with enhanced modern UI."""
        self.app.layout = html.Div([
            # Modern Header with Gradient
            html.Div([
                html.Div([
                    html.Div([
                        html.I(className='fas fa-bolt', style={'fontSize': '48px', 'color': '#f39c12', 'marginRight': '15px'}),
                        html.Div([
                            html.H1("Smart Energy Grid Fault Detection",
                                   style={'margin': '0', 'color': 'white', 'fontSize': '36px', 'fontWeight': '700', 'letterSpacing': '0.5px'}),
                            html.P("Real-time IoT Monitoring & CNN-based Fault Classification",
                                   style={'margin': '5px 0 0 0', 'color': 'rgba(255,255,255,0.9)', 'fontSize': '16px'})
                        ])
                    ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center'}),
                ], style={
                    'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                    'padding': '30px 20px',
                    'borderRadius': '15px',
                    'boxShadow': '0 10px 40px rgba(0,0,0,0.2)',
                    'marginBottom': '25px'
                })
            ]),
            
            # Enhanced Status Cards with Icons and Animations
            html.Div([
                # Grid Status Card
                html.Div([
                    html.Div([
                        html.Div('⚡', style={'fontSize': '40px', 'marginBottom': '10px'}),
                        html.H4("Grid Status", style={'color': '#34495e', 'margin': '0 0 15px 0', 'fontSize': '14px', 'textTransform': 'uppercase', 'letterSpacing': '1px', 'fontWeight': '600'}),
                        html.H2(id='grid-status', style={'color': '#27ae60', 'fontSize': '28px', 'margin': '0 0 10px 0', 'fontWeight': '700'}),
                        html.P(id='status-time', style={'color': '#95a5a6', 'fontSize': '12px', 'margin': '0'})
                    ], style={'textAlign': 'center'})
                ], style={
                    'backgroundColor': 'white',
                    'padding': '25px 20px',
                    'borderRadius': '15px',
                    'boxShadow': '0 4px 15px rgba(0,0,0,0.08)',
                    'width': '23%',
                    'display': 'inline-block',
                    'marginRight': '2%',
                    'transition': 'transform 0.3s ease, box-shadow 0.3s ease',
                    'border': '1px solid #ecf0f1'
                }),
                
                # Detected Fault Card
                html.Div([
                    html.Div([
                        html.Div('🔍', style={'fontSize': '40px', 'marginBottom': '10px'}),
                        html.H4("Detected Fault", style={'color': '#34495e', 'margin': '0 0 15px 0', 'fontSize': '14px', 'textTransform': 'uppercase', 'letterSpacing': '1px', 'fontWeight': '600'}),
                        html.H2(id='detected-fault', style={'fontSize': '26px', 'margin': '0 0 10px 0', 'fontWeight': '700'}),
                        html.P(id='fault-confidence', style={'color': '#95a5a6', 'fontSize': '12px', 'margin': '0'})
                    ], style={'textAlign': 'center'})
                ], style={
                    'backgroundColor': 'white',
                    'padding': '25px 20px',
                    'borderRadius': '15px',
                    'boxShadow': '0 4px 15px rgba(0,0,0,0.08)',
                    'width': '23%',
                    'display': 'inline-block',
                    'marginRight': '2%',
                    'transition': 'transform 0.3s ease',
                    'border': '1px solid #ecf0f1'
                }),
                
                # Alert Level Card
                html.Div([
                    html.Div([
                        html.Div('🚨', style={'fontSize': '40px', 'marginBottom': '10px'}),
                        html.H4("Alert Level", style={'color': '#34495e', 'margin': '0 0 15px 0', 'fontSize': '14px', 'textTransform': 'uppercase', 'letterSpacing': '1px', 'fontWeight': '600'}),
                        html.H2(id='alert-level', style={'fontSize': '28px', 'margin': '0 0 10px 0', 'fontWeight': '700'}),
                        html.P("Current System State", style={'color': '#95a5a6', 'fontSize': '12px', 'margin': '0'})
                    ], style={'textAlign': 'center'})
                ], style={
                    'backgroundColor': 'white',
                    'padding': '25px 20px',
                    'borderRadius': '15px',
                    'boxShadow': '0 4px 15px rgba(0,0,0,0.08)',
                    'width': '23%',
                    'display': 'inline-block',
                    'marginRight': '2%',
                    'transition': 'transform 0.3s ease',
                    'border': '1px solid #ecf0f1'
                }),
                
                # Active Sensors Card
                html.Div([
                    html.Div([
                        html.Div('📡', style={'fontSize': '40px', 'marginBottom': '10px'}),
                        html.H4("Active Sensors", style={'color': '#34495e', 'margin': '0 0 15px 0', 'fontSize': '14px', 'textTransform': 'uppercase', 'letterSpacing': '1px', 'fontWeight': '600'}),
                        html.H2(id='active-sensors', style={'color': '#3498db', 'fontSize': '28px', 'margin': '0 0 10px 0', 'fontWeight': '700'}),
                        html.P("Monitoring Network", style={'color': '#95a5a6', 'fontSize': '12px', 'margin': '0'})
                    ], style={'textAlign': 'center'})
                ], style={
                    'backgroundColor': 'white',
                    'padding': '25px 20px',
                    'borderRadius': '15px',
                    'boxShadow': '0 4px 15px rgba(0,0,0,0.08)',
                    'width': '23%',
                    'display': 'inline-block',
                    'transition': 'transform 0.3s ease',
                    'border': '1px solid #ecf0f1'
                })
            ], style={'marginBottom': '25px'}),

            # Enhanced KPIs Row
            html.Div([
                html.Div([
                    html.Div([
                        html.Span('📊', style={'fontSize': '32px', 'marginRight': '15px'}),
                        html.Div([
                            html.H4("Feature Drift", style={'color': '#34495e', 'margin': '0 0 8px 0', 'fontSize': '13px', 'textTransform': 'uppercase', 'letterSpacing': '1px', 'fontWeight': '600'}),
                            html.H2(id='drift-score', style={'fontSize': '26px', 'color': '#8e44ad', 'margin': '0', 'fontWeight': '700'}),
                        ])
                    ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '10px'}),
                    html.P("Mean shift vs baseline", style={'color': '#95a5a6', 'fontSize': '12px', 'margin': '0', 'textAlign': 'center'})
                ], style={
                    'backgroundColor': 'white',
                    'padding': '25px',
                    'borderRadius': '15px',
                    'boxShadow': '0 4px 15px rgba(0,0,0,0.08)',
                    'width': '32%',
                    'display': 'inline-block',
                    'marginRight': '2%',
                    'border': '1px solid #ecf0f1'
                }),
                html.Div([
                    html.Div([
                        html.Span('⚡', style={'fontSize': '32px', 'marginRight': '15px'}),
                        html.Div([
                            html.H4("Detection Latency", style={'color': '#34495e', 'margin': '0 0 8px 0', 'fontSize': '13px', 'textTransform': 'uppercase', 'letterSpacing': '1px', 'fontWeight': '600'}),
                            html.H2(id='latency-ms', style={'fontSize': '26px', 'color': '#16a085', 'margin': '0', 'fontWeight': '700'}),
                        ])
                    ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '10px'}),
                    html.P("Callback processing time", style={'color': '#95a5a6', 'fontSize': '12px', 'margin': '0', 'textAlign': 'center'})
                ], style={
                    'backgroundColor': 'white',
                    'padding': '25px',
                    'borderRadius': '15px',
                    'boxShadow': '0 4px 15px rgba(0,0,0,0.08)',
                    'width': '32%',
                    'display': 'inline-block',
                    'marginRight': '2%',
                    'border': '1px solid #ecf0f1'
                }),
                html.Div([
                    html.Div([
                        html.Span('🎯', style={'fontSize': '32px', 'marginRight': '15px'}),
                        html.Div([
                            html.H4("Model Accuracy", style={'color': '#34495e', 'margin': '0 0 8px 0', 'fontSize': '13px', 'textTransform': 'uppercase', 'letterSpacing': '1px', 'fontWeight': '600'}),
                            html.H2(id='model-accuracy', style={'fontSize': '26px', 'color': '#e67e22', 'margin': '0', 'fontWeight': '700'}),
                        ])
                    ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '10px'}),
                    html.P("Last 100 predictions", style={'color': '#95a5a6', 'fontSize': '12px', 'margin': '0', 'textAlign': 'center'})
                ], style={
                    'backgroundColor': 'white',
                    'padding': '25px',
                    'borderRadius': '15px',
                    'boxShadow': '0 4px 15px rgba(0,0,0,0.08)',
                    'width': '32%',
                    'display': 'inline-block',
                    'border': '1px solid #ecf0f1'
                })
            ], style={'marginBottom': '25px'}),
            
            # Voltage and Current Charts Side by Side
            html.Div([
                html.Div([
                    html.Div([
                        html.Span('📈', style={'fontSize': '24px', 'marginRight': '10px'}),
                        html.H3("Three-Phase Voltage Signals", style={'margin': '0', 'color': '#34495e', 'fontSize': '18px', 'fontWeight': '600'})
                    ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '15px'}),
                    dcc.Graph(id='voltage-graph', config={'displayModeBar': False})
                ], style={
                    'backgroundColor': 'white',
                    'padding': '25px',
                    'borderRadius': '15px',
                    'boxShadow': '0 4px 15px rgba(0,0,0,0.08)',
                    'width': '49%',
                    'display': 'inline-block',
                    'marginRight': '2%',
                    'border': '1px solid #ecf0f1'
                }),
                
                html.Div([
                    html.Div([
                        html.Span('📉', style={'fontSize': '24px', 'marginRight': '10px'}),
                        html.H3("Three-Phase Current Signals", style={'margin': '0', 'color': '#34495e', 'fontSize': '18px', 'fontWeight': '600'})
                    ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '15px'}),
                    dcc.Graph(id='current-graph', config={'displayModeBar': False})
                ], style={
                    'backgroundColor': 'white',
                    'padding': '25px',
                    'borderRadius': '15px',
                    'boxShadow': '0 4px 15px rgba(0,0,0,0.08)',
                    'width': '49%',
                    'display': 'inline-block',
                    'border': '1px solid #ecf0f1'
                })
            ], style={'marginBottom': '25px'}),

            # Confidence and Events Row
            html.Div([
                html.Div([
                    html.Div([
                        html.Span('📊', style={'fontSize': '24px', 'marginRight': '10px'}),
                        html.H3("Model Confidence Distribution", style={'margin': '0', 'color': '#34495e', 'fontSize': '18px', 'fontWeight': '600'})
                    ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '15px'}),
                    dcc.Graph(id='confidence-hist', config={'displayModeBar': False})
                ], style={
                    'backgroundColor': 'white',
                    'padding': '25px',
                    'borderRadius': '15px',
                    'boxShadow': '0 4px 15px rgba(0,0,0,0.08)',
                    'width': '49%',
                    'display': 'inline-block',
                    'marginRight': '2%',
                    'border': '1px solid #ecf0f1'
                }),
                
                html.Div([
                    html.Div([
                        html.Span('📋', style={'fontSize': '24px', 'marginRight': '10px'}),
                        html.H3("Event Timeline", style={'margin': '0', 'color': '#34495e', 'fontSize': '18px', 'fontWeight': '600'})
                    ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '15px'}),
                    html.Div(id='event-log', style={
                        'maxHeight': '280px',
                        'overflowY': 'auto',
                        'padding': '10px',
                        'backgroundColor': '#f8f9fa',
                        'borderRadius': '8px',
                        'fontSize': '13px',
                        'fontFamily': 'monospace'
                    })
                ], style={
                    'backgroundColor': 'white',
                    'padding': '25px',
                    'borderRadius': '15px',
                    'boxShadow': '0 4px 15px rgba(0,0,0,0.08)',
                    'width': '49%',
                    'display': 'inline-block',
                    'border': '1px solid #ecf0f1'
                })
            ], style={'marginBottom': '25px'}),
            
            # Footer
            html.Div([
                html.P([
                    '© 2025 Smart Grid Fault Detection System | ',
                    html.Span('Powered by CNN & IoT', style={'color': '#667eea', 'fontWeight': '600'}),
                    ' | Last Updated: ',
                    html.Span(id='footer-time', children=time.strftime("%Y-%m-%d %H:%M:%S"))
                ], style={'textAlign': 'center', 'color': '#95a5a6', 'fontSize': '13px', 'margin': '0'})
            ], style={
                'backgroundColor': 'white',
                'padding': '20px',
                'borderRadius': '15px',
                'boxShadow': '0 4px 15px rgba(0,0,0,0.08)',
                'border': '1px solid #ecf0f1'
            }),
            
            # Update interval
            dcc.Interval(
                id='interval-component',
                interval=self.config.get('visualization.update_interval', 1000),  # milliseconds
                n_intervals=0
            )
        ], style={
            'backgroundColor': '#f0f3f7',
            'padding': '25px',
            'fontFamily': '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif',
            'minHeight': '100vh'
        })
    
    def _setup_callbacks(self):
        """Setup dashboard callbacks for real-time updates."""
        
        # Maintain local buffers for histogram and events
        self.confidences = deque(maxlen=self.max_points)
        self.events = deque(maxlen=200)
        self.baseline_mean = None

        @self.app.callback(
            [Output('grid-status', 'children'),
             Output('grid-status', 'style'),
             Output('status-time', 'children'),
             Output('detected-fault', 'children'),
             Output('detected-fault', 'style'),
             Output('fault-confidence', 'children'),
             Output('alert-level', 'children'),
             Output('alert-level', 'style'),
             Output('active-sensors', 'children'),
             Output('voltage-graph', 'figure'),
             Output('current-graph', 'figure'),
             Output('confidence-hist', 'figure'),
             Output('event-log', 'children'),
             Output('drift-score', 'children'),
             Output('latency-ms', 'children'),
             Output('model-accuracy', 'children')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_dashboard(n):
            """Update all dashboard components."""
            t0 = time.time()
            # Simulate sensor readings (in real system, this would come from actual sensors)
            fault_type = np.random.choice(['Normal', 'Normal', 'Normal', 'AG', 'ABC'], p=[0.7, 0.1, 0.1, 0.05, 0.05])
            try:
                readings = self.sensor_network.simulate_readings(fault_type)
            except Exception as e:
                self.logger.error(f"Sensor simulation error: {e}")
                readings = {f'V_{p}': 0.0 for p in ['A','B','C']}
                readings.update({f'I_{p}': 0.0 for p in ['A','B','C']})
                readings['fault_type'] = 'Normal'
                readings['timestamp'] = time.time()
            
            # Update data buffers with type safety
            self.time_data.append(len(self.time_data))
            for phase in ['A', 'B', 'C']:
                v_val = readings.get(f'V_{phase}', 0.0)
                i_val = readings.get(f'I_{phase}', 0.0)
                self.voltage_data[phase].append(float(v_val) if v_val is not None else 0.0)
                self.current_data[phase].append(float(i_val) if i_val is not None else 0.0)
            
            # Simulate fault detection result
            self.latest_detection = {
                'fault_type': fault_type,
                'confidence': np.random.uniform(0.85, 0.99) if fault_type == 'Normal' else np.random.uniform(0.90, 0.99),
                'alert_level': 'INFO' if fault_type == 'Normal' else ('CRITICAL' if fault_type == 'ABC' else 'WARNING'),
                'timestamp': time.time()
            }
            self.confidences.append(self.latest_detection['confidence'])
            if fault_type != 'Normal':
                self.events.appendleft(f"{time.strftime('%H:%M:%S')} - {fault_type} (conf {self.latest_detection['confidence']:.2f})")
            
            # Grid status
            grid_status = "OPERATIONAL" if fault_type == 'Normal' else "FAULT DETECTED"
            grid_color = {'color': '#27ae60'} if fault_type == 'Normal' else {'color': '#e74c3c'}
            status_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.latest_detection['timestamp']))
            
            # Detected fault
            detected_fault = self.latest_detection['fault_type']
            fault_color = {'color': '#27ae60'} if fault_type == 'Normal' else {'color': '#e74c3c'}
            fault_confidence = f"Confidence: {self.latest_detection['confidence']:.1%}"
            
            # Alert level
            alert_level = self.latest_detection['alert_level']
            alert_colors = {'INFO': '#3498db', 'WARNING': '#f39c12', 'CRITICAL': '#e74c3c'}
            alert_color = {'color': alert_colors.get(alert_level, '#3498db')}
            
            # Active sensors
            active_sensors = f"{len(self.sensor_network.sensors)}/7"
            
            # Create enhanced voltage plot
            voltage_fig = go.Figure()
            colors = {'A': '#e74c3c', 'B': '#f39c12', 'C': '#3498db'}
            
            for phase in ['A', 'B', 'C']:
                voltage_fig.add_trace(go.Scatter(
                    x=list(self.time_data),
                    y=list(self.voltage_data[phase]),
                    mode='lines',
                    name=f'Phase {phase}',
                    line=dict(color=colors[phase], width=2.5),
                    hovertemplate='<b>Phase %{fullData.name}</b><br>Time: %{x}<br>Voltage: %{y:.2f}V<extra></extra>'
                ))
            
            voltage_fig.update_layout(
                xaxis=dict(
                    title='Time Steps',
                    showgrid=True,
                    gridcolor='rgba(0,0,0,0.05)',
                    zeroline=False
                ),
                yaxis=dict(
                    title='Voltage (V)',
                    showgrid=True,
                    gridcolor='rgba(0,0,0,0.05)',
                    zeroline=False
                ),
                legend=dict(
                    x=0.02,
                    y=0.98,
                    bgcolor='rgba(255,255,255,0.8)',
                    bordercolor='rgba(0,0,0,0.1)',
                    borderwidth=1
                ),
                margin=dict(l=60, r=30, t=10, b=50),
                height=300,
                hovermode='x unified',
                plot_bgcolor='rgba(248,249,250,0.5)',
                paper_bgcolor='white'
            )
            
            # Create enhanced current plot
            current_fig = go.Figure()
            
            for phase in ['A', 'B', 'C']:
                current_fig.add_trace(go.Scatter(
                    x=list(self.time_data),
                    y=list(self.current_data[phase]),
                    mode='lines',
                    name=f'Phase {phase}',
                    line=dict(color=colors[phase], width=2.5),
                    hovertemplate='<b>Phase %{fullData.name}</b><br>Time: %{x}<br>Current: %{y:.2f}A<extra></extra>'
                ))
            
            current_fig.update_layout(
                xaxis=dict(
                    title='Time Steps',
                    showgrid=True,
                    gridcolor='rgba(0,0,0,0.05)',
                    zeroline=False
                ),
                yaxis=dict(
                    title='Current (A)',
                    showgrid=True,
                    gridcolor='rgba(0,0,0,0.05)',
                    zeroline=False
                ),
                legend=dict(
                    x=0.02,
                    y=0.98,
                    bgcolor='rgba(255,255,255,0.8)',
                    bordercolor='rgba(0,0,0,0.1)',
                    borderwidth=1
                ),
                margin=dict(l=60, r=30, t=10, b=50),
                height=300,
                hovermode='x unified',
                plot_bgcolor='rgba(248,249,250,0.5)',
                paper_bgcolor='white'
            )

            # Enhanced confidence histogram
            conf_fig = go.Figure()
            if len(self.confidences) > 0:
                conf_fig.add_trace(go.Histogram(
                    x=list(self.confidences),
                    nbinsx=15,
                    marker=dict(
                        color='#2ecc71',
                        line=dict(color='#27ae60', width=1)
                    ),
                    hovertemplate='Confidence: %{x:.2f}<br>Count: %{y}<extra></extra>'
                ))
            conf_fig.update_layout(
                xaxis=dict(
                    title='Confidence Score',
                    showgrid=True,
                    gridcolor='rgba(0,0,0,0.05)'
                ),
                yaxis=dict(
                    title='Frequency',
                    showgrid=True,
                    gridcolor='rgba(0,0,0,0.05)'
                ),
                height=280,
                margin=dict(l=60, r=30, t=10, b=50),
                plot_bgcolor='rgba(248,249,250,0.5)',
                paper_bgcolor='white',
                bargap=0.1
            )

            # Drift score as mean absolute shift vs first 20 points baseline
            drift_text = "N/A"
            if self.baseline_mean is None and len(self.time_data) >= 20:
                try:
                    base = np.array([
                        list(self.voltage_data['A']),
                        list(self.voltage_data['B']),
                        list(self.voltage_data['C'])
                    ])
                    if base.shape[1] >= 20:
                        self.baseline_mean = base[:, -20:].mean(axis=1)
                except Exception as e:
                    self.logger.error(f"Baseline mean error: {e}")
                    self.baseline_mean = None
            
            if self.baseline_mean is not None and len(self.time_data) > 0:
                try:
                    cur = np.array([
                        list(self.voltage_data['A']),
                        list(self.voltage_data['B']),
                        list(self.voltage_data['C'])
                    ])
                    if cur.shape[1] > 0:
                        window = min(cur.shape[1], 20)
                        cur_mean = cur[:, -window:].mean(axis=1)
                        drift = float(np.mean(np.abs(cur_mean - self.baseline_mean)))
                        drift_text = f"{drift:.3f} V"
                except Exception as e:
                    self.logger.error(f"Drift calc error: {e}")
                    drift_text = "N/A"

            # Latency
            latency = (time.time() - t0) * 1000.0
            latency_text = f"{latency:.1f} ms"

            # Enhanced event children with styling
            event_children = []
            for i, e in enumerate(list(self.events)):
                event_color = '#e74c3c' if 'ABC' in str(e) else '#f39c12' if any(x in str(e) for x in ['AG', 'BG', 'CG', 'AB', 'BC', 'CA']) else '#95a5a6'
                event_children.append(
                    html.Div([
                        html.Span('●', style={'color': event_color, 'marginRight': '8px', 'fontSize': '14px'}),
                        html.Span(str(e), style={'color': '#34495e'})
                    ], style={
                        'padding': '8px 12px',
                        'marginBottom': '6px',
                        'backgroundColor': 'white',
                        'borderRadius': '6px',
                        'border': f'1px solid {event_color}',
                        'fontSize': '12px'
                    })
                )
            
            if not event_children:
                event_children = [html.Div('No fault events detected', style={'color': '#95a5a6', 'fontStyle': 'italic', 'textAlign': 'center', 'padding': '20px'})]
            
            # Calculate model accuracy (simulated)
            accuracy = np.mean(self.confidences) if len(self.confidences) > 0 else 0.95
            accuracy_text = f"{accuracy * 100:.1f}%"

            return (grid_status, grid_color, status_time,
                    detected_fault, fault_color, fault_confidence,
                    alert_level, alert_color, active_sensors,
                    voltage_fig, current_fig, conf_fig,
                    event_children,
                    drift_text, latency_text, accuracy_text)
    
    def run(self, host='0.0.0.0', port=None, debug=False):
        """
        Run the dashboard server.
        
        Args:
            host: Host address
            port: Port number
            debug: Debug mode
        """
        if port is None:
            port = self.config.get('visualization.dashboard_port', 8050)
        
        self.logger.info(f"Starting dashboard on http://{host}:{port}")
        # Dash >=3 uses app.run instead of deprecated run_server
        self.app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Smart Grid Monitoring Dashboard")
    parser.add_argument('--host', type=str, default='localhost',
                       help='Host address')
    parser.add_argument('--port', type=int, default=8050,
                       help='Port number')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    
    args = parser.parse_args()
    
    # Create and run dashboard
    dashboard = GridMonitoringDashboard()
    
    print("\n" + "="*70)
    print("SMART GRID MONITORING DASHBOARD")
    print("="*70)
    print(f"\nDashboard URL: http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop the server")
    print("="*70 + "\n")
    
    dashboard.run(host=args.host, port=args.port, debug=args.debug)
