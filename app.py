import streamlit as st
import numpy as np
from scipy.ndimage import gaussian_filter
import plotly.graph_objects as go
from time import sleep

class NeuralFieldExplorer:
    def __init__(self, size=100, time_depth=50):
        self.size = size
        self.time_depth = time_depth
        self.energy_flow_history = np.zeros((time_depth, size, size))
        
        # Field parameters
        self.u = np.zeros((size, size))
        self.v = np.zeros((size, size))
        self.phi = np.zeros((size, size))
        
        # Initialize central disturbance
        self.u[size//2, size//2] = 2.0
        
        # Physics parameters
        self.dt = 0.1
        self.dx = 1.0
        self.dy = 1.0
        self.c = 1.0
        self.alpha = 0.05
        self.beta = 0.02
    
    def update_fields(self):
        laplacian = (
            -4 * self.u +
            np.roll(self.u, 1, axis=0) +
            np.roll(self.u, -1, axis=0) +
            np.roll(self.u, 1, axis=1) +
            np.roll(self.u, -1, axis=1)
        ) / (self.dx * self.dy)
        
        quantum_input = np.random.normal(0, 0.1, (self.size, self.size))
        classical_input = np.zeros((self.size, self.size))
        
        a = self.c**2 * laplacian - self.beta * self.v - self.alpha * (self.u**3) + quantum_input + classical_input
        v_new = self.v + a * self.dt
        u_new = self.u + v_new * self.dt
        phi_new = self.phi + (v_new * self.dt)
        
        self.u, self.v, self.phi = u_new, v_new, phi_new
        
    def calculate_energy_flow(self):
        grad_x = np.gradient(self.u, axis=0)
        grad_y = np.gradient(self.u, axis=1)
        energy_flow = np.sqrt(grad_x**2 + grad_y**2)
        energy_flow = gaussian_filter(energy_flow, sigma=1)
        return (energy_flow - energy_flow.min()) / (energy_flow.max() - energy_flow.min() + 1e-8)
    
    def update_history(self, energy_flow):
        self.energy_flow_history = np.roll(self.energy_flow_history, -1, axis=0)
        self.energy_flow_history[-1] = energy_flow

    def create_3d_visualization(self):
        x, y = np.meshgrid(np.arange(self.size), np.arange(self.size))
        
        # Create empty lists for our surface plots
        surfaces = []
        
        # Create a surface for each time slice
        for i in range(0, self.time_depth, 2):
            z = i * np.ones_like(x)
            
            # Create surface with custom coloring
            surfaces.append(
                go.Surface(
                    x=x,
                    y=y,
                    z=z,
                    surfacecolor=self.energy_flow_history[i],
                    showscale=False,
                    opacity=0.3,
                    colorscale='Magma'
                )
            )
        
        return surfaces

def main():
    st.title("🧠 Neural Field Pattern Explorer")
    st.write("Exploring the 3D structure of neural field patterns in real-time!")
    
    # Initialize session state
    if 'explorer' not in st.session_state:
        st.session_state.explorer = NeuralFieldExplorer()
        st.session_state.frame_count = 0
    
    # Control panel
    col1, col2, col3 = st.columns(3)
    with col1:
        running = st.checkbox('Run Simulation', value=True)
    with col2:
        speed = st.slider('Animation Speed', 1, 10, 5)
    with col3:
        transparency = st.slider('Layer Transparency', 0.1, 1.0, 0.3)
    
    # Create placeholders for our visualizations
    plot3d = st.empty()
    plot2d = st.empty()
    
    # Main simulation loop
    while running:
        # Update fields
        st.session_state.explorer.update_fields()
        energy_flow = st.session_state.explorer.calculate_energy_flow()
        st.session_state.explorer.update_history(energy_flow)
        
        # Create 3D visualization
        surfaces = st.session_state.explorer.create_3d_visualization()
        
        # Update 3D plot
        fig3d = go.Figure(data=surfaces)
        fig3d.update_layout(
            title='3D Neural Field Patterns',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Time',
                camera=dict(
                    up=dict(x=0, y=0, z=1),
                    center=dict(x=0, y=0, z=0),
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            width=800,
            height=600
        )
        
        # Update 2D plot
        fig2d = go.Figure(data=go.Heatmap(
            z=energy_flow,
            colorscale='Magma'
        ))
        fig2d.update_layout(
            title='Current Energy Flow',
            width=400,
            height=400
        )
        
        # Display plots
        plot3d.plotly_chart(fig3d, use_container_width=True)
        plot2d.plotly_chart(fig2d, use_container_width=True)
        
        # Control animation speed
        sleep(1.0 / speed)
        
        st.session_state.frame_count += 1
        
        # Break if checkbox is unchecked
        if not running:
            break

if __name__ == "__main__":
    st.set_page_config(page_title="Neural Field Explorer", layout="wide")
    main()