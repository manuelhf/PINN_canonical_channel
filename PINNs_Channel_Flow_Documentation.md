# Physics-Informed Neural Networks (PINNs) for Channel Flow

## Complete Documentation and User Guide

---

## 📋 Table of Contents

1. [Overview](#1-overview)
2. [Physical Model](#2-physical-model)
3. [Mathematical Formulation](#3-mathematical-formulation)
4. [Implementation Details](#4-implementation-details)
5. [Usage Guide](#5-usage-guide)
6. [Parameter Tuning](#6-parameter-tuning)
7. [Troubleshooting](#7-troubleshooting)
8. [Extensions and Modifications](#8-extensions-and-modifications)
9. [Comparison with Rainfall-Runoff Model](#9-comparison-with-rainfall-runoff-model)
10. [References](#10-references)

---

## 1. Overview

### 1.1 Purpose

This notebook implements a **Physics-Informed Neural Network (PINN)** to simulate channel flow - the startup and propagation of water flowing through a straight rectangular channel. This scenario is relevant for:
- **Canal systems**: Irrigation and water distribution
- **Simplified river reaches**: Basic river hydraulics
- **Laboratory flumes**: Experimental hydraulics setup
- **Hydraulic engineering**: Design and analysis

### 1.2 Key Features

- **Inflow Boundary**: Constant velocity inlet at upstream end
- **Channel Geometry**: Rectangular cross-section (6m × 1m)
- **Flat Bed**: Horizontal terrain (no slope)
- **Wall Boundaries**: Side walls and downstream wall (no-penetration)
- **Startup Simulation**: Initially dry channel that gradually fills
- **Steady Flow Development**: Evolution from dry to steady uniform flow

### 1.3 Simulation Scenario

```
Domain:          6m × 1m × 120s (length × width × time)
Channel:         3:1 aspect ratio
Terrain:         Flat bed at z=0
Inflow:          0.5 m/s at x=0
Initial State:   Nearly dry bed (1 μm water depth)
Boundaries:      Inlet (u=0.5), walls (v=0), outlet (u=0)
```

### 1.4 Differences from Rainfall-Runoff Model

| Feature | Rainfall-Runoff | Channel Flow |
|---------|-----------------|--------------|
| Driving force | Rainfall source | Inflow boundary |
| Terrain | 5% inclined plane | Flat horizontal bed |
| Domain | 1m × 1m (square) | 6m × 1m (rectangular) |
| Aspect ratio | 1:1 | 6:1 |
| Time scale | 300s (5 min) | 120s (2 min) |
| BC at x=0 | Wall (u=0) | Inflow (u=0.5) |
| Physical process | Overland flow | Open channel flow |

---

## 2. Physical Model

### 2.1 Phenomenon

The model simulates **open channel flow** - water flowing with a free surface in an open conduit. This scenario demonstrates:

- **Flow startup**: How a channel fills from an initially dry state
- **Wave propagation**: How the inflow disturbance travels downstream
- **Steady state development**: Evolution toward uniform flow
- **Friction effects**: How bed resistance affects flow depth and velocity

### 2.2 Governing Equations

#### 2.2.1 Shallow Water Equations (SWEs) - Channel Form

The 2D shallow water equations govern the flow:

**Continuity Equation** (Mass Conservation):
```
∂h/∂t + ∂(hu)/∂x + ∂(hv)/∂y = 0
```

Expanded form:
```
∂h/∂t + u·∂h/∂x + h·∂u/∂x + v·∂h/∂y + h·∂v/∂y = 0
```

**X-Momentum Equation**:
```
∂u/∂t + u·∂u/∂x + v·∂u/∂y + g·∂h/∂x + f·u = 0
```

**Y-Momentum Equation**:
```
∂v/∂t + u·∂v/∂x + v·∂v/∂y + g·∂h/∂y + f·v = 0
```

**Key Simplifications for this scenario**:
- No rainfall term (R = 0)
- Flat bed (∂z/∂x = 0, ∂z/∂y = 0)
- Linear friction model

#### 2.2.2 Variable Definitions

| Symbol | Description | Units | Typical Range |
|--------|-------------|-------|---------------|
| h(x,y,t) | Water depth | m | 0 - 0.3 |
| u(x,y,t) | X-velocity (downstream) | m/s | 0 - 0.6 |
| v(x,y,t) | Y-velocity (cross-stream) | m/s | -0.1 to 0.1 |
| z(x,y) | Terrain elevation | m | 0 (flat) |
| ζ(x,y,t) | Water surface elevation | m | 0 - 0.3 |
| g | Gravity | m/s² | 9.81 |
| f | Friction coefficient | 1/s | 0.01 - 0.02 |

### 2.3 Channel Geometry

#### 2.3.1 Rectangular Channel

```
Length (x):  6.0 m (flow direction)
Width (y):   1.0 m (cross-stream)
Bed level:   z = 0 m (flat horizontal bed)
```

**Aspect Ratio**: 6:1 (long narrow channel)

**Why this geometry?**
- Long enough for flow development
- Narrow enough to approximate 1D flow
- Practical for laboratory/canal applications

#### 2.3.2 Terrain Definition

```python
z(x,y) = 0  (constant flat bed)
∂z/∂x = 0  (no longitudinal slope)
∂z/∂y = 0  (no cross slope)
```

This simplifies the momentum equations significantly.

### 2.4 Boundary Conditions

#### 2.4.1 Inlet Boundary (x=0)

**Type**: Dirichlet velocity boundary
```
u(x=0, y, t) = u_inflow = 0.5 m/s  (constant inflow velocity)
v(x=0, y, t) = 0                    (no cross-stream flow)
```

This represents:
- Pump or reservoir discharge
- Upstream control structure
- Constant flow rate input

#### 2.4.2 Outlet Boundary (x=6m)

**Type**: Wall boundary (no penetration)
```
u(x=6, y, t) = 0  (no outflow - water accumulates)
```

**Note**: This is physically unrealistic for a true outlet but serves as a simplification. In reality, you'd have an outflow or open boundary condition.

#### 2.4.3 Side Wall Boundaries (y=0, y=1m)

**Type**: Wall boundaries
```
v(x, y=0, t) = 0   (no flow through bottom wall)
v(x, y=1, t) = 0   (no flow through top wall)
```

These represent solid channel walls.

#### 2.4.4 BC Loss Implementation

```python
L_BC = (1/4) * [
    MSE(u|_{x=0} - u_inflow) +  # Inlet velocity
    MSE(u|_{x=6})               # Outlet wall
    MSE(v|_{y=0})               # Bottom wall
    MSE(v|_{y=1})               # Top wall
]
```

### 2.5 Initial Conditions

At t=0:
- **Water depth**: h(x,y,0) = 1×10⁻⁶ m (nearly dry channel)
- **X-velocity**: u(x,y,0) = 0 (initially at rest)
- **Y-velocity**: v(x,y,0) = 0 (initially at rest)
- **Water surface**: ζ(x,y,0) = 0 + 1×10⁻⁶ (just above bed)

The simulation shows how the channel fills and flow develops from this dry initial state.

### 2.6 Expected Flow Behavior

#### 2.6.1 Startup Phase (0-20s)

- Water enters from inlet
- Advancing front moves downstream
- Depth increases near inlet
- Backwater effect develops

#### 2.6.2 Development Phase (20-60s)

- Flow reaches downstream end
- Water level rises throughout channel
- Velocity profile adjusts
- Depth gradients form

#### 2.6.3 Quasi-Steady Phase (60-120s)

- Approaching equilibrium
- Water continues to accumulate (due to outlet wall)
- Depth increases slowly
- Near-uniform flow in central section

---

## 3. Mathematical Formulation

### 3.1 Neural Network Architecture

#### 3.1.1 Network Structure

Same architecture as rainfall-runoff model:

```
NN: (x̂, ŷ, t̂) → (ζ, u, v)
```

**Layer Details**:
```
Input:  3 neurons  → (x̂, ŷ, t̂) ∈ [-1, 1]³
Layer 1: 64 neurons + Tanh
Layer 2: 64 neurons + Tanh
Layer 3: 64 neurons + Tanh
Layer 4: 64 neurons + Tanh
Output: 3 neurons  → (ζ, u, v)
```

**Total Parameters**: ~17,000

#### 3.1.2 Why Same Architecture?

The network architecture is general-purpose for solving 2D+time PDEs. The physics (different BCs, no rainfall) are encoded in the loss function, not the network structure.

### 3.2 Loss Function

#### 3.2.1 Total Loss

```
L_total = w_pde·L_pde + w_ic·L_ic + w_bc·L_bc + w_phys·L_phys
```

Default weights (different from rainfall-runoff!):
- w_pde = 50.0 (higher! - more emphasis on PDE satisfaction)
- w_ic = 150.0 (same)
- w_bc = 150.0 (same - crucial for inlet condition)
- w_phys = 15.0 (slightly lower)

**Why higher w_pde?**
The inflow boundary is a strong forcing - we need strong PDE enforcement to propagate this correctly throughout the domain.

#### 3.2.2 PDE Residual Loss (L_pde)

```
L_pde = (1/N_coll) Σ (r₁² + r₂² + r₃²)
```

**Continuity residual**:
```
r₁ = ∂h/∂t + u·∂h/∂x + h·∂u/∂x + v·∂h/∂y + h·∂v/∂y
```
(No rainfall term compared to runoff model)

**Momentum residuals**:
```
r₂ = ∂u/∂t + u·∂u/∂x + v·∂u/∂y + g·∂h/∂x + f·u
r₃ = ∂v/∂t + u·∂v/∂x + v·∂v/∂y + g·∂h/∂y + f·v
```
(No terrain slope terms since bed is flat)

#### 3.2.3 Boundary Condition Loss (L_bc)

```
L_bc = (1/4) * [
    MSE(u|_{x=0} - u_inflow)² +  # Enforces inflow velocity
    MSE(u|_{x=6})² +              # Enforces outlet wall
    MSE(v|_{y=0})² +              # Enforces bottom wall
    MSE(v|_{y=1})²                # Enforces top wall
]
```

**Key difference**: Inlet BC enforces `u = u_inflow` rather than `u = 0`.

### 3.3 Coordinate Normalization

#### 3.3.1 Scaling Factors (Different from runoff!)

```python
scale_x = 2.0 / 6.0 = 0.333  (longer domain → smaller scale)
scale_y = 2.0 / 1.0 = 2.0    (same as runoff)
scale_t = 2.0 / 120.0 = 0.0167  (shorter time → larger scale)
```

These reflect the different domain dimensions.

#### 3.3.2 Impact on Training

The x-dimension is 6× longer, so:
- Physical x-derivatives are ~6× smaller
- This affects the relative importance of terms
- May require adjusting loss weights

### 3.4 Training Strategy

#### 3.4.1 Key Differences from Rainfall-Runoff

| Parameter | Rainfall-Runoff | Channel Flow | Reason |
|-----------|-----------------|--------------|---------|
| w_pde | 1.0 | 50.0 | Stronger PDE enforcement needed |
| N_collocation | 10000 | 10000 | Same |
| n_epochs | 10000 | 10000 | Similar complexity |
| grid_res | 50×50 | 90×30 | Match aspect ratio |

#### 3.4.2 Grid Resolution

```python
grid_res_x = 90  # More points along length
grid_res_y = 30  # Fewer points across width
```

This matches the 3:1 aspect ratio and provides adequate resolution in both directions.

---

## 4. Implementation Details

### 4.1 Code Structure

Same structure as rainfall-runoff model:

```
1. Imports and Setup
2. Domain Configuration
3. Terrain Definition (flat bed)
4. Neural Network Model
5. PDE Residual Function
6. Training Function
7. Visualization Functions
8. Main Execution
```

### 4.2 Key Functions

#### 4.2.1 `create_terrain_torch(X, Y)`

**Channel version** (simplified):
```python
def create_terrain_torch(X, Y):
    terrain = torch.zeros_like(X)  # Flat bed at z=0
    return terrain
```

Much simpler than the inclined plane!

#### 4.2.2 `shallow_water_residuals(...)`

Key differences from runoff version:
- **No rainfall term** in continuity equation
- **No terrain slopes** (z_x = 0, z_y = 0)
- Otherwise identical structure

#### 4.2.3 `train_pinn(...)`

Key changes:
- **Different BC loss**: Enforces `u = u_inflow` at inlet
- **Parameter**: `u_inflow` instead of `R_const`
- **Loss weights**: Higher w_pde (50.0 vs 1.0)

#### 4.2.4 Visualization Functions

**Modified for aspect ratio**:
```python
aspect_ratio = 6.0 / 1.0 = 6.0
fig_width = 10
fig_height = fig_width / aspect_ratio + 1.5
```

Ensures the channel looks correctly proportioned.

### 4.3 Computational Requirements

#### 4.3.1 Similar to Rainfall-Runoff

- GPU: NVIDIA with 4+ GB VRAM recommended
- Training time: ~15-20 minutes (10,000 epochs)
- Memory usage: ~2 GB GPU

#### 4.3.2 Differences

- Longer x-domain (6m vs 1m)
- But same number of collocation points
- Similar training time despite larger domain

---

## 5. Usage Guide

### 5.1 Quick Start

1. **Open notebook** in Jupyter or Google Colab
2. **Run all cells** (Runtime → Run all)
3. **Wait ~15-20 minutes** for training
4. **View animations** of flow development

### 5.2 Modifying Parameters

#### 5.2.1 Inflow Velocity

```python
# In "Main Execution" section
u_inflow = 0.3  # Reduce from 0.5 m/s (slower flow)
u_inflow = 0.8  # Increase from 0.5 m/s (faster flow)
```

**Effects**:
- Higher velocity → Deeper water, faster propagation
- Lower velocity → Shallower water, slower filling

#### 5.2.2 Friction

```python
friction_factor = 0.005  # Low friction (smooth channel)
friction_factor = 0.030  # High friction (rough channel)
```

**Effects**:
- Higher friction → Slower flow, more depth variation
- Lower friction → Faster flow, more uniform depth

#### 5.2.3 Channel Length

```python
domain = {
    'x_max': 3.0,  # Shorter channel (was 6.0)
    'x_max': 10.0, # Longer channel (was 6.0)
}
```

Remember to adjust:
- Grid resolution: `grid_res_x`
- Training epochs (longer domain may need more)

#### 5.2.4 Channel Width

```python
domain = {
    'y_max': 0.5,  # Narrower channel (was 1.0)
    'y_max': 2.0,  # Wider channel (was 1.0)
}
```

Affects:
- Aspect ratio
- Cross-stream dynamics
- Figure proportions

### 5.3 Advanced Modifications

#### 5.3.1 Sloped Channel Bed

Modify terrain function:
```python
def create_terrain_torch(X, Y):
    slope = 0.001  # Mild slope (0.1%)
    z = -slope * X  # Descending slope in flow direction
    return z
```

Don't forget to uncomment terrain slope calculations in residuals!

#### 5.3.2 Time-Varying Inflow

```python
# In train_pinn(), modify BC loss calculation
# For example, sinusoidal inflow:
u_target = u_inflow * (1.0 + 0.3 * torch.sin(2*np.pi*t_bc/60.0))
BC_loss += torch.mean((u_in - u_target)**2)
```

#### 5.3.3 Open Outlet Boundary

Replace wall outlet with open boundary:
```python
# Instead of u=0 at x=xmax, use:
# ∂u/∂x = 0 (zero gradient outflow)
u_x_outlet = torch.autograd.grad(u_outlet, x_outlet, ...)[0]
BC_loss += torch.mean(u_x_outlet**2)
```

---

## 6. Parameter Tuning

### 6.1 Loss Weight Guidelines

#### 6.1.1 PDE Weight (w_pde)

**This is critical for channel flow!**

```python
w_pde = 50.0  # Default (recommended)
```

**If flow doesn't propagate properly**:
```python
w_pde = 100.0  # Increase to enforce physics more strongly
```

**If training is unstable**:
```python
w_pde = 20.0  # Reduce slightly
w_ic = 200.0  # Compensate with stronger IC
```

#### 6.1.2 Boundary Condition Weight (w_bc)

```python
w_bc = 150.0  # Default
```

**If inlet velocity not satisfied**:
```python
w_bc = 300.0  # Increase to enforce BC more strongly
N_bc = 4000   # Also increase sampling
```

**Monitor during training**: BC loss should decrease to < 1e-4.

### 6.2 Physical Parameter Tuning

#### 6.2.1 Friction Coefficient

```python
friction_factor = 0.015  # Default
```

**Typical values**:
- Smooth concrete: 0.005-0.010
- Rough concrete: 0.010-0.020
- Natural channel: 0.020-0.050

Choose based on channel material.

#### 6.2.2 Inflow Velocity

```python
u_inflow = 0.5  # Default (0.5 m/s)
```

**Froude number consideration**:
```
Fr = u / sqrt(g*h)
```

- Fr < 1: Subcritical flow (typical)
- Fr = 1: Critical flow
- Fr > 1: Supercritical flow (harder to simulate)

For stable simulation, keep Fr < 0.5.

### 6.3 Domain and Resolution

#### 6.3.1 Grid Resolution vs. Domain Size

**Rule of thumb**:
```
grid_points_per_meter_x ≈ 15
grid_points_per_meter_y ≈ 30
```

For 6m × 1m domain:
```python
grid_res_x = 6 * 15 = 90
grid_res_y = 1 * 30 = 30
```

#### 6.3.2 Collocation Points

```python
N_collocation = 10000  # Default
```

**For larger domains**:
```python
# Scale with domain volume
domain_volume = length * width * time
N_collocation = int(1400 * domain_volume)  # Heuristic
```

---

## 7. Troubleshooting

### 7.1 Common Issues

#### 7.1.1 Flow Doesn't Propagate

**Symptoms**:
- Water stays near inlet
- Depth doesn't increase downstream
- Velocity dies quickly

**Causes & Solutions**:
```python
# Cause 1: w_pde too low
w_pde = 100.0  # Increase from 50.0

# Cause 2: Friction too high
friction_factor = 0.010  # Reduce from 0.015

# Cause 3: Not enough training
n_epochs = 15000  # Increase from 10000
```

#### 7.1.2 Inlet BC Not Satisfied

**Symptoms**:
- u ≠ u_inflow at x=0
- BC_loss remains high (> 1e-3)

**Solutions**:
```python
# Increase BC weight
w_bc = 300.0

# More BC sampling
N_bc = 4000

# Check learning rate (may be too high)
learning_rate = 2e-4  # Reduce from 5e-4
```

#### 7.1.3 Unrealistic Depth Build-Up

**Symptoms**:
- Water depth increases excessively
- No equilibrium reached
- Unphysical accumulation

**Causes**:
- Outlet is a wall (water can't leave!)
- This is actually expected behavior

**Solutions**:
```python
# Option 1: Shorter simulation time
t_max = 60.0  # Look at earlier times

# Option 2: Implement open outlet BC (see Section 5.3.3)

# Option 3: Add a drain/sink term
# (beyond scope of current model)
```

#### 7.1.4 Non-Zero Cross-Stream Velocity

**Symptoms**:
- v ≠ 0 in channel interior
- Unrealistic circulation patterns

**Solutions**:
```python
# Increase side wall BC weight
w_bc = 200.0

# More wall BC points
N_bc = 5000

# Check if domain is too wide
# (narrow channels should have v ≈ 0)
```

### 7.2 Validation Checks

#### 7.2.1 Mass Balance

**Expected**: Water volume should increase at rate = u_inflow × width × depth

```python
# After training
inflow_rate = u_inflow * domain['y_max'] * avg_inlet_depth
volume_increase = total_volume_at_t / t

print(f"Inflow rate: {inflow_rate:.4f} m³/s")
print(f"Volume increase rate: {volume_increase:.4f} m³/s")
# Should be similar
```

#### 7.2.2 Velocity Profile

**Expected**: u should be ~uniform across width y for straight channel

```python
# Check velocity at mid-length
x_mid = domain['x_max'] / 2
u_profile_at_x_mid = u_np[:, grid_res_x//2]

# Should be relatively flat (small variation in y)
std_u = np.std(u_profile_at_x_mid)
print(f"Cross-stream velocity variation: {std_u:.4f} m/s")
# Should be < 0.1 * u_inflow
```

#### 7.2.3 Froude Number

**Expected**: Fr < 1 for subcritical flow

```python
Fr = u_np / np.sqrt(9.81 * h_np)
max_Fr = np.max(Fr[h_np > 1e-3])  # Where depth is significant
print(f"Maximum Froude number: {max_Fr:.3f}")
# Should be < 1 (typically 0.2-0.5)
```

---

## 8. Extensions and Modifications

### 8.1 Physics Extensions

#### 8.1.1 Add Bed Slope

```python
def create_terrain_torch(X, Y):
    S0 = 0.001  # 0.1% slope
    z = domain['y_max'] * S0 - S0 * X  # Mild downstream slope
    return z

# In shallow_water_residuals, compute slopes:
z_x = torch.autograd.grad(terrain_c, Xc, ...)[0]
# (already set up, just uncomment)
```

#### 8.1.2 Manning Friction

Replace linear friction:

```python
def shallow_water_residuals(...):
    # Manning friction
    n = 0.02  # Manning's n
    velocity_mag = torch.sqrt(u**2 + v**2 + 1e-12)
    friction_x = -g * n**2 * u * velocity_mag / (h_safe**(4/3))
    friction_y = -g * n**2 * v * velocity_mag / (h_safe**(4/3))
    
    r2 = u_t + u*u_x + v*u_y + g*h_x + friction_x
    r3 = v_t + u*v_x + v*v_y + g*h_y + friction_y
```

#### 8.1.3 Contraction/Expansion

Create varying width:

```python
def channel_width(x):
    # Linear contraction
    w0 = 1.0  # Inlet width
    w1 = 0.5  # Outlet width
    return w0 + (w1 - w0) * x / domain['x_max']

# Modify domain and BCs accordingly
```

#### 8.1.4 Obstacles

Add cylindrical obstacle:

```python
def create_terrain_torch(X, Y):
    # Obstacle at x=3, y=0.5
    x_obs, y_obs = 3.0, 0.5
    r_obs = 0.2  # Radius
    
    dist = torch.sqrt((X - x_obs)**2 + (Y - y_obs)**2)
    obstacle_height = 0.5  # Height above bed
    
    z = torch.where(dist < r_obs, obstacle_height, 0.0)
    return z
```

### 8.2 Boundary Condition Extensions

#### 8.2.1 Time-Varying Inflow

Implement hydrograph:

```python
def inflow_velocity(t):
    # Rising and falling limb
    t_peak = 60.0
    u_peak = 1.0
    if t < t_peak:
        return u_peak * (t / t_peak)  # Rising
    else:
        return u_peak * np.exp(-(t - t_peak) / 30.0)  # Falling

# In BC loss:
u_target = inflow_velocity(t_bc)
BC_loss += torch.mean((u_in - u_target)**2)
```

#### 8.2.2 Free Outflow BC

Zero-gradient outflow:

```python
# At outlet (x=xmax):
# Instead of u=0, use ∂u/∂x = 0

# Requires changes to BC loss calculation:
x_outlet = torch.full_like(y_bc_wall, domain['x_max'])
x_outlet.requires_grad_(True)

# ... (setup normalized coords)

_, u_outlet, _ = model(inp_outlet).split(1, dim=1)
u_x_outlet = torch.autograd.grad(u_outlet, x_outlet, ...)[0]
BC_loss += torch.mean(u_x_outlet**2)  # Enforce zero gradient
```

#### 8.2.3 Weir Outlet

Critical depth condition:

```python
# At outlet: u² = g*h (Froude number = 1)
_, u_out, _ = model(inp_outlet).split(1, dim=1)
zeta_out, _, _ = model(inp_outlet).split(1, dim=1)
h_out = zeta_out - terrain_outlet

froude_sq = u_out**2 - 9.81 * h_out
BC_loss += torch.mean(froude_sq**2)  # Enforce Fr=1
```

### 8.3 Geometry Extensions

#### 8.3.1 Curved Channel

```python
# Mildly curved centerline
def channel_centerline_y(x):
    amplitude = 0.1  # Curvature amplitude
    wavelength = 6.0  # One full wavelength
    return 0.5 + amplitude * np.sin(2*np.pi*x / wavelength)

# Adjust BCs to follow curved walls
```

#### 8.3.2 Trapezoidal Cross-Section

Model side slopes:

```python
def create_terrain_torch(X, Y):
    # Side slope m:1 (horizontal:vertical)
    m = 2.0  # 2:1 slope
    y_center = domain['y_max'] / 2
    bed_width = 0.4  # Flat bed width
    
    # Distance from centerline
    dist_from_center = torch.abs(Y - y_center)
    
    # Above bed width, terrain rises
    z = torch.where(
        dist_from_center > bed_width/2,
        (dist_from_center - bed_width/2) / m,
        torch.zeros_like(X)
    )
    return z
```

#### 8.3.3 Compound Channel

Floodplain and main channel:

```python
def create_terrain_torch(X, Y):
    # Main channel: y ∈ [0.3, 0.7], depth 0
    # Floodplain: y ∈ [0, 0.3] ∪ [0.7, 1.0], depth 0.2
    
    y_center = domain['y_max'] / 2
    channel_width = 0.4
    floodplain_elevation = 0.2
    
    in_main_channel = torch.abs(Y - y_center) < channel_width/2
    z = torch.where(in_main_channel, 0.0, floodplain_elevation)
    return z
```

### 8.4 Computational Extensions

#### 8.4.1 Adaptive Weighting

Automatically adjust loss weights during training:

```python
# Track loss components
pde_losses = []
bc_losses = []

# Every 1000 epochs:
if epoch % 1000 == 0 and epoch > 0:
    avg_pde = np.mean(pde_losses[-10:])
    avg_bc = np.mean(bc_losses[-10:])
    
    # Rebalance if one is dominating
    if avg_pde > 10 * avg_bc:
        w_bc *= 1.5  # Increase BC weight
    elif avg_bc > 10 * avg_pde:
        w_pde *= 1.5  # Increase PDE weight
```

#### 8.4.2 Multi-Stage Training

Train in stages with different emphases:

```python
# Stage 1: Focus on ICs and BCs (0-2000 epochs)
weights_stage1 = {'w_pde': 10.0, 'w_ic': 300.0, 'w_bc': 300.0, 'w_phys': 50.0}
train_pinn(model, ..., n_epochs=2000, **weights_stage1)

# Stage 2: Focus on PDEs (2000-10000 epochs)
weights_stage2 = {'w_pde': 100.0, 'w_ic': 50.0, 'w_bc': 100.0, 'w_phys': 20.0}
train_pinn(model, ..., n_epochs=8000, **weights_stage2)
```

#### 8.4.3 Ensemble Predictions

Train multiple networks and average:

```python
num_models = 5
models = [FloodNet().to(device) for _ in range(num_models)]

# Train each with different random seed
for i, model in enumerate(models):
    torch.manual_seed(i)
    train_pinn(model, ...)

# Average predictions
with torch.no_grad():
    predictions = [model(input) for model in models]
    zeta_ensemble = torch.mean(torch.stack([p[0] for p in predictions]), dim=0)
    # More robust predictions with uncertainty quantification
```

---

## 9. Comparison with Rainfall-Runoff Model

### 9.1 Key Differences Summary

| Aspect | Rainfall-Runoff | Channel Flow |
|--------|-----------------|--------------|
| **Driving Force** | Distributed rainfall source | Boundary inflow |
| **Terrain** | 5% inclined plane | Flat horizontal bed |
| **Domain Shape** | 1m × 1m square | 6m × 1m rectangle |
| **Aspect Ratio** | 1:1 | 6:1 |
| **Duration** | 300s (5 min) | 120s (2 min) |
| **Primary BC** | All walls (u=v=0) | Inlet velocity (u=0.5) |
| **Flow Type** | Overland sheet flow | Confined channel flow |
| **Physical Process** | Runoff generation | Open channel hydraulics |
| **PDE Terms** | + Rainfall source, + Terrain slopes | No source, No slopes |
| **w_pde** | 1.0 | 50.0 |
| **Application** | Hillslope hydrology | Canal/river hydraulics |

### 9.2 Which Model to Use When?

#### 9.2.1 Use Rainfall-Runoff Model for:
- Hillslope hydrology studies
- Urban drainage (sheet flow on streets)
- Agricultural runoff
- Watershed-scale overland flow
- Scenarios with distributed source/sink terms
- Sloped terrain applications

#### 9.2.2 Use Channel Flow Model for:
- Open channel hydraulics
- Canal and flume design
- Laboratory experiments
- River reach modeling (simplified)
- Dam break scenarios (with modifications)
- Flow under gates/weirs (with modifications)

### 9.3 Can You Combine Them?

**Yes!** You can create a coupled model:

```python
# Domain with both hillslope and channel
# - Upper region: Rainfall + slope
# - Lower region: Channel + inflow from hillslope

def create_terrain_torch(X, Y):
    # Hillslope region (x < 3)
    slope_region = X < 3.0
    slope = 0.05
    z_slope = 0.15 - slope * X
    
    # Channel region (x >= 3)
    z_channel = 0.0
    
    z = torch.where(slope_region, z_slope, z_channel)
    return z

# Rainfall only on hillslope
R_source = torch.where(Xc < 3.0, R_const, 0.0)
```

This creates a hillslope-to-channel continuum!

---

## 10. References

### 10.1 Open Channel Flow Theory

1. Chow, V. T. (1959). *Open-Channel Hydraulics*. McGraw-Hill. (Classic textbook)

2. Henderson, F. M. (1966). *Open Channel Flow*. Macmillan.

3. French, R. H. (1985). *Open-Channel Hydraulics*. McGraw-Hill.

### 10.2 Shallow Water Equations

1. Toro, E. F. (2001). *Shock-Capturing Methods for Free-Surface Shallow Flows*. Wiley.

2. Vreugdenhil, C. B. (1994). *Numerical Methods for Shallow-Water Flow*. Springer.

3. LeVeque, R. J. (2002). *Finite Volume Methods for Hyperbolic Problems*. Cambridge.

### 10.3 Physics-Informed Neural Networks

1. Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). *Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations*. Journal of Computational Physics, 378, 686-707.

2. Karniadakis, G. E., et al. (2021). *Physics-informed machine learning*. Nature Reviews Physics, 3(6), 422-440.

3. Cai, S., et al. (2021). *Physics-informed neural networks (PINNs) for fluid mechanics: A review*. Acta Mechanica Sinica, 37(12), 1727-1738.

### 10.4 PINN Applications to Hydraulics

1. Raissi, M., et al. (2020). *Hidden fluid mechanics: Learning velocity and pressure fields from flow visualizations*. Science, 367(6481), 1026-1030.

2. Kissas, G., et al. (2020). *Machine learning in cardiovascular flows modeling: Predicting arterial blood pressure from non-invasive 4D flow MRI data using physics-informed neural networks*. Computer Methods in Applied Mechanics and Engineering, 358, 112623.

---

## Appendix A: Variable Notation

| Symbol | Description | Units | Typical Range |
|--------|-------------|-------|---------------|
| x | Streamwise coordinate (flow direction) | m | 0-6 |
| y | Cross-stream coordinate (width) | m | 0-1 |
| t | Time | s | 0-120 |
| h | Water depth | m | 0-0.3 |
| u | Streamwise velocity | m/s | 0-0.6 |
| v | Cross-stream velocity | m/s | -0.1 to 0.1 |
| z | Bed elevation | m | 0 (flat) |
| ζ | Water surface elevation | m | 0-0.3 |
| g | Gravitational acceleration | m/s² | 9.81 |
| f | Linear friction coefficient | 1/s | 0.01-0.02 |
| Fr | Froude number | - | 0.2-0.8 |

---

## Appendix B: Default Parameters

```python
# Domain
domain = {
    'x_min': 0.0, 'x_max': 6.0,  # 6m channel length
    'y_min': 0.0, 'y_max': 1.0,  # 1m channel width
    't_min': 0.0, 't_max': 120.0  # 2 minutes
}

# Sampling
N_collocation = 10000
N_bc = 2500
N_ic = 3000

# Network
in_dim = 3
hid_dim = 64
out_dim = 3
activation = 'Tanh'

# Training
n_epochs = 10000
learning_rate = 5e-4
scheduler_gamma = 0.997

# Loss weights (KEY DIFFERENCE!)
w_pde = 50.0   # Much higher than rainfall-runoff
w_ic = 150.0
w_bc = 150.0
w_phys = 15.0

# Physics
g = 9.81  # m/s²
friction_factor = 0.015  # 1/s
u_inflow = 0.5  # m/s (inflow velocity)

# Terrain
z = 0.0  # Flat bed (no slope)

# Grid resolution
grid_res_x = 90  # Matches length
grid_res_y = 30  # Matches width
```

---

## Appendix C: Expected Results

### Flow Development Timeline

**t = 0-10s**: 
- Water enters from inlet
- Advancing front at ~1 m from inlet
- Depth near inlet: ~0.05 m

**t = 10-30s**:
- Front reaches mid-channel (x ≈ 3m)
- Backwater curve developing
- Depth near inlet: ~0.10 m

**t = 30-60s**:
- Front reaches outlet
- Water accumulates (no outflow)
- Depth gradient establishing

**t = 60-120s**:
- Quasi-steady profile
- Depth slowly increasing everywhere
- Near-uniform velocity in center section

### Typical Final Values (at t=120s)

- **Inlet depth**: 0.15-0.20 m
- **Mid-channel depth**: 0.10-0.15 m
- **Outlet depth**: 0.05-0.10 m
- **Average velocity**: 0.3-0.5 m/s
- **Froude number**: 0.3-0.5 (subcritical)

---

## Document Information

**Version**: 1.0  
**Date**: February 2026  
**Companion to**: Rainfall-Runoff PINN Documentation  
**Purpose**: Complete reference for PINN channel flow model  

**Document Status**: Complete  
**Review Status**: Reviewed  
**Distribution**: Public  

---

*End of Documentation*
