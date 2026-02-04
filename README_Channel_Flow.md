# Physics-Informed Neural Networks for Channel Flow

## Quick Start Guide

---

## 📦 What's Included

1. **PINNs_Channel_Flow_CLEAN.ipynb** - Jupyter notebook (~40 KB)
2. **PINNs_Channel_Flow_Documentation.md** - Complete documentation
3. **README_Channel_Flow.md** - This quick start guide

---

## 🚀 Quick Start

### Run the Simulation

```bash
# Install dependencies
pip install torch numpy matplotlib tqdm jupyter

# Open notebook
jupyter notebook PINNs_Channel_Flow_CLEAN.ipynb

# Run all cells (Shift + Enter through each, or Cell → Run All)
# Wait 15-20 minutes for training
```

---

## 🌊 What This Simulates

**Channel flow startup**: Water flowing into an initially dry rectangular channel

- **Inlet**: Water enters at 0.5 m/s from the left
- **Channel**: 6m long × 1m wide, flat bed
- **Simulation**: 2 minutes of flow development
- **Physics**: Shallow water equations with friction

### Key Outputs

- **3D Animation**: Water surface rising and propagating downstream
- **2D Animation**: Depth contours with velocity vectors
- **Training curve**: Loss convergence history

---

## 🎯 Key Features

### Differences from Rainfall-Runoff Model

| Feature | Rainfall-Runoff | Channel Flow |
|---------|-----------------|--------------|
| **Input** | Distributed rainfall | Boundary inflow |
| **Terrain** | 5% sloped | Flat horizontal |
| **Domain** | 1m × 1m square | 6m × 1m rectangle |
| **Application** | Hillslope runoff | Canal hydraulics |
| **PDE source** | Rainfall term | No source |

---

## 🔧 Adjustable Parameters

Located in the "Main Execution" section:

### Inflow Velocity
```python
u_inflow = 0.3   # Slower flow (was 0.5)
u_inflow = 0.8   # Faster flow (was 0.5)
```

### Friction
```python
friction_factor = 0.005  # Smooth channel (was 0.015)
friction_factor = 0.030  # Rough channel (was 0.015)
```

### Channel Length
```python
domain = {'x_max': 3.0}   # Shorter (was 6.0)
domain = {'x_max': 10.0}  # Longer (was 6.0)
```

### Loss Weights
```python
w_pde = 100.0   # Stronger PDE enforcement (was 50.0)
w_bc = 300.0    # Stronger boundary conditions (was 150.0)
```

---

## 📊 Expected Results

### Flow Development

**0-30 seconds**: Water enters and front advances downstream  
**30-60 seconds**: Flow reaches outlet, backwater develops  
**60-120 seconds**: Approaching steady state, depth increasing  

### Typical Values (at t=120s)

- **Depth**: 0.05-0.20 m (varies along channel)
- **Velocity**: 0.3-0.5 m/s (streamwise)
- **Froude number**: 0.3-0.5 (subcritical flow)

---

## 🐛 Troubleshooting

### Flow Doesn't Propagate

**Problem**: Water stays near inlet, doesn't move downstream

**Solutions**:
```python
w_pde = 100.0           # Increase PDE weight
friction_factor = 0.010  # Reduce friction
n_epochs = 15000        # Train longer
```

### Inlet Velocity Not Satisfied

**Problem**: Velocity at x=0 doesn't match u_inflow

**Solutions**:
```python
w_bc = 300.0   # Increase BC weight
N_bc = 4000    # More boundary samples
```

### Excessive Water Build-Up

**Problem**: Water depth keeps increasing unrealistically

**Cause**: Outlet is a wall - water can't leave (this is expected!)

**Solutions**:
```python
# Option 1: Look at earlier times only
t_max = 60.0

# Option 2: Implement open outlet (see full docs)
```

---

## 📚 Learning Path

### Beginner (1 hour)
1. Run the notebook
2. Read this README
3. Try changing inflow velocity

### Intermediate (3 hours)
1. Read Documentation Sections 1-3
2. Modify friction and observe effects
3. Adjust loss weights
4. Compare with rainfall-runoff model

### Advanced (1 day)
1. Read full documentation
2. Add bed slope
3. Implement time-varying inflow
4. Create curved channel

---

## 🎓 Key Concepts

### What is a PINN?

A neural network that:
1. Learns the solution to PDEs
2. Trained to satisfy physics (via loss function)
3. No need for grid discretization
4. Continuous representation

### Why Higher w_pde?

Channel flow needs **stronger PDE enforcement** (w_pde=50.0 vs 1.0 for rainfall):

- Boundary inflow is strong forcing
- Must propagate correctly throughout domain
- Flat bed removes terrain gradient "help"
- Longer domain requires consistent physics

### Boundary Conditions

**Inlet (x=0)**: u = 0.5 m/s (Dirichlet velocity)  
**Outlet (x=6)**: u = 0 (wall - simplified)  
**Sides (y=0,1)**: v = 0 (no cross-flow)  

---

## 🔬 Experiments to Try

### Easy (30 min each)

1. **Double the inflow**: `u_inflow = 1.0`
   - Observe: Faster filling, deeper water

2. **Smooth channel**: `friction_factor = 0.005`
   - Observe: More uniform flow, faster propagation

3. **Shorter channel**: `domain = {'x_max': 3.0}`
   - Observe: Quicker steady state

### Medium (1-2 hours each)

4. **Time-varying inflow**: Implement sinusoidal velocity
   - See periodic depth variations

5. **Add mild slope**: `z = -0.001 * X`
   - Observe: Gravity-driven acceleration

6. **Wider channel**: `domain = {'y_max': 2.0}`
   - Observe: More 2D flow patterns

### Advanced (half day each)

7. **Open outlet boundary**: Implement zero-gradient outflow
   - See water actually leave the channel

8. **Curved channel**: Create sinusoidal centerline
   - Observe: Secondary circulation

9. **Obstacle**: Add cylinder in mid-channel
   - See wake formation

---

## 📖 Documentation Guide

### Quick Reference (this file)
- Basic usage
- Parameter adjustment
- Common issues

### Full Documentation
- Complete physics explanation
- Mathematical derivation
- All parameters explained
- Advanced modifications
- Validation methods

---

## 🤝 Comparison with Rainfall-Runoff

Both models use the same:
- Neural network architecture
- Training approach
- Automatic differentiation

Key difference is in the **physics**:
- Rainfall: Source term in continuity equation
- Channel: Boundary condition at inlet

This shows PINN flexibility - same network, different physics!

---

## ⚙️ Technical Details

### Requirements
- Python 3.8+
- PyTorch 1.10+
- ~2 GB GPU memory (or CPU)
- 15-20 minutes training time

### File Sizes
- Notebook: ~40 KB (outputs cleared)
- Documentation: ~170 KB
- Generated videos: ~5 MB each

---

## 📧 Tips for Best Results

1. **Start with defaults** - make sure baseline works
2. **Change one parameter at a time** - easier to understand effects
3. **Monitor BC loss** - should be < 1e-3 for good inlet condition
4. **Check Froude number** - should stay < 1 for stable flow
5. **Use GPU if available** - 10× faster than CPU

---

## 🎯 Success Criteria

You understand this model when you can:

✅ Explain why w_pde is higher than rainfall model  
✅ Predict how changing friction affects depth  
✅ Identify which BC is not satisfied from training logs  
✅ Modify the channel geometry  
✅ Add a new boundary condition type  

---

## 📝 Quick Reference Card

### Default Values
```python
domain: 6m × 1m × 120s
u_inflow: 0.5 m/s
friction: 0.015
w_pde: 50.0 (KEY!)
w_bc: 150.0
grid: 90 × 30
```

### File Structure
```
Cell 1-2: Description
Cell 3-5: Imports & Colab setup
Cell 6-7: Domain & terrain
Cell 8-9: Equations & residuals
Cell 10: Training function
Cell 11: Visualization
Cell 12: Main execution
```

### Training Time
- GPU: ~15 minutes
- CPU: ~2 hours

---

**Version**: 1.0  
**Date**: February 2026  
**Companion**: Rainfall-Runoff PINN Model

---

*Happy channel flow modeling!* 🌊
