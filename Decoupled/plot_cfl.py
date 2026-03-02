#!/usr/bin/env python3
"""Plot CFL and magnetization diagnostics from Rosensweig run."""
import csv
import matplotlib.pyplot as plt
import sys
import os

csv_path = sys.argv[1] if len(sys.argv) > 1 else \
    "/Users/mahdi/Projects/git/PhD-Project/Decoupled/Results/030126_222431_rosensweig_r4/diagnostics.csv"

# Read CSV (skip comment header lines starting with #)
times, cfls, M_maxs, U_maxs, H_maxs = [], [], [], [], []
with open(csv_path) as f:
    reader = csv.reader(f)
    header = None
    for row in reader:
        if row[0].startswith('#'):
            continue
        if header is None:
            header = row
            # Find column indices
            idx_time = header.index('time')
            idx_cfl = header.index('CFL')
            idx_U_max = header.index('U_max')
            idx_H_max = header.index('H_max')
            idx_M_max = header.index('M_mag_max')
            continue
        try:
            t = float(row[idx_time])
            cfl = float(row[idx_cfl])
            U = float(row[idx_U_max])
            H = float(row[idx_H_max])
            M = float(row[idx_M_max])
            times.append(t)
            cfls.append(cfl)
            U_maxs.append(U)
            H_maxs.append(H)
            M_maxs.append(M)
        except (ValueError, IndexError):
            continue

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('Rosensweig Instability Diagnostics (Linear chi/nu, no spin-vorticity)', fontsize=13)

# Plot 1: CFL vs time
ax = axes[0, 0]
ax.plot(times, cfls, 'b-', linewidth=0.8)
ax.set_xlabel('Time')
ax.set_ylabel('CFL')
ax.set_title('CFL vs Time')
ax.grid(True, alpha=0.3)

# Plot 2: CFL vs M_max
ax = axes[0, 1]
ax.plot(M_maxs, cfls, 'r-', linewidth=0.8)
ax.set_xlabel('Max Magnetization |M|')
ax.set_ylabel('CFL')
ax.set_title('CFL vs Magnetization')
ax.grid(True, alpha=0.3)

# Plot 3: U_max vs time
ax = axes[1, 0]
ax.plot(times, U_maxs, 'g-', linewidth=0.8)
ax.set_xlabel('Time')
ax.set_ylabel('U_max')
ax.set_title('Max Velocity vs Time')
ax.grid(True, alpha=0.3)

# Plot 4: H_max and M_max vs time
ax = axes[1, 1]
ax.plot(times, H_maxs, 'b-', linewidth=0.8, label='|H|_max')
ax.plot(times, M_maxs, 'r--', linewidth=0.8, label='|M|_max')
ax.set_xlabel('Time')
ax.set_ylabel('Field magnitude')
ax.set_title('Magnetic Field & Magnetization vs Time')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
out_path = os.path.join(os.path.dirname(csv_path), 'diagnostics_plots.png')
plt.savefig(out_path, dpi=150)
print(f"Saved: {out_path}")
plt.close()
