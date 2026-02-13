"""
Household Microgrid Model — Single Node Simulation
====================================================
Models a single household node with:
  - PV generation (solar pattern with weather noise)
  - Household demand (realistic daily load profile)
  - Battery storage (charge on surplus, discharge on deficit)
  - Net power = Generation - Demand ± Battery

Produces a 6-day time-series plot similar to the reference figure.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from datetime import datetime, timedelta

# ─── Simulation Parameters ───────────────────────────────────────────
np.random.seed(42)

DAYS = 6
DT_MIN = 10  # time resolution: 10 minutes
STEPS_PER_DAY = 24 * 60 // DT_MIN  # 144 steps per day
N = DAYS * STEPS_PER_DAY

# Battery parameters
BATTERY_CAPACITY_KWH = 14.0      # total capacity
BATTERY_MAX_POWER_KW = 5.0      # max charge/discharge rate
BATTERY_EFF = 0.92              # round-trip efficiency (sqrt for each direction)
SOC_MIN = 0.1                   # minimum state of charge
SOC_MAX = 0.9                   # maximum state of charge

# ─── Time Axis ───────────────────────────────────────────────────────
start = datetime(2025, 6, 15, 0, 0)  # summer week
time = [start + timedelta(minutes=DT_MIN * i) for i in range(N)]
hours = np.array([(t.hour + t.minute / 60) for t in time])
day_index = np.array([t.day - start.day for t in time])


# ─── PV Generation Model ────────────────────────────────────────────
def generate_pv(hours, n_steps, days):
    """
    Realistic PV generation:
    - Bell-shaped solar curve peaking around 12:00-13:00
    - Cloud transients modeled as correlated noise
    - Day-to-day variation (some days cloudier)
    """
    pv = np.zeros(n_steps)
    for i in range(n_steps):
        h = hours[i]
        # Solar envelope: sunrise ~5:30, sunset ~21:00 (UK summer)
        if 5.5 < h < 21.0:
            # Gaussian-ish solar curve centered at 13:00
            solar = np.exp(-0.5 * ((h - 13.0) / 3.5) ** 2)
        else:
            solar = 0.0
        pv[i] = solar

    # Scale to realistic peak (~0.8-1.0 kW for a typical UK residential PV)
    pv *= 1.8

    # Day-to-day cloud factor (some days are cloudier)
    daily_cloud = np.array([0.95, 0.55, 0.85, 0.70, 0.90, 0.80])
    for d in range(days):
        mask = day_index == d
        pv[mask] *= daily_cloud[d % len(daily_cloud)]

    # Intra-day cloud transients (correlated noise)
    cloud_noise = np.random.normal(0, 0.08, n_steps)
    # Smooth the noise to make cloud events correlated over ~30-60 min
    kernel_size = 5  # 50 minutes smoothing
    kernel = np.ones(kernel_size) / kernel_size
    cloud_noise = np.convolve(cloud_noise, kernel, mode='same')
    pv = pv + cloud_noise * (pv > 0.05)  # only add noise during daylight

    pv = np.clip(pv, 0, None)
    return pv


# ─── Household Demand Model ─────────────────────────────────────────
def generate_demand(hours, n_steps, days):
    """
    Realistic UK household demand profile:
    - Base load ~0.2-0.3 kW (fridge, standby, etc.)
    - Morning peak ~7:00-9:00 (breakfast, heating water)
    - Evening peak ~17:00-21:00 (cooking, entertainment)
    - Low overnight
    - Random appliance spikes (kettle, oven, etc.)
    """
    demand = np.zeros(n_steps)

    for i in range(n_steps):
        h = hours[i]

        # Base load
        base = 0.25

        # Morning peak
        morning = 0.25 * np.exp(-0.5 * ((h - 7.5) / 1.0) ** 2)

        # Evening peak (larger)
        evening = 0.35 * np.exp(-0.5 * ((h - 18.5) / 1.5) ** 2)

        # Lunch bump
        lunch = 0.10 * np.exp(-0.5 * ((h - 12.5) / 0.8) ** 2)

        # Late night dip
        night_factor = 1.0
        if h < 5 or h > 23:
            night_factor = 0.7

        demand[i] = (base + morning + evening + lunch) * night_factor

    # Day-to-day variation
    daily_factor = np.array([1.0, 0.9, 1.05, 0.95, 1.1, 1.0])
    for d in range(days):
        mask = day_index == d
        demand[mask] *= daily_factor[d % len(daily_factor)]

    # Random appliance spikes (kettle = 2-3 kW for a few minutes, etc.)
    for d in range(days):
        n_spikes = np.random.randint(3, 7)
        for _ in range(n_spikes):
            spike_step = np.random.randint(
                d * STEPS_PER_DAY + 6 * 6,  # after 6 AM
                min((d + 1) * STEPS_PER_DAY - 3 * 6, n_steps - 3)  # before 9:30 PM
            )
            spike_mag = np.random.uniform(0.15, 0.4)
            spike_dur = np.random.randint(1, 4)  # 10-30 minutes
            end = min(spike_step + spike_dur, n_steps)
            demand[spike_step:end] += spike_mag

    # Smooth slightly to avoid unrealistic jagged edges
    kernel = np.ones(3) / 3
    demand = np.convolve(demand, kernel, mode='same')

    return demand


# ─── Battery Control Strategy ───────────────────────────────────────
def simulate_battery(generation, demand, n_steps, dt_hours):
    """
    Simple self-consumption maximization strategy:
    - Surplus PV → charge battery (up to max rate & SOC limit)
    - Deficit → discharge battery (down to min SOC)
    - Net power = what flows to/from the grid through PCC
    """
    soc = np.zeros(n_steps)  # state of charge in kWh
    battery_power = np.zeros(n_steps)  # positive = discharging, negative = charging
    net_power = np.zeros(n_steps)

    soc[0] = BATTERY_CAPACITY_KWH * 0.5  # start at 50% SOC

    for i in range(n_steps):
        surplus = generation[i] - demand[i]  # positive = excess PV

        if surplus > 0:
            # Try to charge battery
            max_charge = min(
                surplus,
                BATTERY_MAX_POWER_KW,
                (BATTERY_CAPACITY_KWH * SOC_MAX - soc[i]) / (dt_hours * np.sqrt(BATTERY_EFF))
            )
            max_charge = max(max_charge, 0)
            battery_power[i] = -max_charge  # negative = charging
            energy_stored = max_charge * dt_hours * np.sqrt(BATTERY_EFF)
            net_power[i] = surplus - max_charge  # remaining surplus → grid

        else:
            # Try to discharge battery to cover deficit
            deficit = -surplus
            max_discharge = min(
                deficit,
                BATTERY_MAX_POWER_KW,
                (soc[i] - BATTERY_CAPACITY_KWH * SOC_MIN) / (dt_hours / np.sqrt(BATTERY_EFF))
            )
            max_discharge = max(max_discharge, 0)
            battery_power[i] = max_discharge  # positive = discharging
            energy_released = max_discharge * dt_hours / np.sqrt(BATTERY_EFF)
            net_power[i] = -(deficit - max_discharge)  # remaining deficit ← grid (negative)

        # Update SOC
        if i < n_steps - 1:
            if battery_power[i] < 0:  # charging
                soc[i + 1] = soc[i] + (-battery_power[i]) * dt_hours * np.sqrt(BATTERY_EFF)
            else:  # discharging
                soc[i + 1] = soc[i] - battery_power[i] * dt_hours / np.sqrt(BATTERY_EFF)
            soc[i + 1] = np.clip(soc[i + 1],
                                  BATTERY_CAPACITY_KWH * SOC_MIN,
                                  BATTERY_CAPACITY_KWH * SOC_MAX)

    return battery_power, net_power, soc


# ─── Run Simulation ─────────────────────────────────────────────────
generation = generate_pv(hours, N, DAYS)
demand = generate_demand(hours, N, DAYS)
dt_hours = DT_MIN / 60.0
battery_power, net_power, soc = simulate_battery(generation, demand, N, dt_hours)


# ─── Plotting (matching reference style) ────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[3, 1],
                          sharex=True, gridspec_kw={'hspace': 0.08})

ax1 = axes[0]

# Fill areas for generation and demand
ax1.fill_between(time, generation, alpha=0.5, color='#4DA6FF', label='Generation (PV)', zorder=2)
ax1.fill_between(time, -demand, alpha=0.5, color='#FF9933', label='Demand', zorder=2)

# Battery power line
ax1.plot(time, battery_power, color='#2D8C2D', linewidth=1.2, label='Battery', zorder=3)

# Net power line
ax1.plot(time, net_power, color='black', linewidth=1.0, label='Net (to grid)', zorder=4)

ax1.axhline(y=0, color='gray', linewidth=0.5, linestyle='-', zorder=1)
ax1.set_ylabel('Power (kW)', fontsize=12)
ax1.set_ylim(-2.0, 2.0)
ax1.legend(loc='upper right', fontsize=10, ncol=4, framealpha=0.9)
ax1.set_title('Household Microgrid — 6-Day Power Balance', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)

# Day separators
for d in range(1, DAYS):
    day_start = start + timedelta(days=d)
    ax1.axvline(x=day_start, color='gray', linewidth=0.8, linestyle='-', alpha=0.5)

# ─── SOC subplot ────────────────────────────────────────────────────
ax2 = axes[1]
soc_pct = soc / BATTERY_CAPACITY_KWH * 100
ax2.fill_between(time, soc_pct, alpha=0.3, color='#2D8C2D')
ax2.plot(time, soc_pct, color='#2D8C2D', linewidth=1.2)
ax2.axhline(y=SOC_MIN * 100, color='red', linewidth=0.8, linestyle='--', alpha=0.6, label='SOC limits')
ax2.axhline(y=SOC_MAX * 100, color='red', linewidth=0.8, linestyle='--', alpha=0.6)
ax2.set_ylabel('Battery SOC (%)', fontsize=12)
ax2.set_ylim(0, 100)
ax2.set_xlabel('Time', fontsize=12)
ax2.legend(loc='upper right', fontsize=9)
ax2.grid(True, alpha=0.3)

# Day separators
for d in range(1, DAYS):
    day_start = start + timedelta(days=d)
    ax2.axvline(x=day_start, color='gray', linewidth=0.8, linestyle='-', alpha=0.5)

# Format x-axis
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
ax2.xaxis.set_major_locator(mdates.DayLocator())
plt.xticks(rotation=0)

plt.tight_layout()
_OUT_DIR = Path(__file__).parent / "output"
_OUT_DIR.mkdir(exist_ok=True)
plt.savefig(_OUT_DIR / 'household_microgrid.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\n✓ Plot saved to {_OUT_DIR / 'household_microgrid.png'}")

# ─── Summary Statistics ─────────────────────────────────────────────
total_gen = np.sum(generation) * dt_hours
total_demand = np.sum(demand) * dt_hours
total_exported = np.sum(net_power[net_power > 0]) * dt_hours
total_imported = -np.sum(net_power[net_power < 0]) * dt_hours
self_consumption = total_gen - total_exported

print(f"\n{'='*50}")
print(f"  Simulation Summary ({DAYS} days)")
print(f"{'='*50}")
print(f"  Total PV generation:   {total_gen:.2f} kWh")
print(f"  Total demand:          {total_demand:.2f} kWh")
print(f"  Self-consumption:      {self_consumption:.2f} kWh ({self_consumption/total_gen*100:.1f}%)")
print(f"  Exported to grid:      {total_exported:.2f} kWh")
print(f"  Imported from grid:    {total_imported:.2f} kWh")
print(f"  Self-sufficiency:      {(1 - total_imported/total_demand)*100:.1f}%")
print(f"{'='*50}")
