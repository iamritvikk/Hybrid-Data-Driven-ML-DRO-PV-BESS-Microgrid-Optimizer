import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings; warnings.filterwarnings('ignore')

np.random.seed(42)

# ── Style ──────────────────────────────────────────────────────
BG   = "#0b0f1a"; CARD = "#111827"; CARD2 = "#1a2236"; BORDER = "#2d3748"
TEXT = "#e8edf5"; T2   = "#8a96aa"; T3    = "#5a6478"
AMBER="#f6ad55";  TEAL ="#4fd1c5";  ACCT  = "#6C63FF"
GREEN="#68d391";  RED  ="#fc8181"
plt.rcParams.update({"figure.facecolor":BG,"axes.facecolor":CARD,"axes.edgecolor":BORDER,
    "axes.labelcolor":T2,"xtick.color":T3,"ytick.color":T3,"text.color":TEXT,
    "grid.color":CARD2,"grid.linewidth":0.8,"font.size":9})

# ── 1. DATA ─────────────────────────────────────────────────────
N = 2500
solar_irr   = np.random.uniform(2.0, 8.0,  N)
load_demand = np.random.uniform(10,  200,   N)
load_var    = np.random.uniform(5,   40,    N)
solar_var   = np.random.uniform(5,   50,    N)

energy = (load_demand*(1+load_var/100*0.45)*(1+solar_var/100*0.12)
          + np.random.normal(0, load_demand*0.03, N))
energy = np.maximum(energy, 5)

battery = (load_demand*0.35
           + load_demand*(load_var/100)*0.55
           + load_demand*(solar_var/100)*0.28
           + np.random.normal(0, load_demand*0.02, N))
battery = np.maximum(battery, 2)

eff_irr = solar_irr*(1 - solar_var/100*0.42)
pv      = (energy/(eff_irr*0.85+0.01)*(1+solar_var/100*0.18)
           + np.random.normal(0, 0.25, N))
pv = np.maximum(pv, 1)

X = np.column_stack([solar_irr, load_demand, load_var, solar_var])
sp = int(0.8*N)
X_tr, X_te = X[:sp], X[sp:]
sc = StandardScaler()
Xs_tr = sc.fit_transform(X_tr); Xs_te = sc.transform(X_te)

tdata = {
    "Energy Usage": (energy,  "kWh", AMBER),
    "Battery Size": (battery, "kWh", TEAL),
    "PV Size":      (pv,      "kWp", ACCT),
}

# ── 2. TRAIN ────────────────────────────────────────────────────
models={}; res={}
print("="*55)
print("  PV-BESS Microgrid — Random Forest Model")
print("="*55)
print(f"\n  {'Target':<16} {'R²':>8} {'RMSE':>8} {'MAE':>8} {'MAPE':>8} {'CV R²':>8}")
print(f"  {'-'*60}")
for name,(y_all,unit,col) in tdata.items():
    y_tr, y_te = y_all[:sp], y_all[sp:]
    rf = RandomForestRegressor(n_estimators=150, max_depth=10,
                                min_samples_leaf=3, max_features="sqrt",
                                random_state=42, n_jobs=1)
    rf.fit(Xs_tr, y_tr)
    pred = rf.predict(Xs_te)
    r2   = r2_score(y_te, pred)
    rmse = np.sqrt(mean_squared_error(y_te, pred))
    mae  = mean_absolute_error(y_te, pred)
    mape = np.mean(np.abs((y_te-pred)/(y_te+1e-8)))*100
    cv   = cross_val_score(rf, Xs_tr, y_tr, cv=3, scoring='r2').mean()
    models[name]=rf
    res[name]=dict(pred=pred,y_te=y_te,r2=r2,rmse=rmse,mae=mae,mape=mape,cv=cv,unit=unit,col=col)
    print(f"  {name:<16} {r2:>8.4f} {rmse:>8.3f} {mae:>8.3f} {mape:>7.2f}% {cv:>8.4f}")

# ── 3. PREDICT FUNCTION ──────────────────────────────────────────
def predict(si, ld, lv, sv):
    x = sc.transform([[si, ld, lv, sv]])
    return {n: round(models[n].predict(x)[0],2) for n in models}

# ── 4. DEMO SCENARIOS ────────────────────────────────────────────
scenarios = [
    ("Rural India",         5.5, 40,  25, 30),
    ("Urban rooftop",       4.0, 120, 15, 20),
    ("Desert microgrid",    7.5, 80,  20, 15),
    ("Coastal",             4.8, 70,  30, 40),
    ("Hospital",            5.0, 150, 10, 25),
]
print(f"\n  {'Scenario':<22} {'Energy(kWh)':>13} {'Battery(kWh)':>13} {'PV(kWp)':>10}")
print(f"  {'-'*60}")
for name,si,ld,lv,sv in scenarios:
    p = predict(si,ld,lv,sv)
    print(f"  {name:<22} {p['Energy Usage']:>13.1f} {p['Battery Size']:>13.1f} {p['PV Size']:>10.1f}")

# ── 5. VISUALISATION ─────────────────────────────────────────────
T_NAMES  = list(tdata.keys())
T_COLORS = [tdata[n][2] for n in T_NAMES]
T_UNITS  = [tdata[n][1] for n in T_NAMES]

fig = plt.figure(figsize=(20, 22))
fig.patch.set_facecolor(BG)
gs  = gridspec.GridSpec(4, 3, figure=fig, hspace=0.52, wspace=0.38)

# Row 0 — Predicted vs Actual
for col,(name,col_c,unit) in enumerate(zip(T_NAMES,T_COLORS,T_UNITS)):
    ax = fig.add_subplot(gs[0,col])
    m  = res[name]
    mn = min(m["y_te"].min(), m["pred"].min())
    mx = max(m["y_te"].max(), m["pred"].max())
    ax.scatter(m["y_te"], m["pred"], alpha=0.28, s=10, color=col_c, zorder=3)
    bx = np.linspace(mn,mx,200)
    ax.fill_between(bx, bx*0.9, bx*1.1, alpha=0.08, color=col_c)
    ax.plot([mn,mx],[mn,mx], color=RED, lw=1.8, linestyle="--", label="Perfect fit")
    ax.set_title(f"Predicted vs Actual\n{name} ({unit})", color=TEXT, fontsize=10, fontweight="bold", pad=8)
    ax.set_xlabel(f"Actual ({unit})"); ax.set_ylabel(f"Predicted ({unit})")
    ax.legend(framealpha=0.15, fontsize=7.5); ax.grid(True, alpha=0.3)
    ax.text(0.04, 0.88, f"R²   = {m['r2']:.4f}\nRMSE = {m['rmse']:.3f}\nMAE  = {m['mae']:.3f}",
            transform=ax.transAxes, fontsize=8.5, color=TEXT,
            bbox=dict(boxstyle="round,pad=0.35", facecolor=CARD2, alpha=0.92))

# Row 1 — Residuals
for col,(name,col_c,unit) in enumerate(zip(T_NAMES,T_COLORS,T_UNITS)):
    ax  = fig.add_subplot(gs[1,col])
    m   = res[name]
    err = m["y_te"] - m["pred"]
    ax.hist(err, bins=40, color=col_c, alpha=0.78, density=True, zorder=3, edgecolor=BG, linewidth=0.4)
    ax.axvline(0,          color=RED,   lw=2.0, linestyle="--", label="Zero error")
    ax.axvline( m["rmse"], color=GREEN, lw=1.2, linestyle=":",  label=f"±RMSE")
    ax.axvline(-m["rmse"], color=GREEN, lw=1.2, linestyle=":")
    ax.set_title(f"Residual Distribution\n{name} ({unit})", color=TEXT, fontsize=10, fontweight="bold", pad=8)
    ax.set_xlabel("Prediction Error"); ax.set_ylabel("Density")
    ax.legend(framealpha=0.15, fontsize=7.5); ax.grid(True, alpha=0.3)
    ax.text(0.62, 0.86, f"μ={err.mean():.2f}\nσ={err.std():.2f}\nMAPE={m['mape']:.1f}%",
            transform=ax.transAxes, fontsize=8.5, color=TEXT,
            bbox=dict(boxstyle="round,pad=0.35", facecolor=CARD2, alpha=0.92))

# Row 2 — Feature Importance
feat_labels = ["Solar\nIrradiance", "Load\nDemand", "Load\nVariability", "Solar\nVariability"]
for col,(name,col_c) in enumerate(zip(T_NAMES,T_COLORS)):
    ax = fig.add_subplot(gs[2,col])
    fi = models[name].feature_importances_
    order = np.argsort(fi)
    bars  = ax.barh([feat_labels[i] for i in order], fi[order],
                    color=[col_c if fi[i]==fi.max() else ACCT for i in order],
                    alpha=0.85, zorder=3, edgecolor=BG, linewidth=0.4)
    for bar,val in zip(bars, fi[order]):
        ax.text(val+0.005, bar.get_y()+bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=9, color=TEXT)
    ax.set_xlim(0, fi.max()*1.28)
    ax.set_title(f"Feature Importance\n{name}", color=TEXT, fontsize=10, fontweight="bold", pad=8)
    ax.set_xlabel("Importance Score"); ax.grid(True, axis="x", alpha=0.3)

# Row 3 — Scenarios + Summary cards
ax_sc = fig.add_subplot(gs[3,:2])
sc_names = [s[0] for s in scenarios]
vals = {n:[] for n in T_NAMES}
for _,si,ld,lv,sv in scenarios:
    p = predict(si,ld,lv,sv)
    for n in T_NAMES: vals[n].append(p[n])

x  = np.arange(len(sc_names)); bw = 0.25
for i,(name,col_c) in enumerate(zip(T_NAMES,T_COLORS)):
    bars = ax_sc.bar(x+(i-1)*bw, vals[name], bw, label=f"{name} ({tdata[name][1]})",
                     color=col_c, alpha=0.90, zorder=3, edgecolor=BG, linewidth=0.5)
    for bar in bars:
        h = bar.get_height()
        ax_sc.text(bar.get_x()+bar.get_width()/2, h+1.5,
                   f"{h:.0f}", ha="center", va="bottom", fontsize=7, color=T2)
ax_sc.set_xticks(x); ax_sc.set_xticklabels(sc_names, fontsize=9)
ax_sc.set_title("Random Forest Predictions — Real-World Scenarios",
                color=TEXT, fontsize=11, fontweight="bold", pad=10)
ax_sc.set_ylabel("Value"); ax_sc.legend(framealpha=0.15, fontsize=9, loc="upper left")
ax_sc.grid(True, axis="y", alpha=0.3)

# Summary metric cards
ax_m = fig.add_subplot(gs[3,2])
ax_m.set_facecolor("#0d1421"); ax_m.axis("off")
ax_m.set_xlim(0,1); ax_m.set_ylim(0,1)
ax_m.text(0.5, 0.96, "MODEL METRICS", ha="center", va="center",
          fontsize=11, fontweight="bold", color=TEXT)
cards = []
for name,col_c,unit in zip(T_NAMES,T_COLORS,T_UNITS):
    m = res[name]; sh = name.split(" ")[0]
    cards += [(f"{sh} R²",    f"{m['r2']:.4f}",         col_c),
              (f"{sh} RMSE",  f"{m['rmse']:.2f} {unit}", col_c),
              (f"{sh} MAE",   f"{m['mae']:.2f} {unit}",  col_c),
              (f"{sh} CV R²", f"{m['cv']:.4f}",          col_c)]
cpr = 3
for i,(label,value,color) in enumerate(cards):
    c  = i % cpr; r = i // cpr
    xp = 0.04 + c*0.315; yp = 0.82 - r*0.22
    rect = FancyBboxPatch((xp,yp-0.16),0.29,0.18,
                           boxstyle="round,pad=0.01",
                           facecolor=CARD, edgecolor=color, linewidth=1.2, alpha=0.95)
    ax_m.add_patch(rect)
    ax_m.text(xp+0.145, yp+0.0,  label, ha="center", va="center", fontsize=7, color=T3, fontweight="bold")
    ax_m.text(xp+0.145, yp-0.09, value, ha="center", va="center", fontsize=10, color=color, fontweight="bold")

plt.suptitle("PV-BESS Microgrid Sizing — Random Forest Model\nSharma et al. · 4 Inputs → 3 Outputs",
             y=0.998, fontsize=14, fontweight="bold", color=TEXT)

out = "/mnt/user-data/outputs/pvbess_rf_model.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"\n  Chart saved → {out}")
print("\n  HOW TO USE:")
print('  p = predict(solar_irr=5.5, load_demand=60, load_var=20, solar_var=25)')
print("  p['Energy Usage']  → kWh")
print("  p['Battery Size']  → kWh")
print("  p['PV Size']       → kWp")
print("\n  Done!")
