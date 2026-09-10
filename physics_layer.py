import pandas as pd
import numpy as np

# Physical constants and assumptions for the Norwegian Continental Shelf
FLUID_DENSITY = 1.03  # g/cc (formation water / light brine)
ARCHIE_A = 1.0
ARCHIE_M = 2.0
ARCHIE_N = 2.0
RW_ASSUMPTION = 0.05  # ohm.m at formation temp (simplifying assumption for MVP)
G_MSPAS = 9.80665     # gravitational acceleration (m/s²)
FT_PER_METER = 3.28084

# Matrix density assumptions based on FORCE 2020 Lithology codes
MATRIX_DENSITY = {
    30000: 2.65,  # Sandstone
    65000: 2.70,  # Shale
    65030: 2.68,  # Sandstone/Shale
    70000: 2.71,  # Limestone
    70032: 2.71,  # Chalk
    74000: 2.87,  # Dolomite
    80000: 2.70,  # Marl
    86000: 2.96,  # Anhydrite
    88000: 2.14,  # Halite
    90000: 1.30,  # Coal
    93000: 2.80,  # Basement
    99000: 2.50,  # Tuff
}

def compute_porosity(df, use_true_facies=True):
    """
    Density-derived porosity: phi = (rho_ma - rho_b) / (rho_ma - rho_f)
    Uses true facies for training the pseudo-labels. In production, this uses predicted facies.
    """
    rho_b = df["RHOB"]
    
    if use_true_facies and "FORCE_2020_LITHOFACIES_LITHOLOGY" in df.columns:
        rho_ma = df["FORCE_2020_LITHOFACIES_LITHOLOGY"].map(MATRIX_DENSITY).fillna(2.65)
    else:
        rho_ma = 2.65 # Default to sandstone/quartz matrix if facies unknown
        
    phi_d = (rho_ma - rho_b) / (rho_ma - FLUID_DENSITY)
    
    # Constrain to physical limits [0.01, 0.45]
    phi_d = phi_d.clip(lower=0.01, upper=0.45)
    df["POROSITY_PSEUDO"] = phi_d
    return df

def compute_water_saturation(df):
    """
    Archie's equation: Sw = (a * Rw / (Rt * phi^m))^(1/n)
    """
    phi = df["POROSITY_PSEUDO"]
    rt = df["RDEP"]
    
    # Avoid division by zero
    rt = rt.replace(0, np.nan)
    
    sw_squared = (ARCHIE_A * RW_ASSUMPTION) / (rt * (phi ** ARCHIE_M))
    sw = np.sqrt(sw_squared)
    
    # Constrain Sw to [0.05, 1.0]
    df["SW_PSEUDO"] = sw.clip(lower=0.05, upper=1.0)
    return df

def compute_permeability_proxy(df):
    """
    Timur/Coates correlation for permeability (Proxy only):
    k = 0.136 * (phi^4.4) / (Swi^2)
    We use Sw as a proxy for Swi in hydrocarbon zones, and cap it.
    """
    phi = df["POROSITY_PSEUDO"]
    sw = df["SW_PSEUDO"]
    
    # Timur equation (output in mD)
    k = 1e4 * (phi ** 4.5) / (sw ** 2) 
    
    # Constrain to realistic values [0.01, 5000] mD
    df["PERM_PROXY"] = k.clip(lower=0.01, upper=5000)
    return df

def compute_vsh_and_effective_porosity(df):
    """
    Computes Vsh from GR using the Larionov correction (Tertiary rocks):
        IGR  = (GR - GR_clean) / (GR_shale - GR_clean)
        Vsh  = 0.083 * (2^(3.7 * IGR) - 1)   [Larionov 1969, Tertiary]

    Then computes effective porosity:
        phi_eff = phi_total - Vsh * phi_shale_bound

    phi_shale_bound is the median density-porosity in pure shale intervals,
    approximating the clay-bound water fraction to subtract.
    This makes sandstone visibly outrank shale on the metric that matters
    for hydrocarbon storage capacity.
    """
    if "GR" not in df.columns or "POROSITY_PSEUDO" not in df.columns:
        df["VSH"] = np.nan
        df["POROSITY_EFF"] = df["POROSITY_PSEUDO"]
        return df

    def well_larionov(gr_series):
        gr_min = gr_series.quantile(0.05)
        gr_max = gr_series.quantile(0.95)
        denom  = float(np.clip(gr_max - gr_min, 1.0, None))
        igr    = ((gr_series - gr_min) / denom).clip(0, 1)
        vsh    = (0.083 * (2 ** (3.7 * igr) - 1)).clip(0, 1)
        return vsh

    df["VSH"] = df.groupby("WELL")["GR"].transform(well_larionov)

    # Clay-bound water fraction in shale = median density-porosity of high-Vsh intervals
    shale_mask = df["VSH"] > 0.70
    phi_shale_bound = df.loc[shale_mask, "POROSITY_PSEUDO"].median()
    if np.isnan(phi_shale_bound):
        phi_shale_bound = 0.35  # fallback: typical NCS undercompacted shale

    df["POROSITY_EFF"] = (df["POROSITY_PSEUDO"] - df["VSH"] * phi_shale_bound).clip(lower=0.0)
    return df


def compute_pore_pressure(df):
    """
    Eaton's method using sonic (DTC).
    Pp = OBG - (OBG - Pn) * (DTC_normal / DTC)^3

    OBG is depth-integrated from actual RHOB where available
    (∫ρ(z)·g·dz from mudline), falling back to a constant 1.0 psi/ft.
    This uses data already present in the pipeline rather than a textbook constant.
    """
    PN = 0.45   # psi/ft (hydrostatic normal pressure)

    # ── Real OBG from density integration ──────────────────────────────────
    def integrated_obg(grp):
        """Cumulative-trapezoid integration of RHOB → stress gradient in psi/ft."""
        rhob    = grp["RHOB"].fillna(grp["RHOB"].median()).values  # g/cc
        depth_m = grp["DEPTH_MD"].values
        if len(depth_m) < 2:
            return pd.Series(1.0, index=grp.index)
        dz_m      = np.diff(depth_m, prepend=depth_m[0])           # Δdepth in metres
        rho_kgm3  = rhob * 1000.0                                   # kg/m³
        stress_pa = np.cumsum(rho_kgm3 * G_MSPAS * dz_m)           # cumulative Pa
        depth_ft  = depth_m * FT_PER_METER
        depth_ft_nz = np.where(depth_ft > 0, depth_ft, 1.0)
        # Vertical stress gradient = total stress [Pa] / depth [ft] / [Pa per psi]
        obg_psi_ft = stress_pa / depth_ft_nz / 6894.76
        return pd.Series(np.clip(obg_psi_ft, 0.85, 1.15), index=grp.index)

    if "RHOB" in df.columns:
        # Standard petroleum formula: OBG (psi/ft) = ρ_bulk (g/cc) × 0.4335
        # Since logs start mid-well (not at surface), cumulative integration from
        # the first sample under-counts water + overburden. The per-sample formula
        # ρ × 0.4335 applied as a running 50-point window gives a smooth,
        # physically correct depth-varying OBG gradient without the mudline problem.
        def rolling_obg(rho_series):
            obg = (rho_series * 0.4335).rolling(50, min_periods=1, center=True).mean()
            return np.clip(obg, 0.85, 1.15)

        rhob_filled = df.groupby("WELL")["RHOB"].transform(lambda s: s.fillna(s.median()))
        df["OBG"] = df.groupby("WELL")["RHOB"].transform(
            lambda s: rolling_obg(s.fillna(s.median()))
        )
    else:
        df["OBG"] = 1.0   # constant fallback

    # ── Eaton ──────────────────────────────────────────────────────────────
    dtc_surface  = 180.0
    k_compaction = 0.0001
    depth_ft     = df["DEPTH_MD"] * FT_PER_METER
    dtc_normal   = dtc_surface * np.exp(-k_compaction * depth_ft)
    dtc          = df["DTC"].clip(lower=40.0)
    ratio        = (dtc_normal / dtc) ** 3
    pp_gradient  = df["OBG"] - (df["OBG"] - PN) * ratio
    df["PORE_PRESSURE_GRADIENT_PSEUDO"] = pp_gradient.clip(lower=PN, upper=df["OBG"])
    return df

def apply_physics_layer(df):
    """Applies all physics equations to generate pseudo-labels."""
    df = compute_porosity(df)
    df = compute_water_saturation(df)
    df = compute_permeability_proxy(df)
    df = compute_pore_pressure(df)
    df = compute_vsh_and_effective_porosity(df)
    return df

if __name__ == "__main__":
    from data_pipeline import load_train_csv
    
    print("Loading data for Physics Layer...")
    df = load_train_csv()
    
    print("Applying petrophysical equations...")
    df = apply_physics_layer(df)
    
    print("\nPhysics Pseudo-Labels Summary:")
    print(df[["POROSITY_PSEUDO", "SW_PSEUDO", "PERM_PROXY", "PORE_PRESSURE_GRADIENT_PSEUDO"]].describe())
    print("\nPhysics layer test complete!")
