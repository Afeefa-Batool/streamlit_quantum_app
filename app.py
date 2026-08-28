"""
Quantum-Enhanced Aircraft Route Optimization System (Streamlit App)
===================================================================
A real-time interactive web application for aircraft flight route selection
comparing Classical Brute-Force Optimization with QAOA (Quantum Approximate Optimization Algorithm).
"""

import time
import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import minimize
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

# ==========================================
# PAGE CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="Quantum Aircraft Route Optimization",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #0b0f19;
        color: #e2e8f0;
    }
    .stApp {
        background: linear-gradient(135deg, #0b0f19 0%, #111827 50%, #0f172a 100%);
    }
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 18px;
        backdrop-filter: blur(10px);
        text-align: center;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    .metric-value {
        font-size: 26px;
        font-weight: 700;
        color: #38bdf8;
        margin-top: 5px;
    }
    .metric-label {
        font-size: 13px;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .badge-success {
        background-color: #059669;
        color: white;
        padding: 4px 10px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 12px;
    }
    .badge-info {
        background-color: #2563eb;
        color: white;
        padding: 4px 10px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 12px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1e293b;
        border-radius: 8px;
        color: #94a3b8;
        padding: 10px 16px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3b82f6 !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# Coordinates for Origins & Destinations
AIRPORT_COORDS = {
    "DEL": {"lat": 28.5562, "lon": 77.1000, "name": "New Delhi (DEL / VIDP)"},
    "JFK": {"lat": 40.6413, "lon": -73.7781, "name": "New York (JFK / KJFK)"}
}

# ==========================================
# DATA LOADING & PREPROCESSING
# ==========================================
@st.cache_data
def load_dataset():
    """Loads the 16-route 21-column aircraft routing dataset."""
    possible_paths = [
        os.path.join(os.path.dirname(__file__), "dataset.csv"),
        os.path.join(os.path.dirname(__file__), "routes_dataset.csv"),
        "dataset.csv",
        "routes_dataset.csv"
    ]
    for p in possible_paths:
        if os.path.exists(p):
            return pd.read_csv(p)
    st.error(f"Dataset file not found. Checked locations: {possible_paths}")
    st.stop()

def calculate_normalized_costs(df, weights):
    """
    Min-Max normalizes numerical metrics and calculates total weighted cost.
    """
    df = df.copy()
    w_fuel, w_time, w_weather, w_airspace = weights
    
    def min_max_scale(series):
        denom = series.max() - series.min()
        return (series - series.min()) / denom if denom > 0 else series * 0
    
    df["Fuel_Norm"] = min_max_scale(df["Fuel_Consumption_kg"])
    df["Time_Norm"] = min_max_scale(df["Flight_Time_min"])
    df["Weather_Norm"] = min_max_scale(df["Weather_Risk_0_10"])
    df["Airspace_Norm"] = min_max_scale(df["Airspace_Risk_0_10"])
    
    df["Total_Cost"] = (
        w_fuel * df["Fuel_Norm"] +
        w_time * df["Time_Norm"] +
        w_weather * df["Weather_Norm"] +
        w_airspace * df["Airspace_Norm"]
    )
    return df

# ==========================================
# OPTIMIZATION ALGORITHMS
# ==========================================
def run_classical_optimization(df):
    """Classical Brute-Force solver (O(N))."""
    start_time = time.perf_counter()
    min_idx = df["Total_Cost"].idxmin()
    best_route = df.loc[min_idx, "Route_ID"]
    best_cost = float(df.loc[min_idx, "Total_Cost"])
    elapsed_time = time.perf_counter() - start_time
    
    return {
        "method": "Classical Brute-Force",
        "route_id": best_route,
        "route_idx": min_idx,
        "cost": best_cost,
        "execution_time": elapsed_time,
        "row_data": df.loc[min_idx]
    }

def build_qaoa_circuit(n_qubits, cost_vector, gamma, beta, penalty=5.0):
    """Builds a p=1 QAOA circuit with log-encoding."""
    qc = QuantumCircuit(n_qubits)
    for q in range(n_qubits):
        qc.h(q)
        
    num_states = 2**n_qubits
    full_costs = np.full(num_states, penalty)
    for i, c in enumerate(cost_vector):
        full_costs[i] = c
        
    phases = np.exp(-1j * gamma * full_costs)
    sv = Statevector.from_instruction(qc)
    sv_phased = Statevector(sv.data * phases)
    
    qc_mixer = QuantumCircuit(n_qubits)
    for q in range(n_qubits):
        qc_mixer.rx(2 * beta, q)
        
    final_sv = sv_phased.evolve(qc_mixer)
    return final_sv, full_costs

def run_qaoa_optimization(df, penalty=5.0, maxiter=60, n_starts=3):
    """QAOA Variational Quantum Solver (4 qubits for up to 16 routes)."""
    start_time = time.perf_counter()
    
    n_routes = len(df)
    n_qubits = int(np.ceil(np.log2(max(n_routes, 2))))
    cost_vector = df["Total_Cost"].values
    
    num_states = 2**n_qubits
    full_costs = np.full(num_states, penalty)
    full_costs[:n_routes] = cost_vector
    
    best_exp_val = float('inf')
    best_params = None
    
    def qaoa_objective(params):
        gamma, beta = params
        final_sv, _ = build_qaoa_circuit(n_qubits, cost_vector, gamma, beta, penalty)
        probs = final_sv.probabilities()
        exp_val = np.sum(probs * full_costs)
        return exp_val

    gamma_starts = np.linspace(0.1, np.pi, n_starts)
    beta_starts = np.linspace(0.1, np.pi / 2.0, n_starts)
    
    for g_init in gamma_starts:
        for b_init in beta_starts:
            res = minimize(qaoa_objective, [g_init, b_init], method="COBYLA", options={"maxiter": maxiter})
            if res.fun < best_exp_val:
                best_exp_val = res.fun
                best_params = res.x
    
    opt_gamma, opt_beta = best_params if best_params is not None else (0.5, 0.5)
    opt_sv, _ = build_qaoa_circuit(n_qubits, cost_vector, opt_gamma, opt_beta, penalty)
    probs = opt_sv.probabilities()
    
    valid_probs = probs[:n_routes]
    best_state_idx = int(np.argmax(valid_probs))
    best_route = df.iloc[best_state_idx]["Route_ID"]
    best_cost = float(df.iloc[best_state_idx]["Total_Cost"])
    
    elapsed_time = time.perf_counter() - start_time
    
    return {
        "method": "QAOA Simulator",
        "route_id": best_route,
        "route_idx": best_state_idx,
        "cost": best_cost,
        "execution_time": elapsed_time,
        "probabilities": valid_probs,
        "optimal_params": (opt_gamma, opt_beta),
        "n_qubits": n_qubits,
        "row_data": df.iloc[best_state_idx]
    }

# ==========================================
# MAIN APPLICATION LAYOUT
# ==========================================
def main():
    # Load base dataset
    df_raw = load_dataset()
    
    # --------------------------------------
    # SIDEBAR: CONTROLS & FILTERS
    # --------------------------------------
    st.sidebar.image("https://img.icons8.com/isometric-line/100/38bdf8/airplane-mode-on.png", width=70)
    st.sidebar.title("Flight Optimizer Controls")
    st.sidebar.caption("Quantum & Multi-Objective Configuration")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚖️ Multi-Objective Weights")
    
    w_fuel = st.sidebar.slider("Fuel Weight (w_fuel)", 0.0, 1.0, 0.4, 0.05)
    w_time = st.sidebar.slider("Flight Time Weight (w_time)", 0.0, 1.0, 0.3, 0.05)
    w_weather = st.sidebar.slider("Weather Risk Weight (w_weather)", 0.0, 1.0, 0.2, 0.05)
    w_airspace = st.sidebar.slider("Airspace Risk Weight (w_airspace)", 0.0, 1.0, 0.1, 0.05)
    
    # Normalize weights so they sum to 1.0
    total_w = w_fuel + w_time + w_weather + w_airspace
    if total_w > 0:
        weights = (w_fuel/total_w, w_time/total_w, w_weather/total_w, w_airspace/total_w)
    else:
        weights = (0.25, 0.25, 0.25, 0.25)
        
    st.sidebar.info(f"Normalized Weights:\nFuel={weights[0]:.2f}, Time={weights[1]:.2f}, Weather={weights[2]:.2f}, Airspace={weights[3]:.2f}")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚛️ QAOA Solver Settings")
    qaoa_penalty = st.sidebar.number_input("Invalid State Penalty Factor", 1.0, 20.0, 5.0, 0.5)
    cobyla_maxiter = st.sidebar.slider("COBYLA Max Iterations", 20, 150, 60, 10)
    grid_density = st.sidebar.slider("Multi-Start Grid Points", 2, 5, 3, 1)
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔍 Dataset Filters")
    countries = ["All"] + sorted(list(df_raw["Waypoint_Country"].unique()))
    selected_country = st.sidebar.selectbox("Filter Waypoint Country", countries)
    
    aircraft_types = ["All"] + sorted(list(set(df_raw["Aircraft_Leg1"].unique()).union(set(df_raw["Aircraft_Leg2"].unique()))))
    selected_aircraft = st.sidebar.selectbox("Filter Aircraft Type", aircraft_types)
    
    # Filter dataset
    df_filtered = df_raw.copy()
    if selected_country != "All":
        df_filtered = df_filtered[df_filtered["Waypoint_Country"] == selected_country]
    if selected_aircraft != "All":
        df_filtered = df_filtered[(df_filtered["Aircraft_Leg1"] == selected_aircraft) | (df_filtered["Aircraft_Leg2"] == selected_aircraft)]
        
    if len(df_filtered) == 0:
        st.warning("No routes match the selected filters. Resetting to full dataset.")
        df_filtered = df_raw.copy()
        
    df_processed = calculate_normalized_costs(df_filtered, weights)
    
    # --------------------------------------
    # HEADER & KPI DASHBOARD
    # --------------------------------------
    st.title("🛸 Quantum-Enhanced Aircraft Route Optimization System")
    st.markdown("*USAF / DoD Collaborative Research Platform — Multi-Objective QAOA vs. Classical Route Solver*")
    
    # Run Solvers
    res_classical = run_classical_optimization(df_processed)
    res_qaoa = run_qaoa_optimization(df_processed, penalty=qaoa_penalty, maxiter=cobyla_maxiter, n_starts=grid_density)
    
    match_status = "YES (Optimal Match)" if res_classical["route_id"] == res_qaoa["route_id"] else "NO (Sub-Optimal)"
    match_color = "#059669" if res_classical["route_id"] == res_qaoa["route_id"] else "#d97706"
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Active Candidate Routes</div><div class="metric-value">{len(df_processed)}</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Min Normalized Cost</div><div class="metric-value">{res_classical["cost"]:.4f}</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Classical Best Route</div><div class="metric-value">{res_classical["route_id"]}</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-card"><div class="metric-label">QAOA Winner Route</div><div class="metric-value">{res_qaoa["route_id"]}</div></div>', unsafe_allow_html=True)
    with col5:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Solver Agreement</div><div class="metric-value" style="color: {match_color};">{match_status}</div></div>', unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # --------------------------------------
    # TABS: MAIN APP SECTIONS
    # --------------------------------------
    tab1, tab2, tab3, tab4 = st.tabs([
        "🗺️ Flight Path Map",
        "⚛️ QAOA vs Classical Solver",
        "📊 Interactive Analytics",
        "📂 Dataset Explorer"
    ])
    
    # --------------------------------------
    # TAB 1: FLIGHT PATH MAP
    # --------------------------------------
    with tab1:
        st.subheader("Global Flight Trajectory Map (DEL -> Waypoint Hub -> JFK)")
        st.caption("Interactive geographic visualization of candidate routes. Delhi (DEL) to New York (JFK) via intermediate international waypoints.")
        
        # Build Map using Plotly Arc / Scatter Mapbox
        fig_map = go.Figure()
        
        del_lat, del_lon = AIRPORT_COORDS["DEL"]["lat"], AIRPORT_COORDS["DEL"]["lon"]
        jfk_lat, jfk_lon = AIRPORT_COORDS["JFK"]["lat"], AIRPORT_COORDS["JFK"]["lon"]
        
        # Add Origin (DEL) and Destination (JFK) markers
        fig_map.add_trace(go.Scattergeo(
            lat=[del_lat, jfk_lat],
            lon=[del_lon, jfk_lon],
            mode="markers+text",
            marker=dict(size=14, color=["#ef4444", "#3b82f6"], symbol="triangle-up"),
            text=["DEL (New Delhi)", "JFK (New York)"],
            textposition="top center",
            name="Airports"
        ))
        
        # Add Waypoints & Arcs for each route
        for idx, row in df_processed.iterrows():
            r_id = row["Route_ID"]
            wp_city = row["Waypoint_City"]
            wp_country = row["Waypoint_Country"]
            wp_lat = row["Waypoint_Lat"]
            wp_lon = row["Waypoint_Lon"]
            cost = row["Total_Cost"]
            
            # Styling based on winner
            if r_id == res_classical["route_id"]:
                line_color = "#00FF66" # Vibrant Green for winner
                line_width = 4.5
                name_label = f"⭐ Winner {r_id} ({wp_city})"
            elif r_id == res_qaoa["route_id"] and r_id != res_classical["route_id"]:
                line_color = "#FF9900" # Orange for QAOA runner-up
                line_width = 3.5
                name_label = f"⚛️ QAOA {r_id} ({wp_city})"
            else:
                line_color = "rgba(56, 189, 248, 0.4)" # Translucent Cyan
                line_width = 1.8
                name_label = f"{r_id} ({wp_city})"
                
            # Flight Leg 1: DEL -> Waypoint
            fig_map.add_trace(go.Scattergeo(
                lat=[del_lat, wp_lat, jfk_lat],
                lon=[del_lon, wp_lon, jfk_lon],
                mode="lines+markers",
                line=dict(width=line_width, color=line_color),
                marker=dict(size=7, color=line_color),
                text=[f"DEL", f"Waypoint: {wp_city}, {wp_country}<br>Route: {r_id}<br>Cost: {cost:.4f}", f"JFK"],
                hoverinfo="text",
                name=name_label
            ))
            
        fig_map.update_layout(
            geo=dict(
                projection_type="natural earth",
                showland=True,
                landcolor="#1e293b",
                showocean=True,
                oceancolor="#0f172a",
                showlakes=True,
                lakecolor="#0f172a",
                showcountries=True,
                countrycolor="#334155",
                coastlinecolor="#475569"
            ),
            margin=dict(l=0, r=0, t=30, b=0),
            height=580,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, font=dict(color="white"))
        )
        
        st.plotly_chart(fig_map, use_container_width=True)
        
        # Route Detail Cards
        st.markdown("### 🏆 Selected Optimum Flight Details")
        c_win, q_win = st.columns(2)
        
        with c_win:
            st.success(f"**Classical Best Route: {res_classical['route_id']}**")
            row_c = res_classical["row_data"]
            st.markdown(f"""
            - **Flight Path:** `{row_c['Flight_Path']}`
            - **Hub / Waypoint:** {row_c['Waypoint_City']}, {row_c['Waypoint_Country']} ({row_c['Waypoint_IATA']})
            - **Total Distance:** {row_c['Distance_km']:,} km
            - **Fuel Consumption:** {row_c['Fuel_Consumption_kg']:,} kg
            - **Flight Time:** {row_c['Flight_Time_min']} mins ({row_c['Flight_Time_min']/60:.1f} hrs)
            - **Weather Risk:** {row_c['Weather_Risk_0_10']} / 10 | **Airspace Risk:** {row_c['Airspace_Risk_0_10']} / 10
            - **Aircraft:** Leg 1: `{row_c['Aircraft_Leg1']}` | Leg 2: `{row_c['Aircraft_Leg2']}`
            """)
            
        with q_win:
            st.info(f"**QAOA Winner Route: {res_qaoa['route_id']}**")
            row_q = res_qaoa["row_data"]
            st.markdown(f"""
            - **Flight Path:** `{row_q['Flight_Path']}`
            - **Hub / Waypoint:** {row_q['Waypoint_City']}, {row_q['Waypoint_Country']} ({row_q['Waypoint_IATA']})
            - **Total Distance:** {row_q['Distance_km']:,} km
            - **Fuel Consumption:** {row_q['Fuel_Consumption_kg']:,} kg
            - **Flight Time:** {row_q['Flight_Time_min']} mins ({row_q['Flight_Time_min']/60:.1f} hrs)
            - **Weather Risk:** {row_q['Weather_Risk_0_10']} / 10 | **Airspace Risk:** {row_q['Airspace_Risk_0_10']} / 10
            - **QAOA State Probability:** {res_qaoa['probabilities'][res_qaoa['route_idx']]*100:.1f}%
            """)

    # --------------------------------------
    # TAB 2: QAOA VS CLASSICAL SOLVER
    # --------------------------------------
    with tab2:
        st.subheader("Quantum (QAOA) vs Classical Solver Benchmark")
        
        col_t1, col_t2 = st.columns(2)
        
        with col_t1:
            st.markdown("#### ⚡ Solver Performance Benchmark Table")
            comp_table = pd.DataFrame([
                {
                    "Method": "Classical Brute-Force",
                    "Winning Route": res_classical["route_id"],
                    "Normalized Cost": f"{res_classical['cost']:.4f}",
                    "Execution Time": f"{res_classical['execution_time']*1000:.2f} ms",
                    "Algorithm Complexity": "O(N) Brute-Force"
                },
                {
                    "Method": f"QAOA Simulator ({res_qaoa['n_qubits']} Qubits)",
                    "Winning Route": res_qaoa["route_id"],
                    "Normalized Cost": f"{res_qaoa['cost']:.4f}",
                    "Execution Time": f"{res_qaoa['execution_time']*1000:.2f} ms",
                    "Algorithm Complexity": f"O(p * 2^{res_qaoa['n_qubits']}) Statevector"
                }
            ])
            st.table(comp_table)
            
            st.markdown("#### 📖 Quantum Circuit & Encoding Breakdown")
            st.markdown(rf"""
            - **Qubit Encoding:** Binary Log-Encoding ($\lceil\log_2 N\rceil = {res_qaoa['n_qubits']}$ qubits for {len(df_processed)} routes).
            - **Variational Depth ($p$):** $p=1$ QAOA Ansatz.
            - **Cost Hamiltonian ($H_C$):** $e^{{-i \gamma H_C}}$ phase separation operator.
            - **Mixer Unitary ($H_M$):** $e^{{-i \beta \sum X_k}} = \bigotimes R_x(2\beta)$ transverse field mixer.
            - **Optimal Variational Angles:** $\gamma^* = {res_qaoa['optimal_params'][0]:.4f}$, $\beta^* = {res_qaoa['optimal_params'][1]:.4f}$.
            """)
            
        with col_t2:
            st.markdown("#### 📈 QAOA State Measurement Probability Distribution")
            probs = res_qaoa["probabilities"]
            route_ids = df_processed["Route_ID"].values
            
            fig_prob = px.bar(
                x=route_ids,
                y=probs,
                labels={"x": "Aircraft Route", "y": "Measurement Probability"},
                title="Quantum Probability Amplification Across Routes",
                color=[1 if r == res_qaoa["route_id"] else 0 for r in route_ids],
                color_continuous_scale=["#38bdf8", "#00FF66"]
            )
            fig_prob.update_layout(
                showlegend=False,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white")
            )
            st.plotly_chart(fig_prob, use_container_width=True)

    # --------------------------------------
    # TAB 3: INTERACTIVE ANALYTICS
    # --------------------------------------
    with tab3:
        st.subheader("Multi-Objective Cost & Trade-off Analysis")
        
        c_a1, c_a2 = st.columns(2)
        
        with c_a1:
            st.markdown("#### 🏷️ Route Total Cost Ranking")
            df_sorted = df_processed.sort_values("Total_Cost")
            fig_cost = px.bar(
                df_sorted,
                x="Route_ID",
                y="Total_Cost",
                color="Total_Cost",
                color_continuous_scale="Viridis",
                hover_data=["Flight_Path", "Distance_km", "Fuel_Consumption_kg", "Weather_Risk_0_10", "Airspace_Risk_0_10"],
                title="Candidate Routes Ranked by Total Cost (Lower is Better)"
            )
            fig_cost.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
            st.plotly_chart(fig_cost, use_container_width=True)
            
        with c_a2:
            st.markdown("#### 🎯 Fuel vs Risk Trade-off Scatter Plot")
            fig_scatter = px.scatter(
                df_processed,
                x="Fuel_Consumption_kg",
                y="Weather_Risk_0_10",
                size="Distance_km",
                color="Total_Cost",
                hover_name="Route_ID",
                text="Route_ID",
                title="Fuel Consumption vs Weather Risk (Bubble Size = Distance)"
            )
            fig_scatter.update_traces(textposition="top center")
            fig_scatter.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
            st.plotly_chart(fig_scatter, use_container_width=True)
            
        st.markdown("#### 📊 Multi-Factor Comparison breakdown")
        fig_grouped = px.bar(
            df_processed,
            x="Route_ID",
            y=["Fuel_Norm", "Time_Norm", "Weather_Norm", "Airspace_Norm"],
            title="Normalized Cost Component Breakdown per Route",
            barmode="group"
        )
        fig_grouped.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
        st.plotly_chart(fig_grouped, use_container_width=True)

    # --------------------------------------
    # TAB 4: DATASET EXPLORER
    # --------------------------------------
    with tab4:
        st.subheader("Raw Dataset Explorer & Export")
        st.caption(f"Showing {len(df_processed)} routes with 21 metadata attributes extracted from Google Sheet dataset.")
        
        # Search filter
        search_query = st.text_input("🔍 Search Routes (e.g. London, Boeing, Dubai, Route A)", "")
        if search_query:
            mask = df_processed.astype(str).apply(lambda row: row.str.contains(search_query, case=False).any(), axis=1)
            df_display = df_processed[mask]
        else:
            df_display = df_processed.copy()
            
        st.dataframe(df_display, use_container_width=True, height=450)
        
        # Download CSV
        csv_data = df_display.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Filtered Dataset CSV",
            data=csv_data,
            file_name="quantum_aircraft_routes_filtered.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()
