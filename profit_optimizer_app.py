import streamlit as st
import numpy as np
from scipy.optimize import linprog

# Note: Matplotlib and Pandas are removed as the visual elements are not needed.

# --- Configuration for Streamlit App ---
st.set_page_config(layout="wide", page_title="Professional Optimization Report")

def solve_lp(objective_coeffs, constraint_matrix, constraint_bounds, var_names):
    """
    Uses the Simplex method via scipy.optimize.linprog to find the optimal solution.
    """
    # Convert maximization problem to minimization (min -Z = max Z)
    c = [-coeff for coeff in objective_coeffs]
    x_bounds = (0, None)
    y_bounds = (0, None)

    result = linprog(
        c=c,
        A_ub=constraint_matrix,
        b_ub=constraint_bounds,
        bounds=[x_bounds, y_bounds],
        method='highs'
    )

    if result.success:
        optimal_x = round(result.x[0], 2)
        optimal_y = round(result.x[1], 2)
        max_profit = round(-result.fun, 2)
        return True, optimal_x, optimal_y, max_profit
    else:
        return False, None, None, None

# --- Streamlit UI Layout ---
st.markdown("<h1 style='text-align: center; color: #3A78BF;'>Professional Optimization Report: Production Planning</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #555;'>Adjust inputs to generate the Optimal Production Plan.</h3>", unsafe_allow_html=True)
st.divider()

# --- INPUT SECTION: Control the Variables and Constraints ---
st.header("1. Input Parameters (The Business Scenario)")

# 1. Define Decision Variables (Profits)
col_p1, col_p2 = st.columns(2)
with col_p1:
    profit_car = st.number_input("Profit per Car ($/unit)", min_value=1.0, value=300.0, step=10.0, format="%.2f", key="p_car")
    car_name = "Car (x)"
with col_p2:
    profit_truck = st.number_input("Profit per Truck ($/unit)", min_value=1.0, value=500.0, step=10.0, format="%.2f", key="p_truck")
    truck_name = "Truck (y)"

VAR_NAMES = [car_name, truck_name]
OBJECTIVE_COEFFS = [profit_car, profit_truck]

st.subheader("Resource Constraints (Maximum Available Hours)")
col_limits_1, col_limits_2 = st.columns(2)
with col_limits_1:
    assembly_limit = st.slider("Max Assembly Hours Available", min_value=10, max_value=200, value=120, step=5, key="limit_a")
with col_limits_2:
    painting_limit = st.slider("Max Painting Hours Available", min_value=10, max_value=200, value=80, step=5, key="limit_p")
CONSTRAINT_BOUNDS = np.array([assembly_limit, painting_limit])

st.subheader("Resource Usage per Vehicle (Required Hours)")
col_usage_1, col_usage_2, col_usage_3, col_usage_4 = st.columns(4)

with col_usage_1:
    assembly_car_usage = st.number_input("Assembly Hours for 1 Car", min_value=0.1, max_value=20.0, value=1.0, step=0.1, key="usage_ac")
with col_usage_2:
    assembly_truck_usage = st.number_input("Assembly Hours for 1 Truck", min_value=0.1, max_value=20.0, value=4.0, step=0.1, key="usage_at")

with col_usage_3:
    painting_car_usage = st.number_input("Painting Hours for 1 Car", min_value=0.1, max_value=20.0, value=2.0, step=0.1, key="usage_pc")
with col_usage_4:
    painting_truck_usage = st.number_input("Painting Hours for 1 Truck", min_value=0.1, max_value=20.0, value=2.0, step=0.1, key="usage_pt")

CONSTRAINT_MATRIX = np.array([
    [assembly_car_usage, assembly_truck_usage], # Assembly Constraint
    [painting_car_usage, painting_truck_usage]  # Painting Constraint
])

st.divider()

# --- Section 2: THE OPTIMIZATION RESULTS ---
st.header("2. The Optimal Production Plan (Manager's Report)")

# Solve the LP problem
success, optimal_x, optimal_y, max_profit = solve_lp(OBJECTIVE_COEFFS, CONSTRAINT_MATRIX, CONSTRAINT_BOUNDS, VAR_NAMES)

if success:
    st.success(f"Based on the inputs, the optimal plan yields a maximum profit of **${max_profit:,.2f}**.")
    
    col_kpi_1, col_kpi_2, col_kpi_3 = st.columns(3)

    with col_kpi_1:
        st.metric(label="Maximum Profit Potential (Target)", value=f"${max_profit:,.2f}", delta="Optimal Solution")

    with col_kpi_2:
        st.metric(label=f"Production Target: {car_name} (x)", value=f"{optimal_x} units")

    with col_kpi_3:
        st.metric(label=f"Production Target: {truck_name} (y)", value=f"{optimal_y} units")
        
    st.markdown("""
        **Actionable Summary:** This report provides the necessary production targets (x and y) to maximize the objective function, P. This is the only type of report generated when solving complex problems (more than two variables).
    """)
    
    # --- Constraint Model Review ---
    st.subheader("Mathematical Model Review (The Inputs)")
    st.code(
        f"""
        Objective Function (Maximize Profit P): P = {profit_car}x + {profit_truck}y

        Constraints (Limits on Resources):
        1. Assembly: {assembly_car_usage}x + {assembly_truck_usage}y <= {assembly_limit}
        2. Painting: {painting_car_usage}x + {painting_truck_usage}y <= {painting_limit}
        3. Non-negativity: x >= 0, y >= 0
        """,
        language="python"
    )

else:
    st.error("The current parameters lead to an infeasible problem (e.g., resource limits are too low). Please adjust the input sliders and usage numbers.")

st.markdown("""
<style>
    /* Styling to make the layout clean and modern */
    .stMetric > div > div:nth-child(2) {
        font-size: 2.5rem;
        color: #1E88E5 !important;
    }
    .stSlider > div > div:nth-child(1) {
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)