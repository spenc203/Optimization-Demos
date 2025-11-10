import streamlit as st
import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt
import pandas as pd

# --- Configuration for Streamlit App ---
st.set_page_config(layout="wide", page_title="Profit Optimization App")

def solve_lp(objective_coeffs, constraint_matrix, constraint_bounds, var_names):
    """
    Uses the Simplex method via scipy.optimize.linprog to find the optimal solution.
    Since we are maximizing, we minimize the negative of the objective function.
    """
    # Convert maximization problem to minimization (min -Z = max Z)
    c = [-coeff for coeff in objective_coeffs]

    # Bounds for x and y (Non-negativity constraint: x >= 0, y >= 0)
    x_bounds = (0, None)
    y_bounds = (0, None)

    # Solve the linear programming problem
    result = linprog(
        c=c,
        A_ub=constraint_matrix,
        b_ub=constraint_bounds,
        bounds=[x_bounds, y_bounds],
        method='highs' # 'highs' is generally fast and robust
    )

    if result.success:
        optimal_x = round(result.x[0], 2)
        optimal_y = round(result.x[1], 2)
        max_profit = round(-result.fun, 2) # Remember to negate the minimized value
        return True, optimal_x, optimal_y, max_profit
    else:
        return False, None, None, None

def plot_feasible_region(A_ub, b_ub, objective_coeffs, var_names):
    """
    Generates the graph for the system of linear inequalities.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlabel(f'{var_names[0]} (units)', fontsize=14)
    ax.set_ylabel(f'{var_names[1]} (units)', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_title("Feasible Region and Optimal Solution (Graphical Method)", fontsize=16)

    # Determine the maximum required axis limit
    max_limit = max(max(b_ub) / min(A_ub[i]) if min(A_ub[i]) > 0 else max(b_ub) for i in range(len(b_ub)))
    max_limit = int(max_limit * 1.5) if max_limit > 0 else 15
    ax.set_xlim(0, max_limit)
    ax.set_ylim(0, max_limit)

    # Create a grid for shading
    x = np.linspace(0, max_limit, 400)
    y = np.linspace(0, max_limit, 400)
    X, Y = np.meshgrid(x, y)
    
    # Initialize mask for feasible region (must satisfy all constraints)
    feasible_mask = (X >= 0) & (Y >= 0)

    # Plot each constraint line and apply mask
    colors = ['r', 'g', 'b', 'm', 'c']
    labels = [
        "Baking Time: 0.5x + 1y ≤ 10",
        "Icing Time: 1x + 0.5y ≤ 8"
    ]

    for i in range(len(A_ub)):
        a1, a2 = A_ub[i]
        b = b_ub[i]

        # Calculate the line equation (y in terms of x): y = (b - a1*x) / a2
        if a2 != 0:
            Y_line = (b - a1 * X) / a2
            
            # Line plot
            x_line = np.linspace(0, max_limit, 100)
            y_line = (b - a1 * x_line) / a2
            ax.plot(x_line, y_line, color=colors[i % len(colors)], linestyle='-', label=f'Constraint {i+1}')
            
            # Apply the constraint mask (A_ub[i,0]*X + A_ub[i,1]*Y <= b_ub[i])
            feasible_mask &= (A_ub[i, 0] * X + A_ub[i, 1] * Y <= b)
    
    # Shade the feasible region
    ax.imshow(feasible_mask.astype(int), extent=(x.min(), x.max(), y.min(), y.max()), origin="lower", cmap='Greens', alpha=0.3, aspect='auto')
    
    # Add the optimal point (if found)
    success, optimal_x, optimal_y, max_profit = solve_lp(objective_coeffs, A_ub, b_ub, var_names)
    if success:
        # Plot optimal vertex
        ax.plot(optimal_x, optimal_y, 'o', color='gold', markersize=10, markeredgecolor='k', label=f"Optimal Point ({optimal_x}, {optimal_y})")
        # Add a text label for the optimal point
        ax.text(optimal_x, optimal_y + max_limit * 0.02, f'Optimal Profit: ${max_profit:.2f}', fontsize=10, ha='center', weight='bold')
    
    ax.legend(loc='upper right')
    return fig, success, optimal_x, optimal_y, max_profit

# --- The Placeholder Problem Data ---
# Decision Variables: x = Cake A (units), y = Cake B (units)
VAR_NAMES = ["Cake A", "Cake B"]

# Objective Function: Maximize P = 20x + 30y
OBJECTIVE_COEFFS = [20, 30]

# Constraints (LHS of inequalities: A_ub * [x, y] <= b_ub)
# Constraint 1 (Baking Time): 0.5x + 1y <= 10
# Constraint 2 (Icing Time): 1x + 0.5y <= 8
CONSTRAINT_MATRIX = np.array([
    [0.5, 1], # x-coeff, y-coeff for Constraint 1
    [1, 0.5]  # x-coeff, y-coeff for Constraint 2
])
CONSTRAINT_BOUNDS = np.array([10, 8]) # RHS limits

# --- Streamlit UI Layout ---
st.markdown("<h1 style='text-align: center; color: #4F80E7;'>Business Optimization App: Maximizing Profit</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #333;'>A Real-World Application of Linear Programming</h3>", unsafe_allow_html=True)
st.divider()

# --- Section 1: The Business Interface (Focus on Results) ---
st.header("1. Business Dashboard: The Optimal Production Plan 📈")
st.markdown(f"**Scenario:** A bakery must decide how many units of {VAR_NAMES[0]} and {VAR_NAMES[1]} to produce to maximize profit, given limited resource hours (Baking Time and Icing Time).")

# Solve the LP problem
success, optimal_x, optimal_y, max_profit = solve_lp(OBJECTIVE_COEFFS, CONSTRAINT_MATRIX, CONSTRAINT_BOUNDS, VAR_NAMES)

if success:
    col_kpi_1, col_kpi_2, col_kpi_3 = st.columns(3)

    with col_kpi_1:
        st.metric(label="Maximum Profit Potential", value=f"${max_profit:,.2f}", delta="Optimal Solution")

    with col_kpi_2:
        st.metric(label=f"Optimal Production of {VAR_NAMES[0]} (x)", value=f"{optimal_x} units")

    with col_kpi_3:
        st.metric(label=f"Optimal Production of {VAR_NAMES[1]} (y)", value=f"{optimal_y} units")

    st.success(f"To achieve the maximum profit of **${max_profit:,.2f}**, the company should produce **{optimal_x} units of {VAR_NAMES[0]}** and **{optimal_y} units of {VAR_NAMES[1]}**.")

else:
    st.error("The optimization problem failed to find a feasible solution. Check constraints.")

st.divider()

# --- Section 2: The Educational Interface (Focus on Math) ---
st.header("2. Mathematical Model: Visualizing the System of Inequalities 📐")
st.markdown("This section shows the high school math that determines the solution, using the **Graphical Method**.")

# 1. Display the Model
st.subheader("Model Formulation")
st.code(
    f"""
    Objective Function (Maximize Profit P):
    P = {OBJECTIVE_COEFFS[0]}x + {OBJECTIVE_COEFFS[1]}y

    Constraints:
    1. {CONSTRAINT_MATRIX[0, 0]}x + {CONSTRAINT_MATRIX[0, 1]}y ≤ {CONSTRAINT_BOUNDS[0]} (Baking Time)
    2. {CONSTRAINT_MATRIX[1, 0]}x + {CONSTRAINT_MATRIX[1, 1]}y ≤ {CONSTRAINT_BOUNDS[1]} (Icing Time)
    3. x ≥ 0
    4. y ≥ 0
    """,
    language="python"
)

col_graph, col_table = st.columns([2, 1])

with col_graph:
    # 2. Display the Graph
    st.subheader("Feasible Region Graph")
    # Plot the graph using the helper function
    fig, _, _, _, _ = plot_feasible_region(CONSTRAINT_MATRIX, CONSTRAINT_BOUNDS, OBJECTIVE_COEFFS, VAR_NAMES)
    st.pyplot(fig)
    st.caption("The green shaded area represents all possible (feasible) production combinations.")

with col_table:
    # 3. Display the Corner Point Evaluation
    st.subheader("Corner Point Evaluation")

    if success:
        # In a real app, you would mathematically find the intersection points (vertices).
        # For simplicity and demonstration, we will just show the optimal vertex calculation.
        
        # NOTE: The actual vertices for the placeholder problem (0, 0), (0, 10), (8, 0), (4, 8)
        # However, the optimal point (4, 8) is an intersection. Let's list a few for demo.
        vertices = [
            (0, 0),
            (8, 0), # Intersection of x <= 8 and y = 0
            (0, 10), # Intersection of 0.5x + y <= 10 and x = 0
            (4, 8) # The actual intersection of the two lines is (4, 8)
        ]

        # Calculate profit at each vertex
        data = []
        for x_v, y_v in vertices:
            P = OBJECTIVE_COEFFS[0] * x_v + OBJECTIVE_COEFFS[1] * y_v
            data.append({
                "Vertex (x, y)": f"({x_v}, {y_v})",
                f"Profit P = {OBJECTIVE_COEFFS[0]}x + {OBJECTIVE_COEFFS[1]}y": f"${P:,.2f}",
                "Is Optimal?": "👑" if x_v == optimal_x and y_v == optimal_y else ""
            })

        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.caption("The maximum profit is always found at a corner (vertex) of the feasible region.")
    else:
        st.info("Cannot display corner points as no feasible region was found.")

st.markdown("""
<style>
    .stCodeBlock {
        background-color: #f7f9fc;
        border-left: 5px solid #4F80E7;
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Placeholder to demonstrate how to find the intersection (vertex) manually
# 0.5x + y = 10  => 1y = 10 - 0.5x
# x + 0.5y = 8   => x + 0.5(10 - 0.5x) = 8
# x + 5 - 0.25x = 8
# 0.75x = 3
# x = 4
# y = 10 - 0.5(4) = 8
# Intersection (4, 8)