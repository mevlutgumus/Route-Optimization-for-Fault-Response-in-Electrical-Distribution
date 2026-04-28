import json
import math
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium
from folium.plugins import AntPath
import pulp as lp
from geopy.distance import geodesic

# =========================================================================
# CONSTANTS AND SETTINGS
# =========================================================================
OPTIMAL_STATUS_CODE = 1

st.set_page_config(page_title="Fault Response Optimization", layout="wide")
st.title("Route Optimization for Fault Response in Electrical Distribution")

# =========================================================================
# CREW LOCATIONS (BEDAS Operating Directorates — European Side of Istanbul)
# =========================================================================
crew_data = {
    "Beyoglu": (41.042942843441594, 28.98187509471993),
    "Beyazit": (41.01255990927693, 28.962134641262114),
    "Bayrampasa": (41.046302182999646, 28.910872668799808),
    "Bakirkoy": (40.98605787570794, 28.89211399154593),
    "Basaksehir": (41.09662872610036, 28.789892375665104),
    "Besyol": (41.02375414632992, 28.790824905276498),
    "Caglayan": (41.07210166553191, 28.982043223356346),
    "Gungoren": (41.02151226059597, 28.887805521157343),
    "Sefakoy": (40.99755768113406, 28.829276198411225),
    "Sariyer": (41.032477772499725, 28.904751568799814)
}
crew_list = list(crew_data.keys())

# =========================================================================
# GEOJSON READER
# =========================================================================
def load_substations_from_geojson(path: str):
    with open(path, "r", encoding="utf-8") as f:
        geojson_data = json.load(f)

    all_substation_coords = {}
    counter = 1

    for feature in geojson_data["features"]:
        geometry = feature.get("geometry", {})
        props = feature.get("properties", {})

        if "power" in props and props["power"] == "substation":
            if geometry.get("type") == "Point":
                lon, lat = geometry["coordinates"]
            elif geometry.get("type") == "MultiPolygon":
                lon, lat = geometry["coordinates"][0][0][0]
            elif geometry.get("type") == "Polygon":
                lon, lat = geometry["coordinates"][0][0]
            else:
                continue

            all_substation_coords[f"Substation_{counter}"] = (lat, lon)
            counter += 1

    return all_substation_coords

# =========================================================================
# GEODESIC DISTANCE CALCULATION — CREW → SUBSTATION
# =========================================================================
def compute_C_ij_geodesic(crew_data, substation_coords):
    C_ij = {}
    for i, (i_lat, i_lon) in crew_data.items():
        C_ij[i] = {}
        crew_loc = (i_lat, i_lon)
        for j, (j_lat, j_lon) in substation_coords.items():
            sub_loc = (j_lat, j_lon)
            distance = geodesic(crew_loc, sub_loc).km
            C_ij[i][j] = round(distance, 2)
    return C_ij

# =========================================================================
# GEODESIC DISTANCE MATRIX — BETWEEN ALL NODES (FOR TSP)
# =========================================================================
def compute_distance_matrix(node_coords):
    """
    node_coords: dict { node_id: (lat, lon), ... }
    Returns a geodesic distance matrix between all node pairs.
    """
    nodes = list(node_coords.keys())
    d = {}
    for j in nodes:
        d[j] = {}
        for k in nodes:
            if j == k:
                d[j][k] = 0.0
            else:
                distance = geodesic(node_coords[j], node_coords[k]).km
                d[j][k] = round(distance, 2)
    return d

# =========================================================================
# SOLVE GAP (PULP)
# =========================================================================
def solve_gap(C_ij, crew_list, fault_list, cap_dict, priority_map=None):
    prob = lp.LpProblem("GAP_Assignment", lp.LpMinimize)
    X_ij = lp.LpVariable.dicts(
        "Assign",
        [(i, j) for i in crew_list for j in fault_list],
        cat=lp.LpBinary
    )

    # Priority weight: critical faults get much lower cost so solver
    # assigns them to the nearest possible crew first
    # P1=0.1, P2=0.3, P3=0.5, P4=1.0
    priority_weight = {1: 0.1, 2: 0.3, 3: 0.5, 4: 1.0}

    # Objective Function: Minimize priority-weighted assignment distance
    prob += lp.lpSum(
        C_ij[i][j] * priority_weight.get(priority_map.get(j, 4) if priority_map else 4, 1.0) * X_ij[(i, j)]
        for i in crew_list for j in fault_list
    )

    # K1: Each fault must be assigned to exactly one crew
    for j in fault_list:
        prob += lp.lpSum(X_ij[(i, j)] for i in crew_list) == 1

    # K2: Crew capacities must not be exceeded
    for i in crew_list:
        prob += lp.lpSum(X_ij[(i, j)] for j in fault_list) <= int(cap_dict[i])

    # K5: Priority distribution — spread critical faults across crews
    # Ensures no crew is overloaded with high-priority faults while others have none
    if priority_map:
        import math
        for p_level in [1, 2, 3]:
            faults_at_p = [j for j in fault_list if priority_map.get(j, 4) == p_level]
            if len(faults_at_p) > 0:
                max_per_crew = math.ceil(len(faults_at_p) / len(crew_list))
                for i in crew_list:
                    prob += lp.lpSum(X_ij[(i, j)] for j in faults_at_p) <= max_per_crew

    prob.solve(lp.PULP_CBC_CMD(msg=0))
    return prob, X_ij

# =========================================================================
# SOLVE PC-TSP (PULP) — SEPARATELY FOR EACH CREW
# =========================================================================
def solve_tsp_for_team(team_name, depot_coord, assigned_faults, substation_coords,
                       priority_map, priority_mode=True, time_limit=60):
    """
    team_name       : Crew name (str)
    depot_coord     : Crew location (lat, lon)
    assigned_faults : List of faults assigned to this crew [str]
    substation_coords: All substation coordinates dict
    priority_map    : { substation_id: priority_level (1-4) }
    priority_mode   : True = K4 constraints active (Emergency), False = distance only (Standard)
    time_limit      : CBC solver time limit (seconds)

    Return: (status, route_order, route_distance, solve_time, visit_order)
    """
    import time

    N_i = assigned_faults  # Fault nodes excluding depot
    n = len(N_i)

    # Edge cases: zero or single fault
    if n == 0:
        return ("Optimal", [team_name], 0.0, 0.0, {})
    if n == 1:
        d_go = geodesic(depot_coord, substation_coords[N_i[0]]).km
        d_back = d_go
        return ("Optimal", [team_name, N_i[0], team_name], round(d_go + d_back, 2), 0.0, {})

    # Node coordinates (depot = "DEPOT")
    node_coords = {"DEPOT": depot_coord}
    for j in N_i:
        node_coords[j] = substation_coords[j]

    all_nodes = list(node_coords.keys())  # DEPOT + fault nodes

    # Distance matrix
    d = compute_distance_matrix(node_coords)

    # ---- MODEL ----
    prob = lp.LpProblem(f"PCTSP_{team_name}", lp.LpMinimize)

    # Decision variables
    # y_jk: Does the crew travel directly from node j to node k? (binary)
    y = lp.LpVariable.dicts("y",
        [(j, k) for j in all_nodes for k in all_nodes if j != k],
        cat=lp.LpBinary)

    # u_j: Visit order of node j in the route (integer, fault nodes only)
    u = lp.LpVariable.dicts("u", N_i, lowBound=1, upBound=n, cat=lp.LpInteger)

    # Objective Function: Minimize total route distance
    prob += lp.lpSum(d[j][k] * y[(j, k)] for j in all_nodes for k in all_nodes if j != k)

    # K1: Each node must be exited exactly once
    for j in all_nodes:
        prob += lp.lpSum(y[(j, k)] for k in all_nodes if k != j) == 1

    # K2: Each node must be entered exactly once
    for k in all_nodes:
        prob += lp.lpSum(y[(j, k)] for j in all_nodes if j != k) == 1

    # K3: MTZ Sub-tour Elimination (for all non-depot node pairs)
    for j in N_i:
        for k in N_i:
            if j != k:
                prob += u[j] - u[k] + n * y[(j, k)] <= n - 1

    # K4: Priority Ranking Constraints (only when priority_mode is active)
    if priority_mode:
        # Build priority sets
        P_sets = {1: [], 2: [], 3: [], 4: []}
        for j in N_i:
            p_level = priority_map.get(j, 4)  # Default P4 (normal)
            P_sets[p_level].append(j)

        # Add constraints for every pair of non-empty priority sets where r < s
        priority_levels = sorted(P_sets.keys())
        for idx_r, r in enumerate(priority_levels):
            for s in priority_levels[idx_r + 1:]:
                if len(P_sets[r]) > 0 and len(P_sets[s]) > 0:
                    for j in P_sets[r]:
                        for k in P_sets[s]:
                            # u_j + 1 <= u_k (strict ordering for integer variables)
                            prob += u[j] + 1 <= u[k]

    # Solve
    start_time = time.time()
    prob.solve(lp.PULP_CBC_CMD(msg=0, timeLimit=time_limit))
    solve_time = round(time.time() - start_time, 3)

    status_text = lp.LpStatus[prob.status]

    if prob.status != OPTIMAL_STATUS_CODE:
        return (status_text, [], 0.0, solve_time, {})

    # Build route (starting from DEPOT, following y_jk = 1)
    route = ["DEPOT"]
    current = "DEPOT"
    visited = set()
    for _ in range(len(all_nodes)):
        visited.add(current)
        for k in all_nodes:
            if k != current and k not in visited:
                if lp.value(y[(current, k)]) is not None and lp.value(y[(current, k)]) > 0.5:
                    route.append(k)
                    current = k
                    break
        else:
            route.append("DEPOT")
            break

    if route[-1] != "DEPOT":
        route.append("DEPOT")

    # Calculate total route distance
    total_dist = 0.0
    for idx in range(len(route) - 1):
        total_dist += d[route[idx]][route[idx + 1]]

    # Replace DEPOT with crew name for display
    display_route = [team_name if x == "DEPOT" else x for x in route]

    # Extract visit order (u_j values)
    visit_order = {}
    for j in N_i:
        val = lp.value(u[j])
        if val is not None:
            visit_order[j] = int(val)

    return (status_text, display_route, round(total_dist, 2), solve_time, visit_order)


# =========================================================================
# SIDEBAR (INPUTS)
# =========================================================================
with st.sidebar:
    st.header("Inputs")

    geojson_path = st.text_input("GeoJSON file path", value="export.geojson")

    try:
        all_substation_coords = load_substations_from_geojson(geojson_path)
        st.success(f"Substations found: {len(all_substation_coords)}")
        geo_ok = True
    except:
        st.error("GeoJSON file could not be read or found.")
        geo_ok = False
        all_substation_coords = {}

    st.subheader("Faulty Substations")
    substation_options = list(all_substation_coords.keys())
    selected_faults = st.multiselect(
        "Select faulty substations",
        options=substation_options,
        default=substation_options[:10] if len(substation_options) >= 10 else substation_options
    )

    # ---- CREW CAPACITIES ----
    st.subheader("Crew Capacities")
    cap_mode = st.selectbox(
        "Capacity type",
        ["Single value (same for all)", "Per crew (individual)", "Optimal (System Recommendation)"]
    )

    cap_dict = {}
    if cap_mode == "Optimal (System Recommendation)":
        if len(crew_list) > 0:
            calc_cap = math.ceil(len(selected_faults) / len(crew_list)) + 1
        else:
            calc_cap = len(selected_faults)
        cap_dict = {i: int(calc_cap) for i in crew_list}
        st.info(f"System assigned **{calc_cap}** capacity to each crew for balanced workload distribution.")
    elif cap_mode == "Single value (same for all)":
        default_cap = max(1, math.ceil(max(len(selected_faults), 1) / max(len(crew_list), 1)) + 1)
        max_cap = st.number_input("Capacity (max_cap)", min_value=1, value=int(default_cap), step=1)
        cap_dict = {i: int(max_cap) for i in crew_list}
    else:
        for i in crew_list:
            cap_dict[i] = int(st.number_input(f"{i} capacity", min_value=0, value=3, step=1))

    # ---- PRIORITY SETTINGS ----
    st.subheader("Priority Settings (TSP)")

    routing_mode = st.radio(
        "Routing Mode",
        ["Standard (Distance Only)", "Emergency Response (Priority-Constrained)"],
        index=1,
        help="Standard: Minimizes total travel distance. Emergency Response: Guarantees P1→P2→P3→P4 visit order."
    )
    priority_mode = (routing_mode == "Emergency Response (Priority-Constrained)")

    priority_map = {}
    if priority_mode and len(selected_faults) > 0:
        st.markdown("**Select critical points** (unselected substations default to P4)")

        # Dynamic key suffix to reset dropdowns when selected faults change
        fault_key = str(len(selected_faults)) + "_" + str(hash(tuple(sorted(selected_faults))) % 10000)

        p1_faults = st.multiselect(
            "P1 — Life-Critical (Hospital, Dialysis Center)",
            options=selected_faults,
            default=[],
            key=f"p1_{fault_key}"
        )
        remaining_after_p1 = [f for f in selected_faults if f not in p1_faults]

        p2_faults = st.multiselect(
            "P2 — Security/Public Order (Military Zone, Police Station)",
            options=remaining_after_p1,
            default=[],
            key=f"p2_{fault_key}"
        )
        remaining_after_p2 = [f for f in remaining_after_p1 if f not in p2_faults]

        p3_faults = st.multiselect(
            "P3 — Socially Sensitive (School, Care Home)",
            options=remaining_after_p2,
            default=[],
            key=f"p3_{fault_key}"
        )

        for f in p1_faults:
            priority_map[f] = 1
        for f in p2_faults:
            priority_map[f] = 2
        for f in p3_faults:
            priority_map[f] = 3
        for f in selected_faults:
            if f not in priority_map:
                priority_map[f] = 4

        # Summary
        st.markdown(f"P1: **{len(p1_faults)}**, P2: **{len(p2_faults)}**, "
                    f"P3: **{len(p3_faults)}**, P4: **{len(selected_faults) - len(p1_faults) - len(p2_faults) - len(p3_faults)}**")
    else:
        for f in selected_faults:
            priority_map[f] = 4

    run_btn = st.button("SOLVE / UPDATE MAP", type="primary")

if not geo_ok:
    st.stop()

# Prepare simulation data
active_faults = [a for a in selected_faults if a in all_substation_coords]
substation_coords = {j: all_substation_coords[j] for j in active_faults}
fault_list = list(substation_coords.keys())

# =========================================================================
# MAIN LAYOUT
# =========================================================================
col_map, col_right = st.columns([1.2, 0.8], gap="large")

if run_btn or ("last_solution" not in st.session_state):
    if len(fault_list) == 0:
        st.warning("No faulty substations selected.")
        st.stop()

    # ---- STAGE 1: GAP ----
    with st.spinner("Stage 1: GAP — Computing geodesic distances and assigning crews..."):
        C_ij = compute_C_ij_geodesic(crew_data, substation_coords)
        prob_gap, X_ij = solve_gap(C_ij, crew_list, fault_list, cap_dict, priority_map)

    if prob_gap.status != OPTIMAL_STATUS_CODE:
        st.error("GAP optimal solution not found. Try increasing crew capacities.")
        st.stop()

    gap_objective = float(lp.value(prob_gap.objective))

    # Extract assignment matrix from GAP results
    Xvals = {(i, j): float(lp.value(X_ij[(i, j)])) for i in crew_list for j in fault_list}

    # Determine faults assigned to each crew
    team_assignments = {}
    for i in crew_list:
        team_assignments[i] = [j for j in fault_list if Xvals.get((i, j), 0) > 0.5]

    # ---- STAGE 2: PC-TSP (Separately for each crew) ----
    tsp_results = {}
    with st.spinner("Stage 2: PC-TSP — Solving route optimization for each crew..."):
        for team_name in crew_list:
            faults = team_assignments[team_name]
            if len(faults) == 0:
                tsp_results[team_name] = ("Optimal", [team_name], 0.0, 0.0, {})
                continue

            result = solve_tsp_for_team(
                team_name=team_name,
                depot_coord=crew_data[team_name],
                assigned_faults=faults,
                substation_coords=substation_coords,
                priority_map=priority_map,
                priority_mode=priority_mode,
                time_limit=60
            )
            tsp_results[team_name] = result

    # Save to session state
    st.session_state.last_solution = {
        "C_ij": C_ij,
        "gap_status": lp.LpStatus[prob_gap.status],
        "gap_objective": gap_objective,
        "Xvals": Xvals,
        "substation_coords": substation_coords,
        "cap_dict": cap_dict,
        "team_assignments": team_assignments,
        "tsp_results": tsp_results,
        "priority_map": priority_map,
        "priority_mode": priority_mode,
    }

# Retrieve results from session state
sol = st.session_state.last_solution
C_ij = sol["C_ij"]
Xvals = sol["Xvals"]
substation_coords = sol["substation_coords"]
cap_dict = sol["cap_dict"]
team_assignments = sol["team_assignments"]
tsp_results = sol["tsp_results"]
priority_map = sol["priority_map"]
is_priority = sol["priority_mode"]

# =========================================================================
# RIGHT PANEL: RESULT TABLES
# =========================================================================
with col_right:
    st.subheader("Summary")
    st.write(f"GAP Status: **{sol['gap_status']}**")
    st.success(f"GAP Total Assignment Distance: **{sol['gap_objective']:.2f} km**")

    # Total TSP distance
    total_tsp = sum(r[2] for r in tsp_results.values())
    st.success(f"TSP Total Route Distance: **{total_tsp:.2f} km**")

    mode_text = "Emergency Response (P1→P2→P3→P4)" if is_priority else "Standard (Distance Only)"
    st.info(f"Routing Mode: **{mode_text}**")

    # Priority legend (only in Emergency Response mode)
    if is_priority:
        st.markdown(
            '<div style="background:#1a1a2e; border:1px solid #333; border-radius:8px; padding:12px 16px; margin-bottom:12px;">'
            '<div style="font-weight:bold; color:#ccc; margin-bottom:8px; font-size:13px;">Priority Legend</div>'
            '<div style="margin:4px 0;"><span style="color:#ff0000; text-shadow:0 0 8px #ff0000; font-weight:bold;">⬤</span> '
            '<span style="color:#ff4444; text-shadow:0 0 6px #ff0000;">P1 — Life-Critical (Hospital, Dialysis Center, etc.)</span></div>'
            '<div style="margin:4px 0;"><span style="color:#ff8c00; text-shadow:0 0 8px #ff8c00; font-weight:bold;">⬤</span> '
            '<span style="color:#ffaa33; text-shadow:0 0 6px #ff8c00;">P2 — Security / Public Order (Military Zone, etc.)</span></div>'
            '<div style="margin:4px 0;"><span style="color:#ffd700; text-shadow:0 0 8px #ffd700; font-weight:bold;">⬤</span> '
            '<span style="color:#ffe44d; text-shadow:0 0 6px #ffd700;">P3 — Socially Sensitive (School, Care Home, etc.)</span></div>'
            '</div>',
            unsafe_allow_html=True
        )

    # ---- ASSIGNMENT LIST ----
    st.subheader("Assignment List (GAP)")
    rows = []
    for j in fault_list:
        assigned_i = None
        for i in crew_list:
            if Xvals.get((i, j), 0) > 0.5:
                assigned_i = i
                break
        if assigned_i:
            p_level = priority_map.get(j, 4)
            p_label = {1: "P1-Life Critical", 2: "P2-Security", 3: "P3-Social", 4: "P4-Normal"}.get(p_level, "P4")
            rows.append({
                "Fault (Substation)": j,
                "Assigned Crew": assigned_i,
                "Priority": p_label,
                "Distance (km)": C_ij[assigned_i][j]
            })

    df_assign = pd.DataFrame(rows).sort_values(["Assigned Crew", "Priority", "Fault (Substation)"], ignore_index=True)
    st.dataframe(df_assign, use_container_width=True, height=300)

    # ---- CREW-LEVEL ROUTE RESULTS ----
    st.subheader("Crew Route Results (TSP)")
    route_rows = []
    for team_name in crew_list:
        result = tsp_results[team_name]
        status = result[0]
        route = result[1]
        dist = result[2]
        solve_time = result[3]
        fault_count = len(team_assignments[team_name])

        if fault_count == 0:
            continue

        route_rows.append({
            "Crew": team_name,
            "Fault Count": fault_count,
            "Capacity": cap_dict[team_name],
            "Route Distance (km)": dist,
            "Solve Time (s)": solve_time,
            "Status": status,
        })

    if route_rows:
        df_route = pd.DataFrame(route_rows)
        st.dataframe(df_route, use_container_width=True, height=280)

    # ---- CREW DETAIL: ROUTE ORDER ----
    st.subheader("Crew Route Details")
    for team_name in crew_list:
        result = tsp_results[team_name]
        route = result[1]
        fault_count = len(team_assignments[team_name])

        if fault_count == 0:
            continue

        with st.expander(f"{team_name} — {fault_count} faults, {result[2]} km"):
            st.write("**Route:** " + " → ".join(route))

            # Visit order table
            if len(result) > 4 and result[4]:
                visit_order = result[4]
                order_rows = []
                for node, order in sorted(visit_order.items(), key=lambda x: x[1]):
                    p_level = priority_map.get(node, 4)
                    p_label = {1: "P1", 2: "P2", 3: "P3", 4: "P4"}.get(p_level, "P4")
                    order_rows.append({
                        "Order (u_j)": order,
                        "Node": node,
                        "Priority": p_label,
                    })
                st.dataframe(pd.DataFrame(order_rows), use_container_width=True, hide_index=True)

# =========================================================================
# LEFT PANEL: MAP
# =========================================================================
with col_map:
    st.subheader("Map — Route Visualization")

    lat_mean = sum(v[0] for v in crew_data.values()) / len(crew_data)
    lon_mean = sum(v[1] for v in crew_data.values()) / len(crew_data)

    m = folium.Map(location=(lat_mean, lon_mean), zoom_start=11, control_scale=True)

    palette = [
        "red", "blue", "green", "purple", "orange",
        "darkred", "cadetblue", "darkgreen", "darkblue", "pink"
    ]
    hex_palette = [
        "#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#e67e22",
        "#c0392b", "#5dade2", "#1abc9c", "#2c3e50", "#ff69b4"
    ]
    crew_color = {crew_list[idx]: palette[idx % len(palette)] for idx in range(len(crew_list))}
    crew_hex = {crew_list[idx]: hex_palette[idx % len(hex_palette)] for idx in range(len(crew_list))}

    # Add crews to map
    for i, (ilat, ilon) in crew_data.items():
        fault_count = len(team_assignments.get(i, []))
        folium.Marker(
            location=(ilat, ilon),
            tooltip=f"Crew: {i} | Capacity: {cap_dict.get(i, 0)} | Assigned: {fault_count}",
            icon=folium.Icon(color=crew_color[i], icon="users", prefix="fa")
        ).add_to(m)

    # Priority color coding (border colors for critical markers)
    priority_border = {1: "#ff0000", 2: "#ff8c00", 3: "#ffd700", 4: "transparent"}

    # Add substations to map
    for j, (jlat, jlon) in substation_coords.items():
        assigned_i = None
        for i in crew_list:
            if Xvals.get((i, j), 0) > 0.5:
                assigned_i = i
                break

        if assigned_i is None:
            color = "gray"
            color_hex_j = "#808080"
            assigned_i_text = "Unassigned"
        else:
            color = crew_color[assigned_i]
            color_hex_j = crew_hex[assigned_i]
            assigned_i_text = assigned_i

        p_level = priority_map.get(j, 4)
        p_label = {1: "P1-Life Critical", 2: "P2-Security", 3: "P3-Social", 4: "P4-Normal"}.get(p_level, "P4")

        # Get visit order
        visit_info = ""
        visit_order_num = ""
        if assigned_i and len(tsp_results.get(assigned_i, ())) > 4:
            vo = tsp_results[assigned_i][4] if len(tsp_results[assigned_i]) > 4 else {}
            if j in vo:
                visit_info = f" | Visit Order: {vo[j]}"
                visit_order_num = str(vo[j])

        tooltip_text = f"Fault: {j} | {p_label} | Crew: {assigned_i_text}{visit_info}"

        if p_level <= 3:
            # Critical faults: pin-shaped marker with "P" label and priority glow
            border_col = priority_border[p_level]
            pin_size = 34
            html = (
                f'<div style="position:relative;width:{pin_size}px;height:{pin_size + 10}px;">'
                f'<div style="'
                f'width:{pin_size}px;height:{pin_size}px;'
                f'background:{color_hex_j};'
                f'border:3px solid {border_col};'
                f'border-radius:50% 50% 50% 0;'
                f'transform:rotate(-45deg);'
                f'display:flex;align-items:center;justify-content:center;'
                f'box-shadow:0 0 10px {border_col}, 0 0 20px {border_col};'
                f'">'
                f'<span style="transform:rotate(45deg);color:white;font-weight:bold;font-size:14px;">P</span>'
                f'</div>'
                f'</div>'
            )
            folium.Marker(
                location=(jlat, jlon),
                tooltip=tooltip_text,
                icon=folium.DivIcon(
                    html=html,
                    icon_size=(pin_size, pin_size + 10),
                    icon_anchor=(pin_size // 2, pin_size + 5)
                )
            ).add_to(m)
        else:
            # Normal faults (P4): standard icon
            folium.Marker(
                location=(jlat, jlon),
                tooltip=tooltip_text,
                icon=folium.Icon(color=color, icon="bolt", prefix="fa")
            ).add_to(m)

    # Add TSP routes to map (directional AntPath arrows)
    for team_name in crew_list:
        result = tsp_results[team_name]
        route = result[1]

        if len(route) <= 2:
            continue

        color_hex = crew_hex[team_name]

        # Build route coordinates
        route_coords = []
        for node in route:
            if node == team_name:
                route_coords.append(crew_data[team_name])
            elif node in substation_coords:
                route_coords.append(substation_coords[node])

        # Directional route with animated arrows (AntPath)
        AntPath(
            locations=route_coords,
            color=color_hex,
            weight=4,
            opacity=0.8,
            dash_array=[15, 30],
            delay=1500,
            tooltip=f"{team_name} route — {result[2]} km"
        ).add_to(m)

        # Add visit order number markers on each stop
        for idx, node in enumerate(route):
            if node == team_name:
                continue
            if node in substation_coords:
                coord = substation_coords[node]
                # Show visit order number on the route
                order_html = (
                    f'<div style="'
                    f'width:22px;height:22px;'
                    f'background:{color_hex};'
                    f'border:2px solid white;'
                    f'border-radius:50%;'
                    f'display:flex;align-items:center;justify-content:center;'
                    f'color:white;font-weight:bold;font-size:11px;'
                    f'box-shadow:0 1px 4px rgba(0,0,0,0.4);'
                    f'">{idx}</div>'
                )
                folium.Marker(
                    location=coord,
                    icon=folium.DivIcon(
                        html=order_html,
                        icon_size=(22, 22),
                        icon_anchor=(11, 11)
                    )
                ).add_to(m)

    st_folium(m, width=None, height=720)

    # Footer credit
    st.markdown(
        "<div style='text-align:right; color:gray; font-size:20px; padding-top:14px;'>"
        "Built by Mevlüt Gümüş</div>",
        unsafe_allow_html=True
    )
