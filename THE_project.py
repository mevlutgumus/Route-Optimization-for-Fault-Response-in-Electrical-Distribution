import json
import math
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium
import pulp as lp
from geopy.distance import geodesic

# =========================================================================
# SABİTLER VE AYARLAR
# =========================================================================
OPTIMAL_STATUS_CODE = 1

st.set_page_config(page_title="Arıza Optimizasyonu", layout="wide")
st.title("Arıza Müdahalesi İçin Atama ve Rota Optimizasyonu (GAP + PC-TSP)")

# =========================================================================
# EKİP KONUMLARI
# =========================================================================
ekip_verileri = {
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
ekip_listesi = list(ekip_verileri.keys())

# =========================================================================
# GEOJSON OKUMA
# =========================================================================
def load_trafos_from_geojson(path: str):
    with open(path, "r", encoding="utf-8") as f:
        geojson_data = json.load(f)

    tum_trafo_konumlari_standart = {}
    trafo_sayaci = 1

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

            tum_trafo_konumlari_standart[f"Trafo_{trafo_sayaci}"] = (lat, lon)
            trafo_sayaci += 1

    return tum_trafo_konumlari_standart

# =========================================================================
# KUŞ UÇUŞU (GEODESIC) MESAFE HESABI — EKİP→TRAFO
# =========================================================================
def compute_C_ij_geodesic(ekip_verileri, trafo_konumlari):
    C_ij = {}
    for i, (i_lat, i_lon) in ekip_verileri.items():
        C_ij[i] = {}
        ekip_loc = (i_lat, i_lon)
        for j, (j_lat, j_lon) in trafo_konumlari.items():
            trafo_loc = (j_lat, j_lon)
            mesafe = geodesic(ekip_loc, trafo_loc).km
            C_ij[i][j] = round(mesafe, 2)
    return C_ij

# =========================================================================
# KUŞ UÇUŞU (GEODESIC) MESAFE MATRİSİ — DÜĞÜMLER ARASI (TSP İÇİN)
# =========================================================================
def compute_distance_matrix(node_coords):
    """
    node_coords: dict { node_id: (lat, lon), ... }
    Tüm düğüm çiftleri arası geodesic mesafe matrisi döndürür.
    """
    nodes = list(node_coords.keys())
    d = {}
    for j in nodes:
        d[j] = {}
        for k in nodes:
            if j == k:
                d[j][k] = 0.0
            else:
                mesafe = geodesic(node_coords[j], node_coords[k]).km
                d[j][k] = round(mesafe, 2)
    return d

# =========================================================================
# GAP ÇÖZ (PULP)
# =========================================================================
def solve_gap(C_ij, ekip_listesi, trafo_listesi, cap_dict):
    prob = lp.LpProblem("GAP_Atama", lp.LpMinimize)
    X_ij = lp.LpVariable.dicts(
        "Atama",
        [(i, j) for i in ekip_listesi for j in trafo_listesi],
        cat=lp.LpBinary
    )

    # Amaç Fonksiyonu
    prob += lp.lpSum(C_ij[i][j] * X_ij[(i, j)] for i in ekip_listesi for j in trafo_listesi)

    # K1: Her arızaya tam olarak 1 ekip atanmalı
    for j in trafo_listesi:
        prob += lp.lpSum(X_ij[(i, j)] for i in ekip_listesi) == 1

    # K2: Ekip kapasiteleri aşılmamalı
    for i in ekip_listesi:
        prob += lp.lpSum(X_ij[(i, j)] for j in trafo_listesi) <= int(cap_dict[i])

    prob.solve(lp.PULP_CBC_CMD(msg=0))
    return prob, X_ij

# =========================================================================
# PC-TSP ÇÖZ (PULP) — HER EKİP İÇİN AYRI
# =========================================================================
def solve_tsp_for_team(team_name, depot_coord, assigned_faults, trafo_konumlari,
                       priority_map, priority_mode=True, time_limit=60):
    """
    team_name     : Ekip adı (str)
    depot_coord   : Ekibin konumu (lat, lon)
    assigned_faults: Bu ekibe atanan arıza listesi [str]
    trafo_konumlari: Tüm trafo koordinatları dict
    priority_map  : { trafo_id: priority_level (1-4) } — seçilen arızaların öncelik bilgisi
    priority_mode : True ise K4 kısıtları aktif (Acil Müdahale), False ise sadece mesafe (Standart)
    time_limit    : CBC solver zaman sınırı (saniye)

    Return: (status, route_order, route_distance, solve_time)
      route_order: sıralı düğüm listesi [depot, fault1, fault2, ..., depot]
      route_distance: toplam rota mesafesi (km)
    """
    import time

    N_i = assigned_faults  # Depot hariç arıza düğümleri
    n = len(N_i)

    # Tek veya sıfır arıza durumu
    if n == 0:
        return ("Optimal", [team_name], 0.0, 0.0)
    if n == 1:
        d_go = geodesic(depot_coord, trafo_konumlari[N_i[0]]).km
        d_back = d_go  # Gidiş-dönüş aynı
        return ("Optimal", [team_name, N_i[0], team_name], round(d_go + d_back, 2), 0.0)

    # Düğüm koordinatları (depot = "DEPOT")
    node_coords = {"DEPOT": depot_coord}
    for j in N_i:
        node_coords[j] = trafo_konumlari[j]

    all_nodes = list(node_coords.keys())  # DEPOT + arıza düğümleri

    # Mesafe matrisi
    d = compute_distance_matrix(node_coords)

    # ---- MODEL ----
    prob = lp.LpProblem(f"PCTSP_{team_name}", lp.LpMinimize)

    # Karar değişkenleri
    # y_jk: j'den k'ya gidilir mi? (binary)
    y = lp.LpVariable.dicts("y",
        [(j, k) for j in all_nodes for k in all_nodes if j != k],
        cat=lp.LpBinary)

    # u_j: j düğümünün ziyaret sırası (integer, sadece arıza düğümleri için)
    u = lp.LpVariable.dicts("u", N_i, lowBound=1, upBound=n, cat=lp.LpInteger)

    # Amaç Fonksiyonu: Toplam rota mesafesini minimize et
    prob += lp.lpSum(d[j][k] * y[(j, k)] for j in all_nodes for k in all_nodes if j != k)

    # K1: Her düğümden tam 1 kez çıkılmalı
    for j in all_nodes:
        prob += lp.lpSum(y[(j, k)] for k in all_nodes if k != j) == 1

    # K2: Her düğüme tam 1 kez girilmeli
    for k in all_nodes:
        prob += lp.lpSum(y[(j, k)] for j in all_nodes if j != k) == 1

    # K3: MTZ Alt-tur Eliminasyonu (DEPOT hariç tüm düğüm çiftleri için)
    for j in N_i:
        for k in N_i:
            if j != k:
                prob += u[j] - u[k] + n * y[(j, k)] <= n - 1

    # K4: Öncelik Sıralama Kısıtları (sadece priority_mode aktifse)
    if priority_mode:
        # Öncelik kümelerini oluştur
        P_sets = {1: [], 2: [], 3: [], 4: []}
        for j in N_i:
            p_level = priority_map.get(j, 4)  # Varsayılan P4 (normal)
            P_sets[p_level].append(j)

        # Her r < s olan boş olmayan küme çifti için kısıt ekle
        priority_levels = sorted(P_sets.keys())
        for idx_r, r in enumerate(priority_levels):
            for s in priority_levels[idx_r + 1:]:
                if len(P_sets[r]) > 0 and len(P_sets[s]) > 0:
                    for j in P_sets[r]:
                        for k in P_sets[s]:
                            # u_j + 1 <= u_k  (strict ordering for integer variables)
                            prob += u[j] + 1 <= u[k]

    # Çöz
    start_time = time.time()
    prob.solve(lp.PULP_CBC_CMD(msg=0, timeLimit=time_limit))
    solve_time = round(time.time() - start_time, 3)

    status_text = lp.LpStatus[prob.status]

    if prob.status != OPTIMAL_STATUS_CODE:
        return (status_text, [], 0.0, solve_time)

    # Rotayı oluştur (DEPOT'tan başlayarak sıralı ziyaret)
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
            # Son düğümden DEPOT'a dönüş
            route.append("DEPOT")
            break

    if route[-1] != "DEPOT":
        route.append("DEPOT")

    # Rota mesafesi
    total_dist = 0.0
    for idx in range(len(route) - 1):
        total_dist += d[route[idx]][route[idx + 1]]

    # DEPOT yerine ekip adı koy
    display_route = [team_name if x == "DEPOT" else x for x in route]

    # Ziyaret sırasını al (debug/tablo için)
    visit_order = {}
    for j in N_i:
        val = lp.value(u[j])
        if val is not None:
            visit_order[j] = int(val)

    return (status_text, display_route, round(total_dist, 2), solve_time, visit_order)


# =========================================================================
# SOL PANEL (GİRDİLER)
# =========================================================================
with st.sidebar:
    st.header("Girdiler")

    geojson_path = st.text_input("GeoJSON dosya adı/yolu", value="export.geojson")

    try:
        tum_trafo_konumlari_standart = load_trafos_from_geojson(geojson_path)
        st.success(f"Trafo bulundu: {len(tum_trafo_konumlari_standart)}")
        geo_ok = True
    except:
        st.error("GeoJSON okunamadı veya bulunamadı.")
        geo_ok = False
        tum_trafo_konumlari_standart = {}

    st.subheader("Arızalı Trafolar")
    trafo_options = list(tum_trafo_konumlari_standart.keys())
    selected_faults = st.multiselect(
        "Arızalı trafoları seç",
        options=trafo_options,
        default=trafo_options[:10] if len(trafo_options) >= 10 else trafo_options
    )

    # ---- EKİP KAPASİTELERİ ----
    st.subheader("Ekip Kapasiteleri")
    cap_mode = st.selectbox(
        "Kapasite tipi",
        ["Tek sayı (hepsine aynı)", "Ekip bazlı (tek tek)", "Optimal (Sistem Önerisi)"]
    )

    cap_dict = {}
    if cap_mode == "Optimal (Sistem Önerisi)":
        if len(ekip_listesi) > 0:
            calc_cap = math.ceil(len(selected_faults) / len(ekip_listesi)) + 1
        else:
            calc_cap = len(selected_faults)
        cap_dict = {i: int(calc_cap) for i in ekip_listesi}
        st.info(f"Sistem, iş yükünü dengelemek için her ekibe **{calc_cap}** kapasite atadı.")
    elif cap_mode == "Tek sayı (hepsine aynı)":
        default_cap = max(1, math.ceil(max(len(selected_faults), 1) / max(len(ekip_listesi), 1)) + 1)
        max_cap = st.number_input("Kapasite (max_cap)", min_value=1, value=int(default_cap), step=1)
        cap_dict = {i: int(max_cap) for i in ekip_listesi}
    else:
        for i in ekip_listesi:
            cap_dict[i] = int(st.number_input(f"{i} kapasite", min_value=0, value=3, step=1))

    # ---- ÖNCELİK AYARLARI ----
    st.subheader("Öncelik Ayarları (TSP)")

    rota_mode = st.radio(
        "Rotalama Modu",
        ["Standart (Sadece Mesafe)", "Acil Müdahale (Öncelik Kısıtlı)"],
        index=1,
        help="Standart: Toplam mesafeyi minimize eder. Acil Müdahale: P1→P2→P3→P4 sırasını garanti eder."
    )
    priority_mode = (rota_mode == "Acil Müdahale (Öncelik Kısıtlı)")

    priority_map = {}
    if priority_mode and len(selected_faults) > 0:
        st.markdown("**Kritik noktaları seçin** (seçilmeyenler otomatik P4 olur)")

        p1_faults = st.multiselect(
            "P1 — Hayati Kritik (Hastane, Diyaliz)",
            options=selected_faults,
            default=[],
            key="p1"
        )
        remaining_after_p1 = [f for f in selected_faults if f not in p1_faults]

        p2_faults = st.multiselect(
            "P2 — Güvenlik/Kamu Düzeni (Askeri, Karakol)",
            options=remaining_after_p1,
            default=[],
            key="p2"
        )
        remaining_after_p2 = [f for f in remaining_after_p1 if f not in p2_faults]

        p3_faults = st.multiselect(
            "P3 — Sosyal Hassas (Okul, Huzurevi)",
            options=remaining_after_p2,
            default=[],
            key="p3"
        )

        for f in p1_faults:
            priority_map[f] = 1
        for f in p2_faults:
            priority_map[f] = 2
        for f in p3_faults:
            priority_map[f] = 3
        # Geri kalanlar P4
        for f in selected_faults:
            if f not in priority_map:
                priority_map[f] = 4

        # Özet
        st.markdown(f"P1: **{len(p1_faults)}**, P2: **{len(p2_faults)}**, "
                    f"P3: **{len(p3_faults)}**, P4: **{len(selected_faults) - len(p1_faults) - len(p2_faults) - len(p3_faults)}**")
    else:
        for f in selected_faults:
            priority_map[f] = 4

    # ---- TSP ZAMAN SINIRI ----
    tsp_time_limit = st.number_input("TSP Solver Zaman Sınırı (sn)", min_value=5, value=60, step=5)

    run_btn = st.button("ÇÖZ / HARİTAYI GÜNCELLE", type="primary")

if not geo_ok:
    st.stop()

# Simülasyon verilerini hazırla
girilen_arizalar = [a for a in selected_faults if a in tum_trafo_konumlari_standart]
trafo_konumlari = {j: tum_trafo_konumlari_standart[j] for j in girilen_arizalar}
trafo_listesi = list(trafo_konumlari.keys())

# =========================================================================
# ANA EKRAN DÜZENİ
# =========================================================================
col_map, col_right = st.columns([1.2, 0.8], gap="large")

if run_btn or ("last_solution" not in st.session_state):
    if len(trafo_listesi) == 0:
        st.warning("Arızalı trafo seçmedin.")
        st.stop()

    # ---- AŞAMA 1: GAP ----
    with st.spinner("Aşama 1: GAP — Kuş uçuşu mesafeler hesaplanıyor ve atama yapılıyor..."):
        C_ij = compute_C_ij_geodesic(ekip_verileri, trafo_konumlari)
        prob_gap, X_ij = solve_gap(C_ij, ekip_listesi, trafo_listesi, cap_dict)

    if prob_gap.status != OPTIMAL_STATUS_CODE:
        st.error("GAP optimal çözüm bulunamadı. Kapasiteleri artırmayı deneyin.")
        st.stop()

    gap_objective = float(lp.value(prob_gap.objective))

    # GAP sonuçlarından atama matrisini çıkar
    Xvals = {(i, j): float(lp.value(X_ij[(i, j)])) for i in ekip_listesi for j in trafo_listesi}

    # Her ekibe atanan arızaları belirle
    team_assignments = {}
    for i in ekip_listesi:
        team_assignments[i] = [j for j in trafo_listesi if Xvals.get((i, j), 0) > 0.5]

    # ---- AŞAMA 2: PC-TSP (Her ekip için ayrı) ----
    tsp_results = {}
    with st.spinner("Aşama 2: PC-TSP — Her ekip için rota optimizasyonu çözülüyor..."):
        for team_name in ekip_listesi:
            faults = team_assignments[team_name]
            if len(faults) == 0:
                tsp_results[team_name] = ("Optimal", [team_name], 0.0, 0.0, {})
                continue

            result = solve_tsp_for_team(
                team_name=team_name,
                depot_coord=ekip_verileri[team_name],
                assigned_faults=faults,
                trafo_konumlari=trafo_konumlari,
                priority_map=priority_map,
                priority_mode=priority_mode,
                time_limit=tsp_time_limit
            )
            tsp_results[team_name] = result

    # Session state'e kaydet
    st.session_state.last_solution = {
        "C_ij": C_ij,
        "gap_status": lp.LpStatus[prob_gap.status],
        "gap_objective": gap_objective,
        "Xvals": Xvals,
        "trafo_konumlari": trafo_konumlari,
        "cap_dict": cap_dict,
        "team_assignments": team_assignments,
        "tsp_results": tsp_results,
        "priority_map": priority_map,
        "priority_mode": priority_mode,
    }

# Sonuçları State'den çek
sol = st.session_state.last_solution
C_ij = sol["C_ij"]
Xvals = sol["Xvals"]
trafo_konumlari = sol["trafo_konumlari"]
cap_dict = sol["cap_dict"]
team_assignments = sol["team_assignments"]
tsp_results = sol["tsp_results"]
priority_map = sol["priority_map"]
is_priority = sol["priority_mode"]

# =========================================================================
# SAĞ: SONUÇ TABLOLARI
# =========================================================================
with col_right:
    st.subheader("Özet Sonuçlar")
    st.write(f"GAP Durumu: **{sol['gap_status']}**")
    st.success(f"GAP Toplam Atama Mesafesi: **{sol['gap_objective']:.2f} km**")

    # Toplam TSP mesafesi
    toplam_tsp = sum(r[2] for r in tsp_results.values())
    st.success(f"TSP Toplam Rota Mesafesi: **{toplam_tsp:.2f} km**")

    mode_text = "Acil Müdahale (P1→P2→P3→P4)" if is_priority else "Standart (Sadece Mesafe)"
    st.info(f"Rotalama Modu: **{mode_text}**")

    # ---- ATAMA LİSTESİ ----
    st.subheader("Atama Listesi (GAP)")
    rows = []
    for j in trafo_listesi:
        assigned_i = None
        for i in ekip_listesi:
            if Xvals.get((i, j), 0) > 0.5:
                assigned_i = i
                break
        if assigned_i:
            p_level = priority_map.get(j, 4)
            p_label = {1: "P1-Hayati", 2: "P2-Güvenlik", 3: "P3-Sosyal", 4: "P4-Normal"}.get(p_level, "P4")
            rows.append({
                "Arıza (Trafo)": j,
                "Atanan Ekip": assigned_i,
                "Öncelik": p_label,
                "Mesafe (km)": C_ij[assigned_i][j]
            })

    df_assign = pd.DataFrame(rows).sort_values(["Atanan Ekip", "Öncelik", "Arıza (Trafo)"], ignore_index=True)
    st.dataframe(df_assign, use_container_width=True, height=300)

    # ---- EKİP BAZLI ROTA SONUÇLARI ----
    st.subheader("Ekip Bazlı Rota Sonuçları (TSP)")
    rota_rows = []
    for team_name in ekip_listesi:
        result = tsp_results[team_name]
        status = result[0]
        route = result[1]
        dist = result[2]
        solve_time = result[3]
        fault_count = len(team_assignments[team_name])

        if fault_count == 0:
            continue

        # Rotayı string olarak göster (sadece arıza düğümleri, depot olmadan)
        route_nodes = [x for x in route if x != team_name]
        route_str = " → ".join(route_nodes) if route_nodes else "-"

        rota_rows.append({
            "Ekip": team_name,
            "Arıza Sayısı": fault_count,
            "Kapasite": cap_dict[team_name],
            "Rota Mesafesi (km)": dist,
            "Çözüm Süresi (sn)": solve_time,
            "Durum": status,
        })

    if rota_rows:
        df_rota = pd.DataFrame(rota_rows)
        st.dataframe(df_rota, use_container_width=True, height=280)

    # ---- EKİP DETAY: ROTA SIRASI ----
    st.subheader("Ekip Rota Detayları")
    for team_name in ekip_listesi:
        result = tsp_results[team_name]
        route = result[1]
        fault_count = len(team_assignments[team_name])

        if fault_count == 0:
            continue

        with st.expander(f"{team_name} — {fault_count} arıza, {result[2]} km"):
            # Rota sırası
            st.write("**Rota:** " + " → ".join(route))

            # Ziyaret sırası tablosu
            if len(result) > 4 and result[4]:
                visit_order = result[4]
                order_rows = []
                for node, order in sorted(visit_order.items(), key=lambda x: x[1]):
                    p_level = priority_map.get(node, 4)
                    p_label = {1: "P1", 2: "P2", 3: "P3", 4: "P4"}.get(p_level, "P4")
                    order_rows.append({
                        "Sıra (u_j)": order,
                        "Düğüm": node,
                        "Öncelik": p_label,
                    })
                st.dataframe(pd.DataFrame(order_rows), use_container_width=True, hide_index=True)

# =========================================================================
# SOL: HARİTA
# =========================================================================
with col_map:
    st.subheader("Harita — Rota Görselleştirme")

    lat_mean = sum(v[0] for v in ekip_verileri.values()) / len(ekip_verileri)
    lon_mean = sum(v[1] for v in ekip_verileri.values()) / len(ekip_verileri)

    m = folium.Map(location=(lat_mean, lon_mean), zoom_start=11, control_scale=True)

    palette = [
        "red", "blue", "green", "purple", "orange",
        "darkred", "cadetblue", "darkgreen", "darkblue", "pink"
    ]
    hex_palette = [
        "#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#e67e22",
        "#c0392b", "#5dade2", "#1abc9c", "#2c3e50", "#ff69b4"
    ]
    ekip_color = {ekip_listesi[idx]: palette[idx % len(palette)] for idx in range(len(ekip_listesi))}
    ekip_hex = {ekip_listesi[idx]: hex_palette[idx % len(hex_palette)] for idx in range(len(ekip_listesi))}

    # Öncelik ikonları
    priority_icons = {1: "hospital-o", 2: "shield", 3: "graduation-cap", 4: "bolt"}

    # Ekipleri Haritaya Ekle
    for i, (ilat, ilon) in ekip_verileri.items():
        fault_count = len(team_assignments.get(i, []))
        folium.Marker(
            location=(ilat, ilon),
            tooltip=f"Ekip: {i} | Kapasite: {cap_dict.get(i, 0)} | Atanan: {fault_count}",
            icon=folium.Icon(color=ekip_color[i], icon="users", prefix="fa")
        ).add_to(m)

    # Trafoları Haritaya Ekle
    for j, (jlat, jlon) in trafo_konumlari.items():
        assigned_i = None
        for i in ekip_listesi:
            if Xvals.get((i, j), 0) > 0.5:
                assigned_i = i
                break

        if assigned_i is None:
            color = "gray"
            assigned_i_text = "Atanmadı"
        else:
            color = ekip_color[assigned_i]
            assigned_i_text = assigned_i

        p_level = priority_map.get(j, 4)
        p_label = {1: "P1-Hayati", 2: "P2-Güvenlik", 3: "P3-Sosyal", 4: "P4-Normal"}.get(p_level, "P4")
        icon_name = priority_icons.get(p_level, "bolt")

        # Ziyaret sırasını bul
        visit_info = ""
        if assigned_i and len(tsp_results.get(assigned_i, ())) > 4:
            vo = tsp_results[assigned_i][4] if len(tsp_results[assigned_i]) > 4 else {}
            if j in vo:
                visit_info = f" | Ziyaret Sırası: {vo[j]}"

        folium.Marker(
            location=(jlat, jlon),
            tooltip=f"Arıza: {j} | {p_label} | Ekip: {assigned_i_text}{visit_info}",
            icon=folium.Icon(color=color, icon=icon_name, prefix="fa")
        ).add_to(m)

    # TSP Rotalarını Haritaya Ekle (sıralı çizgiler)
    for team_name in ekip_listesi:
        result = tsp_results[team_name]
        route = result[1]

        if len(route) <= 2:
            continue

        color_hex = ekip_hex[team_name]

        # Rota koordinatlarını oluştur
        route_coords = []
        for node in route:
            if node == team_name:
                route_coords.append(ekip_verileri[team_name])
            elif node in trafo_konumlari:
                route_coords.append(trafo_konumlari[node])

        # Sıralı rota çizgisi (düz çizgi, numaralı)
        folium.PolyLine(
            locations=route_coords,
            color=color_hex,
            weight=3,
            opacity=0.85,
            tooltip=f"{team_name} rotası — {result[2]} km"
        ).add_to(m)

        # Rota üzerindeki düğümlere sıra numarası ekle (küçük daireler)
        for idx, node in enumerate(route):
            if node == team_name:
                continue
            if node in trafo_konumlari:
                coord = trafo_konumlari[node]
                folium.CircleMarker(
                    location=coord,
                    radius=10,
                    color=color_hex,
                    fill=True,
                    fill_color=color_hex,
                    fill_opacity=0.9,
                    tooltip=f"Sıra: {idx}"
                ).add_to(m)

    st_folium(m, width=None, height=720)
