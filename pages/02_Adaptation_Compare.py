import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import timedelta, datetime
from calc_traffic import get_daily_ESAL
from calc_modulus import generate_Mr, generate_flooded_Mr

# ─── PSI helper functions (identical to main) ───────────────────
def calculate_delta_psi(esal, Mr, SN):
    return (10**((np.log10(esal))-(0.45*(-1.645))-(9.36*np.log10(SN+1))
          +0.2+8.07-(2.32*np.log10(Mr)))*(0.4+(1094/((SN+1)**5.19)))) * (4.2-1.5)

def get_psi_gwt_flood(psi_i, psi_t, esals, design_years,
                      Mrs, Flooded_Mr, flooded_vals, SN):
    days = np.arange(design_years*365+1)
    psi  = np.full_like(days, psi_i, dtype=float)
    last = psi_i
    for year in range(design_years):
        start = year*365+1
        flood = int(flooded_vals[year]*0.25)
        normal=365-flood
        for d in range(normal):
            idx=start+d-1
            last-=calculate_delta_psi(esals[idx],Mrs[year],SN)
            psi[idx]=max(last,psi_t)
            if last<psi_t:
                psi[idx:]=psi_t;return psi
        for d in range(flood):
            idx=start+normal+d-1
            last-=calculate_delta_psi(esals[idx],Flooded_Mr[year],SN)
            psi[idx]=max(last,psi_t)
            if last<psi_t:
                psi[idx:]=psi_t;return psi
    psi[psi==psi_i]=last
    return psi

# ─── PAGE GUARD ─────────────────────────────────────────────────
st.set_page_config(page_title="Compare adaptation alternatives", page_icon="🔀")
if "scenario_inputs" not in st.session_state:
    st.error("Run the base page first.")
    st.stop()

scenarios = st.session_state["scenario_inputs"]
base = scenarios[0]

# one shared ESAL array
base_esals = get_daily_ESAL(base["aadt"], base["traffic_classes"],
                            base["growth_rate"], base["design_years"])

st.title("Compare adaptation alternatives")

MAX_SCEN = 5
for idx, sc in enumerate(scenarios):
    # defaults
    sc.setdefault("a1",0.42); sc.setdefault("a2",0.172); sc.setdefault("m2",0.8)
    sc.setdefault("gwt_depth", sc["surT"]+sc["baseT"]+40)
    sc.setdefault("init_psi",4.2); sc.setdefault("term_psi",1.0)
    
    if idx not in st.session_state:
        st.session_state[idx] = True
    with st.expander(sc["label"], expanded=st.session_state[idx]):
        sc["label"]=st.text_input("Label",sc["label"],key=f"lab{idx}")
        sc["surT"]=st.number_input("Surface (in)",3,26,sc["surT"],1,key=f"sur{idx}")
        sc["baseT"]=st.number_input("Base (in)",6,28,sc["baseT"],1,key=f"bas{idx}")
        sc["base_type"]=st.selectbox("Base type",('A-1-a','A-1-b','A-2-4'),
                                     ('A-1-a','A-1-b','A-2-4').index(sc["base_type"]),
                                     key=f"bt{idx}")
        sc["gwt_rise"]=st.slider("GWT rise /yr (in)",0.0,5.0,sc["gwt_rise"],.1,key=f"gw{idx}")
        sc["flooded_days"]=st.slider("Flood days /yr",0,150,sc["flooded_days"],1,key=f"fd{idx}")

        st.markdown("**Serviceability thresholds**")
        sc["term_psi"]=st.number_input("Terminal PSI",1.0,3.5,sc["term_psi"],.1,key=f"tp{idx}")
        sc["init_psi"]=st.number_input("Initial PSI",sc["term_psi"]+0.1,5.0,
                                       sc["init_psi"],.1,key=f"ip{idx}")

        if len(scenarios)<MAX_SCEN:
            if st.button("➕ Add scenario",key=f"add{idx}"):
                new=scenarios[0].copy(); new["label"]=f"Scenario {len(scenarios)}"
                scenarios.append(new); st.rerun()

st.divider()

if st.button("Compare scenarios",type="primary"):
    soil_map = {
        'A-1-a':1,
        'A-1-b':2,
        'A-2-4':3,
        'A-2-5':4,
        'A-2-6':5,
        'A-2-7':6,
        'A-3':7,
        'A-4':8,
        'A-5':9,
        'A-6':10,
        'A-7-5':11,
        'A-7-6':12
    }
    colours=["black"]+px.colors.qualitative.Set2
    line_fig=go.Figure(); eol_years=[]; eol_psis=[]; labels=[]
    for i,sc in enumerate(scenarios):
        yrs=sc["design_years"]; gwt=[sc["gwt_depth"]-sc["gwt_rise"]*y for y in range(yrs+1)]
        flooded=[sc["flooded_days"]]*(yrs+1)
        params=[{"Surface Thickness":sc["surT"],"Base Thickness":sc["baseT"],
                 "GWT":g,"Base Type":soil_map[sc["base_type"]],
                 "Subgrade Type":soil_map[sc["sg_type"]]} for g in gwt]
        Mrs=generate_Mr(gwt,sc["sg_type"]); F_Mr=generate_flooded_Mr(params)
        SN=sc["a1"]*sc["surT"]+sc["a2"]*sc["baseT"]*sc["m2"]

        psi_full=get_psi_gwt_flood(sc["init_psi"],sc["term_psi"],
                              base_esals,yrs, Mrs, F_Mr, flooded,SN)

        # ── metrics BEFORE trimming ──────────────────────────────
        fail_idx = np.where(psi_full <= sc["term_psi"] + 1e-6)[0]
        if fail_idx.size:
            eol_year = fail_idx[0] / 365          # first day ≤ terminal
            eol_psi  = sc["term_psi"]
            plot_len = fail_idx[0] + 1            # include that point
        else:
            eol_year = yrs                        # lasted full design life
            eol_psi  = psi_full[-1]
            plot_len = len(psi_full)

        eol_years.append(eol_year)
        eol_psis.append(eol_psi)
        labels.append(sc["label"])

        # ── curve for plotting (cut after end‑of‑life) ───────────
        psi_plot = psi_full[:plot_len]
        yrs_arr  = np.arange(plot_len) / 365

        line_fig.add_trace(go.Scatter(
            x=yrs_arr, y=psi_plot,
            name=sc["label"],
            line=dict(width=3, color=colours[i % len(colours)])
        ))
    # dotted threshold lines
    line_fig.add_shape(type="line",x0=0,x1=max(eol_years),
                       y0=base["init_psi"],y1=base["init_psi"],
                       line=dict(color="grey",dash="dash"))
    line_fig.add_shape(type="line",x0=0,x1=max(eol_years),
                       y0=base["term_psi"],y1=base["term_psi"],
                       line=dict(color="grey",dash="dash"))
    line_fig.update_layout(
        title=dict(text="PSI curves", font=dict(size=24)),
        xaxis_title=dict(text="Year", font=dict(size=22)),
        yaxis_title=dict(text="PSI",  font=dict(size=22)),
        xaxis=dict(tickfont=dict(size=20)),
        yaxis=dict(tickfont=dict(size=20), range=[0, 5]),
        legend=dict(
            orientation="h",
            y=-0.25, x=0.5, xanchor="center",
            font=dict(size=20)
        ),
        height=650,
        margin=dict(l=60, r=40, t=60, b=80)
    )
    st.plotly_chart(line_fig, use_container_width=True)
    # bar charts
    bar_col=colours[:len(labels)]
    fig1=go.Figure(go.Bar(x=labels,y=eol_years,marker_color=bar_col))
    fig1.update_layout(
        title=dict(text="End‑of‑life year", font=dict(size=24)),
        yaxis_title=dict(text="Year", font=dict(size=22)),
        xaxis_tickfont=dict(size=20),
        yaxis_tickfont=dict(size=20),
        height=450, margin=dict(l=60, r=40, t=60, b=60)
    )
    fig2=go.Figure(go.Bar(x=labels,y=eol_psis,marker_color=bar_col))
    fig2.update_layout(
    title=dict(text="PSI at design horizon", font=dict(size=24)),
        yaxis_title=dict(text="PSI", font=dict(size=22)),
        yaxis=dict(range=[0, 5], tickfont=dict(size=20)),
        xaxis_tickfont=dict(size=20),
        height=450, margin=dict(l=60, r=40, t=60, b=60)
    )
    st.plotly_chart(fig1,use_container_width=True)
    st.plotly_chart(fig2,use_container_width=True)

    total_days = base["design_years"] * 365
    year_num   = np.arange(total_days) // 365 + 1
    day_num    = np.arange(total_days) % 365  + 1
    date_col   = [f"Year {y}, Day {d}" for y, d in zip(year_num, day_num)]
    esal_col   = np.round(base_esals[:total_days])

    flood_status = []
    for yr in range(base["design_years"]):
        flood_len  = int(base["flooded_days"] * 0.25)
        normal_len = 365 - flood_len
        flood_status.extend(["Normal"] * normal_len + ["Flooding"] * flood_len)

    # start DataFrame with shared columns
    df = pd.DataFrame({
        "Date"    : date_col,
        "ESAL"    : esal_col,
        "Status"  : flood_status
    })

    # add two columns (Modulus, PSI) for each scenario
    for i, sc in enumerate(scenarios):
        yrs  = sc["design_years"]
        gwt  = [sc["gwt_depth"] - sc["gwt_rise"] * y for y in range(yrs+1)]
        floods = [sc["flooded_days"]] * (yrs+1)
        Mrs   = generate_Mr(gwt, sc["sg_type"])
        F_Mr  = generate_flooded_Mr([
            {"Surface Thickness": sc["surT"], "Base Thickness": sc["baseT"],
             "GWT": g, "Base Type": soil_map[sc["base_type"]],
             "Subgrade Type": soil_map[sc["sg_type"]]} for g in gwt
        ])

        mod_series = []
        for yr in range(yrs):
            flood_len  = int(floods[yr] * 0.25)
            normal_len = 365 - flood_len
            mod_series.extend([Mrs[yr]] * normal_len + [F_Mr[yr]] * flood_len)

        psi_series = get_psi_gwt_flood(
            sc["init_psi"], sc["term_psi"], base_esals, yrs,
            Mrs, F_Mr, floods, sc["a1"]*sc["surT"] + sc["a2"]*sc["baseT"]*sc["m2"]
        )[1:total_days+1]

        col_prefix = sc["label"].replace(",", "_")  # safe column name
        df[f"Subgrade resilient modulus in psi ({col_prefix})"] = mod_series[:total_days]
        df[f"PSI ({col_prefix})"]          = psi_series[:total_days]

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download daily data (CSV)",
        data=csv_bytes,
        file_name="daily_comparison_data.csv",
        mime="text/csv",
        type="primary"
    )

