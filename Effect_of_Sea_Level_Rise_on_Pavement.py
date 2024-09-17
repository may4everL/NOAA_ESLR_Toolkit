import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pickle
import math
import plotly.graph_objects as go
from PIL import Image

st.set_page_config(
    page_title='Effect of Sea Level Rise on Pavement Performance',
    page_icon=":arrow_right:"
)

# Font

st.markdown("""
<style>
    /* Attempt to change font sizes inside widgets globally */
    div[data-testid="stMarkdownContainer"] p {
        font-size: 18px;
    }
    div[data-testid="stTickBarMax"] p {
        font-size: 18px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
    <style>
    @media (prefers-color-scheme: dark) {
        img {
            background-color: #ffffff;  /* White background for images in dark mode */
            border: 1px solid #ffffff;  /* Optional: adds a border around the image */
            padding: 10px;  /* Optional: adds space between the border and the image */
        }
    }
    </style>
    """, unsafe_allow_html=True)

# image = Image.open('Figure 3.jpg')
# st.image(image, caption = 'Schematics for the pavement modelling')

#Converter
inch2meter = 0.0254
meter2inch = 39.3701

# Page Title
# st.title('Prediction of Saturation Changes during Inundation')
# st.set_option('deprecation.showPyplotGlobalUse', False)

## Loading ML Models
# def load_models():
#     edge_sat_sur_model = pickle.load(open(f'./Models/edge_sat_sur_model.obj', 'rb'))
#     edge_sat_base_model = pickle.load(open(f'./Models/edge_sat_base_model.obj', 'rb'))
#     edge_sat_sga_model = pickle.load(open(f'./Models/edge_sat_sga_model.obj', 'rb'))
#     edge_pt_model = pickle.load(open(f'./Models/edge_peak_time_model.obj', 'rb'))
#     edge_rt_model = pickle.load(open(f'./Models/edge_rest_time_model.obj', 'rb'))
#     edge_vadose_model = pickle.load(open(f'./Models/edge_sat_vadose_model.obj', 'rb'))
#     wp_sat_sur_model = pickle.load(open(f'./Models/wp_sat_sur_model.obj', 'rb'))
#     wp_sat_base_model = pickle.load(open(f'./Models/wp_sat_base_model.obj', 'rb'))
#     wp_sat_sga_model = pickle.load(open(f'./Models/wp_sat_sga_model.obj', 'rb'))
#     wp_pt_model = pickle.load(open(f'./Models/wp_peak_time_model.obj', 'rb'))
#     wp_rt_model = pickle.load(open(f'./Models/wp_rest_time_model.obj', 'rb'))
#     wp_vadose_model = pickle.load(open(f'./Models/wp_sat_vadose_model.obj', 'rb'))
#     return edge_sat_sur_model, edge_sat_base_model, edge_sat_sga_model, edge_pt_model, edge_rt_model,edge_vadose_model,\
#            wp_sat_sur_model, wp_sat_base_model, wp_sat_sga_model, wp_pt_model, wp_rt_model, wp_vadose_model

# edge_sat_sur_model, edge_sat_base_model, edge_sat_sga_model, edge_pt_model, edge_rt_model, edge_vadose_model,\
#            wp_sat_sur_model, wp_sat_base_model, wp_sat_sga_model, wp_pt_model, wp_rt_model, wp_vadose_model = load_models()

## Deal with input
st.sidebar.markdown("## Input Parameters")  # Main sidebar header with increased size
st.sidebar.markdown("### Pavement Info")  # Subheader for pavement information
with st.sidebar.expander('Design Period'):
    design_years = st.slider('Designed service year:', min_value=5, max_value=30, step=1, value=20)

with st.sidebar.expander("Layer Thickness"):
    surT = st.slider('Thickness of the AC layer (inch)', min_value=3, max_value=26, step=1, value=6)
    baseT = st.slider('Thickness of the base layer (inch)', min_value=6, max_value=28, step=1, value=8)

with st.sidebar.expander("Material Properties"):
    sg_type = st.selectbox("Subgrade type", ('A-1-a','A-1-b', 'A-2-4', 'A-2-5', 'A-2-6', 'A-2-7', 'A-4', 'A-5', 'A-6', 'A-7-5', 'A-7-6'), help='This will affect the hydrodynamic parameters and resilient modulus used for calculation.')
    st.markdown(
    "[AASHTO soil classification](https://transportation.org/technical-training-solutions/wp-content/uploads/sites/64/2023/02/AT-TC3CN025-18-T1-JA021.pdf)",
    unsafe_allow_html=True
)

st.sidebar.markdown("### Groundwater Info")  # Subheader for groundwater information
with st.sidebar.expander("Groundwater Level"):
    gwt = st.slider("Groundwater table: depth from the surface (inch)", min_value=surT+baseT, max_value=160, value=surT+baseT+40)
    gwt = gwt - surT - baseT  # Adjusting groundwater table depth
    st.text(f'{gwt}in from top of subgrade')

st.sidebar.markdown("### Sea Level Rise Impact")  # Subheader for sea level rise
with st.sidebar.expander("Impact Details"):
    gwt_rise = st.slider("Groundwater rise per Year (inch)", min_value=0.0, max_value=5.0, step=0.1, value=1.0)
    flooded_days = st.slider("Flooded days per Year", min_value=0, max_value=150, step=1, value=40)

st.sidebar.markdown("### Traffic Information")  # Subheader for traffic information
with st.sidebar.expander("Traffic"):
    aadt = st.number_input("AADT (Annual Average Daily Traffic)", min_value=2000, max_value=100000, value=5000)

st.sidebar.markdown("### Advanced Settings")  # Subheader for additional settings
with st.sidebar.expander("Layer Coefficients"):
    a1 = st.number_input("a1", min_value=0.0, max_value=1.0, value=0.42)
    a2 = st.number_input("a2", min_value=0.0, max_value=1.0, value=0.172)
    m2 = st.number_input("m2", min_value=0.0, max_value=1.5, value=0.8)

uncertainty = st.sidebar.checkbox('Include Uncertainty', help='This will include uncertainty using Monte Carlo Simulation')
if uncertainty:
    # gwt_rise_std = st.slider("Standard Deviation of Groundwater rise per Year (inch))", min_value=0.0, max_value=2.0, step=0.1, value=0.2)
    flooded_days_std = st.sidebar.slider("Standard Deviation of flooded days per Year (inch))", min_value=0, max_value=80, step=1, value=20)

prediction_state = st.sidebar.button('Show Predictions')
# Parameters
gwt_rise_std = 0.0
soil_params = {}

soil_params['A-1-b'] = {
    'theta_r': 0.045,
    'theta_s': 0.43,
    'a': 0.6665,
    'n': 2.68,
    'm': 0.6268
}

soil_params['A-2-4'] = {
    'theta_r': 0.025,
    'theta_s': 0.403,
    'a': 0.0383,
    'n': 1.3774,
    'm': 0.2740
}

soil_params['A-4'] = {
    'theta_r': 0.01,
    'theta_s': 0.439,
    'a': 0.0314,
    'n': 1.1804,
    'm': 0.1528
}

soil_params['A-5'] = {
    'theta_r': 0.01,
    'theta_s': 0.439,
    'a': 0.0314,
    'n': 1.1804,
    'm': 0.1528
}

soil_params['A-6'] = {
    'theta_r': 0.01,
    'theta_s': 0.614,
    'a': 0.0265,
    'n': 1.1033,
    'm': 0.0936
}

soil_params['A-7'] = {
    'theta_r': 0.01,
    'theta_s': 0.520,
    'a': 0.0367,
    'n': 1.1012,
    'm': 0.0919
}

# Traffic calculation
ADT = aadt
T = 0.18
TF = 0.52
G = 0.04
years = design_years
D = 0.5
L = 0.79

def get_ESAL(ADT, T=0.18, TF=0.52, G=0.04, design_years = 20, D=0.5, L=0.79):
    w = []
    for increment in np.arange(design_years):
        w.append(ADT*T*TF*D*L*365*(((1+G)**max(0.01, increment))-1)/G)
    return np.array(w)

def get_daily_ESAL(ADT, T=0.18, TF=0.52, G=0.04, design_years=21, D=0.5, L=0.79):
    yearly_values = [ADT * T * TF * D * L * 365 * (((1 + G)**max(0.01, year)) - 1) / G for year in range(design_years)]
    daily_values = np.zeros(design_years * 365)  # Array to hold daily ESAL values
    # Fill daily values by interpolating yearly increases
    start_index = 0
    for year in range(design_years):
        end_index = start_index + 365
        if year == 0:
            daily_values[start_index:end_index] = yearly_values[year] / 365
        else:
            # Calculate daily increase from the last year's total to this year's total
            daily_increase = (yearly_values[year] - yearly_values[year - 1]) / 365
            daily_values[start_index:end_index] = daily_values[start_index - 1] + daily_increase
        start_index = end_index

    return daily_values

esals = get_daily_ESAL(ADT)

#Constants
T1 = surT
# a1 = 0.42
# ## Base Layer
T2 = baseT
# a2 = 0.172
# m2 = 0.8
## Subgrade
a = -0.5934
b = 0.4
km = 6.1324
e = 0.516
Gs = 2.723

#W = 0.171
WOpt = 0.165
SrOpt = Gs*WOpt/e

Mr_initial_A1b = 56000 # psi
Mr_initial_A24 = 42000 # psi
Mr_initial_A4 = 25000 # psi
Mr_initial_A6 = 18000 # psi
Mr_initial_A7 = 14000 # psi
GWT_initial = gwt # inch
if not uncertainty:
    gwt_rise_std = 0.0
def get_gwts(g_initial = GWT_initial,g_rise=gwt_rise,gwt_std=0.0, years=design_years):
    gwts = [g_initial + g_rise * i for i in range(design_years+1)]
    if gwt_std > 0.0:
        gwts = [np.random.normal(mean, gwt_std + 0.1*year*gwt_std) for year, mean in enumerate(gwts)]
    return gwts

def generate_Mr(gwt_vals, Mr_initial=Mr_initial_A24, soil_type='A-2-4'):
    a, b, c = 0.000133, -0.0123, 0.928
    # if soil_type == 'A-1-b':
    #     a, b, c = -1e-4, 1.95e-2, 5.592e-1
    # if soil_type == 'A-1-b':
    #     a, b, c = -1e-4, 2.01e-2, 5.598e-1
    # if soil_type == 'A-2-4':
    #     a, b, c = -1e-4, 2.16e-2, 5.601e-1
    # if soil_type == 'A-2-5':
    #     a, b, c = -1e-4, 2.06e-2, 5.610e-1
    # if soil_type == 'A-2-6':
    #     a, b, c = -7.33e-5, 1.81e-2, 5.470e-1
    # if soil_type == 'A-2-7':
    #     a, b, c = 2.06e-5, 9.71e-3, 5.540e-1
    # if soil_type == 'A-3':
    #     a, b, c = -1.84e-7, 7.03e-4, 6.140e-1
    # if soil_type == 'A-4':
    #     a, b, c = -8.05e-5, 2.23e-2, 4.11e-1
    # if soil_type == 'A-5':
    #     a, b, c = -7.00e-5, 2.17e-2, 4.214e-1
    # if soil_type == 'A-6':
    #     a, b, c = 4.00e-5, 9.40e-3, 3.614e-1
    # if soil_type == 'A-7-5':
    #     a, b, c = 7.36e-6, 7.57e-3, 2.550e-1
    # if soil_type == 'A-7-6':
    #     a, b, c = -2.88e-6, 5.95e-3, 2.110e-1
    Mrs = [Mr_initial * (a*(gwt)**2 - b*(gwt) + c) for gwt in gwt_vals]
    return Mrs

# Mrs_A1b = [Mr_initial_A1b * (0.000133*(GWT_initial+i)**2 - 0.0123*(GWT_initial+i) + 0.928) for i in np.arange(20)]
# Mrs_A24 = [Mr_initial_A24 * (0.000133*(GWT_initial+i)**2 - 0.0123*(GWT_initial+i) + 0.928) for i in np.arange(20)]
# Mrs_A4 = [Mr_initial_A4 * (0.000133*(GWT_initial+i)**2 - 0.0123*(GWT_initial+i) + 0.928) for i in np.arange(20)]
# Mrs_A6 = [Mr_initial_A6 * (0.000133*(GWT_initial+i)**2 - 0.0123*(GWT_initial+i) + 0.928) for i in np.arange(20)]
# Mrs_A7 = [Mr_initial_A7 * (0.000133*(GWT_initial+i)**2 - 0.0123*(GWT_initial+i) + 0.928) for i in np.arange(20)]

# flood information
if not uncertainty:
    flooded_days_std = 0.0      
def get_flooded_days(flood_initial=flooded_days, flood_rise_rate=0.1, flooded_days_std=flooded_days_std):
    floods = [int(flood_initial + flood_rise_rate * i) for i in range(design_years+1)]
    if flooded_days_std > 0.0:
        floods = [int(np.random.normal(mean, flooded_days_std + 0.1*year*flooded_days_std)) for year, mean in enumerate(floods)]
    
    return floods

yearly_flood_days = flooded_days
Flooded_Mr_A1b = [Mr_initial_A1b*(0.4-0.02*(i//5)) for i in range(design_years)]
Flooded_Mr_A24 = [Mr_initial_A24*(0.4-0.02*(i//5)) for i in range(design_years)]
Flooded_Mr_A4 = [Mr_initial_A4*(0.4-0.02*(i//5)) for i in range(design_years)]
Flooded_Mr_A6 = [Mr_initial_A6*(0.4-0.02*(i//5)) for i in range(design_years)]
Flooded_Mr_A7 = [Mr_initial_A7*(0.4-0.02*(i//5)) for i in range(design_years)]

gwt_vals = get_gwts(GWT_initial, gwt_rise, gwt_rise_std, design_years)
if sg_type in ['A-1-a','A-1-b']:
    Mr_initial = Mr_initial_A1b
    Mrs = generate_Mr(gwt_vals, Mr_initial_A1b, soil_type=sg_type)
    Flooded_Mr = Flooded_Mr_A1b
elif sg_type in ['A-2-4','A-2-5','A-2-6','A-2-7','A-3']:
    Mr_initial = Mr_initial_A24
    Mrs = generate_Mr(gwt_vals, Mr_initial_A24, soil_type=sg_type)
    Flooded_Mr = Flooded_Mr_A24
elif sg_type in ['A-4','A-5']:
    Mr_initial = Mr_initial_A4
    Mrs = generate_Mr(gwt_vals, Mr_initial_A4, soil_type=sg_type)
    Flooded_Mr = Flooded_Mr_A4
elif sg_type == 'A-6':
    Mr_initial = Mr_initial_A6
    Mrs = generate_Mr(gwt_vals, Mr_initial_A6, soil_type=sg_type)
    Flooded_Mr = Flooded_Mr_A6
elif sg_type in ['A-7-5', 'A-7-6']:
    Mr_initial = Mr_initial_A7
    Mrs = generate_Mr(gwt_vals, Mr_initial_A7, soil_type=sg_type)
    Flooded_Mr = Flooded_Mr_A7

# print(Mrs)

flooded_vals = get_flooded_days(flooded_days, 0.2)
# print(flooded_vals)
def calc_SN(surT=surT, baseT=baseT, a1=a1, a2=a2, m2=m2):
    return a1 * surT + a2 * baseT * m2
SN = calc_SN()
def calculate_delta_psi(esal=5000, SN=5.0):
    return (10**((np.log10(esal))-(0.45*(-1.645))-(9.36*np.log10(SN+1))\
                +0.2+8.07-(2.32*np.log10(15000)))*(0.4+(1094/((SN+1)**5.19))))*(4.2-1.5)

def get_psi_gwt_flood(psi_i=4.2, psi_t=1.0, design_years=20, Mrs=Mrs, Flooded_Mr=Flooded_Mr_A24, flooded_vals=flooded_vals):
    days = np.arange(design_years * 365+1)
    esals = get_daily_ESAL(ADT)
    psi = np.full_like(days, 0, dtype=float)
    psi[0] = psi_i
    last_psi = psi_i
    for year in range(design_years):
        start_day = year * 365 + 1
        yearly_flood_days = flooded_vals[year]
        normal_days = 365 - int(yearly_flood_days * 0.25) # coefficient as placeholder
        flood_effect_days = 365 - int(yearly_flood_days * 0.25)
        flooded_days = int(yearly_flood_days * 0.25)
        end_normal_days = start_day + normal_days
        end_flood_effect_days = start_day + flood_effect_days

        # Calculate daily PSI drop for normal days
        daily_psi_drop = np.zeros(normal_days)
        for day in range(normal_days):
            # Calculate delta_psi assuming some MR calculation here
            Mr_cur = Mrs[(start_day + day)//365]
            SN = T1*a1+T2*a2*m2
            # delta_psi = (4.2-1.5) * (10**((math.log10(esals[start_day + day])-(0.45*(-1.645))-(9.36*math.log10(SN+1))\
            #     +0.2+8.07-(2.32*math.log10(Mr_cur)))*(0.4+(1094/((SN+1)**5.19)))))
            delta_psi = calculate_delta_psi(esals[start_day+day], SN)
            daily_psi_drop[day] = last_psi - delta_psi
            last_psi = daily_psi_drop[day]
            if last_psi < psi_t:
                break
        # print(f"SN: {SN} ----- delta_psi: {delta_psi}")
        if last_psi < psi_t:
            psi[start_day:start_day + normal_days] = daily_psi_drop
            break
        # Interpolate to spread this drop over 355 days
        # extended_psi_drop = interpolate_psi_drops(daily_psi_drop, flood_effect_days)
        # print(extended_psi_drop)
        # Assign the interpolated values
        psi[start_day:start_day + flood_effect_days] = daily_psi_drop
        # print([start_day,start_day + flood_effect_days])
        # print(psi[start_day:start_day + flood_effect_days])
        # Calculate the additional drop due to flooding in the last 10 days
        for day in range(flooded_days):
            # Assuming a different MR for flood affected days
            Mr_cur = Flooded_Mr[(start_day + day)//365]
            delta_psi = (10**((math.log10(esals[end_flood_effect_days + day])-(0.45*(-1.645))-(9.36*np.log10((T1*a1)+(T2*a2*m2)+1))\
                +0.2+8.07-(2.32*math.log10(Mr_cur)))*(0.4+(1094/(((T1*a1)+(T2*a2*m2)+1)**5.19)))))*(4.2-1.5)
            psi[start_day+flood_effect_days+day] = last_psi - delta_psi
            last_psi = psi[start_day+flood_effect_days+day]
        # Ensure we do not calculate beyond the design_years
        if last_psi < psi_t:
            break

    return psi

st.markdown('## Input Information')
st.markdown('### Pavement Configuration')
fig = go.Figure()
y0_bot = -50.0
y1_bot = y0_bot + 2.0
y0_mid = y1_bot
y1_mid = y1_bot + baseT/50.0
y0_top = y1_mid
y1_top = y1_mid + surT/50.0
scale_factor = 1.5
fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
                        marker=dict(color='LightSkyBlue'),
                        name='surface Layer'))
fig.add_shape(type="rect",
            x0=.4*scale_factor, y0=y0_top*scale_factor, x1=1.6*scale_factor, y1=y1_top*scale_factor,
            line=dict(color="RoyalBlue"),
            fillcolor="LightSkyBlue")

fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
                            marker=dict(color='LightBlue'),
                            name='base/subbase Layer'))
fig.add_shape(type="rect",
            x0=.2*scale_factor, y0=y0_mid*scale_factor, x1=1.8*scale_factor, y1=y1_mid*scale_factor,
            line=dict(color="RoyalBlue"),
            fillcolor="LightBlue")
fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
                            marker=dict(color='Blue'),
                            name='subgrade'))
fig.add_shape(type="rect",
            x0=0*scale_factor, y0=y0_bot*scale_factor, x1=2*scale_factor, y1=y1_bot*scale_factor,
            line=dict(color="RoyalBlue"),
            fillcolor="Blue")
fig.add_shape(type='line',
                x0=-.1*scale_factor, x1=2.3*scale_factor, y0=(y1_bot-gwt/50)*scale_factor, y1=(y1_bot-gwt/50)*scale_factor,
                line=dict(color='black', width=3, dash='dash'))
fig.add_shape(
    type="path",
    path=f"M {2.2*scale_factor} {(y1_bot-gwt/50)*scale_factor} L {2.1*scale_factor} {(y1_bot-gwt/50+0.2)*scale_factor} L {2.3*scale_factor} {(y1_bot-gwt/50+0.2)*scale_factor} Z",
    fillcolor="blue",
    line_color="black"
)
fig.add_shape(
    type="path",
    path=f"M {2.3*scale_factor} {(y1_bot-gwt/50-0.05)*scale_factor} L {2.1*scale_factor} {(y1_bot-gwt/50-0.05)*scale_factor} Z",
    fillcolor="blue",
    line_color="black"
)
fig.add_shape(
    type="path",
    path=f"M {2.27*scale_factor} {(y1_bot-gwt/50-0.1)*scale_factor} L {2.13*scale_factor} {(y1_bot-gwt/50-0.1)*scale_factor} Z",
    fillcolor="blue",
    line_color="black"
)
fig.add_shape(
    type="path",
    path=f"M {2.23*scale_factor} {(y1_bot-gwt/50-0.15)*scale_factor} L {2.17*scale_factor} {(y1_bot-gwt/50-0.15)*scale_factor} Z",
    fillcolor="blue",
    line_color="black"
)

fig.update_layout(
    plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
    paper_bgcolor='rgba(0,0,0,0)',  # Transparent paper background
    font=dict(color="black"),
    xaxis_visible=False, yaxis_visible=False, # Hide the axes
    # title_text="pavement layer structure", plot_bgcolor='white',
    showlegend=True, # Optionally hide the legend if not required
    legend=dict(font=dict(size=24)),
    # paper_bgcolor="white",
    margin=dict(l=0, r=0, t=40, b=0),
    height=300,
    width=800
)
fig.update_traces(marker=dict(symbol='square', size=10)) 
st.plotly_chart(fig, use_container_width=True)

st.markdown("""---""")
st.markdown('### Groundwater Rise and Flooded days')
years = np.arange(0, design_years)
if not uncertainty:
    gwt_values = -gwt + gwt_rise * (years)
    flood_values = flooded_vals
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=years, y=gwt_values, mode='lines+markers', name='Groundwater level', line=dict(color='rgba(0,120,100,1)'),))
    fig.add_trace(go.Scatter(x=years, y=flood_values, mode='lines+markers', name='Flood days', line=dict(color='rgba(120,0,100,1)'), yaxis='y2'))
    fig.update_layout(title='Yearly growth of groundwater and flooded days under the pavement',
                    xaxis_title='Year', 
                    yaxis_title = 'Groundwater level (0: surface layer)',
                    plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
                    paper_bgcolor='rgba(0,0,0,0)',  # Transparent paper background
                    font=dict(color="black"),
                    yaxis=dict(tickfont=dict(color='rgba(0,120,100,1)'),
                            titlefont=dict(color='rgba(0,120,100,1)')),
                    yaxis2=dict(
                        range=(min(flood_values), max(flood_values) + 1),
                        title="Flooded Days",
                        titlefont=dict(color="green"),
                        tickfont=dict(color="green"),
                        anchor="x",
                        overlaying="y",
                        side="right",
                        dtick=1.0
                    ),
                    legend=dict(
                        font=dict(size=15),
                        x=0.5,  # Centers the legend horizontally
                        y=-0.3,  # Positions the legend below the plot
                        xanchor='center',  # Anchors the center of the legend at x
                        yanchor='top',  # Anchors the top of the legend at y
                        orientation='h'  # Optional: makes the legend horizontal)
                    ))
    fig.update_xaxes(tickfont=dict(size=20),title_font=dict(size=20), range=[-1, design_years+1])
    fig.update_yaxes(tickfont=dict(size=20),title_font=dict(size=20))
    st.plotly_chart(fig, use_container_width=True)
else:
    gwt_values = -gwt + gwt_rise * (years)
    gwt_lower_bound = gwt_values - 1.96 * (gwt_rise_std * 0.2 * years)
    gwt_upper_bound = gwt_values + 1.96 * (gwt_rise_std * 0.2 * years)
    flood_values = np.int32(flooded_days + 0.2 * (years))
    flood_lower_bound = flood_values - 1.96 * (flooded_days_std * 0.01 * years)
    flood_upper_bound = flood_values + 1.96 * (flooded_days_std * 0.01 * years)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.concatenate([years, years[::-1]]),  # years forward and then backward
                        y=np.concatenate([gwt_upper_bound, gwt_lower_bound[::-1]]),
                        fill='toself', fillcolor='rgba(0,100,80,0.15)',
                        line=dict(color='rgba(255,255,255,100)'),
                        showlegend=False, name='95% Confidence Interval'))
    fig.add_trace(go.Scatter(x=np.concatenate([years, years[::-1]]),  # years forward and then backward
                        y=np.concatenate([flood_upper_bound, flood_lower_bound[::-1]]),
                        fill='toself', fillcolor='rgba(100,0,80,0.15)',
                        line=dict(color='rgba(255,255,255,100)'),
                        showlegend=False, name='95% Confidence Interval', yaxis='y2'))
    fig.add_trace(go.Scatter(x=years, y=gwt_values, mode='lines+markers', name='Groundwater rise', line=dict(color='rgba(0,120,100,1)'),
                                ))
    fig.add_trace(go.Scatter(x=years, y=flood_values, mode='lines+markers', name='Flooded days', line=dict(color='rgba(120,0,100,1)'), yaxis='y2' 
                                ))
    fig.update_traces(hoverlabel=dict(font_size=16,  # Set the font size
                                  font_family='Arial',  # Set the font family (optional)
                                  bgcolor='white',  # Background color of hover labels
                                  font_color='black'))  # Font color
    fig.update_layout(title='Yearly rise of groundwater table and flooded days',
                    xaxis_title='Year', 
                    yaxis_title = 'Groundwater level (0: surface layer)',
                    plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
                    paper_bgcolor='rgba(0,0,0,0)',  # Transparent paper background
                    font=dict(color="black"),
                    yaxis=dict(tickfont=dict(color='rgba(0,120,100,1)'),
                            titlefont=dict(color='rgba(0,120,100,1)')),
                    yaxis2=dict(
                        range=(min(flood_lower_bound), max(flood_upper_bound) + 1),
                        title="Flooded Days",
                        titlefont=dict(color='rgba(120,0,100,1)'),
                        tickfont=dict(color='rgba(120,0,100,1)'),
                        anchor="x",
                        overlaying="y",
                        side="right",
                        dtick=2.0
                    ),
                    legend=dict(
                        font=dict(size=15),
                        x=0.5,  # Centers the legend horizontally
                        y=-0.3,  # Positions the legend below the plot
                        xanchor='center',  # Anchors the center of the legend at x
                        yanchor='top',  # Anchors the top of the legend at y
                        orientation='h'  # Optional: makes the legend horizontal)
                    ))
    fig.update_xaxes(tickfont=dict(size=20),title_font=dict(size=20), range=[-1, design_years+1])
    fig.update_yaxes(tickfont=dict(size=20),title_font=dict(size=20))
    st.plotly_chart(fig, use_container_width=True)

if prediction_state:

    st.markdown('## Pavement Performance Life Curve')
    st.markdown('### PSI curve with impact of rising groundwater and flood')
    if not uncertainty:
        # print(Mrs)
        GWT_and_Flood_psi_data_A24 = get_psi_gwt_flood(Mrs=Mrs, Flooded_Mr=Flooded_Mr, design_years=design_years)
        psi_data = pd.DataFrame({'Year': np.arange(365*design_years+1)/365,'PSI': GWT_and_Flood_psi_data_A24})
        psi_data['PSI'] = psi_data['PSI'].where(psi_data['PSI'] >= 1.0)

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=psi_data['Year'],
            y=psi_data['PSI'],
            mode='lines',
            name='PSI',
            line=dict(color='red'),
            hoverinfo='text',  # Set hover info to custom text
            text=[f'Year {x:.0f} and Day {int(d)}<br>PSI: {y:.2f}'
              for x, d, y in zip(psi_data['Year']*365//365, (psi_data['Year'] - psi_data['Year']*365//365)*365 , psi_data['PSI'])]
        ))

        # Horizontal line at y=4.2 (Initial PSI)
        fig.add_trace(go.Scatter(
            x=[psi_data['Year'].min(), psi_data['Year'].max()],
            y=[4.2, 4.2],
            mode='lines',
            line=dict(color='black', dash='dash'),
            name='Initial PSI'
        ))

        # Horizontal line at y=1.0 (Terminal PSI)
        fig.add_trace(go.Scatter(
            x=[psi_data['Year'].min(), psi_data['Year'].max()],
            y=[1.0, 1.0],
            mode='lines',
            line=dict(color='gray', dash='dash'),
            name='Terminal PSI'
        ))
        fig.update_traces(hoverlabel=dict(font_size=16,  # Set the font size
                                  font_family='Arial',  # Set the font family (optional)
                                  bgcolor='white',  # Background color of hover labels
                                  font_color='black'))  # Font color

        # Set chart layout details
        fig.update_layout(
            # title='PSI Trend over Years',
            xaxis_title='Year',
            yaxis_title='PSI',
            xaxis=dict(range=[0, 10], showgrid=True),
            yaxis=dict(range=[0.0, 4.5], showgrid=True),
            legend=dict(font=dict(size=15),
                        x=0.5,  # Centers the legend horizontally
                        y=-0.3,  # Positions the legend below the plot
                        xanchor='center',  # Anchors the center of the legend at x
                        yanchor='top',  # Anchors the top of the legend at y
                        orientation='h'  # Optional: makes the legend horizontal)
                    ),
            # legend_title_text='Legend',
            plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
            paper_bgcolor='rgba(0,0,0,0)',  # Transparent paper background
            font=dict(color="black"),
            height=600,
            width=800
        )
        ending_year = np.ceil(np.max(psi_data['Year'].where(psi_data['PSI'] >= 1.0)))
        # print(ending_year)
        fig.update_xaxes(tickfont=dict(size=25),title_font=dict(size=25), range=[-1, ending_year+1])
        fig.update_yaxes(tickfont=dict(size=25),title_font=dict(size=25))
        # fig.update_xaxes(range=[-1, ending_year+1])
        # Show plot in Streamlit
        st.plotly_chart(fig, use_container_width=True)
        # fig, ax = plt.subplots(figsize=(10, 7.5))
        # ax.plot(psi_data['Year'], psi_data['PSI'], color='r', linestyle='--')
        # ax.axhline(y=4.2, color='black', linestyle='--', label='Initial PSI')
        # ax.axhline(y=1.0, color='gray', linestyle='--', label='Terminal PSI')
        # ax.set_xlabel('Year')
        # ax.set_ylabel('PSI')
        # ax.set_title('PSI Trend over Years')
        # ax.set_xlim(0, 10)
        # ax.set_ylim(0.0, 4.5)
        # ax.legend()
        # ax.grid(True)
        # st.title('Predicted PSI curve considering the rising groundwater and flood')
        # st.pyplot(fig)
    else:
        num_simulations = 200
        all_psi_results = []
        for _ in range(num_simulations):
            sim_flooded_days = np.array(get_flooded_days(flooded_days_std=flooded_days_std))
            sim_flooded_days[sim_flooded_days<0] = 0
            # print(gwt_rise_std)
            gwts = get_gwts(GWT_initial, gwt_rise, gwt_rise_std)
            # print(gwts)
            # Mrs = generate_Mr(gwts, Mr_initial, soil_type=sg_type)
            # print(Mrs)
            psi_results = get_psi_gwt_flood(flooded_vals=sim_flooded_days, Mrs=Mrs, Flooded_Mr=Flooded_Mr)
            all_psi_results.append(psi_results)

        all_psi_results = np.array(all_psi_results)
        mean_psi = np.mean(all_psi_results, axis=0)
        std_psi = np.std(all_psi_results, axis=0)
        ci_psi = 1.96 * std_psi

        years = np.arange(365 * design_years + 1) / 365
        valid_psi = mean_psi >= 1.0

        fig = go.Figure()
        # Confidence interval area
        fig.add_trace(go.Scatter(
            x=np.concatenate([years[valid_psi], years[valid_psi][::-1]]),  # x, then x reversed
            y=np.concatenate([(mean_psi - ci_psi)[valid_psi], (mean_psi + ci_psi)[valid_psi][::-1]]),  # upper, then lower reversed
            fill='toself',
            fillcolor='pink',
            line=dict(color='rgba(255,255,255,100)'),
            name='95% Confidence Interval',
            showlegend=True,
            hoverinfo='skip'
        ))
        fig.add_trace(go.Scatter(
            x = years[valid_psi],
            y = mean_psi[valid_psi],
            mode='lines',
            name='mean PSI',
            line=dict(color='red', dash='solid'),
            hoverinfo='text',  # Set hover info to custom text
            text=[f'Year {x:.0f} and Day {int(d)}<br>Mean PSI: {y:.2f}<br>Lower CI: {y-c:.2f}<br>Upper CI: {y+c:.2f}'
              for x, d, y, c in zip(years[valid_psi]*365//365, (years[valid_psi] - years[valid_psi]*365//365)*365 , mean_psi[valid_psi], ci_psi[valid_psi])]
        ))

        fig.add_trace(go.Scatter(
            x=[-5, 25],
            y=[4.2, 4.2],
            mode='lines',
            line=dict(color='black', dash='dash'),
            name='Initial PSI'
        ))

        fig.add_trace(go.Scatter(
            x=[-5, 25],
            y=[1.0, 1.0],
            mode='lines',
            line=dict(color='gray', dash='dash'),
            name='Terminal PSI'
        ))

        fig.update_layout(
            # title='PSI Trend over Years',
            xaxis_title='Year',
            yaxis_title='PSI',
            xaxis=dict(range=[0, 10]),
            yaxis=dict(range=[0.0, 4.5]),
            legend=dict(
                font=dict(size=15),
                x=0.5,  # Centers the legend horizontally
                y=-0.3,  # Positions the legend below the plot
                xanchor='center',  # Anchors the center of the legend at x
                yanchor='top',  # Anchors the top of the legend at y
                orientation='h'  # Optional: makes the legend horizontal)
            ),
            plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
            paper_bgcolor='rgba(0,0,0,0)',  # Transparent paper background
            font=dict(color="black"),
        )
        fig.update_traces(hoverlabel=dict(font_size=16,  # Set the font size
                                  font_family='Arial',  # Set the font family (optional)
                                  bgcolor='white',  # Background color of hover labels
                                  font_color='black'))  # Font color
        # Adding grid lines manually
        ending_year = np.where(mean_psi < 1.0)
        if ending_year[0].size == 0:
            ending_year = design_years
        else:
            ending_year = np.where(mean_psi < 1.0)[0][0]//365
        
        # print(ending_year)
        fig.update_xaxes(range=[-1, ending_year+1],tickfont=dict(size=20),title_font=dict(size=20),showgrid=True, gridwidth=1, gridcolor='lightgrey')
        fig.update_yaxes(tickfont=dict(size=20),title_font=dict(size=20),showgrid=True, gridwidth=1, gridcolor='lightgrey')

        # Show plot in Streamlit
        st.plotly_chart(fig, use_container_width=True)
        # fig, ax = plt.subplots(figsize=(10, 7.5))

        # ax.plot(years, mean_psi[valid_psi], color='r', linestyle='--')
        # plt.fill_between(years[valid_psi], (mean_psi - ci_psi)[valid_psi], (mean_psi + ci_psi)[valid_psi], color='pink', alpha=0.7)
        # ax.axhline(y=4.2, color='black', linestyle='--', label='Initial PSI')
        # ax.axhline(y=1.0, color='gray', linestyle='--', label='Terminal PSI')
        # ax.set_xlabel('Year')
        # ax.set_ylabel('PSI')
        # ax.set_title('PSI Trend over Years')
        # ax.set_xlim(0, 10)
        # ax.set_ylim(0.0, 4.5)
        # ax.legend()
        # ax.grid(True)
        # st.title('Predicted PSI curve considering the rising groundwater and flood with uncertainty')
        # st.pyplot(fig)
    # print(edge_sat_pred)
st.divider()