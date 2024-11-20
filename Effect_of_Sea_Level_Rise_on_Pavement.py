import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pickle
import math
import plotly.graph_objects as go
from PIL import Image

from calc_traffic import get_daily_ESAL
from calc_modulus import generate_Mr, generate_flooded_Mr

st.set_page_config(
    page_title='Effect of Sea Level Rise on Pavement Performance',
    page_icon=":arrow_right:",
    menu_items={
        'Get Help': 'mailto:wei.sun@unh.edu',
        'Report a bug': 'mailto:wei.sun@unh.edu',
        'About': "This is a Streamlit app developed by Wei Sun, focusing on the impact of sea level rise on pavement performance"
    }
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
# Add custom CSS to hide the GitHub icon
st.markdown(
    """
    <style>
    .stToolbarActionButton {
        display: none;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# image = Image.open('Figure 3.jpg')
# st.image(image, caption = 'Schematics for the pavement modelling')

#Converter
inch2meter = 0.0254
meter2inch = 39.3701


## Deal with input
st.sidebar.markdown("## Input Parameters")  # Main sidebar header with increased size
st.sidebar.markdown("### Pavement Info")  # Subheader for pavement information
with st.sidebar.expander('Design Period'):
    design_years = st.slider('Designed service year:', min_value=5, max_value=30, step=1, value=20)

with st.sidebar.expander("Layer Thickness"):
    surT = st.slider('Thickness of the AC layer (inch)', min_value=3, max_value=26, step=1, value=6)
    baseT = st.slider('Thickness of the base layer (inch)', min_value=6, max_value=28, step=1, value=8)

with st.sidebar.expander("Material Properties"):
    base_type = st.selectbox("Base type", ('A-1-a','A-1-b', 'A-2-4'), help='This will affect the hydrodynamic parameters and resilient modulus used for calculation.')
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
    growth_rate = st.number_input("Growth Rate", min_value=0.0, max_value=0.5, value=0.04, step=0.01)
    advanced_settings = st.checkbox("Advanced Settings")
    if advanced_settings:
        traffic_class_options = [str(i) for i in range(1, 14)]  # Traffic classes from 1 to 13
        traffic_classes = {}
        for class_label in range(4, 14):  # Traffic classes from 1 to 13
            percentage = st.number_input(
                f"Percentage for Class {class_label}",
                min_value=0,
                max_value=100,
                value=0 if class_label != 5 else 100,
                key=f"percentage_class_{class_label}"
            )
            traffic_classes[class_label] = percentage/100
        cum = 0
        for traffic_class in traffic_classes.keys():
            cum += traffic_classes[traffic_class]
        remaining = 1.0 - cum
        traffic_classes[1] = remaining / 3.0
        traffic_classes[2] = remaining / 3.0
        traffic_classes[3] = remaining / 3.0
    else:
        traffic_classes = {3:0.1,5:0.4,6:0.3,7:0.2}  # Or set a default dictionary if needed

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

esals = get_daily_ESAL(ADT, traffic_perc=traffic_classes, G=growth_rate, design_years=design_years)
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


GWT_initial = gwt # inch
if not uncertainty:
    gwt_rise_std = 0.0
def get_gwts(g_initial = GWT_initial,g_rise=gwt_rise,gwt_std=0.0, years=design_years):
    gwts = [g_initial - g_rise * i for i in range(design_years+1)]
    if gwt_std > 0.0:
        gwts = [np.random.normal(mean, gwt_std + 0.1*year*gwt_std) for year, mean in enumerate(gwts)]
    return gwts


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

# print(Mrs)
gwt_vals = get_gwts(GWT_initial, gwt_rise, gwt_rise_std, design_years)
flooded_vals = get_flooded_days(flooded_days, 0.2)
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
input_params = {
    'Surface Thickness':surT,
    'Base Thickness':baseT,
    'GWT':GWT_initial,
    'Base Type':soil_map[base_type],
    'Subgrade Type':soil_map[sg_type]
}
Mrs = generate_Mr(gwt_vals, soil_type=sg_type)
input_params_list = []
for cur_gwt in gwt_vals:
    input_params_year = input_params.copy()
    input_params_year['GWT'] = cur_gwt
    input_params_list.append(input_params_year)
Flooded_Mr = generate_flooded_Mr(input_params_list)
# print(flooded_vals)
def calc_SN(surT=surT, baseT=baseT, a1=a1, a2=a2, m2=m2):
    return a1 * surT + a2 * baseT * m2
SN = calc_SN()
def calculate_delta_psi(esal=5000, SN=5.0):
    return (10**((np.log10(esal))-(0.45*(-1.645))-(9.36*np.log10(SN+1))\
                +0.2+8.07-(2.32*np.log10(15000)))*(0.4+(1094/((SN+1)**5.19))))*(4.2-1.5)
# print(Mrs)
# print(Flooded_Mr)
def get_psi_gwt_flood(psi_i=4.2, psi_t=1.0, esals=esals, design_years=20, Mrs=Mrs, Flooded_Mr=Flooded_Mr, flooded_vals=flooded_vals):
    days = np.arange(design_years * 365+1)
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
        # Assign the interpolated values
        psi[start_day:start_day + flood_effect_days] = daily_psi_drop
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
# Scale factor to adjust the visual representation
scale_factor = 2.0  # Adjust as needed for your visualization

# Calculate scaled thicknesses
surT_scaled = surT / scale_factor
baseT_scaled = baseT / scale_factor
total_layer_thickness_scaled = (surT + baseT) / scale_factor

# Define the y-values for each layer
y1_top = 0  # Top of the surface layer (ground level)
y0_top = y1_top - surT_scaled  # Bottom of the surface layer

y1_mid = y0_top  # Top of the base layer
y0_mid = y1_mid - baseT_scaled  # Bottom of the base layer

y1_bot = y0_mid  # Top of the subgrade layer
y0_bot = y1_bot - 50  # Arbitrary depth for the subgrade (semi-infinite)

# Calculate groundwater table positions
current_gwt_scaled = gwt / scale_factor
future_gwt_scaled = (gwt - gwt_rise * design_years) / scale_factor
# print(gwt)
# print(future_gwt_scaled)
# Check if the future GWT exceeds the top of the subgrade
fully_saturated_subgrade = False
fully_saturated_base = False
fully_saturated_surface = False
if future_gwt_scaled <= 0:
    if future_gwt_scaled < - (surT + baseT) / scale_factor:
        future_gwt_scaled = - (surT + baseT) / scale_factor
        fully_saturated_surface = True
    elif future_gwt_scaled < - baseT / scale_factor:
        future_gwt_scaled = - baseT / scale_factor
        fully_saturated_base = True
    else:
        future_gwt_scaled = 0
        fully_saturated_subgrade = True

# Calculate y-positions for the GWT lines
y_gwt_current = y1_bot - current_gwt_scaled
y_gwt_future = y1_bot - future_gwt_scaled

# Initialize the figure
fig = go.Figure()

# Draw the subgrade layer (drawn first so it's at the back)
fig.add_shape(
    type="rect",
    x0=0.0, y0=y0_bot, x1=2.0, y1=y1_bot,
    line=dict(color="RoyalBlue"),
    fillcolor="Blue",
    layer='below'  # Drawn below other shapes
)

# Draw the base layer
fig.add_shape(
    type="rect",
    x0=0.1, y0=y0_mid, x1=1.9, y1=y1_mid,
    line=dict(color="RoyalBlue"),
    fillcolor="LightBlue",
    layer='below'
)

# Draw the surface layer
fig.add_shape(
    type="rect",
    x0=0.2, y0=y0_top, x1=1.8, y1=y1_top,
    line=dict(color="RoyalBlue"),
    fillcolor="LightSkyBlue",
    layer='below'
)

# Add the current GWT line and shapes
fig.add_shape(
    type='line',
    x0=-0.1, x1=2.3,
    y0=y_gwt_current, y1=y_gwt_current,
    line=dict(color='black', width=3, dash='dash'),
    layer='above',  # Drawn above other shapes
    name='Current GWT'
)

# Add the triangle and lines for current GWT
x_center = 2.2
x_offset = 0.1
y_offset = 3.0

# Triangle on current GWT line
fig.add_shape(
    type="path",
    path=f"M {x_center} {y_gwt_current} "
         f"L {x_center - x_offset} {y_gwt_current + y_offset} "
         f"L {x_center + x_offset} {y_gwt_current + y_offset} Z",
    fillcolor="blue",
    line_color="black",
    layer='above'
)

# Small lines below current GWT line
for i in range(1, 4):
    y_line = y_gwt_current - 0.5 * i
    x_left = x_center - (x_offset / 2) * (2 - 0.4 * i)
    x_right = x_center + (x_offset / 2) * (2 - 0.4 * i)
    fig.add_shape(
        type="line",
        x0=x_left, y0=y_line, x1=x_right, y1=y_line,
        line=dict(color='black'),
        layer='above'
    )

# Add the future GWT line and shapes
fig.add_shape(
    type='line',
    x0=-0.1, x1=2.3,
    y0=y_gwt_future, y1=y_gwt_future,
    line=dict(color='purple', width=3, dash='dash'),
    layer='above',
    name=f'Future GWT at Year {design_years}'
)

# Triangle on future GWT line
fig.add_shape(
    type="path",
    path=f"M {x_center} {y_gwt_future} "
         f"L {x_center - x_offset} {y_gwt_future + y_offset} "
         f"L {x_center + x_offset} {y_gwt_future + y_offset} Z",
    fillcolor="cyan",
    line_color="purple",
    layer='above'
)

# Small lines below future GWT line
for i in range(1, 4):
    y_line = y_gwt_future - 0.5 * i
    x_left = x_center - (x_offset / 2) * (2 - 0.4 * i)
    x_right = x_center + (x_offset / 2) * (2 - 0.4 * i)
    fig.add_shape(
        type="line",
        x0=x_left, y0=y_line, x1=x_right, y1=y_line,
        line=dict(color='purple'),
        layer='above'
    )

# Add labels for the GWT lines
fig.add_annotation(
    x=2.3,
    y=y_gwt_current,
    text="Current GWT",
    showarrow=False,
    xanchor='left',
    yanchor='middle',
    font=dict(color='black', size=20)
)

fig.add_annotation(
    x=2.3,
    y=y_gwt_future,
    text=f'Future GWT at Year {design_years}',
    showarrow=False,
    xanchor='left',
    yanchor='middle',
    font=dict(color='purple', size=20)
)

# Add legends for layers by adding invisible traces
fig.add_trace(go.Scatter(
    x=[None],
    y=[None],
    mode='markers',
    marker=dict(size=20, color='LightSkyBlue'),
    name='Surface Layer'
))
fig.add_trace(go.Scatter(
    x=[None],
    y=[None],
    mode='markers',
    marker=dict(size=20, color='LightBlue'),
    name='Base Layer'
))
fig.add_trace(go.Scatter(
    x=[None],
    y=[None],
    mode='markers',
    marker=dict(size=20, color='Blue'),
    name='Subgrade'
))

# Update the layout
fig.update_layout(
    title="Pavement Layer Structure",
    xaxis=dict(visible=False),
    yaxis=dict(visible=False),
    showlegend=True,
    legend=dict(
        itemsizing='constant',
        font=dict(size=20)
    ),
    height=500,
    margin=dict(l=0, r=50, t=50, b=0)
)

# Add text description if the subgrade will be fully saturated
if fully_saturated_surface:
    st.write(f"**Note: The :red[surface will be fully saturated] in Year {design_years}.**")
elif fully_saturated_base:
    st.write(f"**Note: The :red[base will be fully saturated] in Year {design_years}.**")
elif fully_saturated_subgrade:
    st.write(f"**Note: The :red[subgrade will be fully saturated] in Year {design_years}.**")

# Display the figure
st.plotly_chart(fig, use_container_width=True)

st.markdown("""---""")
st.markdown('### Traffic Growth and Distribution')
esals = []
x_values = []
hover_texts = []
for day in range(design_years * 365):
    # Compute the year
    year = day // 365
    # Compute the AADT for that year
    current_aadt = aadt * ((1 + growth_rate) ** year)
    # Assuming ESAL per day is proportional to AADT / 365
    daily_esal = current_aadt
    esals.append(daily_esal)
    x_values.append(day)
    year_display = year + 1  # Year numbering starts from 1
    day_of_year = (day % 365) + 1  # Day numbering starts from 1
    hover_text = f"Year {year_display}, Day {day_of_year}"
    hover_texts.append(hover_text)

# Plotting
if traffic_classes:
    # Create ESALS growth line plot
    fig_esals = go.Figure()
    fig_esals.add_trace(go.Scatter(
        x=x_values,
        y=esals,
        mode='lines',
        name='Daily ESALS',
        hovertext=hover_texts,
        hoverinfo='text+y'
    ))

    # Set x-axis ticks at the start of each year to avoid clutter
    x_ticks = [i * 365 * 2 for i in range(design_years + 1)]  # From day 0 to total_days
    x_tick_labels = [str(i + 1) for i in range(0, design_years + 1, 2)]  # Years 1 to design_years
    fig_esals.update_layout(
        title='ESALS Growth Over Design Years',
        xaxis=dict(
            title='Year',
            tickmode='array',
            tickvals=x_ticks,
            ticktext=x_tick_labels,
            showgrid=False,
            tickangle=45
        ),
        yaxis_title='Daily ESALS',
        margin=dict(l=50, r=50, t=50, b=50),
        font=dict(color="black"),
        legend=dict(
            font=dict(size=15),
            x=0.5,  # Centers the legend horizontally
            y=-0.3,  # Positions the legend below the plot
            xanchor='center',  # Anchors the center of the legend at x
            yanchor='top',  # Anchors the top of the legend at y
            orientation='h'  # Optional: makes the legend horizontal)
        )
    )
    fig_esals.update_yaxes(tickfont=dict(size=20),title_font=dict(size=20))
    fig_esals.update_xaxes(tickfont=dict(size=20),title_font=dict(size=20))
    # Create bar chart for traffic class percentages
    class_labels = list(range(1, 14))
    class_percentages = [traffic_classes[cur] * 100 if cur in traffic_classes else 0 for cur in range(1, 14)]
    hover_texts = [f'Traffic Class {cls}, Percentage: {perc:.2f}%' for cls, perc in zip(class_labels, class_percentages)]
    fig_percentages = go.Figure()
    fig_percentages.add_trace(go.Bar(
        x=class_labels,
        y=class_percentages,
        name='Traffic Class Percentages',
        marker_color='orange',
        hovertext=hover_texts,
        hoverinfo='text'
    ))
    fig_percentages.update_layout(
        title='Traffic Class Percentages',
        xaxis=dict(
            title='Traffic Class',
            tickmode='linear',
            dtick=1,
            tick0=1,
            tickangle=45
        ),
        yaxis_title='Percentage (%)',
        yaxis=dict(range=[0, 100]),
        margin=dict(l=50, r=50, t=50, b=50),
                font=dict(color="black"),
        legend=dict(
            font=dict(size=15),
            x=0.5,  # Centers the legend horizontally
            y=-0.3,  # Positions the legend below the plot
            xanchor='center',  # Anchors the center of the legend at x
            yanchor='top',  # Anchors the top of the legend at y
            orientation='h'  # Optional: makes the legend horizontal)
        )
    )
    fig_percentages.update_xaxes(tickfont=dict(size=20),title_font=dict(size=20))
    fig_percentages.update_yaxes(tickfont=dict(size=20),title_font=dict(size=20))
    # Display plots side by side
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_esals, use_container_width=True)
    with col2:
        st.plotly_chart(fig_percentages, use_container_width=True)
else:
    st.write("Please enable Advanced Settings and input traffic class percentages to view the plots.")

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
                    yaxis_title = 'GWT from top of subgrade (inch)',
                    plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
                    paper_bgcolor='rgba(0,0,0,0)',  # Transparent paper background
                    font=dict(color="black"),
                    yaxis=dict(tickfont=dict(color='rgba(0,120,100,1)'),
                            titlefont=dict(color='rgba(0,120,100,1)')),
                    yaxis2=dict(
                        range=(min(flood_values), max(flood_values) + 1),
                        title="Flooded Days",
                        anchor="x",
                        overlaying="y",
                        side="right",
                        dtick=1.0,
                        tickfont=dict(color='rgba(120,0,100,1)'),
                        titlefont=dict(color='rgba(120,0,100,1)'),
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

        st.plotly_chart(fig, use_container_width=True)

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
            psi_results = get_psi_gwt_flood(flooded_vals=sim_flooded_days, Mrs=Mrs, Flooded_Mr=Flooded_Mr, design_years=design_years)
            all_psi_results.append(psi_results)

        all_psi_results = np.array(all_psi_results)
        mean_psi = np.mean(all_psi_results, axis=0)
        std_psi = np.std(all_psi_results, axis=0)
        ci_psi = 1.96 * std_psi

        years = np.arange(365 * design_years + 1) / 365
        valid_psi = mean_psi >= 1.0
        # valid_psi_tmp = np.full_like(years, fill_value=False, dtype=bool)
        # valid_psi_tmp[:len(valid_psi)] = valid_psi
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