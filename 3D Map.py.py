"""
Streamlit app: 3D Exoplanetary Systems Visualization
- Shows exoplanetary systems in an interactive 3D scatter (Plotly).
- Planets are colored by a simple "habitability classification".
- Hover over a planet to see detailed metadata.

How to run:
1. Create a virtualenv and install dependencies:
   pip install streamlit pandas plotly numpy
2. Save this file as `streamlit_exoplanet_3d_app.py` and run:
   streamlit run streamlit_exoplanet_3d_app.py

Input data:
- You can upload your own CSV. Required columns (case-insensitive):
  system_name, planet_name, semi_major_axis_au, planet_radius_earth,
  planet_mass_earth, equilibrium_temperature_k, discovery_year, star_radius_solar

If you don't upload data, the app uses a small generated demo dataset.

Notes:
- The habitability classification here is a simple heuristic for visualization only
  (not a scientific determination):
    * Habitable: radius in [0.8,1.5] R_earth and eq_temp in [240,310] K
    * Potentially Habitable: radius in [0.5,2.5] and eq_temp in [200,350]
    * Uninhabitable: otherwise (or giant planets)
    * Unknown: missing required fields

"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Exoplanet 3D Habitability Map", layout="wide")

st.title("🪐 Exoplanet Classification Map — 3D Visualization")
st.markdown("Hover on planets for details. Color shows habitability classification.")

# --- helpers ---

def standardize_columns(df):
    # lower-case column names for convenience
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    colmap = {c.lower(): c for c in df.columns}
    # common names
    needed = {
        'system_name': ['system_name','star_name','host_star','planet_system'],
        'planet_name': ['planet_name','name'],
        'semi_major_axis_au': ['semi_major_axis_au','a_au','orbital_distance_au','semimajoraxis_au','semi_major_axis'],
        'planet_radius_earth': ['planet_radius_earth','radius_earth','pl_rade','radius'],
        'planet_mass_earth': ['planet_mass_earth','mass_earth','pl_masse','mass'],
        'equilibrium_temperature_k': ['equilibrium_temperature_k','teq_k','eq_temp_k','equilibrium_temperature'],
        'discovery_year': ['discovery_year','year'],
        'star_radius_solar': ['star_radius_solar','st_rad','star_radius']
    }
    out = {}
    for std, candidates in needed.items():
        found = None
        for cand in candidates:
            for col in df.columns:
                if col.lower() == cand:
                    found = col
                    break
            if found:
                break
        if found:
            out[std] = df[found]
        else:
            out[std] = pd.Series([np.nan]*len(df))
    out_df = pd.DataFrame(out)
    # keep original other columns too
    for col in df.columns:
        if col not in out_df.columns:
            out_df[col] = df[col]
    return out_df


def classify_habitability(row):
    # simple heuristic — only for visualization
    try:
        r = float(row['planet_radius_earth'])
        t = float(row['equilibrium_temperature_k'])
    except Exception:
        return 'Unknown'
    if np.isnan(r) or np.isnan(t):
        return 'Unknown'
    if 0.8 <= r <= 1.5 and 240 <= t <= 310:
        return 'Habitable'
    if 0.5 <= r <= 2.5 and 200 <= t <= 350:
        return 'Potentially Habitable'
    # mark giants
    if r > 6:
        return 'Gas Giant (Uninhabitable)'
    return 'Uninhabitable'


def generate_demo_data():
    systems = []
    # System: Kepler-like
    systems.append({
        'system_name':'Kepler-186',
        'planet_name':'Kepler-186 f',
        'semi_major_axis_au':0.36,
        'planet_radius_earth':1.11,
        'planet_mass_earth':1.4,
        'equilibrium_temperature_k':287,
        'discovery_year':2014,
        'star_radius_solar':0.5
    })
    systems.append({
        'system_name':'Kepler-186',
        'planet_name':'Kepler-186 b',
        'semi_major_axis_au':0.02,
        'planet_radius_earth':1.3,
        'planet_mass_earth':2.0,
        'equilibrium_temperature_k':900,
        'discovery_year':2014,
        'star_radius_solar':0.5
    })
    # TRAPPIST-like compact system
    for i, (a,r,t) in enumerate([(0.01,1.0,250),(0.015,0.9,260),(0.02,1.1,240),(0.03,1.4,280)], start=1):
        systems.append({
            'system_name':'TRAPPIST-1',
            'planet_name':f'TRAPPIST-1 {i}',
            'semi_major_axis_au':a,
            'planet_radius_earth':r,
            'planet_mass_earth':0.9,
            'equilibrium_temperature_k':t,
            'discovery_year':2016,
            'star_radius_solar':0.12
        })
    # A system with gas giant
    systems.append({
        'system_name':'HD-209458',
        'planet_name':'HD-209458 b',
        'semi_major_axis_au':0.047,
        'planet_radius_earth':13.5,
        'planet_mass_earth':220,
        'equilibrium_temperature_k':1450,
        'discovery_year':1999,
        'star_radius_solar':1.2
    })
    return pd.DataFrame(systems)

# --- load data ---

uploaded = st.file_uploader("Upload exoplanet CSV", type=['csv'], help="CSV must contain columns like planet_name, semi_major_axis_au, planet_radius_earth, equilibrium_temperature_k")

if uploaded is not None:
    try:
        df_raw = pd.read_csv(uploaded)
        st.success('File loaded successfully!')
    except Exception as e:
        st.error(f'Failed to read CSV: {e}')
        st.stop()
else:
    st.info('No file uploaded — using demo dataset. Upload your own CSV to visualize real data.')
    df_raw = generate_demo_data()

# standardize
df = standardize_columns(df_raw)

# classification
if 'habitability' not in df.columns:
    df['habitability'] = df.apply(classify_habitability, axis=1)

# make sure numeric types
for col in ['semi_major_axis_au','planet_radius_earth','planet_mass_earth','equilibrium_temperature_k','star_radius_solar']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# sidebar controls
st.sidebar.header('Controls')
system_list = df['system_name'].dropna().unique().tolist()
system_list = sorted(system_list)
selected_system = st.sidebar.selectbox('Choose system', ['All Systems'] + system_list)
size_scale = st.sidebar.slider('Planet marker size scaling', 1.0, 20.0, 6.0)
show_stars = st.sidebar.checkbox('Show host stars', value=True)
color_by = st.sidebar.selectbox('Color by', ['habitability','discovery_year','planet_mass_earth','planet_radius_earth'])

# filter by system
if selected_system != 'All Systems':
    plot_df = df[df['system_name'] == selected_system].copy()
else:
    plot_df = df.copy()

if plot_df.empty:
    st.warning('No data available for the selected system.')
    st.stop()

# Prepare plot coordinates
# X: semi-major axis (au), Y: planet radius (R_earth), Z: equilibrium temp (K)
plot_df['x'] = plot_df['semi_major_axis_au']
plot_df['y'] = plot_df['planet_radius_earth']
plot_df['z'] = plot_df['equilibrium_temperature_k']

# Fallback coords if missing: jitter
for c in ['x','y','z']:
    if plot_df[c].isna().any():
        plot_df[c] = plot_df[c].fillna(plot_df[c].median())

# marker sizes
base_sizes = np.clip(plot_df['planet_radius_earth'].fillna(1.0), 0.1, 30)
marker_sizes = (base_sizes ** 0.5) * size_scale

# color mapping
if color_by == 'habitability':
    color_field = 'habitability'
else:
    color_field = color_by

# Create hover info
hover_template = (
    '<b>%{customdata[0]}</b><br>' +
    'System: %{customdata[1]}<br>' +
    'a (AU): %{x}<br>' +
    'Radius (R⊕): %{y}<br>' +
    'Teq (K): %{z}<br>' +
    'Mass (M⊕): %{customdata[2]}<br>' +
    'Discovery: %{customdata[3]}<extra></extra>'
)

customdata = np.stack([
    plot_df['planet_name'].astype(str),
    plot_df['system_name'].astype(str),
    plot_df['planet_mass_earth'].astype(str),
    plot_df['discovery_year'].astype(str)
], axis=-1)

# build plotly figure
fig = px.scatter_3d(
    plot_df,
    x='x', y='y', z='z',
    color=color_field,
    hover_name='planet_name',
    size=base_sizes,  # plotly uses its own scaling; we provide base but override marker size below
    custom_data=['planet_name','system_name','planet_mass_earth','discovery_year'],
    labels={'x':'Semi-major axis (AU)', 'y':'Radius (R⊕)', 'z':'Equilibrium Temp (K)'}
)

# update markers with our calculated sizes
for i, d in enumerate(fig.data):
    # adjust marker sizes
    d.marker.size = marker_sizes
    d.hovertemplate = hover_template

# optionally add star marker at origin for each system (displayed at x=0,y=star_radius,z=star_temp proxy)
if show_stars:
    # if plotting a single system, show its star(s)
    stars = []
    if selected_system != 'All Systems':
        s = plot_df.drop_duplicates('system_name')[['system_name','star_radius_solar']]
        for _, row in s.iterrows():
            stars.append({'system_name':row['system_name'],'star_radius_solar':row['star_radius_solar']})
    else:
        s = df.drop_duplicates('system_name')[['system_name','star_radius_solar']]
        for _, row in s.iterrows():
            stars.append({'system_name':row['system_name'],'star_radius_solar':row['star_radius_solar']})
    # add star scatter as big points at x=0, y=star_radius (just for visual)
    star_x = []
    star_y = []
    star_z = []
    star_text = []
    star_sizes = []
    for s in stars:
        star_x.append(0.0)
        # plot star radius scaled to y axis for visualization
        star_y.append(max(0.01, float(s.get('star_radius_solar', 1.0))))
        star_z.append(plot_df['z'].median() if not plot_df['z'].isna().all() else 300)
        star_text.append(s['system_name'] + ' (star)')
        star_sizes.append(30)
    if len(star_x) > 0:
        fig.add_scatter3d(x=star_x, y=star_y, z=star_z, mode='markers',
                          marker=dict(symbol='circle', size=star_sizes, opacity=0.9),
                          name='Host stars', hovertext=star_text)

fig.update_layout(margin=dict(l=0,r=0,b=0,t=40), legend=dict(itemsizing='constant'))

# show figure
st.plotly_chart(fig, use_container_width=True)

# Sidebar: data table and download
with st.expander('Show data table'):
    st.dataframe(plot_df.drop(columns=['x','y','z']) if 'x' in plot_df.columns else plot_df)

# Download processed dataset
@st.cache_data
def to_csv_bytes(df_):
    return df_.to_csv(index=False).encode('utf-8')

csv_bytes = to_csv_bytes(plot_df)
st.download_button('Download plotted dataset (CSV)', csv_bytes, file_name='plotted_exoplanets.csv')

st.markdown('---')
st.caption('This interactive app is for visualization and teaching. Habitability classification is heuristic and simplified.')
