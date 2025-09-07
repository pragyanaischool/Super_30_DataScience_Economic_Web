import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import base64
import numpy as np
import tempfile
from fpdf import FPDF
import geopandas as gpd

# --- Configuration and Helper Functions ---
st.set_page_config(
    page_title="Advanced Economic Data Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📊"
)
st.title("Advanced Economic Data Analytics & Report Generator")
st.markdown("---")

# Dictionary of pre-configured websites and their scraping hints
WEBSITES = {
    "IMF Data (Placeholder)": {
        "url": "https://www.imf.org/external/datamapper/datasets",
        "description": "IMF's DataMapper section. Scraped tables may be incomplete.",
        "placeholder": True
    },
    "World Bank (Placeholder)": {
        "url": "https://data.worldbank.org/indicator/NY.GDP.MKTP.CD",
        "description": "World Bank GDP data. Scraped tables may be incomplete.",
        "placeholder": True
    },
    "Google Finance": {
        "url": "https://www.google.com/finance/",
        "description": "Google Finance provides real-time stock quotes, charts, and news.",
        "placeholder": True
    }
}

@st.cache_data(ttl=3600)
def scrape_tables_from_url(url):
    """Scrapes all tables from a given URL and returns them as a list of pandas DataFrames."""
    try:
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            tables = soup.find_all('table')
            if not tables:
                st.warning("No tables were found on the provided URL.")
                return []
            dfs = pd.read_html(response.content)
            return dfs
        else:
            st.error(f"Failed to retrieve content from {url}. Status code: {response.status_code}")
            return []
    except Exception as e:
        st.error(f"An error occurred during web scraping: {e}")
        return []

def filter_numeric_range(df, col):
    if not pd.api.types.is_numeric_dtype(df[col]):
        return df
    min_val = df[col].min()
    max_val = df[col].max()
    selected_range = st.sidebar.slider(f"Filter '{col}' range:", float(min_val), float(max_val), (float(min_val), float(max_val)))
    filtered_df = df[(df[col] >= selected_range[0]) & (df[col] <= selected_range[1])]
    return filtered_df

def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Data')
    writer.save()
    processed_data = output.getvalue()
    return processed_data

def create_pdf_report(title, data_md, analysis_text, chart_path=None):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, title, ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    for line in data_md.split('\n'):
        pdf.multi_cell(0, 8, line)
    pdf.ln(10)
    pdf.multi_cell(0, 8, "Analysis:\n" + analysis_text)
    if chart_path:
        pdf.add_page()
        pdf.image(chart_path, x=10, y=20, w=pdf.epw)
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf.output(tmp_file.name)
    return tmp_file.name

# --- Sidebar UI for Data Selection ---
with st.sidebar:
    st.header("Settings")
    selected_website = st.selectbox("Select a Website to Scrape:", list(WEBSITES.keys()))
    st.markdown(f"**Description:** {WEBSITES[selected_website]['description']}")
    if st.button("Scrape Data from Selected Site") or (not st.session_state.get('data_frames')):
        st.session_state.data_frames = scrape_tables_from_url(WEBSITES[selected_website]["url"])
        if not st.session_state.get('data_frames'):
            st.session_state.data_frames = []
        st.session_state.page_state = 'data_loaded'
        st.success("Data scraped successfully! Select table and options below.")

    # Theme selector
    theme = st.selectbox("Select Theme:", ["Light", "Dark"])
    if theme == "Dark":
        st.write('<style>body{background-color:#0e1117;color:white;}</style>', unsafe_allow_html=True)
    else:
        st.write('<style>body{background-color:white;color:black;}</style>', unsafe_allow_html=True)

    # Export button
    if 'filtered_df' in st.session_state:
        df_xlsx = to_excel(st.session_state.filtered_df)
        st.download_button(label="Download Filtered Data as Excel", data=df_xlsx, file_name="filtered_data.xlsx")

# Initialize session state
if 'page_state' not in st.session_state:
    st.session_state.page_state = 'initial'
if 'data_frames' not in st.session_state:
    st.session_state.data_frames = []

# --- Geo Plot function for Country GDP visualization ---
def plot_gdp_geoplot(df, country_col, gdp_col):
    try:
        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        df['country_lower'] = df[country_col].astype(str).str.lower()
        world['name_lower'] = world['name'].str.lower()
        merged = world.merge(df, how='left', left_on='name_lower', right_on='country_lower')

        fig = px.choropleth(
            merged,
            geojson=merged.geometry,
            locations=merged.index,
            color=gdp_col,
            hover_name='name',
            color_continuous_scale='Viridis',
            labels={gdp_col: f'{gdp_col} Value'},
            title=f'Global {gdp_col} by Country'
        )
        fig.update_geos(fitbounds="locations", visible=False)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error generating Geo plot: {e}")

# --- Main Content UI ---
if st.session_state.page_state == 'data_loaded':
    if not st.session_state.data_frames:
        st.warning("No data frames were scraped. Please try again or select a different site.")
    else:
        st.subheader("Tables Overview")

        tabs = st.tabs([f"Table {i+1}" for i in range(len(st.session_state.data_frames))])
        selected_df = None
        for i, tab in enumerate(tabs):
            with tab:
                df = st.session_state.data_frames[i]
                st.write(f"**Shape:** {df.shape}")
                st.dataframe(df)
                if st.checkbox(f"Select this table {i+1} for analysis"):
                    selected_df = df

        if selected_df is not None:
            st.markdown("---")
            st.subheader("Data Filtering")

            numeric_cols = [col for col in selected_df.columns if pd.api.types.is_numeric_dtype(selected_df[col])]
            filtered_df = selected_df.copy()
            for col in numeric_cols:
                filtered_df = filter_numeric_range(filtered_df, col)
            st.session_state.filtered_df = filtered_df
            st.write(f"Filtered Data Shape: {filtered_df.shape}")
            st.dataframe(filtered_df)

            st.markdown("---")
            st.subheader("Summary Statistics")
            st.dataframe(filtered_df.describe())

            # Geo plot
            country_cols = [col for col in filtered_df.columns if 'country' in col.lower()]
            gdp_cols = [col for col in filtered_df.columns if any(key in col.lower() for key in ['gdp', 'income', 'per capita'])]
            if country_cols and gdp_cols:
                st.markdown("---")
                st.subheader("Geospatial GDP Visualization")
                selected_country_col = st.selectbox("Select Country Column for Geo Plot:", country_cols)
                selected_gdp_col = st.selectbox("Select GDP/Income Column for Geo Plot:", gdp_cols)
                plot_gdp_geoplot(filtered_df, selected_country_col, selected_gdp_col)

            st.markdown("---")
            st.subheader("Chart Customization")

            available_columns = list(filtered_df.columns)
            x_axis_col = st.selectbox("Select X-Axis Column:", available_columns, key="x_axis")
            y_axis_col = st.selectbox("Select Y-Axis Column:", numeric_cols, key="y_axis")

            agg_func = st.selectbox("Aggregation Function (applied if grouping):", ["None", "Sum", "Mean", "Median", "Count"], index=0)
            group_col = None
            if agg_func != "None":
                group_col = st.selectbox("Group By Column:", available_columns)

            chart_type = st.selectbox(
                "Select Chart Type:",
                ["Line Chart", "Bar Chart", "Scatter Plot", "Area Chart", "Box Plot", "Histogram"],
                key="chart_type"
            )
            color_col = st.selectbox("Color By (Optional):", [None] + available_columns)

            if st.button("Generate Advanced Chart"):
                try:
                    plot_df = filtered_df.copy()
                    if agg_func != "None" and group_col:
                        if agg_func == "Count":
                            plot_df = plot_df.groupby(group_col)[y_axis_col].count().reset_index(name="Count")
                            y_axis_col_plot = "Count"
                        else:
                            plot_df = plot_df.groupby(group_col)[y_axis_col].agg(agg_func.lower()).reset_index()
                            y_axis_col_plot = y_axis_col
                    else:
                        y_axis_col_plot = y_axis_col

                    plot_df[y_axis_col_plot] = pd.to_numeric(plot_df[y_axis_col_plot], errors='coerce')
                    plot_df = plot_df.dropna(subset=[y_axis_col_plot])

                    if chart_type == "Line Chart":
                        fig = px.line(plot_df, x=x_axis_col if group_col is None else group_col, y=y_axis_col_plot, color=color_col)
                    elif chart_type == "Bar Chart":
                        fig = px.bar(plot_df, x=x_axis_col if group_col is None else group_col, y=y_axis_col_plot, color=color_col)
                    elif chart_type == "Scatter Plot":
                        fig = px.scatter(plot_df, x=x_axis_col, y=y_axis_col_plot, color=color_col)
                    elif chart_type == "Area Chart":
                        fig = px.area(plot_df, x=x_axis_col if group_col is None else group_col, y=y_axis_col_plot, color=color_col)
                    elif chart_type == "Box Plot":
                        fig = px.box(plot_df, x=x_axis_col if group_col is None else group_col, y=y_axis_col_plot, color=color_col)
                    elif chart_type == "Histogram":
                        fig = px.histogram(plot_df, x=y_axis_col_plot, color=color_col)

                    st.plotly_chart(fig, use_container_width=True)

                    chart_path = "chart.png"
                    fig.write_image(chart_path)
                    st.session_state.chart_path = chart_path
                    st.session_state.chart_generated = True

                except Exception as e:
                    st.error(f"Error generating chart: {e}")
            else:
                st.session_state.chart_generated = False

            st.markdown("---")
            st.subheader("Analysis & Report")

            user_analysis = st.text_area("Enter your detailed analysis here:", height=250)

            if st.button("Generate PDF Report"):
                if user_analysis and selected_df is not None:
                    data_md = filtered_df.head(500).to_markdown(index=False)
                    pdf_path = create_pdf_report(
                        title="Economic Data Analysis Report",
                        data_md=data_md,
                        analysis_text=user_analysis,
                        chart_path=st.session_state.chart_path if st.session_state.chart_generated else None
                    )
                    with open(pdf_path, "rb") as f:
                        pdf_bytes = f.read()
                        b64 = base64.b64encode(pdf_bytes).decode()
                        href = f'<a href="data:application/octet-stream;base64,{b64}" download="Economic_Data_Report.pdf">Download PDF Report</a>'
                        st.markdown(href, unsafe_allow_html=True)
                else:
                    st.warning("Please generate a chart and write an analysis before exporting the report.")

else:
    st.info("Select a website and scrape data using the sidebar options.")
