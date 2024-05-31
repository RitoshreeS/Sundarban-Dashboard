import dash
from dash import dcc, html, Input, Output
import pandas as pd
from docx import Document
from flask import json
import plotly.graph_objects as go
from dash import Dash, html, dcc, Input, Output
import requests
from datetime import datetime
import geopandas as gpd
import folium
import plotly.express as px
from plotly.subplots import make_subplots

# Load the Word document
doc = Document('https://github.com/RitoshreeS/Sundarban-Dashboard/blob/main/Data/Crop Data.docx')

# Assuming the table is the first table in the document
table = doc.tables[0]

# Extract data from the table
data = []
for row in table.rows:
    row_data = [cell.text for cell in row.cells]
    data.append(row_data)

# Convert the data to a DataFrame
df_cropdata = pd.DataFrame(data[1:], columns=data[0])

# Convert the "CROP" column to uppercase
df_cropdata['CROP'] = df_cropdata['CROP'].str.upper()

# Read the GeoJSON file
gdf = gpd.read_file("https://github.com/RitoshreeS/Sundarban-Dashboard/blob/main/Data/SB_Landscape_Boundary.shp.geojson")

with open ("https://github.com/RitoshreeS/Sundarban-Dashboard/blob/main/Data/27_villages.shp_all.geojson") as f:
    geojson_data = json.load(f)

# Filter the GeoJSON data for the specified locations
locations = ['Kultali', 'Patharpratima', 'Gosaba']
filtered_gdf = gdf[gdf['sdtname'].isin(locations)]

# Define colors for each location
colors = {
    'Kultali': 'blue',
    'Patharpratima': 'green',
    'Gosaba': 'red'
}

df = pd.read_csv('https://github.com/RitoshreeS/Sundarban-Dashboard/blob/main/Data/QueryData.csv')

# Initialize the Dash app
app = dash.Dash(__name__, assets_folder='assets', assets_url_path='https://github.com/RitoshreeS/Sundarban-Dashboard/tree/main/assets')
server = app.server

# Define image source file path for the WWF India logo
image_link = 'https://github.com/RitoshreeS/Sundarban-Dashboard/tree/main/assets/WWF_logo.png'
app.layout = html.Div(style={'background-image': 'url("https://github.com/RitoshreeS/Sundarban-Dashboard/tree/main/assets/deer.jpg")',
                             'background-size': 'cover',
                             'background-repeat': 'no-repeat',
                             'background-position': 'center',
                             'height': '250vh',
                             'backgroundColor':'rgba(0,0,0,0.7)',
                             'border': 'ridge',
                             'padding': '10px'}, children=[
    dcc.Tabs(id='tabs', value='tab-3', children=[
        dcc.Tab(label='WEATHER', value='tab-3', children=[
            html.Div(style={'display': 'flex', 'align-items': 'center', 'justify-content': 'flex-start'}, children=[
                html.Button(
                    id='wwf-button1',
                    children=[
                        html.Img(src=image_link, alt='WWF India Logo', style={'width': '90px', 'height': 'auto'}),
                    ],
                    style={'border': 'none', 'background': 'white', 'cursor': 'pointer', 'padding': '0px','margin-right': '600px'}
                ),
                html.H1('SUNDARBANS WEATHER DASHBOARD', style={'textAlign': 'center', 'color': 'White', 'font-size': 40, 'font-family': 'Math Italic'}),
            ]),
            # Area for the graphs
            html.Div(dcc.Graph(id='plot1*'), style={'border': 'ridge', 'padding': '20px', 'height': '1200px'}),
            html.Div(style={'display': 'flex', 'justify-content': 'center'}, children=[
                html.Div(dcc.Graph(id='plot2'), style={'border': 'ridge', 'padding': '20px', 'width': '50%', 'height': '550px'}),
                html.Div(dcc.Graph(id='plot3'), style={'border': 'ridge', 'padding': '20px', 'width': '50%', 'height': '550px'})
            ]),
            html.Div(style={'display': 'flex', 'justify-content': 'center'}, children=[
                html.Div(dcc.Graph(id='plot4'), style={'border': 'ridge', 'padding': '20px', 'width': '50%', 'height': '550px'}),
                html.Div(dcc.Graph(id='plot5'), style={'border': 'ridge', 'padding': '20px', 'width': '50%', 'height': '550px'})
            ])
        ]),
        dcc.Tab(label='CROP ADVISORY', value='tab-1', children=[
            html.Div(style={'display': 'flex', 'align-items': 'center', 'justify-content': 'flex-start'}, children=[
                html.Button(
                    id='wwf-button2',
                    children=[
                        html.Img(src=image_link, alt='WWF India Logo', style={'width': '90px', 'height': 'auto'}),
                    ],
                    style={'border': 'none', 'background': 'white', 'cursor': 'pointer', 'padding': '0px','margin-right': '800px'}
                ),
                html.H1("CROP ADVISORY", style={'textAlign': 'center', 'color': 'White', 'font-size': 40}),  # Add the heading here
            ]),
            html.Div([
                dcc.Dropdown(
                    id='crop-dropdown',
                    options=[{'label': crop, 'value': crop} for crop in df_cropdata['CROP'].unique()],
                    value='Crop',  # Default value
                    placeholder="Select a crop",
                    style={'font-size': '18px'}
                ),
            ], style={'margin-bottom': '20px'}),
            html.Div([
                dcc.Dropdown(
                    id='month-dropdown',
                    options=[{'label': month, 'value': month} for month in df_cropdata['MONTH'].unique()],
                    value='Month',  # Default value
                    placeholder="Select a month",
                    style={'font-size': '18px'}
                ),
            ], style={'margin-bottom': '20px'}),
            html.Div(id='crop-info', style={'margin-top': '20px', 'font-size': '25px', 'color': 'black', 'backgroundColor': '#dfe7fd'}),  # Set text color to white

            html.Div(dcc.Graph(id='crop_calendar'), style={'border': 'ridge', 'padding': '10px', 'backgroundColor': 'black', 'height': '1000px'}),
            
            #Crop Advisory content
            
        ]),
        dcc.Tab(label='QUERIES', value='tab-2', children=[
            html.Div(style={'display': 'flex', 'align-items': 'center', 'justify-content': 'flex-start'}, children=[
                html.Button(
                    id='wwf-button3',
                    children=[
                        html.Img(src=image_link, alt='WWF India Logo', style={'width': '90px', 'height': 'auto'}),
                    ],
                    style={'border': 'none', 'background': 'white', 'cursor': 'pointer', 'padding': '0px','margin-right': '450px'}
                ),
                html.H1('AGRICULTURAL QUERIES IN SUNDARBAN LANDSCAPE', style={'textAlign': 'center', 'color': 'White', 'font-size': 40}),
            ]),
            # Area for the Folium map
            html.Div(dcc.Graph(id='query_map'), style={'border': 'ridge', 'padding': '10px', 'height': '1300px'}),
                
            # Area for the Plotly table
            html.Div(id='plot_query', style={'border': 'ridge', 'padding': '10px', 'height': '1000px'}),
                        
            # Query Map content
        ]),
        dcc.Tab(label='SOIL CONTENT', value='tab-4', children=[
            html.Div(style={'display': 'flex', 'align-items': 'center', 'justify-content': 'flex-start'}, children=[
                html.Button(
                    id='wwf-button4',
                    children=[
                        html.Img(src=image_link, alt='WWF India Logo', style={'width': '90px', 'height': 'auto'}),
                    ],
                    style={'border': 'none', 'background': 'white', 'cursor': 'pointer', 'padding': '0px','margin-right': '700px'}
                ),
                html.H1('SUNDARBANS SOIL DASHBOARD', style={'textAlign': 'center', 'color': 'White', 'font-size': 40}),
            ]),
            # Tabs section
            dcc.Tabs(id="soil-tabs", value='tab-1', children=[
                dcc.Tab(label='PH', value='tab-1'),
                dcc.Tab(label='ELECTRICAL CONDUCTIVITY', value='tab-2'),
                dcc.Tab(label='ORGANIC CARBON', value='tab-3'),
                dcc.Tab(label='NITROGEN', value='tab-4'),
                dcc.Tab(label='PHOSPHORUS', value='tab-5'),
                dcc.Tab(label='POTASSIUM', value='tab-6'),
                dcc.Tab(label='COPPER', value='tab-7'),
                dcc.Tab(label='ZINC', value='tab-8'),
                dcc.Tab(label='IRON', value='tab-9'),
                dcc.Tab(label='MANGANESE', value='tab-10'),
                dcc.Tab(label='BORON', value='tab-11'),
                dcc.Tab(label='SULPHUR', value='tab-12'),
                dcc.Tab(label='SOIL TYPE',value='tab-13')
            ]),
            # Tab content
            html.Div(id='soil-tabs-content'),
            # Another plot division
            html.Div([
                dcc.Graph(id='independent-plot')
            ], style={'border': 'ridge', 'padding': '20px', 'height': '800px'})
        ]),
    ])
])



#___________________________________________________________________________________________________________________________________________________________________

# Create callback decorator
@app.callback(
    [Output(component_id='plot1*', component_property='figure'),
     Output(component_id='plot2', component_property='figure'),
     Output(component_id='plot3', component_property='figure'),
     Output(component_id='plot4', component_property='figure'),
     Output(component_id='plot5', component_property='figure')],
    [Input(component_id='wwf-button1', component_property='n_clicks')]
)
def update_graphs(n_clicks):
    
    
    df_weather = pd.read_csv('https://github.com/RitoshreeS/Sundarban-Dashboard/blob/main/Data/Weather.csv')
    # Load the GeoJSON file using geopandas
    gdf = gpd.read_file("https://github.com/RitoshreeS/Sundarban-Dashboard/blob/main/Data/SB_Landscape_Boundary.shp.geojson")

    # Filter the GeoDataFrame to include only the desired values of "sdtname"
    filtered_gdf = gdf[gdf['sdtname'].isin(['Gosaba', 'Patharpratima', 'Kultali'])]

    # Sample data for demonstration
    block_data = {
        'BLOCK': ['Patharpratima', 'Kultali', 'Gosaba'],
        'LATITUDE': [21.79206941, 21.8820151, 22.16557075],
        'LONGITUDE': [88.3555912, 88.5265151, 88.80817531],
    }
    df_blocks = pd.DataFrame(block_data)

    # Define the OpenWeatherMap API endpoint
    API_ENDPOINT = "https://api.openweathermap.org/data/2.5/forecast"

    # Define the API parameters for each block
    weather_data = []
    for idx, row in df_blocks.iterrows():
        lat, lon = row['LATITUDE'], row['LONGITUDE']
        block = row['BLOCK']

        API_PARAMS = {
            "lat": lat,
            "lon": lon,
            "units": "metric",
            "appid": "419c843a9bee437e8437ad22e1c4288e",  # Replace with your OpenWeatherMap API key
            "cnt": 10,  # Number of forecast entries (max is 40)
        }

        # Fetch real-time 10-day weather forecast data for each block
        response = requests.get(API_ENDPOINT, params=API_PARAMS)
        forecast_data = response.json()

        # Extract the weather forecast data for each block
        for forecast in forecast_data['list']:
            weather_data.append({
                'Block': block,
                'Date': datetime.fromtimestamp(forecast['dt']).strftime('%a %b %d %H:%M'),
                'Weather': forecast['weather'][0]['description'],
                'Temperature': forecast['main']['temp'],
                'Humidity': forecast['main']['humidity'],
                'Wind Speed': forecast['wind']['speed'],
                'Pressure': forecast['main']['pressure'],
            })

    # Create DataFrame for the weather forecast data
    df_forecast = pd.DataFrame(weather_data)
    
    # Filter the GeoDataFrame to include only the desired values of "sdtname"
    filtered_gdf = gdf[gdf['sdtname'].isin(['Gosaba', 'Patharpratima', 'Kultali'])]
    

    # Plot the GeoDataFrame using Plotly Express
    fig1 = px.choropleth_mapbox(filtered_gdf, 
                                geojson=filtered_gdf.geometry, 
                                locations=filtered_gdf.index, 
                                color='sdtname',
                                mapbox_style="open-street-map",
                                center={"lat": filtered_gdf.centroid.y.mean(), "lon": filtered_gdf.centroid.x.mean()},
                                zoom=8,
                                opacity=0.5,
                                hover_name='sdtname',
                
                            )
    fig1.update_geos(fitbounds="locations", visible=False)
    
    # Add polygons to the figure
    for index, row in gdf.iterrows():
        if row.geometry.geom_type == 'Polygon':
            # For single polygons
            x, y = row.geometry.exterior.xy
            fig1.add_trace(go.Scattermapbox(lon=list(x), lat=list(y), mode='lines', fill='none', line=dict(width=2, color='black'), name=row['sdtname'],showlegend=False))
        elif row.geometry.geom_type == 'MultiPolygon':
            # For multipolygons
            for polygon in row.geometry.geoms:
                x, y = polygon.exterior.xy
                fig1.add_trace(go.Scattermapbox(lon=list(x), lat=list(y), mode='lines', fill='none', line=dict(width=2, color='black'), name=row['sdtname'],showlegend=False))
    fig1.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(width=5, color='black'), name="Sundarban Landscape"))
    
    
    # Load the GeoJSON file containing additional polygon data
    with open("https://github.com/RitoshreeS/Sundarban-Dashboard/blob/main/Data/Soil_villages.geojson") as f:
        geo = json.load(f)
        
    # Add polygons from the additional GeoJSON file
    for feature in geo['features']:
        # Extract coordinates for the polygon
        lon = [coord[0] for coord in feature['geometry']['coordinates'][0]]
        lat = [coord[1] for coord in feature['geometry']['coordinates'][0]]

        # Add the polygon to the figure
        fig1.add_trace(go.Scattermapbox(
            lon=lon + [lon[0]],  # Close the polygon by repeating the first coordinate
            lat=lat + [lat[0]],  # Close the polygon by repeating the first coordinate
            mode='lines',
            fill='toself',  # Fill the polygon
            line=dict(color='yellow', width=1),
            hoverinfo='text',  # Show hover text
            opacity=0.7,
            hovertext=feature['properties']['name'],  # Use village name as hover text
            showlegend=False,
        ))
    fig1.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(width=5, color='yellow'), name="Villages"))
    
    # Add weather data as hover text on the polygons
    for idx, row in filtered_gdf.iterrows():
        block_forecast = df_forecast[df_forecast['Block'] == row['sdtname']]
        hover_text = "<b>Date:</b> " + block_forecast['Date'] + "<br>" + \
                    "<b>Weather:</b> " + block_forecast['Weather'] + "<br>" + \
                    "<b>Temperature:</b> " + block_forecast['Temperature'].astype(str) + "°C<br>" + \
                    "<b>Humidity:</b> " + block_forecast['Humidity'].astype(str) + "%<br>" + \
                    "<b>Wind Speed:</b> " + block_forecast['Wind Speed'].astype(str) + " m/s<br>" + \
                    "<b>Pressure:</b> " + block_forecast['Pressure'].astype(str) + " hPa<br>"
        
        fig1.add_trace(go.Scattermapbox(
            lon=[row['geometry'].centroid.x],
            lat=[row['geometry'].centroid.y],
            mode='markers',
            marker=dict(size=5),
            hoverinfo='text',
            text=hover_text,
            hoverlabel=dict(bgcolor='white', font=dict(color='black', size=14)), 
            showlegend=False
        ))
    fig1.update_layout(height=1200,
                    legend=dict(traceorder="normal", title="<b>BLOCKS<b>"),
                    paper_bgcolor='rgba(0,0,0,0.7)',
                    font=dict(
                    color='white',
                    size=16  # Adjust the font size as needed
        ))


    # Create a Plotly figure for plot2
    fig_plot2 = make_subplots(rows=1, cols=1, subplot_titles=("Temperature Data"))

    # Define colors for each block
    colors = {'Gosaba': 'red', 'Kultali': 'blue', 'Patharpratima': 'green'}

    # Add traces for min and max temperature for each block
    for block, color in colors.items():
        block_df = df_weather[df_weather['BLOCK'] == block]
        fig_plot2.add_trace(go.Scatter(x=block_df['DATES'], y=block_df['Temperature-Advisory Data'],
                                    mode='lines', name=f'{block} -<b>Advisory Data<b>', line=dict(color=color)),
                            row=1, col=1)
        fig_plot2.add_trace(go.Scatter(x=block_df['DATES'], y=block_df['Temperature-Realtime Data'],
                                    mode='lines', name=f'{block} -<b>Realtime Data<b>', line=dict(color=color, dash='dash')),
                            row=1, col=1)

        # Update layout for plot2
        fig_plot2.update_layout(title="<b>TEMPERATURE DATA<b>",
                                xaxis_title="Date",
                                yaxis_title="Temperature (°C)",
                                template='plotly_dark',
                                paper_bgcolor='rgba(0,0,0,0.7)',
                                height=550,
                                font=dict(
                                    family="Arial, sans-serif",
                                    size=14,  # Increase font size to 14
                                    color="white"
                                )
                                )

    # Add dropdown menu for block selection
    buttons = []
    for block in colors.keys():
        visible_traces = [i for i, trace in enumerate(fig_plot2.data) if block in trace.name]
        buttons.append(dict(label=block,
                            method='update',
                            args=[{'visible': [True if i in visible_traces else False for i in range(len(fig_plot2.data))]}]))

    fig_plot2.update_layout(updatemenus=[dict(buttons=buttons,
                                            direction="down",
                                            pad={"r": 10, "t": 10},
                                            showactive=True,
                                            x=0.1,
                                            xanchor="left",
                                            y=1.1,
                                            yanchor="top")])

        # Create a Plotly figure for plot3 (Humidity Variation)
    fig_plot3 = make_subplots(rows=1, cols=1, subplot_titles=("Humidity Data"))

    # Define colors for each block
    colors = {'Gosaba': 'red', 'Kultali': 'blue', 'Patharpratima': 'green'}

    # Add traces for min and max humidity for each block
    for block, color in colors.items():
        block_df = df_weather[df_weather['BLOCK'] == block]
        fig_plot3.add_trace(go.Scatter(x=block_df['DATES'], y=block_df['Humidity-Advisory Data'],
                                    mode='lines', name=f'{block} -<b>Advisory Data<b>', line=dict(color=color)),
                            row=1, col=1)
        fig_plot3.add_trace(go.Scatter(x=block_df['DATES'], y=block_df['Humidity- Realtime Data'],
                                    mode='lines', name=f'{block} -<b>Realtime Data<b>', line=dict(color=color, dash='dash')),
                            row=1, col=1)

    # Update layout for plot3
    fig_plot3.update_layout(title="<b>HUMIDITY DATA<b>",
                            xaxis_title="Date",
                            yaxis_title="Humidity (%)",
                            paper_bgcolor='rgba(0,0,0,0.7)',
                            template='plotly_dark',
                            height=550,
                            font=dict(
                                family="Arial, sans-serif",
                                size=14,  # Increase font size to 14
                                color="white"
                            ))

    # Add dropdown menu for block selection
    buttons = []
    for block in colors.keys():
        visible_traces = [i for i, trace in enumerate(fig_plot3.data) if block in trace.name]
        buttons.append(dict(label=block,
                            method='update',
                            args=[{'visible': [True if i in visible_traces else False for i in range(len(fig_plot3.data))]}]))

    fig_plot3.update_layout(updatemenus=[dict(buttons=buttons,
                                            direction="down",
                                            pad={"r": 10, "t": 10},
                                            showactive=True,
                                            x=0.1,
                                            xanchor="left",
                                            y=1.1,
                                            yanchor="top")])

    
    # Create line graph for plot4 (Rainfall Data)
    fig_plot4 = px.line(df_weather, x='DATES', y='PRECIPITATION(mm)', color='BLOCK',
                        title='<b>RAINFALLL DATA<b>', labels={'DATES': 'Date', 'PRECIPITATION(mm)': 'Precipitation (mm)'})

    # Update layout for plot4 to set the background color to black
    fig_plot4.update_layout(
        plot_bgcolor='black',  # Set plot background color
        paper_bgcolor='rgba(0,0,0,0.7)',  # Set paper background color (outside plot area)
        font=dict(color='white', size=15),  # Set font color to white for better visibility on dark background,
        xaxis=dict(gridcolor='rgba(255, 255, 255, 0.3)'),  # Fade x-axis grid color (white with 30% opacity)
        height=550,
        yaxis=dict(gridcolor='rgba(255, 255, 255, 0.3)')  # Fade y-axis grid color (white with 30% opacity)
    )

    # Define the wind direction order
    wind_direction_order = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']

    # Create a Plotly figure for plot5 (Wind rose plot)
    blocks = ['Gosaba', 'Patharpratima', 'Kultali']
    filtered_df = df_weather[df_weather['BLOCK'].isin(blocks)].sort_values(by='STRENGTH')

    # Calculate the color scale based on wind speed with a difference of 3
    max_wind_speed = filtered_df['WIND SPEED(km/h)'].max()
    color_scale = ['#f8f8ff', '#cce6ff', '#99ccff', '#66b3ff', '#3399ff', '#0066cc']

    # Initialize a list to hold traces for each block
    traces = []
    # Create traces for each block with color-coded bars based on wind speed
    for block in blocks:
        block_df = filtered_df[filtered_df['BLOCK'] == block]
        # Sort theta values based on wind_direction_order
        sorted_theta = sorted(block_df['WIND DIRECTION'], key=lambda x: wind_direction_order.index(x))
        trace = go.Barpolar(
            r=block_df['WIND SPEED(km/h)'],
            theta=sorted_theta,  # Use sorted theta values
            name=block,
            customdata=block_df['DATES'],  # Use DATES column as custom data
            hovertemplate='<b>%{fullData.name}</b><br>' +
                        'Date: %{customdata}<br>' +  # Update customdata reference
                        'Wind Speed: %{r} km/h<br>' +
                        'Wind Direction: %{theta}<extra></extra>',
            marker_color=block_df['WIND SPEED(km/h)'],  # Color based on wind speed
            marker=dict(colorscale=color_scale, cmin=0, cmax=max_wind_speed, colorbar=dict(title='Wind Speed(km/h)')),
            hoverlabel=dict(font=dict(size=18))  # Increase font size of hover labels
        )
        traces.append(trace)

    # Create the wind rose figure using the traces
    fig_plot5 = go.Figure(traces)

    # Define the dropdown menu for blocks selection
    dropdown_menu = []
    for trace, block in zip(traces, blocks):
        dropdown_menu.append(dict(
            args=[{"visible": [True if trace == t else False for t in fig_plot5.data]}],
            label=block,
            method="update"
        ))

    # Update layout to include the dropdown menu and color bar
    fig_plot5.update_layout(
        paper_bgcolor='rgba(0,0,0,0.7)',
        updatemenus=[
            dict(
                
                buttons=dropdown_menu,
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.1,
                xanchor="left",
                y=1.1,
                yanchor="top"
            ),
        ],
        polar=dict(
            radialaxis=dict(visible=True, tickvals=[], ticktext=[]),  # Remove default tick values and labels
            angularaxis=dict(showline=False, visible=True)  # Adjust angular axis
        ),
        autosize=False,
        width=1000,  # Adjust width as needed
        height=550,  # Adjust height as needed
        font=dict(size=16),  # Increase font size
        template='plotly_dark',  # Use Plotly's dark theme
        title='<b>WIND SPEED AND DIRECTION<b>',  # Add the title
    )



    return fig1, fig_plot2, fig_plot3, fig_plot4, fig_plot5  # Replace None with other figures as needed

#__________________________________________________________________________________________________________________________________________________________________

# Define callback to update the crop information based on dropdown selection
@app.callback(
    [Output(component_id='crop_calendar', component_property='figure'),
     Output('crop-info', 'children')],
    [Input('crop-dropdown', 'value'),
     Input('month-dropdown', 'value')]
)


def update_crop_info(selected_crop, selected_month):
    # Filter the DataFrame based on selected crop and month
    df_filtered = df_cropdata[(df_cropdata['CROP'] == selected_crop) & (df_cropdata['MONTH'] == selected_month)]
    
    # Create a list of paragraphs containing the information
    crop_info = []
    for index, row in df_filtered.iterrows():
        crop_info.append(html.Div([
        html.B(f"{row['CROP']} ({row['MONTH']})", style={'margin-bottom': '10px'}),
        html.Div([
            html.B("SOWING", style={'color': '#001845'}),
            html.P(row['SOWING'] + '/' + row['SOWING'], style={'font-size': '25px'}),  # Increase font size        
        ], style={'margin-bottom': '20px'}),
        html.Div([
            html.B("SEED/SEEDBED", style={'color': '#001845'}),
            html.P(row['SEED'] + '/' + row['SEEDBED'], style={'font-size': '25px'}),  # Increase font size
        ], style={'margin-bottom': '20px'}),
        html.Div([
            html.B("AGRONOMIC PRACTICES", style={'color': '#001845'}),
            html.P(row['AGRONOMIC PRACTICES'], style={'font-size': '25px'}),  # Increase font size
        ], style={'margin-bottom': '20px'}),
        html.Div([
            html.B("FERTILIZATION:", style={'color': '#001845'}),
            html.P(row['FERTILIZATION'], style={'font-size': '25px'}),  # Increase font size
        ], style={'margin-bottom': '20px'}),
        html.Div([
            html.B("DISEASES", style={'color': '#001845'}),
            html.P(row['DISEASES'], style={'font-size': '25px'}),  # Increase font size
        ], style={'margin-bottom': '20px'}),
        html.Div([
            html.B("PEST CONTROL", style={'color': '#001845'}),
            html.P(row['PEST CONTROL'], style={'font-size': '25px'}),  # Increase font size
        ], style={'margin-bottom': '20px'}),
        html.Div([
            html.B("PRUNING/HARVEST", style={'color': '#001845'}),
            html.P(row['PRUNING/ HARVEST'], style={'font-size': '25px'}),  # Increase font size
        ], style={'margin-bottom': '20px'}),
    ], style={'margin-bottom': '40px'}))

    # Read the CSV file into DataFrame
    df_calendar1 = pd.read_csv('https://github.com/RitoshreeS/Sundarban-Dashboard/blob/main/Data/cropcalendar.csv')

    # Convert START and FINISH columns to datetime objects
    df_calendar1['START'] = pd.to_datetime(df_calendar1['START'])
    df_calendar1['FINISH'] = pd.to_datetime(df_calendar1['FINISH'])

    # Create a Gantt chart using Plotly Express
    fig_cal = px.timeline(df_calendar1, x_start='START', x_end='FINISH', y='CROPS', color='STAGES', labels={'STAGES': 'STAGES'})
    fig_cal.update_yaxes(categoryorder='total ascending')  # Arrange crops in ascending order
    fig_cal.update_layout(title='<b>CROP CALENDAR<b>', xaxis_title='Date', yaxis_title='Crops', height=1000)
    fig_cal.update_xaxes(tickfont=dict(size=14))
    fig_cal.update_yaxes(tickfont=dict(size=14))
    fig_cal.update_layout(legend=dict(font=dict(size=14)))

    # Arrange crops in ascending order
    fig_cal.update_yaxes(categoryorder='total ascending')
    
    
    return fig_cal, crop_info


#__________________________________________________________________________________________________________________________________________________________________________
# Define callback to update the plot based on button click
@app.callback(
    Output(component_id='query_map', component_property='figure'),
    Output(component_id='plot_query', component_property='children'),
    [Input(component_id='wwf-button2', component_property='n_clicks')]
)
def update_graphs(n_clicks):
    
    
    # Load the GeoJSON file using geopandas
    gdf = gpd.read_file("https://github.com/RitoshreeS/Sundarban-Dashboard/blob/main/Data/SB_Landscape_Boundary.shp.geojson")

    # Read the query data
    df_query = pd.read_csv('https://github.com/RitoshreeS/Sundarban-Dashboard/blob/main/Data/QueryData.csv')

    # Filter the GeoDataFrame to include only specific sdtname values
    sdtnames_to_include = ['Gosaba', 'Patharpratima', 'Kultali']
    gdf_filtered = gdf[gdf['sdtname'].isin(sdtnames_to_include)]

    # Plot the filtered GeoDataFrame using Plotly Express
    fig_query = px.choropleth_mapbox(gdf_filtered, 
                                geojson=gdf_filtered.geometry, 
                                locations=gdf_filtered.index, 
                                color='sdtname',  # Color by sdtname
                                mapbox_style="open-street-map",
                                center={"lat": gdf_filtered.centroid.y.mean(), "lon": gdf_filtered.centroid.x.mean()},
                                zoom=8,
                                opacity=0.7,
                                hover_name='sdtname'
                            )

    fig_query.update_geos(fitbounds="locations", visible=False)

    # Increase border width and color for all polygons
    fig_query.update_traces(line=dict(width=6, color='black'), selector=dict(type='scattermapbox', fill='none'))

    # Load the GeoJSON file containing additional polygon data
    with open("D:/wwf india/Website Dashboard/plotly-dash/Polygon/Soil_villages.geojson") as f:
        geo = json.load(f)

    # Add polygons from the additional GeoJSON file
    for feature in geo['features']:
        # Extract coordinates for the polygon
        lon = [coord[0] for coord in feature['geometry']['coordinates'][0]]
        lat = [coord[1] for coord in feature['geometry']['coordinates'][0]]

        # Add the polygon to the figure
        fig_query.add_trace(go.Scattermapbox(
            lon=lon + [lon[0]],  # Close the polygon by repeating the first coordinate
            lat=lat + [lat[0]],  # Close the polygon by repeating the first coordinate
            mode='lines',
            fill='toself',  # Fill the polygon
            line=dict(color='yellow', width=1),
            hoverinfo='text',  # Show hover text
            opacity=1,
            hovertext=feature['properties']['name'],  # Use village name as hover text
            showlegend=False,
        ))

    # Add a dummy legend for the black line
    fig_query.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(width=7, color='black'), name="Sundarban Landscape"))
    fig_query.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(width=7, color='yellow'), name="Villages"))
    fig_query.add_trace(go.Scatter(x=[None], y=[None], mode='markers', line=dict(width=7, color='black'), name="Query"))

    # Add markers to the figure with hover text including BLOCK, QUERY, and REASON
    fig_query.add_trace(go.Scattermapbox(
        lat=df_query['LATITUDE'],
        lon=df_query['LONGITUDE'],
        mode='markers',
        hovertext=df_query.apply(lambda row: f"BLOCK: {row['BLOCK']}<br>QUERY: {row['QUERY']}<br>REASON: {row['REASON']}", axis=1),
        marker=dict(
            size=25,
            color='black',  # Change marker color to black
            opacity=1,
        ),
        hoverinfo='none',
        name='Queries'
    ))
    
     # Add markers to the figure with hover text including BLOCK, QUERY, and REASON
    fig_query.add_trace(go.Scattermapbox(
        lat=df_query['LATITUDE'],
        lon=df_query['LONGITUDE'],
        mode='markers',
        hovertext=df_query.apply(lambda row: f"BLOCK: {row['BLOCK']}<br>QUERY: {row['QUERY']}<br>REASON: {row['REASON']}", axis=1),
        marker=dict(
            size=13,
            color='white',  # Change marker color to black
            opacity=1,
        ),
        hoverinfo='text',
        name='Queries'
    ))
    
    fig_query.update_layout(
    hoverlabel=dict(
        font=dict(
            size=20
        )
      )
    )
    
    
    # Add polygons to the figure
    for index, row in gdf.iterrows():
        if row.geometry.geom_type == 'Polygon':
            # For single polygons
            x, y = row.geometry.exterior.xy
            fig_query.add_trace(go.Scattermapbox(lon=list(x), lat=list(y), mode='lines', fill='none', line=dict(width=1.5, color='black'), name=row['sdtname']))
        elif row.geometry.geom_type == 'MultiPolygon':
            # For multipolygons
            for polygon in row.geometry.geoms:
                x, y = polygon.exterior.xy
                fig_query.add_trace(go.Scattermapbox(lon=list(x), lat=list(y), mode='lines', fill='none', line=dict(width=1.5, color='black'), name=row['sdtname']))
                
    # Customize the layout
    fig_query.update_layout(
        mapbox_style="open-street-map",
        mapbox_zoom=8,
        mapbox_center={"lat": gdf.centroid.y.mean(), "lon": gdf.centroid.x.mean()},
        legend=dict(traceorder="normal", title="<b>BLOCKS<b>"),  # Set the trace order for the legend and change the title
        title_x=0.5,  # Position the title at the center horizontally
        title_y=0.98,  # Position the title at the top vertically
    )

    # Remove legend for Scattermapbox traces
    for item in fig_query.data:
        if isinstance(item, go.Scattermapbox):
            item.showlegend = False

    fig_query.update_layout(width=2100, height=1200)

#_________________________________________________________________________________
    

    # Read the query data
    df_query = pd.read_csv('https://github.com/RitoshreeS/Sundarban-Dashboard/blob/main/Data/QueryData.csv')
    df_query.dropna(subset=['LATITUDE', 'LONGITUDE'], inplace=True)
    
    # Select columns to display in the table (excluding LATITUDE and LONGITUDE)
    df_display = df_query.drop(columns=['LATITUDE', 'LONGITUDE'])

    # Create the table
    table = dcc.Graph(
        figure=go.Figure(
            data=[go.Table(
                header=dict(values=list(df_display.columns),
                            fill_color='#52b69a',
                            align='left'),
                cells=dict(values=[df_display[col] for col in df_display.columns],
                            fill_color='#d8f3dc',
                            align='left'))
            ],
            layout=dict(title='QUERY TABLE', height=1000, font=dict(size=18))
        )
    )
    return fig_query,table
#______________________________________________________________________________________________________________________________________________________________________

# Callback to update the tab content
@app.callback(Output('soil-tabs-content', 'children'),
              [Input('soil-tabs', 'value')])

def render_content(tab):
    
    gdf = gpd.read_file("https://github.com/RitoshreeS/Sundarban-Dashboard/blob/main/Data/SB_Landscape_Boundary.shp.geojson")
        # Read the CSV file containing soil data
    df_soilavg = pd.read_csv('https://github.com/RitoshreeS/Sundarban-Dashboard/blob/main/Data/Soil_avg.csv')

    # Load the GeoJSON file containing polygon data
    with open("https://github.com/RitoshreeS/Sundarban-Dashboard/blob/main/Data/Soil_villages.geojson") as f:
        geo = json.load(f)
    
    # Define colors for each sdtname
    color_mapping = {
        'Patharpratima': 'green',
        'Kultali': 'blue',
        'Gosaba': 'red',
        'Basanti': 'yellow',
        'Hingalganj': 'black'
    }

    
    #PH______________________________________________________________________________________________________________________________
    if tab == 'tab-1':
        # Correcting the line that sets the 'text' column for pH
        df_soilavg['text'] = 'Village: ' + df_soilavg['Village'] + '<br>' +\
                            'GP: ' + df_soilavg['Gram Panchayat'] + '<br>' +\
                            'Block: ' + df_soilavg['BLOCK'] + '<br>' +\
                            'pH: ' + df_soilavg['PH'].astype(str)

        # Define pH range names and colors
        pH_range_names = {
            (0, 5.5): 'Strongly acidic',
            (5.5, 6.5): 'Acidic',
            (6.5, 7.5): 'Neutral',
            (7.5, 30): 'Alkaline'
        }
        pH_colors = ["#9d0208", "#dc2f02", "blue", "#3a86ff"]

        # Create a Plotly figure for pH
        fig_pH = go.Figure()

        # Add dummy traces for all pH range names
        for lim, color in zip(pH_range_names.keys(), pH_colors):
            fig_pH.add_trace(go.Scatter(
                x=[None],
                y=[None],
                mode='markers',
                marker=dict(color=color, opacity=1),  # Invisible marker
                name=pH_range_names[lim]  # Set legend name based on pH range
            ))

        # Add traces for soil data based on pH range
        for lim, color in zip(pH_range_names.keys(), pH_colors):
            # Filter DataFrame based on pH range using boolean indexing
            df_sub = df_soilavg[(df_soilavg['PH'] >= lim[0]) & (df_soilavg['PH'] < lim[1])]
            
            # Filter DataFrame further to include only Bali-I and Bali-II
            df_sub = df_sub[df_sub['Village'].isin(['Bali-I', 'Bali-II'])]
            
            # Calculate bubble size based on the range
            bubble_size = 25
            fig_pH.add_trace(go.Scattermapbox(
                lon=df_sub['Longitude'],
                lat=df_sub['Latitude'],
                text=df_sub['text'],
                marker=dict(
                    size=bubble_size,
                    color=color,
                    opacity=1,
                ),
                name='',  # Empty name to prevent duplicate legends
                showlegend=False  # Exclude these traces from the legend
            ))

        # Add polygons from GeoJSON file to the map for villages other than "Bally"
        for feature in geo['features']:
            village_name = feature['properties']['name']
            if village_name != 'Bally':
                try:
                    pH_value = df_soilavg[df_soilavg['Village'] == village_name]['PH'].values[0]
                    for lim, color in zip(pH_range_names.keys(), pH_colors):
                        if lim[0] <= pH_value < lim[1]:
                            fill_color = color
                            break

                    if feature['geometry']['type'] == 'Polygon':
                        # Extract coordinates for the polygon
                        lon = [coord[0] for coord in feature['geometry']['coordinates'][0]]
                        lat = [coord[1] for coord in feature['geometry']['coordinates'][0]]

                        fig_pH.add_trace(go.Scattermapbox(
                            lon=lon + [lon[0]],  # Close the polygon by repeating the first coordinate
                            lat=lat + [lat[0]],  # Close the polygon by repeating the first coordinate
                            mode='lines',
                            fill='toself',  # Fill the polygon with color
                            fillcolor=fill_color,  # Use trace color for filling
                            line=dict(color='black', width=1),
                            hoverinfo='text',  # Show hover text
                            hovertext=df_soilavg[df_soilavg['Village'] == village_name]['text'].values[0],  # Use df_soilavg['text'] as hover text
                            showlegend=False,
                        ))
                except IndexError:
                    pass  # Skip villages that don't have corresponding data in the DataFrame
        
        # Add polygons to the figure
        for index, row in gdf.iterrows():
            if row.geometry.geom_type == 'Polygon':
                # For single polygons
                x, y = row.geometry.exterior.xy
                fig_pH.add_trace(go.Scattermapbox(lon=list(x), lat=list(y), mode='lines', fill='none', line=dict(width=2, color='black'), name=row['sdtname'],showlegend=False))
            elif row.geometry.geom_type == 'MultiPolygon':
                # For multipolygons
                for polygon in row.geometry.geoms:
                    x, y = polygon.exterior.xy
                    fig_pH.add_trace(go.Scattermapbox(lon=list(x), lat=list(y), mode='lines', fill='none', line=dict(width=2, color='black'), name=row['sdtname'],showlegend=False))
        fig_pH.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(width=5, color='black'), name="Sundarban Landscape"))
        
        # Update layout of the map for pH
        fig_pH.update_layout(
            title_text='<b>Sundarban Soil Data - pH<b>',
            mapbox=dict(
                style="open-street-map",  # Set the Mapbox style
                center=dict(lat=df_soilavg['Latitude'].mean(), lon=df_soilavg['Longitude'].mean()),  # Center the map
                zoom=10,  # Set the initial zoom level
                bearing=0,
                pitch=0,
                layers=[],  # Hide layers control
            ),
            showlegend=True,
            legend=dict(
                title='<b>pH Content<b>',  # Set legend title
                itemsizing='constant',  # Keep legend item sizes constant,
                font=dict(size=16)
            ),
            height=1200,  # Set the height of the map
            width=2150,  # Set the width of the map
        )
        
        return dcc.Graph(figure=fig_pH)
    elif tab == 'tab-2':
        
        #EC____________________________________________________________________________________________________________________________
        # Correcting the line that sets the 'text' column
        df_soilavg['text'] = 'Village: ' + df_soilavg['Village'] + '<br>' +\
                            'GP: ' + df_soilavg['Gram Panchayat'] + '<br>' +\
                            'Block: ' + df_soilavg['BLOCK'] + '<br>' +\
                            'EC1:2 (ds/m): ' + df_soilavg['EC1:2 (ds/m)'].astype(str)

        limits = [(0, 2), (2, 4), (4, 8), (8, 16), (16, 30)]
        colors=["yellow","lightseagreen","blue","orange","crimson"]

        # Define EC range names
        ec_range_names = {
            (0, 2): 'Non Saline',
            (2, 4): 'Very Slightly Saline',
            (4, 8): 'Slightly Saline',
            (8, 16): 'Moderately Saline',
            (16, 30): 'Strongly Alkaline'
        }

        # Create a Plotly figure
        fig_ec = go.Figure()

        # Add dummy traces for all range names
        for lim, color in zip(limits, colors):
            fig_ec.add_trace(go.Scatter(
                x=[None],
                y=[None],
                mode='markers',
                marker=dict(color=color, opacity=1),  # Invisible marker
                name=ec_range_names[lim]  # Set legend name based on EC range
            ))

        # Add traces for soil data based on EC range
        for lim, color in zip(limits, colors):
            # Filter DataFrame based on EC range using boolean indexing
            df_sub = df_soilavg[(df_soilavg['EC1:2 (ds/m)'] >= lim[0]) & (df_soilavg['EC1:2 (ds/m)'] < lim[1])]
            
            # Filter DataFrame further to include only Bali-I and Bali-II
            df_sub = df_sub[df_sub['Village'].isin(['Bali-I', 'Bali-II'])]
            
            # Calculate bubble size based on the range
            bubble_size =25
            fig_ec.add_trace(go.Scattermapbox(
                lon=df_sub['Longitude'],
                lat=df_sub['Latitude'],
                text=df_sub['text'],
                marker=dict(
                    size=bubble_size,
                    color=color,
                    opacity=1,
                ),
                name='',  # Empty name to prevent duplicate legends
                showlegend=False  # Exclude these traces from the legend
            ))

        # Add polygons from GeoJSON file to the map for villages other than "Bally"
        for feature in geo['features']:
            village_name = feature['properties']['name']
            if village_name != 'Bally':
                try:
                    ec_value = df_soilavg[df_soilavg['Village'] == village_name]['EC1:2 (ds/m)'].values[0]
                    for lim, color in zip(limits, colors):
                        if lim[0] <= ec_value < lim[1]:
                            fill_color = color
                            break

                    if feature['geometry']['type'] == 'Polygon':
                        # Extract coordinates for the polygon
                        lon = [coord[0] for coord in feature['geometry']['coordinates'][0]]
                        lat = [coord[1] for coord in feature['geometry']['coordinates'][0]]

                        fig_ec.add_trace(go.Scattermapbox(
                            lon=lon + [lon[0]],  # Close the polygon by repeating the first coordinate
                            lat=lat + [lat[0]],  # Close the polygon by repeating the first coordinate
                            mode='lines',
                            fill='toself',  # Fill the polygon with color
                            fillcolor=fill_color,  # Use trace color for filling
                            line=dict(color='black', width=1),
                            hoverinfo='text',  # Show hover text
                            hovertext=df_soilavg[df_soilavg['Village'] == village_name]['text'].values[0],  # Use df_soilavg['text'] as hover text
                            showlegend=False,
                        ))
                except IndexError:
                    pass  # Skip villages that don't have corresponding data in the DataFrame
        
         # Add polygons to the figure
        for index, row in gdf.iterrows():
            if row.geometry.geom_type == 'Polygon':
                # For single polygons
                x, y = row.geometry.exterior.xy
                fig_ec.add_trace(go.Scattermapbox(lon=list(x), lat=list(y), mode='lines', fill='none', line=dict(width=2, color='black'), name=row['sdtname'],showlegend=False))
            elif row.geometry.geom_type == 'MultiPolygon':
                # For multipolygons
                for polygon in row.geometry.geoms:
                    x, y = polygon.exterior.xy
                    fig_ec.add_trace(go.Scattermapbox(lon=list(x), lat=list(y), mode='lines', fill='none', line=dict(width=2, color='black'), name=row['sdtname'],showlegend=False))
        fig_ec.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(width=5, color='black'), name="Sundarban Landscape"))
        
        
        # Update layout of the map
        fig_ec.update_layout(
            title_text='<b>Sundarban Soil Data - Electrical Conductivity<b>',
            mapbox=dict(
                style="open-street-map",  # Set the Mapbox style
                center=dict(lat=df_soilavg['Latitude'].mean(), lon=df_soilavg['Longitude'].mean()),  # Center the map
                zoom=10,  # Set the initial zoom level
                bearing=0,
                pitch=0,
                layers=[],  # Hide layers control
            ),
            showlegend=True,
            legend=dict(
                title='<b>EC Content<b>',  # Set legend title
                itemsizing='constant',  # Keep legend item sizes constant
                font=dict(size=16)
            ),
            height=1200,  # Set the height of the map
            width=2150,  # Set the width of the map
        )
        
        return dcc.Graph(figure=fig_ec)
    
    elif tab == 'tab-3':
        #OC____________________________________________________________________________________________________________________
        # Correcting the line that sets the 'text' column
        df_soilavg['text'] = 'Village: ' + df_soilavg['Village'] + '<br>' +\
                            'GP: ' + df_soilavg['Gram Panchayat'] + '<br>' +\
                            'Block: ' + df_soilavg['BLOCK'] + '<br>' +\
                            'Organic Carbon (%): ' + df_soilavg['O/C (%)'].astype(str)

        # Define OC concentration ranges
        oc_limits = [(0, 0.5), (0.5, 0.75), (0.75, 30)]
        oc_colors = ["#FFBB5C", "#FF9B50", "#fb5607"]

        # Define OC range names
        oc_range_names = {
            (0, 0.5): 'Organic carbon-Low',
            (0.5, 0.75): 'Organic carbon-Medium',
            (0.75, 30): 'Organic carbon-High'
        }

        # Create a Plotly figure
        fig_oc= go.Figure()

        # Add dummy traces for all range names
        for lim, color in zip(oc_limits, oc_colors):
            fig_oc.add_trace(go.Scatter(
                x=[None],
                y=[None],
                mode='markers',
                marker=dict(color=color, opacity=1),  # Invisible marker
                name=oc_range_names[lim]  # Set legend name based on OC range
            ))

        # Add traces for soil data based on OC range
        for lim, color in zip(oc_limits, oc_colors):
            # Filter DataFrame based on OC range using boolean indexing
            df_sub = df_soilavg[(df_soilavg['O/C (%)'] >= lim[0]) & (df_soilavg['O/C (%)'] < lim[1])]
            
            # Filter DataFrame further to include only Bali-I and Bali-II
            df_sub = df_sub[df_sub['Village'].isin(['Bali-I', 'Bali-II'])]
            
            # Calculate bubble size based on the range
            bubble_size = 25
            fig_oc.add_trace(go.Scattermapbox(
                lon=df_sub['Longitude'],
                lat=df_sub['Latitude'],
                text=df_sub['text'],
                marker=dict(
                    size=bubble_size,
                    color=color,
                    opacity=1,
                ),
                name='',  # Empty name to prevent duplicate legends
                showlegend=False  # Exclude these traces from the legend
            ))

        # Add polygons from GeoJSON file to the map for villages other than "Bally"
        for feature in geo['features']:
            village_name = feature['properties']['name']
            if village_name != 'Bally':
                try:
                    oc_value = df_soilavg[df_soilavg['Village'] == village_name]['O/C (%)'].values[0]
                    for lim, color in zip(oc_limits, oc_colors):
                        if lim[0] <= oc_value < lim[1]:
                            fill_color = color
                            break

                    if feature['geometry']['type'] == 'Polygon':
                        # Extract coordinates for the polygon
                        lon = [coord[0] for coord in feature['geometry']['coordinates'][0]]
                        lat = [coord[1] for coord in feature['geometry']['coordinates'][0]]

                        fig_oc.add_trace(go.Scattermapbox(
                            lon=lon + [lon[0]],  # Close the polygon by repeating the first coordinate
                            lat=lat + [lat[0]],  # Close the polygon by repeating the first coordinate
                            mode='lines',
                            fill='toself',  # Fill the polygon with color
                            fillcolor=fill_color,  # Use trace color for filling
                            line=dict(color='black', width=1),
                            hoverinfo='text',  # Show hover text
                            hovertext=df_soilavg[df_soilavg['Village'] == village_name]['text'].values[0],  # Use df_soilavg['text'] as hover text
                            showlegend=False,
                        ))
                except IndexError:
                    pass  # Skip villages that don't have corresponding data in the DataFrame
        
         # Add polygons to the figure
        for index, row in gdf.iterrows():
            if row.geometry.geom_type == 'Polygon':
                # For single polygons
                x, y = row.geometry.exterior.xy
                fig_oc.add_trace(go.Scattermapbox(lon=list(x), lat=list(y), mode='lines', fill='none', line=dict(width=2, color='black'), name=row['sdtname'],showlegend=False))
            elif row.geometry.geom_type == 'MultiPolygon':
                # For multipolygons
                for polygon in row.geometry.geoms:
                    x, y = polygon.exterior.xy
                    fig_oc.add_trace(go.Scattermapbox(lon=list(x), lat=list(y), mode='lines', fill='none', line=dict(width=2, color='black'), name=row['sdtname'],showlegend=False))
        fig_oc.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(width=5, color='black'), name="Sundarban Landscape"))
        
        
        # Update layout of the map
        fig_oc.update_layout(
            title_text='<b>Sundarban Soil Data - Organic Carbon<b>',
            mapbox=dict(
                style="open-street-map",  # Set the Mapbox style
                center=dict(lat=df_soilavg['Latitude'].mean(), lon=df_soilavg['Longitude'].mean()),  # Center the map
                zoom=10,  # Set the initial zoom level
                bearing=0,
                pitch=0,
                layers=[],  # Hide layers control
            ),
            showlegend=True,
            legend=dict(
                title='<b>OC Content<b>',  # Set legend title
                itemsizing='constant' , # Keep legend item sizes constant
                font=dict(size=16)
            ),
            height=1200,  # Set the height of the map
            width=2150,  # Set the width of the map
        )
        return dcc.Graph(figure=fig_oc)
    
    elif tab == 'tab-4':
        #N____________________________________________________________________________________________________        
        # Correcting the line that sets the 'text' column
        df_soilavg['text'] = 'Village: ' + df_soilavg['Village'] + '<br>' +\
                            'GP: ' + df_soilavg['Gram Panchayat'] + '<br>' +\
                            'Block: ' + df_soilavg['BLOCK'] + '<br>' +\
                            'Nitrogen (kg/ha): ' + df_soilavg['N (kg/ha)'].astype(str)

        # Define nitrogen concentration ranges
        n_limits = [(0, 240), (240, 480), (480, 1000)]
        n_colors = ["#ff758f", "#ff4d6d", "#c9184a"]

        # Define nitrogen range names
        n_range_names = {
            (0, 240): 'Nitrogen-Low',
            (240, 480): 'Nitrogen-Medium',
            (480, 1000): 'Nitrogen-High'
        }

        # Create a Plotly figure
        fig_n = go.Figure()

        # Add dummy traces for all range names
        for lim, color in zip(n_limits, n_colors):
            fig_n.add_trace(go.Scatter(
                x=[None],
                y=[None],
                mode='markers',
                marker=dict(color=color, opacity=1),  # Invisible marker
                name=n_range_names[lim]  # Set legend name based on nitrogen range
            ))

        # Add traces for soil data based on nitrogen range
        for lim, color in zip(n_limits, n_colors):
            # Filter DataFrame based on nitrogen range using boolean indexing
            df_sub = df_soilavg[(df_soilavg['N (kg/ha)'] >= lim[0]) & (df_soilavg['N (kg/ha)'] < lim[1])]
            
            # Filter DataFrame further to include only Bali-I and Bali-II
            df_sub = df_sub[df_sub['Village'].isin(['Bali-I', 'Bali-II'])]
            
            # Calculate bubble size based on the range
            bubble_size = 25
            fig_n.add_trace(go.Scattermapbox(
                lon=df_sub['Longitude'],
                lat=df_sub['Latitude'],
                text=df_sub['text'],
                marker=dict(
                    size=bubble_size,
                    color=color,
                    opacity=1,
                ),
                name='',  # Empty name to prevent duplicate legends
                showlegend=False  # Exclude these traces from the legend
            ))

        # Add polygons from GeoJSON file to the map for villages other than "Bally"
        for feature in geo['features']:
            village_name = feature['properties']['name']
            if village_name != 'Bally':
                try:
                    n_value = df_soilavg[df_soilavg['Village'] == village_name]['N (kg/ha)'].values[0]
                    for lim, color in zip(n_limits, n_colors):
                        if lim[0] <= n_value < lim[1]:
                            fill_color = color
                            break

                    if feature['geometry']['type'] == 'Polygon':
                        # Extract coordinates for the polygon
                        lon = [coord[0] for coord in feature['geometry']['coordinates'][0]]
                        lat = [coord[1] for coord in feature['geometry']['coordinates'][0]]

                        fig_n.add_trace(go.Scattermapbox(
                            lon=lon + [lon[0]],  # Close the polygon by repeating the first coordinate
                            lat=lat + [lat[0]],  # Close the polygon by repeating the first coordinate
                            mode='lines',
                            fill='toself',  # Fill the polygon with color
                            fillcolor=fill_color,  # Use trace color for filling
                            line=dict(color='black', width=1),
                            hoverinfo='text',  # Show hover text
                            hovertext=df_soilavg[df_soilavg['Village'] == village_name]['text'].values[0],  # Use df_soilavg['text'] as hover text
                            showlegend=False,
                        ))
                except IndexError:
                    pass  # Skip villages that don't have corresponding data in the DataFrame

         # Add polygons to the figure
        for index, row in gdf.iterrows():
            if row.geometry.geom_type == 'Polygon':
                # For single polygons
                x, y = row.geometry.exterior.xy
                fig_n.add_trace(go.Scattermapbox(lon=list(x), lat=list(y), mode='lines', fill='none', line=dict(width=2, color='black'), name=row['sdtname'],showlegend=False))
            elif row.geometry.geom_type == 'MultiPolygon':
                # For multipolygons
                for polygon in row.geometry.geoms:
                    x, y = polygon.exterior.xy
                    fig_n.add_trace(go.Scattermapbox(lon=list(x), lat=list(y), mode='lines', fill='none', line=dict(width=2, color='black'), name=row['sdtname'],showlegend=False))
        fig_n.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(width=5, color='black'), name="Sundarban Landscape"))
        

        # Update layout of the map
        fig_n.update_layout(
            title_text='<b>Sundarban Soil Data - Nitrogen<b>',
            mapbox=dict(
                style="open-street-map",  # Set the Mapbox style
                center=dict(lat=df_soilavg['Latitude'].mean(), lon=df_soilavg['Longitude'].mean()),  # Center the map
                zoom=10,  # Set the initial zoom level
                bearing=0,
                pitch=0,
                layers=[],  # Hide layers control
            ),
            showlegend=True,
            legend=dict(
                title='<b>N Content<b>',  # Set legend title
                itemsizing='constant',  # Keep legend item sizes constant
                font=dict(size=16)  # Increase legend font size
            ),
            height=1200,  # Set the height of the map
            width=2150,  # Set the width of the map
        )
        return dcc.Graph(figure=fig_n)
    
    elif tab == 'tab-5':
        #P_______________________________________________________________________________________________________________________
        # Correcting the line that sets the 'text' column
        df_soilavg['text'] = 'Village: ' + df_soilavg['Village'] + '<br>' +\
                            'GP: ' + df_soilavg['Gram Panchayat'] + '<br>' +\
                            'Block: ' + df_soilavg['BLOCK'] + '<br>' +\
                            'Phosphorus (kg/ha): ' + df_soilavg['P (kg/ha)'].astype(str)

        limits = [(0, 11), (11, 22), (22, 200)]
        colors = ["#E2F4C5", "#A8CD9F", "#2b9348"]

        # Define Nitrogen range names
        p_range_names = {
            (0, 11): 'Phosphorus-Low',
            (11, 22): 'Phosphorus-Medium',
            (22, 200): 'Phosphorus-High'
        }

        # Create a Plotly figure
        fig_p = go.Figure()

        # Add dummy traces for all range names
        for lim, color in zip(limits, colors):
            fig_p.add_trace(go.Scatter(
                x=[None],
                y=[None],
                mode='markers',
                marker=dict(color=color, opacity=1),  # Invisible marker
                name=p_range_names[lim]  # Set legend name based on Nitrogen range
            ))

        # Add traces for soil data based on Nitrogen range
        for lim, color in zip(limits, colors):
            # Filter DataFrame based on Nitrogen range using boolean indexing
            df_sub = df_soilavg[(df_soilavg['P (kg/ha)'] >= lim[0]) & (df_soilavg['P (kg/ha)'] < lim[1])]
            
            # Filter DataFrame further to include only Bali-I and Bali-II
            df_sub = df_sub[df_sub['Village'].isin(['Bali-I', 'Bali-II'])]
            
            # Calculate bubble size based on the range
            bubble_size = 100
            fig_p.add_trace(go.Scattermapbox(
                lon=df_sub['Longitude'],
                lat=df_sub['Latitude'],
                text=df_sub['text'],
                marker=dict(
                    size=bubble_size,
                    color=color,
                    opacity=1,
                ),
                name='',  
                showlegend=False  
            ))

        # Add polygons from GeoJSON file to the map for villages other than "Bally"
        for feature in geo['features']:
            village_name = feature['properties']['name']
            if village_name != 'Bally':
                try:
                    p_value = df_soilavg[df_soilavg['Village'] == village_name]['P (kg/ha)'].values[0]
                    for lim, color in zip(limits, colors):
                        if lim[0] <= p_value < lim[1]:
                            fill_color = color
                            break

                    if feature['geometry']['type'] == 'Polygon':
                        # Extract coordinates for the polygon
                        lon = [coord[0] for coord in feature['geometry']['coordinates'][0]]
                        lat = [coord[1] for coord in feature['geometry']['coordinates'][0]]

                        fig_p.add_trace(go.Scattermapbox(
                            lon=lon + [lon[0]],  
                            lat=lat + [lat[0]], 
                            mode='lines',
                            fill='toself',  
                            fillcolor=fill_color,
                            line=dict(color='black', width=1),
                            hoverinfo='text',  
                            hovertext=df_soilavg[df_soilavg['Village'] == village_name]['text'].values[0], 
                            showlegend=False,
                        ))
                except IndexError:
                    pass 
        
         # Add polygons to the figure
        for index, row in gdf.iterrows():
            if row.geometry.geom_type == 'Polygon':
                # For single polygons
                x, y = row.geometry.exterior.xy
                fig_p.add_trace(go.Scattermapbox(lon=list(x), lat=list(y), mode='lines', fill='none', line=dict(width=2, color='black'), name=row['sdtname'],showlegend=False))
            elif row.geometry.geom_type == 'MultiPolygon':
                # For multipolygons
                for polygon in row.geometry.geoms:
                    x, y = polygon.exterior.xy
                    fig_p.add_trace(go.Scattermapbox(lon=list(x), lat=list(y), mode='lines', fill='none', line=dict(width=2, color='black'), name=row['sdtname'],showlegend=False))
        fig_p.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(width=5, color='black'), name="Sundarban Landscape"))
        
        
        # Update layout of the map
        fig_p.update_layout(
            title_text='<b>Sundarban Soil Data - Phosphorus<b>',
            mapbox=dict(
                style="open-street-map",  
                center=dict(lat=df_soilavg['Latitude'].mean(), lon=df_soilavg['Longitude'].mean()),  
                zoom=10,  
                bearing=0,
                pitch=0,
                layers=[],  
            ),
            showlegend=True,
            legend=dict(
                title='<b>Ph Content<b>',  
                itemsizing='constant', 
                font=dict(size=16)  
            ),
            height=1200,
            width=2150,  
)

        return dcc.Graph(figure=fig_p)
    
    elif tab == 'tab-6':
        #K_____________________________________________________________________________________________________________________________________________
        # Correcting the line that sets the 'text' column
        df_soilavg['text'] = 'Village: ' + df_soilavg['Village'] + '<br>' +\
                            'GP: ' + df_soilavg['Gram Panchayat'] + '<br>' +\
                            'Block: ' + df_soilavg['BLOCK'] + '<br>' +\
                            'K (kg/ha): ' + df_soilavg['K (kg/ha)'].astype(str)

        limits = [(0, 110), (110, 280), (280, 1300)]
        k_colors = ["#9290C3", "#535C91", "#1B1A55"]

        # Define Potassium (K) range names
        k_range_names = {
            (0, 110): 'Potassium-Low',
            (110, 280): 'Potassium-Medium',
            (280, 1300): 'Potassium-High'
        }

        # Create a Plotly figure
        fig_k= go.Figure()

        # Add dummy traces for all range names
        for lim, color in zip(limits, k_colors):
            fig_k.add_trace(go.Scatter(
                x=[None],
                y=[None],
                mode='markers',
                marker=dict(color=color, opacity=1),  
                name=k_range_names[lim]  
            ))

        # Add traces for soil data based on Potassium (K) range
        for lim, color in zip(limits, k_colors):
            # Filter DataFrame based on Potassium (K) range using boolean indexing
            df_sub = df_soilavg[(df_soilavg['K (kg/ha)'] >= lim[0]) & (df_soilavg['K (kg/ha)'] < lim[1])]
            
            # Filter DataFrame further to include only Bali-I and Bali-II
            df_sub = df_sub[df_sub['Village'].isin(['Bali-I', 'Bali-II'])]
            
            # Calculate bubble size based on the range
            bubble_size = 25
            fig_k.add_trace(go.Scattermapbox(
                lon=df_sub['Longitude'],
                lat=df_sub['Latitude'],
                text=df_sub['text'],
                marker=dict(
                    size=bubble_size,
                    color=color,
                    opacity=1,
                ),
                name='',  
                showlegend=False  
            ))

        # Add polygons from GeoJSON file to the map for villages other than "Bally"
        for feature in geo['features']:
            village_name = feature['properties']['name']
            if village_name != 'Bally':
                try:
                    k_value = df_soilavg[df_soilavg['Village'] == village_name]['K (kg/ha)'].values[0]
                    for lim, color in zip(limits, k_colors):
                        if lim[0] <= k_value < lim[1]:
                            fill_color = color
                            break

                    if feature['geometry']['type'] == 'Polygon':
                        # Extract coordinates for the polygon
                        lon = [coord[0] for coord in feature['geometry']['coordinates'][0]]
                        lat = [coord[1] for coord in feature['geometry']['coordinates'][0]]

                        fig_k.add_trace(go.Scattermapbox(
                            lon=lon + [lon[0]],  
                            lat=lat + [lat[0]],  
                            mode='lines',
                            fill='toself',  
                            fillcolor=fill_color,  
                            line=dict(color='black', width=1),
                            hoverinfo='text', 
                            hovertext=df_soilavg[df_soilavg['Village'] == village_name]['text'].values[0], 
                            showlegend=False,
                        ))
                except IndexError:
                    pass  # Skip villages that don't have corresponding data in the DataFrame
        
         # Add polygons to the figure
        for index, row in gdf.iterrows():
            if row.geometry.geom_type == 'Polygon':
                # For single polygons
                x, y = row.geometry.exterior.xy
                fig_k.add_trace(go.Scattermapbox(lon=list(x), lat=list(y), mode='lines', fill='none', line=dict(width=2, color='black'), name=row['sdtname'],showlegend=False))
            elif row.geometry.geom_type == 'MultiPolygon':
                # For multipolygons
                for polygon in row.geometry.geoms:
                    x, y = polygon.exterior.xy
                    fig_k.add_trace(go.Scattermapbox(lon=list(x), lat=list(y), mode='lines', fill='none', line=dict(width=2, color='black'), name=row['sdtname'],showlegend=False))
        fig_k.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(width=5, color='black'), name="Sundarban Landscape"))
        
        
        # Update layout of the map
        fig_k.update_layout(
            title_text='<b>Sundarban Soil Data - Potassium<b>',
            mapbox=dict(
                style="open-street-map",  
                center=dict(lat=df_soilavg['Latitude'].mean(), lon=df_soilavg['Longitude'].mean()),  
                zoom=10,  
                bearing=0,
                pitch=0,
                layers=[], 
            ),
            showlegend=True,
            legend=dict(
                title='<b>K Content<b>', 
                itemsizing='constant',  
                font=dict(size=16) \
            ),
            height=1200,  
            width=2150,  
            )
        return dcc.Graph(figure=fig_k)

    elif tab == 'tab-7':
        #Cu_____________________________________________________________________________________________________________________________________--
        # Correcting the line that sets the 'text' column
        df_soilavg['text'] = 'Village: ' + df_soilavg['Village'] + '<br>' +\
                            'GP: ' + df_soilavg['Gram Panchayat'] + '<br>' +\
                            'Block: ' + df_soilavg['BLOCK'] + '<br>' +\
                            'Copper (ppm): ' + df_soilavg['Cu (ppm)'].astype(str)

        limits = [(0, 1), (1, 50), (50, 100)]
        colors = ["#ffea00", "#ffd000", "#ff8800"]

        # Define Cu range names
        cu_range_names = {
            (0, 1): 'Copper-Low',
            (1, 50): 'Copper-Permissible limit',
            (50, 100): 'Copper-High'
        }

        # Create a Plotly figure
        fig_cu = go.Figure()

        # Add dummy traces for all range names
        for lim, color in zip(limits, colors):
            fig_cu.add_trace(go.Scatter(
                x=[None],
                y=[None],
                mode='markers',
                marker=dict(color=color, opacity=1),  
                name=cu_range_names[lim]  
            ))

        # Add traces for soil data based on Cu range
        for lim, color in zip(limits, colors):
            # Filter DataFrame based on Cu range using boolean indexing
            df_sub = df_soilavg[(df_soilavg['Cu (ppm)'] >= lim[0]) & (df_soilavg['Cu (ppm)'] < lim[1])]
            
            # Filter DataFrame further to include only Bali-I and Bali-II
            df_sub = df_sub[df_sub['Village'].isin(['Bali-I', 'Bali-II'])]
            
            # Calculate bubble size based on the range
            bubble_size = 25
            fig_cu.add_trace(go.Scattermapbox(
                lon=df_sub['Longitude'],
                lat=df_sub['Latitude'],
                text=df_sub['text'],
                marker=dict(
                    size=bubble_size,
                    color=color,
                    opacity=1,
                ),
                name='',  
                showlegend=False  
            ))

        # Add polygons from GeoJSON file to the map for villages other than "Bally"
        for feature in geo['features']:
            village_name = feature['properties']['name']
            if village_name != 'Bally':
                try:
                    cu_value = df_soilavg[df_soilavg['Village'] == village_name]['Cu (ppm)'].values[0]
                    for lim, color in zip(limits, colors):
                        if lim[0] <= cu_value < lim[1]:
                            fill_color = color
                            break

                    if feature['geometry']['type'] == 'Polygon':
                        # Extract coordinates for the polygon
                        lon = [coord[0] for coord in feature['geometry']['coordinates'][0]]
                        lat = [coord[1] for coord in feature['geometry']['coordinates'][0]]

                        fig_cu.add_trace(go.Scattermapbox(
                            lon=lon + [lon[0]],  
                            lat=lat + [lat[0]],  
                            mode='lines',
                            fill='toself', 
                            fillcolor=fill_color,  
                            line=dict(color='black', width=1),
                            hoverinfo='text',  
                            hovertext=df_soilavg[df_soilavg['Village'] == village_name]['text'].values[0], 
                            showlegend=False,
                        ))
                except IndexError:
                    pass  
        
         # Add polygons to the figure
        for index, row in gdf.iterrows():
            if row.geometry.geom_type == 'Polygon':
                # For single polygons
                x, y = row.geometry.exterior.xy
                fig_cu.add_trace(go.Scattermapbox(lon=list(x), lat=list(y), mode='lines', fill='none', line=dict(width=2, color='black'), name=row['sdtname'],showlegend=False))
            elif row.geometry.geom_type == 'MultiPolygon':
                # For multipolygons
                for polygon in row.geometry.geoms:
                    x, y = polygon.exterior.xy
                    fig_cu.add_trace(go.Scattermapbox(lon=list(x), lat=list(y), mode='lines', fill='none', line=dict(width=2, color='black'), name=row['sdtname'],showlegend=False))
        fig_cu.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(width=5, color='black'), name="Sundarban Landscape"))
        
        
        # Update layout of the map
        fig_cu.update_layout(
            title_text='<b>Sundarban Soil Data - Copper<b>',
            mapbox=dict(
                style="open-street-map",
                center=dict(lat=df_soilavg['Latitude'].mean(), lon=df_soilavg['Longitude'].mean()),
                zoom=10,
                bearing=0,
                pitch=0,
                layers=[], 
            ),
            showlegend=True,
            legend=dict(
                title='<b>Cu Content<b>', 
                itemsizing='constant',  
                font=dict(size=16)
            ),
            height=1200, 
            width=2150,
        )

        return dcc.Graph(figure=fig_cu)
    
    elif tab == 'tab-8':
        #Zn____________________________________________________________________________________________________________________________________
        # Correcting the line that sets the 'text' column
        df_soilavg['text'] = 'Village: ' + df_soilavg['Village'] + '<br>' +\
                            'GP: ' + df_soilavg['Gram Panchayat'] + '<br>' +\
                            'Block: ' + df_soilavg['BLOCK'] + '<br>' +\
                            'Zinc (ppm): ' + df_soilavg['Zn (ppm)'].astype(str)

        limits = [(0, 10), (10, 300), (300, 400)]
        zn_colors = ["#F5CCA0", "#E48F45", "#994D1C"]

        # Define Zinc range names
        zn_range_names = {
            (0, 10): 'Zinc-Low',
            (10, 300): 'Zinc-Permissible limit',
            (300, 400): 'Zinc-High'
        }

        # Create a Plotly figure for Zinc
        fig_zn = go.Figure()

        # Add dummy traces for all range names
        for lim, color in zip(limits, zn_colors):
            fig_zn.add_trace(go.Scatter(
                x=[None],
                y=[None],
                mode='markers',
                marker=dict(color=color, opacity=1),
                name=zn_range_names[lim]  
            ))

        # Add traces for soil data based on Zinc range
        for lim, color in zip(limits, zn_colors):
            # Filter DataFrame based on Zinc range using boolean indexing
            df_sub = df_soilavg[(df_soilavg['Zn (ppm)'] >= lim[0]) & (df_soilavg['Zn (ppm)'] < lim[1])]
            
            # Filter DataFrame further to include only Bali-I and Bali-II
            df_sub = df_sub[df_sub['Village'].isin(['Bali-I', 'Bali-II'])]
            
            # Calculate bubble size based on the range
            bubble_size = 25
            fig_zn.add_trace(go.Scattermapbox(
                lon=df_sub['Longitude'],
                lat=df_sub['Latitude'],
                text=df_sub['text'],
                marker=dict(
                    size=bubble_size,
                    color=color,
                    opacity=1,
                ),
                name='',  
                showlegend=False  
            ))

        # Add polygons from GeoJSON file to the map for villages other than "Bally'
        for feature in geo['features']:
            village_name = feature['properties']['name']
            if village_name != 'Bally':
                try:
                    zn_value = df_soilavg[df_soilavg['Village'] == village_name]['Zn (ppm)'].values[0]
                    for lim, color in zip(limits, zn_colors):
                        if lim[0] <= zn_value < lim[1]:
                            fill_color = color
                            break

                    if feature['geometry']['type'] == 'Polygon':
                        # Extract coordinates for the polygon
                        lon = [coord[0] for coord in feature['geometry']['coordinates'][0]]
                        lat = [coord[1] for coord in feature['geometry']['coordinates'][0]]

                        fig_zn.add_trace(go.Scattermapbox(
                            lon=lon + [lon[0]],  
                            lat=lat + [lat[0]],  
                            mode='lines',
                            fill='toself',  
                            fillcolor=fill_color, 
                            line=dict(color='black', width=1),
                            hoverinfo='text',
                            hovertext=df_soilavg[df_soilavg['Village'] == village_name]['text'].values[0],  
                            showlegend=False,
                        ))
                except IndexError:
                    pass  
        
         # Add polygons to the figure
        for index, row in gdf.iterrows():
            if row.geometry.geom_type == 'Polygon':
                # For single polygons
                x, y = row.geometry.exterior.xy
                fig_zn.add_trace(go.Scattermapbox(lon=list(x), lat=list(y), mode='lines', fill='none', line=dict(width=2, color='black'), name=row['sdtname'],showlegend=False))
            elif row.geometry.geom_type == 'MultiPolygon':
                # For multipolygons
                for polygon in row.geometry.geoms:
                    x, y = polygon.exterior.xy
                    fig_zn.add_trace(go.Scattermapbox(lon=list(x), lat=list(y), mode='lines', fill='none', line=dict(width=2, color='black'), name=row['sdtname'],showlegend=False))
        fig_zn.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(width=5, color='black'), name="Sundarban Landscape"))
        
        
        # Update layout of the map for Zinc
        fig_zn.update_layout(
            title_text='<b>Sundarban Soil Data - Zinc<b>',
            mapbox=dict(
                style="open-street-map",  
                center=dict(lat=df_soilavg['Latitude'].mean(), lon=df_soilavg['Longitude'].mean()), 
                zoom=10,  
                bearing=0,
                pitch=0,
                layers=[],  
            ),
            showlegend=True,
            legend=dict(
                title='<b>Zn Content<b>',  
                itemsizing='constant',  
                font=dict(size=16)  
            ),
            height=1200,  
            width=2150,  
        )
        return dcc.Graph(figure=fig_zn)
    
    elif tab == 'tab-9':
        #Fe________________________________________________________________________________________________________________________________-
        # Correcting the line that sets the 'text' column
        df_soilavg['text'] = 'Village: ' + df_soilavg['Village'] + '<br>' +\
                            'GP: ' + df_soilavg['Gram Panchayat'] + '<br>' +\
                            'Block: ' + df_soilavg['BLOCK'] + '<br>' +\
                            'Iron (ppm): ' + df_soilavg['Fe (ppm)'].astype(str)

        fe_limits = [(0, 50), (50, 100), (100, 400)]
        fe_colors = ["#6fffe9", "#5bc0be", "#3a506b"]

        # Define Iron range names
        fe_range_names = {
            (0, 50): 'Iron-Low',
            (50, 100): 'Iron-Permissible limit',
            (100, 400): 'Iron-High'
        }

        # Create a Plotly figure for Iron
        fig_fe = go.Figure()

        # Add dummy traces for all range names
        for lim, color in zip(fe_limits, fe_colors):
            fig_fe.add_trace(go.Scatter(
                x=[None],
                y=[None],
                mode='markers',
                marker=dict(color=color, opacity=1), 
                name=fe_range_names[lim]  
            ))

        # Add traces for soil data based on Iron range
        for lim, color in zip(fe_limits, fe_colors):
            # Filter DataFrame based on Iron range using boolean indexing
            df_sub = df_soilavg[(df_soilavg['Fe (ppm)'] >= lim[0]) & (df_soilavg['Fe (ppm)'] < lim[1])]
            
            # Filter DataFrame further to include only Bali-I and Bali-II
            df_sub = df_sub[df_sub['Village'].isin(['Bali-I', 'Bali-II'])]
            
            # Calculate bubble size based on the range
            bubble_size = 25
            fig_fe.add_trace(go.Scattermapbox(
                lon=df_sub['Longitude'],
                lat=df_sub['Latitude'],
                text=df_sub['text'],
                marker=dict(
                    size=bubble_size,
                    color=color,
                    opacity=1,
                ),
                name='',  
                showlegend=False  
            ))

        # Add polygons from GeoJSON file to the map for villages other than "Bally"
        for feature in geo['features']:
            village_name = feature['properties']['name']
            if village_name != 'Bally':
                try:
                    fe_value = df_soilavg[df_soilavg['Village'] == village_name]['Fe (ppm)'].values[0]
                    for lim, color in zip(fe_limits, fe_colors):
                        if lim[0] <= fe_value < lim[1]:
                            fill_color = color
                            break

                    if feature['geometry']['type'] == 'Polygon':
                        # Extract coordinates for the polygon
                        lon = [coord[0] for coord in feature['geometry']['coordinates'][0]]
                        lat = [coord[1] for coord in feature['geometry']['coordinates'][0]]

                        fig_fe.add_trace(go.Scattermapbox(
                            lon=lon + [lon[0]],  
                            lat=lat + [lat[0]],  
                            mode='lines',
                            fill='toself',  
                            fillcolor=fill_color,  
                            line=dict(color='black', width=1),
                            hoverinfo='text',  
                            hovertext=df_soilavg[df_soilavg['Village'] == village_name]['text'].values[0],  
                            showlegend=False,
                        ))
                except IndexError:
                    pass  # Skip villages that don't have corresponding data in the DataFrame

         # Add polygons to the figure
        for index, row in gdf.iterrows():
            if row.geometry.geom_type == 'Polygon':
                # For single polygons
                x, y = row.geometry.exterior.xy
                fig_fe.add_trace(go.Scattermapbox(lon=list(x), lat=list(y), mode='lines', fill='none', line=dict(width=2, color='black'), name=row['sdtname'],showlegend=False))
            elif row.geometry.geom_type == 'MultiPolygon':
                # For multipolygons
                for polygon in row.geometry.geoms:
                    x, y = polygon.exterior.xy
                    fig_fe.add_trace(go.Scattermapbox(lon=list(x), lat=list(y), mode='lines', fill='none', line=dict(width=2, color='black'), name=row['sdtname'],showlegend=False))
        fig_fe.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(width=5, color='black'), name="Sundarban Landscape"))
        

        # Update layout of the map for Iron
        fig_fe.update_layout(
            title_text='<b>Sundarban Soil Data - Iron<b>',
            mapbox=dict(
                style="open-street-map", 
                center=dict(lat=df_soilavg['Latitude'].mean(), lon=df_soilavg['Longitude'].mean()),  
                zoom=10,
                bearing=0,
                pitch=0,
                layers=[],  
            ),
            showlegend=True,
            legend=dict(
                title='<b>Fe Content<b>',
                itemsizing='constant',  
                font=dict(size=16)  
            ),
            height=1200,  
            width=2150,
        )
        return dcc.Graph(figure=fig_fe)
    
    elif tab == 'tab-10':
        #Mn__________________________________________________________________________________________________________________________________________
        # Correcting the line that sets the 'text' column
        df_soilavg['text'] = 'Village: ' + df_soilavg['Village'] + '<br>' +\
                            'GP: ' + df_soilavg['Gram Panchayat'] + '<br>' +\
                            'Block: ' + df_soilavg['BLOCK'] + '<br>' +\
                            'Manganese (ppm): ' + df_soilavg['Mn (ppm)'].astype(str)

        limits = [(0, 40), (40, 60), (60, 100)]
        mn_colors = ["#FFBB5C", "#E25E3E", "#f35b04"]

        # Define Manganese range names
        mn_range_names = {
            (0, 40): 'Manganese-Low',
            (40, 60): 'Manganese-Permissible limit',
            (60, 100): 'Manganese-High'
        }

        # Create a Plotly figure for Manganese
        fig_mn = go.Figure()

        # Add dummy traces for all range names
        for lim, color in zip(limits, mn_colors):
            fig_mn.add_trace(go.Scatter(
                x=[None],
                y=[None],
                mode='markers',
                marker=dict(color=color, opacity=1), 
                name=mn_range_names[lim]  
            ))

        # Add traces for soil data based on Manganese range
        for lim, color in zip(limits, mn_colors):
            # Filter DataFrame based on Manganese range using boolean indexing
            df_sub = df_soilavg[(df_soilavg['Mn (ppm)'] >= lim[0]) & (df_soilavg['Mn (ppm)'] < lim[1])]
            
            # Filter DataFrame further to include only Bali-I and Bali-II
            df_sub = df_sub[df_sub['Village'].isin(['Bali-I', 'Bali-II'])]
            
            # Calculate bubble size based on the range
            bubble_size = 25
            fig_mn.add_trace(go.Scattermapbox(
                lon=df_sub['Longitude'],
                lat=df_sub['Latitude'],
                text=df_sub['text'],
                marker=dict(
                    size=bubble_size,
                    color=color,
                    opacity=1,
                ),
                showlegend=False 
            ))

        # Add polygons from GeoJSON file to the map for villages other than "Bally"
        for feature in geo['features']:
            village_name = feature['properties']['name']
            if village_name != 'Bally':
                try:
                    mn_value = df_soilavg[df_soilavg['Village'] == village_name]['Mn (ppm)'].values[0]
                    for lim, color in zip(limits, mn_colors):
                        if lim[0] <= mn_value < lim[1]:
                            fill_color = color
                            break

                    if feature['geometry']['type'] == 'Polygon':
                        # Extract coordinates for the polygon
                        lon = [coord[0] for coord in feature['geometry']['coordinates'][0]]
                        lat = [coord[1] for coord in feature['geometry']['coordinates'][0]]

                        fig_mn.add_trace(go.Scattermapbox(
                            lon=lon + [lon[0]],  
                            lat=lat + [lat[0]], 
                            mode='lines',
                            fill='toself',  
                            fillcolor=fill_color,  
                            line=dict(color='black', width=1),
                            hoverinfo='text',
                            hovertext=df_soilavg[df_soilavg['Village'] == village_name]['text'].values[0],  
                            showlegend=False,
                        ))
                except IndexError:
                    pass
        
         # Add polygons to the figure
        for index, row in gdf.iterrows():
            if row.geometry.geom_type == 'Polygon':
                # For single polygons
                x, y = row.geometry.exterior.xy
                fig_mn.add_trace(go.Scattermapbox(lon=list(x), lat=list(y), mode='lines', fill='none', line=dict(width=2, color='black'), name=row['sdtname'],showlegend=False))
            elif row.geometry.geom_type == 'MultiPolygon':
                # For multipolygons
                for polygon in row.geometry.geoms:
                    x, y = polygon.exterior.xy
                    fig_mn.add_trace(go.Scattermapbox(lon=list(x), lat=list(y), mode='lines', fill='none', line=dict(width=2, color='black'), name=row['sdtname'],showlegend=False))
        fig_mn.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(width=5, color='black'), name="Sundarban Landscape"))
        
        
        # Update layout of the map for Manganese
        fig_mn.update_layout(
            title_text='<b>Sundarban Soil Data - Manganese<b>',
            mapbox=dict(
                style="open-street-map",  
                center=dict(lat=df_soilavg['Latitude'].mean(), lon=df_soilavg['Longitude'].mean()),  
                zoom=10,  
                bearing=0,
                pitch=0,
                layers=[],  
            ),
            showlegend=True,
            legend=dict(
                title='<b>Mn Content<b>',  
                itemsizing='constant',
            font=dict(size=16) 
            ),
            height=1200,  
            width=2150,
        )
        return dcc.Graph(figure=fig_mn)
    
    elif tab == 'tab-11':
        #B________________________________________________________________________________________________________________________________________
        
        # Correcting the line that sets the 'text' column
        df_soilavg['text'] = 'Village: ' + df_soilavg['Village'] + '<br>' +\
                            'GP: ' + df_soilavg['Gram Panchayat'] + '<br>' +\
                            'Block: ' + df_soilavg['BLOCK'] + '<br>' +\
                            'Boron (ppm): ' + df_soilavg['B (ppm)'].astype(str)

        limits = [(0, 0.5), (0.5, 1), (1, 10)]
        b_colors = ["green", "blue", "#f35b04"]

        # Define Boron range names
        b_range_names = {
            (0, 0.5): 'Boron-Deficient',
            (0.5, 1): 'Boron-Sufficient',
            (1, 10): 'Boron-Toxic'
        }

        # Create a Plotly figure for Boron
        fig_b = go.Figure()

        # Add dummy traces for all range names
        for lim, color in zip(limits, b_colors):
            fig_b.add_trace(go.Scatter(
                x=[None],
                y=[None],
                mode='markers',
                marker=dict(color=color,),
                name=b_range_names[lim]  
            ))

        # Add traces for soil data based on Boron range
        for lim, color in zip(limits, b_colors):
            # Filter DataFrame based on Boron range using boolean indexing
            df_sub = df_soilavg[(df_soilavg['B (ppm)'] >= lim[0]) & (df_soilavg['B (ppm)'] < lim[1])]
            
            # Filter DataFrame further to include only Bali-I and Bali-II
            df_sub = df_sub[df_sub['Village'].isin(['Bali-I', 'Bali-II'])]
            
            # Calculate bubble size based on the range
            bubble_size = 25
            fig_b.add_trace(go.Scattermapbox(
                lon=df_sub['Longitude'],
                lat=df_sub['Latitude'],
                text=df_sub['text'],
                marker=dict(
                    size=bubble_size,
                    color=color,
                    opacity=1,
                ),
                name='',  
                showlegend=False 
            ))

        # Add polygons from GeoJSON file to the map for villages other than "Bally"
        for feature in geo['features']:
            village_name = feature['properties']['name']
            if village_name != 'Bally':
                try:
                    b_value = df_soilavg[df_soilavg['Village'] == village_name]['B (ppm)'].values[0]
                    for lim, color in zip(limits, b_colors):
                        if lim[0] <= b_value < lim[1]:
                            fill_color = color
                            break

                    if feature['geometry']['type'] == 'Polygon':
                        # Extract coordinates for the polygon
                        lon = [coord[0] for coord in feature['geometry']['coordinates'][0]]
                        lat = [coord[1] for coord in feature['geometry']['coordinates'][0]]

                        fig_b.add_trace(go.Scattermapbox(
                            lon=lon + [lon[0]],  
                            lat=lat + [lat[0]],
                            mode='lines',
                            fill='toself',  
                            fillcolor=fill_color,  
                            line=dict(color='black', width=1),
                            hoverinfo='text', 
                            hovertext=df_soilavg[df_soilavg['Village'] == village_name]['text'].values[0],  
                            showlegend=False,
                        ))
                except IndexError:
                    pass  # Skip villages that don't have corresponding data in the DataFrame
        
         # Add polygons to the figure
        for index, row in gdf.iterrows():
            if row.geometry.geom_type == 'Polygon':
                # For single polygons
                x, y = row.geometry.exterior.xy
                fig_b.add_trace(go.Scattermapbox(lon=list(x), lat=list(y), mode='lines', fill='none', line=dict(width=2, color='black'), name=row['sdtname'],showlegend=False))
            elif row.geometry.geom_type == 'MultiPolygon':
                # For multipolygons
                for polygon in row.geometry.geoms:
                    x, y = polygon.exterior.xy
                    fig_b.add_trace(go.Scattermapbox(lon=list(x), lat=list(y), mode='lines', fill='none', line=dict(width=2, color='black'), name=row['sdtname'],showlegend=False))
        fig_b.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(width=5, color='black'), name="Sundarban Landscape"))
        
        
        # Update layout of the map for Boron
        fig_b.update_layout(
            title_text='<b>Sundarban Soil Data - Boron<b>',
            mapbox=dict(
                style="open-street-map",  
                center=dict(lat=df_soilavg['Latitude'].mean(), lon=df_soilavg['Longitude'].mean()), 
                zoom=10,  
                bearing=0,
                pitch=0,
                layers=[], 
            ),
            showlegend=True,
            legend=dict(
                title='<b>B Content<b>',  
                itemsizing='constant',  
                font=dict(size=16) 
            ),
            height=1200,
            width=2150,  
        )
        return dcc.Graph(figure=fig_b)
    
    elif tab == 'tab-12':
        #S____________________________________________________________________________________________________________________________________
        # Create a Plotly figure for Sulphur
        fig_s = go.Figure()
        
        # Correcting the line that sets the 'text' column
        df_soilavg['text'] = 'Village: ' + df_soilavg['Village'] + '<br>' +\
                            'GP: ' + df_soilavg['Gram Panchayat'] + '<br>' +\
                            'Block: ' + df_soilavg['BLOCK'] + '<br>' +\
                            'Sulphur (ppm): ' + df_soilavg['S (ppm)'].astype(str)


        # Define Sulphur range limits and colors
        s_limits = [(0, 15), (15, 60)]
        s_colors = ["crimson", "blue"]

        # Define Sulphur range names
        s_range_names = {
            (0, 15): 'Sulphur-Deficient',
            (15, 60): 'Sulphur-Good'
        }

        # Add dummy traces for all range names
        for lim, color in zip(s_limits, s_colors):
            fig_s.add_trace(go.Scatter(
                x=[None],
                y=[None],
                mode='markers',
                marker=dict(color=color, opacity=1),  
                name=s_range_names[lim]  
            ))

        # Add traces for soil data based on Sulphur range
        for lim, color in zip(s_limits, s_colors):
            # Filter DataFrame based on Sulphur range using boolean indexing
            df_sub = df_soilavg[(df_soilavg['S (ppm)'] >= lim[0]) & (df_soilavg['S (ppm)'] < lim[1])]
            
            # Filter DataFrame further to include only Bali-I and Bali-II
            df_sub = df_sub[df_sub['Village'].isin(['Bali-I', 'Bali-II'])]
            
            # Calculate bubble size based on the range
            bubble_size = 25
            fig_s.add_trace(go.Scattermapbox(
                lon=df_sub['Longitude'],
                lat=df_sub['Latitude'],
                text=df_sub['text'],
                marker=dict(
                    size=bubble_size,
                    color=color,
                    opacity=1,
                ),
                name='', 
                showlegend=False
            ))

        # Add polygons from GeoJSON file to the map for villages other than "Bally"
        for feature in geo['features']:
            village_name = feature['properties']['name']
            if village_name != 'Bally':
                try:
                    s_value = df_soilavg[df_soilavg['Village'] == village_name]['S (ppm)'].values[0]
                    for lim, color in zip(s_limits, s_colors):
                        if lim[0] <= s_value < lim[1]:
                            fill_color = color
                            break

                    if feature['geometry']['type'] == 'Polygon':
                        
                        lon = [coord[0] for coord in feature['geometry']['coordinates'][0]]
                        lat = [coord[1] for coord in feature['geometry']['coordinates'][0]]

                        fig_s.add_trace(go.Scattermapbox(
                            lon=lon + [lon[0]],  
                            lat=lat + [lat[0]],  
                            mode='lines',
                            fill='toself',
                            fillcolor=fill_color,  
                            line=dict(color='black', width=1),
                            hoverinfo='text',  
                            hovertext=df_soilavg[df_soilavg['Village'] == village_name]['text'].values[0],  
                            showlegend=False,
                        ))
                except IndexError:
                    pass 
        
         # Add polygons to the figure
        for index, row in gdf.iterrows():
            if row.geometry.geom_type == 'Polygon':
                # For single polygons
                x, y = row.geometry.exterior.xy
                fig_s.add_trace(go.Scattermapbox(lon=list(x), lat=list(y), mode='lines', fill='none', line=dict(width=2, color='black'), name=row['sdtname'],showlegend=False))
            elif row.geometry.geom_type == 'MultiPolygon':
                # For multipolygons
                for polygon in row.geometry.geoms:
                    x, y = polygon.exterior.xy
                    fig_s.add_trace(go.Scattermapbox(lon=list(x), lat=list(y), mode='lines', fill='none', line=dict(width=2, color='black'), name=row['sdtname'],showlegend=False))
        fig_s.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(width=5, color='black'), name="Sundarban Landscape"))
        
        
        # Update layout of the map for Sulphur
        fig_s.update_layout(
            title_text='<b>Sundarban Soil Data - Sulphur <b>',
            mapbox=dict(
                style="open-street-map",
                center=dict(lat=df_soilavg['Latitude'].mean(), lon=df_soilavg['Longitude'].mean()),  
                zoom=10,  
                bearing=0,
                pitch=0,
                layers=[],  
                
            ),
            showlegend=True,
            legend=dict(
                title='<b>S Content<b>', 
                itemsizing='constant' ,  
                font=dict(size=16)  
            ),
            height=1200, 
            width=2150,  
        )
        return dcc.Graph(figure=fig_s)
    
    elif tab == 'tab-13':
        #Soil____________________________________________________________________________________________________________________________-
    
        df_soilavg['text'] = 'Village: ' + df_soilavg['Village'] + '<br>' +\
                            'GP: ' + df_soilavg['Gram Panchayat'] + '<br>' +\
                            'Block: ' + df_soilavg['BLOCK'] + '<br>' +\
                            'Soil Type: ' + df_soilavg['Soil Type'].astype(str)

        # Define colors for different soil types
        soil_colors = {
            'Clay': 'rgb(255,0,51)',          
            'Sandy Clay': 'rgb(255,92,66)',   
            'Clay Loam': 'rgb(255,153,102)',  
            'Silty Clay': 'rgb(255,204,153)',  
            'Silty Clay Loam': 'rgb(255,255,153)', 
            'Sandy Clay Loam': 'rgb(255,255,204)', 
            'Sandy Loam': 'rgb(204,255,153)',  
            'Loam': 'rgb(153,255,102)',       
            'Silt Loam': 'rgb(51,204,102)',    
            'Silt': 'rgb(0,153,76)',          
            'Loamy Sand': 'rgb(0,102,51)',     
            'Sand': 'rgb(0,76,61)'             
        }

        # Create a Plotly figure for Soil Type
        fig_soil_type = go.Figure()

        # Add dummy traces for all soil types
        for soil_type, color in soil_colors.items():
            fig_soil_type.add_trace(go.Scatter(
                x=[None],
                y=[None],
                mode='markers',
                marker=dict(color=color, opacity=1,), 
                name=soil_type 
            ))

        # Add traces for soil data based on Soil Type
        for soil_type, color in soil_colors.items():
            # Filter DataFrame based on Soil Type
            df_sub = df_soilavg[df_soilavg['Soil Type'] == soil_type]
                
            # Filter DataFrame further to include only Bali-I and Bali-II
            df_sub = df_sub[df_sub['Village'].isin(['Bali-I', 'Bali-II'])]
            
            # Calculate bubble size
            bubble_size = 25
            fig_soil_type.add_trace(go.Scattermapbox(
                lon=df_sub['Longitude'],
                lat=df_sub['Latitude'],
                text=df_sub['text'],
                marker=dict(
                    size=bubble_size,
                    color=color,
                    opacity=1,
                ),
                name=soil_type, 
                showlegend=False 
            ))

        # Add polygons from GeoJSON file to the map
        for feature in geo['features']:
            village_name = feature['properties']['name']
            if village_name != 'Bally':
                try:
                    soil_type = df_soilavg[df_soilavg['Village'] == village_name]['Soil Type'].values[0]
                    fill_color = soil_colors.get(soil_type, 'rgb(211,211,211)')

                    if feature['geometry']['type'] == 'Polygon':
               
                        lon = [coord[0] for coord in feature['geometry']['coordinates'][0]]
                        lat = [coord[1] for coord in feature['geometry']['coordinates'][0]]

                        fig_soil_type.add_trace(go.Scattermapbox(
                            lon=lon + [lon[0]], 
                            lat=lat + [lat[0]], 
                            mode='lines',
                            fill='toself', 
                            fillcolor=fill_color, 
                            line=dict(color='black', width=1),
                            hoverinfo='text',  
                            showlegend=False,
                        ))
                except IndexError:
                    pass  

        
        for index, row in gdf.iterrows():
            if row.geometry.geom_type == 'Polygon':
                # For single polygons
                x, y = row.geometry.exterior.xy
                fig_soil_type.add_trace(go.Scattermapbox(lon=list(x), lat=list(y), mode='lines', fill='none', line=dict(width=2, color='black'), name=row['sdtname'],showlegend=False))
            elif row.geometry.geom_type == 'MultiPolygon':
                # For multipolygons
                for polygon in row.geometry.geoms:
                    x, y = polygon.exterior.xy
                    fig_soil_type.add_trace(go.Scattermapbox(lon=list(x), lat=list(y), mode='lines', fill='none', line=dict(width=2, color='black'), name=row['sdtname'],showlegend=False))
        fig_soil_type.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(width=5, color='black'), name="Sundarban Landscape"))
            
        
        
        fig_soil_type.update_layout(
            title_text='<b>Sundarban Soil Data - Soil Type<b>',
            mapbox=dict(
                style="open-street-map",  
                center=dict(lat=df_soilavg['Latitude'].mean(), lon=df_soilavg['Longitude'].mean()),  
                zoom=10,  
                bearing=0,
                pitch=0,
                layers=[],  
            ),
            showlegend=True,
            legend=dict(
                title='<b>Soil Types<b>',  
                font=dict(size=16),  
            ),
            height=1200, 
            width=2150,  
        )

       

        return dcc.Graph(figure=fig_soil_type)
    

@app.callback(
    Output('independent-plot', 'figure'),
    [Input('wwf-button3', 'n_clicks')]
)
def update_independent_plot(n_clicks):

    
    df_soilavg=pd.read_csv('https://github.com/RitoshreeS/Sundarban-Dashboard/blob/main/Data/Soil_avg.csv')

    fig = px.bar(df_soilavg, x="Village", y=["PH","EC1:2 (ds/m)","O/C (%)","N (kg/ha)","P (kg/ha)","K (kg/ha)","Cu (ppm)","Zn (ppm)","Fe (ppm)","Mn (ppm)","B (ppm)","S (ppm)"], title="<b>Soil Composition Per Village<b>")
    fig.update_layout(height=800)
    
    return fig

# Run the app and open it in a new tab
if __name__ == '__main__':
    app.run_server(debug=True, port=8051)
    
