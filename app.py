import base64
import datetime
from sklearn.metrics import r2_score
import io
import joblib 
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
from dash.dependencies import Input, Output, State
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


#Fancy Styles and fonts provided to us by nice people
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']


app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


data_wind = pd.read_csv('data/wind_generation_data.csv')   #reading csv file
data_wind
data_wind.isnull().any()                    #checking null values
data_wind['Days'] = data_wind.index             #setting days as index of dataframe
data_wind.set_index('Days', inplace=True)     
data_wind
X = data_wind[['wind speed','direction']].values    #predictor values
Y = data_wind['Power Output'].values                 #target values
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25,random_state=42)  #test train split

rf = RandomForestRegressor()  #calling and fitting the model
rf.fit(X_train, y_train)
y_predicted = rf.predict(X_test)    #model predictions
y_predicted                        
joblib.dump(rf, "rf.pkl")           #serialise data with pickle

data_solar = pd.read_csv('data/solar_generation_data.csv')     #reading csv file
data_solar
data_solar.isnull().any()                   #checking null values
#cleaning dataframe
data_clean = data_solar.fillna(method='ffill')   #filling null values
data_clean
data_clean.isnull().any()     #confirming filling of null values
data_clean.dtypes
deg= data_clean['Temp Hi'] = data_clean['Temp Hi'].str.replace("°", "")
deg_1= data_clean['Temp Low'] = data_clean['Temp Low'].str.replace("°", "")
#changing entries of two columns from string to float
data_clean['Temp Hi'] = data_clean['Temp Hi'].astype(float)
data_clean['Temp Low'] = data_clean['Temp Low'].astype(float)
data_clean
data_clean.dtypes
X1 = data_clean[['Temp Hi','Temp Low','Solar','Cloud Cover Percentage','Rainfall in mm']].values
Y1 = data_clean['Power Generated in MW'].values
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, Y1, test_size=0.25,random_state=42)
scaler = StandardScaler()
scaler.fit(X1_train)
X1_train = scaler.transform(X1_train)
X1_test = scaler.transform(X1_test)
forest = RandomForestRegressor()
forest.fit(X1_train, y1_train)
y1_predicted = forest.predict(X1_test)
y1_predicted
#score = r2_score(y1_test, y1_predicted)
#score
joblib.dump(forest, "forest.pkl")         

rf_from_joblib = joblib.load('data/rf.pkl')
Power = rf_from_joblib.predict(X_test)
days = data_clean['Day'].head()

forest_from_joblib = joblib.load('data/forest.pkl')
Power_1 = forest_from_joblib.predict(X1_test)
days_1 = data_clean['Day'].head()

# This structure represents the bar graph
# id: The html id of the bar graph
# figure: info that actually goes into the bar graph. It is a python dictionary
# data: The data for the x and y axis. The type of the graph and the name of the graph. Different types and names documented by plotly (https://plot.ly/python/bar-charts/) or (https://plot.ly/python/)

	# *bar1

bar1 = dcc.Graph(
        id='barChart-p1',	
        figure={
            'data': [
                {'x': days, 'y': Power, 'type': 'bar', 'name': 'Power'}
                
                    ],
            'layout': {
                'title': 'Predicted Power Output (Wind)'
            }
        }
    )

	# *bar2
bar2 = dcc.Graph(
        id='barChart-p2',	
        figure={
            'data': [
                {'x': days_1, 'y': Power_1, 'type': 'bar', 'name': 'Power'}
                
                    ],
            'layout': {
                'title': 'Predicted Power Output (Solar)'
            }
        }
    )

	# *bar3
bar3 = dcc.Graph(
        id='barChart-p3',	
        figure={
            'data': [
                {'x': days, 'y': Power, 'type': 'bar', 'name': 'Wind'},
                {'x': days_1, 'y': Power_1, 'type': 'bar', 'name': 'Solar'},
                
                    ],
            'layout': {
                'title': 'Power plant Wind and Solar'
            }
        }
    )


def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    return html.Div([
        html.H5(filename),
        html.H6(datetime.datetime.fromtimestamp(date)),

        dash_table.DataTable(
            data=df.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in df.columns]
        ),

        html.Hr(),  # horizontal line

        # For debugging, display the raw contents provided by the web browser
        html.Div('Raw Content'),
        html.Pre(contents[0:200] + '...', style={
            'whiteSpace': 'pre-wrap',
            'wordBreak': 'break-all'
        })
    ])


@app.callback(Output('output-data-upload', 'children'),
              [Input('upload-data', 'contents')],
              [State('upload-data', 'filename'),
               State('upload-data', 'last_modified')])
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children




app.layout = html.Div([
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=True
    ),

    html.Div(children='''
        Emmanuel Mireku Summative Assessment Dashboard.
    '''),
     html.Div(id='output-data-upload'),

    bar1,
    bar2,
    bar3,
])

if __name__ == '__main__':
    app.run_server(debug=True)