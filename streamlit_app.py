import ast
import json
import base64
from io import BytesIO
from collections import Counter

import requests
from PIL import Image
import streamlit as st
from loguru import logger
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go


def redirect_loguru_to_streamlit():
    def _filter_warning(record):
        return record["level"].no == logger.level("WARNING").no    
    if 'warning_logger' not in st.session_state:
        st.session_state['warning_logger'] = logger.add(st.warning, filter=_filter_warning, level='INFO')
    if 'error_logger' not in st.session_state:
        st.session_state['error_logger'] = logger.add(st.error, level='ERROR')

redirect_loguru_to_streamlit()

def download_sample_data(api_url, token):
    try:
        # Set the path for eval API
        eval_url = api_url + "/prod/eval"
        
        # Set the authorization based on query parameter 'token', 
        # it is obtainable once you logged in to the modelshare website
        headers = {
            "Content-Type": "application/json", 
            "authorizationToken": token,
        }

        # Set the body indicating we want to get sample data from eval API
        data = {
            "exampledata": "TRUE"
        }
        data = json.dumps(data)

        # Send the request
        sample_images = requests.request("POST", eval_url, 
                                         headers=headers, data=data).json()
        logger.warning(len(sample_images['exampledata'].split(",")))
        # sample_images["totalfiles"]
        # sample_images["exampledata"]
    except Exception as e:
        logger.error(e)

def display_result(images, labels, statuses):
    status_label = {
        True: "Success",
        False: "Failed",
    }
    for (image, label, status) in zip(images, labels, statuses):
        # Display the image with filename as caption
        with st.container():
            col1, col2 = st.columns(2)

            with col1:
                st.image(
                    image, 
                    caption=image.name, 
                    use_column_width=True,
                )

            # Display prediction details
            with col2:
                st.write("Status: {}".format(status_label[status]))
                st.write("Label: {}".format(label))

def display_pie_chart(sizes, labels):
    fig = go.Figure(data=[go.Pie(labels=labels, values=sizes)])
    st.plotly_chart(fig, use_container_width=True)
    
def display_bar_chart(freqs, labels):
    fig = px.bar(x=labels, y=freqs)
    st.plotly_chart(fig, use_container_width=True)
    
def display_stats(labels):
    # Pie chart, where the slices will be ordered and plotted counter-clockwise:
    counter = Counter(labels)
    unique_labels = list(counter.keys())
    freqs = list(counter.values())

    sizes = [float(x) / sum(freqs) * 100 for x in freqs]

    display_pie_chart(sizes, unique_labels)
    display_bar_chart(freqs, unique_labels)

# Set the API url accordingly based on AIModelShare Playground API.
api_url = "https://5l7clbsyu5.execute-api.us-east-1.amazonaws.com"

# Get the query parameter
params = st.experimental_get_query_params()
token = params['token'][0]

st.header("Flower Image Classification")
st.write("Choose any image and get the prediction:")

uploaded_files = st.file_uploader(
    label="Choose an image...",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True,
)

if st.button('Download sample data'):
    download_sample_data(api_url, token)
    
labels = []
statuses = []
images = []    
if uploaded_files:
    for uploaded_file in uploaded_files:
        try:
            # Prepare the uploaded image into base64 encoded string
            images.append(uploaded_file)
            image = Image.open(uploaded_file)
            buffered = BytesIO()
            image.save(buffered, format="JPEG")
            encoded_string = base64.b64encode(buffered.getvalue())
            data = json.dumps({"data": encoded_string.decode('utf-8')})

            # Set the path for prediction API
            pred_url = api_url + "/prod/m"
            
            # Set the authorization based on query parameter 'token', 
            # it is obtainable once you logged in to the modelshare website
            headers = {
                "Content-Type": "application/json", 
                "authorizationToken": token,
            }

            # Send the request
            prediction = requests.request("POST", pred_url, 
                                          headers=headers, data=data)

            # Parse the prediction
            label = ast.literal_eval(prediction.text)[0]
            
            # Insert the label into labels
            labels.append(label)
            
            # Insert the API call status into statuses
            statuses.append(True)
        except Exception as e:
            logger.error(e)

            # add label as None if necessary
            if len(labels) < len(images):
                labels.append(None)
            statuses.append(False)

    display_result(images, labels, statuses)
    display_stats(labels)