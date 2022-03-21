import ast
import json
import base64
import logging
import zipfile
from io import BytesIO
from collections import Counter

import requests
from PIL import Image
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from streamlit_echarts import st_echarts


def download_data_sample(api_url, token):
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

        # Parsing the base64 encoded images
        images = sample_images['exampledata'].split(",")

        # Prepare the data sample in zip
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
            for i, image in enumerate(images):
                file_name = "image_sample{}.png".format(i)
                image_buffer = BytesIO()
                image_buffer.write(base64.b64decode(image))
                zip_file.writestr(file_name, image_buffer.getvalue())
        
        # Setup a download button
        btn = st.download_button(
            label="Download data sample",
            data=zip_buffer.getvalue(),
            file_name="data_sample.zip",
            mime="application/zip"
        )
    except Exception as e:
        logging.error(e)

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
    data = [{"value": sizes[i], "name": labels[i]} for i in range(len(sizes))]
    options = {
        "tooltip": {"trigger": "item"},
        "legend": {"top": "5%", "left": "center"},
        "series": [
            {
                "name": "Prediction Statistics",
                "type": "pie",
                "radius": ["20%", "70%"],
                "avoidLabelOverlap": False,
                "itemStyle": {
                    "borderRadius": 10,
                    "borderColor": "#fff",
                    "borderWidth": 2,
                },
                "label": {"show": False, "position": "center"},
                "emphasis": {
                    "label": {"show": True, "fontSize": "40", "fontWeight": "bold"}
                },
                "labelLine": {"show": False},
                "data": data,
            }
        ],
    }
    st_echarts(
        options=options, height="500px",
    )
    
def display_bar_chart(freqs, labels):
    options = {
        "xAxis": {
            "type": "category",
            "data": labels,
        },
        "yAxis": {"type": "value"},
        "series": [{"data": freqs, "type": "bar"}],
    }
    st_echarts(options=options, height="500px")
    
def display_stats(labels):
    counter = Counter(labels)
    unique_labels = list(counter.keys())
    freqs = list(counter.values()) # frequency of each labels

    # Size or portion in pie chart
    sizes = [float(x) / sum(freqs) * 100 for x in freqs]

    # Display prediction details
    with st.container():
        col1, col2 = st.columns(2)

        with col1:
            display_pie_chart(sizes, unique_labels)

        with col2:
            display_bar_chart(freqs, unique_labels)

def predict(uploaded_file, api_url, token):
    # Prepare the uploaded image into base64 encoded string
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
    return label

def main():
    # Set the API url accordingly based on AIModelShare Playground API.
    api_url = "https://5l7clbsyu5.execute-api.us-east-1.amazonaws.com"

    # Get the query parameter
    params = st.experimental_get_query_params()
    if "token" not in params:
        st.warning("Please insert the auth token as query parameter. " 
                   "e.g. https://share.streamlit.io/raudipra/"
                   "streamlit-image-classification/main?token=secret")
        token = ""
    else:
        token = params['token'][0]

    st.header("Flower Image Classification")

    uploaded_files = st.file_uploader(
        label="Choose any image and get the prediction",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
    )

    download_data_sample(api_url, token)
        
    labels = []
    statuses = []
    images = []    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            try:
                # Classify the image
                images.append(uploaded_file)
                label = predict(uploaded_file, api_url, token)
                
                # Insert the label into labels
                labels.append(label)
                
                # Insert the API call status into statuses
                statuses.append(True)
            except Exception as e:
                logging.error(e)

                # add label as None if necessary
                if len(labels) < len(images):
                    labels.append(None)
                statuses.append(False)

        display_stats(labels)
        display_result(images, labels, statuses)

if __name__ == "__main__":
    main()