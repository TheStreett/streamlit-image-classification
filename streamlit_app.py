import ast
import json
import base64
import logging
from io import BytesIO
from collections import Counter

import requests
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt


def download_sample_data(api_url, token):
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
    logging.info("sending request")
    sample_images = requests.request("POST", eval_url, 
                                     headers=headers, data=data).json()
    logging.info("request done")
    logging.info(sample_images)
    # sample_images["totalfiles"]
    # sample_images["exampledata"]

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
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    st.pyplot(fig)

def display_bar_chart(freqs, labels):
    fig, ax = plt.subplots()
    ax.bar(labels, freqs)
    st.pyplot(fig)

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

st.button('Download sample data', on_click=download_sample_data, 
          args=(api_url, token))
    
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
            logging.info(e)

            # add label as None if necessary
            if len(labels) < len(images):
                labels.append(None)
            statuses.append(False)

    display_result(images, labels, statuses)
    display_stats(labels)