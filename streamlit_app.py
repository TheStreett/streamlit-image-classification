import ast
import uuid
import json
import base64
import logging
import zipfile
from io import BytesIO
from datetime import datetime
from collections import Counter

import requests
import pandas as pd
from PIL import Image
import streamlit as st
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

def display_result(images, labels, statuses, datetimes, uuids):
    status_label = {
        True: "Success",
        False: "Failed",
    }
    with st.container():
        col1, col2, col3, col4, col5 = st.columns([1, 2, 1, 1, 1])

        with col1:
            st.write("Time")
        
        with col2:
            st.write("Input")
        
        with col3:
            st.write("Filename")
        
        with col4:
            st.write("Request Status")
        
        with col5:
            st.write("Result")
    
    for (image, label, status, date_time) in zip(images, labels, statuses, datetimes):
        # Display prediction details
        with st.container():
            col1, col2, col3, col4, col5 = st.columns([1, 2, 1, 1, 1])

            # Display datetime
            with col1:
                st.write(date_time)

            # Display the image
            with col2:
                st.image(
                    image[0], 
                    use_column_width=False,
                )
            
            # Display filename
            with col3:
                st.caption(image[1])
                
            with col4:
                st.write(status_label[status])
                
            with col5:
                st.write(label)

    # Create dataframe
    data_frame = pd.DataFrame()
    data_frame = data_frame.assign(result=labels)
    data_frame = data_frame.assign(status=[status_label[x] for x in statuses])
    data_frame = data_frame.assign(filename=[x[1] for x in images])
    data_frame = data_frame.assign(unique_id=uuids)
    data_frame = data_frame.assign(time=datetimes)

    # Prepare the data sample in csv
    csv_data = data_frame.to_csv(index=False).encode('utf-8')

    # Setup a download button
    st.download_button(
        label="Download result",
        data=csv_data,
        file_name="export.csv",
        mime="text/csv"
    )

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

def predict(uploaded_file, uuid_str, api_url, token):
    # Prepare the uploaded image into base64 encoded string
    image = Image.open(uploaded_file)
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    encoded_string = base64.b64encode(buffered.getvalue())
    data = json.dumps({
        "data": encoded_string.decode('utf-8'),
        "uuid": uuid_str,
        "version": "mostrecent"
    })

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

    print(prediction.text)
    # Parse the prediction
    label = ast.literal_eval(prediction.text)[0]
    return label

# Resize image and extract the image filename
def transform_image(uploaded_file):
    image = Image.open(uploaded_file)
    MAX_SIZE = (150, 150)
    image.thumbnail(MAX_SIZE)
    return (image, uploaded_file.name)

# Add custom CSS for streamlit component
def local_css(file_name):
    with open(file_name) as f:
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)

def main():
    # Set custom CSS for whole layout
    local_css("style.css")

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

    labels = []
    statuses = []
    images = []
    uuids = []
    datetimes = []

    st.header("Flower Image Classification")
    
    with st.expander("Show instruction"):
        st.write("To build and run modelshare's streamlit app, you will need "
                  "authorization token and modelshare's playground URL. "
                  "You can obtain the auth token by signing in to www.modelshare.org "
                  "and the playground URL by choosing any available playground in "
                  "www.modelshare.org. Use the sample code below and pass the auth token to the "
                  "app as a query parameter 'token' on streamlit's URL, e.g. "
                  "https://share.streamlit.io/user/apps-name/main?token=secret.")

        st.write("Here are some important part of codes to classify an image"
                 " using modelshare's playground url")

        code = """
import ast
import json
import base64
import requests
from PIL import Image
import streamlit as st
from io import BytesIO

api_url = "https://5l7clbsyu5.execute-api.us-east-1.amazonaws.com"
token = st.experimental_get_query_params()['token'][0]
image = Image.open(image_file)
def predict(image, api_url, token):
    # Prepare the uploaded image into base64 encoded string
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
label = predict(data, api_url, token)
        """
        st.code(code, "python")

    with st.container():
        col1, col2 = st.columns([1, 1])

        with col1:
            uploaded_files = st.file_uploader(
                label="Choose any image and get the prediction",
                type=["png", "jpg", "jpeg"],
                accept_multiple_files=True,
            )

            download_data_sample(api_url, token)

        with col2:
            metric_placeholder = st.empty()
            metric_placeholder.metric(label="Request count", value=len(statuses))
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            try:
                # Keep the resized image and filename for prediction display
                images.append(transform_image(uploaded_file))
                
                # Create identifier for this prediction
                uuid_str = str(uuid.uuid4())
                uuids.append(uuid_str)
                
                # Capture the timestamp of prediction
                now = datetime.now()
                date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
                datetimes.append(date_time)

                # Classify the image
                label = predict(uploaded_file, uuid_str, api_url, token)
                
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

        metric_placeholder.metric(label="Request count", value=len(statuses))
        display_stats(labels)
        display_result(images, labels, statuses, datetimes, uuids)

if __name__ == "__main__":
    main()