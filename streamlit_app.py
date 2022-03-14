import streamlit as st
import streamlit_authenticator as stauth
from numpy import load
from numpy import expand_dims
from matplotlib import pyplot
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import ast
import json
import base64
import logging
import requests
from io import BytesIO

names = ['John Smith', 'Rebecca Briggs']
usernames = ['jsmith', 'rbriggs']
passwords = ['123', '456']
hashed_passwords = stauth.hasher(passwords).generate()
authenticator = stauth.authenticate(names, usernames, hashed_passwords,
                                    'some_cookie_name', 'some_signature_key',
                                    cookie_expiry_days=30)
name, authentication_status = authenticator.login('Login','main')
if authentication_status:
    st.write('Welcome *%s*' % (name))
elif authentication_status == False:
    st.error('Username/password is incorrect')
elif authentication_status == None:
    st.warning('Please enter your username and password')

if st.session_state['authentication_status']:
    params = st.experimental_get_query_params()
    logging.info(params)

    st.header("Flower Image Classification")
    st.write("Choose any image and get the prediction:")

    uploaded_file = st.file_uploader("Choose an image...")

    if uploaded_file is not None:
        #src_image = load_image(uploaded_file)
        image = Image.open(uploaded_file)
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        encoded_string = base64.b64encode(buffered.getvalue())
        data = json.dumps({"data": encoded_string.decode('utf-8') })

        api_url = "https://5l7clbsyu5.execute-api.us-east-1.amazonaws.com/prod/m"
        headers = {"Content-Type": "application/json", "authorizationToken": params['token'][0]}

        prediction = requests.request("POST", api_url, headers = headers, data=data)
        label = ast.literal_eval(prediction.text)
        logging.info(label)

        st.image(uploaded_file, caption="Label: {}".format(label[0]), use_column_width=True)
