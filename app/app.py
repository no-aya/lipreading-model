# Importing dependencies
import streamlit as st
import os
import tensorflow as tf
import imageio

# Importing our own modules
from utils import load_data, num_to_char
from modelutil import load_model

# Set te layout 
st.set_page_config(
    page_title="Lip Reading App",
    page_icon="assets\logo.png",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "Lip reading app. This is a lip reading app that uses a deep learning model to predict the text being said in a video."
    }
)

# Setting up the sidebar
with st.sidebar :
    st.image('assets\logo.png')
    st.title("Lip Reading")
    st.write("This is a lip reading app that uses a deep learning model to predict the text being said in a video.")
    st.info("This app is originally made from LipNet's deep Learning Model ")

# Setting up the main page
st.title("Lip Reading App")

# Setting up options 
options = ['Camera', 'Sample Video']
choice = st.radio("Choose an option", options)

# Generating columns _(similar to bootsrap)
if choice == 'Sample Video':
    # Select a  sample 
    files = os.listdir(os.path.join('..','data','sequences'))
    selected_file = st.selectbox("Select a sample", files)
    col1, col2 = st.columns(2)
    if files :
        with col1 :
            st.info("This video displays the converted text")
            # Issue : The app doesn't process the .mpg format
            # st.video(os.path.join('..','data','sequences',selected_file))
            file_path = os.path.join('..','data','sequences',selected_file)
            # Convert the video format 
            os.system(f"ffmpeg -i {file_path} -vcodec libx264 test_video.mp4 -y") 
            # Render the video
            video = open('test_video.mp4', 'rb')
            video_bytes = video.read()
            st.video(video_bytes) 
        with col2 :
            st.info("What the model uses to predict")
            video, annotations = load_data(tf.convert_to_tensor(file_path))
            imageio.mimsave('test.gif', video, fps=10)
            st.image('test.gif',use_column_width=True)

            st.info('This is the output of the model as tokens')
            # CTC 
            model = load_model()
            predictions = model.predict(tf.expand_dims(video, axis=0))
            #st.text(tf.argmax(predictions, axis=1))
            decoder = tf.keras.backend.ctc_decode(predictions, [75], greedy=True)[0][0].numpy()
            st.write(decoder)
            st.info('This is the output of the model as text')
            #NUm to char
            text = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
            st.text(text)
if choice == 'Camera' :
    st.info("This is a lip reading app that uses a deep learning model to predict the text being said in a video.")
    # Open the camera
    

#%%
