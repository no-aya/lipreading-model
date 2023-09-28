import os
import tensorflow as tf
from typing import List
import cv2

vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'1234567890? "]
char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(),
    oov_token="",
    invert=True
)

# Load videos
def load_video(path:str) -> List[float]:
    """
    Loads a video from a given path, reduces the size of the video and standardizes it.
    :param path: Path to the video
    :return: A list of standardized frames
    """

    cap = cv2.VideoCapture(path)
    frames = []
    for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        ret, frame = cap.read()
        frame = tf.image.rgb_to_grayscale(frame) # Less data to process
        frames.append(frame[190:236,80:220,:]) # This is the region of interest (ROI) that contains the mouth
        # However we can replace it with a face detector library like dlib
    cap.release()

    # We standardize the frames so that they have a mean of 0 and standard deviation of 1.
    # (Scaling data, good practice)
    mean = tf.math.reduce_mean(frames)
    casted_frames= tf.cast(frames, tf.float32)
    std = tf.math.reduce_std(casted_frames)
    return tf.cast((frames - mean), tf.float32) / std

# Load alignments
# Alignments are the text that is being said in the video
def load_alignments(path:str) -> List[str]:
    with open(path, 'r') as f:
        alignments = f.readlines()
    tokens = []
    for alignment in alignments:
        alignment = alignment.split()
        if alignment[2] != 'sil':
            tokens = [*tokens, ' ', alignment[2]]
    tokens_unicode= tf.strings.unicode_split(tokens, 'UTF-8')
    return char_to_num(tf.reshape(tokens_unicode, [-1]))[1:]

# Load data from the dataset
def load_data(path:str):
    """
    This function will load the data from the folder and call the load_video and load_alignments functions to import the frames and alignments simultaneously
    :param path: Path to the dataset
    :return: frames, alignments
    """
    path = bytes.decode(path.numpy()) # Convert the path to a string
    file_name = path.split('\\')[-1].split('.')[0] # Get the file name
    video_path = os.path.join('..','data','sequences',f'{file_name}.mpg') # Get the video path
    alignment_path = os.path.join('..','data','alignments','s1',f'{file_name}.align') # Get the alignment path
    return load_video(video_path), load_alignments(alignment_path)
