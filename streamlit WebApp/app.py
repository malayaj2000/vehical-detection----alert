import cv2
import streamlit as st
from utils import *
from detect import detect
st.title("Webcam Live Feed")
run = st.checkbox('Run')
FRAME_WINDOW = st.image([])
camera = cv2.VideoCapture(0)

model = torch.hub.load('facebookresearch/detr:main', 'detr_resnet50', pretrained=True)
model.eval()
model.to(device)

while run:
    _, frame = camera.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    ######################################################
    frame = detect(frame,model)
    ######################################################
    FRAME_WINDOW.image(frame)
else:
    st.write('Stopped')