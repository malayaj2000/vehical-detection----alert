from utils import *
def detections(path):
    model = torch.hub.load('facebookresearch/detr:main', 'detr_resnet50', pretrained=True)
    model.eval()
    model.to(device)

    cap = cv2.VideoCapture(path)

    if(cap.isOpened() == False):
        print("Error opening video stream or file")

    ret,frame = cap.read()
    fps_start = 0
    fps = 0
    while(ret):
        ret,frame = cap.read()
            
        im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(im)
        img = transform(im).unsqueeze(0)
        img = img.to(device)
        outputs = model(img)
        probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
        probas = probas*mask
        keep = probas.max(-1).values > 0.9
        
        bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)
        
        fps_end = time.time()
        fps = 1/(fps_end - fps_start)
        fps_start = fps_end
        show(im, probas[keep], bboxes_scaled,fps)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    cap.release()