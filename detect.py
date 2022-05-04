from utils import *
def detect(frame,model):
    im = Image.fromarray(frame)
    img = transform(im).unsqueeze(0)
    img = img.to(device)
    outputs = model(img)
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    probas = probas*mask
    keep = probas.max(-1).values > 0.9
        
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)
    return draw(im, probas[keep], bboxes_scaled)



