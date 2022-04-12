from predict import predict
from helpers import draw_bboxs, resizer

def get_quads(image, weights = "best.pt"):
    image_path = image
    
    pred_list, pred = predict(
    weights=weights,
    source=image_path,
    imgsz=[1344, 768])
    
    classes = np.array(pred[0])[:, -1]
    
    big_quads = []
    small_quads = []
    for i in range(len(pred_list)):
        pred_list[i].append(classes[i])
        if int(classes[i]) == 1:
            big_quads.append(pred_list[i])
        else:
            small_quads.append(pred_list[i])
            
    return big_quads, small_quads