from PIL import Image
import numpy as np
import torch

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''


    # TODO: Process a PIL image for use in a PyTorch model

    im = Image.open(image)
    im = im.resize((256,256))
    im = im.crop((0,0,224,224))
    np_image = np.array(im, dtype=np.float32) / 255
    np_image -= np.array([0.485, 0.456, 0.406])
    np_image /= np.array([0.229, 0.224, 0.225])

    return np_image.transpose(2,0,1)


def predict(image_path, model, topn, device):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    # TODO: Implement the code to predict the class from an image file
    image = process_image(image_path)
    image = torch.from_numpy(image)
    image = image.to(device)
    model.eval()
    with torch.no_grad():
      output = model(image.unsqueeze(0)) # model processes batches
      ps = torch.exp(output)
      top_p, top_class = ps.topk(topn, dim=1)
    top_p = top_p.to('cpu')
    top_class = top_class.to('cpu')
    top_class = top_class.numpy().reshape(-1)

    invert_dict = {value:key for key,value in model.class_to_idx.items()}
    for c in top_class:
      if c in invert_dict.keys():
        value = invert_dict[c]
        top_class[np.where(top_class == c)] = value



    return top_p.numpy().reshape(-1), top_class

