from neural_net import *
from util import *
from linear import *
from act_func import *
import pickle
import numpy as np

def generate_image_for_class(model, target_class):
    """
    This function Generates a random image that will be classified
    as class target_class by the neural network.
    Parameters:
    ------------------------------------
    model: neural network model object
    target_class: integer, target_class to which the network classifies the image
    alpha: each pixel in the image is initialized by sampling from
            uniform distribution over (-alpha, alpha)
    """
    image = np.random.uniform(-0.1, 0.1,(1,784))
    y_probs = model.forward(image)
    yhat = np.argmax(y_probs)
    while (yhat != target_class):
        image += -0.1 * model.grad_wrt_input(np.array([image]), np.array([target_class]))
        y_probs = model.forward(image)
        yhat = np.argmax(y_probs)

    filename = "targeted_random_img_class_[{target}].png".format(target = target_class)
    adv_image = visualize_example(image, y_probs, filename = filename)
    return adv_image

def main():
    model = None
    with open('trained_model.pkl', 'rb') as fid:
        model = pickle.load(fid)
    for c in range(10):
        generate_image_for_class(model, c)
        

if __name__ == "__main__":
    main()
