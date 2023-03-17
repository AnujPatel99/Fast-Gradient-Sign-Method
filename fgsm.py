from neural_net import *
from linear import *
from act_func import *
from util import *
import pickle
import numpy as np

def fgsm(x_test, y_test, model, eps=0.05):
    yhat = np.argmax(model.forward(x_test))
    x = x_test
    while((yhat == y_test)):
        x = x + eps*np.sign(model.grad_wrt_input(np.array([x]), np.array([yhat])))
        yhat = np.argmax(model.forward(x))
    return x

def main():
    # load datasets
    mnist = None
    with open('mnist.pkl', 'rb') as fid:
        mnist = pickle.load(fid)
    # load model
    model = None
    with open('trained_model.pkl', 'rb') as fid:
        model = pickle.load(fid)
        
    x_test = mnist['test_images'][0]
    y_test = mnist['test_labels'][0]
    
    x_adv = fgsm(x_test, y_test, model) 
    y_probs = model.forward(x_adv)
    visualize_example(x_adv, y_probs, filename = "FGSM_untargeted.png")
    
if __name__ == "__main__":
    main()
