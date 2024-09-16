import matplotlib.pyplot as plt
from model import Perceptron

def visualization_error(X, y, ppn):
   
    ppn.fit(X, y)
    
    plt.plot(range(1, len(ppn.errors) + 1), ppn.errors, marker = 'o')
    plt.xlabel('Epochs')
    plt.ylabel('Number of updates')

    plt.show()