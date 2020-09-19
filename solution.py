import numpy as np
import imageio
import glob

# Activation function
def sigmoid(t):
    return 1/(1+np.exp(-t))

# Derivative of sigmoid
def sigmoid_derivative(p):
    return p * (1 - p)

#preprocess images
def image_preprocessing(title):
    arr = imageio.imread(title, as_gray=True)
    data = 255.0 - arr.reshape(1600)
    data = (data / 255.0 * 0.99) + 0.01
    return data
    
class neuralNetwork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, grate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.gr = grate
        self.weights1 = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.weights2 = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))
    
    def train(self, inputs, targets):
        hiddeninputs = np.dot(self.weights1, np.array(inputs, ndmin=2).T)
        hiddenoutputs = sigmoid(hiddeninputs)
        finalinputs = np.dot(self.weights2, hiddenoutputs)
        finaloutputs = sigmoid(finalinputs)
        self.weights1 += self.gr * np.dot(((np.dot(self.weights2.T, np.array(targets, ndmin=2).T - finaloutputs)) * sigmoid_derivative(hiddenoutputs)), np.transpose(np.array(inputs, ndmin=2).T))
        self.weights2 += self.gr * np.dot(((np.array(targets, ndmin=2).T - finaloutputs) * sigmoid_derivative(finaloutputs)), np.transpose(hiddenoutputs))
       
    def test(self, inputs):
        return sigmoid(np.dot(self.weights2, sigmoid(np.dot(self.weights1, np.array(inputs, ndmin=2).T))))
        
def training():
    rate = float(input('Input learning rate: '))
    epochs = int(input('Input epochs: '))
    images = []
    trsymbols = []

    n = neuralNetwork(1600, 1000, 10, rate)
    
    for image_file_name in glob.glob('templates/tp?.png'):
        print(image_file_name)
        images.append(image_preprocessing(image_file_name))
        trsymbols.append(image_file_name[-5])
        

    for i in range(0, epochs):
        for item in range(0, len(images)):
            index = 0
            for oper in range(0, len(symbols)):
                if symbols[oper]==trsymbols[item]:
                    index = oper
            targets = np.zeros(10) + 0.01
            targets[index] = 0.99
            n.train(images[item], targets)
    return n


def testing(n):
    print('!','&','^','~','=','↑','→','↓','↔','v')
    symbol = input('Input a symbol for which you would like to check test image: ')
    s="test/ts"+symbol+".png"
    res = n.test(image_preprocessing(s))
    summ = 0
    for i in res:
        summ += i
    print(np.argmax(res))
    print('Posibility: ',(res[np.argmax(res)]/summ)*100,'%')
    print("Actual Output: " + symbol)
    print("Predicted Output: " + symbols[np.argmax(res)])
    

symbols = ['!','&','^','~','=','↑','→','↓','↔','v']


print("Hi user!")
ans = 1
n = training()
while ans!=0:
    ans=int(input("Choose an option 1-to train , 2 - to test, 0 - exit: "))
    if ans==1:
        n = training()
    elif ans==2:
        testing(n)
    