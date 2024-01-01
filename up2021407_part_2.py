def OR_perceptron(x1, x2):
    w1 = 1
    w2 = 1
    theta = 1
    yin = (x1 * w1) + (x2 * w2)
    if yin >= theta:
        return 1
    else:
        return 0

#test
print("OR Perceptron")
print(OR_perceptron(0, 0))
print(OR_perceptron(0, 1))
print(OR_perceptron(1, 0))
print(OR_perceptron(1, 1))
print("----------------------------------")

def XOR_perceptron(x1, x2):
    w1 = 1
    w2 = -1
    theta = 0
    yin = (x1 * w1) + (x2 * w2)
    if yin >= theta:
        return 1
    else:
        return 0

#test
print("XOR Perceptron")
print(XOR_perceptron(0, 0))
print(XOR_perceptron(0, 1))
print(XOR_perceptron(1, 0))
print(XOR_perceptron(1, 1))
