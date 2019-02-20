import numpy as np

# =========================================================================
# 1. Funkcje aktywacyjne. p to 2elementowa lista pozycji w macierzy funkcyj aktywacyjnych
def activationFunction(x, loc_vector):

    x = list(x)

    layers_vs_activation_functions = {0: 'None', 1: 'ReLu', 2: 'None'}

    if loc_vector[0] == 0: # funkcja aktywacyjna

        if layers_vs_activation_functions[loc_vector[1]] == 'None':
            y = x
        elif layers_vs_activation_functions[loc_vector[1]] == 'ReLu':
            y = [max(i,0) for i in x]
        elif layers_vs_activation_functions[loc_vector[1]] == 'Sigm':
            y = np.exp(x)/(np.exp(x)+1)

    else: # pochodna funkcji aktywacyjnej

        if layers_vs_activation_functions[loc_vector[1]] == 'None':
            y = [1]*len(x)
        elif layers_vs_activation_functions[loc_vector[1]] == 'ReLu':
            y = np.heaviside(x,0.5)
        elif layers_vs_activation_functions[loc_vector[1]] == 'Sigm':
            y = np.exp(x)/(np.exp(x)+1)**2

    return np.array(y)

# -----------------------------------------------------------------------------------
# 2. Generacja danych uczących (i testowych)
def generateTrainingSet(n_training, n_inputs, input_limits):

    dataMatrix = np.zeros([n_training,n_inputs+2])

    for i in range(n_training):

        inputy = np.zeros(n_inputs)
        for j, lims in enumerate(input_limits):
            inputy[j] = np.random.rand()*lims[1]

        dataMatrix[i,0:n_inputs] = inputy
        dataMatrix[i,n_inputs] = inputy[0]/9.81*2*np.sin(inputy[1])*np.cos(inputy[1])
        dataMatrix[i,n_inputs+1] = inputy[0]**2/9.81/2*(np.sin(inputy[1]))**2

    return dataMatrix

# -------------------------------------------------------------------------------------------
# 3. Liczy output sieci na podstawie inputów i wag
def getNetworkOutput(y1, W, n_inputs, n_hidden, n_outputs):
    W1 = W[0:n_hidden,0:n_inputs+1,0]
    W2 = W[0:n_outputs,0:n_hidden+1,1]

    z1 = activationFunction(y1,[0,0])

    y2 = np.matmul(z1.transpose(),W1[:,:-1].transpose()) + W1[:,-1]   # kombinacja liniowa na neuronach hidden
    z2 = activationFunction(y2,[0,1])

    y3 = np.matmul(z2.transpose(),W2[:,:-1].transpose()) + W2[:,-1]   # kombinacja liniowa na neuronach output
    z3 = activationFunction(y3,[0,2])

    return z3

# --------------------------------------------------------------------------------------------
# 4. Liczy koszt sieci dla danego zestawu testowego i kompletu wag (oraz n_inputs dla ujednoznacznienia trainingSet)
def calcCost(trainingSet, W, n_inputs, n_hidden, n_outputs):
    ''' n_inputs pierwszych kolumn macierzy trainingSet to inputy, reszta to outputy '''
    W1 = W[0:n_hidden,0:n_inputs+1,0]
    W2 = W[0:n_outputs,0:n_hidden+1,1]

    koszty = []                   # lista kosztów sieci dla kolejnych przykładów testowych
    for i in range(np.shape(trainingSet)[0]):    # iteruje po wierszach (przykładach uczących)
        unit_cost = 0        # unit_cost to koszt po wszystkich outputach dla jednego przykładu uczącego

        # iteruje po wszystkich outputach dla i-tego przypadku testowego (w tej sieci po dwóch outputach)
        for j in range(n_inputs,np.shape(trainingSet)[1]):
            unit_cost += ( trainingSet[i,j] - getNetworkOutput(trainingSet[i,0:n_inputs],W,n_inputs, n_hidden, n_outputs)[j-n_inputs] )**2

        koszty.append(unit_cost)

    return np.mean(koszty)

# --------------------------------------------------------------------------------------------------------------------
# 5. Oblicza przyczynek od jednego n. outputowego i jednego testu do jednego elementu 3D macierzy gradientów.
# Działa dla każdego z 6 rodzaju wag ale przy jednym wywołaniu odnosi się do jednej wagi

# n - indeks neuronu outputowego, Z - macierz outputów, W - macierz wag, y_n - oczekiwana wartość neuronu n, Y - macierz preoutputów
# case - dwuelementowy wektor, wskazuje na pozycję w macierzy typów wagi w
# bound - lista pozycji w swojej warstwie 2 neuronów wiązanych przez daną wagę. Najpierw n.docelowy, potem ten z którego wychodzi

def calcGradElement(n, Z, W, y_n, Y, case, bound, n_inputs):
    chain_rule1 = 2*(Z[n,2] - y_n)
    chain_rule2 = activationFunction([Y[n,2]],[1,2])

    if case == [0,0]:
        chain_rule3 = 0
    elif case == [1,0]:
        chain_rule3 = 0
    elif case == [0,1]:
        m = bound[1]
        chain_rule3 = Z[m,1]
    elif case == [1,1]:
        chain_rule3 = 1
    elif case == [0,2]:
        suma = np.dot(Z[0:n_inputs,0], W[bound[0], 0:n_inputs, 0] ) + W[bound[0],n_inputs,0]
        chain_rule3 = W[n, bound[0], 1] * activationFunction([suma],[0,1]) * Z[bound[1],0]
    elif case == [1,2]:
        suma = np.dot(Z[0:n_inputs,0], W[bound[0], 0:n_inputs, 0] ) + W[bound[0],n_inputs,0]
        chain_rule3 = W[n, bound[0], 1] * activationFunction([suma],[0,1])

    return -chain_rule1*chain_rule2*chain_rule3








#
