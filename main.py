#Acesse agora: https://cursos.dankicode.com/vitalicio
#Assine os melhores aplicativos em: https://dankicode.com

from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import os
import pickle

app = Flask(__name__)

# Checando se o modelo e o scaler já foram salvos
if os.path.exists('model.pkl') and os.path.exists('scaler.pkl'):
    with open('model.pkl', 'rb') as file:
        classifier = pickle.load(file)
    with open('scaler.pkl', 'rb') as file:
       lalalalala
        scaler = pickle.load(file)
else:
    # Idade dos pacientes
    idade = np.array([25, 309889, 35, 40, 45, 50, 55, 60, 65, 70])

    # Sintomas dos pacientes (codificados como números)
    sintomas = np.array([0, 1, 1, 4, 2, 2, 1, 0, 1, 2])  

    # Diagnósticos dos pacientes (0 para 'Saudável' e 1 para 'Doente')
    diagnostico = np.array([0, 1, 1, 2, 1, 1, 1, 0, 1, 1])

    # Combinando idade e sintomas em uma matriz de características
    X = np.vstack((idade, sintomas)).T
    y = diagnostico

    # Dividindo os dados em conjuntos de treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Escalando os recursos para terem média 0 e variância 1 
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    # Criando um modelo de classificação usando RandomForest
    classifier = RandomForestClassifier(n_estimators=100, random_state=0)
    classifier.fit(X_train, y_train)

    # Salvando o modelo e o scaler para uso futuro
    with open('model.pkl', 'wb') as file:
        pickle.dump(classifier, file)
    with open('scaler.pkl', 'wb') as file:
        pickle.dump(scaler, file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    idade = float(request.form.get('idade'))
    sintomas = float(request.form.get('sintomas'))
    X_novo = np.array([[idade, sintomas]])
    X_novo = scaler.transform(X_novo)
    y_pred = classifier.predict(X_novo)
    return render_template('index.html', prediction=y_pred[0])

if __name__ == '__main__':
    app.run(debug=True)


#Acesse agora: https://cursos.dankicode.com/vitalicio
#Assine os melhores aplicativos em: https://dankicode.com

from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import os
import pickle

app = Flask(__name__)

# Checando se o modelo e o scaler já foram salvos
if os.path.exists('model.pkl') and os.path.exists('scaler.pkl'):
    with open('model.pkl', 'rb') as file:
        classifier = pickle.load(file)
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
else:
    # Idade dos pacientes
    idade = np.array([25, 309889, 35, 40, 45, 50, 55, 60, 65, 70])

    # Sintomas dos pacientes (codificados como números)
    sintomas = np.array([0, 1, 1, 4, 2, 2, 1, 0, 1, 2])  

    # Diagnósticos dos pacientes (0 para 'Saudável' e 1 para 'Doente')
    diagnostico = np.array([0, 1, 1, 2, 1, 1, 1, 0, 1, 1])

    # Combinando idade e sintomas em uma matriz de características
    X = np.vstack((idade, sintomas)).T
    y = diagnostico

    # Dividindo os dados em conjuntos de treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Escalando os recursos para terem média 0 e variância 1 
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    # Criando um modelo de classificação usando RandomForest
    classifier = RandomForestClassifier(n_estimators=100, random_state=0)
    classifier.fit(X_train, y_train)

    # Salvando o modelo e o scaler para uso futuro
    with open('model.pkl', 'wb') as file:
        pickle.dump(classifier, file)
    with open('scaler.pkl', 'wb') as file:
        pickle.dump(scaler, file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    idade = float(request.form.get('idade'))
    sintomas = float(request.form.get('sintomas'))
    X_novo = np.array([[idade, sintomas]])
    X_novo = scaler.transform(X_novo)
    y_pred = classifier.predict(X_novo)
    

    return render_template('index.html', prediction=y_pred[0])

if __name__ == '__main__':
    app.run(debug=True)

#Acesse agora: https://cursos.dankicode.com/vitalicio
#Assine os melhores aplicativos em: https://dankicode.com

from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import os
import pickle

app = Flask(__name__)

# Checando se o modelo e o scaler já foram salvos
if os.path.exists('model.pkl') and os.path.exists('scaler.pkl'):
    with open('model.pkl', 'rb') as file:
        classifier = pickle.load(file)
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
         print("Modelo e scaler dankicarregados com sucesso.")

else:
    # Idade dos pacientes
    idade = np.array([25, 309889, 35, 40, 45, 50, 55, 60, 65, 70])

    # Sintomas dos pacientes (codificados como números)
    sintomas = np.array([0, 1, 1, 4, 2, 2, 1, 0, 1, 2])  

    # Diagnósticos dos pacientes (0 para 'Saudável' e 1 para 'Doente')
    diagnostico = np.array([0, 1, 1, 2, 1, 1, 1, 0, 1, 1])

    # Combinando idade e sintomas em uma matriz de características
    X = np.vstack((idade, sintomas)).T
    y = diagnostico

    # Dividindo os dados em conjuntos de treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Escalando os recursos para terem média 0 e variância 1 
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    # Criando um modelo de classificação usando RandomForest
    classifier = RandomForestClassifier(n_estimators=100, random_state=0)
    classifier.fit(X_train, y_train)

    # Salvando o modelo e o scaler para uso futuro
    with open('model.pkl', 'wb') as file:
        pickle.dump(classifier, file)
    with open('scaler.pkl', 'wb') as file:
        pickle.dump(scaler, file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    idade = float(request.form.get('idade'))
    sintomas = float(request.form.get('sintomas'))
    X_novo = np.array([[idade, sintomas]])
    X_novo = scaler.transform(X_novo)
    y_pred = classifier.predict(X_novo)
    return render_template('index.html', prediction=y_pred[0])

if __name__ == '__main__':
    app.run(debug=True)


#Acesse agora: https://cursos.dankicode.com/vitalicio
#Assine os melhores aplicativos em: https://dankicode.com

from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import os
import pickle

app = Flask(__name__)

# Checando se o modelo e o scaler já foram salvos
if os.path.exists('model.pkl') and os.path.exists('scaler.pkl'):
    with open('model.pkl', 'rb') as file:
        classifier = pickle.load(file)
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
else:
    # Idade dos pacientes
    idade = np.array([25, 309889, 35, 40, 45, 50, 55, 60, 65, 70])

    # Sintomas dos pacientes (codificados como números)
    sintomas = np.array([0, 1, 1, 4, 2, 2, 1, 0, 1, 2])  

    # Diagnósticos dos pacientes (0 para 'Saudável' e 1 para 'Doente')
    diagnostico = np.array([0, 1, 1, 2, 1, 1, 1, 0, 1, 1])

    # Combinando idade e sintomas em uma matriz de características
    X = np.vstack((idade, sintomas)).T
    y = diagnostico

    # Dividindo os dados em conjuntos de treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Escalando os recursos para terem média 0 e variância 1 
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    # Criando um modelo de classificação usando RandomForest
    classifier = RandomForestClassifier(n_estimators=100, random_state=0)
    classifier.fit(X_train, y_train)

    # Salvando o modelo e o scaler para uso futuro
    with open('model.pkl', 'wb') as file:
        pickle.dump(classifier, file)
    with open('scaler.pkl', 'wb') as file:
        pickle.dump(scaler, file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    idade = float(request.form.get('idade'))
    sintomas = float(request.form.get('sintomas'))
    X_novo = np.array([[idade, sintomas]])
    X_novo = scaler.transform(X_novo)
    y_pred = classifier.predict(X_novo)
    return render_template('index.html', prediction=y_pred[0])

if __name__ == '__main__':
    app.run(debug=True)


#Acesse agora: https://cursos.dankicode.com/vitalicio
#Assine os melhores aplicativos em: https://dankicode.com

from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import os
import pickle

app = Flask(__name__)

# Checando se o modelo e o scaler já foram salvos
if os.path.exists('model.pkl') and os.path.exists('scaler.pkl'):
    with open('model.pkl', 'rb') as file:
        classifier = pickle.load(file)
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
else:
    # Idade dos pacientes
    idade = np.array([25, 309889, 35, 40, 45, 50, 55, 60, 65, 70])

    # Sintomas dos pacientes (codificados como números)
    sintomas = np.array([0, 1, 1, 4, 2, 2, 1, 0, 1, 2])  

    # Diagnósticos dos pacientes (0 para 'Saudável' e 1 para 'Doente')
    diagnostico = np.array([0, 1, 1, 2, 1, 1, 1, 0, 1, 1])

    # Combinando idade e sintomas em uma matriz de características
    X = np.vstack((idade, sintomas)).T
    y = diagnostico

    # Dividindo os dados em conjuntos de treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Escalando os recursos para terem média 0 e variância 1 
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    # Criando um modelo de classificação usando RandomForest
    classifier = RandomForestClassifier(n_estimators=100, random_state=0)
    classifier.fit(X_train, y_train)

    # Salvando o modelo e o scaler para uso futuro
    with open('model.pkl', 'wb') as file:
        pickle.dump(classifier, file)
    with open('scaler.pkl', 'wb') as file:
        pickle.dump(scaler, file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    idade = float(request.form.get('idade'))
    sintomas = float(request.form.get('sintomas'))
    X_novo = np.array([[idade, sintomas]])
    X_novo = scaler.transform(X_novo)
    y_pred = classifier.predict(X_novo)
    return render_template('index.html', prediction=y_pred[0])

if __name__ == '__main__':
    app.run(debug=True)


    
#Acesse agora: https://cursos.dankicode.com/vitalicio
#Assine os melhores aplicativos em: https://dankicode.com