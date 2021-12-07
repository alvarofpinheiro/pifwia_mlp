#MultiLayer Perceptron (MLP)

#Enquanto o perceptron de uma unica camada possa resolver problemas lineares, ele é incapaz de resolver problemas mais complexos. Para isso podemos utilizar o Perceptron Multicamadas.

#Etapas do Perceptron Multicamadas:

#1º Passo: Inicialização**
# Atribuir valores aleatórios para os pesos e viés

#2º Passo: Ativação**
# Calcular os valores dos neurônios da camada oculta
# Calcular os valores dos neurônios da camada de saída

#3º Passo: Treinar os Pesos**
# Calcular os erros dos neurônios das camadas de saída e oculta
# Calcular a correção dos pesos
# Atualizar os pesos dos neurônios das camadas de saída e oculta

#4º Passo: Iteração**
# Repetir o processo a partir do passo 2 até que satisfaça o critério de erro

import random
import math

# Os indices 0, 1 e 2 são entradas e o ultimo é a saida
entradas = [[0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1],]
resultado_esperado = [1,1,1,1,0,0,1,1]

numero_inputs = len(entradas[0])
numero_output = 1
camada_oculta = 5
taxa_de_aprendizado = 0.01

# treinamento da rede
pesos_inputs_hidden = []
pesos_hidden_output = []
bias_hidden = []
bias_output = []

for i in range(numero_inputs * camada_oculta):
  pesos_inputs_hidden.append(random.random()*2 -1)

for i in range(numero_output * camada_oculta):
  pesos_hidden_output.append(random.random()*2 -1)

for i in range(camada_oculta):
  bias_hidden.append(random.random()*2 -1)

for i in range(numero_output):
  bias_output.append(random.random()*2 -1)

def sigmoid(x):
  return 1.0/(1.0  + math.exp(-x))

def sigmoid_derivada(x):
  # No caso x deve ser um valor q ja passou pela sigmoid
  return x * (1 - x)

def valor_saida(entrada, pesos, indice, bias, tamanho):
  valor_camada = 0
  for i in range(0, len(entrada)):
    valor_camada += pesos[i*tamanho + indice]*entrada[i]
  valor_camada += bias
  valor_camada = sigmoid(valor_camada)
  return valor_camada

def gerar_solucao(entrada, pesos_input, pesos_output, bias_hidden, bias_output):

  # dada uma entrada gera uma camada oculta
  camadas_oculta = []
  for i in range(0, camada_oculta):
    camadas_oculta.append(valor_saida(entrada, pesos_input, i, bias_hidden[i], camada_oculta))
  
  # dada uma camada oculta gera um output
  output = valor_saida(camadas_oculta, pesos_hidden_output, 0, bias_output[0], numero_output)

  return output

def treinar_rede(inputs, solucoes, pesos_inputs_hidden, pesos_hidden_output):
  
  for p in range(0, 5000):
    for i in range(0, len(inputs)):
      # dada uma entrada gera uma camada oculta
      camadas_oculta = []
      for j in range(0, camada_oculta):
        camadas_oculta.append(valor_saida(inputs[i], pesos_inputs_hidden, j, bias_hidden[j], camada_oculta))
    
      # dada uma camada oculta gera um output
      output = valor_saida(camadas_oculta, pesos_hidden_output, 0, bias_output[0], numero_output)

      # calcula o feedback do output para a camada oculta
      erro = solucoes[i] - output
      saida = sigmoid_derivada(output) * erro * taxa_de_aprendizado
      bias_output[0] += saida

      # calcula o feedback da camada oculta para os inputs
      erros_camada_oculta = []
      saida_oculta = []
      peso_hidden_total = sum(pesos_hidden_output)
      for j in range(0, camada_oculta):
        erros_camada_oculta.append(pesos_hidden_output[j]/peso_hidden_total)
        erros_camada_oculta[-1] *= erro
        saida_oculta.append(sigmoid_derivada(camadas_oculta[j]))
        saida_oculta[-1] *= erros_camada_oculta[-1]
        saida_oculta[-1] *= taxa_de_aprendizado

      # atualiza pesos entre hidden e output
      for j in range(0, len(pesos_hidden_output)):
        pesos_hidden_output[j] += saida * camadas_oculta[j]

      # atualiza pesos entre input e hidden
      for k in range(0, camada_oculta):
        for j in range(0, len(inputs[i])):
          pesos_inputs_hidden[j*camada_oculta + k] += saida_oculta[k] * inputs[i][j]
        bias_hidden[k] += saida_oculta[k]

  resultado = []
  for i in range(0, len(inputs)):
    aux = gerar_solucao(inputs[i], pesos_inputs_hidden, pesos_hidden_output, bias_hidden, bias_output)
    resultado.append(aux)
  return resultado

# testes
correto = 0
for i in range (0, 10):
  pesos_inputs_hidden = []
  pesos_hidden_output = []
  bias_hidden = []
  bias_output = []

  for j in range(numero_inputs * camada_oculta):
    pesos_inputs_hidden.append(random.random()*2 -1)

  for j in range(numero_output * camada_oculta):
    pesos_hidden_output.append(random.random()*2 -1)

  for j in range(camada_oculta):
    bias_hidden.append(random.random()*2 -1)

  for j in range(numero_output):
    bias_output.append(random.random()*2 -1)


  aux = treinar_rede(entradas,resultado_esperado, pesos_inputs_hidden, pesos_hidden_output)

  equal = True
  for j in range(len(aux)):
    if aux[j] > 0.5:
      aux[j] = 1
    else: 
      aux[j] = 0
      
    if aux[j] != resultado_esperado[j]:
      equal = False
  if equal:
    correto += 1
  print(aux)
print(correto)