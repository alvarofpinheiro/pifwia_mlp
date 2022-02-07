#instalar biblioteca Orange Canvas
!pip install Orange3

#importar biblioteca Orange Canvas
import Orange

#importar dados do disco local
from google.colab import files  
files.upload()

#instanciar objeto de dados com base no caminho gerado com a importação do arquivo
dados = Orange.data.Table("/content/Simulacao-1.csv")

#imprimir os primeiros 5 registros
for d in dados[:5]:
  print(d)

#explorar os metadados
qtde_campos = len(dados.domain.attributes)
qtde_cont = sum(1 for a in dados.domain.attributes if a.is_continuous)
qtde_disc = sum(1 for a in dados.domain.attributes if a.is_discrete)
print("%d metadados: %d continuos, %d discretos" % (qtde_campos, qtde_cont, qtde_disc))
print("Nome dos metadados:", ", ".join(dados.domain.attributes[i].name for i in range(qtde_campos)),)

#explorar domínios
dados.domain.attributes

#explorar classe
dados.domain.class_var

#explorar dados
print("Campos:", ", ".join(c.name for c in dados.domain.attributes))
print("Registros:", len(dados))
print("Valor do registro 1 da coluna 1:", dados[0][0])
print("Valor do registro 2 da coluna 2:", dados[1][1])

#criar amostra
qtde100 = len(dados)
qtde70 = len(dados) * 70 / 100
qtde30 = len(dados) * 30 / 100
print("Qtde 100%:", qtde100)
print("Qtde  70%:", qtde70)
print("Qtde  30%:", qtde30)
amostra = Orange.data.Table(dados.domain, [d for d in dados if d.row_index < qtde70])
print("Registros:", len(dados))
print("Registros:", len(amostra))

#Técnica Multi-Layer Perceptron (MLP)
mlp = Orange.classification.NNClassificationLearner(hidden_layer_sizes=(100,), activation='relu', solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, preprocessors=None)

#ligar técnica MLP aos dados
dados_mlp = mlp(dados)

#treinar e testar técnica MLP com os dados
avalia_mlp = Orange.evaluation.CrossValidation(dados, [mlp], k=5)

#avaliar os indicadores para o SVM
print("Acurácia:  %.3f" % Orange.evaluation.scoring.CA(avalia_mlp)[0])
print("Precisão:  %.3f" % Orange.evaluation.scoring.Precision(avalia_mlp)[0])
print("Revocação: %.3f" % Orange.evaluation.scoring.Recall(avalia_mlp)[0])
print("F1:        %.3f" % Orange.evaluation.scoring.F1(avalia_mlp)[0])
print("ROC:       %.3f" % Orange.evaluation.scoring.AUC(avalia_mlp)[0])

#comparar a técnica MLP com outras 2 técnicas
svm = Orange.classification.SVMLearner(C=1.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, max_iter=-1, preprocessors=None)
knn = Orange.classification.KNNLearner(n_neighbors=5, metric='euclidean', weights='distance', algorithm='auto', metric_params=None, preprocessors=None)
dados_svm = svm(dados)
dados_knn = knn(dados)
aprendizado = [mlp, svm, knn]
avaliacao = Orange.evaluation.CrossValidation(dados, aprendizado, k=5)

#imprimir os indicadores para as 3 técnicas
print(" " * 10 + " | ".join("%-4s" % learner.name for learner in aprendizado))
print("Acurácia  %s" % " | ".join("%.2f" % s for s in Orange.evaluation.CA(avaliacao)))
print("Precisão  %s" % " | ".join("%.2f" % s for s in Orange.evaluation.Precision(avaliacao)))
print("Revocação %s" % " | ".join("%.2f" % s for s in Orange.evaluation.Recall(avaliacao)))
print("F1        %s" % " | ".join("%.2f" % s for s in Orange.evaluation.F1(avaliacao)))
print("ROC       %s" % " | ".join("%.2f" % s for s in Orange.evaluation.AUC(avaliacao)))
