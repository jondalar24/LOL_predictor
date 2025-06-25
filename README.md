
# LOL Predictor - ¿Puede una red neuronal predecir quién ganará en League of Legends?

Este proyecto implementa una red neuronal simple en PyTorch con el objetivo de predecir el resultado de una partida de **League of Legends** (win/loss) basándose en estadísticas numéricas obtenidas del final de la partida.

---

## Objetivo

Explorar los límites del aprendizaje automático en contextos complejos como los videojuegos multijugador, donde existen múltiples factores humanos, tácticos y situacionales que afectan al resultado.

---

## Contenido del proyecto

- `logistic_regression_lol.py`: script principal con todo el flujo de procesamiento, entrenamiento y evaluación del modelo.
- `league_of_legends_data_large.csv`: dataset con estadísticas de partidas.
- Imágenes de salida:
  - `confusion_matrix.png`
  - `roc_curve.png`
  - `feature_importance.png`

---

##  Enfoque técnico

- Librerías: `PyTorch`, `Scikit-Learn`, `Matplotlib`, `Pandas`
- Red neuronal: `Linear(input_dim, 1)` + `Sigmoid`
- Optimización: SGD + momentum + weight decay (L2)
- Métricas evaluadas:
  - Matriz de confusión
  - Reporte de clasificación
  - Curva ROC
  - AUC
  - Importancia de características (pesos)

---

##  Resultados

```plaintext
Train Accuracy: 0.5275
Test Accuracy:  0.5400
AUC:            0.58
```

-  El modelo tiene dificultades para generalizar.
-  Ajustes como learning rate, momentum y número de épocas no han mejorado sustancialmente el rendimiento.
-  La AUC cercana a 0.5 sugiere que el modelo apenas supera al azar.

---

## Reflexiones

> Los resultados sugieren que **las estadísticas post-partida no son suficientes para predecir con precisión el resultado**.  
> Elementos como decisiones tácticas en tiempo real, habilidades humanas, atención y contexto escapan a los datos recogidos.  
> A veces, el problema no es el modelo, sino **la falta de representación completa del fenómeno a modelar**.

---

##  ¿Quieres probarlo?

1. Clona este repositorio:
```bash
git clone https://github.com/jondalar24/LOL_predictor.git
```

2. Instala los requisitos:
```bash
pip install torch scikit-learn pandas matplotlib
```

3. Añade el dataset `league_of_legends_data_large.csv` al directorio raíz.

4. Ejecuta el script:
```bash
python logistic_regression_lol.py
```

---

##  ¿Ideas para mejorar?

Estoy abierto a propuestas. Puedes escribirme por LinkedIn o abrir un issue en este repositorio.

---

##  Hashtags sugeridos para compartir:

```
#MachineLearning #DeepLearning #PyTorch #IA #LeagueOfLegends #GamingAnalytics 
#AprendizajeAutomatico #DataScience #OpenToLearning #NeuralNetworks
```
