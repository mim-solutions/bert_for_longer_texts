# Roberta For Longer Texts

## Opis

Cele projektu:

1. Stworzenie wygodnego interfejsu modelu RoBERTa za pomocą klasy z metodami fit i predict
2. Implementacja modelu dla tekstów dłuższych niż 512 tokenów za pomocą max poolingu

## Instalacja środowiska
Biblioteka wymaga instalacji torcha w wersji kompatybilnej z maszyną i CUDA. Następnie instalujemy pozostałe paczki za pomocą ```bash env_setup.sh```. Bardziej szczegółowy opis znajduje się w [Setup środowiska dla nowego projektu](docs/setup_env.md).

## Ściągnięcie modelu
W pierwszej kolejności ściągamy model RoBERTa wytrenowany na korpusie języka polskiego. Jest to plik ```roberta_base_transformers.zip``` na  [stronie](https://github.com/sdadas/polish-roberta/releases). Po ściągnięciu, rozpakowujemy pliki. Ścieżkę do katalogu kopiujemy do pliku z konfiguracją ```config.py``` np. ```ROBERTA_PATH = "../resources/roberta"```

## Konfiguracja
W pliku ```config.py``` podajemy ścieżkę do ściągniętego modelu oraz podać GPU, na którym chcemy puszczać model. Można wybrać kilka GPU np. ```VISIBLE_GPUS = "0,2,3"```.

## Testy
Żeby upewnić się, że wszystko działa, puszczamy testy poleceniami ```python test/test_units.py``` oraz ```python test/test_model.py```.

## Przykład użycia

```
import pandas as pd
import numpy as np

from config import VISIBLE_GPUS

import os
os.environ["CUDA_VISIBLE_DEVICES"]= VISIBLE_GPUS
import torch

from sklearn.model_selection import train_test_split
from lib.roberta_main import RobertaClassificationModel

SAMPLE_DATA_PATH = 'test/sample_data/sample_data.csv'

# Loading data for tests
df = pd.read_csv(SAMPLE_DATA_PATH)

texts = df['sentence'].tolist() # list of texts
labels = df['target'].tolist() # list of 0/1 labels

# Train test split
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Loading model
model = RobertaClassificationModel()
# Fitting a model to training data for 5 epochs
model.fit(X_train,y_train,epochs = 5)
# Predicted probability for test set
preds = model.predict(X_test)

predicted_classes = (np.array(preds) >= 0.5)
accurate = sum(predicted_classes == np.array(y_test).astype(bool))
accuracy = accurate/len(y_test)

print(f'Test accuracy: {accuracy}')
```

 Wynik powyższego kodu:
 ```
Epoch: 0, Train accuracy: 0.5, Train loss: 0.6915183782577514
Epoch: 1, Train accuracy: 0.6125, Train loss: 0.668085078895092
Epoch: 2, Train accuracy: 0.75, Train loss: 0.6126224309206009
Epoch: 3, Train accuracy: 0.9, Train loss: 0.5083391539752483
Epoch: 4, Train accuracy: 0.9875, Train loss: 0.30732202231884004
Test accuracy: 1.0
 ```
