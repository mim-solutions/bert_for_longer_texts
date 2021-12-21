# Roberta For Longer Texts

## Opis

Cele projektu:

1. Stworzenie wygodnego interfejsu modelu RoBERTa za pomocą klasy z metodami fit i predict
2. Implementacja modelu dla tekstów dłuższych niż 512 tokenów za pomocą max poolingu

## Instalacja środowiska
Biblioteka wymaga instalacji torcha w wersji kompatybilnej z maszyną i CUDA. Następnie instalujemy pozostałe paczki za pomocą ```bash env_setup.sh```. Bardziej szczegółowy opis znajduje się w [Setup środowiska dla nowego projektu](docs/setup_env.md).

## Ściągnięcie modelu
W pierwszej kolejności ściągamy model RoBERTa wytrenowany na korpusie języka polskiego. Jest to plik ```roberta_base_transformers.zip``` na  [stronie](https://github.com/sdadas/polish-roberta/releases). Po ściągnięciu, rozpakowujemy pliki. Ścieżkę do katalogu kopiujemy do pliku z konfiguracją ```config.py``` np. ```ROBERTA_PATH = "../resources/roberta"```.

## Konfiguracja
W pliku ```config.py``` podajemy ścieżkę do ściągniętego modelu oraz podać GPU, na którym chcemy puszczać model. Można wybrać kilka GPU np. ```VISIBLE_GPUS = "0,2,3"```.

## Testy
Żeby upewnić się, że wszystko działa, puszczamy testy poleceniem ```python3 -m pytest test```.

## Przykład użycia - metody fit i predict

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

Wynik powyższego kodu (przykładowy, może się różnić z uwagi na losowość):
 ```
Epoch: 0, Train accuracy: 0.590625, Train loss: 0.6770698979496956
Epoch: 1, Train accuracy: 0.721875, Train loss: 0.58055414929986
Epoch: 2, Train accuracy: 0.89375, Train loss: 0.3515076955780387
Epoch: 3, Train accuracy: 0.9125, Train loss: 0.2523562053218484
Epoch: 4, Train accuracy: 0.940625, Train loss: 0.164382476080209
Test accuracy: 0.925
 ```

## Przykład użycia - metoda train_and_evaluate

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
result = model.train_and_evaluate(X_train, X_test, y_train, y_test,epochs = 5)
```

 Wynik powyższego kodu (przykładowy, może się różnić z uwagi na losowość):

```
Epoch: 0, Train accuracy: 0.5875, Train loss: 0.6660354651510716
Epoch: 0, Test accuracy: 0.65, Test loss: 0.5904278859496117
Epoch: 1, Train accuracy: 0.78125, Train loss: 0.5293301593512296
Epoch: 1, Test accuracy: 0.925, Test loss: 0.35317784547805786
Epoch: 2, Train accuracy: 0.88125, Train loss: 0.34443826507776976
Epoch: 2, Test accuracy: 0.95, Test loss: 0.1830226019024849
Epoch: 3, Train accuracy: 0.9375, Train loss: 0.20902621131390334
Epoch: 3, Test accuracy: 0.9375, Test loss: 0.17358638979494573
Epoch: 4, Train accuracy: 0.96875, Train loss: 0.12159209074452519
Epoch: 4, Test accuracy: 0.95, Test loss: 0.12857716977596284
```

Dodatkowo metryki są zapisane w zmiennej `result`.