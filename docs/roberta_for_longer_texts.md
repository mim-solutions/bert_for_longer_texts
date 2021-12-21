# Modyfikacja modelu dla dłuższych tekstów

## Motywacja
Domyślnie modele typu BERT mogą przetwarzać tekst o maksymalnej długości 512 tokenów (w dużym przybliżeniu 1 token odpowiada 1 słowu). Jest wymuszone samą architekturą sieci neuronowej i nie da się łatwo obejść. Dyskusję na ten temat można znaleźć [tutaj](https://github.com/google-research/bert/issues/27).

## Metoda
Metoda została opisana przez Devlina (jednego z autorów BERTa) w powyższej dyskusji: [komentarz](https://github.com/google-research/bert/issues/27#issuecomment-435265194).
Zgodnie z powyższym, schemat działania jest następujący:
1. Przygotowanie pojedynczego tekstu - postępujemy według [instrukcji](https://www.kdnuggets.com/2021/04/apply-transformers-any-length-text.html)
- zamieniamy cały tekst na tokeny
- dzielimy ciąg tokenów na kawałki długości `size`, nakładające się na siebie z krokiem `step`, minimalna długość kawałka wynosi `minimal_length`
- dla każdego kawałka dodajemy specjalne tokeny na początku i końcu
- używamy padding tokens, żeby każdy kawałek miał tyle samo tokenów
- sklejamy tak przetworzone kawałki w jeden tensor za pomocą torch.stack
2. Ewaluacja modelu
- tak utworzony tensor przepuszczamy przez model jako mini-batch
- otrzymujemy N prawdopodobieństw (N - liczba kawałków tekstu)
- ostateczne prawdopodobieństwo otrzymujemy jako średnia/maximum z prawdopodobieństw (w zależności od hiperparametru `pooling_strategy`)
3. Transfer learning
- przy trenowaniu modelu robimy te same kroki, co w punkcie 2, kluczowe jest, żeby wszystkie operacje typu `cat/stack/split/mean/max` były wykonywane na tensorach z liczonym gradientem, żeby `loss.backward()` działało poprawnie. Wszelkie konwersje na listy/arraye nie są dozwolone
- ponieważ liczba kawałków dla poszczególnych tekstów może być różna, teksty po przetworzeniu w punkcie 1 są tensorami o zmiennej wielkości. Domyślny `DataLoader` w torchu nie dopuszcza takiej możliwości. Stąd tworzymy customowy `DataLoader` z nadpisaną metodą `collate_fn` - problem jest opisany [tutaj](https://discuss.pytorch.org/t/dataloader-for-various-length-of-data/6418)

## Uwagi
- wspomniane hiperparametry modelu `size`, `step`, `minimal_length`, `pooling_strategy` są parametrami klasy `RobertaClassificationModelWithPooling`. Ich domyślne wartości można też zmienić w pliku `config.py`