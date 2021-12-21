# Transfer learning modelu RoBERTa

## Motywacja
Istniejące instrukcje na temat transfer learningu modeli typu BERT wymagają zgłębiania szczegółów dotyczących architektury modelu, tokenizacji, konkretnych bibliotek, etc.. Rzecz jasna, taka wiedza jest wartościowa, jednak z punktu widzenia użytkownika, przydatne jest posiadanie prostego narzędzia z minimalnym interfejsem do trenowania modelu i predykcji podając minimalne dane tj. surowe teksty oraz ich labele. Dzięki temu można dużo szybciej w pierwszej kolejności zbudować prototyp i zobaczyć, jakie daje wyniki.

Przykładowe alternatywne rozwiązania:
- [Transfer Learning for NLP: Fine-Tuning BERT for Text Classification](https://www.analyticsvidhya.com/blog/2020/07/transfer-learning-for-nlp-fine-tuning-bert-for-text-classification/) - złożona implementacja krok po kroku
- [Fine-tuning a pretrained model](https://huggingface.co/docs/transformers/training) - stosunkowa prosta implementacja, jednak nadal wymagana jest pewna znajomość biblioteki `transformers` oraz jawne przekształcanie tekstu i tworzenie datasetu

Zalety przygotowanego rozwiązania:
- inputem są minimalne surowe dane, tj. listy tekstów i oznaczeń. Wszelkie przekształcenia, tokenizacje, przygotowanie datasetów i dataloaderów dzieją się automatycznie
- minimalny interfejs - klasa modelu wyposażona w metody `fit` oraz `predict`.