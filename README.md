# eksploracja-danych

## Instrukcja użycia
Głównym program jest zawarty w procedurze `main()` z pliku `main.py`. Można go także przywołać z poziomu `explore.ipynb`.

Upewnij się że lemmatizer i lista stopwordów są pobrane. W razie potrzeby można to zrobić za pomocą procedur `download_lemmatizer()` i `download_stopwords()` z `src/utils/processing.py` (także dostępne z poziomu `explore.ipynb`).

### Uruchamianie programu
`main(N_UNIGRAMS: int, TRAIN_SIZE: int, TEST_SIZE: int, show_timer: bool = False, use_linear_svm: bool = False)`
* `N_UNIGRAMS` - liczba najczęściej występujących unigramów, która jest brana pod uwagę przy tworzeniu wektorów obecności unigramów
* `TRAIN_SIZE` - liczba recenzji pozytywnych/negatywnych w zbiorze trenującym (rozmiar całego zbioru trenującego będzie wynosić dwukrotność tego parametru)
* `TEST_SIZE` - liczba recenzji pozytywnych/negatywnych w zbiorze testowym (rozmiar całego zbioru testowego będzie wynosić dwukrotność tego parametru)
* `show_timer` - powoduje dodatkowo pokazanie czasu przetwarzania poszczególnych części programu
* `use_linear_svm` - zmienia używaną implementację SVM z `sklearn.svm.SVC` na `sklearn.svm.LinearSVC`, którego szybkość uczenia się znacznie lepiej skaluje się z rozmiarem zbioru trenującego