# Setup środowiska dla nowego projektu

0. Uruchamiamy condę. Na kulfonie poleceniem:
```
source /opt/tljh/user/bin/activate
```

1. Tworzymy nowe środowisko. Instalujemy 'ipykernel', jeśli korzystamy JupyterHub:
```
conda create --name roberta_for_longer_texts python=3.8 pip ipykernel
```

2. Aktywujemy środowisko:
```
conda activate roberta_for_longer_texts
```

3. Instalujemy Pytorch oraz cudatoolkit. Ten punkt niestety zależy od maszyny: sprawdzamy wersję driverów GPU (jeśli w ogóle mamy) poleceniem `nvidia-smi` (np. na kulfonie `470.63.01`), i wybieramy najnowszą wersję cudatoolkit kompatybilną z tymi driverami wg [tej tabelki](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html) (np. 11.1).
Jednocześnie condą instalujemy pytorcha, żeby dostać kompatybilny build. Rekomendujemy dodać wymagania minimalnej wersji oraz tego by pakiet pytorch pochodził z kanału pytorch, jak poniżej (inaczej wszelkie dalsze zmiany w condzie mogą nam niechcący podmienić na niższą wersję lub wersję cpu zamiast cuda).
Cudzysłów jest konieczny (inaczej bash interpretuje `>` jako przekierowanie do pliku `=1.9`).

Przykładowa instalacja na szklance (starsze sterowniki)

```
conda install cudatoolkit=10.1 "pytorch::pytorch>=1.8" "torchvision>=0.9" -c pytorch -c conda-forge
```

Instalacja na kulfonie:

```
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
```

4. *Konfiguracja na gcloud/dockerfile sprowadza się do tego kroku (zakładając, że w obrazach mamy już Pytorcha i cudatoolkit)*. Z uwagi na kwadratowy algorytm rozwiązywania zależności przez condę, jak również rozmaite problemy cross-platformowe, staramy się instalować pip-em. Instalujemy tylko główne paczki. W tym celu odpalamy z poziomu root-a projektu:

```
bash env_setup.sh
```

5. Jeśli doinstalowujemy nową paczkę i będziemy z niej korzystać w projekcie (albo zauważyliśmy, że jakiejś brakuje), powinniśmy ją dodać w odpowiednim miejscu w pliku `./env_setup.sh`.