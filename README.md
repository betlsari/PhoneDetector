\# PhoneDetector



Python, OpenCV ve YOLO kullanarak görüntülerde cep telefonu tespiti yapan bilgisayarlı görü projesi.



\## Özellikler



\* Görüntülerde cep telefonu tespiti

\* YOLO modeliyle tahmin

\* Tespit sonuçlarını görüntüye kaydetme

\* CPU üzerinde çalışabilme



\## Gereksinimler



\* Python 3.11+

\* Ultralytics

\* OpenCV

\* NumPy



\## Kurulum



Projeyi klonladıktan sonra proje klasöründe aşağıdaki komutları çalıştırın:



```bash

python -m venv .venv

```



Git Bash üzerinde sanal ortamı etkinleştirmek için:



```bash

source .venv/Scripts/activate

```



Gerekli paketleri yükleyin:



```bash

pip install -r requirements.txt

```



\## Kullanım



```bash

python main.py --image "gorsel.jpg" --output "sonuc.jpg"

```



\*\*Not:\*\* Eğitilmiş model ve veri seti bu depoya dahil değildir. Model dosyası, `main.py` içinde tanımlanan konumda bulunmalıdır.



\## Teknolojiler



\* Python

\* Ultralytics YOLO

\* OpenCV



