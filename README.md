# Gürültü Giderme Otokodlayıcı Projesi

Görüntü restorasyonu için gürültü giderme otokodlayıcılarının kapsamlı bir uygulaması - çoklu gürültü türleri ve performans değerlendirme metrikleri içerir.

## 🎯 Proje Genel Bakış

Bu proje, MNIST rakam görüntülerinden çeşitli gürültü türlerini kaldırmak için TensorFlow/Keras kullanarak gürültü giderme otokodlayıcılarını uygular ve değerlendirir. Sistem, gerçek dünya görüntü işleme senaryolarında yaygın olarak bulunan farklı gürültü desenlerine karşı konvolüsyonel otokodlayıcıların etkinliğini karşılaştırır.

## 🔧 Özellikler

- **Çoklu Gürültü Türleri**: Gauss, Tuz ve Biber, Benek, Poisson ve Düzgün gürültü
- **Konvolüsyonel Mimari**: CNN tabanlı kodlayıcı-kod çözücü kullanan derin öğrenme yaklaşımı
- **Performans Metrikleri**: PSNR ve SSIM değerlendirmesi
- **Kapsamlı Görselleştirme**: Öncesi/sonrası karşılaştırmaları ve performans grafikleri
- **Modüler Tasarım**: Temiz, nesne yönelimli uygulama

## 📊 Temel Sonuçlar

### Gürültü Türüne Göre Performans Karşılaştırması

| Gürültü Türü | PSNR (dB) | SSIM Skoru | Zorluk Seviyesi |
|---------------|-----------|------------|------------------|
| **Poisson** | 14.90 | 0.6411 | ⭐⭐ Kolay |
| **Düzgün** | 14.72 | 0.5885 | ⭐⭐ Kolay |
| **Benek** | 13.91 | 0.5171 | ⭐⭐⭐ Orta |
| **Tuz ve Biber** | 12.76 | 0.2789 | ⭐⭐⭐⭐ Zor |
| **Gauss** | 12.15 | 0.1605 | ⭐⭐⭐⭐⭐ En Zor |

### Temel Bulgular

1. **Poisson Gürültüsü**: En iyi yeniden yapılandırma kalitesi (PSNR: 14.90 dB, SSIM: 0.6411)
2. **Gauss Gürültüsü**: Gürültü giderme için en zorlayıcı (PSNR: 12.15 dB, SSIM: 0.1605)
3. **Tuz ve Biber**: Orta seviye PSNR'ye rağmen düşük yapısal benzerlik gösterir
4. **Eğitim Verimliliği**: Tüm modeller erken durdurma kullanarak 5 epoch içinde yakınsadı

## 🏗️ Mimari

### Otokodlayıcı Yapısı
```
Giriş (28x28x1) → Conv2D(32) → MaxPool → Conv2D(32) → MaxPool → 
Kodlanmış (7x7x32) → Conv2D(32) → UpSample → Conv2D(32) → UpSample → 
Çıkış (28x28x1)
```

### Teknik Özellikler
- **Framework**: TensorFlow 2.x + Keras
- **Optimizer**: Adam
- **Kayıp Fonksiyonu**: Binary Crossentropy
- **Metrikler**: MSE, PSNR, SSIM
- **Callback'ler**: Erken Durdurma, Öğrenme Oranı Azaltma

## 📈 Performans Metrikleri

### PSNR (Tepe Sinyal-Gürültü Oranı)
- **Aralık**: Yüksek değerler daha iyi kaliteyi gösterir
- **En İyi Performans**: Poisson gürültüsü (14.90 dB)
- **En Kötü Performans**: Gauss gürültüsü (12.15 dB)

### SSIM (Yapısal Benzerlik İndeksi)
- **Aralık**: 0-1 (1 = mükemmel benzerlik)
- **En İyi Performans**: Poisson gürültüsü (0.6411)
- **En Kötü Performans**: Gauss gürültüsü (0.1605)

## 🛠️ Kurulum ve Kullanım

### Önkoşullar
```bash
pip install tensorflow numpy matplotlib scikit-image scikit-learn seaborn
```

### Projeyi Çalıştırma
```python
# Projeyi başlat
project = DenoisingAutoencoderProject()

# Veriyi yükle ve ön işle
(x_train, y_train), (x_test, y_test) = project.load_and_preprocess_data('mnist')

# Kapsamlı gürültü karşılaştırma deneyini çalıştır
results = project.run_noise_comparison_experiment()

# Sonuçları görselleştir
plot_performance_comparison(results)
```

## 💻 Google Colab Desteği

Bu proje Google Colab üzerinde çalıştırılmaya uygundur
https://colab.research.google.com/drive/1Tn9DYV53gmAEq9SDVk-1d3a3-kakR08M?usp=sharing

## 🔍 Gürültü Türleri Açıklaması

### 1. Gauss Gürültüsü
- **Özellikler**: Normal dağılım izleyen eklemeli beyaz gürültü
- **Gerçek Dünya**: Sensör gürültüsü, termal gürültü
- **Zorluk Seviyesi**: En Yüksek (tüm frekans spektrumuna yayılır)

### 2. Tuz ve Biber Gürültüsü
- **Özellikler**: Rastgele siyah/beyaz pikseller
- **Gerçek Dünya**: İletim hataları, sensör arızaları
- **Zorluk Seviyesi**: Yüksek (keskin süreksizlikler oluşturur)

### 3. Benek Gürültüsü
- **Özellikler**: Çarpmalı gürültü
- **Gerçek Dünya**: Radar, tıbbi görüntüleme (ultrason)
- **Zorluk Seviyesi**: Orta (sinyal bağımlı)

### 4. Poisson Gürültüsü
- **Özellikler**: Atış gürültüsü, sinyal bağımlı
- **Gerçek Dünya**: Düşük ışık fotoğrafçılığı, tıbbi görüntüleme
- **Zorluk Seviyesi**: Düşük (sinyal istatistiklerini takip eder)

### 5. Düzgün Gürültü
- **Özellikler**: Düzgün dağılımla eklemeli gürültü
- **Gerçek Dünya**: Nicemleme hataları, ADC gürültüsü
- **Zorluk Seviyesi**: Düşük (sınırlı frekans aralığı)

## 📊 Görsel Sonuçlar

Proje üç tür görselleştirme oluşturur:

1. **Gürültü Karşılaştırma Tablosu**: Orijinal görüntüyü 5 gürültü türüyle gösterir
2. **Gürültü Giderme Sonuçları**: Her gürültü türü için öncesi/sonrası karşılaştırması
3. **Performans Grafikleri**: PSNR ve SSIM karşılaştırma çubuk grafikleri

## 🎯 Model Performans Analizi

### Eğitim Özellikleri
- **Yakınsama**: Hızlı yakınsama (3-5 epoch)
- **Kararlılık**: Çalıştırmalar arası tutarlı performans
- **Genelleme**: Görülmemiş test verilerinde iyi performans

### Mimari İçgörüler
- MNIST için basit CNN mimarisi yeterli
- Uzamsal boyut azaltma için MaxPooling etkili
- Uygun piksel değer aralığı için Sigmoid aktivasyon kritik


### Yazılım Bağımlılıkları
- Python 3.7+
- TensorFlow 2.x
- NumPy, Matplotlib, Scikit-learn
- Görselleştirme için Seaborn


## 📜 Lisans

Bu proje MIT Lisansı altında lisanslanmıştır - detaylar için LICENSE dosyasına bakın.

## 🔗 Referanslar

- [Gürültü Giderme Otokodlayıcıları](https://www.cs.toronto.edu/~larocheh/publications/icml-2008-denoising-autoencoders.pdf)
- [Görüntü Kalitesi Değerlendirme](https://ieeexplore.ieee.org/document/1284395)
- [Konvolüsyonel Otokodlayıcılar](https://blog.keras.io/building-autoencoders-in-keras.html)


