import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist, fashion_mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dense, Flatten, Reshape
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import seaborn as sns
from skimage.metrics import structural_similarity as ssim
import warnings
warnings.filterwarnings('ignore')

class DenoisingAutoencoderProject:
    def __init__(self):
        self.models = {}
        self.histories = {}
        self.noise_factors = [0.1, 0.3, 0.5, 0.7]
        
    def load_and_preprocess_data(self, dataset='mnist'):
        """Veri yükleme ve ön işleme"""
        print(f"📊 {dataset.upper()} veri seti yükleniyor...")
        
        if dataset == 'mnist':
            (x_train, y_train), (x_test, y_test) = mnist.load_data()
        else:
            (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
            
        # Normalizasyon
        x_train = x_train.astype("float32") / 255.0
        x_test = x_test.astype("float32") / 255.0
        
        # Reshape
        x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
        x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))
        
        print(f"✅ Veri yüklendi - Train: {x_train.shape}, Test: {x_test.shape}")
        return (x_train, y_train), (x_test, y_test)
    
    def add_noise(self, data, noise_factor):
        """Veri setine gürültü ekleme"""
        noisy_data = data + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=data.shape)
        return np.clip(noisy_data, 0., 1.)
    
    def create_simple_autoencoder(self, input_shape=(28, 28, 1)):
        """Basit Autoencoder modeli"""
        input_img = Input(shape=input_shape)
        
        # Encoder
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        encoded = MaxPooling2D((2, 2), padding='same')(x)
        
        # Decoder
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
        
        autoencoder = Model(input_img, decoded)
        encoder = Model(input_img, encoded)
        
        autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['mse'])
        return autoencoder, encoder
    
    def create_deep_autoencoder(self, input_shape=(28, 28, 1)):
        """Daha derin Autoencoder modeli"""
        input_img = Input(shape=input_shape)
        
        # Encoder
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
        encoded = MaxPooling2D((2, 2), padding='same')(x)
        
        # Decoder
        x = Conv2D(16, (3, 3), activation='relu', padding='same')(encoded)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
        
        autoencoder = Model(input_img, decoded)
        encoder = Model(input_img, encoded)
        
        autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['mse'])
        return autoencoder, encoder
    
    def calculate_psnr(self, original, denoised):
        """PSNR hesaplama"""
        mse = np.mean((original - denoised) ** 2)
        if mse == 0:
            return float('inf')
        max_pixel = 1.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
        return psnr
    
    def calculate_ssim(self, original, denoised):
        """SSIM hesaplama"""
        ssim_values = []
        for i in range(original.shape[0]):
            ssim_val = ssim(original[i].squeeze(), denoised[i].squeeze(), data_range=1.0)
            ssim_values.append(ssim_val)
        return np.mean(ssim_values)
    
    def train_model(self, model, x_train_noisy, x_train_clean, x_test_noisy, x_test_clean, 
                   model_name, epochs=50):
        """Model eğitimi"""
        print(f"🚀 {model_name} eğitimi başlıyor...")
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
        ]
        
        history = model.fit(x_train_noisy, x_train_clean,
                           epochs=epochs,
                           batch_size=128,
                           shuffle=True,
                           validation_data=(x_test_noisy, x_test_clean),
                           callbacks=callbacks,
                           verbose=1)
        
        print(f"✅ {model_name} eğitimi tamamlandı!")
        return history
    
    def evaluate_models(self, x_test_clean, x_test_noisy):
        """Model performanslarını değerlendirme"""
        results = {}
        
        for model_name, model in self.models.items():
            print(f"📊 {model_name} değerlendiriliyor...")
            
            # Tahmin
            decoded_imgs = model.predict(x_test_noisy, verbose=0)
            
            # Metrikler
            psnr = self.calculate_psnr(x_test_clean, decoded_imgs)
            ssim_score = self.calculate_ssim(x_test_clean, decoded_imgs)
            mse = np.mean((x_test_clean - decoded_imgs) ** 2)
            
            results[model_name] = {
                'psnr': psnr,
                'ssim': ssim_score,
                'mse': mse,
                'decoded_imgs': decoded_imgs
            }
            
            print(f"  PSNR: {psnr:.2f} dB")
            print(f"  SSIM: {ssim_score:.4f}")
            print(f"  MSE: {mse:.6f}")
        
        return results
    
    def test_noise_levels(self, x_train, x_test):
        """Farklı gürültü seviyelerinde test"""
        print("🔊 Farklı gürültü seviyelerinde test ediliyor...")
        
        noise_results = {}
        
        for noise_factor in self.noise_factors:
            print(f"\n📊 Gürültü faktörü: {noise_factor}")
            
            # Gürültülü veri oluştur
            x_train_noisy = self.add_noise(x_train, noise_factor)
            x_test_noisy = self.add_noise(x_test, noise_factor)
            
            # Basit model eğit
            model, _ = self.create_simple_autoencoder()
            history = self.train_model(model, x_train_noisy, x_train, 
                                     x_test_noisy, x_test, 
                                     f"Noise_{noise_factor}", epochs=30)
            
            # Değerlendir
            decoded_imgs = model.predict(x_test_noisy, verbose=0)
            psnr = self.calculate_psnr(x_test, decoded_imgs)
            ssim_score = self.calculate_ssim(x_test, decoded_imgs)
            
            noise_results[noise_factor] = {
                'psnr': psnr,
                'ssim': ssim_score,
                'model': model,
                'history': history,
                'decoded_imgs': decoded_imgs,
                'noisy_imgs': x_test_noisy
            }
        
        return noise_results
    
    def feature_extraction_classification(self, x_train, y_train, x_test, y_test, encoder):
        """Özellik çıkarımı ve sınıflandırma"""
        print("🔍 Feature extraction ve sınıflandırma testi...")
        
        # Encoder ile özellik çıkarımı
        train_features = encoder.predict(x_train, verbose=0)
        test_features = encoder.predict(x_test, verbose=0)
        
        # Flatten
        train_features = train_features.reshape(train_features.shape[0], -1)
        test_features = test_features.reshape(test_features.shape[0], -1)
        
        # Ham verilerle karşılaştırma için
        x_train_flat = x_train.reshape(x_train.shape[0], -1)
        x_test_flat = x_test.reshape(x_test.shape[0], -1)
        
        results = {}
        
        # Random Forest ile test
        print("  🌲 Random Forest ile test...")
        
        # Encoded features ile
        rf_encoded = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_encoded.fit(train_features, y_train)
        accuracy_encoded = rf_encoded.score(test_features, y_test)
        
        # Ham veriler ile
        rf_raw = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_raw.fit(x_train_flat, y_train)
        accuracy_raw = rf_raw.score(x_test_flat, y_test)
        
        results['random_forest'] = {
            'encoded_accuracy': accuracy_encoded,
            'raw_accuracy': accuracy_raw,
            'improvement': accuracy_encoded - accuracy_raw
        }
        
        print(f"    Encoded features: {accuracy_encoded:.4f}")
        print(f"    Raw features: {accuracy_raw:.4f}")
        print(f"    İyileştirme: {accuracy_encoded - accuracy_raw:.4f}")
        
        return results
    
    def plot_training_history(self, histories):
        """Eğitim geçmişini görselleştirme"""
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 2, 1)
        for name, history in histories.items():
            plt.plot(history.history['loss'], label=f'{name} - Train Loss')
            plt.plot(history.history['val_loss'], label=f'{name} - Val Loss')
        plt.title('Model Loss Comparison')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        for name, history in histories.items():
            if 'mse' in history.history:
                plt.plot(history.history['mse'], label=f'{name} - Train MSE')
                plt.plot(history.history['val_mse'], label=f'{name} - Val MSE')
        plt.title('Model MSE Comparison')
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def plot_noise_comparison(self, noise_results):
        """Gürültü seviyesi karşılaştırması"""
        noise_factors = list(noise_results.keys())
        psnr_values = [noise_results[nf]['psnr'] for nf in noise_factors]
        ssim_values = [noise_results[nf]['ssim'] for nf in noise_factors]
        
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(noise_factors, psnr_values, 'o-', linewidth=2, markersize=8)
        plt.title('PSNR vs Noise Level')
        plt.xlabel('Noise Factor')
        plt.ylabel('PSNR (dB)')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(noise_factors, ssim_values, 'o-', linewidth=2, markersize=8, color='orange')
        plt.title('SSIM vs Noise Level')
        plt.xlabel('Noise Factor')
        plt.ylabel('SSIM')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def plot_results(self, x_test_clean, x_test_noisy, results, n=10):
        """Sonuçları görselleştirme"""
        n_models = len(results)
        fig, axes = plt.subplots(n_models + 2, n, figsize=(20, 4 * (n_models + 2)))
        
        for i in range(n):
            # Orijinal görüntü
            axes[0, i].imshow(x_test_clean[i].reshape(28, 28), cmap='gray')
            axes[0, i].set_title('Original')
            axes[0, i].axis('off')
            
            # Gürültülü görüntü
            axes[1, i].imshow(x_test_noisy[i].reshape(28, 28), cmap='gray')
            axes[1, i].set_title('Noisy')
            axes[1, i].axis('off')
            
            # Model sonuçları
            for j, (model_name, result) in enumerate(results.items()):
                axes[j + 2, i].imshow(result['decoded_imgs'][i].reshape(28, 28), cmap='gray')
                axes[j + 2, i].set_title(f'{model_name}')
                axes[j + 2, i].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def run_complete_project(self, dataset='mnist'):
        """Tam proje akışını çalıştırma"""
        print("🎯 Denoising Autoencoder Projesi Başlatılıyor...")
        print("=" * 60)
        
        # 1. Veri yükleme
        (x_train, y_train), (x_test, y_test) = self.load_and_preprocess_data(dataset)
        
        # 2. Gürültü ekleme (ana test için)
        main_noise_factor = 0.5
        x_train_noisy = self.add_noise(x_train, main_noise_factor)
        x_test_noisy = self.add_noise(x_test, main_noise_factor)
        
        # 3. Modelleri oluştur ve eğit
        print("\n📋 Model Eğitimleri:")
        print("-" * 30)
        
        # Basit Autoencoder
        simple_model, simple_encoder = self.create_simple_autoencoder()
        simple_history = self.train_model(simple_model, x_train_noisy, x_train,
                                        x_test_noisy, x_test, "Simple Autoencoder", epochs=50)
        self.models['Simple'] = simple_model
        self.histories['Simple'] = simple_history
        
        # Derin Autoencoder
        deep_model, deep_encoder = self.create_deep_autoencoder()
        deep_history = self.train_model(deep_model, x_train_noisy, x_train,
                                      x_test_noisy, x_test, "Deep Autoencoder", epochs=50)
        self.models['Deep'] = deep_model
        self.histories['Deep'] = deep_history
        
        # 4. Model karşılaştırması
        print("\n📊 Model Performans Karşılaştırması:")
        print("-" * 40)
        results = self.evaluate_models(x_test, x_test_noisy)
        
        # 5. Farklı gürültü seviyelerinde test
        print("\n🔊 Gürültü Seviyesi Analizi:")
        print("-" * 35)
        noise_results = self.test_noise_levels(x_train[:5000], x_test[:1000])  # Küçük subset ile hızlı test
        
        # 6. Feature extraction test
        print("\n🔍 Feature Extraction ve Sınıflandırma:")
        print("-" * 45)
        classification_results = self.feature_extraction_classification(
            x_train[:5000], y_train[:5000], x_test[:1000], y_test[:1000], simple_encoder)
        
        # 7. Görselleştirmeler
        print("\n📈 Sonuçlar Görselleştiriliyor...")
        print("-" * 35)
        
        # Eğitim geçmişi
        self.plot_training_history(self.histories)
        
        # Gürültü karşılaştırması
        self.plot_noise_comparison(noise_results)
        
        # Sonuç görüntüleri
        self.plot_results(x_test, x_test_noisy, results)
        
        # 8. Özet rapor
        print("\n📋 PROJE ÖZET RAPORU")
        print("=" * 60)
        print("\n🎯 Ana Sonuçlar:")
        for model_name, result in results.items():
            print(f"  {model_name}:")
            print(f"    - PSNR: {result['psnr']:.2f} dB")
            print(f"    - SSIM: {result['ssim']:.4f}")
            print(f"    - MSE: {result['mse']:.6f}")
        
        print("\n🔊 Gürültü Analizi:")
        for noise_factor, result in noise_results.items():
            print(f"  Gürültü {noise_factor}: PSNR={result['psnr']:.2f}dB, SSIM={result['ssim']:.4f}")
        
        print("\n🔍 Sınıflandırma Sonuçları:")
        for classifier, result in classification_results.items():
            print(f"  {classifier}:")
            print(f"    - Encoded features: {result['encoded_accuracy']:.4f}")
            print(f"    - Raw features: {result['raw_accuracy']:.4f}")
            print(f"    - İyileştirme: {result['improvement']:.4f}")
        
        print("\n✅ Proje başarıyla tamamlandı!")
        
        return {
            'models': self.models,
            'results': results,
            'noise_results': noise_results,
            'classification_results': classification_results,
            'histories': self.histories
        }

# Projeyi çalıştırma
if __name__ == "__main__":
    project = DenoisingAutoencoderProject()
    
    # MNIST ile tam proje
    all_results = project.run_complete_project('mnist')
    
    # İsteğe bağlı: Fashion-MNIST ile de test edebilirsiniz
    # all_results_fashion = project.run_complete_project('fashion_mnist')