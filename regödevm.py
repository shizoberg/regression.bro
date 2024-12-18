import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import statsmodels.api as sm
import plotly.express as px
from pandastable import Table

def load_csv():
    """CSV dosyasını seçmek ve yüklemek."""
    global df  # DataFrame'i global olarak tanımlıyoruz
    file_path = filedialog.askopenfilename(
        title="CSV Dosyasını Seçin",
        filetypes=[("CSV Dosyaları", "*.csv"), ("Tüm Dosyalar", "*.*")]
    )
    if not file_path:
        return  # Kullanıcı dosya seçimini iptal etti
    try:
        # CSV dosyasını pandas ile yükle
        df = pd.read_csv(file_path)

        # NaN değerleri kontrol et ve kaldır
        df.dropna(inplace=True)

        # Sayısal olmayan sütunları kategori kodlarına dönüştür
        for col in df.select_dtypes(include=['object', 'category']).columns:
            df[col] = df[col].astype('category').cat.codes

        display_table(df)
    except Exception as e:
        messagebox.showerror("Hata", f"Dosya yüklenirken hata oluştu:\n{e}")

def display_table(dataframe):
    """Yüklenen veriyi tablo olarak görüntülemek için fonksiyon."""
    table_window = tk.Toplevel(root)
    table_window.title("CSV Verisi")
    frame = tk.Frame(table_window)
    frame.pack(fill=tk.BOTH, expand=1)

    # Pandastable kullanarak dataframe'i göster
    pt = Table(frame, dataframe=dataframe)
    pt.show()

    # Korelasyon Matrisi Butonu
    correlation_button = tk.Button(table_window, text="Korelasyon Matrisi Göster", command=show_correlation_matrix)
    correlation_button.pack(pady=10)

    # Regresyon Analizi Butonu
    analyze_button = tk.Button(table_window, text="Regresyon Analizi Yap", command=setup_regression)
    analyze_button.pack(pady=10)

def show_correlation_matrix():
    """Korelasyon matrisini plotly ile görselleştir."""
    if df is None:
        messagebox.showerror("Hata", "Lütfen önce bir CSV dosyası yükleyin.")
        return
    try:
        # Korelasyon matrisini hesapla
        corr_matrix = df.corr()

        # Plotly ile interaktif korelasyon matrisi
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            title="Korelasyon Matrisi",
            color_continuous_scale="Viridis"
        )
        fig.show()
    except Exception as e:
        messagebox.showerror("Hata", f"Korelasyon matrisi oluşturulurken hata oluştu:\n{e}")

def setup_regression():
    """Regresyon analizi için değişkenleri seçme arayüzü."""
    if df is None:
        messagebox.showerror("Hata", "Lütfen önce bir CSV dosyası yükleyin.")
        return

    regression_window = tk.Toplevel(root)
    regression_window.title("Regresyon Analizi")

    tk.Label(regression_window, text="Bağımlı Değişkeni Seçin:").pack(pady=5)
    dependent_var = ttk.Combobox(regression_window, values=list(df.columns), width=50)
    dependent_var.pack()

    tk.Label(regression_window, text="Bağımsız Değişkenleri Seçin (CTRL ile birden fazla seçim yapabilirsiniz):").pack(pady=5)
    independent_vars = tk.Listbox(regression_window, selectmode="multiple", width=50, height=10)
    for column in df.columns:
        independent_vars.insert(tk.END, column)
    independent_vars.pack()

    run_button = tk.Button(
        regression_window,
        text="Analizi Çalıştır",
        command=lambda: run_regression(dependent_var.get(), [independent_vars.get(i) for i in independent_vars.curselection()])
    )
    run_button.pack(pady=10)

def run_regression(dependent_var, independent_vars):
    """Seçilen değişkenlerle regresyon analizi yap ve sonucu göster."""
    if not dependent_var or not independent_vars:
        messagebox.showerror("Hata", "Lütfen bağımlı ve bağımsız değişkenleri seçin.")
        return

    try:
        # Sayısal sütunları kontrol et
        numeric_df = df.select_dtypes(include=['number'])
        if dependent_var not in numeric_df.columns or any(var not in numeric_df.columns for var in independent_vars):
            messagebox.showerror("Hata", "Lütfen yalnızca sayısal değişkenler seçin.")
            return

        # Bağımlı ve bağımsız değişkenleri al
        X = numeric_df[independent_vars]
        y = numeric_df[dependent_var]
        
        X = sm.add_constant(X)  # Sabit terim ekle
        model = sm.OLS(y, X).fit()
        results_summary = model.summary()

        # Sonuçları göstermek için yeni pencere aç
        results_window = tk.Toplevel(root)
        results_window.title("Regresyon Sonuçları")

        text_area = tk.Text(results_window, wrap=tk.WORD, width=100, height=30)
        text_area.insert(tk.END, results_summary)
        text_area.pack()
    except KeyError as e:
        messagebox.showerror("Hata", f"Sayısal olmayan sütunlar nedeniyle hata oluştu:\n{e}")
    except Exception as e:
        messagebox.showerror("Hata", f"Regresyon analizi sırasında hata oluştu:\n{e}")

# Ana Tkinter penceresi
root = tk.Tk()
root.title("CSV Görüntüleyici ve Regresyon Analizi")

# Yükleme düğmesi
load_button = tk.Button(root, text="CSV Yükle ve Görüntüle", command=load_csv)
load_button.pack(pady=20)

# Çıkış düğmesi
exit_button = tk.Button(root, text="Çıkış", command=root.quit)
exit_button.pack(pady=10)

# Arayüzü çalıştır
root.mainloop()