import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import shapiro
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
import numpy as np
from statsmodels.formula.api import ols
import openai
from statsmodels.stats.anova import anova_lm

# Global variables
data = None
dependent_var = None

# Function to load dataset
def load_file():
    global data
    file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    if file_path:
        try:
            data = pd.read_csv(file_path)
            clean_data()
            update_columns()
            messagebox.showinfo("Success", "File loaded and cleaned successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Error loading file: {e}")

# Function to clean dataset
def clean_data():
    global data
    if data is not None:
        # Remove rows with NaN or inf values
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data.dropna(inplace=True)

# Function to update column dropdowns
def update_columns():
    if data is not None:
        numerical_cols = list(data.select_dtypes(include=['number']).columns)
        dependent_var_dropdown['values'] = numerical_cols

# Function to set dependent variable
def set_dependent_var(event):
    global dependent_var
    dependent_var = dependent_var_dropdown.get()

# Function to perform regression
def perform_regression():
    if data is not None and dependent_var:
        try:
            independent_vars = [col for col in data.select_dtypes(include=['number']).columns if col != dependent_var]
            X = data[independent_vars]
            y = data[dependent_var]
            X = sm.add_constant(X)
            model = sm.OLS(y, X).fit()
            text_output.delete("1.0", tk.END)
            text_output.insert(tk.END, model.summary().as_text())
            return model
        except Exception as e:
            messagebox.showerror("Error", f"Error performing regression: {e}")
    else:
        messagebox.showerror("Error", "No dependent variable selected or dataset loaded!")

# Function to plot diagnostic plots
def plot_diagnostics():
    model = perform_regression()
    if model:
        residuals = model.resid
        fitted = model.fittedvalues
        
        # Residuals vs Fitted
        plt.figure(figsize=(8, 6))
        sns.residplot(x=fitted, y=residuals, lowess=True, line_kws={'color': 'red'})
        plt.title("Residuals vs Fitted")
        plt.xlabel("Fitted values")
        plt.ylabel("Residuals")
        plt.axhline(0, color='black', linestyle='--', linewidth=1)
        plt.grid(alpha=0.7, linestyle='--')
        plt.show()
        
        # Normal Q-Q Plot
        sm.qqplot(residuals, line="45", fit=True)
        plt.title("Normal Q-Q")
        plt.grid(alpha=0.7, linestyle='--')
        plt.show()
        
        # Scale-Location Plot
        plt.figure(figsize=(8, 6))
        plt.scatter(fitted, abs(residuals) ** 0.5, alpha=0.6)
        plt.title("Scale-Location")
        plt.xlabel("Fitted values")
        plt.ylabel("√|Residuals|")
        plt.axhline(0, color='red', linestyle='--', linewidth=1)
        plt.grid(alpha=0.7, linestyle='--')
        plt.show()
        
        # Residuals vs Leverage
        influence = model.get_influence()
        leverage = influence.hat_matrix_diag
        cooks_d, _ = influence.cooks_distance
        
        plt.figure(figsize=(8, 6))
        plt.scatter(leverage, residuals, alpha=0.6)
        plt.title("Residuals vs Leverage")
        plt.xlabel("Leverage")
        plt.ylabel("Residuals")
        plt.grid(alpha=0.7, linestyle='--')
        plt.show()

# Function to check outliers using Cook's Distance
def check_outliers():
    model = perform_regression()
    if model:
        influence = model.get_influence()
        cooks_d, _ = influence.cooks_distance
        leverage = influence.hat_matrix_diag

        # Cook's Distance plot
        plt.figure(figsize=(8, 6))
        plt.stem(range(len(cooks_d)), cooks_d, markerfmt=",", basefmt=" ")
        plt.title("Cook's Distance")
        plt.xlabel("Observation Index")
        plt.ylabel("Cook's Distance")
        plt.axhline(0.5, color='red', linestyle='--', label="Threshold = 0.2")
        plt.legend()
        plt.grid(alpha=0.7, linestyle='--')
        plt.show()

        # Text output for high leverage points
        critical_leverage = 2 * (model.df_model + 1) / len(data)
        high_leverage_points = [i for i, h in enumerate(leverage) if h > critical_leverage]
        text_output.delete("1.0", tk.END)
        text_output.insert(tk.END, f"Max Cook's Distance: {max(cooks_d):.4f}\n")
        text_output.insert(tk.END, f"Critical Leverage Value: {critical_leverage:.4f}\n")
        text_output.insert(tk.END, f"High Leverage Points: {high_leverage_points}\n")

# Function to test for homoscedasticity (constant variance)
def test_homoscedasticity():
    model = perform_regression()
    if model:
        residuals = model.resid
        fitted = model.fittedvalues
        
        # Breusch-Pagan Test
        test_result = het_breuschpagan(residuals, model.model.exog)
        bp_stat, bp_pvalue, _, _ = test_result
        
        # Display results
        text_output.delete("1.0", tk.END)
        text_output.insert(tk.END, "Breusch-Pagan Test for Homoscedasticity\n")
        text_output.insert(tk.END, f"Test Statistic: {bp_stat:.4f}\n")
        text_output.insert(tk.END, f"P-Value: {bp_pvalue:.4e}\n")
        if bp_pvalue < 0.05:
            text_output.insert(tk.END, "Conclusion: Heteroscedasticity detected (variance is not constant).\n")
        else:
            text_output.insert(tk.END, "Conclusion: No evidence of heteroscedasticity (variance is constant).\n")

        # Scale-Location Plot
        plt.figure(figsize=(8, 6))
        plt.scatter(fitted, abs(residuals) ** 0.5, alpha=0.6)
        plt.title("Scale-Location Plot")
        plt.xlabel("Fitted values")
        plt.ylabel("√|Residuals|")
        plt.axhline(0, color='red', linestyle='--', linewidth=1)
        plt.grid(alpha=0.7, linestyle='--')
        plt.show()
    else:
        messagebox.showerror("Error", "Regression model is not available!")
# Function to test for normality of residuals
# Function to test for normality of residuals
def test_normality():
    model = perform_regression()  # Modelin oluşturulduğundan emin olmak için
    if model:
        residuals = model.resid  # Artık değerler (residuals)
        
        # Shapiro-Wilk Normality Test
        shapiro_stat, shapiro_pvalue = shapiro(residuals)
        
        # Shapiro-Wilk Test Sonuçları
        text_output.delete("1.0", tk.END)
        text_output.insert(tk.END, "Shapiro-Wilk Test for Normality:\n")
        text_output.insert(tk.END, f"Test Statistic (W): {shapiro_stat:.4f}\n")
        text_output.insert(tk.END, f"P-Value: {shapiro_pvalue:.4e}\n")
        if shapiro_pvalue < 0.05:
            text_output.insert(tk.END, "Conclusion: Residuals are not normally distributed.\n")
        else:
            text_output.insert(tk.END, "Conclusion: Residuals are normally distributed.\n")
        
        # Normal Q-Q Plot
        plt.figure(figsize=(10, 5))
        sm.qqplot(residuals, line="45", fit=True)
        plt.title("Normal Q-Q Plot")
        plt.grid(alpha=0.7, linestyle='--')
        plt.show()
        
        # Histogram of Residuals
        plt.figure(figsize=(10, 5))
        sns.histplot(residuals, kde=True, bins=20, color="pink", edgecolor="black")
        plt.title("Histogram of Residuals")
        plt.xlabel("Residuals")
        plt.ylabel("Frequency")
        plt.grid(alpha=0.7, linestyle='--')
        plt.show()
    else:
        messagebox.showerror("Error", "Regression model is not available!")
    
# Function to calculate and display VIF results
def visualize_vif():
    if data is not None and dependent_var:
        try:
            # Bağımsız değişkenleri seç (const hariç)
            independent_vars = [col for col in data.select_dtypes(include=['number']).columns if col != dependent_var]
            X = data[independent_vars]
            X = sm.add_constant(X)

            # VIF hesapla (const hariç) ve değerleri 10'a böl
            vif_data = pd.DataFrame()
            vif_data["Variable"] = X.columns[1:]  # const sütununu atla
            vif_data["VIF"] = [variance_inflation_factor(X.values, i) / 10 for i in range(1, X.shape[1])]

            # Text output'a VIF sonuçlarını yaz
            text_output.delete("1.0", tk.END)
            text_output.insert(tk.END, "Variance Inflation Factor (VIF) Results (divided by 10):\n\n")
            text_output.insert(tk.END, vif_data.to_string(index=False))

            # VIF için bar chart oluştur
            plt.figure(figsize=(10, 6))
            sns.barplot(data=vif_data, x="Variable", y="VIF", palette="viridis")
    
            plt.title("Variance Inflation Factor (VIF)")
            plt.xlabel("Variables")
            plt.ylabel("VIF (divided by 10)")
            plt.legend()
            plt.grid(alpha=0.5, linestyle="--")
            plt.show()

        except Exception as e:
            messagebox.showerror("Error", f"Error visualizing VIF: {e}")
    else:
        messagebox.showerror("Error", "No dependent variable selected or dataset loaded!")
def plot_correlation_heatmap():

    correlation_matrix = data.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.show()
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Global variables for train-test data
X_train, X_test, y_train, y_test = None, None, None, None

# Function to split dataset into train and test sets
def split_dataset():
    global X_train, X_test, y_train, y_test
    if data is not None and dependent_var:
        try:
            independent_vars = [col for col in data.select_dtypes(include=['number']).columns if col != dependent_var]
            X = data[independent_vars]
            y = data[dependent_var]
            
            # Split the dataset
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            messagebox.showinfo("Success", "Dataset successfully split into train (80%) and test (20%) sets!")
        except Exception as e:
            messagebox.showerror("Error", f"Error splitting dataset: {e}")
    else:
        messagebox.showerror("Error", "No dependent variable selected or dataset loaded!")

# Function to train model and test it
def train_and_test_model():
    global X_train, X_test, y_train, y_test
    if X_train is not None and y_train is not None:
        try:
            # Train the model
            X_train_const = sm.add_constant(X_train)
            model = sm.OLS(y_train, X_train_const).fit()

            # Test the model
            X_test_const = sm.add_constant(X_test)
            y_pred = model.predict(X_test_const)

            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Display results
            text_output.delete("1.0", tk.END)
            text_output.insert(tk.END, "Model Trained and Tested Successfully!\n")
            text_output.insert(tk.END, f"Mean Squared Error (MSE): {mse:.4f}\n")
            text_output.insert(tk.END, f"R-squared (Test Set): {r2:.4f}\n")
            text_output.insert(tk.END, "\nModel Summary (Train Set):\n")
            text_output.insert(tk.END, model.summary().as_text())
        except Exception as e:
            messagebox.showerror("Error", f"Error training/testing model: {e}")
    else:
        messagebox.showerror("Error", "Train and test datasets are not available!")

import itertools

# Stepwise Regression (Forward and Backward Selection)
def stepwise_selection(data, dependent_var):
    independent_vars = [col for col in data.select_dtypes(include=['number']).columns if col != dependent_var]
    X = data[independent_vars]
    y = data[dependent_var]
    X = sm.add_constant(X)

    # Initial setup
    included = []
    best_models = []
    while True:
        changed = False

        # Forward step: try adding all variables not in the model
        excluded = list(set(X.columns) - set(included) - {"const"})
        new_pvalues = pd.Series(index=excluded, dtype=float)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(data[included + [new_column]])).fit()
            new_pvalues[new_column] = model.pvalues[new_column]

        best_pvalue = new_pvalues.min()
        if best_pvalue < 0.05:
            best_feature = new_pvalues.idxmin()
            included.append(best_feature)
            changed = True

        # Backward step: try removing each variable in the model
        model = sm.OLS(y, sm.add_constant(data[included])).fit()
        pvalues = model.pvalues.iloc[1:]  # Exclude intercept
        worst_pvalue = pvalues.max()
        if worst_pvalue > 0.1:
            worst_feature = pvalues.idxmax()
            included.remove(worst_feature)
            changed = True

        if not changed:
            break

        best_models.append(model)

    return best_models[-1], included

# Best Subset Regression
def best_subset_regression(data, dependent_var):
    independent_vars = [col for col in data.select_dtypes(include=['number']).columns if col != dependent_var]
    X = data[independent_vars]
    y = data[dependent_var]
    X = sm.add_constant(X)

    best_models = []
    for k in range(1, len(independent_vars) + 1):
        for combo in itertools.combinations(independent_vars, k):
            model = sm.OLS(y, sm.add_constant(data[list(combo)])).fit()
            best_models.append((model.rsquared_adj, model))

    best_models.sort(reverse=True, key=lambda x: x[0])
    return best_models[0][1]  # Return model with highest adjusted R-squared

def run_stepwise():
    if data is not None and dependent_var:
        try:
            best_model, included_vars = stepwise_selection(data, dependent_var)
            text_output.delete("1.0", tk.END)
            text_output.insert(tk.END, "Stepwise Regression Results:\n")
            text_output.insert(tk.END, f"Included Variables: {included_vars}\n")
            text_output.insert(tk.END, best_model.summary().as_text())
        except Exception as e:
            messagebox.showerror("Error", f"Error in stepwise regression: {e}")
    else:
        messagebox.showerror("Error", "No dependent variable selected or dataset loaded!")

def run_best_subset():
    if data is not None and dependent_var:
        try:
            best_model = best_subset_regression(data, dependent_var)
            text_output.delete("1.0", tk.END)
            text_output.insert(tk.END, "Best Subset Regression Results:\n")
            text_output.insert(tk.END, best_model.summary().as_text())
        except Exception as e:
            messagebox.showerror("Error", f"Error in best subset regression: {e}")
    else:
        messagebox.showerror("Error", "No dependent variable selected or dataset loaded!")
def plot_individual_boxplots():
    if data is not None:
        try:
            numerical_cols = data.select_dtypes(include=['number']).columns
            for col in numerical_cols:
                plt.figure(figsize=(8, 6))
                sns.boxplot(y=data[col])
                plt.title(f"Box Plot of {col}")
                plt.ylabel(col)
                plt.grid(alpha=0.7, linestyle='--')
                plt.show()
        except Exception as e:
            messagebox.showerror("Error", f"Error generating box plots: {e}")
    else:
        messagebox.showerror("Error", "No dataset loaded!")

from sklearn.metrics import mean_absolute_error, mean_squared_error

# Function to calculate metrics for multiple models
# Function to calculate metrics for multiple models
def compare_models():
    if data is not None and dependent_var:
        try:
            independent_vars = [col for col in data.select_dtypes(include=['number']).columns if col != dependent_var]
            y = data[dependent_var]
            
            # Model 1: Using the first 3 independent variables
            model1_vars = independent_vars[:3]
            X1 = sm.add_constant(data[model1_vars])
            model1 = sm.OLS(y, X1).fit()

            # Model 2: Using the first 4 independent variables
            model2_vars = independent_vars[:4]
            X2 = sm.add_constant(data[model2_vars])
            model2 = sm.OLS(y, X2).fit()

            # Model 3: Using all independent variables
            X3 = sm.add_constant(data[independent_vars])
            model3 = sm.OLS(y, X3).fit()

            # Metrics for each model
            metrics = pd.DataFrame(columns=["Model", "Variables", "RMSE", "MAE", "R-squared", "Adj. R-squared"])
            
            for i, (model, vars_used) in enumerate([(model1, model1_vars), (model2, model2_vars), (model3, independent_vars)], 1):
                y_pred = model.predict(sm.add_constant(data[vars_used]))
                rmse = mean_squared_error(y, y_pred, squared=False)
                mae = mean_absolute_error(y, y_pred)
                r2 = model.rsquared
                adj_r2 = model.rsquared_adj

                metrics = pd.concat([
                    metrics,
                    pd.DataFrame([{
                        "Model": f"Model {i}",
                        "Variables": vars_used,
                        "RMSE": rmse,
                        "MAE": mae,
                        "R-squared": r2,
                        "Adj. R-squared": adj_r2
                    }])
                ], ignore_index=True)

            # Display metrics in text output
            text_output.delete("1.0", tk.END)
            text_output.insert(tk.END, "Model Comparison Results:\n\n")
            text_output.insert(tk.END, metrics.to_string(index=False))

            # Display models' summaries
            for i, model in enumerate([model1, model2, model3], 1):
                text_output.insert(tk.END, f"\n\nSummary of Model {i}:\n")
                text_output.insert(tk.END, model.summary().as_text())

        except Exception as e:
            messagebox.showerror("Error", f"Error comparing models: {e}")
    else:
        messagebox.showerror("Error", "No dependent variable selected or dataset loaded!")
def test_and_plot_residuals_normality():
    model = perform_regression()  # Modelin oluşturulduğundan emin olmak için
    if model:
        residuals = model.resid  # Artık değerler (residuals)
        
        # Shapiro-Wilk Normality Test
        shapiro_stat, shapiro_pvalue = shapiro(residuals)
        
        # Shapiro-Wilk Test Sonuçları
        text_output.delete("1.0", tk.END)
        text_output.insert(tk.END, "Shapiro-Wilk Test for Normality:\n")
        text_output.insert(tk.END, f"Test Statistic (W): {shapiro_stat:.4f}\n")
        text_output.insert(tk.END, f"P-Value: {shapiro_pvalue:.4e}\n")
        if shapiro_pvalue < 0.05:
            text_output.insert(tk.END, "Conclusion: Residuals are not normally distributed.\n")
        else:
            text_output.insert(tk.END, "Conclusion: Residuals are normally distributed.\n")
        
        # Normal Q-Q Plot
        plt.figure(figsize=(10, 5))
        sm.qqplot(residuals, line="45", fit=True)
        plt.title("Normal Q-Q Plot")
        plt.grid(alpha=0.7, linestyle='--')
        plt.show()
        
        # Histogram of Residuals
        plt.figure(figsize=(10, 5))
        sns.histplot(residuals, kde=True, bins=20, color="pink", edgecolor="black")
        plt.title("Histogram of Residuals")
        plt.xlabel("Residuals")
        plt.ylabel("Frequency")
        plt.grid(alpha=0.7, linestyle='--')
        plt.show()
    else:
        messagebox.showerror("Error", "Regression model is not available!")
from statsmodels.stats.diagnostic import het_breuschpagan

def test_homoscedasticity():
    model = perform_regression()
    if model:
        residuals = model.resid
        exog = model.model.exog

        # Breusch-Pagan Test
        bp_stat, bp_pvalue, _, _ = het_breuschpagan(residuals, exog)

        # Sonuçları ekrana yazdır
        text_output.delete("1.0", tk.END)
        text_output.insert(tk.END, "Breusch-Pagan Test for Homoscedasticity\n")
        text_output.insert(tk.END, f"Test Statistic (BP): {bp_stat:.4f}\n")
        text_output.insert(tk.END, f"P-Value: {bp_pvalue:.4e}\n")
        if bp_pvalue < 0.05:
            text_output.insert(tk.END, "Conclusion: Residuals do not have constant variance (heteroscedasticity detected).\n")
        else:
            text_output.insert(tk.END, "Conclusion: Residuals have constant variance (homoscedasticity).\n")
    else:
        messagebox.showerror("Error", "Regression model is not available!")
from statsmodels.stats.stattools import durbin_watson

def test_independence():
    model = perform_regression()
    if model:
        residuals = model.resid

        # Durbin-Watson Test
        dw_stat = durbin_watson(residuals)

        # Sonuçları ekrana yazdır
        text_output.delete("1.0", tk.END)
        text_output.insert(tk.END, "Durbin-Watson Test for Independence of Residuals\n")
        text_output.insert(tk.END, f"Durbin-Watson Statistic: {dw_stat:.4f}\n")
        if dw_stat < 1.5 or dw_stat > 2.5:
            text_output.insert(tk.END, "Conclusion: Evidence of autocorrelation in residuals.\n")
        else:
            text_output.insert(tk.END, "Conclusion: No evidence of autocorrelation in residuals.\n")
    else:
        messagebox.showerror("Error", "Regression model is not available!")

def determine_final_model():
    if data is not None and dependent_var:
        try:
            # Bağımsız değişkenlerin listesi
            independent_vars = [col for col in data.select_dtypes(include=['number']).columns if col != dependent_var]
            X = data[independent_vars]
            y = data[dependent_var]
            X = sm.add_constant(X)

            # Başlangıç modeli
            model = sm.OLS(y, X).fit()

            # Adım adım değişken eleme (p-value > 0.05 olan değişkenleri çıkarma)
            while True:
                p_values = model.pvalues.iloc[1:]  # Intercept'i hariç tut
                max_p_value = p_values.max()
                if max_p_value > 0.05:
                    # En yüksek p-value'yu çıkar
                    max_p_variable = p_values.idxmax()
                    independent_vars.remove(max_p_variable)
                    X = data[independent_vars]
                    X = sm.add_constant(X)
                    model = sm.OLS(y, X).fit()
                else:
                    break

            # Sonuçları yazdır
            text_output.delete("1.0", tk.END)
            text_output.insert(tk.END, "Final Model Results:\n")
            text_output.insert(tk.END, model.summary().as_text())

            return model
        except Exception as e:
            messagebox.showerror("Error", f"Error determining final model: {e}")
    else:
        messagebox.showerror("Error", "No dependent variable selected or dataset loaded!")
def calculate_influence_measures():
    # Regresyon modelini çalıştır
    model = perform_regression()
    if model:
        try:
            # Etki ölçütlerini hesapla
            influence = model.get_influence()
            cooks_d, _ = influence.cooks_distance  # Cook's Distance
            leverage = influence.hat_matrix_diag  # Leverage
            standardized_residuals = influence.resid_studentized_internal  # Standardized residuals

            # Etki ölçütlerini bir DataFrame'de sakla
            influence_data = pd.DataFrame({
                "Index": range(len(cooks_d)),
                "Cook's Distance": cooks_d,
                "Leverage": leverage,
                "Standardized Residuals": standardized_residuals,
            })

            # Tabloda kritik değerleri işaretle
            critical_cooks_d = 4 / len(data)
            critical_leverage = 2 * (model.df_model + 1) / len(data)
            influence_data["High Cook's D"] = influence_data["Cook's Distance"] > critical_cooks_d
            influence_data["High Leverage"] = influence_data["Leverage"] > critical_leverage

            # Sonuçları text_output ve bir matplotlib tablosunda göster
            text_output.delete("1.0", tk.END)
            text_output.insert(tk.END, "Influence Measures (Cook's D, Leverage, Residuals):\n")
            text_output.insert(tk.END, influence_data.to_string(index=False))
            
            # Plot: Cook's Distance
            plt.figure(figsize=(10, 6))
            plt.stem(influence_data["Index"], cooks_d, markerfmt=",", basefmt=" ", use_line_collection=True)
            plt.axhline(critical_cooks_d, color="red", linestyle="--", label=f"Critical Cook's D = {critical_cooks_d:.4f}")
            plt.title("Cook's Distance")
            plt.xlabel("Observation Index")
            plt.ylabel("Cook's Distance")
            plt.legend()
            plt.grid(alpha=0.5, linestyle="--")
            plt.show()

            # Plot: Leverage vs Standardized Residuals
            plt.figure(figsize=(10, 6))
            plt.scatter(leverage, standardized_residuals, alpha=0.6, edgecolor="k")
            plt.axhline(y=0, color="red", linestyle="--")
            plt.axvline(x=critical_leverage, color="red", linestyle="--", label=f"Critical Leverage = {critical_leverage:.4f}")
            plt.title("Leverage vs Standardized Residuals")
            plt.xlabel("Leverage")
            plt.ylabel("Standardized Residuals")
            plt.legend()
            plt.grid(alpha=0.5, linestyle="--")
            plt.show()

        except Exception as e:
            messagebox.showerror("Error", f"Error calculating influence measures: {e}")
    else:
        messagebox.showerror("Error", "Regression model is not available!")

def scatterplot_matrix():
    if data is not None:
        try:
            # Scatterplot matrix için seaborn pairplot kullanıyoruz
            sns.set(style="whitegrid")
            plt.figure(figsize=(15, 15))
            pairplot = sns.pairplot(data, diag_kind="kde", plot_kws={'alpha': 0.6, 'color': 'purple'})
            
            # Başlık ekle
            pairplot.fig.suptitle("Scatterplot Matrix", y=1.02)
            plt.show()
        except Exception as e:
            messagebox.showerror("Error", f"Error generating scatterplot matrix: {e}")
    else:
        messagebox.showerror("Error", "No dataset loaded!")

def plot_all_distributions():
    if data is not None:
        try:
            numerical_cols = data.select_dtypes(include=['number']).columns  # Sayısal sütunları seç
            for col in numerical_cols:
                plt.figure(figsize=(8, 6))
                sns.histplot(data[col], kde=True, bins=20, color="blue", edgecolor="black")
                plt.title(f"Distribution of {col}")
                plt.xlabel(col)
                plt.ylabel("Frequency")
                plt.grid(alpha=0.5, linestyle='--')
                plt.show()
        except Exception as e:
            messagebox.showerror("Error", f"Error plotting distributions: {e}")
    else:
        messagebox.showerror("Error", "No dataset loaded!")

from statsmodels.sandbox.regression.predstd import wls_prediction_std

      
def ols_step_backward():
    if data is not None and dependent_var:
        try:
            independent_vars = [col for col in data.select_dtypes(include=['number']).columns if col != dependent_var]
            X = data[independent_vars]
            y = data[dependent_var]
            X = sm.add_constant(X)

            # Başlangıç modeli
            model = sm.OLS(y, X).fit()

            # Değişken eleme
            while True:
                p_values = model.pvalues.iloc[1:]  # Intercept hariç
                max_p_value = p_values.max()
                if max_p_value > 0.05:  # Eğer p-değeri 0.05'ten büyükse
                    max_p_variable = p_values.idxmax()
                    independent_vars.remove(max_p_variable)
                    X = data[independent_vars]
                    X = sm.add_constant(X)
                    model = sm.OLS(y, X).fit()
                else:
                    break

            # Sonuçları göster
            text_output.delete("1.0", tk.END)
            text_output.insert(tk.END, "OLS Step Backward Results:\n")
            text_output.insert(tk.END, model.summary().as_text())
        except Exception as e:
            messagebox.showerror("Error", f"Error performing OLS Step Backward: {e}")
    else:
        messagebox.showerror("Error", "No dependent variable selected or dataset loaded!")

def ols_step_forward():
    if data is not None and dependent_var:
        try:
            independent_vars = [col for col in data.select_dtypes(include=['number']).columns if col != dependent_var]
            selected_vars = []
            y = data[dependent_var]

            while True:
                remaining_vars = [col for col in independent_vars if col not in selected_vars]
                best_p_value = 1
                best_var = None

                for var in remaining_vars:
                    X = data[selected_vars + [var]]
                    X = sm.add_constant(X)
                    model = sm.OLS(y, X).fit()
                    if model.pvalues[var] < best_p_value:
                        best_p_value = model.pvalues[var]
                        best_var = var

                if best_p_value < 0.05:
                    selected_vars.append(best_var)
                else:
                    break

            # Son Model
            X = data[selected_vars]
            X = sm.add_constant(X)
            final_model = sm.OLS(y, X).fit()

            # Sonuçları göster
            text_output.delete("1.0", tk.END)
            text_output.insert(tk.END, "OLS Step Forward Results:\n")
            text_output.insert(tk.END, final_model.summary().as_text())
        except Exception as e:
            messagebox.showerror("Error", f"Error performing OLS Step Forward: {e}")
    else:
        messagebox.showerror("Error", "No dependent variable selected or dataset loaded!")

def ols_all_step_possible():
    if data is not None and dependent_var:
        try:
            independent_vars = [col for col in data.select_dtypes(include=['number']).columns if col != dependent_var]
            y = data[dependent_var]

            best_models = []
            for k in range(1, len(independent_vars) + 1):
                for combo in itertools.combinations(independent_vars, k):
                    X = data[list(combo)]
                    X = sm.add_constant(X)
                    model = sm.OLS(y, X).fit()
                    best_models.append((model.rsquared_adj, model))

            best_models.sort(reverse=True, key=lambda x: x[0])
            best_model = best_models[0][1]

            # Sonuçları göster
            text_output.delete("1.0", tk.END)
            text_output.insert(tk.END, "OLS All Step Possible Results:\n")
            text_output.insert(tk.END, best_model.summary().as_text())
        except Exception as e:
            messagebox.showerror("Error", f"Error performing OLS All Step Possible: {e}")
    else:
        messagebox.showerror("Error", "No dependent variable selected or dataset loaded!")

def calculate_prediction_intervals(input_values):
    global data, dependent_var
    if data is None:
        messagebox.showerror("Error", "No dataset loaded! Please load a dataset first.")
        return
    if dependent_var is None:
        messagebox.showerror("Error", "No dependent variable selected! Please select one.")
        return
    
    try:
        # Bağımsız değişkenlerin seçimi
        independent_vars = [col for col in data.select_dtypes(include=['number']).columns if col != dependent_var]
        new_data = pd.DataFrame([input_values])

        # Eksik sütunları doldurun
        for var in independent_vars:
            if var not in new_data.columns:
                new_data[var] = 0

        # Sabit sütun ekleyin
        new_data = new_data[independent_vars]
        new_data = sm.add_constant(new_data)

        # Regresyon modelini oluştur
        model = perform_regression()

        # Tahmin yap ve standart hata ile güven aralıklarını hesapla
        prediction, std_err, confidence_interval = wls_prediction_std(model, exog=new_data, alpha=0.05)

        # Tahmin aralığı hesaplama
        prediction_value = float(prediction[0])
        lower_bound_value = float(prediction[0] - 1.96 * std_err)
        upper_bound_value = float(prediction[0] + 1.96 * std_err)

        # Sonuçları ekrana yazdır
        text_output.delete("1.0", tk.END)
        text_output.insert(tk.END, "Tahmin Değeri için Güven ve Kestirim Aralıkları:\n")
        text_output.insert(tk.END, f"Tahmin Değeri: {prediction_value:.4f}\n")
        text_output.insert(tk.END, f"%95 Güven Aralığı: [{lower_bound_value:.4f}, {upper_bound_value:.4f}]\n")
    except Exception as e:
        messagebox.showerror("Error", f"Error calculating prediction intervals: {e}")
def analyze_with_openai(prompt):
    try:
        response = openai.Completion.create(
            engine="text-davinci-003",  # OpenAI'nin güçlü dil modeli
            prompt=prompt,
            max_tokens=150,  # Yanıtın uzunluğunu belirler
            temperature=0.7  # Yanıtın yaratıcılığını kontrol eder
        )
        return response.choices[0].text.strip()
    except Exception as e:
        messagebox.showerror("Error", f"Error communicating with OpenAI API: {e}")
        return None
def generate_prompt_for_openai():
    # Text output'taki analiz sonuçlarını al
    analysis_results = text_output.get("1.0", tk.END).strip()
    if not analysis_results:
        messagebox.showerror("Error", "No analysis results to analyze!")
        return None

    # Prompt'u oluştur
    prompt = (
        "Please analyze the following statistical results and provide a concise explanation. "
        "Focus on the implications, potential insights, and any recommendations for improvement:\n\n"
        f"{analysis_results}\n"
    )
    return prompt

def analyze_with_openai(prompt):
    try:
        response = openai.Completion.create(
            engine="text-davinci-003",  # GPT-4 yerine "text-davinci-003" kullanılabilir
            prompt=prompt,
            max_tokens=150,
            temperature=0.7
        )
        return response.choices[0].text.strip()
    except Exception as e:
        messagebox.showerror("Error", f"Error communicating with OpenAI API: {e}")
        return None
def interpret_analysis():
    prompt = generate_prompt_for_openai()
    if prompt:
        ai_response = analyze_with_openai(prompt) 
        if ai_response:
            text_output.delete("1.0", tk.END)
            text_output.insert(tk.END, "OpenAI Analysis:\n")
            text_output.insert(tk.END, ai_response)
# Örnek veri
example_input = {
    'GDP.per.capita': 1.7,
    'Social.support': 1.65,
    'Healthy.life.expectancy': 1.1,
    'Freedom.to.make.life.choices': 0.7
}

import itertools
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

def best_subset_per_column():
    global data, dependent_var
    if data is not None and dependent_var:
        try:
            independent_vars = [col for col in data.select_dtypes(include=['number']).columns if col != dependent_var]
            y = data[dependent_var]

            # Initialize results dictionary
            best_results = []

            # Iterate through all variables as independent columns
            for k in range(1, len(independent_vars) + 1):
                best_cp = float('inf')
                best_model = None
                best_predictors = None

                # Iterate through all combinations of length k
                for combo in itertools.combinations(independent_vars, k):
                    X = data[list(combo)]
                    X = sm.add_constant(X)  # Add constant
                    model = sm.OLS(y, X).fit()

                    # Calculate Cp
                    sigma_squared = np.var(model.resid)
                    rss = sum(model.resid ** 2)
                    cp = (rss / sigma_squared) - (len(y) - 2 * (len(combo) + 1))  # +1 for constant

                    # Check if this is the best Cp for this combination
                    if cp < best_cp:
                        best_cp = cp
                        best_model = model
                        best_predictors = combo

                # Store the best result for this combination length
                best_results.append({
                    'Predictors': ", ".join(best_predictors[:3]) + ("..." if len(best_predictors) > 3 else ""),
                    'Adj. R-Squared': round(best_model.rsquared_adj, 4),
                    'AIC': round(best_model.aic, 2),
                    'BIC': round(best_model.bic, 2),
                    'Cp': round(best_cp, 2)
                })

            # Convert results to DataFrame
            best_results_df = pd.DataFrame(best_results).sort_values(by='Adj. R-Squared', ascending=False)

            # Display results in text output
            text_output.delete("1.0", tk.END)
            text_output.insert(tk.END, "Best Subset Regression Results (Compact):\n")
            text_output.insert(tk.END, best_results_df.to_string(index=False))

        except Exception as e:
            messagebox.showerror("Error", f"Error performing best subset regression per column: {e}")
    else:
        messagebox.showerror("Error", "No dependent variable selected or dataset loaded!")
import csv

def evaluate_and_export_model():
    global data, dependent_var
    if data is not None and dependent_var:
        try:
            # 1. Model Performansı
            independent_vars = [col for col in data.select_dtypes(include=['number']).columns if col != dependent_var]
            X = data[independent_vars]
            y = data[dependent_var]
            X = sm.add_constant(X)  # Sabit sütunu ekle

            # Model oluştur
            model = sm.OLS(y, X).fit()

            # Performans metrikleri
            model_summary = {
                "R-Squared": model.rsquared,
                "Adj. R-Squared": model.rsquared_adj,
                "AIC": model.aic,
                "BIC": model.bic,
                "p-Values": model.pvalues.to_dict(),
            }

            # 2. Artık Analizi
            residuals = model.resid
            shapiro_stat, shapiro_pvalue = shapiro(residuals)
            bp_stat, bp_pvalue, _, _ = het_breuschpagan(residuals, model.model.exog)

            normality_test = {
                "Shapiro-Wilk Test Statistic": shapiro_stat,
                "Shapiro-Wilk p-Value": shapiro_pvalue,
                "Breusch-Pagan Test Statistic": bp_stat,
                "Breusch-Pagan p-Value": bp_pvalue,
            }

            # 3. Model Validasyonu
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            validation_model = sm.OLS(y_train, X_train).fit()
            y_pred = validation_model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2_test = r2_score(y_test, y_pred)

            validation_metrics = {
                "MSE (Test Set)": mse,
                "R-Squared (Test Set)": r2_test,
            }

            # 4. Tahmin Yapma
            # Tahmin için veri oluştur ve sabit sütunu ekle
            prediction_data = pd.DataFrame({col: [data[col].mean()] for col in independent_vars})
            prediction_data = sm.add_constant(prediction_data)  # Sabit sütunu ekle

            # Modeldeki sütun sırasını koru
            for col in model.model.exog_names:
                if col not in prediction_data.columns:
                    prediction_data[col] = 0  # Eksik sütunları doldur
            prediction_data = prediction_data[model.model.exog_names]  # Sıralamayı koru

            # Tahmin yap
            prediction = model.predict(prediction_data)
            prediction_results = {
                "Prediction Value": prediction[0],
            }

            # 5. Sonuçları CSV Olarak Kaydet
            export_data = {
                "Model Summary": model_summary,
                "Normality Test": normality_test,
                "Validation Metrics": validation_metrics,
                "Prediction Results": prediction_results,
            }

            # CSV Dosyası oluştur
            with open("model_evaluation_results.csv", "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["Category", "Metric", "Value"])
                for category, metrics in export_data.items():
                    for key, value in metrics.items():
                        if isinstance(value, dict):
                            for sub_key, sub_value in value.items():
                                writer.writerow([category, sub_key, sub_value])
                        else:
                            writer.writerow([category, key, value])

            # Kullanıcıya bilgi ver
            text_output.delete("1.0", tk.END)
            text_output.insert(tk.END, "Model evaluation and export completed successfully!\n")
            text_output.insert(tk.END, "Results saved to 'model_evaluation_results.csv'\n")
        except Exception as e:
            messagebox.showerror("Error", f"Error during model evaluation: {e}")
    else:
        messagebox.showerror("Error", "No dependent variable selected or dataset loaded!")

def best_subset_and_export():
    global data, dependent_var
    if data is not None and dependent_var:
        try:
            independent_vars = [col for col in data.select_dtypes(include=['number']).columns if col != dependent_var]
            y = data[dependent_var]
            results = []  # Tüm kombinasyonların sonuçlarını tutacak

            # Her değişken kombinasyonu için en iyi modeli bul
            for k in range(1, len(independent_vars) + 1):
                best_cp = float("inf")
                best_model = None
                best_predictors = None

                # K kombinasyonlarını dene
                for combo in itertools.combinations(independent_vars, k):
                    X = data[list(combo)]
                    X = sm.add_constant(X)
                    model = sm.OLS(y, X).fit()

                    # Cp değerini hesapla
                    sigma_squared = np.var(model.resid)
                    rss = sum(model.resid ** 2)
                    cp = (rss / sigma_squared) - (len(y) - 2 * (len(combo) + 1))  # +1 constant için

                    # En iyi modeli belirle
                    if cp < best_cp:
                        best_cp = cp
                        best_model = model
                        best_predictors = combo

                # Sonuçları kaydet
                results.append({
                    'Predictors': ", ".join(best_predictors),
                    'Adj. R-Squared': best_model.rsquared_adj,
                    'AIC': best_model.aic,
                    'BIC': best_model.bic,
                    'Cp': best_cp
                })

            # Tüm sonuçları bir DataFrame'e dönüştür
            results_df = pd.DataFrame(results)

            # En iyi sonucu seç
            best_result = results_df.loc[results_df['Adj. R-Squared'].idxmax()]

            # Tüm sonuçları bir CSV dosyasına kaydet
            results_df.to_csv("best_subset_results.csv", index=False)

            # En iyi sonucu ayrı bir CSV dosyasına kaydet
            best_result.to_frame().T.to_csv("best_model_result.csv", index=False)

            # Kullanıcıya bilgi ver
            text_output.delete("1.0", tk.END)
            text_output.insert(tk.END, "Best subset regression completed successfully!\n")
            text_output.insert(tk.END, "All results saved to 'best_subset_results.csv'\n")
            text_output.insert(tk.END, "Best model result saved to 'best_model_result.csv'\n")

        except Exception as e:
            messagebox.showerror("Error", f"Error performing best subset regression: {e}")
    else:
        messagebox.showerror("Error", "No dependent variable selected or dataset loaded!")
def compare_models_and_select_best():
    global data, dependent_var
    if data is not None and dependent_var:
        try:
            # Bağımsız değişkenleri belirle
            independent_vars = [col for col in data.select_dtypes(include=['number']).columns if col != dependent_var]
            y = data[dependent_var]
            
            # Model sonuçlarını saklamak için bir liste
            model_results = []
            
            # Farklı sayıda değişkenle model oluştur
            for i in range(1, len(independent_vars) + 1):
                # i değişkenle model oluştur
                selected_vars = independent_vars[:i]
                X = data[selected_vars]
                X = sm.add_constant(X)  # Sabit sütun ekle
                
                # Model oluştur ve eğit
                model = sm.OLS(y, X).fit()
                
                # Test metriklerini hesapla
                rmse = np.sqrt(np.mean(model.resid ** 2))
                r_squared = model.rsquared
                adj_r_squared = model.rsquared_adj
                
                # Modelin sonuçlarını kaydet
                model_results.append({
                    "Model": f"Model {i}",
                    "Variables": selected_vars,
                    "RMSE": rmse,
                    "R-Squared": r_squared,
                    "Adj. R-Squared": adj_r_squared
                })
                
                # Ekrana yazdır
                text_output.insert(tk.END, f"\nModel {i} Results:\n")
                text_output.insert(tk.END, f"Variables: {selected_vars}\n")
                text_output.insert(tk.END, f"RMSE: {rmse:.4f}\n")
                text_output.insert(tk.END, f"R-Squared: {r_squared:.4f}\n")
                text_output.insert(tk.END, f"Adjusted R-Squared: {adj_r_squared:.4f}\n")
            
            # En iyi modeli seç (Adjusted R-Squared'a göre)
            best_model = max(model_results, key=lambda x: x["Adj. R-Squared"])
            
            # En iyi modeli ekrana yazdır
            text_output.insert(tk.END, "\nBest Model:\n")
            text_output.insert(tk.END, f"{best_model['Model']} with Variables: {best_model['Variables']}\n")
            text_output.insert(tk.END, f"Adjusted R-Squared: {best_model['Adj. R-Squared']:.4f}\n")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error comparing models: {e}")
    else:
        messagebox.showerror("Error", "No dependent variable selected or dataset loaded!")
   




def train_test_and_export_model(model_name):
    global data, dependent_var
    if data is not None and dependent_var:
        try:
            # Train-Test Split
            independent_vars = [col for col in data.select_dtypes(include=['number']).columns if col != dependent_var]
            X = data[independent_vars]
            y = data[dependent_var]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train the Model
            X_train_const = sm.add_constant(X_train)
            model = sm.OLS(y_train, X_train_const).fit()

            # Test the Model
            X_test_const = sm.add_constant(X_test)
            y_pred = model.predict(X_test_const)

            # Calculate Metrics
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            # Export Metrics and Model Summary
            export_data = {
                "Model Name": model_name,
                "MSE (Test Set)": mse,
                "R-Squared (Test Set)": r2,
                "Adjusted R-Squared (Train Set)": model.rsquared_adj,
                "AIC": model.aic,
                "BIC": model.bic
            }

            # Save to CSV
            with open(f"{model_name}_results.csv", "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["Metric", "Value"])
                for key, value in export_data.items():
                    writer.writerow([key, value])

            # Display results in GUI
            text_output.delete("1.0", tk.END)
            text_output.insert(tk.END, f"Model {model_name} Trained and Tested Successfully!\n")
            text_output.insert(tk.END, f"Results exported to {model_name}_results.csv\n")
            text_output.insert(tk.END, f"MSE (Test Set): {mse:.4f}\n")
            text_output.insert(tk.END, f"R-Squared (Test Set): {r2:.4f}\n")
            text_output.insert(tk.END, f"Adjusted R-Squared (Train Set): {model.rsquared_adj:.4f}\n")
            text_output.insert(tk.END, f"\nModel Summary:\n")
            text_output.insert(tk.END, model.summary().as_text())
        except Exception as e:
            messagebox.showerror("Error", f"Error training and testing model: {e}")
    else:
        messagebox.showerror("Error", "No dependent variable selected or dataset loaded!")





def display_confidence_intervals():
    global data, dependent_var
    if data is not None and dependent_var:
        try:
            # Bağımsız değişkenleri belirleyin
            independent_vars = [col for col in data.select_dtypes(include=['number']).columns if col != dependent_var]
            X = data[independent_vars]
            y = data[dependent_var]
            X = sm.add_constant(X)  # Sabit sütunu ekle

            # Model oluştur
            model = sm.OLS(y, X).fit()

            # Güven aralıklarını hesapla
            conf_intervals = model.conf_int(alpha=0.05)
            conf_intervals.columns = ["2.5%", "97.5%"]
            conf_intervals["Coefficient"] = model.params
            conf_intervals = conf_intervals[["Coefficient", "2.5%", "97.5%"]]

            # Sonuçları GUI'ye yazdır
            text_output.delete("1.0", tk.END)
            text_output.insert(tk.END, "Confidence Intervals (%95):\n\n")
            text_output.insert(tk.END, conf_intervals.to_string())

            # Güven aralıklarını görselleştir
            plt.figure(figsize=(10, 6))
            plt.errorbar(conf_intervals.index, conf_intervals["Coefficient"], 
                         yerr=[conf_intervals["Coefficient"] - conf_intervals["2.5%"],
                               conf_intervals["97.5%"] - conf_intervals["Coefficient"]],
                         fmt='o', color='blue', ecolor='red', capsize=5)
            plt.axhline(0, color='gray', linestyle='--')
            plt.title("Confidence Intervals for Coefficients")
            plt.xlabel("Variables")
            plt.ylabel("Coefficient")
            plt.grid(alpha=0.7, linestyle='--')
            plt.xticks(rotation=45)
            plt.show()
        except Exception as e:
            messagebox.showerror("Error", f"Error calculating confidence intervals: {e}")
    else:
        messagebox.showerror("Error", "No dependent variable selected or dataset loaded!")

def summarize_data():
    global data
    if data is not None and not data.empty:
        try:
            # Genel Bilgiler
            general_info = {
                "Number of Rows": data.shape[0],
                "Number of Columns": data.shape[1],
                "Missing Values": data.isnull().sum().sum(),
                "Numeric Columns": len(data.select_dtypes(include=['number']).columns),
                "Categorical Columns": len(data.select_dtypes(include=['object', 'category']).columns)
            }

            # Sayısal Değişkenlerin Özet İstatistikleri
            if len(data.select_dtypes(include=['number']).columns) > 0:
                numeric_summary = data.describe().T
            else:
                numeric_summary = "No numeric columns available."

            # Kategorik Değişkenlerin Özet İstatistikleri
            if len(data.select_dtypes(include=['object', 'category']).columns) > 0:
                categorical_summary = data.select_dtypes(include=['object', 'category']).describe().T
            else:
                categorical_summary = "No categorical columns available."

            # GUI'ye yazdırma
            text_output.delete("1.0", tk.END)
            text_output.insert(tk.END, "General Information:\n")
            for key, value in general_info.items():
                text_output.insert(tk.END, f"{key}: {value}\n")

            text_output.insert(tk.END, "\nNumeric Columns Summary:\n")
            if isinstance(numeric_summary, pd.DataFrame):
                text_output.insert(tk.END, numeric_summary.to_string())
            else:
                text_output.insert(tk.END, numeric_summary)

            text_output.insert(tk.END, "\n\nCategorical Columns Summary:\n")
            if isinstance(categorical_summary, pd.DataFrame):
                text_output.insert(tk.END, categorical_summary.to_string())
            else:
                text_output.insert(tk.END, categorical_summary)

            # Terminalde kontrol için
            print("General Information:", general_info)
            if isinstance(numeric_summary, pd.DataFrame):
                print("\nNumeric Columns Summary:\n", numeric_summary)
            else:
                print("\nNumeric Columns Summary:", numeric_summary)

            if isinstance(categorical_summary, pd.DataFrame):
                print("\nCategorical Columns Summary:\n", categorical_summary)
            else:
                print("\nCategorical Columns Summary:", categorical_summary)

        except Exception as e:
            messagebox.showerror("Error", f"Error summarizing data: {e}")
    else:
        messagebox.showerror("Error", "Dataset is empty or not loaded!")

def add_quadratic_term_and_evaluate(variable):
    global data, dependent_var
    if data is None:
        messagebox.showerror("Error", "No dataset loaded! Please load a dataset first.")
        return

    if dependent_var is None:
        messagebox.showerror("Error", "No dependent variable selected! Please select one.")
        return

    if variable not in data.columns:
        messagebox.showerror("Error", f"The variable '{variable}' does not exist in the dataset!")
        return

    try:
        # Yeni sütun ekle (karesini al)
        new_col_name = f"{variable}_squared"
        data[new_col_name] = data[variable] ** 2

        # Regresyon Modeli
        independent_vars = [col for col in data.select_dtypes(include=['number']).columns if col != dependent_var]
        X = data[independent_vars]
        y = data[dependent_var]
        X = sm.add_constant(X)
        model = sm.OLS(y, X).fit()

        # Model Performans Çıktıları
        text_output.delete("1.0", tk.END)
        text_output.insert(tk.END, f"Model Performance After Adding {new_col_name}:\n")
        text_output.insert(tk.END, model.summary().as_text())

        # Grafiksel Analiz
        residuals = model.resid
        fitted = model.fittedvalues

        # Residuals vs Fitted
        plt.figure(figsize=(8, 6))
        sns.residplot(x=fitted, y=residuals, lowess=True, line_kws={'color': 'red'})
        plt.title("Residuals vs Fitted")
        plt.xlabel("Fitted Values")
        plt.ylabel("Residuals")
        plt.axhline(0, color='black', linestyle='--', linewidth=1)
        plt.grid(alpha=0.5, linestyle='--')
        plt.show()

        # Normal Q-Q Plot
        sm.qqplot(residuals, line="45", fit=True)
        plt.title("Normal Q-Q Plot")
        plt.grid(alpha=0.5, linestyle='--')
        plt.show()

        # Scale-Location Plot
        plt.figure(figsize=(8, 6))
        plt.scatter(fitted, abs(residuals) ** 0.5, alpha=0.6)
        plt.title("Scale-Location Plot")
        plt.xlabel("Fitted Values")
        plt.ylabel("√|Residuals|")
        plt.axhline(0, color='red', linestyle='--', linewidth=1)
        plt.grid(alpha=0.5, linestyle='--')
        plt.show()

        # Histogram of Residuals
        plt.figure(figsize=(8, 6))
        sns.histplot(residuals, kde=True, bins=20, color="purple", edgecolor="black")
        plt.title("Histogram of Residuals")
        plt.xlabel("Residuals")
        plt.ylabel("Frequency")
        plt.grid(alpha=0.5, linestyle='--')
        plt.show()

        # Performance Metrics
        r_squared = model.rsquared
        adj_r_squared = model.rsquared_adj
        aic = model.aic
        bic = model.bic
        text_output.insert(tk.END, "\nModel Evaluation Metrics:\n")
        text_output.insert(tk.END, f"R-Squared: {r_squared:.4f}\n")
        text_output.insert(tk.END, f"Adjusted R-Squared: {adj_r_squared:.4f}\n")
        text_output.insert(tk.END, f"AIC: {aic:.4f}\n")
        text_output.insert(tk.END, f"BIC: {bic:.4f}\n")

        # VIF Hesaplama
        vif_data = pd.DataFrame()
        vif_data["Variable"] = X.columns[1:]  # const hariç
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) / 10 for i in range(1, X.shape[1])]  # Değerleri 10'a böl 

        # VIF Sonuçlarını Yazdır
        text_output.insert(tk.END, "\nVariance Inflation Factor (VIF) Results:\n")
        text_output.insert(tk.END, vif_data.to_string(index=False))

    except Exception as e:
        messagebox.showerror("Error", f"Error evaluating model: {e}")


def perform_anova():
    global data, dependent_var
    if data is not None and dependent_var:
        try:
            # Automatically generate formula for all independent variables
            independent_vars = [col for col in data.select_dtypes(include=['number']).columns if col != dependent_var]
            formula = f"{dependent_var} ~ {' + '.join(independent_vars)}"
            
            # Print formula for debugging
            print(f"Generated formula for ANOVA: {formula}")
            
            # Create a formula-based model
            model = ols(formula, data=data).fit()
            
            # Perform ANOVA analysis
            anova_results = anova_lm(model, typ=2)  # typ=2: Type II ANOVA
            
            # Display results in the text output area
            text_output.delete("1.0", tk.END)
            text_output.insert(tk.END, "ANOVA Results:\n\n")
            text_output.insert(tk.END, anova_results.to_string())
            
            # Optional: Print to console
            print("ANOVA Results:\n", anova_results)
        except Exception as e:
            messagebox.showerror("Error", f"Error performing ANOVA: {e}")
    else:
        messagebox.showerror("Error", "No dependent variable selected or dataset loaded!")

def perform_anova():
    global data, dependent_var
    if data is not None and dependent_var:
        try:
            # Tüm bağımsız değişkenleri belirleyin
            independent_vars = [col for col in data.select_dtypes(include=['number']).columns if col != dependent_var]
            formula = f"{dependent_var} ~ {' + '.join(independent_vars)}"
            
            # ANOVA için model oluştur
            model = ols(formula, data=data).fit()
            
            # ANOVA analizi
            anova_results = anova_lm(model, typ=2)  # Type II ANOVA
            
            # GUI'de göster
            text_output.delete("1.0", tk.END)
            text_output.insert(tk.END, "ANOVA Results:\n\n")
            text_output.insert(tk.END, anova_results.to_string())
            
            # Konsolda göster
            print("ANOVA Results:\n", anova_results)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error performing ANOVA: {e}")
    else:
        messagebox.showerror("Error", "No dependent variable selected or dataset loaded!")



def calculate_influence_and_outliers():
    model = perform_regression()  # Regresyon modelini çağır
    if model:
        try:
            # Etki ölçütlerini hesapla
            influence = model.get_influence()
            cooks_d, _ = influence.cooks_distance  # Cook's Distance
            leverage = influence.hat_matrix_diag  # Leverage
            standardized_residuals = influence.resid_studentized_internal  # Standardized Residuals
            studentized_residuals = influence.resid_studentized_external  # Studentized Residuals
            
            # Standart sapmadan uzak gözlemleri bul
            outliers = [i for i, r in enumerate(standardized_residuals) if abs(r) > 3]

            # Etki ölçütlerini bir DataFrame olarak düzenle
            influence_data = pd.DataFrame({
                "Index": range(len(cooks_d)),
                "Cook's Distance": cooks_d,
                "Leverage": leverage,
                "Standardized Residuals": standardized_residuals,
                "Studentized Residuals": studentized_residuals
            })

            # Kritik değerler
            critical_cooks_d = 4 / len(data)
            critical_leverage = 2 * (model.df_model + 1) / len(data)
            influence_data["High Cook's D"] = influence_data["Cook's Distance"] > critical_cooks_d
            influence_data["High Leverage"] = influence_data["Leverage"] > critical_leverage

            # Text output'a yazdır
            text_output.delete("1.0", tk.END)
            text_output.insert(tk.END, "Influence Measures:\n")
            text_output.insert(tk.END, influence_data.to_string(index=False))
            text_output.insert(tk.END, "\n\nOutliers (Standardized Residuals > |3|):\n")
            text_output.insert(tk.END, str(outliers))

            # Konsola yazdır
            print("Influence Measures:\n", influence_data)
            print("\nOutliers (Standardized Residuals > |3|):", outliers)

            # Cook's Distance grafiği
            plt.figure(figsize=(10, 6))
            plt.stem(influence_data["Index"], cooks_d, markerfmt=",", basefmt=" ", use_line_collection=True)
            plt.axhline(critical_cooks_d, color="red", linestyle="--", label=f"Critical Cook's D = {critical_cooks_d:.4f}")
            plt.title("Cook's Distance")
            plt.xlabel("Observation Index")
            plt.ylabel("Cook's Distance")
            plt.legend()
            plt.grid(alpha=0.5, linestyle="--")
            plt.show()

            # Leverage vs Studentized Residuals grafiği
            plt.figure(figsize=(10, 6))
            plt.scatter(leverage, studentized_residuals, alpha=0.6, edgecolor="k")
            plt.axhline(y=0, color="red", linestyle="--")
            plt.axhline(y=3, color="blue", linestyle="--", label="Threshold ±3")
            plt.axhline(y=-3, color="blue", linestyle="--")
            plt.axvline(x=critical_leverage, color="green", linestyle="--", label=f"Critical Leverage = {critical_leverage:.4f}")
            plt.title("Leverage vs Studentized Residuals")
            plt.xlabel("Leverage")
            plt.ylabel("Studentized Residuals")
            plt.legend()
            plt.grid(alpha=0.5, linestyle="--")
            plt.show()

        except Exception as e:
            messagebox.showerror("Error", f"Error calculating influence measures and detecting outliers: {e}")
    else:
        messagebox.showerror("Error", "Regression model is not available!")











# GUI setup
app = tk.Tk()
app.title("Python Diagnostic Plot Tool")
app.geometry("1200x800")

# Frames for layout
left_frame = tk.Frame(app, width=300, bg="#000000")
left_frame.pack(side="left", fill="y", padx=10, pady=10)

right_frame = tk.Frame(app, width=900)
right_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)

# Left Frame: Buttons and dependent variable dropdown
tk.Label(left_frame, text="Select Dependent Variable:", bg="#000000", fg="white").grid(row=0, column=0, columnspan=2, pady=5)
dependent_var_dropdown = ttk.Combobox(left_frame, state="readonly", width=30)
dependent_var_dropdown.bind("<<ComboboxSelected>>", set_dependent_var)
dependent_var_dropdown.grid(row=1, column=0, columnspan=2, pady=5)

# Buttons in two-column layout

button_list = [
    ("Load Dataset", load_file),
    ("Perform Regression", perform_regression),
    ("Check Outliers", check_outliers),
    ("Show Diagnostic Plots", plot_diagnostics),
    ("Test Homoscedasticity", test_homoscedasticity),
    ("Test Normality of Residuals", test_normality),
    ("Visualize VIF", visualize_vif),
    ("Plot Correlation Heatmap", plot_correlation_heatmap),
    ("Split Dataset (Train/Test)", split_dataset),
    ("Train and Test Model", train_and_test_model),
    ("Run Stepwise Regression", run_stepwise),
    ("Run Best Subset Regression", run_best_subset),
    ("Plot Individual Boxplots", plot_individual_boxplots),
    ("Compare Models", compare_models),
    ("Test Independence (Durbin-Watson)", test_independence),
    ("Determine Final Model", determine_final_model),
    ("Scatterplot Matrix", scatterplot_matrix),
    ("Plot All Distributions", plot_all_distributions),
    ("OLS Step Backward", ols_step_backward),
    ("OLS Step Forward", ols_step_forward),
    ("OLS All Step Possible", ols_all_step_possible),
    ("Calculate Prediction Intervals", lambda: calculate_prediction_intervals(example_input)),
    ("Best Subset Regression (per column)", best_subset_per_column),
    ("Evaluate and Export Model", evaluate_and_export_model),
    ("Best Subset and Export", best_subset_and_export),
    ("Compare Models & Select Best", compare_models_and_select_best),
    ("Train Test and Export Model 1", lambda: train_test_and_export_model("Model1")),
    ("Display Confidence Intervals", display_confidence_intervals),
    ("Summarize Data", summarize_data),
    ("Add Quadratic Term & Evaluate", lambda: add_quadratic_term_and_evaluate("Sonuçlar")),
    ("Perform ANOVA", perform_anova),
    ("Calculate Influence & Detect Outliers", calculate_influence_and_outliers),

   

     
    
     

    
]

for idx, (text, command) in enumerate(button_list):
    tk.Button(left_frame, text=text, command=command, width=20).grid(row=2 + idx // 2, column=idx % 2, padx=5, pady=5)
# Right Frame: Text output
text_output = tk.Text(right_frame, height=25, wrap="word")
text_output.pack(fill="both", expand=True, padx=10, pady=10)

# Run the app
app.mainloop()
