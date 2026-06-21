import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, 
    confusion_matrix, classification_report, roc_auc_score,
    balanced_accuracy_score, matthews_corrcoef
)
from sklearn.utils.class_weight import compute_class_weight
from datasets import load_dataset
import urllib.parse
import re
import time
import warnings
import logging
import os
import gc

# --- CONFIGURACIÓN ---
# Suprimir warnings
warnings.filterwarnings('ignore')
logging.getLogger("datasets").setLevel(logging.ERROR)

# Configurar token de Hugging Face (REEMPLAZA CON TU TOKEN REAL)
HF_TOKEN = "hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"

# Configurar el token en variable de entorno
os.environ['HF_TOKEN'] = HF_TOKEN
print("✅ Token de Hugging Face configurado")

print("="*80)
print("🛡️ SISTEMA DE DETECCIÓN DE PHISHING - CON DATASET PHRESHPHISH")
print("="*80)

# ============================================
# SELECCIÓN DEL PORCENTAJE DE MUESTRAS
# ============================================
print("\n" + "="*80)
print("📊 SELECCIÓN DEL TAMAÑO DEL DATASET")
print("="*80)
print("\nOpciones de muestra:")
print("  1. 1%   (~5,000 muestras)  - Ultra rápido")
print("  2. 5%   (~25,000 muestras) - Rápido")
print("  3. 10%  (~50,000 muestras) - Recomendado para pruebas")
print("  4. 30%  (~150,000 muestras) - Buen rendimiento")
print("  5. 50%  (~250,000 muestras) - Excelente rendimiento")
print("  6. 100% (~500,000 muestras) - Máxima precisión (requiere mucha RAM)")
print("  7. Personalizado (especificar porcentaje)")

while True:
    try:
        opcion_muestra = input("\n👉 Selecciona una opción (1-7): ").strip()
        
        if opcion_muestra == '1':
            porcentaje = 1
            break
        elif opcion_muestra == '2':
            porcentaje = 5
            break
        elif opcion_muestra == '3':
            porcentaje = 10
            break
        elif opcion_muestra == '4':
            porcentaje = 30
            break
        elif opcion_muestra == '5':
            porcentaje = 50
            break
        elif opcion_muestra == '6':
            porcentaje = 100
            break
        elif opcion_muestra == '7':
            porcentaje = float(input("Ingresa el porcentaje (1-100): "))
            if 1 <= porcentaje <= 100:
                break
            else:
                print("❌ Porcentaje inválido. Debe ser entre 1 y 100.")
        else:
            print("❌ Opción no válida. Selecciona 1-7.")
    except ValueError:
        print("❌ Ingresa un número válido.")

# Convertir porcentaje a string para el split
if porcentaje == 100:
    split_str = 'train'
else:
    split_str = f'train[:{porcentaje}%]'

print(f"\n✅ Cargando {porcentaje}% del dataset ({split_str})")

start_time = time.time()

# 1. Cargar dataset
print("\n[1] Cargando dataset PhreshPhish desde Hugging Face...")

try:
    print(f"Cargando muestra del {porcentaje}% del dataset...")
    
    # PASAR EL TOKEN DIRECTAMENTE EN load_dataset
    dataset = load_dataset(
        'phreshphish/phreshphish', 
        split=split_str,
        token=HF_TOKEN  # <--- TOKEN PASADO DIRECTAMENTE
    )
    
    print("✅ Dataset cargado exitosamente!")
    
    # Convertir a DataFrame optimizado
    print("Convirtiendo a DataFrame...")
    df = pd.DataFrame({
        'URL': dataset['url'],
        'Result': [0 if label == 'benign' else 1 for label in dataset['label']]
    })
    
    # Liberar memoria del dataset original
    del dataset
    gc.collect()
    
    print(f"📊 Total de muestras: {len(df)}")
    print(f"📊 Legítimo (0): {sum(df['Result']==0)} ({sum(df['Result']==0)/len(df)*100:.1f}%)")
    print(f"📊 Phishing (1): {sum(df['Result']==1)} ({sum(df['Result']==1)/len(df)*100:.1f}%)")

except Exception as e:
    print(f"❌ Error cargando el dataset: {e}")
    print("\n📌 Soluciones:")
    print("1. Usa un porcentaje más pequeño")
    print("2. Verifica tu conexión a internet")
    print("3. Asegúrate de tener suficiente memoria RAM")
    print("4. Verifica que tu token sea válido en: https://huggingface.co/settings/tokens")
    exit(1)

# 2. Extracción de características optimizada
print("\n[2] Extrayendo características de URLs...")

def extraer_caracteristicas_url(url):
    """Extrae características de una URL para el modelo"""
    try:
        parsed = urllib.parse.urlparse(url)
        domain = parsed.netloc
        path = parsed.path
        query = parsed.query
        
        if not domain:
            return np.zeros(18)
        
        caracteristicas = []
        
        # 1. IP en lugar de dominio
        ip_pattern = r'^\d+\.\d+\.\d+\.\d+$'
        caracteristicas.append(1 if re.match(ip_pattern, domain) else -1)
        
        # 2. Longitud de URL
        if len(url) > 75:
            caracteristicas.append(1)
        elif len(url) < 54:
            caracteristicas.append(-1)
        else:
            caracteristicas.append(0)
        
        # 3. Servicio de acortamiento
        acortadores = ['bit.ly', 'goo.gl', 'tinyurl', 'ow.ly', 'is.gd', 't.co']
        caracteristicas.append(1 if any(x in domain for x in acortadores) else -1)
        
        # 4. Símbolo @
        caracteristicas.append(1 if '@' in url else -1)
        
        # 5. Doble slash
        if '//' in url[url.find('://')+3:]:
            caracteristicas.append(1)
        else:
            caracteristicas.append(-1)
        
        # 6. Guiones en el dominio
        caracteristicas.append(1 if '-' in domain else -1)
        
        # 7. Número de subdominios
        sub_count = domain.count('.')
        if sub_count > 2:
            caracteristicas.append(1)
        elif sub_count == 2:
            caracteristicas.append(0)
        else:
            caracteristicas.append(-1)
        
        # 8. SSL (HTTP vs HTTPS)
        caracteristicas.append(1 if url.startswith('http://') else -1)
        
        # 9. Edad del dominio (simulado)
        caracteristicas.append(1 if any(x in url.lower() for x in ['archive', 'old']) else -1)
        
        # 10. Token HTTPS
        caracteristicas.append(1 if 'https' in url.lower() else -1)
        
        # 11. Longitud del path
        if len(path) > 20:
            caracteristicas.append(1)
        elif len(path) < 5:
            caracteristicas.append(-1)
        else:
            caracteristicas.append(0)
        
        # 12. Presencia de parámetros en query
        caracteristicas.append(1 if len(query) > 0 else -1)
        
        # 13. Número de parámetros en query
        param_count = len(query.split('&')) if query else 0
        if param_count > 3:
            caracteristicas.append(1)
        elif param_count == 0:
            caracteristicas.append(-1)
        else:
            caracteristicas.append(0)
        
        # 14. Presencia de fragmento (#)
        caracteristicas.append(1 if parsed.fragment else -1)
        
        # 15. El dominio contiene números
        caracteristicas.append(1 if any(c.isdigit() for c in domain) else -1)
        
        # 16. URL contiene caracteres especiales
        special_chars = set('!@#$%^&*()_+{}|:"<>?`~')
        caracteristicas.append(1 if any(c in special_chars for c in url) else -1)
        
        # 17. Longitud del dominio
        if len(domain) > 20:
            caracteristicas.append(1)
        elif len(domain) < 8:
            caracteristicas.append(-1)
        else:
            caracteristicas.append(0)
        
        # 18. Contiene nombre de marca conocida
        marcas = ['banamex', 'bbva', 'banorte', 'santander', 'google', 
                 'facebook', 'paypal', 'amazon', 'microsoft', 'apple']
        caracteristicas.append(1 if any(marca in domain.lower() for marca in marcas) else -1)
        
        return np.array(caracteristicas)
        
    except:
        return np.zeros(18)

# Extraer características con progreso
print(f"Extrayendo características de {len(df)} URLs...")
X_features = []

for i, url in enumerate(df['URL']):
    if i % 1000 == 0 and i > 0:
        print(f"  Procesadas {i} de {len(df)} URLs...")
    X_features.append(extraer_caracteristicas_url(url))

X_features = np.array(X_features)
y = df['Result'].values

# Liberar memoria del DataFrame
del df
gc.collect()

print(f"✅ Características extraídas: {X_features.shape}")

# 3. Dividir datos
print("\n[3] Dividiendo datos en entrenamiento y prueba...")

X_train, X_test, y_train, y_test = train_test_split(
    X_features, y, test_size=0.2, random_state=42, stratify=y
)

# Liberar memoria de X_features
del X_features
gc.collect()

# 4. Escalar características
print("\n[4] Escalando características...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. PCA
print("\n[5] Aplicando PCA...")
n_components = min(15, X_train_scaled.shape[1])
pca = PCA(n_components=n_components)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print(f"✅ Varianza explicada por PCA: {pca.explained_variance_ratio_.sum():.3f}")

# 6. Entrenar modelo
print("\n[6] Entrenando modelo Random Forest...")

# Calcular pesos de clase
class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y_train)
class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}

# Ajustar número de estimadores según el tamaño del dataset
if porcentaje <= 5:
    n_estimators = 50
elif porcentaje <= 20:
    n_estimators = 100
elif porcentaje <= 50:
    n_estimators = 150
else:
    n_estimators = 200

print(f"📊 Usando {n_estimators} árboles para {porcentaje}% del dataset")

rf_model = RandomForestClassifier(
    n_estimators=n_estimators,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    class_weight=class_weight_dict,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train_pca, y_train)
print("✅ Modelo entrenado!")

# 7. Evaluación
print("\n[7] Evaluando modelo...")

y_pred = rf_model.predict(X_test_pca)
y_proba = rf_model.predict_proba(X_test_pca)[:, 1]

# Métricas
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
balanced_acc = balanced_accuracy_score(y_test, y_pred)
mcc = matthews_corrcoef(y_test, y_pred)
auc_roc = roc_auc_score(y_test, y_proba)

print(f"\n📊 RESULTADOS DEL MODELO CON PHRESHPHISH ({porcentaje}%):")
print(f"  ✅ Accuracy:          {accuracy:.4f}")
print(f"  ✅ F1-Score:          {f1:.4f}")
print(f"  ✅ Precision:         {precision:.4f}")
print(f"  ✅ Recall:            {recall:.4f}")
print(f"  ✅ Balanced Accuracy: {balanced_acc:.4f}")
print(f"  ✅ MCC:               {mcc:.4f}")
print(f"  ✅ AUC-ROC:           {auc_roc:.4f}")

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
print(f"\n📊 Matriz de Confusión:")
print(f"               Predicho")
print(f"               Legítimo  Phishing")
print(f"  Legítimo     {cm[0][0]:5d}   {cm[0][1]:5d}")
print(f"  Phishing     {cm[1][0]:5d}   {cm[1][1]:5d}")

# Tasas de error
fp_rate = cm[0][1] / (cm[0][0] + cm[0][1]) if (cm[0][0] + cm[0][1]) > 0 else 0
fn_rate = cm[1][0] / (cm[1][0] + cm[1][1]) if (cm[1][0] + cm[1][1]) > 0 else 0

print(f"\n⚠️ TASAS DE ERROR:")
print(f"  Falsos Positivos: {fp_rate:.3f} ({cm[0][1]} sitios legítimos mal clasificados)")
print(f"  Falsos Negativos: {fn_rate:.3f} ({cm[1][0]} sitios phishing no detectados)")

# Interpretación
print(f"\n📈 INTERPRETACIÓN DEL MODELO:")
if balanced_acc > 0.85:
    print("  ✅ EXCELENTE - El modelo distingue perfectamente entre clases")
elif balanced_acc > 0.75:
    print("  ✅ BUENO - Modelo funcional y confiable")
elif balanced_acc > 0.65:
    print("  ⚠️ ACEPTABLE - Mejorable con más características")
else:
    print("  ❌ DEFICIENTE - El modelo no está aprendiendo adecuadamente")

if mcc > 0.7:
    print("  ✅ Correlación excelente")
elif mcc > 0.5:
    print("  ✅ Buena correlación")
elif mcc > 0.3:
    print("  ⚠️ Correlación moderada")
else:
    print("  ❌ Correlación débil (casi aleatorio)")

# 8. Validación cruzada
print("\n[8] Validación cruzada...")
cv_scores = cross_val_score(rf_model, X_train_pca, y_train, cv=3, scoring='f1')
print(f"✅ F1-Score promedio CV: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

elapsed_time = time.time() - start_time
print(f"\n⏱️ Tiempo total de ejecución: {elapsed_time:.2f} segundos")

# 9. Funciones de detección
print("\n[9] Inicializando detector...")

# Lista de dominios legítimos
DOMINIOS_LEGITIMOS = [
    'banamex.com', 'www.banamex.com', 'bancanet.banamex.com',
    'bbva.mx', 'www.bbva.mx', 'banorte.com', 'www.banorte.com',
    'santander.com.mx', 'www.santander.com.mx', 'citibanamex.com',
    'hsbc.com.mx', 'www.hsbc.com.mx', 'scotiabank.com.mx',
    'inbursa.com', 'www.inbursa.com', 'afirme.com', 'www.afirme.com',
    'google.com', 'www.google.com', 'gmail.com',
    'facebook.com', 'www.facebook.com', 'twitter.com', 'www.twitter.com',
    'instagram.com', 'www.instagram.com', 'youtube.com', 'www.youtube.com',
    'linkedin.com', 'www.linkedin.com', 'whatsapp.com', 'telegram.org',
    'amazon.com', 'www.amazon.com', 'amazon.com.mx', 'www.amazon.com.mx',
    'mercadolibre.com.mx', 'www.mercadolibre.com.mx', 'walmart.com.mx',
    'paypal.com', 'www.paypal.com', 'netflix.com', 'www.netflix.com',
    'spotify.com', 'www.spotify.com', 'uber.com', 'www.uber.com',
    'gob.mx', 'www.gob.mx', 'sat.gob.mx', 'www.sat.gob.mx',
    'edu.mx', 'unam.mx', 'www.unam.mx', 'ipn.mx', 'www.ipn.mx',
    'outlook.com', 'hotmail.com', 'yahoo.com',
    'office.com', 'microsoft.com', 'apple.com', 'icloud.com'
]

PATRONES_PHISHING = [
    r'banamex[^c]', r'b4namex', r'banamexx', r'baanamex',
    r'banamex[0-9]', r'banamex\-', r'bbva[^.]', r'bbva[0-9]',
    r'banorte[^.]', r'banorte[0-9]', r'santander[^.]', r'santander[0-9]',
    r'paypal[^.]', r'paypal[0-9]', r'amazon[^.]', r'amazon[0-9]',
    r'google[^.]', r'google[0-9]', r'facebook[^.]', r'facebook[0-9]'
]

TLDS_SOSPECHOSOS = [
    '.az', '.xyz', '.top', '.club', '.gq', '.ml', 
    '.tk', '.cf', '.ga', '.cc', '.co', '.buzz', '.live',
    '.online', '.site', '.tech', '.space', '.host'
]

def es_dominio_legitimo(domain):
    domain_clean = domain.lower().replace('https://', '').replace('http://', '')
    domain_clean = domain_clean.split('/')[0]
    
    for legitimo in DOMINIOS_LEGITIMOS:
        if domain_clean == legitimo or domain_clean.endswith('.' + legitimo):
            return True
    return False

def es_sospechoso_por_nombre(domain):
    domain_clean = domain.lower()
    
    for patron in PATRONES_PHISHING:
        if re.search(patron, domain_clean):
            return True
    
    if any(domain_clean.endswith(tld) for tld in TLDS_SOSPECHOSOS):
        return True
    
    return False

def analizar_url(url):
    """Analiza una URL y determina si es phishing"""
    try:
        parsed = urllib.parse.urlparse(url)
        domain = parsed.netloc
        
        if not domain:
            return {
                'resultado': '❌ URL inválida',
                'phishing': False,
                'probabilidad': 0,
                'razon': 'URL no válida',
                'confianza': 'BAJA'
            }
        
        # 1. Lista blanca
        if es_dominio_legitimo(domain):
            if url.startswith('http://'):
                return {
                    'resultado': '⚠️ LEGÍTIMO PERO INSEGURO',
                    'phishing': False,
                    'probabilidad': 0.15,
                    'razon': 'Dominio legítimo con HTTP inseguro',
                    'confianza': 'ALTA'
                }
            else:
                return {
                    'resultado': '✅ LEGÍTIMO',
                    'phishing': False,
                    'probabilidad': 0.01,
                    'razon': 'Dominio verificado en lista blanca',
                    'confianza': 'ALTA'
                }
        
        # 2. Patrones sospechosos
        if es_sospechoso_por_nombre(domain):
            return {
                'resultado': '🚨 PHISHING DETECTADO',
                'phishing': True,
                'probabilidad': 0.95,
                'razon': 'Patrón de imitación detectado',
                'confianza': 'ALTA'
            }
        
        # 3. HTTP inseguro
        if url.startswith('http://'):
            palabras_clave = ['login', 'signin', 'account', 'bank', 'banco', 
                            'security', 'payment', 'verify']
            if any(p in url.lower() for p in palabras_clave):
                return {
                    'resultado': '⚠️ POSIBLE PHISHING',
                    'phishing': True,
                    'probabilidad': 0.85,
                    'razon': 'HTTP inseguro en página sensible',
                    'confianza': 'MEDIA'
                }
        
        # 4. Modelo ML
        caracteristicas = extraer_caracteristicas_url(url)
        if len(caracteristicas) == 0:
            caracteristicas = np.zeros(18)
        
        caracteristicas_scaled = scaler.transform(caracteristicas.reshape(1, -1))
        caracteristicas_pca = pca.transform(caracteristicas_scaled)
        
        prediccion = rf_model.predict(caracteristicas_pca)[0]
        probabilidad = rf_model.predict_proba(caracteristicas_pca)[0][1]
        
        # Ajustes
        if url.startswith('http://'):
            probabilidad = min(0.95, probabilidad + 0.2)
        
        es_phishing = prediccion == 1
        
        # Confianza
        if probabilidad > 0.8:
            confianza = 'ALTA'
        elif probabilidad > 0.5:
            confianza = 'MEDIA'
        else:
            confianza = 'BAJA'
        
        return {
            'resultado': '🚨 PHISHING' if es_phishing else '✅ LEGÍTIMO',
            'phishing': es_phishing,
            'probabilidad': probabilidad,
            'razon': 'Modelo ML' if not es_phishing else 'Modelo ML detecta phishing',
            'confianza': confianza
        }
        
    except Exception as e:
        return {
            'resultado': '❌ ERROR',
            'phishing': False,
            'probabilidad': 0,
            'razon': f'Error: {str(e)}',
            'confianza': 'BAJA'
        }

# 10. Interfaz de usuario
print("\n" + "="*80)
print("🔍 ANALIZADOR DE PHISHING CON PHRESHPHISH")
print("="*80)
print(f"\n✅ Dataset: {porcentaje}% ({len(y_train) + len(y_test)} muestras)")
print(f"✅ Entrenamiento completado en {elapsed_time:.2f} segundos")
print("="*80)

EJEMPLOS = [
    "https://www.google.com",
    "https://www.banamex.com",
    "https://www.bbva.mx",
    "https://www.facebook.com",
    "https://paypal.com",
    "http://www.banamex.com",
    "https://banamex-login.top",
    "https://www.b4namex.xyz",
    "https://paypal-verify-security.com",
    "https://www.banamexx.com",
]

while True:
    print("\n" + "-"*80)
    print("📋 OPCIONES:")
    print("  1. Analizar una URL")
    print("  2. Probar con ejemplos")
    print("  3. Ver dominios legítimos")
    print("  4. Ver estadísticas")
    print("  5. Cambiar porcentaje de muestras")
    print("  6. Salir")
    
    opcion = input("\n👉 Selecciona una opción (1-6): ").strip()
    
    if opcion == '1':
        print("\n" + "="*80)
        url = input("🔗 Ingresa la URL: ").strip()
        
        if not url:
            print("❌ No ingresaste ninguna URL")
            continue
            
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        print("\n" + "="*80)
        print("🔄 ANALIZANDO...")
        print("="*80)
        
        resultado = analizar_url(url)
        
        print(f"\n📌 URL: {url}")
        print(f"📊 Resultado: {resultado['resultado']}")
        print(f"🎯 Probabilidad: {resultado['probabilidad']:.1%}")
        print(f"📝 Razón: {resultado['razon']}")
        print(f"🎯 Confianza: {resultado['confianza']}")
        
        if resultado['phishing']:
            print("\n" + "!"*60)
            print("🚨 ¡ALERTA DE SEGURIDAD!")
            print("NO ingreses información personal o bancaria.")
            print("!"*60)
        else:
            print("\n" + "✓"*60)
            print("✅ Sitio verificado - Sin amenazas")
            print("✓"*60)
            
    elif opcion == '2':
        print("\n" + "="*80)
        print("📋 EJEMPLOS DE PRUEBA")
        print("="*80)
        
        print("\nLista de ejemplos para analizar:")
        for i, ej in enumerate(EJEMPLOS, 1):
            print(f"{i:2}. {ej}")
        
        try:
            seleccion = input("\nIngresa el número (1-10) o 'todos': ").strip()
            
            if seleccion.lower() == 'todos':
                urls_a_analizar = EJEMPLOS
            else:
                num = int(seleccion) - 1
                if 0 <= num < len(EJEMPLOS):
                    urls_a_analizar = [EJEMPLOS[num]]
                else:
                    print("❌ Número inválido")
                    continue
            
            print("\n" + "="*80)
            print("📊 RESULTADOS DEL ANÁLISIS")
            print("="*80)
            
            for i, url in enumerate(urls_a_analizar, 1):
                resultado = analizar_url(url)
                
                print(f"\n{i}. URL: {url}")
                print(f"   Resultado: {resultado['resultado']}")
                print(f"   Probabilidad: {resultado['probabilidad']:.1%}")
                print(f"   Confianza: {resultado['confianza']}")
                print(f"   Razón: {resultado['razon']}")
                
        except ValueError:
            print("❌ Ingresa un número válido o 'todos'")
            
    elif opcion == '3':
        print("\n" + "="*80)
        print(f"✅ DOMINIOS LEGÍTIMOS ({len(DOMINIOS_LEGITIMOS)})")
        print("="*80)
        print("\nCategorías:")
        print("  🏦 Bancos: Banamex, BBVA, Banorte, Santander, HSBC, Inbursa, Afirme")
        print("  🌐 Redes Sociales: Google, Facebook, Twitter, Instagram, YouTube, LinkedIn")
        print("  🛍️ Comercio: Amazon, MercadoLibre, Walmart, PayPal, Netflix, Spotify")
        print("  🏛️ Gobierno: gob.mx, SAT, UNAM, IPN")
        print("  📧 Correo: Gmail, Outlook, Hotmail, Yahoo, Microsoft, Apple")
        
        ver_todos = input("\n¿Ver todos los dominios? (s/n): ").strip().lower()
        if ver_todos == 's':
            print("\nLista completa de dominios legítimos:")
            for i, dominio in enumerate(sorted(DOMINIOS_LEGITIMOS), 1):
                print(f"{i:3}. {dominio}")
        
    elif opcion == '4':
        print("\n" + "="*80)
        print("📊 ESTADÍSTICAS DEL MODELO")
        print("="*80)
        print(f"📊 Porcentaje de datos: {porcentaje}%")
        print(f"📊 Muestras totales: {len(y_train) + len(y_test)}")
        print(f"📊 Muestras entrenamiento: {len(y_train)}")
        print(f"📊 Muestras prueba: {len(y_test)}")
        print(f"📊 Componentes PCA: {n_components}")
        print(f"📊 Árboles Random Forest: {n_estimators}")
        print(f"\n🎯 Accuracy:          {accuracy:.4f}")
        print(f"🎯 F1-Score:          {f1:.4f}")
        print(f"🎯 Precision:         {precision:.4f}")
        print(f"🎯 Recall:            {recall:.4f}")
        print(f"🎯 Balanced Accuracy: {balanced_acc:.4f}")
        print(f"🎯 MCC:               {mcc:.4f}")
        print(f"🎯 AUC-ROC:           {auc_roc:.4f}")
        print(f"\n📊 Falsos Positivos: {fp_rate:.3f} ({cm[0][1]} sitios)")
        print(f"📊 Falsos Negativos: {fn_rate:.3f} ({cm[1][0]} sitios)")
        print(f"\n⏱️ Tiempo entrenamiento: {elapsed_time:.2f}s")
        
    elif opcion == '5':
        print("\n" + "="*80)
        print("🔄 CAMBIAR PORCENTAJE DE MUESTRAS")
        print("="*80)
        print("\n⚠️ Esto recargará el dataset y reentrenará el modelo.")
        print("   El proceso puede tomar varios minutos.")
        
        confirmar = input("\n¿Continuar? (s/n): ").strip().lower()
        if confirmar == 's':
            # Recargar el script con nuevo porcentaje
            print("\n🔄 Reiniciando con nuevo porcentaje...")
            # Guardar el porcentaje seleccionado y reiniciar
            exec(open(__file__).read())
            break
        else:
            print("❌ Operación cancelada")
            
    elif opcion == '6':
        print("\n" + "="*80)
        print("👋 ¡GRACIAS POR USAR EL ANALIZADOR!")
        print("Mantente seguro en línea. 🔒")
        print("="*80)
        break
        
    else:
        print("❌ Opción no válida. Selecciona 1, 2, 3, 4, 5 o 6.")

# Resumen final
print("\n" + "="*80)
print("📊 RESUMEN DEL SISTEMA")
print("="*80)
print(f"Modelo: Random Forest con PCA")
print(f"Porcentaje de datos: {porcentaje}%")
print(f"Muestras: {len(y_train) + len(y_test)}")
print(f"Componentes PCA: {n_components}")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"Balanced Accuracy: {balanced_acc:.4f}")
print(f"Dominios en lista blanca: {len(DOMINIOS_LEGITIMOS)}")
print("="*80)
