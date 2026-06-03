import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.preprocessing.image import img_to_array

# ============================================
# НАСТРОЙКА СТРАНИЦЫ
# ============================================
st.set_page_config(
    page_title="Vehicle Classifier",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🚗 Классификация транспортных средств")
st.markdown("Загрузите изображение, и модель определит тип транспортного средства")

# ============================================
# КЛАССЫ
# ============================================
class_names = ['Auto Rickshaws', 'Bikes', 'Cars', 'Motorcycles', 'Planes', 'Ships', 'Trains']

# ============================================
# ЗАГРУЗКА МОДЕЛЕЙ
# ============================================
@st.cache_resource
def load_models():
    """Загрузка обученных моделей"""
    models = {}
    
    model_configs = {
        "Model A (MobileNet + SE)": {
            "path": "model_a_mobilenet_se.keras",
            "size": (224, 224),
            "normalize": "none",
            "type": "single"
        },
        "Model B (CBAM + Parallel Pooling)": {
            "path": "model_b_mobilenet_cbam.keras",
            "size": (224, 224),
            "normalize": "none",
            "type": "single"
        },
        "Ансамбль (конкатенация)": {
            "path": "concat_ensemble.keras",
            "size": (224, 224),
            "normalize": "none",
            "type": "single"
        },
        "Своя модель (64x64)": {
            "path": "model_after_finetuning.keras",
            "size": (64, 64),
            "normalize": "rescale",
            "type": "single"
        },
        "Предобученная MobileNetV2": {
            "path": "pretrained_model.keras",
            "size": (224, 224),
            "normalize": "rescale",
            "type": "single"
        }
    }
    
    for name, config in model_configs.items():
        try:
            if os.path.exists(config["path"]):
                model = tf.keras.models.load_model(config["path"], compile=False, safe_mode=False)
                models[name] = {
                    "model": model,
                    "size": config["size"],
                    "normalize": config["normalize"],
                    "type": config["type"]
                }
                st.success(f"✅ {name} загружен")
            else:
                st.warning(f"⚠️ {name} не найден: {config['path']}")
        except Exception as e:
            st.error(f"❌ Ошибка загрузки {name}: {str(e)[:80]}...")
    
    return models

# ============================================
# ФУНКЦИЯ ПРЕДОБРАБОТКИ
# ============================================
def preprocess_image(image, target_size, normalize_type):
    """Предобработка изображения для конкретной модели"""
    image = image.resize(target_size)
    img_array = img_to_array(image)
    
    if normalize_type == "rescale":
        img_array = img_array / 255.0
    elif normalize_type == "none":
        pass
    
    img_array = np.expand_dims(img_array, axis=0)
    return img_array.astype(np.float32)

# ============================================
# ФУНКЦИЯ ПРЕДСКАЗАНИЯ
# ============================================
def predict_single(model_info, img_array):
    """Предсказание для одной модели"""
    model = model_info["model"]
    prediction = model.predict(img_array, verbose=0)
    return prediction[0]

def predict_all_models(models, original_image):
    """Предсказание для всех моделей (каждая со своим размером)"""
    results = {}
    for name, model_info in models.items():
        try:
            # Предобработка с правильным размером для каждой модели
            img_array = preprocess_image(
                original_image,
                target_size=model_info["size"],
                normalize_type=model_info["normalize"]
            )
            probs = predict_single(model_info, img_array)
            results[name] = probs
        except Exception as e:
            st.error(f"Ошибка при предсказании {name}: {e}")
            results[name] = None
    return results

# ============================================
# ВИЗУАЛИЗАЦИЯ
# ============================================
def display_comparison(image, all_results, top_n=3):
    """Отображение сравнения всех моделей"""
    
    st.image(image, caption="Загруженное изображение", use_container_width=True)
    
    st.subheader("📊 Сравнение результатов всех моделей")
    
    # Таблица сравнения
    comparison_data = []
    for model_name, probs in all_results.items():
        if probs is not None:
            top_indices = np.argsort(probs)[::-1][:top_n]
            best_class = class_names[top_indices[0]]
            best_conf = probs[top_indices[0]]
            
            comparison_data.append({
                "Модель": model_name,
                "Лучший класс": best_class,
                "Уверенность": f"{best_conf:.2%}"
            })
    
    df = pd.DataFrame(comparison_data)
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    # График сравнения
    st.subheader("📊 Сравнение уверенности лучших предсказаний")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = [d["Модель"] for d in comparison_data]
    confidences = [float(d["Уверенность"].replace('%', '')) / 100 for d in comparison_data]
    
    bars = ax.barh(models, confidences, color='skyblue')
    ax.set_xlabel('Уверенность')
    ax.set_title('Уверенность лучшего предсказания по моделям')
    ax.set_xlim(0, 1)
    
    for bar, conf in zip(bars, confidences):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
               f'{conf:.2%}', va='center')
    
    st.pyplot(fig)

def display_single_model(image, model_info, model_name, top_n=3):
    """Отображение результата одной модели"""
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image(image, caption="Загруженное изображение", use_container_width=True)
    
    with col2:
        st.subheader(f"📊 Результаты: {model_name}")
        
        # Предобработка и предсказание
        img_array = preprocess_image(
            image,
            target_size=model_info["size"],
            normalize_type=model_info["normalize"]
        )
        probs = predict_single(model_info, img_array)
        
        if probs is not None:
            top_indices = np.argsort(probs)[::-1][:top_n]
            for idx in top_indices:
                class_name = class_names[idx]
                confidence = probs[idx]
                st.write(f"**{class_name}**")
                st.progress(float(confidence))
                st.write(f"Уверенность: {confidence:.2%}")
                st.write("---")
            
            best_class = class_names[top_indices[0]]
            best_conf = probs[top_indices[0]]
            st.success(f"🎯 **Лучшее предсказание: {best_class}**")
            st.info(f"📈 Уверенность: {best_conf:.2%}")
            
            # График
            st.subheader("📊 График уверенности")
            fig, ax = plt.subplots(figsize=(10, 4))
            top_classes = [class_names[i] for i in top_indices]
            top_probs = [probs[i] for i in top_indices]
            
            bars = ax.barh(top_classes, top_probs, color='skyblue')
            ax.set_xlim(0, 1)
            ax.set_xlabel('Уверенность')
            
            for bar, prob in zip(bars, top_probs):
                ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                       f'{prob:.2%}', va='center')
            
            st.pyplot(fig)
            
            # Таблица
            st.subheader("📋 Детальная таблица")
            df = pd.DataFrame([(class_names[i], probs[i]) for i in top_indices], 
                             columns=["Класс", "Уверенность"])
            df["Уверенность (%)"] = df["Уверенность"].apply(lambda x: f"{x:.2%}")
            st.dataframe(df, use_container_width=True, hide_index=True)


# ============================================
# ОСНОВНАЯ ЧАСТЬ
# ============================================
def main():
    # Загрузка моделей
    with st.spinner("Загрузка моделей..."):
        models = load_models()
    
    if not models:
        st.error("❌ Не удалось загрузить ни одну модель.")
        st.info("""
        **Требуемые файлы:**
        - model_a_final.keras
        - model_b_final.keras
        - concat_ensemble.keras
        - model_after_finetuning.keras
        - pretrained_model.keras
        """)
        return
    
    # Боковая панель
    st.sidebar.header("⚙️ Настройки")
    
    mode = st.sidebar.radio(
        "Режим работы:",
        ["Сравнение всех моделей", "Выбор конкретной модели"]
    )
    
    top_n = st.sidebar.slider("Количество отображаемых классов (Top-N):", 1, 7, 3)

    
    st.sidebar.markdown("---")
    st.sidebar.subheader("🚗 Доступные классы")
    emojis = {"Auto Rickshaws": "🛺", "Bikes": "🏍️", "Cars": "🚗", 
              "Motorcycles": "🏍️", "Planes": "✈️", "Ships": "🛳️", "Trains": "🚂"}
    for i, name in enumerate(class_names, 1):
        st.sidebar.write(f"{emojis.get(name, '📷')} {i}. {name}")
    
    # Основной контент
    uploaded_file = st.file_uploader(
        "📁 Загрузите изображение...",
        type=['jpg', 'jpeg', 'png', 'bmp', 'webp']
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        if st.button("🔍 Распознать", type="primary"):
            with st.spinner("Анализ изображения..."):
                
                if mode == "Сравнение всех моделей":
                    all_results = predict_all_models(models, image)
                    display_comparison(image, all_results, top_n)
                    
                else:  # Выбор конкретной модели
                    model_names = list(models.keys())
                    selected_model = st.selectbox("Выберите модель:", model_names)
                    
                    if selected_model:
                        model_info = models[selected_model]
                        display_single_model(image, model_info, selected_model, top_n)
    
    else:
        st.info("👈 Загрузите изображение для начала работы")

if __name__ == "__main__":
    main()