import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
from PIL import Image
import joblib



# Título de la aplicación con subtítulo para las autoras
st.title('NSCD - Clasificador de Nevus')
st.subheader('Autoras: Silvia Prieto y Celia Fernández')

# Sección para subir una imagen y completar los datos del paciente
st.header("Sube una imagen y completa los datos del paciente")

# Subir imagen
uploaded_file = st.file_uploader("Elige una imagen para analizar...", type=["jpg", "png"])

# Formulario para ingresar datos numéricos
st.subheader("Introduce los datos numéricos")
numerical_columns = ['age_approx', 'clin_size_long_diam_mm', 'tbp_lv_A', 'tbp_lv_Aext', 'tbp_lv_B', 'tbp_lv_Bext', 'tbp_lv_C', 
    'tbp_lv_Cext', 'tbp_lv_H', 'tbp_lv_Hext', 'tbp_lv_L', 'tbp_lv_Lext', 'tbp_lv_areaMM2', 
    'tbp_lv_area_perim_ratio', 'tbp_lv_color_std_mean', 'tbp_lv_deltaA', 'tbp_lv_deltaB', 'tbp_lv_deltaL', 
    'tbp_lv_deltaLB', 'tbp_lv_deltaLBnorm', 'tbp_lv_eccentricity', 'tbp_lv_minorAxisMM', 'tbp_lv_norm_border', 
    'tbp_lv_norm_color', 'tbp_lv_perimeterMM', 'tbp_lv_radial_color_std_max', 'tbp_lv_stdL', 
    'tbp_lv_stdLExt', 'tbp_lv_symm_2axis', 'tbp_lv_symm_2axis_angle']

# Inicializar un diccionario para almacenar los valores ingresados
inputs = {label: None for label in numerical_columns}

# Layout para mostrar los inputs numéricos en varias filas
for i in range(0, len(numerical_columns), 3):  # Dividir en grupos de 3 por fila
    cols = st.columns(3)
    for j, label in enumerate(numerical_columns[i:i+3]):
        inputs[label] = cols[j].text_input(label)


# Formulario para ingresar datos categóricos
st.subheader("Introduce los datos categóricos")
categorical_columns = ['sex', 'anatom_site_general', 'tbp_lv_location_simple']
col4, col5, col6 = st.columns(3)

with col4:
    sex = st.selectbox('Sexo', ['male', 'female'])

with col5:
    anatom_site_general = st.selectbox('Sitio anatómico general', ['lower extremity', 'head/neck', 'posterior torso',
        'anterior torso', 'upper extremity'])

with col6:
    tbp_lv_location_simple = st.selectbox('Localización simple', ['Right Leg', 'Head & Neck', 'Torso Back', 'Torso Front',
        'Right Arm', 'Left Leg', 'Left Arm'])

# Barra para configurar el umbral de clasificación
st.subheader("Configura el umbral para clasificar maligno")
umbral = st.slider("Elige el umbral de probabilidad para maligno", 0.0, 1.0, 0.5, 0.01)

# Botón para confirmar la entrada de datos
if st.button('Confirmar datos'):

    # Cargar el modelo y preprocesador solo cuando se presiona el botón
    modelo = load_model('modelos/best_combined_model20240920_094128_.keras')
    preprocessor = joblib.load('datos/pipeline_preprocesado.pkl')

    # Verificar que todos los inputs estén completos
    if all(value is not None for value in inputs.values()) and sex and anatom_site_general and tbp_lv_location_simple:
        # Crear un DataFrame con los datos ingresados
        new_data = pd.DataFrame({
            'age_approx': [inputs['age_approx']],
            'clin_size_long_diam_mm': [inputs['clin_size_long_diam_mm']],
            'tbp_lv_A': [inputs['tbp_lv_A']],
            'tbp_lv_Aext': [inputs['tbp_lv_Aext']],
            'tbp_lv_B': [inputs['tbp_lv_B']],
            'tbp_lv_Bext': [inputs['tbp_lv_Bext']],
            'tbp_lv_C': [inputs['tbp_lv_C']],
            'tbp_lv_Cext': [inputs['tbp_lv_Cext']],
            'tbp_lv_H': [inputs['tbp_lv_H']],
            'tbp_lv_Hext': [inputs['tbp_lv_Hext']],
            'tbp_lv_L': [inputs['tbp_lv_L']],
            'tbp_lv_Lext': [inputs['tbp_lv_Lext']],
            'tbp_lv_areaMM2': [inputs['tbp_lv_areaMM2']],
            'tbp_lv_area_perim_ratio': [inputs['tbp_lv_area_perim_ratio']],
            'tbp_lv_color_std_mean': [inputs['tbp_lv_color_std_mean']],
            'tbp_lv_deltaA': [inputs['tbp_lv_deltaA']],
            'tbp_lv_deltaB': [inputs['tbp_lv_deltaB']],
            'tbp_lv_deltaL': [inputs['tbp_lv_deltaL']],
            'tbp_lv_deltaLB': [inputs['tbp_lv_deltaLB']],
            'tbp_lv_deltaLBnorm': [inputs['tbp_lv_deltaLBnorm']],
            'tbp_lv_eccentricity': [inputs['tbp_lv_eccentricity']],
            'tbp_lv_minorAxisMM': [inputs['tbp_lv_minorAxisMM']],
            'tbp_lv_norm_border': [inputs['tbp_lv_norm_border']],
            'tbp_lv_norm_color': [inputs['tbp_lv_norm_color']],
            'tbp_lv_perimeterMM': [inputs['tbp_lv_perimeterMM']],
            'tbp_lv_radial_color_std_max': [inputs['tbp_lv_radial_color_std_max']],
            'tbp_lv_stdL': [inputs['tbp_lv_stdL']],
            'tbp_lv_stdLExt': [inputs['tbp_lv_stdLExt']],
            'tbp_lv_symm_2axis': [inputs['tbp_lv_symm_2axis']],
            'tbp_lv_symm_2axis_angle': [inputs['tbp_lv_symm_2axis_angle']],
            'sex': [sex],
            'anatom_site_general': [anatom_site_general],
            'tbp_lv_location_simple': [tbp_lv_location_simple]
        })

        # Mostrar los datos ingresados en una tabla
        st.subheader("Datos ingresados")
        st.table(new_data)

        # Proceder con el procesamiento de la imagen y la predicción si se ha subido una imagen
        if uploaded_file is not None:
            # Procesar la imagen
            image = Image.open(uploaded_file)
            st.image(image, caption='Imagen cargada.', use_column_width=True)

            # Preprocesar la imagen
            image = image.resize((224, 224))  # Cambiar al tamaño esperado
            image = np.array(image) / 255.0  # Normalizar
            image = np.expand_dims(image, axis=0)  # Añadir una dimensión extra

            # Preprocesar los datos numéricos y categóricos
            preprocessed_data = preprocessor.transform(new_data)

            # Combinar la imagen y los datos preprocesados
            combined_input = [image, preprocessed_data]
            
            # Realizar predicciones (obtener probabilidades)
            prediccion_prob = modelo.predict(combined_input)

            # Aplicar el umbral para clasificar como maligno o benigno
            prediccion = (prediccion_prob[:, 1] > umbral).astype(int)  

            # Mostrar el resultado de la predicción
            st.subheader("Resultado de la predicción")
            st.write(f"Predicción: {'Maligno' if prediccion == 1 else 'Benigno'} con un umbral de {umbral}")
            st.write(f"Probabilidad de maligno: {prediccion_prob[0][1]}")

    else:
        st.warning("Por favor, completa todos los campos requeridos.")
