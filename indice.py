import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import folium
from scipy.interpolate import griddata
from streamlit_folium import st_folium
import tempfile

# Configuración de la página
st.set_page_config(page_title="Interpolación de Nutrientes", layout="wide")

# Subir archivo Excel
st.title("Visualización de Nutrientes en Cultivo de Papa")
uploaded_file = st.file_uploader(
    "Sube el archivo Excel con los datos de los nutrientes", type=["xlsx"])

if uploaded_file:
    # Leer datos
    data = pd.read_excel(uploaded_file)

    # Mostrar los primeros datos
    st.subheader("Datos Cargados")
    st.write(data.head())

    # Crear mapa interactivo con Folium
    st.subheader("Mapa Interactivo de Nutrientes")
    mapa = folium.Map(
        location=[data["Latitud"].mean(), data["Longitud"].mean()], zoom_start=16)

    for _, row in data.iterrows():
        folium.CircleMarker(
            location=[row["Latitud"], row["Longitud"]],
            radius=5,
            color="blue",
            fill=True,
            fill_color="blue",
            fill_opacity=0.9,
            popup=f"N: {row['N']}, P: {row['P']}, K: {row['K']}",
        ).add_to(mapa)

    # Mostrar el mapa en Streamlit
    st_map = st_folium(mapa, width=700, height=500)

    # Parámetro de interpolación
    paso = 0.00001
    cordx = np.arange(data["Longitud"].min(), data["Longitud"].max(), paso)
    cordy = np.arange(data["Latitud"].min(), data["Latitud"].max(), paso)
    xx, yy = np.meshgrid(cordx, cordy)

    def interpolar_nutriente(data, nutriente):
        matz = np.zeros((len(cordy), len(cordx)))
        for i in range(len(cordx)):
            for j in range(len(cordy)):
                distancia = 1 / \
                    (np.sqrt((data["Latitud"] - cordy[j]) **
                     2 + (data["Longitud"] - cordx[i])**2))
                indice = np.isinf(distancia)
                if not indice.any():
                    factor = distancia / np.sum(distancia)
                    matz[j, i] = np.sum(data[nutriente] * factor)
                else:
                    matz[j, i] = data[nutriente][indice.argmax()]
        return matz

    # Visualización de gráficos interactivos
    nutriente_seleccionado = st.selectbox(
        "Selecciona el nutriente para interpolar", ["N", "P", "K"])
    matz = interpolar_nutriente(data, nutriente_seleccionado)

    fig = go.Figure()
    fig.add_trace(go.Contour(
        z=matz,
        x=cordx,
        y=cordy,
        contours=dict(start=0, end=max(
            data[nutriente_seleccionado]), size=1, showlabels=False),
        line_smoothing=1,
    ))
    fig.add_trace(go.Scatter(
        x=data["Longitud"],
        y=data["Latitud"],
        mode="markers",
        marker=dict(color="black", size=5),
        name="Puntos originales",
    ))
    fig.update_layout(
        width=1000,
        height=800,
        title=f"Interpolación de {nutriente_seleccionado}",
        xaxis_title="Longitud",
        yaxis_title="Latitud",
    )

    st.plotly_chart(fig, use_container_width=True)

    # Consulta de valor interpolado
    st.subheader("Consulta de Valor Interpolado")
    latitud_consulta = st.number_input(
        "Ingresa la latitud:", format="%.6f", value=data["Latitud"].mean())
    longitud_consulta = st.number_input(
        "Ingresa la longitud:", format="%.6f", value=data["Longitud"].mean())

    def obtener_valor_interpolado(latitud, longitud, data, nutriente):
        distancia = 1 / \
            (np.sqrt((data["Latitud"] - latitud) **
             2 + (data["Longitud"] - longitud)**2))
        indice = np.isinf(distancia)
        if not indice.any():
            factor = distancia / np.sum(distancia)
            valor_interpolado = np.sum(data[nutriente] * factor)
        else:
            valor_interpolado = data[nutriente][indice.argmax()]
        return valor_interpolado

    valor_interpolado = obtener_valor_interpolado(
        latitud_consulta, longitud_consulta, data, nutriente_seleccionado)
    st.write(
        f"El valor interpolado de {nutriente_seleccionado} en ({latitud_consulta}, {longitud_consulta}) es: {valor_interpolado:.2f}")
