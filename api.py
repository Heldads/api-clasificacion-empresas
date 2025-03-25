# 📌 Importamos librerías necesarias
from fastapi import FastAPI
import joblib
import numpy as np
from pydantic import BaseModel

# 📌 Cargamos el modelo y el escalador
modelo = joblib.load("C:/Users/Helmut Garcia/Downloads/clasificacion de rentabilidad de salud de empresa/mi api rentabilidad/modelo_empresas.pkl")  # Cargar el modelo entrenado
scaler = joblib.load("C:/Users/Helmut Garcia/Downloads/clasificacion de rentabilidad de salud de empresa/mi api rentabilidad/escalador.pkl")  # Cargar el escalador de datos

# 📌 Inicializamos la API
app = FastAPI()

# 📌 Definimos la estructura de los datos de entrada
class DatosEntrada(BaseModel):
    ROE: float
    Margen_Utilidad_Neta: float
    Sector: str

# 📌 Lista de sectores disponibles
sectores = ["Tecnología", "Manufactura", "Retail", "Finanzas", "Salud"]

# 📌 Endpoint para la raíz
@app.get("/")
def read_root():
    return {"message": "Bienvenido a la API de Clasificación de Empresas"}

# 📌 Endpoint para favicon
@app.get("/favicon.ico")
def favicon():
    return {"message": "Favicon not found"}, 204

# 📌 Endpoint para predecir la clasificación de una empresa
@app.post("/predecir/")
def predecir(datos: DatosEntrada):
    try:
        # 📌 Extraemos los valores del JSON recibido
        roe = datos.ROE
        margen = datos.Margen_Utilidad_Neta
        sector = datos.Sector

        # 📌 Verificamos si el sector es válido
        if sector not in sectores:
            return {"error": "Sector no válido. Usa uno de estos: " + ", ".join(sectores)}

        # 📌 Codificamos el sector en formato de variables dummy
        sector_data = [1 if s == sector else 0 for s in sectores[1:]]  # Omitimos la primera categoría

        # 📌 Creamos un array con los datos de entrada
        input_data = np.array([[roe, margen] + sector_data])
        input_data = scaler.transform(input_data)  # Normalizamos los datos

        # 📌 Realizamos la predicción con el modelo
        prediccion = modelo.predict(input_data)[0]

        # 📌 Retornamos el resultado
        return {"Clasificación": prediccion}
    
    except Exception as e:
        return {"error": str(e)}
