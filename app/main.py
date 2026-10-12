from fastapi import FastAPI

app = FastAPI()

"""Quand accès à l'URL et envoi d'une requête GET : appel de root() pour envoi de réponse en .json"""

@app.get("/") # Création d'une route HTTP GET pour l'échange d'informations entre mon ordinateur et l'API (avec "/" comme racine de l'API)
def root():
    return {"message": "The API is activated. More content soon"}
