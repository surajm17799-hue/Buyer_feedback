from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive

def get_gdrive():
    gauth = GoogleAuth()
    gauth.LocalWebserverAuth()  # Opens a browser for OAuth
    return GoogleDrive(gauth)
