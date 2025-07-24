import zipfile
import os

with zipfile.ZipFile("original.zip", mode="w") as archive:
    for filename in os.listdir("./original/"):
        archive.write(f"./original/{filename}")
