import zipfile
import os

with zipfile.ZipFile("resolts.zip", mode="w") as archive:
    for filename in os.listdir("./results/french_probing"):
        archive.write(f"./results/french_probing/{filename}")
        try: 
            for probe_name in os.listdir(f"./results/french_probing/{filename}"):
                print(probe_name)
                archive.write(f"./results/french_probing/{filename}/{probe_name}")
        except:
            continue