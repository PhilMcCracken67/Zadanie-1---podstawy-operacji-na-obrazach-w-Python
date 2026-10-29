import urllib.request
import cv2
import numpy as np
import matplotlib.pyplot as plt



url = "https://upload.wikimedia.org/wikipedia/commons/3/3a/Cat03.jpg"


naglowki = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
zapytanie = urllib.request.Request(url, headers=naglowki)


req = urllib.request.urlopen(zapytanie)
arr = np.asarray(bytearray(req.read()), dtype=np.uint8)


img_bgr = cv2.imdecode(arr, -1)


img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img_rgb)
plt.title("Oryginalny obraz")
plt.axis('off')



nowa_szerokosc = int(img_bgr.shape[1] * 0.5)
nowa_wysokosc = int(img_bgr.shape[0] * 0.5)
wymiary = (nowa_szerokosc, nowa_wysokosc)


img_resized = cv2.resize(img_bgr, wymiary, interpolation=cv2.INTER_AREA)

img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

img_rotated = cv2.rotate(img_gray, cv2.ROTATE_90_CLOCKWISE)


plt.subplot(1, 2, 2)
plt.imshow(img_rotated, cmap='gray') 
plt.title("Zmniejszony, szary i obrocony")
plt.axis('off')


plt.show()


print("Wymiary wynikowej macierzy (Wysokosc x Szerokosc):", img_rotated.shape)
print("\nMacierz obrazu (wartosci pikseli 0-255):")
print(img_rotated)