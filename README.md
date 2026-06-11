# Squats mit KI

**Autor:** Bohdan Samoliuk

**GitHub Repository:**
https://github.com/dentist202/Zwischenprodukt

## Projektbeschreibung

Dieses Programm erkennt die Körperhaltung einer Person in Echtzeit mithilfe von MediaPipe. Dabei werden die Körperpunkte (Pose Landmarks) sowohl in 2D als auch in 3D dargestellt.

Zusätzlich berechnet das Programm:

* den rechten Kniewinkel
* den Neigungswinkel des Oberkörpers

Die Werte werden direkt während der Bewegung angezeigt und können zur Analyse einer Kniebeuge und Neigungswinkel des Oberkörpers verwendet werden.

---

## Voraussetzungen

Empfohlen wird **Python 3.11**, da diese Version mit MediaPipe 0.10.9 oder 0.10.35 zuverlässig funktioniert.

Benötigte Bibliotheken:

```bash
pip install mediapipe==0.10.35
pip install opencv-python
pip install numpy
pip install matplotlib
```

---

## Installation

### 1. Modell herunterladen

Laden Sie die Datei `pose_landmarker_full.task` herunter:

https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task

Speichern Sie die Datei im selben Ordner wie das Python-Projekt.

### 2. Videoquelle auswählen

In `main.py` kann entweder ein Video oder die Webcam verwendet werden.

Video:

```python
cap = cv2.VideoCapture("Pfad/zum/Video.mov")
```

Webcam:

```python
cap = cv2.VideoCapture(0)
```

### 3. Programm starten

```bash
python main.py
```

Zum Beenden die Taste **q** drücken.

---

## Ausgabe

Während der Ausführung werden folgende Informationen angezeigt:

| Wert                   | Beschreibung                   |
| ---------------------- | ------------------------------ |
| Right Knee Angle       | Kniewinkel des rechten Beins   |
| Right Torso Lean Angle | Neigungswinkel des Oberkörpers |

Typische Werte:

| Bewegung        | Kniewinkel |
| --------------- | ---------- |
| Stehen          | ca. 170°   |
| Tiefe Kniebeuge | ca. 70–80° |

Der Oberkörperwinkel ist beim aufrechten Stand nahe 0-10° und steigt beim Vorbeugen an.
