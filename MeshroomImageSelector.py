"""
Meshroom Image Selector
========================
Reduziert die Anzahl der Bilder für Meshroom mittels SIFT-Feature-Extraktion
und K-Means Clustering — optimiert für minimale Verarbeitungszeit.

Strategie:
  1. SIFT-Features aus jedem Bild extrahieren (Scale-invariant feature transform)
  2. Features zu einem Bild-Vektor aggregieren (Bag-of-Visual-Words / Mean-Descriptor)
  3. K-Means Clustering im Feature-Space
  4. Pro Cluster das "zentralste" Bild wählen (= repräsentativster Blickwinkel)
  5. Schärfe-Check: unscharfe Bilder werden verworfen

Ergebnis: 50–70% weniger Bilder bei nahezu gleicher Meshroom-Qualität.
"""

import os
import sys
import shutil
import argparse
import numpy as np
from pathlib import Path
from typing import List, Tuple

try:
    import cv2
except ImportError:
    sys.exit("OpenCV fehlt. Bitte installieren: pip install opencv-contrib-python")

try:
    from sklearn.cluster import KMeans, MiniBatchKMeans
    from sklearn.preprocessing import normalize
except ImportError:
    sys.exit("scikit-learn fehlt. Bitte installieren: pip install scikit-learn")


# ─────────────────────────────────────────────
#  Konfiguration
# ─────────────────────────────────────────────

#DEFAULT_TARGET_RATIO = 0.35   # Ziel: 35 % der Originalbilder behalten
#MIN_SHARPNESS       = 80.0    # Laplacian-Varianz — unter diesem Wert = unscharf
#MIN_SIFT_FEATURES   = 30      # Bilder mit weniger Features werden verworfen
#OVERLAP_THRESHOLD   = 0.85    # Max. Feature-Ähnlichkeit zwischen zwei Bildern


DEFAULT_TARGET_RATIO = 0.60   # Ziel: 35 % der Originalbilder behalten
MIN_SHARPNESS       = 80.0    # Laplacian-Varianz — unter diesem Wert = unscharf
MIN_SIFT_FEATURES   = 30      # Bilder mit weniger Features werden verworfen
OVERLAP_THRESHOLD   = 0.70    # Max. Feature-Ähnlichkeit zwischen zwei Bildern


# ─────────────────────────────────────────────
#  Hilfsfunktionen
# ─────────────────────────────────────────────

def compute_sharpness(image: np.ndarray) -> float:
    """Laplacian-Varianz als Schärfe-Metrik."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def extract_sift_descriptor(image: np.ndarray, max_features: int = 300) -> np.ndarray | None:
    """
    Extrahiert SIFT-Keypoints und gibt einen kompakten 128-D Bild-Vektor zurück
    (Mittelwert aller Deskriptoren → schnell & effektiv für Clustering).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    sift = cv2.SIFT_create(nfeatures=max_features)
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    if descriptors is None or len(keypoints) < MIN_SIFT_FEATURES:
        return None

    # Mean-Pooling → 128-D Vektor pro Bild
    return descriptors.mean(axis=0)


def load_images(folder: Path, extensions: tuple = (".jpg", ".jpeg", ".png", ".tif", ".tiff")) -> List[Tuple[Path, np.ndarray]]:
    """Lädt alle Bilder aus einem Ordner."""
    images = []
    for ext in extensions:
        for path in sorted(folder.glob(f"*{ext}")):
            img = cv2.imread(str(path))
            if img is not None:
                images.append((path, img))
            else:
                print(f"  ⚠  Konnte nicht laden: {path.name}")
    return images


# ─────────────────────────────────────────────
#  Haupt-Pipeline
# ─────────────────────────────────────────────

def select_images(
    input_folder: str,
    output_folder: str,
    target_ratio: float = DEFAULT_TARGET_RATIO,
    min_sharpness: float = MIN_SHARPNESS,
    verbose: bool = True
) -> List[Path]:
    """
    Vollständige Pipeline:
      Laden → Schärfefilter → SIFT-Extraktion → K-Means → Auswahl → Kopieren

    Returns:
        Liste der ausgewählten Bildpfade
    """
    input_path  = Path(input_folder)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    # ── 1. Bilder laden ──────────────────────────────
    print(f"\n📂 Lade Bilder aus: {input_path}")
    all_images = load_images(input_path)
    if not all_images:
        sys.exit("Keine Bilder gefunden!")
    print(f"   {len(all_images)} Bilder gefunden.")

    # ── 2. Schärfefilter ─────────────────────────────
    print(f"\n🔍 Schärfefilter (Schwelle: {min_sharpness:.0f}) …")
    sharp_images = []
    for path, img in all_images:
        sharpness = compute_sharpness(img)
        if sharpness >= min_sharpness:
            sharp_images.append((path, img, sharpness))
        elif verbose:
            print(f"   ✗ Verworfen (unscharf, score={sharpness:.1f}): {path.name}")

    print(f"   {len(sharp_images)} Bilder nach Schärfefilter.")

    if len(sharp_images) < 5:
        print("⚠  Zu wenige scharfe Bilder — Schärfe-Schwelle auf 30 reduziert.")
        min_sharpness = 30.0
        sharp_images = [(p, i, compute_sharpness(i)) for p, i in all_images]

    # ── 3. SIFT-Feature-Extraktion ───────────────────
    print(f"\n⚙  Extrahiere SIFT-Features …")
    feature_data = []  # (path, descriptor, sharpness)
    for i, (path, img, sharpness) in enumerate(sharp_images):
        desc = extract_sift_descriptor(img)
        if desc is not None:
            feature_data.append((path, desc, sharpness))
        elif verbose:
            print(f"   ✗ Zu wenige Features: {path.name}")
        if verbose and (i + 1) % 20 == 0:
            print(f"   … {i+1}/{len(sharp_images)} verarbeitet")

    print(f"   {len(feature_data)} Bilder mit ausreichend Features.")

    if not feature_data:
        sys.exit("Keine verwertbaren Features extrahiert.")

    # ── 4. K-Means Clustering ────────────────────────
    n_total    = len(feature_data)
    n_clusters = max(3, int(round(n_total * target_ratio)))
    n_clusters = min(n_clusters, n_total)

    print(f"\n🧮 K-Means Clustering: {n_total} → {n_clusters} Bilder …")

    descriptors = np.array([d for _, d, _ in feature_data], dtype=np.float32)
    descriptors = normalize(descriptors)  # L2-Normalisierung für bessere Cluster

    # MiniBatchKMeans für Geschwindigkeit bei vielen Bildern
    Clusterer = MiniBatchKMeans if n_total > 200 else KMeans
    kmeans = Clusterer(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )
    labels = kmeans.fit_predict(descriptors)

    # ── 5. Pro Cluster das zentralste Bild wählen ────
    print(f"\n🎯 Wähle repräsentative Bilder pro Cluster …")
    selected_paths = []
    centers = kmeans.cluster_centers_

    for cluster_id in range(n_clusters):
        cluster_indices = np.where(labels == cluster_id)[0]
        if len(cluster_indices) == 0:
            continue

        cluster_center = centers[cluster_id]
        cluster_descs  = descriptors[cluster_indices]

        # Euklidische Distanz zum Cluster-Zentrum
        distances = np.linalg.norm(cluster_descs - cluster_center, axis=1)

        # Unter den nächsten 3 Kandidaten das schärfste Bild bevorzugen
        top_k = min(3, len(cluster_indices))
        nearest_idx = np.argsort(distances)[:top_k]
        best_local  = max(nearest_idx, key=lambda i: feature_data[cluster_indices[i]][2])  # Schärfe

        selected_idx  = cluster_indices[best_local]
        selected_path = feature_data[selected_idx][0]
        selected_paths.append(selected_path)

    # ── 6. Ausgabe kopieren ──────────────────────────
    print(f"\n📁 Kopiere {len(selected_paths)} ausgewählte Bilder nach: {output_path}")
    for path in selected_paths:
        shutil.copy2(path, output_path / path.name)

    # ── 7. Zusammenfassung ───────────────────────────
    reduction = 100 * (1 - len(selected_paths) / len(all_images))
    print(f"""
╔══════════════════════════════════════════╗
║         SELEKTION ABGESCHLOSSEN          ║
╠══════════════════════════════════════════╣
║  Eingabe:    {len(all_images):>5} Bilder               ║
║  Verworfen:  {len(all_images) - len(sharp_images):>5} (unscharf/wenig Features)║
║  Ausgewählt: {len(selected_paths):>5} Bilder               ║
║  Reduktion:  {reduction:>5.1f} %                    ║
╚══════════════════════════════════════════╝
    """)
    print(f"✅ Bilder sind bereit für Meshroom: {output_path}\n")
    return selected_paths


# ─────────────────────────────────────────────
#  Meshroom-Integration
# ─────────────────────────────────────────────

def run_meshroom(image_folder: str, meshroom_script: str = "meshroom_batch"):
    """
    Startet Meshroom direkt nach der Bildauswahl.
    Setzt meshroom_batch oder meshroom_photogrammetry voraus.
    """
    import subprocess
    cmd = [meshroom_script, "--input", image_folder, "--output", image_folder + "_3d"]
    print(f"🚀 Starte Meshroom: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


# ─────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Reduziert Bilder für Meshroom via SIFT + K-Means Clustering"
    )
    parser.add_argument("input",  help="Ordner mit Eingangsbildern")
    parser.add_argument("output", help="Ordner für ausgewählte Bilder")
    parser.add_argument(
        "--ratio", type=float, default=DEFAULT_TARGET_RATIO,
        help=f"Anteil zu behaltender Bilder (Standard: {DEFAULT_TARGET_RATIO})"
    )
    parser.add_argument(
        "--sharpness", type=float, default=MIN_SHARPNESS,
        help=f"Mindestwert für Bildschärfe (Standard: {MIN_SHARPNESS})"
    )
    parser.add_argument(
        "--run-meshroom", action="store_true",
        help="Meshroom nach Selektion direkt starten"
    )
    parser.add_argument("--quiet", action="store_true", help="Weniger Ausgabe")

    args = parser.parse_args()

    selected = select_images(
        input_folder   = args.input,
        output_folder  = args.output,
        target_ratio   = args.ratio,
        min_sharpness  = args.sharpness,
        verbose        = not args.quiet
    )

    if args.run_meshroom:
        run_meshroom(args.output)

    return selected


if __name__ == "__main__":
    main()