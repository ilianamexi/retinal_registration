import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mutual_info_score


DATASET_PATH = r"FIRE" 

IMAGES_DIR = os.path.join(DATASET_PATH, "Images")
GT_DIR = os.path.join(DATASET_PATH, "Ground Truth", "Control Points") 


# Επιλέγουμε 10 ζεύγη εικόνων 
PAIRS_TO_PROCESS = [
    ("A01_1.jpg", "A01_2.jpg", "control_points_A01_1_2.txt"),
    ("A02_1.jpg", "A02_2.jpg", "control_points_A02_1_2.txt"),
    ("A03_1.jpg", "A03_2.jpg", "control_points_A03_1_2.txt"),
    ("A04_1.jpg", "A04_2.jpg", "control_points_A04_1_2.txt"),
    ("A05_1.jpg", "A05_2.jpg", "control_points_A05_1_2.txt"),
    ("A06_1.jpg", "A06_2.jpg", "control_points_A06_1_2.txt"),
    ("A07_1.jpg", "A07_2.jpg", "control_points_A07_1_2.txt"),
    ("A08_1.jpg", "A08_2.jpg", "control_points_A08_1_2.txt"),
    ("A09_1.jpg", "A09_2.jpg", "control_points_A09_1_2.txt"),
    ("A10_1.jpg", "A10_2.jpg", "control_points_A10_1_2.txt"),
]

def read_fire_points(filepath):
    """
    Διαβάζει τα σημεία από το FIRE dataset format.
    Format: x1 y1 x2 y2
    """
    pts1, pts2 = [], []
    if not os.path.exists(filepath):
        # Δοκιμή εναλλακτικού path (απευθείας στο Ground Truth)
        filepath = filepath.replace("Control Points", "")
        filepath = filepath.replace("//", "/") # fix slash
        if not os.path.exists(filepath):
            print(f"  [!] Δεν βρέθηκε το αρχείο σημείων: {filepath}")
            return None, None

    with open(filepath, 'r') as f:
        for line in f:
            coords = [float(x) for x in line.strip().split()]
            if len(coords) >= 4:
                pts1.append([coords[0], coords[1]])
                pts2.append([coords[2], coords[3]])
    
    return np.array(pts1, dtype=np.float32), np.array(pts2, dtype=np.float32)

def calculate_metrics_masked(img1, img2):
    """
    Υπολογίζει Mutual Information (MI) και Correlation Coefficient (CC).
    Εξαιρεί τα μαύρα pixels (φόντο).
    """
    # Μετατροπή σε Grayscale
    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # Resize g2 να ταιριάζει στο g1 (αν χρειάζεται λόγω μετασχηματισμού)
    if g1.shape != g2.shape:
        g2 = cv2.resize(g2, (g1.shape[1], g1.shape[0]))

    # Μάσκα: Κρατάμε pixels που δεν είναι μαύρα και στις δύο εικόνες
    # Χρησιμοποιούμε threshold > 5 για να αποφύγουμε θόρυβο στο μαύρο
    mask = (g1 > 5) & (g2 > 5)
    
    val1 = g1[mask].ravel()
    val2 = g2[mask].ravel()
    
    if len(val1) == 0:
        return 0.0, 0.0

    # Correlation Coefficient
    cc = np.corrcoef(val1, val2)[0, 1]
    
    # Mutual Information
    mi = mutual_info_score(val1, val2)
    
    return cc, mi

def visualize_fusion(img1, img2, title, save_path=None):
    """
    Δημιουργεί εικόνα fusion (50% img1 + 50% img2).
    """
    # Resize για σιγουριά
    h, w = img1.shape[:2]
    img2_res = cv2.resize(img2, (w, h))
    
    # Add weighted
    fusion = cv2.addWeighted(img1, 0.5, img2_res, 0.5, 0)
    
    # Plotting
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(fusion, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


# MAIN LOOP
print(f"{'Ζεύγος':<10} | {'a11':<8} {'a12':<8} {'a21':<8} {'a22':<8} | {'MI Πριν':<8} {'MI Μετά':<8} | {'CC Πριν':<8} {'CC Μετά':<8}")
print("-" * 100)

for idx, (f1, f2, txt) in enumerate(PAIRS_TO_PROCESS):
    path1 = os.path.join(IMAGES_DIR, f1)
    path2 = os.path.join(IMAGES_DIR, f2)
    path_txt = os.path.join(GT_DIR, txt)
    
    # 1. Διάβασμα Εικόνων 
    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)
    
    if img1 is None or img2 is None:
        print(f"Skipping {f1}-{f2}: Images not found.")
        continue

    # 2. Metrics ΠΡΙΝ 
    cc_pre, mi_pre = calculate_metrics_masked(img1, img2)
    
    # Προβολή Fusion ΠΡΙΝ
    # visualize_fusion(img1, img2, f"Fusion Before {f1}")

    # 3. Διάβασμα Σημείων & Υπολογισμός Affine [cite: 11, 13]
    p1, p2 = read_fire_points(path_txt)
    
    if p1 is None or len(p1) < 3:
        print(f"Skipping {f1}: Not enough points.")
        continue
        
    # Υπολογισμός πίνακα Affine (2x3) ώστε το img2 να πάει στο img1
    # Χρησιμοποιούμε estimateAffine2D που είναι πιο robust (RANSAC)
    M, _ = cv2.estimateAffine2D(p2, p1) 
    
    if M is None:
        print(f"Transformation failed for {f1}")
        continue

    # 4. Εφαρμογή Μετασχηματισμού 
    rows, cols = img1.shape[:2]
    img2_registered = cv2.warpAffine(img2, M, (cols, rows))

    # 5. Metrics ΜΕΤΑ 
    cc_post, mi_post = calculate_metrics_masked(img1, img2_registered)
    
    # Προβολή Fusion ΜΕΤΑ 
    # visualize_fusion(img1, img2_registered, f"Fusion After {f1}")

    # 6. Εκτύπωση για τον Πίνακα της Αναφοράς 
    # Ο πίνακας Affine είναι [[a11, a12, tx], [a21, a22, ty]]
    a11, a12 = M[0, 0], M[0, 1]
    a21, a22 = M[1, 0], M[1, 1]
    
    pair_name = f1.split('.')[0] # π.χ. A01_1
    tx, ty = M[0, 2], M[1, 2]
    print(f"{pair_name:<6} | a11:{a11:.4f} a12:{a12:.4f} tx:{tx:.2f} | a21:{a21:.4f} a22:{a22:.4f} ty:{ty:.2f} | MI:{mi_post:.3f} CC:{cc_post:.3f}")