import os
import pandas as pd
from PIL import Image
from tqdm import tqdm


def get_image_stats(folder, ext=".jpg"):
    files = [f for f in os.listdir(folder) if f.endswith(ext)]
    if not files:
        return {}

    sizes, file_sizes_kb = [], []

    for f in tqdm(files, desc=f"Scanning {os.path.basename(folder)}"):
        path = os.path.join(folder, f)
        file_sizes_kb.append(os.path.getsize(path) / 1024)
        try:
            with Image.open(path) as img:
                sizes.append(img.size)
        except Exception:
            pass

    widths  = [s[0] for s in sizes]
    heights = [s[1] for s in sizes]
    unique  = set(sizes)

    return {
        "count"             : len(files),
        "max_width"         : max(widths),
        "min_width"         : min(widths),
        "max_height"        : max(heights),
        "min_height"        : min(heights),
        "unique_resolutions": len(unique),
        "all_same_size"     : "Yes" if len(unique) == 1 else "No",
        "common_resolution" : str(max(unique, key=lambda s: sizes.count(s))),
        "avg_file_size_kb"  : round(sum(file_sizes_kb) / len(file_sizes_kb), 2),
        "max_file_size_kb"  : round(max(file_sizes_kb), 2),
        "min_file_size_kb"  : round(min(file_sizes_kb), 2),
        "total_size_mb"     : round(sum(file_sizes_kb) / 1024, 2),
    }


def get_metadata_stats(csv_path, label_col='dx'):
    df = pd.read_csv(csv_path)
    stats = {
        "rows"          : len(df),
        "columns"       : len(df.columns),
        "missing_values": int(df.isnull().sum().sum()),
    }

    if label_col and label_col in df.columns:
        vc = df[label_col].value_counts()
        stats.update({
            "classes"           : df[label_col].nunique(),
            "class_distribution": vc.to_dict(),
            "majority_class"    : vc.idxmax(),
            "minority_class"    : vc.idxmin(),
            "imbalance_ratio"   : round(vc.max() / vc.min(), 2),  # higher = more skewed
        })

    if 'age' in df.columns:
        stats.update({
            "age_missing": int(df['age'].isnull().sum()),
            "age_mean"   : round(df['age'].mean(), 2),
            "age_min"    : df['age'].min(),
            "age_max"    : df['age'].max(),
        })

    if 'sex' in df.columns:
        stats["sex_distribution"] = df['sex'].value_counts().to_dict()

    if 'localization' in df.columns:
        stats["unique_localizations"] = df['localization'].nunique()
        stats["top_localization"]     = df['localization'].value_counts().idxmax()

    if 'dx_type' in df.columns:
        stats["dx_type_distribution"]  = df['dx_type'].value_counts().to_dict()
        # histo = biopsy confirmed, most reliable label
        stats["biopsy_confirmed_pct"]  = round(
            df['dx_type'].value_counts().get('histo', 0) / len(df) * 100, 2
        )

    if 'lesion_id' in df.columns:
        stats["unique_lesions"]          = df['lesion_id'].nunique()
        # same lesion photographed multiple times — important for correct train/val split
        stats["duplicate_lesion_images"] = len(df) - df['lesion_id'].nunique()

    return df, stats


# ── paths ──────────────────────────────────────────────────────────────────────
FOLDERS = {
    "part_1"      : "data/raw/images/part_1",
    "part_2"      : "data/raw/images/part_2",
    "test_images" : "data/raw/test_images",
    "segmentations": "data/raw/segmentations",
}
os.makedirs("data/reports", exist_ok=True)


# ── 1. image stats per folder ──────────────────────────────────────────────────
print("\n=== Scanning image folders ===")
img_rows = []
for name, path in FOLDERS.items():
    if not os.path.exists(path):
        print(f"  Skipping {name} — not found")
        continue
    ext   = ".png" if name == "segmentations" else ".jpg"
    stats = get_image_stats(path, ext=ext)
    stats["folder"] = name
    img_rows.append(stats)

df_img = pd.DataFrame(img_rows).set_index("folder")
df_img.to_csv("data/reports/image_stats.csv")


# ── 2. metadata stats ──────────────────────────────────────────────────────────
df_train, train_stats = get_metadata_stats("data/raw/HAM10000_metadata.csv", label_col='dx')
df_test,  test_stats  = get_metadata_stats("data/raw/ISIC2018_Task3_Test_GroundTruth.csv", label_col=None)


# ── 3. detect missing test images dynamically ──────────────────────────────────
test_imgs      = set(f.replace(".jpg", "") for f in os.listdir(FOLDERS["test_images"]) if f.endswith(".jpg"))
test_meta_ids  = set(df_test['image_id'])
KNOWN_MISSING  = test_meta_ids - test_imgs  # images in CSV but not on disk

df_test_clean  = df_test[~df_test['image_id'].isin(KNOWN_MISSING)]
test_stats["rows_after_drop"] = len(df_test_clean)
test_stats["known_missing"]   = list(KNOWN_MISSING)


# ── 4. train file integrity check ─────────────────────────────────────────────
all_train_imgs = set(
    f.replace(".jpg", "")
    for folder in [FOLDERS["part_1"], FOLDERS["part_2"]]
    for f in os.listdir(folder)
)
train_missing = set(df_train['image_id']) - all_train_imgs

file_check = pd.DataFrame([
    {"check": "Part 1 images",           "count": len(os.listdir(FOLDERS["part_1"])), "status": "OK"},
    {"check": "Part 2 images",           "count": len(os.listdir(FOLDERS["part_2"])), "status": "OK"},
    {"check": "Train metadata rows",     "count": len(df_train),                       "status": "OK"},
    {"check": "Train missing images",    "count": len(train_missing),                  "status": "OK" if not train_missing else "FAIL"},
    {"check": "Test images on disk",     "count": len(test_imgs),                      "status": "OK"},
    {"check": "Test metadata raw",       "count": len(df_test),                        "status": "WARN" if KNOWN_MISSING else "OK"},
    {"check": "Test metadata clean",     "count": len(df_test_clean),                  "status": "OK"},
    {"check": "Known missing test imgs", "count": len(KNOWN_MISSING),                  "status": f"{list(KNOWN_MISSING)}"},
])
file_check.to_csv("data/reports/file_check_summary.csv", index=False)


# ── 5. class distribution ──────────────────────────────────────────────────────
df_train['dx'].value_counts().rename_axis('class').reset_index(name='count') \
    .to_csv("data/reports/class_distribution.csv", index=False)


# ── 6. flatten all stats into master summary ───────────────────────────────────
def flatten(stats, source, category):
    return [{"category": category, "source": source, "metric": k, "value": str(v)}
            for k, v in stats.items()]

summary = []

for _, row in df_img.iterrows():
    summary += flatten(row.to_dict(), row.name, "image_folder")

summary += flatten(train_stats, "train_metadata", "metadata")
summary += flatten(test_stats,  "test_metadata",  "metadata")

for _, row in file_check.iterrows():
    summary.append({
        "category": "file_check", "source": "dataset",
        "metric": row["check"], "value": f"{row['count']} | {row['status']}"
    })

pd.DataFrame(summary).to_csv("data/reports/raw_data_summary.csv", index=False)


print("\n=== Reports saved to data/reports/ ===")
print("  raw_data_summary.csv     — master")
print("  image_stats.csv          — dimensions + file sizes per folder")
print("  class_distribution.csv   — per class counts")
print("  file_check_summary.csv   — integrity check")