"""
YOLOv11 信号機専用モデル 学習スクリプト
========================================
使い方:
    python train.py --data traffic_light.yaml --epochs 100
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def train(data: str, epochs: int, imgsz: int, batch: int, base_model: str):
    print(f"📦 ベースモデル: {base_model}")
    print(f"📊 データセット: {data}")
    print(f"🔁 エポック数:   {epochs}")

    model = YOLO(base_model)

    results = model.train(
        data=data,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        name="yolo11_traffic",
        patience=20,
        save=True,
    )

    # 最良モデルを models/ にコピー
    best = Path(f"runs/detect/yolo11_traffic/weights/best.pt")
    if best.exists():
        dest = Path("models/yolo11_traffic.pt")
        dest.parent.mkdir(exist_ok=True)
        import shutil
        shutil.copy(best, dest)
        print(f"\n✅ モデルを保存しました: {dest}")
    else:
        print("\n⚠️ best.pt が見つかりませんでした。runs/ を確認してください。")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv11 Traffic Light Training")
    parser.add_argument("--data",       default="traffic_light.yaml", help="データセット設定ファイル")
    parser.add_argument("--epochs",     type=int, default=100)
    parser.add_argument("--imgsz",      type=int, default=640)
    parser.add_argument("--batch",      type=int, default=16)
    parser.add_argument("--base-model", default="yolo11n.pt", help="ベースモデル (yolo11n/s/m/l/x.pt)")
    args = parser.parse_args()

    train(args.data, args.epochs, args.imgsz, args.batch, args.base_model)
