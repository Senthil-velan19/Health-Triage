#!/usr/bin/env python3
"""
EdgeTriage Machine Learning Pipeline
-----------------------------------
This script contains a complete, production-ready pipeline to fine-tune,
export, and quantize a lightweight Vision Transformer (MobileViT) for
offline-first deployment in resource-constrained rural clinics.

Architecture Highlights:
1. Base Model: apple/mobilevit-small (~20MB parameters) combining CNN and ViT.
2. Fine-Tuning: Binary Classification (Normal vs. Anomaly/Pneumonia) using Hugging Face Trainer.
3. ONNX Conversion: optimum.onnxruntime for high-performance cross-platform execution.
4. INT8 Quantization: Dynamic quantization reducing model size (targeting <15MB)
   and latency on edge CPUs without requiring a GPU.

Requirements to install before running:
    pip install torch torchvision transformers datasets optimum[onnxruntime]
"""

# ==========================================================
# 0. DYNAMIC COMPATIBILITY PATCHES (TORCH & PROTOBUF)
# ==========================================================
import torch
import sys

# Patch missing torch attributes that onnxruntime helpers look up dynamically
if not hasattr(torch, "int4"):
    # Map int4 to int8 since we are not exporting an int4 model
    torch.int4 = getattr(torch, "int8", None)
if not hasattr(torch, "uint4"):
    torch.uint4 = getattr(torch, "uint8", None)

# Patch Protobuf 5.x/6.x MessageFactory GetPrototype deprecation/removal
try:
    from google.protobuf import message_factory
    if hasattr(message_factory, "MessageFactory") and not hasattr(message_factory.MessageFactory, "GetPrototype"):
        message_factory.MessageFactory.GetPrototype = lambda self, desc: message_factory.GetMessageClass(desc)
except ImportError:
    pass

import os
import shutil
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    MobileViTImageProcessor,
    MobileViTForImageClassification,
    TrainingArguments,
    Trainer,
    EvalPrediction
)
from optimum.onnxruntime import ORTModelForImageClassification, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)


# ==========================================
# 1. DUMMY CHEST X-RAY DATASET CREATION
# ==========================================
class DummyChestXrayDataset(Dataset):
    """
    A dummy dataset simulating Chest X-ray images for binary classification.
    Generates synthetic grayscale images representing lung fields with and without anomalies.
    """
    def __init__(self, num_samples: int, image_processor: MobileViTImageProcessor):
        self.num_samples = num_samples
        self.image_processor = image_processor
        
        # 0: Normal, 1: Anomaly (Pneumonia)
        self.labels = np.random.randint(0, 2, size=self.num_samples)
        
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        # Create a synthetic 224x224 grayscale image representing a chest structure
        # Normal (0) will have a clean dark lung cavity, Anomaly (1) will have random white infiltrates
        img_array = np.zeros((224, 224, 3), dtype=np.uint8)
        
        # Draw background cage/rib contours (static pattern)
        for y in range(40, 180, 20):
            img_array[y-2:y+2, 50:170] = 180  # Rib bones simulator
            
        # Draw lung cavities (dark regions)
        img_array[50:170, 40:100] = 30  # Left lung
        img_array[50:170, 120:180] = 30 # Right lung
        
        label = self.labels[idx]
        if label == 1:
            # Simulate "Pneumonia/Infiltrate" (patchy consolidations/higher intensity areas in lungs)
            # Add random cloud-like noise in lung cavities to represent fluid/opacity
            for _ in range(5):
                cy, cx = np.random.randint(70, 150), np.random.choice([60, 140])
                radius = np.random.randint(15, 30)
                y, x = np.ogrid[-cy:224-cy, -cx:224-cx]
                mask = y*y + x*x <= radius*radius
                img_array[mask] = np.clip(img_array[mask] + np.random.randint(80, 150), 0, 255)
                
        # Convert array to PIL Image
        pil_img = Image.fromarray(img_array)
        
        # Apply Hugging Face ImageProcessor for MobileViT (handles resize, scale, normalize)
        inputs = self.image_processor(images=pil_img, return_tensors="pt")
        
        # Reformat pixel values to eliminate batch dimension since PyTorch DataLoader wraps it
        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "label": int(label)
        }


# ==========================================
# 2. METRICS COMPILATION FUNCTION
# ==========================================
def compute_metrics(p: EvalPrediction):
    """
    Computes standard classification metrics: Accuracy, Precision, Recall, and F1.
    """
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    
    tp = np.sum((preds == 1) & (labels == 1))
    fp = np.sum((preds == 1) & (labels == 0))
    fn = np.sum((preds == 0) & (labels == 1))
    tn = np.sum((preds == 0) & (labels == 0))
    
    accuracy = (tp + tn) / len(labels)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# ==========================================
# MAIN EXECUTION ROUTINE
# ==========================================
def main():
    print("[1/5] Initializing MobileViT Model and Preprocessor...")
    model_id = "apple/mobilevit-small"
    
    # Load Image Processor & Model with 2 labels (Normal vs. Anomaly)
    image_processor = MobileViTImageProcessor.from_pretrained(model_id)
    model = MobileViTForImageClassification.from_pretrained(
        model_id,
        num_labels=2,
        ignore_mismatched_sizes=True, # Allows resetting classifier head
        id2label={0: "Normal", 1: "Pneumonia"},
        label2id={"Normal": 0, "Pneumonia": 1}
    )
    
    print("[2/5] Creating synthetic Chest X-ray train/validation datasets...")
    # Instantiate dummy dataset
    train_dataset = DummyChestXrayDataset(num_samples=100, image_processor=image_processor)
    val_dataset = DummyChestXrayDataset(num_samples=20, image_processor=image_processor)
    
    print(f"Generated {len(train_dataset)} training and {len(val_dataset)} validation samples.")
    
    # Setup training configurations
    training_args = TrainingArguments(
        output_dir="./vit_results",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=5,
        learning_rate=5e-5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        remove_unused_columns=False,
        report_to="none" # Disable W&B/TensorBoard logging for this demo
    )
    
    # Initialize HF Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )
    
    print("[3/5] Commencing MobileViT Fine-Tuning...")
    trainer.train()
    
    # Save the fine-tuned model locally
    fine_tuned_path = "./model_finetuned"
    trainer.save_model(fine_tuned_path)
    image_processor.save_pretrained(fine_tuned_path)
    print(f"Fine-tuned model successfully saved at: {fine_tuned_path}")
    
    print("[4/5] Converting Fine-Tuned PyTorch Model to ONNX format...")
    # Export using Hugging Face Optimum ORTModel
    onnx_path = "./model_onnx"
    onnx_model = ORTModelForImageClassification.from_pretrained(
        fine_tuned_path, 
        export=True
    )
    # Save ONNX model
    onnx_model.save_pretrained(onnx_path)
    print(f"ONNX Model saved successfully at: {onnx_path}")
    
    # Show pre-quantization size
    raw_onnx_file = os.path.join(onnx_path, "model.onnx")
    raw_size_mb = os.path.getsize(raw_onnx_file) / (1024 * 1024)
    print(f"Original ONNX model file size: {raw_size_mb:.2f} MB")
    
    print("[5/5] Performing INT8 Dynamic Quantization...")
    # Initialize ORTQuantizer
    quantizer = ORTQuantizer.from_pretrained(onnx_path)
    
    # Setup dynamic quantization targeting ARM64 / generic CPU architectures
    # Dynamic quantization quantizes weights to INT8 and dynamically scales activations at runtime.
    # It requires zero calibration data and offers the safest size/accuracy tradeoff for edge browsers.
    qconfig = AutoQuantizationConfig.arm64(
        is_static=False, 
        per_channel=False
    )
    
    quantized_path = "./model_quantized_dir"
    quantizer.quantize(
        save_dir=quantized_path,
        quantization_config=qconfig
    )
    quantized_file = os.path.join(quantized_path, "model_quantized.onnx")
    final_output_file = "./model_quantized.onnx"
    
    if os.path.exists(quantized_file):
        shutil.copy(quantized_file, final_output_file)
        quantized_size_mb = os.path.getsize(final_output_file) / (1024 * 1024)
        print(f"\nSuccess!")
        print(f"Quantized Model saved at: {final_output_file}")
        print(f"Quantized ONNX model file size: {quantized_size_mb:.2f} MB (Target: <15MB)")
        print(f"Size Reduction: {((raw_size_mb - quantized_size_mb) / raw_size_mb) * 100:.1f}%")
    else:
        print("Error: Could not locate the quantized ONNX file.")

if __name__ == "__main__":
    main()
