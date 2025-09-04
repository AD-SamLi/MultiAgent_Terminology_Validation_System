#!/usr/bin/env python3
"""
Test script to verify NLLB model loading with CPU+GPU hybrid setup
"""

import torch
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def test_model_loading():
    """Test loading the NLLB model with accelerate"""
    
    print("🧪 NLLB Model Loading Test")
    print("=" * 40)
    
    # Check system info
    print(f"🐍 Python: {torch.__version__}")
    print(f"🔥 PyTorch: {torch.__version__}")
    print(f"🎮 CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name()}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    
    # Test model loading
    model_name = "facebook/nllb-200-1.3B"  # Smaller model
    
    try:
        print(f"\n📥 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("✅ Tokenizer loaded successfully")
        
        print(f"\n🧠 Loading model with accelerate...")
        
        # Create offload directory
        offload_dir = "./model_offload"
        os.makedirs(offload_dir, exist_ok=True)
        
        # Load with accelerate device mapping
        if torch.cuda.is_available():
            print("🎮 Using GPU + CPU hybrid loading...")
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,  # Half precision
                device_map="auto",          # Auto device mapping
                offload_folder=offload_dir, # Disk offloading if needed
                low_cpu_mem_usage=True,     # Minimize CPU memory
                offload_state_dict=True     # Offload state dict
            )
        else:
            print("💻 Using CPU with memory optimization...")
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                device_map="auto",          # Auto device mapping
                low_cpu_mem_usage=True,     # Minimize CPU memory
                offload_folder=offload_dir, # Disk offloading if needed
                torch_dtype=torch.float32   # Full precision for CPU
            )
        
        print("✅ Model loaded successfully with hybrid setup")
        
        # Print memory usage
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            print(f"🎮 GPU Memory - Allocated: {allocated:.1f}GB, Reserved: {reserved:.1f}GB")
        
        # Test a simple translation
        print(f"\n🔄 Testing translation...")
        
        # Prepare input
        text = "Hello world"
        tokenizer.src_lang = "eng_Latn"
        inputs = tokenizer(text, return_tensors="pt")
        
        # Generate translation
        with torch.no_grad():
            # Get the language token ID
            spa_token_id = tokenizer.convert_tokens_to_ids("spa_Latn")
            generated_tokens = model.generate(
                **inputs,
                forced_bos_token_id=spa_token_id,
                max_length=50
            )
        
        # Decode result
        translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        print(f"✅ Translation test: '{text}' -> '{translation}'")
        
        print(f"\n🎉 Model loading test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pipeline():
    """Test using transformers pipeline"""
    
    print(f"\n🔧 Testing transformers pipeline...")
    
    try:
        from transformers import pipeline
        
        # Create pipeline without specifying device (let accelerate handle it)
        translator = pipeline(
            "translation",
            model="facebook/nllb-200-1.3B",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        
        print("✅ Pipeline created successfully")
        
        # Test translation
        result = translator("Hello world", src_lang="eng_Latn", tgt_lang="spa_Latn")
        print(f"✅ Pipeline test: {result}")
        
        return True
        
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🌟 NLLB MODEL TESTING")
    print("=" * 50)
    
    # Test 1: Direct model loading
    success1 = test_model_loading()
    
    # Test 2: Pipeline approach
    success2 = test_pipeline()
    
    # Summary
    print(f"\n📊 TEST SUMMARY")
    print("=" * 30)
    print(f"Direct loading: {'✅ PASS' if success1 else '❌ FAIL'}")
    print(f"Pipeline approach: {'✅ PASS' if success2 else '❌ FAIL'}")
    
    if success1 or success2:
        print(f"\n🎉 At least one method works! The system should be functional.")
    else:
        print(f"\n❌ Both methods failed. Check your setup:")
        print("1. Install requirements: pip install -r requirements.txt")
        print("2. Check CUDA setup if using GPU")
        print("3. Ensure sufficient memory (8GB+ recommended)")
