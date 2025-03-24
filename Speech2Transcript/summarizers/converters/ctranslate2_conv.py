import os
import ctranslate2
from huggingface_hub import snapshot_download
from transformers import AutoModel, AutoTokenizer
from ctranslate2.converters import TransformersConverter

class CTranslate2Converter:

    def _load_model_with_ctranslate2(self, model_name, model_type):
            """Load a model with CTranslate2 acceleration"""
            model_path = snapshot_download(model_name, cache_dir=self.cache_dir)
            ct2_model_path = f"{model_path}_ct2_{model_type}"
            
            setattr(self, f"{model_type}_tokenizer", AutoTokenizer.from_pretrained(model_name, cache_dir=self.cache_dir))
            
            try:
                if not os.path.exists(ct2_model_path):
                    print(f"Converting {model_name} to CTranslate2 format")
                    try:
                        converter = TransformersConverter(model_path)
                        converter.convert(ct2_model_path, quantization=self.compute_type)
                    except:
                        print(f"Using subprocess for {model_name} conversion")
                        import subprocess
                        subprocess.run([
                            "ct2-transformers-converter",
                            "--model", model_path,
                            "--output_dir", ct2_model_path,
                            "--quantization", self.compute_type
                        ])
                
                ct2_model = ctranslate2.Encoder(
                    ct2_model_path, 
                    device=self.ct2_device, 
                    compute_type=self.compute_type
                )
                setattr(self, f"{model_type}_ct2", ct2_model)
                setattr(self, f"{model_type}_using_ct2", True)
                print(f"Loaded {model_name} using CTranslate2")
                
            except Exception as e:
                print(f"Error loading with CTranslate2: {e}")
                print(f"Falling back to PyTorch for {model_name}")
                setattr(self, model_type, AutoModel.from_pretrained(model_name, cache_dir=self.cache_dir).to(self.device))
                setattr(self, f"{model_type}_using_ct2", False)