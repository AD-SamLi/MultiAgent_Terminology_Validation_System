#!/usr/bin/env python3
"""
NLLB-200 Translation Tool with GPU Support
Uses Facebook's NLLB-200-3.3B model for multilingual translation
Supports all 200 languages from the NLLB-200 model
"""

import os
import json
import time
import logging
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import warnings

# Set CUDA memory allocation configuration for better memory management
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Suppress transformers warnings
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("transformers").setLevel(logging.ERROR)

@dataclass
class TranslationResult:
    """Container for translation results"""
    original_text: str
    translated_text: str
    source_lang: str
    target_lang: str
    is_same: bool
    confidence_score: float = 0.0
    error: Optional[str] = None

class NLLBTranslationTool:
    """
    NLLB-200 Translation Tool with GPU acceleration and CPU+GPU hybrid loading
    
    Supports all 200 languages from Facebook's NLLB-200 model family:
    - facebook/nllb-200-1.3B (smaller, faster, less memory)
    - facebook/nllb-200-3.3B (larger, better quality, more memory)
    
    Uses accelerate for intelligent CPU+GPU memory management.
    """
    
    # Available NLLB model sizes
    MODEL_SIZES = {
        "small": "facebook/nllb-200-1.3B",      # ~2.6GB, faster inference
        "medium": "facebook/nllb-200-3.3B",     # ~6.7GB, better quality
        "1.3B": "facebook/nllb-200-1.3B",
        "3.3B": "facebook/nllb-200-3.3B"
    }
    
    # NLLB-200 language codes (200 languages)
    NLLB_LANGUAGES = {
        # Major languages
        "ace_Arab": "Acehnese (Arabic script)",
        "ace_Latn": "Acehnese (Latin script)", 
        "acm_Arab": "Mesopotamian Arabic",
        "acq_Arab": "Ta'izzi-Adeni Arabic",
        "aeb_Arab": "Tunisian Arabic",
        "afr_Latn": "Afrikaans",
        "ajp_Arab": "South Levantine Arabic",
        "aka_Latn": "Akan",
        "amh_Ethi": "Amharic",
        "apc_Arab": "North Levantine Arabic",
        "arb_Arab": "Modern Standard Arabic",
        "ars_Arab": "Najdi Arabic",
        "ary_Arab": "Moroccan Arabic",
        "arz_Arab": "Egyptian Arabic",
        "asm_Beng": "Assamese",
        "ast_Latn": "Asturian",
        "awa_Deva": "Awadhi",
        "ayr_Latn": "Central Aymara",
        "azb_Arab": "South Azerbaijani",
        "azj_Latn": "North Azerbaijani",
        "bak_Cyrl": "Bashkir",
        "bam_Latn": "Bambara",
        "ban_Latn": "Balinese",
        "bel_Cyrl": "Belarusian",
        "bem_Latn": "Bemba",
        "ben_Beng": "Bengali",
        "bho_Deva": "Bhojpuri",
        "bjn_Arab": "Banjar (Arabic script)",
        "bjn_Latn": "Banjar (Latin script)",
        "bod_Tibt": "Standard Tibetan",
        "bos_Latn": "Bosnian",
        "bug_Latn": "Buginese",
        "bul_Cyrl": "Bulgarian",
        "cat_Latn": "Catalan",
        "ceb_Latn": "Cebuano",
        "ces_Latn": "Czech",
        "cjk_Latn": "Chokwe",
        "ckb_Arab": "Central Kurdish",
        "crh_Latn": "Crimean Tatar",
        "cym_Latn": "Welsh",
        "dan_Latn": "Danish",
        "deu_Latn": "German",
        "dik_Latn": "Southwestern Dinka",
        "dyu_Latn": "Dyula",
        "dzo_Tibt": "Dzongkha",
        "ell_Grek": "Greek",
        "eng_Latn": "English",
        "epo_Latn": "Esperanto",
        "est_Latn": "Estonian",
        "eus_Latn": "Basque",
        "ewe_Latn": "Ewe",
        "fao_Latn": "Faroese",
        "pes_Arab": "Western Persian",
        "fij_Latn": "Fijian",
        "fin_Latn": "Finnish",
        "fon_Latn": "Fon",
        "fra_Latn": "French",
        "fur_Latn": "Friulian",
        "fuv_Latn": "Nigerian Fulfulde",
        "gla_Latn": "Scottish Gaelic",
        "gle_Latn": "Irish",
        "glg_Latn": "Galician",
        "grn_Latn": "Guarani",
        "guj_Gujr": "Gujarati",
        "hat_Latn": "Haitian Creole",
        "hau_Latn": "Hausa",
        "heb_Hebr": "Hebrew",
        "hin_Deva": "Hindi",
        "hne_Deva": "Chhattisgarhi",
        "hrv_Latn": "Croatian",
        "hun_Latn": "Hungarian",
        "hye_Armn": "Armenian",
        "ibo_Latn": "Igbo",
        "ilo_Latn": "Ilocano",
        "ind_Latn": "Indonesian",
        "isl_Latn": "Icelandic",
        "ita_Latn": "Italian",
        "jav_Latn": "Javanese",
        "jpn_Jpan": "Japanese",
        "kab_Latn": "Kabyle",
        "kac_Latn": "Jingpho",
        "kam_Latn": "Kamba",
        "kan_Knda": "Kannada",
        "kas_Arab": "Kashmiri (Arabic script)",
        "kas_Deva": "Kashmiri (Devanagari script)",
        "kat_Geor": "Georgian",
        "knc_Arab": "Central Kanuri (Arabic script)",
        "knc_Latn": "Central Kanuri (Latin script)",
        "kaz_Cyrl": "Kazakh",
        "kbp_Latn": "Kabiyè",
        "kea_Latn": "Kabuverdianu",
        "khm_Khmr": "Khmer",
        "kik_Latn": "Kikuyu",
        "kin_Latn": "Kinyarwanda",
        "kir_Cyrl": "Kyrgyz",
        "kmb_Latn": "Kimbundu",
        "kon_Latn": "Kikongo",
        "kor_Hang": "Korean",
        "kmr_Latn": "Northern Kurdish",
        "lao_Laoo": "Lao",
        "lvs_Latn": "Standard Latvian",
        "lij_Latn": "Ligurian",
        "lim_Latn": "Limburgish",
        "lin_Latn": "Lingala",
        "lit_Latn": "Lithuanian",
        "lmo_Latn": "Lombard",
        "ltg_Latn": "Latgalian",
        "ltz_Latn": "Luxembourgish",
        "lua_Latn": "Luba-Kasai",
        "lug_Latn": "Ganda",
        "luo_Latn": "Luo",
        "lus_Latn": "Mizo",
        "mag_Deva": "Magahi",
        "mai_Deva": "Maithili",
        "mal_Mlym": "Malayalam",
        "mar_Deva": "Marathi",
        "min_Arab": "Minangkabau (Arabic script)",
        "min_Latn": "Minangkabau (Latin script)",
        "mkd_Cyrl": "Macedonian",
        "plt_Latn": "Plateau Malagasy",
        "mlt_Latn": "Maltese",
        "mni_Beng": "Meitei (Bengali script)",
        "khk_Cyrl": "Halh Mongolian",
        "mos_Latn": "Mossi",
        "mri_Latn": "Maori",
        "zsm_Latn": "Standard Malay",
        "mya_Mymr": "Burmese",
        "nld_Latn": "Dutch",
        "nno_Latn": "Norwegian Nynorsk",
        "nob_Latn": "Norwegian Bokmål",
        "npi_Deva": "Nepali",
        "nso_Latn": "Northern Sotho",
        "nus_Latn": "Nuer",
        "nya_Latn": "Nyanja",
        "oci_Latn": "Occitan",
        "gaz_Latn": "West Central Oromo",
        "ory_Orya": "Odia",
        "pag_Latn": "Pangasinan",
        "pan_Guru": "Eastern Panjabi",
        "pap_Latn": "Papiamento",
        "pol_Latn": "Polish",
        "por_Latn": "Portuguese",
        "prs_Arab": "Dari",
        "pbt_Arab": "Southern Pashto",
        "quy_Latn": "Ayacucho Quechua",
        "ron_Latn": "Romanian",
        "run_Latn": "Rundi",
        "rus_Cyrl": "Russian",
        "sag_Latn": "Sango",
        "san_Deva": "Sanskrit",
        "sat_Olck": "Santali",
        "scn_Latn": "Sicilian",
        "shn_Mymr": "Shan",
        "sin_Sinh": "Sinhala",
        "slk_Latn": "Slovak",
        "slv_Latn": "Slovenian",
        "smo_Latn": "Samoan",
        "sna_Latn": "Shona",
        "snd_Arab": "Sindhi",
        "som_Latn": "Somali",
        "sot_Latn": "Southern Sotho",
        "spa_Latn": "Spanish",
        "als_Latn": "Tosk Albanian",
        "srd_Latn": "Sardinian",
        "srp_Cyrl": "Serbian",
        "ssw_Latn": "Swati",
        "sun_Latn": "Sundanese",
        "swe_Latn": "Swedish",
        "swh_Latn": "Swahili",
        "szl_Latn": "Silesian",
        "tam_Taml": "Tamil",
        "tat_Cyrl": "Tatar",
        "tel_Telu": "Telugu",
        "tgk_Cyrl": "Tajik",
        "tgl_Latn": "Tagalog",
        "tha_Thai": "Thai",
        "tir_Ethi": "Tigrinya",
        "taq_Latn": "Tamasheq (Latin script)",
        "taq_Tfng": "Tamasheq (Tifinagh script)",
        "tpi_Latn": "Tok Pisin",
        "tsn_Latn": "Tswana",
        "tso_Latn": "Tsonga",
        "tuk_Latn": "Turkmen",
        "tum_Latn": "Tumbuka",
        "tur_Latn": "Turkish",
        "twi_Latn": "Twi",
        "tzm_Tfng": "Central Atlas Tamazight",
        "uig_Arab": "Uyghur",
        "ukr_Cyrl": "Ukrainian",
        "umb_Latn": "Umbundu",
        "urd_Arab": "Urdu",
        "uzn_Latn": "Northern Uzbek",
        "vec_Latn": "Venetian",
        "vie_Latn": "Vietnamese",
        "war_Latn": "Waray",
        "wol_Latn": "Wolof",
        "xho_Latn": "Xhosa",
        "ydd_Hebr": "Eastern Yiddish",
        "yor_Latn": "Yoruba",
        "yue_Hant": "Yue Chinese",
        "zho_Hans": "Chinese (Simplified)",
        "zho_Hant": "Chinese (Traditional)",
        "zul_Latn": "Zulu"
    }
    
    def __init__(self, model_name: str = "facebook/nllb-200-1.3B", device: str = "auto", batch_size: int = 8, offload_dir: str = None):
        """
        Initialize the NLLB translation tool
        
        Args:
            model_name: HuggingFace model name or size key ('small', 'medium', '1.3B', '3.3B')
            device: Device to use ('auto', 'cuda', 'cpu')
            batch_size: Batch size for translation
            offload_dir: Directory for model offloading (default: "./model_offload")
        """
        # Resolve model name if it's a size key
        if model_name in self.MODEL_SIZES:
            self.model_name = self.MODEL_SIZES[model_name]
            print(f"📋 Using {model_name} model: {self.model_name}")
        else:
            self.model_name = model_name
        
        self.batch_size = batch_size
        self.device = self._setup_device(device)
        self.offload_dir = offload_dir or "./model_offload"
        
        print(f"🚀 Initializing NLLB-200 Translation Tool")
        print(f"🤖 Model: {self.model_name}")
        print(f"📱 Device strategy: {self.device}")
        print(f"🔢 Batch size: {batch_size}")
        
        # Initialize model and tokenizer
        self.tokenizer = None
        self.model = None
        self.translator = None
        
        self._load_model()
        
        print(f"✅ NLLB-200 tool ready with {len(self.NLLB_LANGUAGES)} languages")
    
    def _setup_device(self, device: str) -> str:
        """Setup computation device"""
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
                print(f"🎮 CUDA available: {torch.cuda.get_device_name()}")
                print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            else:
                device = "cpu"
                print("💻 Using CPU (CUDA not available)")
        elif device.startswith("cuda:"):
            # Normalize cuda:N to cuda for our optimizations to work
            print(f"🎮 Specific CUDA device requested: {device}")
            device = "cuda"  # Normalize to trigger GPU optimizations
        
        return device
    
    def _load_model(self):
        """Load NLLB model and tokenizer with CPU+GPU hybrid setup"""
        
        print(f"📥 Loading tokenizer from {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        print("✅ Tokenizer loaded successfully")
        
        print(f"🧠 Loading model with accelerate device mapping...")
        
        # Create offload directory only if offload strategy requires it
        import os
        # Only create directory if we're actually using disk offloading
        # Check if offload_dir is meaningful (not just a default path)
        if self.offload_dir and self.offload_dir != "./model_offload" and "model_offload" not in self.offload_dir:
            os.makedirs(self.offload_dir, exist_ok=True)
        # Skip creating model_offload directories since they're not used with strategy="none"
        
        # Configure loading parameters based on device strategy
        if self.device == "cuda" and torch.cuda.is_available():
            print("🎮 Using GPU + CPU hybrid loading with optimized memory management...")
            
            # Calculate optimal max_memory for Tesla T4 with 3 workers
            if torch.cuda.device_count() > 0:
                gpu_props = torch.cuda.get_device_properties(0)
                gpu_memory_gb = gpu_props.total_memory / (1024**3)
                gpu_name = gpu_props.name
                
                # Set max_memory based on GPU type and worker count
                if "Tesla T4" in gpu_name:
                    # Tesla T4 optimization: Use 95% of memory instead of default 90%
                    max_memory_per_device = f"{gpu_memory_gb * 0.95:.1f}GB"
                    print(f"   🎯 Tesla T4 detected: Using {max_memory_per_device} max memory (95% of {gpu_memory_gb:.1f}GB)")
                else:
                    # Other GPUs: Use 92% of memory
                    max_memory_per_device = f"{gpu_memory_gb * 0.92:.1f}GB"
                    print(f"   🎯 {gpu_name}: Using {max_memory_per_device} max memory (92% of {gpu_memory_gb:.1f}GB)")
                
                max_memory_config = {0: max_memory_per_device}
            else:
                max_memory_config = None
            
            load_kwargs = {
                "torch_dtype": torch.float16,  # Half precision for memory efficiency
                "device_map": "auto",          # Auto-distribute across available devices
                "max_memory": max_memory_config,  # Optimize memory usage per device
                "offload_folder": self.offload_dir, # Offload to disk if needed
                "low_cpu_mem_usage": True,     # Minimize CPU memory usage
                "offload_state_dict": True     # Offload state dict to save memory
            }
        else:
            print("💻 Using CPU with memory optimization...")
            load_kwargs = {
                "device_map": "auto",          # Let accelerate handle placement
                "low_cpu_mem_usage": True,     # Minimize CPU memory usage
                "offload_folder": self.offload_dir, # Use disk offloading if needed
                "torch_dtype": torch.float32   # Full precision for CPU
            }
        
        # Load the model
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name,
            **load_kwargs
        )
        
        print("✅ Model loaded with hybrid memory management")
        
        # Print memory info if CUDA is available
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            print(f"🎮 GPU Memory - Allocated: {allocated:.1f}GB, Reserved: {reserved:.1f}GB")
        
        # Create translation pipeline (let accelerate handle device placement)
        print("🔧 Creating translation pipeline...")
        self.translator = pipeline(
            "translation",
            model=self.model,
            tokenizer=self.tokenizer,
            batch_size=self.batch_size
        )
        
        print("✅ Translation pipeline ready")
    
    def get_available_languages(self) -> List[str]:
        """Get list of available language codes"""
        return list(self.NLLB_LANGUAGES.keys())
    
    def get_language_info(self, lang_code: str) -> Optional[str]:
        """Get language name from code"""
        return self.NLLB_LANGUAGES.get(lang_code)
    
    def is_valid_language(self, lang_code: str) -> bool:
        """Check if language code is valid"""
        return lang_code in self.NLLB_LANGUAGES
    
    def translate_text(self, text: str, src_lang: str, tgt_lang: str) -> TranslationResult:
        """
        Translate text from source to target language
        
        Args:
            text: Text to translate
            src_lang: Source language code (NLLB format)
            tgt_lang: Target language code (NLLB format)
            
        Returns:
            TranslationResult object
        """
        if not text.strip():
            return TranslationResult(
                original_text=text,
                translated_text=text,
                source_lang=src_lang,
                target_lang=tgt_lang,
                is_same=True,
                error="Empty text"
            )
        
        if not self.is_valid_language(src_lang):
            return TranslationResult(
                original_text=text,
                translated_text=text,
                source_lang=src_lang,
                target_lang=tgt_lang,
                is_same=True,
                error=f"Invalid source language: {src_lang}"
            )
        
        if not self.is_valid_language(tgt_lang):
            return TranslationResult(
                original_text=text,
                translated_text=text,
                source_lang=src_lang,
                target_lang=tgt_lang,
                is_same=True,
                error=f"Invalid target language: {tgt_lang}"
            )
        
        try:
            # Set source language
            self.tokenizer.src_lang = src_lang
            
            # Perform translation
            result = self.translator(
                text,
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                max_length=512
            )
            
            if isinstance(result, list) and len(result) > 0:
                translated_text = result[0]['translation_text']
            else:
                translated_text = str(result.get('translation_text', text))
            
            # Check if translation is the same as original
            is_same = text.strip().lower() == translated_text.strip().lower()
            
            return TranslationResult(
                original_text=text,
                translated_text=translated_text,
                source_lang=src_lang,
                target_lang=tgt_lang,
                is_same=is_same,
                confidence_score=1.0  # NLLB doesn't provide confidence scores
            )
            
        except Exception as e:
            return TranslationResult(
                original_text=text,
                translated_text=text,
                source_lang=src_lang,
                target_lang=tgt_lang,
                is_same=True,
                error=f"Translation error: {str(e)}"
            )
    
    def translate_batch(self, texts: List[str], src_lang: str, tgt_lang: str) -> List[TranslationResult]:
        """
        Translate multiple texts in batch
        
        Args:
            texts: List of texts to translate
            src_lang: Source language code
            tgt_lang: Target language code
            
        Returns:
            List of TranslationResult objects
        """
        if not texts:
            return []
        
        # Filter out empty texts
        text_indices = [(i, text) for i, text in enumerate(texts) if text.strip()]
        if not text_indices:
            return [TranslationResult(
                original_text=text,
                translated_text=text,
                source_lang=src_lang,
                target_lang=tgt_lang,
                is_same=True,
                error="Empty text"
            ) for text in texts]
        
        # Validate languages
        if not self.is_valid_language(src_lang) or not self.is_valid_language(tgt_lang):
            error_msg = f"Invalid language codes: {src_lang} -> {tgt_lang}"
            return [TranslationResult(
                original_text=text,
                translated_text=text,
                source_lang=src_lang,
                target_lang=tgt_lang,
                is_same=True,
                error=error_msg
            ) for text in texts]
        
        results = [None] * len(texts)
        
        try:
            # Set source language
            self.tokenizer.src_lang = src_lang
            
            # Extract just the texts for translation
            batch_texts = [text for _, text in text_indices]
            
            # Perform batch translation
            translations = self.translator(
                batch_texts,
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                max_length=512,
                batch_size=self.batch_size
            )
            
            # Process results
            for (original_idx, original_text), translation in zip(text_indices, translations):
                if isinstance(translation, dict):
                    translated_text = translation.get('translation_text', original_text)
                else:
                    translated_text = str(translation)
                
                is_same = original_text.strip().lower() == translated_text.strip().lower()
                
                results[original_idx] = TranslationResult(
                    original_text=original_text,
                    translated_text=translated_text,
                    source_lang=src_lang,
                    target_lang=tgt_lang,
                    is_same=is_same,
                    confidence_score=1.0
                )
            
            # Fill in empty text results
            for i, text in enumerate(texts):
                if results[i] is None:
                    results[i] = TranslationResult(
                        original_text=text,
                        translated_text=text,
                        source_lang=src_lang,
                        target_lang=tgt_lang,
                        is_same=True,
                        error="Empty text"
                    )
            
            return results
            
        except Exception as e:
            error_msg = f"Batch translation error: {str(e)}"
            return [TranslationResult(
                original_text=text,
                translated_text=text,
                source_lang=src_lang,
                target_lang=tgt_lang,
                is_same=True,
                error=error_msg
            ) for text in texts]
    
    def translate_to_all_languages(self, text: str, src_lang: str = "eng_Latn") -> Dict[str, TranslationResult]:
        """
        Translate text to all available languages
        
        Args:
            text: Text to translate
            src_lang: Source language code (default: English)
            
        Returns:
            Dictionary mapping target language codes to TranslationResult objects
        """
        results = {}
        target_languages = [lang for lang in self.NLLB_LANGUAGES.keys() if lang != src_lang]
        
        print(f"🌍 Translating to {len(target_languages)} languages...")
        
        # Process in batches to avoid memory issues
        batch_size = 10  # Smaller batches for multi-language translation
        
        for i in range(0, len(target_languages), batch_size):
            batch_langs = target_languages[i:i + batch_size]
            print(f"📦 Processing batch {i//batch_size + 1}/{(len(target_languages) + batch_size - 1)//batch_size}")
            
            for tgt_lang in batch_langs:
                try:
                    result = self.translate_text(text, src_lang, tgt_lang)
                    results[tgt_lang] = result
                    
                    if result.error:
                        print(f"⚠️  {tgt_lang}: {result.error}")
                    elif result.is_same:
                        print(f"🟡 {tgt_lang}: Same as original")
                    else:
                        print(f"✅ {tgt_lang}: Translated")
                        
                except Exception as e:
                    results[tgt_lang] = TranslationResult(
                        original_text=text,
                        translated_text=text,
                        source_lang=src_lang,
                        target_lang=tgt_lang,
                        is_same=True,
                        error=f"Translation failed: {str(e)}"
                    )
                    print(f"❌ {tgt_lang}: Failed - {str(e)}")
            
            # Small delay between batches to prevent overheating
            time.sleep(1)
        
        return results
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage"""
        if self.device == "cuda" and torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            return {
                "allocated_gb": allocated,
                "reserved_gb": reserved,
                "device": self.device
            }
        else:
            return {"device": self.device, "memory_tracking": "not_available"}


def test_nllb_tool():
    """Test the NLLB translation tool"""
    print("🧪 Testing NLLB Translation Tool")
    print("=" * 50)
    
    # Initialize tool with small model
    tool = NLLBTranslationTool(model_name="small", batch_size=4)
    
    # Test single translation
    print("\n1️⃣ Testing single translation:")
    result = tool.translate_text("Hello world", "eng_Latn", "spa_Latn")
    print(f"Original: {result.original_text}")
    print(f"Translation: {result.translated_text}")
    print(f"Same: {result.is_same}")
    print(f"Error: {result.error}")
    
    # Test batch translation
    print("\n2️⃣ Testing batch translation:")
    texts = ["Hello", "world", "computer", "software"]
    results = tool.translate_batch(texts, "eng_Latn", "fra_Latn")
    for i, result in enumerate(results):
        print(f"{texts[i]} -> {result.translated_text} (same: {result.is_same})")
    
    # Test memory usage
    print("\n3️⃣ Memory usage:")
    memory = tool.get_memory_usage()
    print(json.dumps(memory, indent=2))
    
    print("\n✅ Test completed!")


if __name__ == "__main__":
    test_nllb_tool()
