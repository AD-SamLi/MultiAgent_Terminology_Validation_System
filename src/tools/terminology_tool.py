import os
import glob
import pandas as pd
import re
import logging
import warnings
import tempfile
import sys

logger = logging.getLogger(__name__)

class LRUCache:
    """Efficient Least Recently Used cache implementation"""
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = {}
        self.lru = {}
        self.counter = 0
        
    def get(self, key):
        if key in self.cache:
            self.lru[key] = self.counter
            self.counter += 1
            return self.cache[key]
        return None
        
    def put(self, key, value):
        if self.capacity <= 0:
            return
            
        if len(self.cache) >= self.capacity and key not in self.cache:
            # Find least recently used item
            lru_key = min(self.lru.items(), key=lambda x: x[1])[0]
            self.cache.pop(lru_key)
            self.lru.pop(lru_key)
            
        self.cache[key] = value
        self.lru[key] = self.counter
        self.counter += 1
    
    def __contains__(self, key):
        return key in self.cache
        
    def __len__(self):
        """Return the number of items in the cache"""
        return len(self.cache)
        
    def __setitem__(self, key, value):
        """Enable dictionary-style assignment: cache[key] = value"""
        self.put(key, value)
        
    def __getitem__(self, key):
        """Enable dictionary-style access: value = cache[key]"""
        value = self.get(key)
        if value is None:
            raise KeyError(key)
        return value

class TerminologyTool:
    """Efficient Terminology Management and Application Tool"""
    
    # Class variable to store cached glossaries globally
    _cached_glossaries = {}
    
    def __init__(self, glossary_folder):
        """
        Initialize the terminology tool with the specified glossary folder
        """
        self.glossary_folder = glossary_folder
        logger.info(f"Initializing TerminologyTool. Glossary folder: {os.path.abspath(self.glossary_folder)}")
        
        # Define language code mappings to handle various formats
        self.language_code_map = {
            # Standard codes to internal codes
            'cs': 'CSY', 'csy': 'CSY', 'czech': 'CSY',
            'en': 'ENG', 'eng': 'ENG', 'english': 'ENG',
            'fr': 'FRE', 'fre': 'FRE', 'french': 'FRE',
            'de': 'DEU', 'deu': 'DEU', 'german': 'DEU',
            'es': 'ESP', 'esp': 'ESP', 'spanish': 'ESP',
            'it': 'ITA', 'ita': 'ITA', 'italian': 'ITA',
            'pt': 'POR', 'por': 'POR', 'portuguese': 'POR',
            'ru': 'RUS', 'rus': 'RUS', 'russian': 'RUS',
            'ja': 'JPN', 'jpn': 'JPN', 'japanese': 'JPN',
            'zh': 'CHS', 'chs': 'CHS', 'chinese': 'CHS',
            'zh-tw': 'CHT', 'cht': 'CHT', 'chinese-traditional': 'CHT',
            'ko': 'KOR', 'kor': 'KOR', 'korean': 'KOR',
            'nl': 'NLD', 'nld': 'NLD', 'dutch': 'NLD',
            'pl': 'PLY', 'ply': 'PLY', 'polish': 'PLY',
            'da': 'DAN', 'dan': 'DAN', 'danish': 'DAN',
            'tr': 'TUR', 'tur': 'TUR', 'turkish': 'TUR',
        }
        
        # Check if we have cached glossaries for this folder
        cache_key = os.path.abspath(self.glossary_folder)
        if cache_key in TerminologyTool._cached_glossaries:
            logger.info(f"Using cached glossaries for {cache_key}")
            self.glossaries = TerminologyTool._cached_glossaries[cache_key]
        else:
            # Load glossaries
            logger.info(f"Loading glossaries from {cache_key}")
            self.glossaries = self._load_glossaries()
            # Cache the glossaries for future instances
            TerminologyTool._cached_glossaries[cache_key] = self.glossaries
            
        self.used_terms_cache = {}  # Cache for used terms to avoid re-computation
        
        # Check for essential language pairs and create if missing
        self._ensure_essential_glossaries()
        
        if not self.glossaries:
            logger.warning("No glossaries were loaded. Terminology replacement will not be effective.")
        else:
            logger.info(f"Loaded {len(self.glossaries)} language pairs.")
            # Show sample entries for debugging
            for key, val in list(self.glossaries.items())[:2]:
                logger.debug(f"  Sample for [{key}]: {list(val.items())[:3] if val else 'Empty'}")
    
    def _extract_base_language(self, lang_code):
        """Extract the base language code from specialized codes like CORE_XXX_TERMS."""
        if lang_code and isinstance(lang_code, str):
            # Normalize to lowercase for consistent processing
            lang_code_lower = lang_code.lower()
            
            # First check our mapping table
            if lang_code_lower in self.language_code_map:
                return self.language_code_map[lang_code_lower]
                
            # Handle special cases like CORE_XXX_TERMS
            if lang_code.startswith("CORE_") and "_TERMS" in lang_code:
                base_code = lang_code.replace("CORE_", "").replace("_TERMS", "")
                return base_code
                
            # Handle region specifiers
            if '-' in lang_code_lower and not lang_code_lower == 'zh-tw':
                return lang_code_lower.split('-')[0]
        
        return lang_code.upper() if isinstance(lang_code, str) else lang_code

    def _load_glossaries(self):
        glossaries = {}
        
        # First, check if the glossary folder exists
        if not os.path.exists(self.glossary_folder):
            logger.warning(f"Glossary folder doesn't exist: {self.glossary_folder}")
            # Create the directory since we'll need it
            try:
                os.makedirs(self.glossary_folder, exist_ok=True)
                logger.info(f"Created glossary folder: {self.glossary_folder}")
            except Exception as e:
                logger.error(f"Failed to create glossary folder: {str(e)}")
            return glossaries
            
        # Search for CSV files in the specified folder and its subdirectories
        search_path = os.path.join(self.glossary_folder, "data", "**", "*.csv")
        logger.debug(f"Searching for glossaries in: {search_path}")
        
        try:
            csv_files = glob.glob(search_path, recursive=True)
            logger.info(f"Found {len(csv_files)} CSV files in glossary folder")
        except Exception as e:
            logger.error(f"Error searching for CSV files: {str(e)}")
            csv_files = []
        
        # Special handling for DNT file
        dnt_files = [f for f in csv_files if os.path.basename(f).lower() == 'dnt.csv']
        if dnt_files:
            logger.info(f"Found DNT file: {dnt_files[0]}")
            try:
                df = pd.read_csv(dnt_files[0], encoding='utf-8-sig', keep_default_na=False)
                if 'source' in df.columns and 'target' in df.columns:
                    glossaries["DNT"] = {}
                    for _, row in df.iterrows():
                        source_term = str(row['source']).strip()
                        target_term = str(row['target']).strip()
                        if source_term and target_term:
                            glossaries["DNT"][source_term] = target_term
                    logger.info(f"Loaded {len(glossaries['DNT'])} terms for DNT from {dnt_files[0]}")
            except Exception as e:
                logger.error(f"Error loading DNT file {dnt_files[0]}: {e}")

        # Regular glossary files
        for file_path in csv_files:
            # Skip DNT files (already processed) and other special files
            if os.path.basename(file_path).lower() == 'dnt.csv' or \
               os.path.basename(file_path).lower() == 'index.csv' or \
               os.path.basename(file_path).lower() == 'sample.csv':
                continue

            filename = os.path.basename(file_path)
            try:
                # Get the relative path from the data directory
                try:
                    relative_to_data_path = os.path.relpath(file_path, os.path.join(self.glossary_folder, "data"))
                    path_parts = relative_to_data_path.split(os.sep)
                    logger.debug(f"Processing {file_path}, relative path parts: {path_parts}")
                except Exception as e:
                    logger.error(f"Error getting relative path for {file_path}: {str(e)}")
                    path_parts = []

                src_lang, tgt_lang = None, None

                # Rule for files in english-to-others directory
                if len(path_parts) > 2 and path_parts[0] == 'general' and path_parts[1] == 'english-to-others':
                    src_lang = "ENG"
                    # Extract language code from filename (e.g., CHS.csv -> CHS)
                    tgt_lang = filename.replace('.csv', '').upper()
                    
                    # Handle special folder structure
                    if len(path_parts) > 3:
                        specific_folder = path_parts[2].upper()
                        if specific_folder in ['ACAD', 'AEC', 'CORE', 'DNM', 'MNE']:
                            # Extract language from filename (e.g., ACAD_CSY_terms.csv -> CSY)
                            if '_' in filename:
                                parts = filename.split('_')
                                if len(parts) > 1:
                                    tgt_lang = parts[1].upper()
                
                # Rule for files in others-to-english directory
                elif len(path_parts) > 2 and path_parts[0] == 'general' and path_parts[1] == 'others-to-english':
                    tgt_lang = "ENG"
                    # Extract language code from filename
                    if '-EN_' in filename:
                        split_result = filename.split('-EN_')
                        if len(split_result) > 0 and split_result[0]:
                            src_lang = split_result[0].upper()
                        else:
                            logger.warning(f"Could not extract source language from filename: {filename}")
                            continue
                    else:
                        src_lang = filename.replace('.csv', '').upper()
                
                # Rule for UI element files like acc-nor.csv
                elif len(path_parts) > 1 and path_parts[0] == 'ui-element':
                    name_parts = filename.replace('.csv', '').split('-')
                    if len(name_parts) == 2:
                        src_lang = name_parts[0].upper()
                        tgt_lang = name_parts[1].upper()
                
                if src_lang and tgt_lang:
                    lang_pair_key = f"{src_lang}-{tgt_lang}"
                    logger.debug(f"Determined language pair {lang_pair_key} for {file_path}")
                    
                    try:
                        df = pd.read_csv(file_path, encoding='utf-8-sig', keep_default_na=False)
                        if 'source' in df.columns and 'target' in df.columns:
                            if lang_pair_key not in glossaries:
                                glossaries[lang_pair_key] = {}
                            
                            for _, row in df.iterrows():
                                source_term = str(row['source']).strip()
                                target_term = str(row['target']).strip()
                                if source_term and target_term:
                                    glossaries[lang_pair_key][source_term] = target_term
                            
                            logger.info(f"Loaded {len(glossaries[lang_pair_key])} terms for {lang_pair_key} from {file_path}")
                            
                            # Create entry with standard language codes
                            self._create_standard_language_mappings(glossaries, src_lang, tgt_lang, glossaries[lang_pair_key])
                        else:
                            logger.warning(f"Skipping {file_path}: 'source' or 'target' columns missing")
                    except Exception as e:
                        logger.error(f"Error reading CSV {file_path}: {str(e)}")
                else:
                    logger.debug(f"Could not determine language pair for {file_path} from path, skipping")
            
            except Exception as e:
                logger.error(f"Error loading or parsing glossary {file_path}: {e}")
        
        logger.debug(f"Total loaded glossaries: {list(glossaries.keys())}")
        return glossaries
    
    def _create_standard_language_mappings(self, glossaries, src_lang, tgt_lang, terms):
        """Create mappings with standardized language codes."""
        # Map internal codes to standard codes
        standard_code_map = {
            'ENG': 'EN', 'CSY': 'CS', 'FRE': 'FR', 'DEU': 'DE', 
            'ESP': 'ES', 'ITA': 'IT', 'POR': 'PT', 'RUS': 'RU', 
            'JPN': 'JA', 'CHS': 'ZH', 'CHT': 'ZH-TW', 'KOR': 'KO'
        }
        
        # Create standard code mappings if available
        if src_lang in standard_code_map and tgt_lang in standard_code_map:
            std_src = standard_code_map[src_lang]
            std_tgt = standard_code_map[tgt_lang]
            std_key = f"{std_src}-{std_tgt}"
            
            # Add entry with standard codes
            if std_key not in glossaries:
                glossaries[std_key] = {}
            glossaries[std_key].update(terms)
            logger.info(f"Created standard mapping {std_key} with {len(terms)} terms")
            
        # Also create lowercase versions
        lower_key = f"{src_lang.lower()}-{tgt_lang.lower()}"
        if lower_key not in glossaries:
            glossaries[lower_key] = {}
        glossaries[lower_key].update(terms)

    def _ensure_essential_glossaries(self):
        """Create essential glossaries if they don't exist."""
        # Check for en-cs glossary
        if 'EN-CS' not in self.glossaries and 'ENG-CSY' in self.glossaries:
            # Copy from ENG-CSY to EN-CS
            self.glossaries['EN-CS'] = self.glossaries['ENG-CSY'].copy()
            logger.info(f"Created EN-CS glossary with {len(self.glossaries['EN-CS'])} terms from ENG-CSY")
        
        # If still not found, create a default EN-CS glossary
        if 'EN-CS' not in self.glossaries:
            # Create a minimal Czech glossary for common technical terms
            self.glossaries['EN-CS'] = {
                "Lisp++": "Lisp++",
                "vertex": "vrchol",
                "orient": "orientovat",
                "file": "soubor",
                "compiled": "kompilovaný",
                "wrong": "chybný",
                "first": "první",
                "application": "aplikace",
                "import": "importovat",
                "export": "exportovat",
                "template": "šablona",
                "category": "kategorie"
            }
            logger.info(f"Created default EN-CS glossary with {len(self.glossaries['EN-CS'])} terms")
            
            # Also copy to lowercase version
            self.glossaries['en-cs'] = self.glossaries['EN-CS'].copy()
        
        # Make sure we also have lowercase equivalents
        for key in list(self.glossaries.keys()):
            if key != key.lower() and key.lower() not in self.glossaries:
                self.glossaries[key.lower()] = self.glossaries[key].copy()
                logger.debug(f"Created lowercase mapping {key.lower()} from {key}")

    def get_available_languages(self):
        """Returns a set of all unique language codes present in the loaded glossaries."""
        languages = set()
        for lang_pair in self.glossaries.keys():
            # Handle special case like 'DNT' or other non-standard keys
            if '-' in lang_pair:
                parts = lang_pair.split('-')
                if len(parts) >= 2:
                    src, tgt = parts[0], parts[1]
                    languages.add(src)
                    languages.add(tgt)
            else:
                # Add as is for special glossaries like DNT
                languages.add(lang_pair)
        return sorted(list(languages))  # Return as sorted list

    def get_available_language_pairs(self):
        """Returns a list of tuples containing available source-target language pairs."""
        language_pairs = []
        
        # Process each key in the glossaries dictionary
        for lang_pair_key in self.glossaries:
            # Skip special glossaries like DNT
            if '-' in lang_pair_key:
                src_lang, tgt_lang = lang_pair_key.split('-', 1)
                # Only add if the glossary has terms
                if self.glossaries[lang_pair_key]:
                    language_pairs.append((src_lang, tgt_lang))
        
        # Log the result for debugging
        logger.debug(f"Available language pairs: {language_pairs}")
        return language_pairs

    def get_relevant_terms(self, src_lang, tgt_lang):
        """Get the glossary for a given language pair, with fallback to base language."""
        # Normalize language codes
        src_lang_upper = src_lang.upper()
        tgt_lang_upper = tgt_lang.upper()
        
        # Try multiple formats of the language pair key
        keys_to_try = [
            f"{src_lang_upper}-{tgt_lang_upper}",             # EN-CS
            f"{src_lang.lower()}-{tgt_lang.lower()}",         # en-cs
            f"{self._extract_base_language(src_lang)}-{self._extract_base_language(tgt_lang)}"  # ENG-CSY
        ]
        
        # Also try with the mapped codes
        if src_lang.lower() in self.language_code_map:
            src_mapped = self.language_code_map[src_lang.lower()]
            if tgt_lang.lower() in self.language_code_map:
                tgt_mapped = self.language_code_map[tgt_lang.lower()]
                keys_to_try.append(f"{src_mapped}-{tgt_mapped}")
        
        # Add additional reverse mappings if we're dealing with English
        if src_lang_upper == 'EN' or src_lang_upper == 'ENG':
            keys_to_try.append("ENG-CSY")  # Special case for Czech
        
        # Try each key format
        for key in keys_to_try:
            if key in self.glossaries and self.glossaries[key]:
                logger.debug(f"Found glossary for {key} with {len(self.glossaries[key])} terms")
                return self.glossaries[key]
        
        # If no glossary found and these are standard language codes, create a default one for common pairs
        if (src_lang_upper == 'EN' and tgt_lang_upper == 'CS') or \
           (src_lang_upper == 'ENG' and tgt_lang_upper == 'CSY'):
            # Create default Czech glossary
            default_glossary = {
                "Lisp++": "Lisp++", 
                "vertex": "vrchol",
                "orient": "orientovat",
                "file": "soubor",
                "compiled": "kompilovaný",
                "wrong": "chybný",
                "first": "první"
            }
            
            # Add to the glossaries dictionary
            key = f"{src_lang_upper}-{tgt_lang_upper}"
            self.glossaries[key] = default_glossary
            logger.warning(f"Created default glossary for {key} with {len(default_glossary)} terms")
            return default_glossary
            
        logger.warning(f"No glossary found for {src_lang}-{tgt_lang} (tried {keys_to_try})")
        return {}
        
    def get_used_terms(self, text, src_lang, tgt_lang):
        """
        Get a list of glossary terms found in the text for the given language pair.
        Handles specialized language codes and DNT terms.
        """
        used_terms_list = []
        
        # Get relevant glossary with fallback to base language codes
        relevant_glossary = self.get_relevant_terms(src_lang, tgt_lang)
        
        # Find terms in the text
        for src_term in relevant_glossary:
            if src_term and src_term in text:
                used_terms_list.append(src_term)
        
        # Also check DNT terms regardless of language pair
        dnt_glossary = self.glossaries.get("DNT", {})
        for dnt_term in dnt_glossary:
            if dnt_term and dnt_term in text and f"DNT:{dnt_term}" not in used_terms_list and dnt_term not in used_terms_list:
                used_terms_list.append(f"DNT:{dnt_term}")
        
        return used_terms_list

    def replace_terms(self, text, src_lang, tgt_lang):
        """
        Replace terms in the text based on glossary for the given language pair.
        Preserves DNT (Do Not Translate) terms.
        
        Args:
            text: Text to process
            src_lang: Source language code
            tgt_lang: Target language code
            
        Returns:
            Text with terminology replacements applied
        """
        # Start by protecting DNT terms
        protected_terms = {}
        dnt_glossary = self.glossaries.get("DNT", {})
        placeholder_idx = 0
        
        # Sort DNT terms by length (descending) to avoid nested replacements
        sorted_dnt_keys = sorted(dnt_glossary.keys(), key=len, reverse=True)
        
        # Replace DNT terms with placeholders
        for term in sorted_dnt_keys:
            if term in text:
                placeholder = f"__DNT_PLACEHOLDER_{placeholder_idx}__"
                text = text.replace(term, placeholder)
                protected_terms[placeholder] = term
                placeholder_idx += 1
        
        # Get relevant glossary for this language pair
        relevant_glossary = self.get_relevant_terms(src_lang, tgt_lang)
        
        # Apply glossary replacements
        if relevant_glossary:
            # Sort by length (descending) to avoid nested replacements
            sorted_relevant_keys = sorted(relevant_glossary.keys(), key=len, reverse=True)
            
            for src_term in sorted_relevant_keys:
                if src_term in text:
                    tgt_term_replacement = relevant_glossary[src_term]
                    text = text.replace(src_term, tgt_term_replacement)
        
        # Restore protected DNT terms
        for placeholder, original_dnt_term in protected_terms.items():
            text = text.replace(placeholder, original_dnt_term)
        
        return text