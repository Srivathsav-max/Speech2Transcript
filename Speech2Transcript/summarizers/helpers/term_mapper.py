import os
import json

class TerminologyMapper:
    """Maps extracted medical entities to standard terminologies like UMLS/SNOMED CT"""
    
    def __init__(self, terminology_path=None, cache_dir=None):
        self.cache_dir = cache_dir
        self.terminology_dict = {}
        
        # Load terminology mapping
        if terminology_path and os.path.exists(terminology_path):
            self._load_terminology(terminology_path)
        else:
            self._initialize_default_mappings()
    
    def _load_terminology(self, path):
        """Load terminology mapping from file"""
        with open(path, 'r') as f:
            self.terminology_dict = json.load(f)
    
    def _initialize_default_mappings(self):
        """Initialize default mappings for common medical terms"""
        # Basic mapping for common medical terms
        self.terminology_dict = {
            "hypertension": {
                "cui": "C0020538",
                "snomed": "38341003",
                "preferred": "Hypertension"
            },
            "diabetes": {
                "cui": "C0011849",
                "snomed": "73211009",
                "preferred": "Diabetes mellitus"
            },
            "myocardial infarction": {
                "cui": "C0027051",
                "snomed": "22298006",
                "preferred": "Myocardial infarction"
            },
            "high blood pressure": {
                "cui": "C0020538",
                "snomed": "38341003",
                "preferred": "Hypertension"
            },
            "high bp": {
                "cui": "C0020538",
                "snomed": "38341003",
                "preferred": "Hypertension"
            },
            "diabetes mellitus": {
                "cui": "C0011849",
                "snomed": "73211009",
                "preferred": "Diabetes mellitus"
            },
            "type 2 diabetes": {
                "cui": "C0011860",
                "snomed": "44054006",
                "preferred": "Type 2 diabetes mellitus"
            },
            "type 1 diabetes": {
                "cui": "C0011854",
                "snomed": "46635009",
                "preferred": "Type 1 diabetes mellitus"
            }
        }
    
    def map_entity(self, entity):
        """Map entity to standard terminology"""
        term_lower = entity.value.lower()
        
        # Check exact match
        if term_lower in self.terminology_dict:
            mapping = self.terminology_dict[term_lower]
            entity.metadata["cui"] = mapping["cui"]
            entity.metadata["snomed"] = mapping.get("snomed")
            entity.normalized_value = mapping["preferred"]
            return entity
        
        # Check partial matches
        best_match = None
        best_score = 0
        
        for standard_term, mapping in self.terminology_dict.items():
            # Calculate similarity score
            similarity = self._string_similarity(term_lower, standard_term)
            if similarity > 0.8 and similarity > best_score:  # 80% similarity threshold
                best_score = similarity
                best_match = mapping
        
        if best_match:
            entity.metadata["cui"] = best_match["cui"]
            entity.metadata["snomed"] = best_match.get("snomed")
            entity.metadata["match_confidence"] = best_score
            entity.normalized_value = best_match["preferred"]
        
        return entity
    
    def _string_similarity(self, s1, s2):
        """Simple string similarity calculation"""
        if s1 == s2:
            return 1.0
        
        # Check if one is substring of the other
        if s1 in s2:
            return len(s1) / len(s2)
        if s2 in s1:
            return len(s2) / len(s1)
        
        # Simple character-based similarity
        common_chars = set(s1) & set(s2)
        return len(common_chars) / max(len(set(s1)), len(set(s2)))
