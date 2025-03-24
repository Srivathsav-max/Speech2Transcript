def _load_medical_terminologies():
        return {
            "symptoms": {
                "shortness of breath": ["sob", "dyspnea", "breathing difficulty", "trouble breathing", "breathlessness"],
                "chest pain": ["angina", "chest discomfort", "chest tightness", "chest pressure"],
                "high blood pressure": ["hypertension", "htn", "elevated bp", "elevated blood pressure"],
                "low blood pressure": ["hypotension", "low bp"],
                "elevated blood sugar": ["hyperglycemia", "high glucose", "high blood sugar", "high sugar", "elevated glucose"],
                "low blood sugar": ["hypoglycemia", "low glucose", "low sugar"],
                "dizziness": ["lightheadedness", "vertigo", "feeling faint", "spinning sensation"],
                "fatigue": ["tiredness", "exhaustion", "weakness", "lethargy", "malaise"],
                "nausea": ["feeling sick", "upset stomach", "queasy"],
                "heart attack": ["myocardial infarction", "mi", "cardiac arrest", "coronary"],
                "stroke": ["cva", "cerebrovascular accident", "brain attack", "cerebral infarction"],
                "diabetes": ["diabetes mellitus", "dm", "t1dm", "t2dm", "type 1 diabetes", "type 2 diabetes", "adult onset diabetes", "sugar disease"]
            },
            
            # Medical concept normalizations
            "medications": {
                "blood pressure medication": ["antihypertensive", "bp medication", "bp med", "antihypertensive medication"],
                "blood thinner": ["anticoagulant", "antiplatelet", "blood thinner medication", "warfarin", "coumadin", "apixaban", "eliquis", "rivaroxaban", "xarelto"],
                "diabetes medication": ["antidiabetic", "insulin", "oral hypoglycemic", "glucose lowering medication", "sugar medication", "sugar pill"],
                "cholesterol medication": ["statin", "lipid lowering", "anticholesterol", "cholesterol lowering", "cholesterol reducer"]
            },
            
            # Unit standardizations
            "units": {
                "blood_glucose": {
                    "mg/dl": ["mg/dl", "milligrams per deciliter", "mg per dl", "mg/deciliter", "mg per deciliter"],
                    "mmol/l": ["mmol/l", "millimoles per liter", "mmol per l", "mmol/liter", "mmol per liter"]
                },
                "blood_pressure": {
                    "mmHg": ["mmHg", "mm Hg", "millimeters of mercury", "mm of mercury", "millimeters mercury"]
                },
                "weight": {
                    "kg": ["kg", "kilograms", "kilogram", "kgs", "kilo", "kilos"],
                    "lb": ["lb", "lbs", "pounds", "pound"]
                },
                "temperature": {
                    "°C": ["°C", "degrees celsius", "celsius", "centigrade", "degrees C", "C"],
                    "°F": ["°F", "degrees fahrenheit", "fahrenheit", "degrees F", "F"]
                }
            }
        }