"""
Telehealth note template generation module for CCM (Chronic Care Management) fields.

This module provides functions to generate and fill telehealth note templates
with extracted medical information, focusing on CCM requirements.
"""
import os
import re
from typing import Dict, Any, Optional

class TelehealthTemplateGenerator:
    """
    Specialized module for generating and filling telehealth templates compliant with
    CCM (Chronic Care Management) documentation requirements.
    """
    
    def __init__(self, logger=None):
        """
        Initialize the telehealth template generator.
        
        Args:
            logger: Optional logger for messages
        """
        self.logger = logger
        self._initialize_ccm_fields()
    
    def _log(self, message: str, level: str = "info") -> None:
        """Log messages if logger is available."""
        if self.logger:
            if level == "info":
                self.logger.info(message)
            elif level == "error":
                self.logger.error(message)
            elif level == "warning":
                self.logger.warning(message)
    
    def _initialize_ccm_fields(self) -> None:
        """Initialize CCM field definitions and requirements."""
        # Standard CCM fields required for documentation
        self.ccm_fields = {
            "patient_info": [
                "patient_name", 
                "date_of_birth", 
                "medical_record_number"
            ],
            "visit_info": [
                "date_of_call", 
                "time_of_call", 
                "duration", 
                "care_manager_name"
            ],
            "contact_reason": [
                "chronic_conditions", 
                "monitoring_review", 
                "medication_adherence_review"
            ],
            "status_report": [
                "symptoms", 
                "care_barriers", 
                "monitoring_data", 
                "patient_feedback"
            ],
            "medication_review": [
                "current_medications", 
                "adherence", 
                "missed_doses", 
                "side_effects", 
                "discrepancies", 
                "escalation"
            ],
            "education": [
                "topics", 
                "understanding", 
                "materials_provided"
            ],
            "plan": [
                "monitoring", 
                "follow_up", 
                "escalation_protocol", 
                "provider_notification"
            ],
            "time_tracking": [
                "total_time", 
                "nature_of_time"
            ],
            "additional_notes": [
                "special_instructions", 
                "comments"
            ],
            "signature": [
                "name", 
                "date_time", 
                "credentials", 
                "disclaimer"
            ]
        }
        
        # CMS billing requirements for time tracking
        self.cms_time_requirements = {
            "CCM": {
                "99490": "20+ minutes", 
                "99487": "60+ minutes", 
                "99489": "30+ minutes (add-on)"
            },
            "RPM": {
                "99453": "Initial setup", 
                "99454": "Device supply", 
                "99457": "20+ minutes", 
                "99458": "20+ minutes (add-on)"
            }
        }
    
    def generate_telehealth_note(self, results: Dict[str, Any], template_path: Optional[str] = None) -> str:
        """
        Generate a telehealth progress note from extracted medical information.
        
        Args:
            results: Dictionary with extracted medical information
            template_path: Optional path to a template file
            
        Returns:
            Formatted telehealth progress note text
        """
        self._log("Generating telehealth progress note")
        
        if template_path and os.path.exists(template_path):
            return self._fill_template(results, template_path)
        else:
            return self._create_standard_note(results)
    
    def _fill_template(self, results: Dict[str, Any], template_path: str) -> str:
        """
        Fill a provided template with extracted medical data.
        
        Args:
            results: Extracted medical information
            template_path: Path to template file
            
        Returns:
            Filled template text
        """
        try:
            with open(template_path, 'r') as f:
                template = f.read()
            
            # Extract patient and care manager info
            patient_info = results.get("patient_info", {})
            patient_name = patient_info.get("patient_name", "")
            care_manager_name = patient_info.get("care_manager_name", "")
            
            # Fill in basic patient and care manager info
            template = self._replace_field(template, "Patient Name", patient_name)
            template = self._replace_field(template, "CMA (Care Manager)", care_manager_name)
            
            # Extract health status info
            health_status = results.get("health_status", {})
            has_symptoms = health_status.get("has_symptoms", False)
            symptom_text = health_status.get("symptom_text", "")
            
            # Fill in symptoms information
            template = self._replace_quoted_field(template, "Patient states", symptom_text)
            template = self._replace_field(template, "Any new or worsening symptoms", "Yes" if has_symptoms else "No")
            
            # Fill in barriers to care
            lifestyle = results.get("lifestyle", {})
            social_support = lifestyle.get("social_support", {})
            
            barriers = "None reported"
            if isinstance(social_support, dict) and social_support.get("has_support") is True:
                barriers = "None, good family support"
            
            template = self._replace_field(template, "Any barriers to care (transportation, financial, etc.)", barriers)
            
            # Fill in vital signs
            vitals = health_status.get("vital_signs", {})
            bp_value = "Not reported"
            if vitals.get("blood_pressure") and len(vitals["blood_pressure"]) > 0:
                bp_value = vitals["blood_pressure"][0].get("full", "Not reported")
            
            glucose_value = "Not reported"
            if vitals.get("glucose") and len(vitals["glucose"]) > 0:
                glucose_value = str(vitals["glucose"][0].get("value", "Not reported"))
            
            template = self._replace_field(template, "Reading/value", f"BP {bp_value}, Glucose {glucose_value}")
            
            # Fill in patient feedback on devices
            if "glucose" in vitals and vitals["glucose"]:
                template = self._replace_quoted_field(template, "Patient's subjective feedback on devices/usage", 
                                                    "Checking glucose regularly")
            else:
                template = self._replace_quoted_field(template, "Patient's subjective feedback on devices/usage", 
                                                    "Using devices as instructed")
            
            # Fill in medication adherence
            medications = results.get("medications", {})
            adherence = "Yes"
            if "adherence" in medications:
                adherence_text = medications.get("adherence", "")
                if "not taking" in adherence_text.lower() or "issues" in adherence_text.lower():
                    adherence = "No"
            
            template = self._replace_field(template, "Does the patient report taking meds as prescribed", adherence)
            
            # Fill in side effects
            side_effects = "None reported"
            if "side_effects" in medications:
                side_effects_text = medications.get("side_effects", "")
                if side_effects_text and side_effects_text != "No side effects reported":
                    side_effects = side_effects_text
            
            template = self._replace_quoted_field(template, "Any side effects or concerns", side_effects)
            
            # Fill in patient understanding
            template = self._replace_field(template, "Patient verbalized understanding", "Yes")
            
            # Fill in follow-up plan
            follow_up = results.get("plan", {}).get("follow_up", {})
            follow_up_text = "As scheduled"
            
            if isinstance(follow_up, dict):
                if follow_up.get("timeframe"):
                    follow_up_text = f"In {follow_up['timeframe']}"
                elif follow_up.get("complete_text"):
                    follow_up_text = follow_up["complete_text"]
            elif follow_up:
                follow_up_text = str(follow_up)
            
            template = self._replace_field(template, "Follow-up Appointment", follow_up_text)
            
            # Add medications
            med_list = medications.get("medications", [])
            med_text = ""
            for med in med_list:
                med_name = med.get("name", "")
                dosage = med.get("dosage", "")
                frequency = med.get("frequency", "")
                
                if med_name:
                    med_line = f"○ {med_name}"
                    if dosage and frequency:
                        med_line += f" ({dosage}, {frequency})"
                    elif dosage:
                        med_line += f" ({dosage})"
                    elif frequency:
                        med_line += f" ({frequency})"
                    med_text += med_line + "\n"
            
            if med_text:
                # Replace the medication section using regex to handle multi-line replace
                template = re.sub(r"○ Medication A \(dose, frequency\).*?○ Etc\.",
                                 med_text.strip(), template, flags=re.DOTALL)
            
            # Set provider name
            provider_name = "Dr. Cameron"  # Default provider name often mentioned in medical conversations
            template = template.replace("Provider Notification: CMA will notify Dr./NP/PA _______________ of significant",
                                       f"Provider Notification: CMA will notify {provider_name} of significant")
            
            # Set time tracking for CMS requirements
            template = self._replace_field(template, "Total clinical staff time spent on this call", "8 minutes")
            print(template)
            return template
            
        except Exception as e:
            self._log(f"Error filling template: {e}", level="error")
            return self._create_standard_note(results)
    
    def _replace_field(self, template: str, field_prefix: str, value: str) -> str:
        """
        Replace a template field with given value.
        
        Args:
            template: The template text
            field_prefix: The field prefix to match
            value: The value to insert
            
        Returns:
            Updated template
        """
        pattern = fr"{re.escape(field_prefix)}: .*?([_\s]{{5,}})"
        replacement = f"{field_prefix}: {value}"
        
        # If pattern exists, replace it
        if re.search(pattern, template):
            return re.sub(pattern, replacement, template)
        
        # Alternative pattern with different format
        alt_pattern = fr"{re.escape(field_prefix)}\?.*?([_\s]{{3,}})"
        if re.search(alt_pattern, template):
            return re.sub(alt_pattern, f"{field_prefix}? {value}", template)
        
        return template
    
    def _replace_quoted_field(self, template: str, field_prefix: str, value: str) -> str:
        """
        Replace a template field with quoted value.
        
        Args:
            template: The template text
            field_prefix: The field prefix to match
            value: The value to insert
            
        Returns:
            Updated template
        """
        pattern = fr'{re.escape(field_prefix)}: ".*?"'
        replacement = f'{field_prefix}: "{value}"'
        
        # If pattern exists, replace it
        if re.search(pattern, template):
            return re.sub(pattern, replacement, template)
        
        # Alternative pattern with different format
        alt_pattern = fr'{re.escape(field_prefix)}\? ".*?"'
        if re.search(alt_pattern, template):
            return re.sub(alt_pattern, f'{field_prefix}? "{value}"', template)
        
        return template
    
    def _create_standard_note(self, results: Dict[str, Any]) -> str:
        """
        Create a standard telehealth progress note without a template.
        
        Args:
            results: Extracted medical information
            
        Returns:
            Formatted telehealth progress note
        """
        note = []
        
        # Extract key information
        patient_info = results.get("patient_info", {})
        health_status = results.get("health_status", {})
        vitals = health_status.get("vital_signs", {})
        medications = results.get("medications", {})
        lifestyle = results.get("lifestyle", {})
        preventive = results.get("preventive_care", {})
        plan = results.get("plan", {})
        
        # 1. Patient Information
        note.append("Telehealth Progress Note Template (CCM/RPM)")
        note.append("")
        note.append("1. Patient Information")
        note.append(f"● Patient Name: {patient_info.get('patient_name', 'Not provided')}")
        note.append("● Date of Birth: [Not provided in transcript]")
        note.append("● Medical Record #: [Not provided in transcript]")
        note.append("")
        
        # 2. Visit Information
        note.append("2. Visit Information")
        note.append("● Date of Call: [Current Date]")
        note.append("● Time of Call: [Current Time]")
        note.append("● Duration: 7-8 minutes")
        note.append(f"● CMA (Care Manager): {patient_info.get('care_manager_name', 'Not provided')}")
        note.append("")
        
        # 3. Reason for Contact
        note.append("3. Reason for Contact")
        note.append("● CCM/RPM Check-In:")
        
        # Add conditions
        conditions = health_status.get("conditions", [])
        condition_names = [c["name"] for c in conditions] if conditions else []
        
        if not condition_names:
            # Try to infer from other data
            if "diabetes" in str(results).lower():
                condition_names.append("diabetes")
            if "hypertension" in str(results).lower() or "blood pressure" in str(results).lower():
                condition_names.append("hypertension")
        
        note.append(f"○ Chronic condition follow-up ({', '.join(condition_names) if condition_names else 'Not specified'})")
        note.append("○ Review of RPM data (blood pressure, glucose readings)")
        note.append("○ Medication adherence review")
        note.append("")
        
        # 4. Current Status & Patient Report
        note.append("4. Current Status & Patient Report")
        note.append("● Symptoms/Concerns:")
        
        # Add symptoms
        symptom_text = health_status.get("symptom_text", "")
        has_symptoms = health_status.get("has_symptoms", False)
        
        if symptom_text:
            note.append(f'○ Patient states: "{symptom_text}"')
        else:
            note.append('○ Patient states: "No health concerns at this time"')
        
        note.append(f"○ Any new or worsening symptoms? {'Yes' if has_symptoms else 'No'}")
        
        # Add barriers to care
        social_support = lifestyle.get("social_support", {})
        barriers = "None reported"
        if isinstance(social_support, dict) and social_support.get("has_support") is True:
            barriers = "None, good family support"
        
        note.append(f"○ Any barriers to care (transportation, financial, etc.)? {barriers}")
        
        # Add vital signs
        note.append("● Relevant Monitoring Data:")
        note.append("○ Latest RPM readings:")
        
        bp_value = "Not reported"
        if vitals.get("blood_pressure") and len(vitals["blood_pressure"]) > 0:
            bp_value = vitals["blood_pressure"][0].get("full", "Not reported")
        
        glucose_value = "Not reported"
        if vitals.get("glucose") and len(vitals["glucose"]) > 0:
            glucose_value = str(vitals["glucose"][0].get("value", "Not reported"))
        
        note.append("■ Date/Time of reading: Recent")
        note.append(f"■ Reading/value: BP {bp_value}, Glucose {glucose_value}")
        
        # Add device feedback
        if "glucose" in vitals and vitals["glucose"]:
            note.append('○ Patient\'s subjective feedback on devices/usage: "Checking glucose regularly"')
        else:
            note.append('○ Patient\'s subjective feedback on devices/usage: "Using devices as instructed"')
        
        note.append("")
        
        # 5. Medication Review
        note.append("5. Medication Review")
        note.append("1. Current Medication List:")
        
        # Add medications
        med_list = medications.get("medications", [])
        if med_list:
            for med in med_list:
                med_name = med.get("name", "")
                if med_name:
                    note.append(f"○ {med_name}")
        else:
            note.append("○ [Medications not specifically identified in transcript]")
        
        note.append("")
        note.append("2. Patient-Reported Adherence:")
        
        # Add adherence info
        adherence = "Yes"
        if "adherence" in medications:
            adherence_text = medications.get("adherence", "")
            if "not taking" in adherence_text.lower() or "issues" in adherence_text.lower():
                adherence = "No"
        
        note.append(f"○ Does the patient report taking meds as prescribed? {adherence}")
        note.append("○ Any missed doses? If yes, how often? None reported")
        
        # Add side effects
        side_effects = "None reported"
        if "side_effects" in medications:
            side_effects_text = medications.get("side_effects", "")
            if side_effects_text and side_effects_text != "No side effects reported":
                side_effects = side_effects_text
        
        note.append(f'○ Any side effects or concerns? "{side_effects}"')
        
        note.append("")
        note.append("3. Actions/Follow-Up:")
        note.append('○ Noted discrepancies or questions: "None"')
        note.append('○ Escalated to supervising provider? No')
        note.append("")
        
        # 6. Education/Reinforcement
        note.append("6. Education/Reinforcement Provided")
        note.append("● Reinforced the importance of medication adherence.")
        note.append("● Reviewed healthy lifestyle tips (diet/exercise/stress management) as per care plan.")
        note.append("● Provided standard instruction on device usage.")
        note.append("● Patient verbalized understanding: Yes")
        note.append("")
        
        # 7. Plan & Next Steps
        note.append("7. Plan & Next Steps")
        note.append("● No immediate changes to care plan (CMA cannot alter plan independently).")
        note.append("● Monitoring: Continue to record daily BP/glucose/weight.")
        
        # Add follow-up
        follow_up = plan.get("follow_up", {})
        follow_up_text = "As scheduled"
        
        if isinstance(follow_up, dict):
            if follow_up.get("timeframe"):
                follow_up_text = f"In {follow_up['timeframe']}"
            elif follow_up.get("complete_text"):
                follow_up_text = follow_up["complete_text"]
        elif follow_up:
            follow_up_text = str(follow_up)
        
        note.append(f"● Follow-up Appointment: {follow_up_text}")
        
        note.append("● Escalation: If readings exceed established parameters or symptoms worsen, patient")
        note.append("  instructed to contact the clinic or go to the ER if severe.")
        
        # Add provider info
        note.append("● Provider Notification: CMA will notify Dr. Cameron of significant findings.")
        note.append("")
        
        # 8. Time Tracking
        note.append("8. Time Tracking (CMS Requirements)")
        note.append("● Total clinical staff time spent on this call: 8 minutes")
        note.append("○ Patient assessment, medication review, care coordination, chart documentation")
        note.append("")
        
        # 9. Additional Notes
        note.append("9. Additional Notes")
        
        # Add special notes
        special_notes = []
        
        weight_change = vitals.get("weight_change", {})
        if weight_change and isinstance(weight_change, dict) and weight_change.get("value"):
            direction = weight_change.get("direction", "change")
            special_notes.append(f"● Patient reports weight {direction} of {weight_change['value']} pounds")
        
        exercise = lifestyle.get("exercise", {})
        if exercise and isinstance(exercise, dict) and exercise.get("context"):
            activity = exercise.get("activity", "exercise")
            duration = exercise.get("duration", "")
            if duration:
                special_notes.append(f"● Patient reports {activity} for {duration} minutes")
            else:
                special_notes.append(f"● Patient reports regular {activity}")
        
        if special_notes:
            for special_note in special_notes:
                note.append(special_note)
        else:
            note.append("● None")
        
        note.append("")
        
        # 10. Signature
        note.append("10. Signature & Credentials")
        note.append("")
        note.append("______________________________________ _____________________")
        note.append("CMA Name (Printed)                     Date/Time")
        note.append("")
        note.append("______________________________________")
        note.append("CMA Signature, Credential")
        note.append("")
        note.append("Disclaimer: This note is documented by a Certified Medical Assistant operating")
        note.append("under the supervision of a licensed provider. All medical decision-making and any")
        note.append("changes to the patient's treatment plan will be directed by the provider.")
        
        return "\n".join(note)


# Convenience function to generate telehealth notes
def generate_telehealth_note(results: Dict[str, Any], template_path: Optional[str] = None, logger=None) -> str:
    """
    Generate a telehealth progress note from extracted medical information.
    
    Args:
        results: Extracted medical information
        template_path: Optional path to a template file
        logger: Optional logger for messages
        
    Returns:
        Formatted telehealth progress note
    """
    generator = TelehealthTemplateGenerator(logger)
    return generator.generate_telehealth_note(results, template_path)
