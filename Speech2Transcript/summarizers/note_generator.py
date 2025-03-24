"""
Medical note generation module for creating SOAP notes and telehealth progress notes.
"""
import os
import re
from typing import Dict, Any
from .base_extractor import BaseExtractor

class NoteGenerator(BaseExtractor):
    """
    Specialized module for generating formatted medical notes including SOAP notes
    and telehealth progress notes based on extracted medical information.
    """
    
    def __init__(self, logger=None):
        """Initialize the note generator."""
        super().__init__(logger)
    
    def generate_soap_note(self, results: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate a comprehensive SOAP note from extracted information.
        
        Args:
            results: Dictionary with extracted medical information
            
        Returns:
            Dictionary with SOAP note sections
        """
        self._log("Generating SOAP note")
        
        soap = {
            "Subjective": "",
            "Objective": "",
            "Assessment": "",
            "Plan": ""
        }
        
        # Build Subjective section
        soap["Subjective"] = self._build_subjective_section(results)
        
        # Build Objective section
        soap["Objective"] = self._build_objective_section(results)
        
        # Build Assessment section
        soap["Assessment"] = self._build_assessment_section(results)
        
        # Build Plan section
        soap["Plan"] = self._build_plan_section(results)
        
        return soap
    
    def generate_telehealth_note(self, results: Dict[str, Any], template_path: str = None) -> str:
        """
        Generate a telehealth progress note from extracted information.
        
        Args:
            results: Dictionary with extracted medical information
            template_path: Optional path to a template file
            
        Returns:
            Formatted telehealth progress note as a string
        """
        self._log("Generating telehealth progress note")
        
        if template_path and os.path.exists(template_path):
            return self._fill_template(results, template_path)
        else:
            return self._create_standard_note(results)
    
    def _build_subjective_section(self, results: Dict[str, Any]) -> str:
        """Build the Subjective section of the SOAP note."""
        subjective_parts = []
        
        # Add symptom information
        health_status = results.get("health_status", {})
        if health_status.get("has_symptoms"):
            subjective_parts.append(f"Patient reports: {health_status.get('symptom_text', '')}")
        else:
            subjective_parts.append("Patient denies any unusual symptoms or health concerns.")
        
        # Add medication adherence
        medications = results.get("medications", {})
        if medications.get("adherence"):
            subjective_parts.append(medications["adherence"])
        
        # Add lifestyle information
        lifestyle = results.get("lifestyle", {})
        
        # Exercise information
        exercise = lifestyle.get("exercise", {})
        if exercise and isinstance(exercise, dict) and exercise.get("activity"):
            exercise_text = f"Exercise: {exercise.get('activity', 'exercise')}"
            if exercise.get("duration"):
                exercise_text += f" for {exercise['duration']} minutes"
            if exercise.get("frequency"):
                exercise_text += f" {exercise['frequency']}"
            subjective_parts.append(exercise_text)
        
        # Diet information
        diet = lifestyle.get("diet", {})
        if diet and isinstance(diet, dict) and (diet.get("quality") or diet.get("type")):
            diet_text = "Diet: "
            if diet.get("quality") and diet.get("type"):
                diet_text += f"{diet['quality']} {diet['type']} diet"
            elif diet.get("quality"):
                diet_text += f"{diet['quality']}"
            elif diet.get("type"):
                diet_text += f"{diet['type']} diet"
            else:
                diet_text += "Patient discussed dietary habits"
            subjective_parts.append(diet_text)
        
        # Smoking information
        smoking = lifestyle.get("smoking", {})
        if smoking and isinstance(smoking, dict) and smoking.get("status"):
            subjective_parts.append(f"Smoking: {smoking['status']}")
        
        # Alcohol information
        alcohol = lifestyle.get("alcohol", {})
        if alcohol and isinstance(alcohol, dict) and alcohol.get("status"):
            subjective_parts.append(f"Alcohol: {alcohol['status']}")
        
        return "\n".join(subjective_parts)
    
    def _build_objective_section(self, results: Dict[str, Any]) -> str:
        """Build the Objective section of the SOAP note."""
        objective_parts = []
        
        # Add vital signs
        vitals = results.get("health_status", {}).get("vital_signs", {})
        vital_parts = []
        
        # Blood pressure
        bp_list = vitals.get("blood_pressure", [])
        if bp_list and len(bp_list) > 0:
            bp = bp_list[0].get("full", "")
            if bp:
                vital_parts.append(f"- BP: {bp}")
        
        # Glucose
        glucose_list = vitals.get("glucose", [])
        if glucose_list and len(glucose_list) > 0:
            glucose = glucose_list[0].get("value", "")
            if glucose:
                vital_parts.append(f"- Glucose: {glucose} mg/dL")
        
        # Weight change
        weight_change = vitals.get("weight_change", {})
        if weight_change and isinstance(weight_change, dict):
            direction = weight_change.get("direction", "")
            value = weight_change.get("value", "")
            if direction and value:
                vital_parts.append(f"- Weight Change: {direction.capitalize()} {value} pounds")
        
        if vital_parts:
            objective_parts.append("Vital Signs:")
            objective_parts.extend(vital_parts)
        
        # Add medications
        med_list = results.get("medications", {}).get("medications", [])
        if med_list:
            objective_parts.append("\nMedications:")
            for med in med_list:
                med_line = f"- {med['name']}"
                if med.get("dosage"):
                    med_line += f" {med['dosage']}"
                if med.get("frequency"):
                    med_line += f" ({med['frequency']})"
                if med.get("is_active") is False:
                    med_line += " (discontinued)"
                objective_parts.append(med_line)
        
        # Add preventive care information
        preventive = results.get("preventive_care", {})
        preventive_parts = []
        
        for check_type, status in preventive.items():
            if status:
                preventive_parts.append(f"- {check_type.replace('_', ' ').title()}: {status}")
        
        if preventive_parts:
            objective_parts.append("\nPreventive Care:")
            objective_parts.extend(preventive_parts)
        
        return "\n".join(objective_parts)
    
    def _build_assessment_section(self, results: Dict[str, Any]) -> str:
        """Build the Assessment section of the SOAP note."""
        assessment_parts = []
        
        # Add medical conditions
        conditions = results.get("health_status", {}).get("conditions", [])
        if conditions:
            assessment_parts.append("Conditions:")
            for condition in conditions:
                condition_text = f"- {condition['name']}"
                if condition.get("severity"):
                    condition_text += f" ({condition['severity']})"
                assessment_parts.append(condition_text)
        
        # Overall assessment
        assessment_parts.append("\nOverall Assessment:")
        
        # Check vital signs and conditions for control assessment
        vitals = results.get("health_status", {}).get("vital_signs", {})
        bp_list = vitals.get("blood_pressure", [])
        glucose_list = vitals.get("glucose", [])
        
        # Blood pressure assessment
        if bp_list and len(bp_list) > 0:
            bp = bp_list[0]
            if isinstance(bp, dict) and "systolic" in bp and "diastolic" in bp:
                systolic = bp["systolic"]
                diastolic = bp["diastolic"]
                if systolic > 140 or diastolic > 90:
                    assessment_parts.append("- Hypertension not optimally controlled")
                else:
                    assessment_parts.append("- Blood pressure well controlled")
        
        # Glucose assessment
        if glucose_list and len(glucose_list) > 0:
            glucose = glucose_list[0].get("value", 0)
            if glucose > 140:
                assessment_parts.append("- Blood glucose elevated")
            else:
                assessment_parts.append("- Blood glucose well controlled")
        
        # Medication adherence assessment
        adherence = results.get("medications", {}).get("adherence", "")
        if "taking medications as prescribed" in adherence.lower():
            assessment_parts.append("- Good medication adherence")
        elif "issues with medication adherence" in adherence.lower():
            assessment_parts.append("- Medication adherence needs follow-up")
        else:
            assessment_parts.append("- Medication adherence to be monitored")
        
        # Weight management assessment
        weight_change = vitals.get("weight_change", {})
        if weight_change and isinstance(weight_change, dict) and weight_change.get("direction") == "loss":
            assessment_parts.append(f"- Positive weight management with {weight_change.get('value', '')} pound weight loss")
        
        return "\n".join(assessment_parts)
    
    def _build_plan_section(self, results: Dict[str, Any]) -> str:
        """Build the Plan section of the SOAP note."""
        plan_parts = []
        
        # Add follow-up plan
        follow_up = results.get("plan", {}).get("follow_up", {})
        if follow_up:
            if isinstance(follow_up, dict):
                if follow_up.get("timeframe") and follow_up.get("type"):
                    plan_parts.append(f"Follow-up: {follow_up['type']} appointment in {follow_up['timeframe']}" + 
                                     (f" with {follow_up['with_who']}" if follow_up.get('with_who') else ""))
                elif follow_up.get("complete_text"):
                    plan_parts.append(f"Follow-up: {follow_up['complete_text']}")
                else:
                    plan_parts.append("Follow-up: Continue routine monitoring")
            else:
                plan_parts.append(f"Follow-up: {follow_up}")
        else:
            plan_parts.append("Follow-up: Continue routine monitoring")
        
        # Recommendations
        plan_parts.append("\nRecommendations:")
        
        # Medication recommendations
        med_list = results.get("medications", {}).get("medications", [])
        if med_list:
            plan_parts.append("- Continue current medications")
            
            # Check for any medications that need adjustment
            if any(not med.get("is_active", True) for med in med_list):
                plan_parts.append("- Note medication changes in the record")
        
        # Monitoring recommendations
        monitor_parts = []
        conditions = results.get("health_status", {}).get("conditions", [])
        if conditions:
            if any("diabetes" in condition["name"].lower() for condition in conditions):
                monitor_parts.append("daily blood glucose")
            if any("hypertension" in condition["name"].lower() or "blood pressure" in condition["name"].lower() for condition in conditions):
                monitor_parts.append("blood pressure")
        
        if monitor_parts:
            plan_parts.append(f"- Continue monitoring {' and '.join(monitor_parts)}")
        
        # Lifestyle recommendations
        lifestyle = results.get("lifestyle", {})
        lifestyle_recs = []
        
        if lifestyle.get("diet") and isinstance(lifestyle["diet"], dict) and lifestyle["diet"].get("context"):
            lifestyle_recs.append("dietary habits")
        if lifestyle.get("exercise") and isinstance(lifestyle["exercise"], dict) and lifestyle["exercise"].get("context"):
            lifestyle_recs.append("physical activity")
            
        if lifestyle_recs:
            plan_parts.append(f"- Continue healthy {' and '.join(lifestyle_recs)}")
        else:
            plan_parts.append("- Maintain healthy lifestyle habits")
        
        # Preventive care recommendations
        preventive = results.get("preventive_care", {})
        preventive_recs = []
        
        for check_type, status in preventive.items():
            if status and "due" in str(status).lower():
                check_name = check_type.replace("_", " ")
                preventive_recs.append(f"schedule {check_name}")
        
        if preventive_recs:
            plan_parts.append("- " + ", ".join(preventive_recs))
        
        return "\n".join(plan_parts)
    
    def _fill_template(self, results: Dict[str, Any], template_path: str) -> str:
        """
        Fill a provided template with extracted data.
        
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
            
            # Fill in patient and care manager names
            template = template.replace("Patient Name: ____________________________",
                                       f"Patient Name: {patient_name}")
            template = template.replace("CMA (Care Manager): ____________________________",
                                       f"CMA (Care Manager): {care_manager_name}")
            
            # Extract health status info
            health_status = results.get("health_status", {})
            has_symptoms = health_status.get("has_symptoms", False)
            symptom_text = health_status.get("symptom_text", "")
            
            # Fill in symptoms
            template = template.replace('Patient states: "____________________________________"',
                                       f'Patient states: "{symptom_text}"')
            template = template.replace("Any new or worsening symptoms? ___________________",
                                       f"Any new or worsening symptoms? {'Yes' if has_symptoms else 'No'}")
            
            # Fill in barriers to care
            lifestyle = results.get("lifestyle", {})
            social_support = lifestyle.get("social_support", {})
            
            barriers = "None reported"
            if isinstance(social_support, dict) and social_support.get("has_support") is True:
                barriers = "None, good family support"
            
            template = template.replace("Any barriers to care (transportation, financial, etc.)? _______",
                                       f"Any barriers to care (transportation, financial, etc.)? {barriers}")
            
            # Fill in vital signs
            vitals = health_status.get("vital_signs", {})
            bp_value = "Not reported"
            if vitals.get("blood_pressure") and len(vitals["blood_pressure"]) > 0:
                bp_value = vitals["blood_pressure"][0].get("full", "Not reported")
            
            glucose_value = "Not reported"
            if vitals.get("glucose") and len(vitals["glucose"]) > 0:
                glucose_value = str(vitals["glucose"][0].get("value", "Not reported"))
            
            template = template.replace("Reading/value: _______________",
                                       f"Reading/value: BP {bp_value}, Glucose {glucose_value}")
            
            # Fill in medication adherence
            medications = results.get("medications", {})
            adherence = "Yes"
            if "adherence" in medications:
                adherence_text = medications.get("adherence", "")
                if "not taking" in adherence_text.lower() or "issues" in adherence_text.lower():
                    adherence = "No"
            
            template = template.replace("Does the patient report taking meds as prescribed? _____",
                                       f"Does the patient report taking meds as prescribed? {adherence}")
            
            # Fill in side effects
            side_effects = "None reported"
            if "side_effects" in medications:
                side_effects_text = medications.get("side_effects", "")
                if side_effects_text and side_effects_text != "No side effects reported":
                    side_effects = side_effects_text
            
            template = template.replace('Any side effects or concerns? "______________"',
                                       f'Any side effects or concerns? "{side_effects}"')
            
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
            
            template = template.replace("Follow-up Appointment: _______________",
                                       f"Follow-up Appointment: {follow_up_text}")
            
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
                template = re.sub(r"○ Medication A \(dose, frequency\).*?○ Etc.",
                                 med_text.strip(), template, flags=re.DOTALL)
            
            # Set provider name
            provider_name = "Dr. Cameron"  # Default provider name often mentioned in medical conversations
            template = template.replace("Provider Notification: CMA will notify Dr./NP/PA _______________ of significant",
                                       f"Provider Notification: CMA will notify {provider_name} of significant")
            
            return template
            
        except Exception as e:
            self._log(f"Error filling template: {e}", level="error")
            return self._create_standard_note(results)
    
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
