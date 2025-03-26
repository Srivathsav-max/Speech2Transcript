"""
CCM (Chronic Care Management) Template Generator

This module provides specialized generation of CCM templates using the extracted
information from advanced NLP processing.
"""
import os
import re
from datetime import datetime
from typing import Dict, List, Any, Optional

class CCMTemplateGenerator:
    """
    Specialized module for generating CCM templates compliant with
    Medicare requirements for CCM billing and documentation.
    
    This class uses advanced template filling to create human-readable
    and regulatory-compliant CCM documentation.
    """
    
    def __init__(self, template_path: Optional[str] = None, logger = None):
        """
        Initialize the CCM template generator.
        
        Args:
            template_path: Optional path to a custom template file
            logger: Optional logger for messages
        """
        self.logger = logger
        self.template_path = template_path
        self._initialize_default_template()
        
    def _log(self, message: str, level: str = "info") -> None:
        """Log messages if logger is available."""
        if self.logger:
            if level == "info":
                self.logger.info(message)
            elif level == "error":
                self.logger.error(message)
            elif level == "warning":
                self.logger.warning(message)
                
    def _initialize_default_template(self) -> None:
        """Initialize the default CCM template."""
        self.default_template = """
# Chronic Care Management Note

Date: {date}
Patient: {patient_name}
Care Manager: {care_manager_name}
Call Duration: {duration} minutes
CCM Billing Code: {billing_code}

## Chronic Conditions Addressed
{chronic_conditions}

## Current Status
{status_report}

## Vital Signs/Monitoring Data
{vital_signs}

## Medication Management
{medication_management}

## Care Plan Status and Updates
{care_plan_updates}

## Patient Goals
{patient_goals}

## Patient Education Provided
{education}

## Identified Social Determinants of Health
{social_determinants}

## Care Coordination
{care_coordination}

## Follow-up Plan
{follow_up}

## Time Spent
Total clinical staff time: {time_spent} minutes
- Assessment: {assessment_time} minutes
- Care coordination: {coordination_time} minutes
- Patient education: {education_time} minutes
- Documentation: {documentation_time} minutes

---
Note completed by: {provider_name}
Date/Time: {completion_date}
"""

    def generate(self, extracted_data: Dict[str, Any]) -> str:
        """
        Generate a CCM note from extracted data.
        
        Args:
            extracted_data: Dictionary with extracted CCM information
            
        Returns:
            Formatted CCM note
        """
        self._log("Generating CCM note")
        
        # If a custom template path is provided and exists, use it
        if self.template_path and os.path.exists(self.template_path):
            try:
                with open(self.template_path, 'r') as f:
                    template = f.read()
            except Exception as e:
                self._log(f"Error reading template file: {e}", level="error")
                template = self.default_template
        else:
            template = self.default_template
        
        # Get basic information
        patient_info = extracted_data.get("patient_info", {})
        patient_name = patient_info.get("patient_name", "Unknown Patient")
        care_manager_name = patient_info.get("care_manager_name", "Unknown Provider")
        
        # Format time information
        time_info = extracted_data.get("ccm_data", {}).get("time_spent", {})
        total_minutes = time_info.get("total_minutes", 0)
        if not total_minutes and time_info.get("time_mentions"):
            time_mentions = time_info.get("time_mentions", [])
            total_minutes = max([mention.get("minutes", 0) for mention in time_mentions]) if time_mentions else 8
        
        # If no time information is available, use default values
        if not total_minutes:
            total_minutes = 8  # Default value
            
        # Calculate sub-times based on heuristics
        assessment_time = int(total_minutes * 0.4)  # 40% of time on assessment
        coordination_time = int(total_minutes * 0.2)  # 20% of time on coordination
        education_time = int(total_minutes * 0.2)  # 20% of time on education
        documentation_time = total_minutes - assessment_time - coordination_time - education_time
        
        # Get suggested billing code
        billing_info = extracted_data.get("ccm_data", {}).get("billing_codes", {})
        billing_code = "99490"  # Default code (20+ minutes)
        
        if billing_info and billing_info.get("suggested_codes"):
            suggested_codes = billing_info.get("suggested_codes")
            if "99487" in suggested_codes:  # 60+ minutes
                billing_code = "99487"
            elif "99490" in suggested_codes:  # 20+ minutes
                billing_code = "99490"
                
        # Format chronic conditions
        chronic_conditions = self._format_chronic_conditions(
            extracted_data.get("ccm_data", {}).get("chronic_conditions", [])
        )
        
        # Format status report
        status_report = self._format_status_report(
            extracted_data.get("health_status", {}),
            extracted_data.get("ccm_data", {}).get("care_plan_updates", {})
        )
        
        # Format vital signs
        vital_signs = self._format_vital_signs(
            extracted_data.get("health_status", {}).get("vital_signs", {})
        )
        
        # Format medication management
        medication_management = self._format_medication_management(
            extracted_data.get("ccm_data", {}).get("medication_management", {}),
            extracted_data.get("medications", {})
        )
        
        # Format care plan updates
        care_plan_updates = self._format_care_plan_updates(
            extracted_data.get("ccm_data", {}).get("care_plan_updates", {})
        )
        
        # Format patient goals
        patient_goals = self._format_patient_goals(
            extracted_data.get("ccm_data", {}).get("patient_goals", [])
        )
        
        # Format patient education
        education = self._format_education(
            extracted_data.get("health_status", {}),
            extracted_data.get("medications", {})
        )
        
        # Format social determinants
        social_determinants = self._format_social_determinants(
            extracted_data.get("ccm_data", {}).get("social_determinants", {})
        )
        
        # Format care coordination
        care_coordination = self._format_care_coordination(
            extracted_data.get("ccm_data", {}).get("care_coordination", [])
        )
        
        # Format follow-up plan
        follow_up = self._format_follow_up(
            extracted_data.get("plan", {}).get("follow_up", {})
        )
        
        # Format dates
        current_date = datetime.now().strftime("%Y-%m-%d")
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M")
        
        # Fill in template
        formatted_note = template.format(
            date=current_date,
            patient_name=patient_name,
            care_manager_name=care_manager_name,
            provider_name=care_manager_name,
            duration=total_minutes,
            time_spent=total_minutes,
            assessment_time=assessment_time,
            coordination_time=coordination_time,
            education_time=education_time,
            documentation_time=documentation_time,
            billing_code=billing_code,
            chronic_conditions=chronic_conditions,
            status_report=status_report,
            vital_signs=vital_signs,
            medication_management=medication_management,
            care_plan_updates=care_plan_updates,
            patient_goals=patient_goals,
            education=education,
            social_determinants=social_determinants,
            care_coordination=care_coordination,
            follow_up=follow_up,
            completion_date=current_datetime
        )
        
        return formatted_note
    
    def _format_chronic_conditions(self, conditions: List[Dict]) -> str:
        """Format chronic conditions section."""
        if not conditions:
            return "No chronic conditions identified from transcript."
            
        formatted_text = []
        for condition in conditions:
            condition_text = f"- {condition.get('name', '').capitalize()}"
            context = condition.get('context', '')
            if context:
                # Extract a shorter, cleaner context
                context = re.sub(r'\s+', ' ', context).strip()
                condition_text += f": \"{context}\""
                
            formatted_text.append(condition_text)
            
        return "\n".join(formatted_text)
    
    def _format_status_report(self, health_status: Dict, care_updates: Dict) -> str:
        """Format current status section."""
        formatted_text = []
        
        # Add symptom information
        has_symptoms = health_status.get("has_symptoms", False)
        symptom_text = health_status.get("symptom_text", "")
        
        if has_symptoms and symptom_text:
            formatted_text.append(f"Patient reports: \"{symptom_text}\"")
        elif has_symptoms:
            formatted_text.append("Patient reports symptoms, details not specified in transcript.")
        else:
            formatted_text.append("Patient denies any new or worsening symptoms.")
        
        # Add condition status if available
        conditions = health_status.get("conditions", [])
        for condition in conditions:
            name = condition.get("name", "")
            status = condition.get("status", "")
            if name and status:
                formatted_text.append(f"- {name.capitalize()}: {status}")
        
        if not formatted_text:
            formatted_text.append("No specific status information available from transcript.")
            
        return "\n".join(formatted_text)
    
    def _format_vital_signs(self, vitals: Dict) -> str:
        """Format vital signs section."""
        formatted_text = []
        
        # Blood pressure
        if vitals.get("blood_pressure") and len(vitals["blood_pressure"]) > 0:
            bp = vitals["blood_pressure"][0].get("full", "")
            if bp:
                formatted_text.append(f"- Blood Pressure: {bp}")
        
        # Glucose
        if vitals.get("glucose") and len(vitals["glucose"]) > 0:
            glucose = vitals["glucose"][0].get("value", "")
            if glucose:
                formatted_text.append(f"- Glucose: {glucose} mg/dL")
        
        # Weight
        if vitals.get("weight"):
            weight = vitals["weight"].get("value", "")
            if weight:
                formatted_text.append(f"- Weight: {weight} lbs")
        
        # Weight change
        if vitals.get("weight_change") and isinstance(vitals["weight_change"], dict):
            weight_change = vitals["weight_change"]
            direction = weight_change.get("direction", "")
            value = weight_change.get("value", "")
            if direction and value:
                formatted_text.append(f"- Weight Change: {direction} {value} lbs")
        
        # Heart rate
        if vitals.get("heart_rate"):
            heart_rate = vitals["heart_rate"].get("value", "")
            if heart_rate:
                formatted_text.append(f"- Heart Rate: {heart_rate} bpm")
        
        # Temperature
        if vitals.get("temperature"):
            temp = vitals["temperature"].get("value", "")
            if temp:
                formatted_text.append(f"- Temperature: {temp}°F")
        
        if not formatted_text:
            formatted_text.append("No vital signs reported in transcript.")
            
        return "\n".join(formatted_text)
    
    def _format_medication_management(self, med_management: Dict, medications: Dict) -> str:
        """Format medication management section."""
        formatted_text = []
        
        # Add current medications
        med_list = medications.get("medications", []) or med_management.get("medications", [])
        if med_list:
            formatted_text.append("Current medications:")
            for med in med_list:
                med_name = med.get("name", "")
                dosage = med.get("dosage", "")
                frequency = med.get("frequency", "")
                
                med_text = f"- {med_name}"
                if dosage and frequency:
                    med_text += f" ({dosage}, {frequency})"
                elif dosage:
                    med_text += f" ({dosage})"
                elif frequency:
                    med_text += f" ({frequency})"
                    
                formatted_text.append(med_text)
            formatted_text.append("")
        
        # Add adherence information
        adherence = medications.get("adherence", "") or med_management.get("adherence", "")
        if adherence:
            formatted_text.append(f"Medication adherence: {adherence}")
        
        # Add side effects
        side_effects = medications.get("side_effects", "") or med_management.get("side_effects", "")
        if side_effects and side_effects != "No side effects reported":
            formatted_text.append(f"Side effects: {side_effects}")
        else:
            formatted_text.append("No medication side effects reported.")
        
        if not formatted_text:
            formatted_text.append("No medication information available from transcript.")
            
        return "\n".join(formatted_text)
    
    def _format_care_plan_updates(self, care_updates: Dict) -> str:
        """Format care plan updates section."""
        formatted_text = []
        
        # Add monitoring changes
        monitoring_changes = care_updates.get("monitoring_changes", [])
        if monitoring_changes:
            formatted_text.append("Monitoring changes:")
            for change in monitoring_changes:
                formatted_text.append(f"- {change.get('text', '')}")
            formatted_text.append("")
        
        # Add medication changes
        med_changes = care_updates.get("medication_changes", [])
        if med_changes:
            formatted_text.append("Medication changes:")
            for change in med_changes:
                formatted_text.append(f"- {change.get('text', '')}")
            formatted_text.append("")
        
        # Add lifestyle recommendations
        lifestyle_recs = care_updates.get("lifestyle_recommendations", [])
        if lifestyle_recs:
            formatted_text.append("Lifestyle recommendations:")
            for rec in lifestyle_recs:
                formatted_text.append(f"- {rec.get('text', '')}")
            formatted_text.append("")
        
        # Add referrals
        referrals = care_updates.get("referrals", [])
        if referrals:
            formatted_text.append("Referrals:")
            for referral in referrals:
                formatted_text.append(f"- {referral.get('text', '')}")
            formatted_text.append("")
        
        if not monitoring_changes and not med_changes and not lifestyle_recs and not referrals:
            formatted_text.append("No changes to care plan identified from transcript.")
            formatted_text.append("Continue with current care plan as previously established.")
            
        return "\n".join(formatted_text)
    
    def _format_patient_goals(self, goals: List[Dict]) -> str:
        """Format patient goals section."""
        if not goals:
            return "No specific patient goals identified from transcript."
            
        formatted_text = []
        for goal in goals:
            goal_text = f"- {goal.get('text', '')}"
            category = goal.get('category', '')
            if category and category != "other":
                goal_text += f" (Category: {category})"
                
            formatted_text.append(goal_text)
            
        return "\n".join(formatted_text)
    
    def _format_education(self, health_status: Dict, medications: Dict) -> str:
        """Format patient education section."""
        formatted_text = [
            "Education provided on:",
            "- Importance of medication adherence",
            "- Self-monitoring of chronic conditions"
        ]
        
        # Add condition-specific education
        conditions = health_status.get("conditions", []) 
        for condition in conditions:
            name = condition.get("name", "").lower()
            if "diabetes" in name:
                formatted_text.append("- Blood glucose monitoring and management")
            elif "hypertension" in name or "blood pressure" in name:
                formatted_text.append("- Blood pressure control and monitoring")
            elif "heart" in name or "cardiac" in name:
                formatted_text.append("- Heart health and cardiovascular disease management")
            elif "copd" in name or "pulmonary" in name or "lung" in name:
                formatted_text.append("- Breathing techniques and respiratory management")
        
        # Add medication-specific education if medications present
        if medications.get("medications"):
            formatted_text.append("- Medication purpose, proper usage, and potential side effects")
        
        # Add general health education
        formatted_text.extend([
            "- Nutrition and healthy eating habits",
            "- Physical activity recommendations",
            "- Stress management techniques"
        ])
        
        formatted_text.append("\nPatient verbalized understanding of education provided.")
        
        return "\n".join(formatted_text)
    
    def _format_social_determinants(self, sdoh: Dict) -> str:
        """Format social determinants section."""
        formatted_text = []
        
        # Check each SDOH category
        for category, data in sdoh.items():
            if data and isinstance(data, dict) and data.get("identified"):
                category_name = category.replace("_", " ").capitalize()
                formatted_text.append(f"- {category_name}: {data.get('text', '')}")
        
        if not formatted_text:
            formatted_text.append("No social determinants of health barriers identified.")
            
        return "\n".join(formatted_text)
    
    def _format_care_coordination(self, coordination: List[Dict]) -> str:
        """Format care coordination section."""
        if not coordination:
            return "No care coordination activities identified from transcript."
            
        formatted_text = []
        types_seen = set()
        
        for item in coordination:
            activity_type = item.get("type", "")
            text = item.get("text", "")
            
            if activity_type not in types_seen:
                type_name = activity_type.replace("_", " ").capitalize()
                formatted_text.append(f"- {type_name}: {text}")
                types_seen.add(activity_type)
            
        return "\n".join(formatted_text)
    
    def _format_follow_up(self, follow_up: Dict) -> str:
        """Format follow-up plan section."""
        if not follow_up:
            return "Routine follow-up as previously scheduled."
            
        formatted_text = ["Follow-up plan:"]
        
        if isinstance(follow_up, dict):
            # Try to get timeframe
            timeframe = follow_up.get("timeframe")
            follow_type = follow_up.get("type")
            with_who = follow_up.get("with_who")
            complete_text = follow_up.get("complete_text")
            
            if timeframe and follow_type:
                text = f"- {follow_type.capitalize()} follow-up in {timeframe}"
                if with_who:
                    text += f" with {with_who}"
                formatted_text.append(text)
            elif complete_text:
                formatted_text.append(f"- {complete_text}")
            else:
                formatted_text.append("- Continue with routine follow-up as previously scheduled.")
        else:
            formatted_text.append(f"- {follow_up}")
        
        # Add standard instructions
        formatted_text.extend([
            "",
            "Patient instructed to:",
            "- Continue monitoring vital signs",
            "- Take medications as prescribed",
            "- Contact office for any worsening symptoms"
        ])
        
        return "\n".join(formatted_text)


# Convenience function to generate CCM notes
def generate_ccm_note(results: Dict[str, Any], template_path: Optional[str] = None, logger = None) -> str:
    """
    Generate a CCM note from extracted medical information.
    
    Args:
        results: Extracted medical information
        template_path: Optional path to a template file
        logger: Optional logger for messages
        
    Returns:
        Formatted CCM note
    """
    generator = CCMTemplateGenerator(template_path, logger)
    return generator.generate(results)