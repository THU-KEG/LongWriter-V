from typing import List, Dict, Optional
from dataclasses import dataclass
from inference.api.gpt import GPT_Interface, DeepSeek_Interface
import logging
import json
from pathlib import Path
from utils import extract_json, count_words
import hashlib
import requests
from pymongo import MongoClient
from datetime import datetime
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ScriptPolishAgent")

@dataclass
class ModificationAnalysis:
    """Analysis of user modifications to scripts"""
    patterns: List[str]  # Patterns found in user modifications
    style_changes: List[str]  # Style changes made by user
    content_changes: List[str]  # Content changes made by user
    polish_prompt: str  # Generated prompt for polishing

    def to_dict(self) -> Dict:
        """Convert analysis to dictionary for logging"""
        return {
            "patterns": self.patterns,
            "style_changes": self.style_changes,
            "content_changes": self.content_changes,
            "polish_prompt": self.polish_prompt
        }

@dataclass
class PolishResult:
    """Result of script polishing process"""
    polished_script: str
    is_satisfied: bool
    feedback: str
    iterations: int

    def to_dict(self) -> Dict:
        """Convert result to dictionary for logging"""
        return {
            "is_satisfied": self.is_satisfied,
            "feedback": self.feedback,
            "iterations": self.iterations,
            "script_length": len(self.polished_script)
        }

class ScriptPolishAgent:
    def __init__(self, model_name: str = "gpt-4o-2024-05-13", config_path: str = "config.json"):
        self.model_name = model_name
        self.max_iterations = 3
        
        # Load configuration for MongoDB
        with open(config_path) as f:
            config = json.load(f)
        
        # Initialize MongoDB connection
        self.mongo_client = MongoClient(
            host=config["mongo_cache_host"],
            port=config["mongo_cache_port"]
        )
        self.db = self.mongo_client[config["mongo_cache_db"]]
        self.embeddings_collection = self.db["clip_embeddings"]
        
        # Create indexes
        self.embeddings_collection.create_index([("ppt_id", 1), ("slide_index", 1)], unique=True)
        
        # CLIP service URL
        self.clip_url = config["clip_url"]
        
        logger.info(f"Initialized ScriptPolishAgent with model: {model_name}, max_iterations: {self.max_iterations}")
   
    def _store_image_embedding(self, image: str, ppt_id: str, slide_index: int) -> bool:
        """Store image embedding in MongoDB via CLIP service"""
        try:
            # Get embedding from CLIP service
            response = requests.post(
                f"{self.clip_url}/embed",
                json={"image": image}
            )
            response.raise_for_status()
            result = response.json()
            
            # Store in MongoDB
            self.embeddings_collection.update_one(
                {"ppt_id": ppt_id, "slide_index": slide_index},
                {
                    "$set": {
                        "embedding": result["embedding"],
                        "timestamp": datetime.now()
                    }
                },
                upsert=True
            )
            
            logger.info(f"Stored embedding for PPT {ppt_id}, slide {slide_index}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing embedding: {str(e)}")
            return False
    
    def _get_embedding(self, image: str) -> np.ndarray:
        """Get CLIP embedding for an image"""
        try:
            response = requests.post(
                f"{self.clip_url}/embed",
                json={"image": image}
            )
            response.raise_for_status()
            return np.array(response.json()["embedding"])
        except Exception as e:
            logger.error(f"Error getting embedding: {str(e)}")
            return None
    
    def _find_similar_slides(self, embeddings: List[np.ndarray], query_embedding: np.ndarray, top_k: int = 3) -> List[Dict]:
        """Find similar slides using pre-computed embeddings"""
        try:
            
            # Calculate similarities with all other slides
            similarities = []
            for i, embedding in enumerate(embeddings):
                similarity = np.dot(query_embedding, embedding)
                similarities.append({
                    "slide_index": i,
                    "similarity": float(similarity)
                })
            
            # Sort by similarity and get top-k
            similarities.sort(key=lambda x: x["similarity"], reverse=True)
            return [similarities[i]["slide_index"] for i in range(top_k)]
            
        except Exception as e:
            logger.error(f"Error finding similar slides: {str(e)}")
            return []

    def _call_gpt(self, messages: List[Dict]) -> str:
        """Call GPT model with messages"""
        try:
            logger.debug(f"Calling GPT with {len(messages)} messages")
            response = GPT_Interface.call(
                model=self.model_name,
                messages=messages,
                temperature=0.7,
                use_cache=True
            )
            logger.debug("GPT call successful")
            return response
        except Exception as e:
            logger.error(f"Error calling GPT: {str(e)}")
            return ""

    def analyze_modifications(self, edited: List[Dict], to_polish_image: str) -> ModificationAnalysis:
        """Analyze all modifications together to understand overall patterns"""
        
        # If we have more than 3 modifications, calculate similarities
        if len(edited) > 3:
            logger.info("More than 3 modifications detected. Calculating similarities.")
            
            # Get embeddings for all slides
            embeddings = []
            for edit in edited:
                embedding = self._get_embedding(edit["image"])
                if embedding is not None:
                    embeddings.append(embedding)
            
            to_polish_embedding = self._get_embedding(to_polish_image)
            
            # Find similar slides for the to-polish image
            similar_slides_idxs = self._find_similar_slides(embeddings, to_polish_embedding)
            edited = [e for i, e in enumerate(edited) if i in similar_slides_idxs]
            print(f"Found {len(edited)} similar slides")

        # Continue with existing analysis logic
        analysis_prompt = """Analyze the differences between the original scripts and user-modified scripts to extract general writing style patterns.
        Focus on understanding the user's overall writing style and approach to lecture scripting.
        
        Modified Scripts:
        {modifications}
        
        Return your analysis in the following JSON format:
        {{
            "writing_style": {{
                "tone_characteristics": [
                    "how the writer generally approaches formality/informality",
                    "typical emotional tone used",
                    "level of personal connection with audience"
                ],
                "structural_preferences": [
                    "how the writer typically organizes information",
                    "preferred ways of transitioning between topics",
                    "typical paragraph/section length patterns"
                ],
                "engagement_patterns": [
                    "how the writer typically engages the audience",
                    "preferred types of rhetorical devices",
                    "patterns in audience interaction"
                ]
            }},
            "visual_style": {{
                "reference_patterns": [
                    "how the writer typically refers to visuals",
                    "patterns in timing visual references",
                    "style of guiding visual attention"
                ],
                "integration_approach": [
                    "how writer typically integrates visual and verbal content",
                    "patterns in visual-verbal synchronization"
                ]
            }},
            "explanation_style": {{
                "complexity_level": [
                    "typical level of detail in explanations",
                    "patterns in concept breakdown",
                    "preferred depth of examples"
                ],
                "example_patterns": [
                    "types of examples typically used",
                    "how examples are usually introduced",
                    "patterns in example complexity"
                ]
            }},
            "polish_guidelines": {{
                "tone_guidance": [
                    "general tone to maintain",
                    "level of formality to target"
                ],
                "structure_guidance": [
                    "preferred organizational patterns",
                    "typical section flow"
                ],
                "engagement_guidance": [
                    "how to approach audience interaction",
                    "preferred engagement methods"
                ],
                "visual_guidance": [
                    "how to handle visual references",
                    "preferred visual integration style"
                ],
                "explanation_guidance": [
                    "target complexity level",
                    "how to approach examples"
                ]
            }}
        }}"""
        
        # Format modifications with their images
        modifications_text = []
        for i, edit in enumerate(edited, 1):
            mod_text = f"""
            Modification {i}:
            Original Script:
            {edit['original_content']}
            
            Modified Script:
            {edit['edited_content']}
            
            [Image {i}]
            """
            modifications_text.append(mod_text)
        
        content = [
            {"type": "text", "text": analysis_prompt.format(
                modifications="\n---\n".join(modifications_text)
            )}
        ]
        
        # Add all images
        for edit in edited:
            content.append({"type": "image_url", "image_url": {"url": edit["image"]}})
        
        messages = [
            {"role": "system", "content": "You are an expert writing analyst focused on lecture effectiveness. Extract specific, actionable patterns from script modifications. Return analysis in valid JSON format. Avoid generic observations. Focus on concrete changes that can be replicated."},
            {"role": "user", "content": content}
        ]
        
        response = self._call_gpt(messages)
        logger.info(f"Received analysis response from GPT: {response}")
        
        try:
            analysis_data = extract_json(response)
            
            # Extract patterns from all writing pattern categories
            patterns = []
            for category, items in analysis_data["writing_style"].items():
                patterns.extend(items)
            
            # Extract style changes from visual alignment
            style_changes = analysis_data["visual_style"]["reference_patterns"]
            
            # Extract content changes
            content_changes = analysis_data["explanation_style"]["complexity_level"]
            
            # Build polish prompt from structured instructions
            polish_sections = []
            
            if analysis_data["writing_style"]["tone_characteristics"]:
                polish_sections.append("Writing Tone:\n" + 
                    "\n".join(f"- Follow this style: {p}" for p in analysis_data["writing_style"]["tone_characteristics"]))
            
            if analysis_data["writing_style"]["structural_preferences"]:
                polish_sections.append("Structure and Organization:\n" + 
                    "\n".join(f"- Use this pattern: {s}" for s in analysis_data["writing_style"]["structural_preferences"]))
            
            if analysis_data["writing_style"]["engagement_patterns"]:
                polish_sections.append("Audience Engagement:\n" + 
                    "\n".join(f"- Engage using: {e}" for e in analysis_data["writing_style"]["engagement_patterns"]))
            
            if analysis_data["visual_style"]["reference_patterns"]:
                polish_sections.append("Visual Integration:\n" + 
                    "\n".join(f"- Handle visuals like: {p}" for p in analysis_data["visual_style"]["reference_patterns"]))
            
            if analysis_data["explanation_style"]["complexity_level"]:
                polish_sections.append("Explanation Style:\n" + 
                    "\n".join(f"- Explain with: {e}" for e in analysis_data["explanation_style"]["complexity_level"]))
            
            if analysis_data["polish_guidelines"]["tone_guidance"]:
                polish_sections.append("Additional Style Guidelines:\n" + 
                    "\n".join(f"- {g}" for g in analysis_data["polish_guidelines"]["tone_guidance"] + 
                             analysis_data["polish_guidelines"]["structure_guidance"] + 
                             analysis_data["polish_guidelines"]["engagement_guidance"] + 
                             analysis_data["polish_guidelines"]["visual_guidance"] + 
                             analysis_data["polish_guidelines"]["explanation_guidance"]))
            
            polish_prompt = """Please polish this script by mimicking the following writing style characteristics. 
                Focus on matching the general style patterns rather than specific words or phrases.
                
                """ + "\n\n".join(polish_sections)
            
            analysis = ModificationAnalysis(
                patterns=patterns,
                style_changes=style_changes,
                content_changes=content_changes,
                polish_prompt=polish_prompt.strip()
            )
            
            logger.info("Modification analysis completed")
            logger.info(f"Analysis results: {json.dumps(analysis.to_dict(), indent=2, ensure_ascii=False)}")
            
            return analysis
            
        except (ValueError, KeyError) as e:
            logger.error(f"Failed to parse or process response: {e}")
            # Fallback to empty analysis
            return ModificationAnalysis([], [], [], "Please improve the script while maintaining its core message.")

    def polish_script(self, script: str, polish_prompt: str, image: str = None) -> str:
        """Polish script using generated prompt"""
        logger.info("Starting script polishing")
        logger.info(f"Polish prompt length: {len(polish_prompt)}, Script length: {len(script)}")
        
        # Detect script language (assuming Chinese if contains Chinese characters)
        has_chinese = any('\u4e00' <= char <= '\u9fff' for char in script)
        language_instruction = """Maintain the primary language of the original script (Chinese if it contains Chinese, English otherwise).
        Technical terms, jargon, and established English phrases can be kept in English when they appear in the original script.
        Format the script for natural spoken delivery, not as a structured document.""" if has_chinese else "Use English language only."
        
        polish_system_prompt = f"""You are an expert lecture script editor. Your task is to polish the given script by:
        1. Making it flow naturally for spoken delivery
        2. Using a conversational teaching style
        3. Ensuring smooth transitions between ideas
        4. Maintaining engagement through natural speech patterns
        
        IMPORTANT:
        - {language_instruction}
        - Output ONLY the polished script text
        - Write as if speaking directly to students
        - Use natural transitions and flow
        - Avoid bullet points or structured formatting
        - Keep technical terms and jargon in their original form
        - Preserve the original mix of languages where appropriate
        - Maintain a conversational teaching tone
        
        DO NOT:
        - Add unnecessary content
        - Use document-style formatting (bullets, numbers, etc.)
        - Include any meta text or special characters
        - Make it sound like a written document
        - Arbitrarily translate established terms between languages"""
        
        content = [{"type": "text", "text": f"""Polish this script by applying the identified writing style patterns:
        {polish_prompt}
        
        Original Script (maintain this language and style):
        {script}
        
        Return ONLY the polished script text."""},
                   {"type": "image_url", "image_url": {"url": image}}]
            
        messages = [
            {"role": "system", "content": polish_system_prompt},
            {"role": "user", "content": content}
        ]
        
        polished = self._call_gpt(messages)
        
        # Clean up any formatting
        if "```" in polished or "{" in polished or ":" in polished.split("\n")[0]:
            lines = polished.split("\n")
            cleaned_lines = []
            for line in lines:
                if "```" in line or "{" in line or ":" in line:
                    continue
                if line.strip() and not line.startswith("-") and not line.startswith("#"):
                    cleaned_lines.append(line)
            polished = "\n".join(cleaned_lines)
        
        logger.info(f"Script polishing completed. Polished script:\n{polished}")
        return polished.strip()

    def evaluate_polish(self, original_script: str, polished_script: str, polish_prompt: str, image: str = None) -> tuple[bool, str]:
        """Evaluate if polished script meets requirements"""
        logger.info("Evaluating polished script")
        
        eval_prompt = f"""Evaluate how well the polished script applies the writing style patterns identified in the polish requirements.
        Consider both how well it maintains the core message and how effectively it adopts the identified style patterns.
        
        Polish Requirements:
        {polish_prompt}
        
        Original Script:
        {original_script}
        
        Polished Script:
        {polished_script}
        
        Return your evaluation in the following JSON format:
        {{
            "style_adoption": {{
                "score": 1-5,
                "successful_patterns": ["pattern successfully applied 1", "pattern successfully applied 2"],
                "missing_patterns": ["pattern not applied 1", "pattern not applied 2"]
            }},
            "content_preservation": {{
                "score": 1-5,
                "maintained_elements": ["element preserved 1", "element preserved 2"],
                "lost_elements": ["element lost 1", "element lost 2"]
            }},
            "overall_effectiveness": {{
                "score": 1-5,
                "strengths": ["strength 1", "strength 2"],
                "weaknesses": ["weakness 1", "weakness 2"]
            }},
            "satisfied": true/false,
            "improvement_suggestions": [
                "suggestion 1",
                "suggestion 2"
            ]
        }}"""
        
        content = [{"type": "text", "text": eval_prompt}, {"type": "image_url", "image_url": {"url": image}}]
            
        messages = [
            {"role": "system", "content": "You are an expert in analyzing writing style and effectiveness. Evaluate how well the polished script adopts the identified patterns while maintaining the core message."},
            {"role": "user", "content": content}
        ]
        
        response = self._call_gpt(messages)
        
        try:
            eval_data = extract_json(response)
            satisfied = eval_data["satisfied"]
            
            # Build detailed feedback from JSON structure
            feedback_parts = []
            
            # Add style adoption feedback
            feedback_parts.append(f"Style Adoption: [Score {eval_data['style_adoption']['score']}]")
            if eval_data['style_adoption']['successful_patterns']:
                feedback_parts.append("Successfully Applied Patterns:")
                feedback_parts.extend(f"- {pattern}" for pattern in eval_data['style_adoption']['successful_patterns'])
            if eval_data['style_adoption']['missing_patterns']:
                feedback_parts.append("Missing Patterns:")
                feedback_parts.extend(f"- {pattern}" for pattern in eval_data['style_adoption']['missing_patterns'])
            
            # Add content preservation feedback
            feedback_parts.append(f"\nContent Preservation: [Score {eval_data['content_preservation']['score']}]")
            if eval_data['content_preservation']['maintained_elements']:
                feedback_parts.append("Maintained Elements:")
                feedback_parts.extend(f"- {element}" for element in eval_data['content_preservation']['maintained_elements'])
            if eval_data['content_preservation']['lost_elements']:
                feedback_parts.append("Lost Elements:")
                feedback_parts.extend(f"- {element}" for element in eval_data['content_preservation']['lost_elements'])
            
            # Add overall effectiveness feedback
            feedback_parts.append(f"\nOverall Effectiveness: [Score {eval_data['overall_effectiveness']['score']}]")
            if eval_data['overall_effectiveness']['strengths']:
                feedback_parts.append("Strengths:")
                feedback_parts.extend(f"- {strength}" for strength in eval_data['overall_effectiveness']['strengths'])
            if eval_data['overall_effectiveness']['weaknesses']:
                feedback_parts.append("Weaknesses:")
                feedback_parts.extend(f"- {weakness}" for weakness in eval_data['overall_effectiveness']['weaknesses'])
            
            # Add improvement suggestions
            if eval_data["improvement_suggestions"]:
                feedback_parts.append("\nImprovement Suggestions:")
                feedback_parts.extend(f"- {suggestion}" for suggestion in eval_data["improvement_suggestions"])
            
            feedback = "\n".join(feedback_parts)
            
        except (ValueError, KeyError) as e:
            logger.error(f"Failed to parse or process response: {e}")
            satisfied = False
            feedback = response  # Use raw response as feedback
        
        logger.info(f"Evaluation completed. Satisfied: {satisfied}")
        logger.info(f"Evaluation feedback: {feedback}")
        
        return satisfied, feedback

    def analyze_continuity(self, prev_script: str, current_script: str, next_script: str) -> tuple[bool, str]:
        """Analyze if there are continuity problems between scripts"""
        logger.info("Analyzing script continuity")
        
        analysis_prompt = f"""Analyze if there are continuity problems between these lecture scripts.
        Focus on transitions, logical flow, and topic connections.
        
        Previous Script:
        {prev_script if prev_script else "[No previous script]"}
        
        Current Script:
        {current_script}
        
        Next Script:
        {next_script if next_script else "[No next page]"}
        
        Return your analysis in JSON format:
        {{
            "has_continuity_problems": true/false,
            "issues": [
                "specific issue 1",
                "specific issue 2"
            ],
            "severity": "low/medium/high",
            "recommendations": [
                "specific recommendation 1",
                "specific recommendation 2"
            ]
        }}
        
        IMPORTANT: Only mark severity as "high" if there are significant disruptions to lecture flow that would confuse students, such as:
        - Abrupt topic changes without any connection
        - Contradictory information between slides
        - Missing critical context needed to understand the current content
        - Completely disconnected ideas between slides
        
        Mark severity as "medium" or "low" for minor issues like:
        - Slightly awkward transitions
        - Could use better connecting phrases
        - Minor improvements possible but content still flows adequately
        """
        
        try:
            response = self._call_gpt(
                messages=[
                    {"role": "system", "content": "You are an expert in analyzing lecture flow and continuity."},
                    {"role": "user", "content": analysis_prompt}
                ]
            )
            
            analysis = extract_json(response)
            has_problems = analysis.get("has_continuity_problems", False)
            severity = analysis.get("severity", "low")
            
            # Build detailed feedback
            feedback = []
            if analysis.get("issues"):
                feedback.append("Issues Found:")
                feedback.extend(f"- {issue}" for issue in analysis["issues"])
            if severity:
                feedback.append(f"\nSeverity: {severity}")
            if analysis.get("recommendations"):
                feedback.append("\nRecommendations:")
                feedback.extend(f"- {rec}" for rec in analysis["recommendations"])
            
            feedback_str = "\n".join(feedback)
            logger.info(f"Continuity analysis completed. Has problems: {has_problems}, Severity: {severity}")
            if has_problems:
                logger.info(f"Continuity issues:\n{feedback_str}")
            
            # Only return True if severity is high
            needs_rewrite = has_problems and severity.lower() == "high"
            return needs_rewrite, feedback_str
            
        except Exception as e:
            logger.error(f"Error in continuity analysis: {str(e)}")
            return False, "Error analyzing continuity"

    def rewrite_for_continuity(self, prev_script: str, current_script: str, next_script: str) -> str:
        """Rewrite the current script to ensure smooth transitions with previous and next pages"""
        logger.info("Starting continuity rewrite phase")
        
        # First analyze if there are high-severity continuity problems
        needs_rewrite, analysis = self.analyze_continuity(prev_script, current_script, next_script)
        
        if not needs_rewrite:
            logger.info("No high-severity continuity problems detected. Keeping original script.")
            return current_script
            
        logger.info("High-severity continuity problems detected. Proceeding with rewrite.")
        logger.info(f"Analysis:\n{analysis}")
        
        original_length = count_words(current_script)
        logger.info(f"Original script word count: {original_length}")
        
        rewrite_prompt = f"""You are an expert in creating smooth, natural lecture flow. 
        Rewrite the current script to address the following continuity issues:
        
        {analysis}
        
        Previous Page Script:
        {prev_script if prev_script else "[No previous page]"}
        
        Current Script:
        {current_script}
        
        Next Page Script:
        {next_script if next_script else "[No next page]"}
        
        STRICT REQUIREMENTS:
        1. Keep EXACTLY the same number of words ({original_length}) as the original script
        2. Maintain the same teaching style and core content
        3. Add minimal transitions to/from adjacent content by rephrasing existing content
        4. Keep the same language mix (technical terms, etc.)
        5. Ensure the script flows naturally for spoken delivery
        
        DO NOT:
        - Add any new content or examples
        - Expand explanations
        - Include additional details
        - Make the script significantly longer than the original
        
        Return ONLY the rewritten script."""
        
        try:
            response = self._call_gpt(
                messages=[
                    {"role": "system", "content": "You are an expert in creating natural, flowing lecture scripts. Your task is to add minimal transitions while maintaining EXACTLY the same length as the original script."},
                    {"role": "user", "content": rewrite_prompt}
                ]
            )
            
            rewritten = response.strip()
            new_length = count_words(rewritten)
            logger.info(f"Rewritten script word count: {new_length}")
            
            # If the rewritten script is significantly longer, return the original
            if new_length > original_length * 1.5:  # Allow 10% margin
                logger.warning(f"Rewritten script too long ({new_length} words vs {original_length}). Keeping original.")
                return current_script
                
            logger.info("Continuity rewrite completed successfully")
            return rewritten
            
        except Exception as e:
            logger.error(f"Error in continuity rewrite: {str(e)}")
            return current_script

    def agentic_polish(self, edited: List[Dict], to_polish: Dict) -> PolishResult:
        """Main agentic workflow for polishing scripts"""
        logger.info("Starting agentic polish workflow")
        
        # Step 1: Analyze all modifications together
        logger.info("Step 1: Analyzing all modifications")
        analysis = self.analyze_modifications(edited, to_polish['image'])
        
        # Step 2: Iterative polishing process
        logger.info("Step 2: Starting iterative polishing process")
        current_script = to_polish["original_content"]
        iterations = 0
        final_feedback = ""
        satisfied = False
        
        while iterations < self.max_iterations:
            iterations += 1
            logger.info(f"Iteration {iterations}/{self.max_iterations}")
            
            # Polish script with image context
            polished_script = self.polish_script(
                current_script, 
                analysis.polish_prompt, 
                to_polish["image"]
            )
            
            # Evaluate polish with image context
            satisfied, feedback = self.evaluate_polish(
                current_script,
                polished_script,
                analysis.polish_prompt,
                to_polish["image"]
            )
            final_feedback = feedback
            
            if satisfied:
                logger.info("Polish requirements satisfied")
                current_script = polished_script
                break
            else:
                logger.info("Polish requirements not satisfied, continuing iterations")
            current_script = polished_script
        
        # Move rewrite phase outside the loop
        if satisfied and "context" in to_polish:
            logger.info("Step 3: Starting continuity rewrite phase")
            context = to_polish["context"]
            rewritten_script = self.rewrite_for_continuity(
                context.get("prev_script", ""),
                current_script,
                context.get("next_script", "")
            )
            current_script = rewritten_script
            logger.info(f"Continuity rewrite completed. Final script:\n{current_script}")
        
        result = PolishResult(
            polished_script=current_script,
            is_satisfied=satisfied,
            feedback=final_feedback,
            iterations=iterations
        )
        
        logger.info("Agentic polish workflow completed")
        logger.info(f"Final result: {json.dumps(result.to_dict(), indent=2, ensure_ascii=False)}")
        
        return result
