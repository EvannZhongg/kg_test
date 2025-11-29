from typing import Any

PROMPTS: dict[str, Any] = {}

# LightRAG 风格的分隔符定义
PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|#|>"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"

# OpenIE 开放抽取 System Prompt
PROMPTS["open_ie_extraction_system_prompt"] = """---Role---
You are an expert Open Information Extraction (OpenIE) Specialist. Your task is to extract a comprehensive knowledge graph from the input text without being limited by a predefined schema.

---Instructions---
1.  **Entity Extraction:**
    * **Identification:** Identify ALL meaningful entities (concepts, objects, people, organizations, events, locations, abstract ideas, etc.) mentioned in the text.
    * **Entity Details:**
        * `entity_name`: The canonical name of the entity. Capitalize proper nouns. Use the most complete form found in the text.
        * `entity_type`: Automatically infer the most appropriate type for the entity based on context (e.g., "Person", "Technology", "Concept", "Metric", "Strategy"). Do not restrict yourself to a predefined list.
        * `entity_description`: A concise description based *solely* on the input text. Avoid external knowledge.
    * **Output Format:** `entity{tuple_delimiter}entity_name{tuple_delimiter}entity_type{tuple_delimiter}entity_description`

2.  **Relationship Extraction:**
    * **Identification:** Extract clearly stated relationships between the identified entities.
    * **Relationship Details:**
        * `source_entity`: Name of the source entity (must match an extracted entity name).
        * `target_entity`: Name of the target entity (must match an extracted entity name).
        * `relationship_keywords`: Key verb or phrase describing the relation (e.g., "increases", "is located in", "collaborates with").
        * `relationship_description`: A concise explanation of the connection based *solely* on the input text.
    * **Output Format:** `relation{tuple_delimiter}source_entity{tuple_delimiter}target_entity{tuple_delimiter}relationship_keywords{tuple_delimiter}relationship_description`

3.  **General Rules:**
    * **Completeness:** Extract as much information as possible.
    * **Language:** Output in {language}. Proper nouns should be retained in their original language.
    * **Tone:** All output must be in the third person, avoiding pronouns like 'I', 'you', 'this article'.
    * **Delimiter:** Use `{tuple_delimiter}` strictly as the field separator.
    * **End:** Output `{completion_delimiter}` on the final line when finished.

---Real Data to be Processed---
Text:
{input_text}

"""

PROMPTS["summarize_entity_descriptions"] = """---Role---
You are a Knowledge Graph Specialist. Your task is to synthesize a list of descriptions for a given entity or relationship into a single, comprehensive, and cohesive summary.

---Input---
Target: {entity_name}
Existing Description: "{existing_desc}"
New Description: "{new_desc}"

---Instructions---
1.  **Synthesis:** Create a comprehensive summary that combines ALL unique details from both descriptions.
2.  **Conflict Resolution:** If there are conflicting details, mention both or try to reconcile them based on context.
3.  **Conciseness:** Keep the summary informative but concise. Avoid redundancy.
4.  **Language:** Output the summary in Chinese.
5.  **Format:** Output ONLY the summarized text. Do not include labels like "Summary:".

---Output---
"""