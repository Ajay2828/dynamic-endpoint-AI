AI_RESPONSE_SCHEMA_FLEXIBLE = {
    "type": "object",
    "required": ["key_findings", "potential_issues", "recommendations"],
    "properties": {
        "key_findings": {
            "type": "array", 
            "items": {
                "type": "object",
                "properties": {
                    "description": {"type": "string"}
                },
                "required": ["description"]
            }
        },
        "potential_issues": {
            "type": "array", 
            "items": {
                "type": "object",
                "properties": {
                    "description": {"type": "string"}
                },
                "required": ["description"]
            }
        },
        "recommendations": {
            "type": "array", 
            "items": {
                "type": "object",
                "properties": {
                    "description": {"type": "string"}
                },
                "required": ["description"]
            }
        },
        "type_specific_insights": {
            "type": "array", 
            "items": {
                "type": "object",
                "properties": {
                    "description": {"type": "string"}
                },
                "required": ["description"]
            }
        }
    }
}