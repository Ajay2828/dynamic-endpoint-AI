def generate_ai_insights(query: str, data: List[Dict], analysis_type: str = "standard", custom_prompt: str = None, ml_context: Dict = None) -> Dict:
    """Generate AI analysis with optional ML-enhanced prompts"""
    if not data:
        return {"error": "No data provided for analysis"}

    access_token = get_access_token_from_service_account()
    
    # Build the base prompt
    base_prompt = f"""
    [INST] <<SYS>>
    You are a data analyst. Provide {analysis_type} analysis.
    Focus only on the provided data. Format response as JSON:
    {{
        "key_findings": [],
        "potential_issues": [],
        "recommendations": [],
        "type_specific_insights": []
    }}
    <</SYS>>
    """
    
    # Add ML context if available
    if ml_context:
        ml_section = f"""
    ML ANALYSIS CONTEXT:
    {json.dumps(ml_context, indent=2)}
    
    Please incorporate these ML insights into your analysis and explain how they support or contradict your findings.
    """
        base_prompt += ml_section
    
    # Add custom instructions if provided
    if custom_prompt:
        custom_section = f"""
    ADDITIONAL INSTRUCTIONS:
    {custom_prompt}
    """
        base_prompt += custom_section
    
    # Add data section
    data_section = f"""
    DATA SAMPLE: {json.dumps(data[:3], indent=2)}
    RECORD COUNT: {len(data)}
    [/INST]
    """
    
    prompt_template = base_prompt + data_section

    try:
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "meta/llama-3.1-405b-instruct-maas",
            "messages": [{"role": "user", "content": prompt_template}],
            "response_format": {"type": "json_object"},
            "temperature": 0.3 if analysis_type == "business" else 0.1,
            "max_tokens": 1024
        }

        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        response_json = response.json()
        ai_content = response_json['choices'][0]['message']['content']
        
        try:
            parsed_content = json.loads(ai_content)
            transformed_content = transform_ai_response(parsed_content)
            return validate_ai_response(transformed_content)
        except json.JSONDecodeError as json_err:
            return {"error": f"Invalid JSON response from AI: {json_err}"}

    except requests.exceptions.RequestException as req_err:
        return {"error": f"API request failed: {req_err}"}
    except Exception as e:
        return {"error": f"AI analysis failed: {str(e)}"}


@app.route('/dynamic/<path:endpoint_name>', methods=['GET'])
@limiter.limit("60 per minute")
@api_key_required
@api_error_handler
def dynamic_endpoint(endpoint_name):
    """Handle dynamic endpoint requests with ML/AI options"""
    if endpoint_name not in endpoint_registry:
        return jsonify({"error": f"Endpoint /{endpoint_name} not found"}), 404
    
    # Process filters/parameters
    filters = request.args.to_dict()
    non_sql_keys = ['ai_analyze', 'analysis_type','ml_analyze']
    filtered_params = {k: v for k, v in filters.items() if k not in non_sql_keys}
    filter_values = list(filtered_params.values())

    query = endpoint_registry[endpoint_name]["query"]
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    
    try:
        cursor.execute(sql.SQL(query), list(request.args.values()))
        columns = [desc[0] for desc in cursor.description]
        data = [dict(zip(columns, row)) for row in cursor.fetchall()]
    finally:
        cursor.close()
        conn.close()

    response = {
        "metadata": {
            "endpoint": endpoint_name,
            "record_count": len(data)
        },
        "data": data
    }

    # Get analysis parameters
    wants_ml = request.args.get('ml_analyze', '').lower() in ('true', '1', 'yes')
    wants_ai = request.args.get('ai_analyze', '').lower() in ('true', '1', 'yes')
    analysis_type = request.args.get('analysis_type', 'standard')
    
    ml_insights = None
    
    # ML Analysis
    if wants_ml:
        ml_insights = ml_generator.generate_insights(endpoint_name, data)
        response["ml_insights"] = ml_insights
    
    # AI Analysis (with or without ML context)
    if wants_ai:
        if wants_ml and ml_insights:
            # Enhanced AI analysis with ML context
            custom_instructions = f"Focus on {analysis_type} analysis and provide business implications of the ML findings."
            response["ai_analysis"] = generate_ai_insights(
                query, 
                data, 
                analysis_type,
                custom_prompt=custom_instructions,
                ml_context=ml_insights
            )
        else:
            # Standard AI analysis
            response["ai_analysis"] = generate_ai_insights(query, data, analysis_type)

    return jsonify(response)