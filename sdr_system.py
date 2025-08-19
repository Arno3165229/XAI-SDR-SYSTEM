import os
import sqlite3
import json
import re
from flask import Flask, request, render_template, jsonify
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime
import logging
from time import sleep

# Configure logging for debugging and monitoring
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
XAI_API_KEY = os.getenv("XAI_API_KEY")
if not XAI_API_KEY:
    logger.error("XAI_API_KEY not found in environment variables")
    raise ValueError("XAI_API_KEY is required. Set it using: export XAI_API_KEY='your-api-key'")

# Load default scoring configuration
def load_scoring_config():
    """Load and validate default scoring configuration from scoring_config.json"""
    try:
        with open('scoring_config.json', 'r') as f:
            config = json.load(f)
        required_keys = ['industry_relevance', 'company_size', 'completeness']
        if not all(key in config for key in required_keys):
            raise ValueError("Missing required keys in scoring_config.json")
        logger.info("Default scoring configuration loaded successfully")
        return config
    except FileNotFoundError:
        logger.error("scoring_config.json not found")
        raise ValueError("scoring_config.json is required")
    except json.JSONDecodeError:
        logger.error("Invalid JSON in scoring_config.json")
        raise ValueError("Invalid JSON format in scoring_config.json")
    except Exception as e:
        logger.error(f"Error loading scoring_config.json: {str(e)}")
        raise

# Load default config on startup
default_scoring_config = load_scoring_config()
custom_scoring_config = None  # Store user-defined config in memory

# Initialize Flask app
app = Flask(__name__)

# Initialize Grok API client
client = OpenAI(
    api_key=XAI_API_KEY,
    base_url="https://api.x.ai/v1"
)

# Test API key validity
def test_api_key(model="grok-4-0709"):
    """Test if the API key is valid by making a simple request"""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Test API key"}],
            max_tokens=10,
            temperature=0.3
        )
        logger.info(f"API key test successful with model {model}")
        return True, None, model
    except Exception as e:
        logger.error(f"API key test failed with model {model}: {str(e)}")
        return False, f"API key test failed: {str(e)}. Verify your key at console.x.ai and ensure access to model {model}. Contact xAI support with Team ID: 87ddf35c-aacb-44ee-b6bd-5cb3d873a0b5.", model

# Run API key test on startup with fallback model
api_key_valid, api_error, model = test_api_key()
if not api_key_valid and model == "grok-4-0709":
    logger.warning("Retrying API key test with fallback model grok-beta")
    api_key_valid, api_error, model = test_api_key(model="grok-beta")
if not api_key_valid:
    logger.warning(f"API key test failed, proceeding with mock responses: {api_error}")

# Database setup
def init_db():
    """Initialize SQLite database for lead management"""
    try:
        conn = sqlite3.connect('sdr_database.db', timeout=5)
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS leads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                company_name TEXT NOT NULL,
                contact_name TEXT,
                industry TEXT,
                company_size INTEGER,
                score INTEGER,
                stage TEXT,
                last_interaction TEXT,
                interaction_history TEXT
            )
        ''')
        conn.commit()
        conn.close()
        logger.info("Database initialized")
    except Exception as e:
        logger.error(f"Database initialization failed: {str(e)}")
        raise

# Initialize database on startup
init_db()

# Lead qualification prompt
def generate_qualification_prompt(lead_data, config):
    """Generate dynamic qualification prompt based on scoring config"""
    industry_rules = "\n".join([f"- {rule['industry'] if isinstance(rule['industry'], list) else rule['industry']}: +{rule['score']}" for rule in config['industry_relevance']['rules']])
    size_rules = "\n".join([f"- {rule['min_size']}{'-' + str(rule['max_size']) if rule['max_size'] else '+'} employees: +{rule['score']}" for rule in config['company_size']['rules']])
    completeness_rules = f"- All fields provided: +{config['completeness']['rules']['all_fields_provided']}\n- Per missing field: {config['completeness']['rules']['missing_field_penalty']}"
    
    return f"""
You are a sales development representative. Given:
- Company: {lead_data.get('company_name', '')}
- Contact: {lead_data.get('contact_name', '')}
- Industry: {lead_data.get('industry', '')}
- Size: {lead_data.get('company_size', 0)} employees

Score the lead from 0-100 based on following factors.
However, don't just add up the scores. Do some analysis and compare with other leads.
Industry relevance:
{industry_rules}
Company size:
{size_rules}
Completeness (fields: {', '.join(config['completeness']['rules']['fields'])}):
{completeness_rules}

Return JSON:
```json
{{
  "score": <integer>,
  "reasoning": "<explanation>"
}}
```
"""

# Personalized outreach prompt
OUTREACH_PROMPT = """
You are a sales development representative crafting a personalized outreach email.
Given the lead information:
- Company Name: {company_name}
- Contact Name: {contact_name}
- Industry: {industry}
- Company Size: {company_size} employees
- Lead Score: {lead_score}

Generate a concise, professional email (100-150 words) tailored to the lead's industry and company size.
Highlight the value of an AI-powered sales tool for their business.
Address the email to the contact name if available, otherwise use a generic greeting.
Return a JSON object with:
- subject (string)
- body (string)
Ensure the response is valid JSON.
"""

def mock_response(expected_keys, lead_data, config):
    """Generate a mock response using the scoring configuration"""
    if 'score' in expected_keys:
        score = 50
        reasoning_parts = []
        
        # Industry score
        industry = lead_data.get('industry', '').lower()
        industry_score = 0
        for rule in config['industry_relevance']['rules']:
            if isinstance(rule['industry'], list) and industry in [i.lower() for i in rule['industry']]:
                industry_score = rule['score']
                reasoning_parts.append(f"Industry ({industry}): +{industry_score}")
                break
            elif rule['industry'] == 'other' and industry and industry not in [i.lower() for r in config['industry_relevance']['rules'][:-1] for i in (r['industry'] if isinstance(r['industry'], list) else [r['industry']])]:
                industry_score = rule['score']
                reasoning_parts.append(f"Industry (other): +{industry_score}")
                break
            elif rule['industry'] == '' and not industry:
                industry_score = rule['score']
                reasoning_parts.append(f"Industry (missing): +{industry_score}")
                break
        
        # Company size score
        size = lead_data.get('company_size', 0)
        size_score = 0
        for rule in config['company_size']['rules']:
            if (rule['min_size'] <= size and (rule['max_size'] is None or size <= rule['max_size'])):
                size_score = rule['score']
                size_range = f"{rule['min_size']}{'-' + str(rule['max_size']) if rule['max_size'] else '+'}"
                reasoning_parts.append(f"Size ({size_range}): +{size_score}")
                break
        
        # Completeness score
        missing_fields = sum(1 for field in config['completeness']['rules']['fields'] if not lead_data.get(field) or (field == 'company_size' and lead_data.get(field) == 0))
        completeness_score = config['completeness']['rules']['all_fields_provided'] + (config['completeness']['rules']['missing_field_penalty'] * missing_fields)
        reasoning_parts.append(f"Completeness ({missing_fields} missing): {completeness_score}")
        
        score = max(0, min(100, industry_score + size_score + completeness_score))
        return {
            'score': score,
            'reasoning': f"Mock score: {', '.join(reasoning_parts)}"
        }
    if 'subject' in expected_keys:
        greeting = f"Dear {lead_data.get('contact_name', 'Valued Customer')},"
        return {
            'subject': f"Boost Your {lead_data.get('industry', 'Business')} with AI-Powered Sales Tools",
            'body': f"{greeting}\n\nThis is a mock email due to API issues. Our AI-powered sales tool can streamline your {lead_data.get('industry', 'business')} operations, saving time and boosting efficiency for your {lead_data.get('company_size', 0)}-employee company. Contact us to learn more!\n\nBest,\nDemo Team"
        }
    return {}

def evaluate_grok_response(response, expected_keys, prompt):
    """Evaluate Grok's response for completeness and validity"""
    try:
        logger.info(f"Raw API response: {response.model_dump_json()}")
        if not response.choices:
            return False, f"No choices in API response for prompt: {prompt[:200]}..."
        if not response.choices[0].message.content:
            return False, f"Empty message content for prompt: {prompt[:200]}..."
        
        # Strip Markdown code block markers
        content = response.choices[0].message.content
        content = re.sub(r'^```json\n|\n```$', '', content, flags=re.MULTILINE).strip()
        
        # Parse JSON
        parsed_content = json.loads(content)
        for key in expected_keys:
            if key not in parsed_content:
                return False, f"Missing key: {key} in response for prompt: {prompt[:200]}..."
        return True, "Valid response", parsed_content
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {str(e)}, raw content: {content}")
        return False, f"Invalid JSON format in response: {str(e)} for prompt: {prompt[:200]}...", None
    except Exception as e:
        logger.error(f"Error evaluating response: {str(e)}")
        return False, f"Error evaluating response: {str(e)} for prompt: {prompt[:200]}...", None

def qualify_lead(lead_data, model="grok-4-0709"):
    """Use Grok to qualify and score a lead with retry and mock fallback"""
    retries = 3
    config = custom_scoring_config if custom_scoring_config else default_scoring_config
    prompt = generate_qualification_prompt(lead_data, config)
    for attempt in range(retries):
        try:
            logger.info(f"Sending prompt to Grok (attempt {attempt+1}, model {model}): {prompt[:200]}...")
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0.3
            )
            is_valid, message, content = evaluate_grok_response(response, ['score', 'reasoning'], prompt)
            if not is_valid:
                logger.error(f"Lead qualification failed: {message}")
                if attempt == retries - 1:
                    if model == "grok-4-0709":
                        logger.warning("Retrying with fallback model grok-beta")
                        return qualify_lead(lead_data, model="grok-beta")
                    logger.warning("Using mock response due to API failure")
                    return mock_response(['score', 'reasoning'], lead_data, config), "Mock response used due to API failure"
                sleep(2 ** attempt)
                continue
            
            logger.info(f"Lead qualified: {content['score']}")
            return content['score'], content['reasoning']
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error in qualify_lead (attempt {attempt+1}, model {model}): {error_msg}")
            if "429" in error_msg and attempt < retries - 1:
                logger.warning(f"Rate limit hit, retrying in {2 ** attempt} seconds")
                sleep(2 ** attempt)
                continue
            if "404" in error_msg:
                error_msg = f"Model {model} not found or inaccessible. Verify your API key at console.x.ai and ensure access to model {model}. Contact xAI support with Team ID: 87ddf35c-aacb-44ee-b6bd-5cb3d873a0b5."
            if attempt == retries - 1:
                if model == "grok-4-0709":
                    logger.warning("Retrying with fallback model grok-beta")
                    return qualify_lead(lead_data, model="grok-beta")
                logger.warning("Using mock response due to API failure")
                return mock_response(['score', 'reasoning'], lead_data, config), f"API error: {error_msg}. Using mock response"
            sleep(2 ** attempt)
    return mock_response(['score', 'reasoning'], lead_data, config), "Max retries reached. Using mock response"

def generate_outreach(lead_data, lead_score, model="grok-4-0709"):
    """Generate personalized outreach email using Grok with retry and mock fallback"""
    retries = 3
    config = custom_scoring_config if custom_scoring_config else default_scoring_config
    prompt = OUTREACH_PROMPT.format(
        company_name=lead_data.get('company_name', ''),
        contact_name=lead_data.get('contact_name', ''),
        industry=lead_data.get('industry', ''),
        company_size=lead_data.get('company_size', 0),
        lead_score=lead_score
    )
    for attempt in range(retries):
        try:
            logger.info(f"Sending outreach prompt to Grok (attempt {attempt+1}, model {model}): {prompt[:200]}...")
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0.5
            )
            is_valid, message, content = evaluate_grok_response(response, ['subject', 'body'], prompt)
            if not is_valid:
                logger.error(f"Outreach generation failed: {message}")
                if attempt == retries - 1:
                    if model == "grok-4-0709":
                        logger.warning("Retrying with fallback model grok-beta")
                        return generate_outreach(lead_data, lead_score, model="grok-beta")
                    logger.warning("Using mock response due to API failure")
                    return mock_response(['subject', 'body'], lead_data, config), "Mock response used due to API failure"
                sleep(2 ** attempt)
                continue
            
            logger.info(f"Outreach email generated for {lead_data.get('company_name')}")
            return content, None
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error in generate_outreach (attempt {attempt+1}, model {model}): {error_msg}")
            if "429" in error_msg and attempt < retries - 1:
                logger.warning(f"Rate limit hit, retrying in {2 ** attempt} seconds")
                sleep(2 ** attempt)
                continue
            if "404" in error_msg:
                error_msg = f"Model {model} not found or inaccessible. Verify your API key at console.x.ai and ensure access to model {model}. Contact xAI support with Team ID: 87ddf35c-aacb-44ee-b6bd-5cb3d873a0b5."
            if attempt == retries - 1:
                if model == "grok-4-0709":
                    logger.warning("Retrying with fallback model grok-beta")
                    return generate_outreach(lead_data, lead_score, model="grok-beta")
                logger.warning("Using mock response due to API failure")
                return mock_response(['subject', 'body'], lead_data, config), f"API error: {error_msg}"
            sleep(2 ** attempt)
    return mock_response(['subject', 'body'], lead_data, config), "Max retries reached for email generation"

# Flask Routes
@app.route('/')
def index():
    """Render the main dashboard"""
    try:
        conn = sqlite3.connect('sdr_database.db', timeout=5)
        c = conn.cursor()
        c.execute("SELECT id, company_name, contact_name, industry, company_size, score, stage FROM leads")
        leads = c.fetchall()
        conn.close()
        return render_template('index.html', leads=leads)
    except Exception as e:
        logger.error(f"Error in index: {str(e)}")
        return jsonify({'error': f"Server error: {str(e)}"}), 500

@app.route('/add_lead', methods=['GET', 'POST'])
def add_lead():
    """Handle adding a new lead"""
    if request.method == 'POST':
        try:
            lead_data = {
                'company_name': request.form.get('company_name'),
                'contact_name': request.form.get('contact_name'),
                'industry': request.form.get('industry'),
                'company_size': int(request.form.get('company_size', 0))
            }
            if not lead_data['company_name']:
                return jsonify({'status': 'error', 'reasoning': 'Company name is required'}), 400
            
            # Qualify lead using Grok
            score, reasoning = qualify_lead(lead_data)
            
            # Log interaction
            interaction = {
                'timestamp': datetime.now().isoformat(),
                'action': 'Lead Created',
                'details': f"Scored {score}: {reasoning}"
            }
            
            # Store in database
            conn = sqlite3.connect('sdr_database.db', timeout=5)
            c = conn.cursor()
            c.execute('''
                INSERT INTO leads (company_name, contact_name, industry, company_size, score, stage, last_interaction, interaction_history)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                lead_data['company_name'],
                lead_data['contact_name'],
                lead_data['industry'],
                lead_data['company_size'],
                score,
                'New',
                interaction['timestamp'],
                json.dumps([interaction])
            ))
            conn.commit()
            conn.close()
            logger.info(f"Lead added: {lead_data['company_name']}")
            return jsonify({'status': 'success', 'score': score, 'reasoning': reasoning})
        except ValueError as e:
            logger.error(f"Error in add_lead: {str(e)}")
            return jsonify({'status': 'error', 'reasoning': f"Invalid input: {str(e)}"}), 400
        except Exception as e:
            logger.error(f"Error in add_lead: {str(e)}")
            return jsonify({'status': 'error', 'reasoning': f"Server error: {str(e)}"}), 500
    return render_template('add_lead.html')

@app.route('/scoring_config', methods=['GET', 'POST'])
def scoring_config_route():
    """Handle scoring configuration form"""
    global custom_scoring_config
    if request.method == 'POST':
        try:
            config_str = request.form.get('scoring_config', '').strip()
            if not config_str:
                custom_scoring_config = None
                logger.info("Cleared custom scoring config, using default")
                return jsonify({'status': 'success', 'reasoning': 'Default scoring configuration restored'})
            
            config = json.loads(config_str)
            required_keys = ['industry_relevance', 'company_size', 'completeness']
            if not all(key in config for key in required_keys):
                return jsonify({'status': 'error', 'reasoning': 'Missing required keys in scoring config'}), 400
            if not isinstance(config['industry_relevance']['rules'], list):
                return jsonify({'status': 'error', 'reasoning': 'industry_relevance.rules must be a list'}), 400
            if not isinstance(config['company_size']['rules'], list):
                return jsonify({'status': 'error', 'reasoning': 'company_size.rules must be a list'}), 400
            if not isinstance(config['completeness']['rules'], dict):
                return jsonify({'status': 'error', 'reasoning': 'completeness.rules must be a dict'}), 400
            
            custom_scoring_config = config
            logger.info("Custom scoring configuration saved")
            return jsonify({'status': 'success', 'reasoning': 'Scoring configuration saved successfully'})
        except json.JSONDecodeError:
            logger.error("Invalid JSON in scoring config form")
            return jsonify({'status': 'error', 'reasoning': 'Invalid JSON format in scoring config'}), 400
        except Exception as e:
            logger.error(f"Error saving scoring config: {str(e)}")
            return jsonify({'status': 'error', 'reasoning': f"Server error: {str(e)}"}), 500
    return render_template('scoring_config.html')

@app.route('/generate_email/<int:lead_id>')
def generate_email(lead_id):
    """Generate outreach email for a lead"""
    try:
        conn = sqlite3.connect('sdr_database.db', timeout=5)
        c = conn.cursor()
        c.execute("SELECT company_name, contact_name, industry, company_size, score FROM leads WHERE id = ?", (lead_id,))
        lead = c.fetchone()
        if not lead:
            logger.error(f"Lead not found: ID {lead_id}")
            return jsonify({'error': 'Lead not found'}), 404
        
        lead_data = {
            'company_name': lead[0],
            'contact_name': lead[1],
            'industry': lead[2],
            'company_size': lead[3]
        }
        lead_score = lead[4]
        
        email_content, error = generate_outreach(lead_data, lead_score)
        if error:
            logger.warning(f"Email generation used mock response: {error}")
            
        # Update interaction history
        interaction = {
            'timestamp': datetime.now().isoformat(),
            'action': 'Email Generated',
            'details': f"Subject: {email_content['subject']}"
        }
        c.execute("SELECT interaction_history FROM leads WHERE id = ?", (lead_id,))
        history = json.loads(c.fetchone()[0])
        history.append(interaction)
        c.execute("UPDATE leads SET interaction_history = ?, last_interaction = ?, stage = ? WHERE id = ?",
                  (json.dumps(history), interaction['timestamp'], 'Contacted', lead_id))
        conn.commit()
        conn.close()
        
        return jsonify(email_content)
    except Exception as e:
        logger.error(f"Error in generate_email: {str(e)}")
        return jsonify({'error': f"Server error: {str(e)}"}), 500

@app.route('/leads/<int:lead_id>')
def view_lead(lead_id):
    """View lead details and interaction history"""
    try:
        conn = sqlite3.connect('sdr_database.db', timeout=5)
        c = conn.cursor()
        c.execute("SELECT * FROM leads WHERE id = ?", (lead_id,))
        lead = c.fetchone()
        conn.close()
        if not lead:
            logger.error(f"Lead not found: ID {lead_id}")
            return jsonify({'error': 'Lead not found'}), 404
        
        lead_data = {
            'id': lead[0],
            'company_name': lead[1],
            'contact_name': lead[2],
            'industry': lead[3],
            'company_size': lead[4],
            'score': lead[5],
            'stage': lead[6],
            'last_interaction': lead[7],
            'interaction_history': json.loads(lead[8])
        }
        return render_template('lead_details.html', lead=lead_data)
    except Exception as e:
        logger.error(f"Error in view_lead: {str(e)}")
        return jsonify({'error': f"Server error: {str(e)}"}), 500

# Evaluation framework
def run_evaluation():
    """Run evaluation on Grok's performance for lead qualification and outreach"""
    results = []
    config = custom_scoring_config if custom_scoring_config else default_scoring_config
    test_cases = [
        {
            'lead_data': {
                'company_name': 'TechCorp',
                'contact_name': 'John Doe',
                'industry': 'Software',
                'company_size': 600
            },
            'expected_score_range': (70, 100),
            'expected_email_length': (100, 150)
        },
        {
            'lead_data': {
                'company_name': 'RetailCo',
                'contact_name': 'Jack Huo',
                'industry': 'Retail',
                'company_size': 50
            },
            'expected_score_range': (20, 40),
            'expected_email_length': (100, 150)
        }
    ]
    
    if not api_key_valid:
        logger.warning("Skipping API evaluation, testing mock responses")
        for test in test_cases:
            score, reasoning = mock_response(['score', 'reasoning'], test['lead_data'], config)
            score_pass = test['expected_score_range'][0] <= score <= test['expected_score_range'][1]
            
            email_content = mock_response(['subject', 'body'], test['lead_data'], config)
            email_words = len(email_content['body'].split())
            email_pass = test['expected_email_length'][0] <= email_words <= test['expected_email_length'][1]
            
            results.append({
                'company': test['lead_data']['company_name'],
                'score': score,
                'score_pass': score_pass,
                'reasoning': reasoning,
                'email_words': email_words,
                'email_pass': email_pass,
                'error': 'Mock response used due to invalid API key'
            })
    else:
        for test in test_cases:
            score, reasoning = qualify_lead(test['lead_data'])
            score_pass = test['expected_score_range'][0] <= score <= test['expected_score_range'][1] if score else False
            
            email_content, error = generate_outreach(test['lead_data'], score)
            email_pass = False
            email_words = 0
            if email_content and not error:
                email_words = len(email_content['body'].split())
                email_pass = test['expected_email_length'][0] <= email_words <= test['expected_email_length'][1]
            
            results.append({
                'company': test['lead_data']['company_name'],
                'score': score,
                'score_pass': score_pass,
                'reasoning': reasoning,
                'email_words': email_words,
                'email_pass': email_pass,
                'error': error
            })
    
    # Log evaluation results
    logger.info("Evaluation Results:")
    for result in results:
        logger.info(json.dumps(result, indent=2))
    
    return results

if __name__ == '__main__':
    # Run evaluation before starting the server
    evaluation_results = run_evaluation()
    # Start Flask server
    app.run(debug=True, host='127.0.0.1', port=5000)