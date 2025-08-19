# Grok-Powered SDR System

A Sales Development Representative (SDR) system powered by xAI's Grok API for intelligent lead qualification, personalized outreach, and pipeline management. The system allows users to define custom scoring priorities via a web form, with a default configuration in `scoring_config.json`. It is designed for local development and Docker deployment, ensuring portability and ease of use.

## Overview
The Grok-Powered SDR System automates key sales tasks:
- **Lead Qualification**: Scores leads (0-100) based on industry, company size, and data completeness, using customized score that Grok interprets dynamically.
- **Personalized Outreach**: Generates tailored emails (100-150 words) highlighting AI-powered sales tools.
- **Pipeline Management**: Tracks leads with stages (e.g., New, Contacted) and interaction history in a SQLite database.
- **Custom Scoring**: Users can define priority-based scoring rules via a JSON form, with Grok assigning scores based on these priorities.
- **Frontend**: Intuitive dashboard built with Flask, Jinja2, and TailwindCSS for lead management and configuration.

Key deliverables:
- **Grok API Integration**: Optimized prompts for scoring and email generation, with retries and mock fallbacks.
- **Evaluation Framework**: Tests Grok’s performance on sample leads, logging results and recommendations.
- **Flexible Scoring**: Priority-based scoring without hardcoded values, customizable via web form.
- **Data Management**: SQLite database for leads and interaction history, with search and CRUD operations.
- **Deployment**: Local Flask server or Docker for reproducibility.
- **Documentation**: This README with setup, usage, and troubleshooting instructions.

## Features
- **Custom Scoring Mechanism**: Define qualitative priorities for industry, company size, and completeness. Grok assigns dynamic scores based on these priorities.
- **Lead Qualification**: Scores leads with detailed reasoning (e.g., “TechCorp: Software is high priority (+35), size 600 is high priority (+35), all fields provided (+25) = 95”).
- **Personalized Outreach**: Generates professional emails tailored to lead data and industry.
- **Pipeline Management**: Tracks stages (New, Qualified, Contacted) and logs interactions (e.g., scoring, email generation).
- **Frontend Interface**: Dashboard to add/view leads, configure scoring, and generate emails.
- **Evaluation Framework**: Runs on startup, testing scoring and email generation on test cases (e.g., TechCorp, RetailCo), logging results and recommendations (e.g., refine prompt for edge cases).
- **Error Handling**: Handles API failures with retries, mock responses, and robust JSON parsing for Markdown-wrapped responses.
- **Deployment Options**: Local run or Dockerized for consistent environments.

## Technical Specs
- **Backend**: Flask (Python) for API routes and logic.
- **AI Integration**: Grok API (model: grok-4-0709, fallback: grok-beta) via OpenAI client.
- **Frontend**: Jinja2 templates with TailwindCSS (v2.2.19) and vanilla JavaScript for interactivity.
- **Database**: SQLite (`sdr_database.db`) for leads and interaction history.
- **Configuration**: `scoring_config.json` for default scoring; web form for custom priorities.
- **Dependencies**: Listed in `requirements.txt`.

## Setup and Installation
### Prerequisites
- Python 3.9+ (for local deployment).
- Docker and Docker Compose (for containerized deployment).
- XAI API key 

### Local Setup
1. Clone the repository:
   ```bash
   git clone <repository-url>
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set the your API key:
   ```bash
   export XAI_API_KEY='your-api-key'
   ```
4. Verify `scoring_config.json` exists in the project root (see default in repository).
5. Run the application:
   ```bash
   python sdr_system.py
   ```
6. Access at `http://127.0.0.1:5000`.

### Docker Deployment
1. Ensure Docker and Docker Compose are installed:
   ```bash
   docker --version
   docker-compose --version
   ```
2. Set the API key:
   ```bash
   export XAI_API_KEY='your-api-key'
   ```
3. Use the automated build script (`build.sh`) to install dependencies, build, and run:
   ```bash
   chmod +x build.sh
   ./build.sh
   ```
4. Access at `http://127.0.0.1:5000`.
5. Stop the container:
   ```bash
   docker-compose down
   ```

## Automated Build Script
The `build.sh` script automates dependency installation, Docker image building, and container running. Place it in the project root and make it executable.