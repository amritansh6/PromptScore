Project README
Overview
This project provides a comprehensive suite of tools for evaluating and fine-tuning prompt scoring models using LLama. It includes scripts for adding base prompts with constraints to a database, fine-tuning LLama models, generating performance graphs, and comparing multiple models based on specified criteria.

Getting Started
Prerequisites
Ensure you have Python 3 installed on your system. You can download it from Python's official website.

Installation
Clone this repository to your local machine using:

bash
Copy code
git clone [repository-url]
Navigate to the project directory:

bash
Copy code
cd [project-directory]
Database Setup
A pre-populated database is provided with the project to facilitate immediate usage of the graph scripts. Ensure the database is correctly configured and connected before proceeding with the scripts below.

Usage
Adding Base Prompts
To add a new base prompt and continue adding constraints which are stored in the database for evaluation, run:

bash
Copy code
python3 PromptScore.py
Follow the on-screen prompts to input base prompts and constraints.

Fine-Tuning LLama
To fine-tune the LLama model for prompt scoring evaluation, execute:

bash
Copy code
python3 PromptScore_Llama.py
This script will run a training session based on the parameters specified in the file.

Generating Performance Graphs
To generate performance graphs for each model, run the corresponding script in the /llms directory:

bash
Copy code
python3 /llms/file_name.py
Replace file_name.py with the actual name of the script you wish to run.

Model Comparison
To compare multiple models based on each criterion, run:

bash
Copy code
python3 compare.py
This will execute comparisons and generate output based on the predefined criteria in the script.

Support
For any issues or questions, please refer to the documentation or contact the project maintainer.

