# WhatsApp Group Members Classification

[![Python version](https://img.shields.io/badge/python-3.x-brightgreen.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Transformers](https://img.shields.io/badge/Transformers-v4.26.0-blue.svg)](https://huggingface.co/transformers/)


In this project, we aim to perform group members classification on WhatsApp group chat data using NLP. The goal is to analyze the text in the chat and predict who said what.

We use the `Transformers` library from `Huggingface` to fine-tune the `Bert` pre-trained model.

Since the messages in my personal groups are in Hebrew, I used the `Transformers` library to translate the data from Hebrew to English.

## Pipeline Steps
1. Convert the group chat txt data into CSV format
2. Preprocess the data
3. Translate the data
4. Prepare the data for classification
5. Train the classification
6. Evaluate on test data (TODO)
7. Analyze the results (TODO)

## Getting Started

### Prerequisites

To run this project on your machine, you need the following:

1. WhatsApp chat data. To export the data:
    - Enter the relevant chat on your phone.
    - Click on three dots.
    - Click `more`.
    - Click `export chat` (without media).
    - Choose the relevant sharing method you want to use.
2. Python 3 installed on your computer
3. Docker installed. The Docker image is provided.

### Installing

To create and run the Docker image, run the following commands:

```bash
docker build -t <image-name> .
docker run -it <image-name> -p 8888:8888 -v $(pwd)/code:/opt/code -v $(pwd)/data:/opt/data --rm <image-name>

```

I used VSCODE and the Dev Containers extension to make it all work. You can follow this link for more information: https://code.visualstudio.com/docs/devcontainers/containers

## Code Extensions

The config.py file is private as it contains my friends' names, so an example is provided below. It should include:

- The global data path
- The subdirectories to save the outputs
- The mapping between the members' names to the names used in the code. In my case, the members' names were in Hebrew, so I mapped them to English names.
- The names you want to use for classification.

```python

global_datapath = '/workspaces/group_members_classification/data'
config = {

    'raw_data_path': f'{global_datapath}/raw',
    'raw_data_file_names' : ['group1_data', 'group2_data', 'group3_data'],
    'prepared_raw_data_path': f'{global_datapath}/processed/prepared_raw_data',
    'prepared_for_machine_translation_path': f'{global_datapath}/processed/prepared_for_machine_translation',
    'prepared_for_classification_path': f'{global_datapath}/processed/prepared_for_classification',
    'classification_model_path': f'{global_datapath}/processed/classification_model',
    'train_file_names': ['group1_data', 'group2_data'],
    'names_dictionary' : {
        'NaMe1': 'name1',
        'NaMe2': 'name2',
        'NaMe3': 'name3',
        'NaMe4': 'name4',
                            },
    'names_to_classify': ['name1', 'name2', 'name3', 'name4']
}

```



## Milestones

### Achieved
- Date: 2023-02-04
  - Initial release of the project
  - Training model code

### Future plans
- Use the model to predict on test data
- Evaluate the performance
- Perform text visualization of easy and hard examples
