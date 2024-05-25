# GitHub Repository Q&A App

The GitHub Repository Q&A App is a web-based application designed to facilitate the exploration and understanding of GitHub repositories by providing a question-and-answer interface. It utilizes powerful language models to generate contextual answers from the contents of any given repository. This tool is particularly useful for developers, researchers, and anyone looking to quickly understand code and documentation without manually sifting through files.

## Features

- **Repository Cloning**: Clone any public GitHub repository directly into the application.
- **Automatic Text Analysis**: Process and analyze text from various code and documentation files within the repository.
- **Question & Answer System**: Submit questions and receive answers based on the content of the repository, leveraging a state-of-the-art language model.
- **Efficient Searching**: Uses FAISS (Facebook AI Similarity Search) for efficient high-dimensional vector searches, allowing rapid retrieval of relevant documents.
- **User-Friendly Interface**: Simple and intuitive web interface built with Streamlit, allowing users to interact easily with the backend capabilities.

## Installation

To set up and run this application, follow these steps:

### Prerequisites

- Python 3.8 or higher
- pip for Python package installation

### Clone the Repository

Start by cloning this repository to your local machine:

```bash
https://github.com/aswin-bs/Github_Repo_Qna.git
```

### Install Dependencies

Install the necessary Python packages using pip:

```bash
pip install -r requirements.txt
```

### Set Up Environment Variables

Create a `.env` file in the root directory of the project and populate it with your API keys:

```plaintext
GROQ_API_KEY=<your_groq_api_key_here>
OPENAI_API_KEY=<your_openai_api_key_here>
```

Replace `<your_groq_api_key_here>` and `<your_openai_api_key_here>` with your actual API keys.

### Run the Application

Launch the application by running:

```bash
streamlit run app.py
```

Navigate to `http://localhost:8501` in your web browser to view and interact with the application.

## Usage

Once the application is running, follow these steps to use it:

1. **Enter the GitHub Repository URL**: Input the URL of the GitHub repository you want to analyze.
2. **Load Repository**: Click the "Load Repository" button to clone and process the repository.
3. **Ask Questions**: Enter questions in the provided text input field to get answers based on the repository's content.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

- [Streamlit](https://streamlit.io/) for creating an amazing tool for rapid data science application development.
- [LangChain](https://langchain.com/) for providing the conversational AI framework.
- [Hugging Face](https://huggingface.co/) for the robust models and embeddings.
- [FAISS](https://github.com/facebookresearch/faiss) for enabling efficient similarity searches.

Feel free to test the application with various repositories to see its capabilities in action. Happy coding!
